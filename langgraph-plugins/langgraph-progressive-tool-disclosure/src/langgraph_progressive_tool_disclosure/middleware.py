"""Progressive tool disclosure: what one LangChain model call is told about tools.

Every call carries, in ``request.tools``, only the tools that are callable on it: the two disclosure
tools (``find_tools`` and ``get_tool_details``), the always-available ones and the ones loaded by
``get_tool_details`` and still in use. A loaded tool is released after ``ttl_cycles`` cycles without a
call, and every call renews it, so a tool used in a stretch stays loaded without reloading and the tool
list still goes back to the mandatory set once the work moves on. Every other registered tool reaches
the model as one line of a catalog appended to the system prompt — its name and a summary of its
description, at most ``catalog_chars`` characters long — under the rule that a listed tool is not
callable and has to be loaded first.

In the messages each call sends, an exchange of a tool that call does not carry is folded to one
sentence — ``The tool X was called and the result was: Y`` — so the model keeps the evidence without a
call shape to repeat, and the disclosure tools' own exchanges are dropped outright.

Everything above is a decision of :func:`context_core.disclosure` over the neutral message and tool
shapes. This module holds only what touches LangChain: reading specifications off ``BaseTool``,
rewriting the request with ``request.override``, the two tools, the per-thread TTL bookkeeping in graph
state, and the guard that cancels a call to a name the model could not see.

State, not instance attributes
-----------------------------
A LangGraph agent is a graph, and its per-thread facts belong in its state so they survive a
checkpoint. ``loaded_tools`` maps a tool name to the cycle it was loaded on, merged across parallel
writes by :func:`_merge_loads`. The cycle counter is not stored at all: it is the number of
``AIMessage`` objects in the call's messages, which makes it a function of the state rather than a
second thing to keep in step with it. Renewal is derived the same way — a tool called in an
``AIMessage`` at cycle *k* counts as used at *k* — so nothing has to be written back when a tool runs.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Container, Mapping, Sequence
from typing import Annotated, Any, NotRequired, TypeAlias

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langchain.tools import ToolRuntime
from langgraph.types import Command

from context_core.disclosure import (
    DEFAULT_CATALOG_CHARS,
    FIND_TOOLS_NAME,
    GET_TOOL_DETAILS_NAME,
    PLUGIN_TOOL_NAMES,
    LexicalToolIndex,
    SummaryCache,
    ToolIndex,
    ToolMatch,
    ToolSpec,
    build_catalog,
    clamp_summary,
    estimate_tokens,
    summary_line,
    fold_closed_exchanges,
)

from ._adapter import to_langchain, to_neutral_list
from ._compat import AgentMiddleware, AgentState, ModelRequest, ModelResponse, ToolCallRequest

__all__ = ["DisclosureState", "ProgressiveToolDisclosureMiddleware"]

logger = logging.getLogger(__name__)

DEFAULT_TTL_CYCLES = 3
"""Cycles a loaded tool survives without a call. Every call renews it."""

DEFAULT_TOP_K = 3
"""How many tools one search lists."""

ToolSummarizer: TypeAlias = "Callable[[ToolSpec, int], str]"
"""Writes a catalog line for a description that does not fit ``catalog_chars``.

Receives ``(spec, max_chars)`` and returns the summary. What it returns is clamped to the limit, and one
that raises or answers nothing falls back to truncation at a sentence or word boundary."""

_MATCHES_HEADER = (
    f"Tools that match. Nothing is loaded yet: call `{GET_TOOL_DETAILS_NAME}` with the names you want, then call them."
)
"""Opens the search result. A search only finds; loading is the other tool's job, and this says so."""

_DETAILS_LOADED_HEADER = (
    "Loaded. These tools are callable with their full parameters on your next call. A tool left unused for "
    f"a few calls is unloaded; to call it after that, call `{GET_TOOL_DETAILS_NAME}` again:"
)
"""Opens the loading result. The specification itself travels in the tool list, not in this text: a tool
result is resident in the history, the tool list is per call and forgettable."""

_DETAILS_UNKNOWN = "Not a tool, ignored: {names}. Use names from the catalog or from `" + FIND_TOOLS_NAME + "`."
"""Line naming the requested names the agent does not have."""

_DETAILS_EMPTY_GUIDANCE = (
    "Pass the names of the tools you want loaded, as a list. Take them from the catalog, or call `"
    + FIND_TOOLS_NAME
    + "` first."
)
"""Answer to a loading call that named nothing usable."""

_EMPTY_NEED_GUIDANCE = "Describe what you are trying to do, in your own words, then call this tool again."
"""Answer to an empty need. Guidance rather than an error: the model can recover on its own turn."""

_NO_MATCH_GUIDANCE = "No tool matches that description. Try different wording, or answer directly."
"""Answer when nothing usable was found. Names the two ways out so the model does not retry blindly."""

_SEARCH_FAILED_GUIDANCE = "Tool search is unavailable right now. Try a different description, or answer directly."
"""Answer when the search itself raised. Worded as guidance rather than as an error, and offering the
same two ways out as a no-match: from where the model stands the two cases are the same one, and the
failure is the middleware's to log, not the model's to reason about."""

_SUMMARY_CONCURRENCY = 8
"""Summaries requested at once when an index build meets many new tools. Bounds the burst, not the total."""

_SUMMARY_SYSTEM_PROMPT = """\
You write catalog lines for tools. You are given one tool's name and description. Reply with a single \
summary of what the tool does, at most {max_chars} characters, in the language of the description. Keep \
what tells this tool apart from similar ones: the object it acts on and what it returns. No preamble, no \
quotes, no tool name, no trailing period needed."""
"""Instruction of the default summarizer, verbatim from the Strands plugin. Asks for the discriminating
content, which is what the model reading the catalog needs to decide whether to load the tool."""

_PREMATURE_CALL_MESSAGE = (
    "'{name}' did not run: it is not loaded, so its parameters are unknown to you. Call `"
    + GET_TOOL_DETAILS_NAME
    + '` with ["{name}"] first, then call \'{name}\' with its real parameters.'
)
"""Cancellation message of a guessed call. Names the tool and points at ``get_tool_details``. The guard
does NOT load the tool on the model's behalf: a recovery that loaded it would teach the model that
calling a catalog name directly works, which is the very shortcut the catalog rule forbids."""


# --------------------------------------------------------------------------------------------------
# State: which tools are loaded, and since when.
# --------------------------------------------------------------------------------------------------


def _merge_loads(left: Mapping[str, int] | None, right: Mapping[str, int] | None) -> dict[str, int]:
    """Merge two ``loaded_tools`` maps, keeping the later cycle for a name present in both.

    A reducer rather than a replacement, because several ``get_tool_details`` calls can land in one
    superstep and a plain assignment would let one of them win outright. Keeping the *maximum* cycle is
    what makes the merge order-independent: the same set of loads produces the same map whichever order
    the writes arrive in.

    The map is therefore never pruned — it is bounded by the number of distinct tools ever loaded, and
    expiry is a read-time decision (:func:`_active_names`) rather than a deletion.

    Args:
        left: The map already in state, or ``None`` on the first write.
        right: The incoming update, or ``None``.

    Returns:
        A new merged map. Neither argument is mutated.
    """
    merged = dict(left or {})
    for name, cycle in (right or {}).items():
        merged[name] = max(merged.get(name, cycle), cycle)
    return merged


class DisclosureState(AgentState):
    """Agent state extended with the disclosure bookkeeping.

    Attributes:
        loaded_tools: Tool name to the cycle ``get_tool_details`` loaded it on. Absent until the first
            load, which is the state of an agent that has not needed a hidden tool yet.
    """

    loaded_tools: NotRequired[Annotated[dict[str, int], _merge_loads]]


# --------------------------------------------------------------------------------------------------
# Reading tool specifications off whatever the request carries.
# --------------------------------------------------------------------------------------------------


def _schema_of(bound: BaseTool) -> dict[str, Any]:
    """Return ``bound``'s JSON input schema, wrapped the way ``ToolSpec`` expects.

    The full schema rather than ``BaseTool.args``: the index and the guard both read it, the first for
    the parameter descriptions that separate two tools whose one-line summaries read alike and the
    second for the ``required`` list. A tool whose schema cannot be rendered yields an empty object
    schema, which reads as "takes no arguments" — the safe answer, since it exempts the tool from the
    guard rather than cancelling a legitimate call.

    Args:
        bound: The tool to read. Left unmodified.

    Returns:
        ``{"json": schema}``.
    """
    try:
        return {"json": bound.tool_call_schema.model_json_schema()}
    except Exception:
        logger.debug("tool schema unavailable | tool=<%s>", getattr(bound, "name", "?"), exc_info=True)
        return {"json": {"type": "object", "properties": {}}}


def _spec_of(bound: BaseTool | Mapping[str, Any]) -> ToolSpec | None:
    """Convert one entry of ``request.tools`` to a neutral :class:`ToolSpec`.

    ``request.tools`` is ``list[BaseTool | dict]``: a caller may bind a provider-native tool declaration
    straight through, and both the OpenAI ``{"type": "function", "function": {...}}`` envelope and a bare
    ``{"name": ..., "description": ...}`` are accepted. An entry with no name is skipped rather than
    guessed at, since a name is what every decision here is keyed by.

    Args:
        bound: The entry to convert. Left unmodified.

    Returns:
        The specification, or ``None`` when the entry carries no name.
    """
    if isinstance(bound, BaseTool):
        return {"name": bound.name, "description": bound.description or "", "inputSchema": _schema_of(bound)}

    if isinstance(bound, Mapping):
        body = bound.get("function") if isinstance(bound.get("function"), Mapping) else bound
        name = body.get("name")
        if not isinstance(name, str) or not name:
            return None
        schema = body.get("parameters") or body.get("input_schema") or body.get("inputSchema") or {}
        return {
            "name": name,
            "description": str(body.get("description") or ""),
            "inputSchema": schema if "json" in schema else {"json": schema},
        }

    return None


def _specs_of(tools: Sequence[BaseTool | Mapping[str, Any]]) -> list[ToolSpec]:
    """Convert every entry of ``request.tools`` to a neutral specification, in arrival order."""
    specs = [_spec_of(bound) for bound in tools]
    return [spec for spec in specs if spec is not None]


def _requires_parameters(spec: ToolSpec) -> bool:
    """Report whether ``spec`` declares at least one required parameter.

    Only the ``required`` list decides. A tool whose parameters are all optional is callable with no
    arguments, so an empty call to it is a legitimate call and not the symptom of a schema the model
    never saw — which is why such a tool is exempt from the guard.

    Args:
        spec: Specification to read. Left unmodified.

    Returns:
        ``True`` when the schema lists at least one required parameter; ``False`` when it lists none,
        and for any schema shape this cannot read.
    """
    input_schema: object = spec.get("inputSchema")
    if not isinstance(input_schema, dict):
        return False
    root = input_schema.get("json", input_schema)
    if not isinstance(root, dict):
        return False
    required = root.get("required")
    return isinstance(required, list) and len(required) > 0


# --------------------------------------------------------------------------------------------------
# The cycle counter and the TTL, both derived from the messages.
# --------------------------------------------------------------------------------------------------


def _cycle(messages: Sequence[BaseMessage]) -> int:
    """Return the cycle the *next* model call runs on: the number of assistant messages so far.

    Each model call appends exactly one ``AIMessage``, so counting them is the cycle counter, and
    deriving it from the messages means there is no separate counter to keep in step with the state or
    to lose on a checkpoint restore.

    Args:
        messages: Messages of the call, system message excluded.

    Returns:
        The cycle index, ``0`` on the first call of a thread.
    """
    return sum(1 for message in messages if isinstance(message, AIMessage))


def _last_used(messages: Sequence[BaseMessage], loaded: Mapping[str, int], before: int) -> dict[str, int]:
    """Return each loaded tool's last use on a cycle strictly before ``before``.

    Renewal is read out of the history rather than written when a tool runs. A tool called by the
    ``AIMessage`` of cycle *k* counts as used at *k*, so a tool used in a stretch of cycles keeps renewing
    itself with no state write at all.

    Two restrictions, and each one exists because of a case that breaks without it:

    - Only a tool that ``loaded`` already holds is renewed. A call is not a load: renewing on any call
      would let a name the model guessed at install itself in the live set, which is exactly the
      shortcut the guard exists to refuse.
    - A use on cycle ``before`` or later does not count. At model-call time no message is that recent, so
      this is a no-op; at guard time the call being judged is itself an ``AIMessage`` on the current
      cycle, and counting it would have the call vouch for its own tool.

    Args:
        messages: Messages of the call, system message excluded.
        loaded: The ``loaded_tools`` map from state.
        before: The cycle being decided for. Uses on it and after it are ignored.

    Returns:
        Tool name to the cycle of its last use, over the keys of ``loaded``. Never mutates ``loaded``.
    """
    # A load recorded on a cycle LATER than the one being decided cannot be real: it was numbered on a
    # longer history, before a middleware removed messages from state (the relevance filter drops its
    # closed retrieval exchanges at the end of a run). Left as is, the tool would stay uncallable until
    # the count caught up, and the model would reload it cycle after cycle. It is treated as loaded just
    # before. A load ON the decided cycle is left alone: that is the same-batch case the guard refuses.
    last = {name: (used if used <= before else before - 1) for name, used in loaded.items()}
    cycle = 0
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        if cycle >= before:
            break
        for call in message.tool_calls or ():
            name = call.get("name")
            if name in last:
                last[name] = max(last[name], cycle)
        cycle += 1
    return last


def _active_names(
    last_used: Mapping[str, int],
    cycle: int,
    ttl_cycles: int,
    always_available: Container[str],
    catalog_names: Container[str],
) -> set[str]:
    """Return the names callable on cycle ``cycle``: the disclosure tools, the always-available, the live.

    A load or a use counts only when it happened on an EARLIER cycle than ``cycle``. That one
    inequality does two jobs: it is trivially true for the normal path — a tool loaded on cycle *k* is
    callable from *k+1* — and it is what makes the guard correct, because a sibling ``get_tool_details``
    that ran in the same batch as a guessed call recorded the *current* cycle and so cannot make the
    guess look sanctioned.

    An exposure at exactly ``cycle - last_use == ttl_cycles`` is kept: the boundary belongs to the live
    side, matching the reference implementation.

    Args:
        last_used: Tool name to the cycle of its last load or call.
        cycle: Cycle the decision is being made for.
        ttl_cycles: Cycles a loaded tool survives without a call.
        always_available: Names configured to be callable on every call.
        catalog_names: Every name the agent has. A name outside it is ignored, so a tool unregistered
            since it was loaded cannot resurrect.

    Returns:
        The callable names.
    """
    active = {name for name in PLUGIN_TOOL_NAMES if name in catalog_names}
    active |= {name for name in always_available if name in catalog_names}
    active |= {
        name
        for name, used in last_used.items()
        if name in catalog_names and used < cycle and cycle - used <= ttl_cycles
    }
    return active


# --------------------------------------------------------------------------------------------------
# Rewriting the request.
# --------------------------------------------------------------------------------------------------


def _with_catalog(system_message: SystemMessage | None, block: str) -> SystemMessage | None:
    """Return ``system_message`` with ``block`` appended as one more text block.

    A NEW message: ``SystemMessage.content_blocks`` is a read-only property in ``langchain-core``, so the
    block list is read off it and a replacement message is built from it. The existing blocks are kept as
    separate blocks rather than flattened into one string, because a caller using the list form is
    usually placing cache checkpoints between them and collapsing it would move them.

    Appending rather than prepending is deliberate: the caller's own prompt keeps the opening position,
    so on a provider that caches by prefix the operator's text stays at a stable offset.

    Args:
        system_message: The request's system message, or ``None`` when the caller set none. Not mutated.
        block: Text to append. An empty block returns ``system_message`` unchanged, by identity.

    Returns:
        The extended message, or a new one carrying only the catalog when there was none.
    """
    if not block:
        return system_message
    if system_message is None:
        return SystemMessage(content=block)
    return SystemMessage(content=[*system_message.content_blocks, {"type": "text", "text": block}])


def _keep_active(bound: Sequence[Any], active: Container[str]) -> list[Any]:
    """Return the entries of ``bound`` whose name is in ``active``, in arrival order.

    Arrival order rather than the order of ``active``: two calls with the same disclosure state then
    produce the same tool list, and a provider's prompt cache is not invalidated by a reordering alone.
    Every entry kept is the caller's own object, verbatim — there is no reduced form of a tool in the
    list, only presence or absence.

    Args:
        bound: Entries of ``request.tools``, ``BaseTool`` or provider-native dict alike.
        active: Names callable on this call.

    Returns:
        The kept entries. An entry with no readable name is dropped, since nothing can vouch for it.
    """
    kept: list[Any] = []
    for entry in bound:
        spec = _spec_of(entry)
        if spec is not None and spec["name"] in active:
            kept.append(entry)
    return kept


def _fold_messages(messages: Sequence[BaseMessage], active: Container[str]) -> list[BaseMessage]:
    """Return ``messages`` with every closed exchange of a tool this call cannot call folded to a sentence.

    The decision is :func:`context_core.disclosure.fold_closed_exchanges`, over the neutral shape. What
    happens here is the round trip, and the one thing the round trip must not lose is object identity:
    the core returns the very same neutral message object for anything it did not rewrite, so those are
    mapped back to the ORIGINAL ``BaseMessage`` rather than reconstructed. That is what keeps the turn in
    flight — and a reasoning model's latest assistant message, whose signature a rebuild would
    invalidate — byte-identical to what arrived.

    Args:
        messages: Messages of the call, system message excluded. Read only.
        active: Names carried in this call's tool list.

    Returns:
        ``messages`` as a list when nothing was folded; otherwise the folded messages.
    """
    originals = list(messages)
    neutral = to_neutral_list(originals)
    folded = fold_closed_exchanges(neutral, active)
    if folded is neutral:
        return originals

    by_identity = {id(item): original for item, original in zip(neutral, originals, strict=True)}
    out: list[BaseMessage] = []
    for item in folded:
        original = by_identity.get(id(item))
        if original is not None:
            out.append(original)
        else:
            out.extend(to_langchain(item))
    return out


# --------------------------------------------------------------------------------------------------
# Configuration validation. Every check runs in the constructor, before anything is registered.
# --------------------------------------------------------------------------------------------------


def _validate_positive_int(value: object, parameter: str) -> None:
    """Reject anything that is not an integer of at least ``1``.

    ``bool`` is rejected explicitly: it passes as an integer in Python, and ``ttl_cycles=True`` silently
    meaning "one cycle" is the kind of configuration that looks like it works. A float is rejected as
    well, including ``5.0``: the value is compared against a cycle counter, so a non-integer has no
    meaning here.

    Raises:
        ValueError: When ``value`` is not an ``int`` of at least ``1``.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{parameter}=<{value!r}> | must be an integer greater than or equal to 1")


def _validate_catalog_chars(catalog_chars: object) -> None:
    """Reject anything that is neither ``None`` nor an integer of at least ``1``.

    ``None`` suppresses the catalog altogether, which is a supported configuration; ``0`` is not, as a
    limit of zero characters would emit catalog lines nothing fits in.

    Raises:
        ValueError: When ``catalog_chars`` is neither ``None`` nor an ``int`` of at least ``1``.
    """
    if catalog_chars is None:
        return
    if isinstance(catalog_chars, bool) or not isinstance(catalog_chars, int) or catalog_chars < 1:
        raise ValueError(f"catalog_chars=<{catalog_chars!r}> | must be None or an integer greater than or equal to 1")


def _validate_always_available(always_available: object) -> None:
    """Reject anything that is not a sequence of non-empty strings.

    A bare string is rejected even though it is a sequence of strings: ``always_available="current_time"``
    would silently configure one name per character, so the mistake is caught rather than honored.

    Raises:
        ValueError: When ``always_available`` is not a sequence, is a string, or holds anything other
            than non-empty strings.
    """
    if isinstance(always_available, (str, bytes)) or not isinstance(always_available, Sequence):
        raise ValueError(f"always_available=<{always_available!r}> | must be a list or tuple of non-empty strings")
    for name in always_available:
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"always_available=<{always_available!r}> | must be a list or tuple of non-empty "
                f"strings, got the element <{name!r}>"
            )


def _validate_index(index: object) -> None:
    """Reject anything that is neither ``None`` nor an object exposing ``build`` and ``search``.

    Checked by member rather than by ``isinstance``: :class:`ToolIndex` is a structural protocol, so any
    object carrying both operations is a valid implementation and a double used in tests never has to
    inherit from anything.

    Raises:
        ValueError: When ``index`` is not ``None`` and lacks a callable ``build`` or ``search``.
    """
    if index is None:
        return
    for member in ("build", "search"):
        if not callable(getattr(index, member, None)):
            raise ValueError(f"index=<{index!r}> | must expose a callable '{member}' member")


def _validate_summarizer(summarizer: object) -> None:
    """Reject anything that is neither ``None`` nor callable.

    Raises:
        ValueError: When ``summarizer`` is not ``None`` and is not callable.
    """
    if summarizer is None or callable(summarizer):
        return
    raise ValueError(f"summarizer=<{summarizer!r}> | must be None or a callable taking (spec, max_chars)")


# --------------------------------------------------------------------------------------------------
# The middleware.
# --------------------------------------------------------------------------------------------------


class ProgressiveToolDisclosureMiddleware(AgentMiddleware):
    """Send a catalog in the system prompt plus two small tools on each call, instead of every full schema.

    Every tool stays bound to the agent and stays callable. What changes is what one call is told: the
    tool list carries only ``find_tools``, ``get_tool_details``, the tools configured as always available
    and the tools loaded and still in use, and every other tool is one line of a catalog appended to the
    system prompt — its name and a summary of its description.

    The flow is catalog -> ``get_tool_details([names])`` -> call. A loaded tool is released after
    ``ttl_cycles`` cycles without a call and every call renews it, so the tool list goes back to the
    mandatory set once the work moves on. ``find_tools`` stays for a need the model cannot map to a
    listed name: it searches and lists matches, and loading them is still ``get_tool_details``' job. In
    the messages a call sends, the exchanges of tools it does not carry are folded to plain sentences.

    No failure here leaves the model without tools. A summary that cannot be written falls back to a
    boundary truncation, a search that raises returns guidance, and any failure on the rewrite path
    degrades to the request as received, which is the behaviour without the middleware.

    Example:
        ```python
        from langchain.agents import create_agent
        from langgraph_progressive_tool_disclosure import ProgressiveToolDisclosureMiddleware

        agent = create_agent(
            model="...",
            tools=[...],
            middleware=[ProgressiveToolDisclosureMiddleware(catalog_chars=80)],
        )
        ```
    """

    state_schema = DisclosureState

    def __init__(
        self,
        *,
        catalog_chars: int | None = DEFAULT_CATALOG_CHARS,
        summarizer: ToolSummarizer | None = None,
        ttl_cycles: int = DEFAULT_TTL_CYCLES,
        always_available: Sequence[str] = (),
        index: ToolIndex | None = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        """Fix the configuration of the instance. Nothing is indexed or summarized and no call goes out here.

        Args:
            catalog_chars: Character limit of one catalog line's summary, or ``None`` to add no catalog at
                all and leave the two disclosure tools' descriptions as the only hint that other tools
                exist. That is the cheapest configuration and the one with the least to go on: with no
                name to recognize, the model may well answer from what it knows instead of searching.
            summarizer: What writes a catalog line when a description does not fit ``catalog_chars``.
                Receives ``(spec, max_chars)`` and returns the summary. ``None`` uses the agent's own
                model -- one plain call per tool, with no tools, no history and no middleware, as the
                Strands plugin does -- and adds its token usage to :attr:`summary_usage`. A description
                that already fits is used verbatim either way, and a summarizer that fails falls back to
                truncation at a sentence or word boundary.
            ttl_cycles: Cycles a loaded tool survives without a call. Each call renews it.
            always_available: Names that are callable on every call, skipping the discovery cycle.
            index: Search implementation behind ``find_tools``. Defaults to
                :class:`~context_core.disclosure.LexicalToolIndex`, which needs no network.
            top_k: How many tools one search lists.

        Raises:
            ValueError: When any parameter is outside its accepted values. Every check runs before any
                state is set up, so a construction that fails registers nothing on any agent.
        """
        _validate_catalog_chars(catalog_chars)
        _validate_summarizer(summarizer)
        _validate_positive_int(ttl_cycles, "ttl_cycles")
        _validate_positive_int(top_k, "top_k")
        _validate_always_available(always_available)
        _validate_index(index)

        self._catalog_chars = catalog_chars
        self._ttl_cycles = ttl_cycles
        # A tuple, so the sequence the caller keeps cannot change the configuration after the fact.
        self._always_available = tuple(always_available)
        self._top_k = top_k
        # Instantiated but not built: building reads the specifications of a call, which the first
        # rewrite is what has.
        self._index: ToolIndex = LexicalToolIndex() if index is None else index
        self._fingerprint: frozenset[tuple[str, str]] | None = None
        # Keyed by (name, description) inside the cache, so a tool re-registered with a different
        # description gets a new line and the same description is never summarized twice.
        self._summaries = SummaryCache(summarizer)
        # No summarizer means the agent's own model writes the lines, primed into the cache before the
        # rewrite reads it. A caller-supplied summarizer is called by the cache itself.
        self._model_summarizes = summarizer is None
        self.summary_usage: dict[str, int] = {}
        """Token usage of the default summarizer: ``calls``, ``inputTokens``, ``outputTokens``. An
        auxiliary cost of the strategy, kept visible next to what it saves."""
        self.premature_cancellations = 0
        """Guessed calls the guard cancelled, over the life of this instance."""
        self.tools = self._build_tools()
        super().__init__()

    # ---------------------------------------------------------------------------------------------
    # The rewrite: one model call's tools, system prompt and messages.
    # ---------------------------------------------------------------------------------------------

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Rewrite ``tools``, ``system_message`` and ``messages`` for this one call, then hand it on.

        The only place any of the three is rewritten. Any failure on this path degrades to the request as
        received: the call goes out with every tool, which is the behaviour without the middleware.
        Nothing is remembered about the failure, so the very next call attempts the rewrite again.

        Args:
            request: The request as assembled by the agent. Never mutated — ``override`` returns a copy.
            handler: The rest of the middleware chain, ending at the model.

        Returns:
            Whatever ``handler`` returns.
        """
        try:
            self._prime_summaries(request)
            request = self._rewrite(request)
        except Exception:
            logger.warning("tool disclosure failed | passing the request through unchanged", exc_info=True)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Any],
    ) -> Any:
        """Async twin of :meth:`wrap_model_call`. Only the default summarizer's calls are awaited."""
        try:
            await self._aprime_summaries(request)
            request = self._rewrite(request)
        except Exception:
            logger.warning("tool disclosure failed | passing the request through unchanged", exc_info=True)
        return await handler(request)

    # ---------------------------------------------------------------------------------------------
    # The default summarizer: the agent's own model, called directly so no middleware sees the call.
    # ---------------------------------------------------------------------------------------------

    def _missing_summaries(self, request: ModelRequest) -> list[ToolSpec]:
        """Specs that need a model-written line: over the limit, not a disclosure tool, not cached yet."""
        if not self._model_summarizes or self._catalog_chars is None or request.model is None:
            return []
        specs = _specs_of(list(request.tools or ()))
        if not PLUGIN_TOOL_NAMES <= {spec["name"] for spec in specs}:
            return []
        return [
            spec
            for spec in specs
            if spec["name"] not in PLUGIN_TOOL_NAMES
            and len(spec.get("description") or "") > self._catalog_chars
            and self._summaries.get(spec) is None
        ]

    def _summary_messages(self, spec: ToolSpec) -> list[BaseMessage]:
        """The Strands summarizer's request: the instruction as system prompt, name and description as the ask."""
        return [
            SystemMessage(content=_SUMMARY_SYSTEM_PROMPT.format(max_chars=self._catalog_chars)),
            HumanMessage(content=f"Tool: {spec['name']}\n\nDescription:\n{spec.get('description') or ''}"),
        ]

    def _store_summary(self, spec: ToolSpec, answer: Any) -> None:
        """Account the call's usage and cache its line, falling back to truncation on an empty answer."""
        assert self._catalog_chars is not None
        self.summary_usage["calls"] = self.summary_usage.get("calls", 0) + 1
        usage = getattr(answer, "usage_metadata", None) or {}
        for ours, theirs in (("inputTokens", "input_tokens"), ("outputTokens", "output_tokens")):
            if theirs in usage:
                self.summary_usage[ours] = self.summary_usage.get(ours, 0) + int(usage[theirs])
        text = answer.text if isinstance(getattr(answer, "text", None), str) else getattr(answer, "content", "")
        line = clamp_summary(text, self._catalog_chars) or summary_line(spec, self._catalog_chars)
        self._summaries.prime(spec["name"], spec.get("description") or "", line, self._catalog_chars)

    def _store_fallback(self, spec: ToolSpec) -> None:
        """Cache the truncation for a spec whose summary call failed, so it is not retried every call."""
        assert self._catalog_chars is not None
        logger.warning("tool summary failed | tool=<%s> | falling back to truncation", spec["name"], exc_info=True)
        self._summaries.prime(
            spec["name"], spec.get("description") or "", summary_line(spec, self._catalog_chars), self._catalog_chars
        )

    def _prime_summaries(self, request: ModelRequest) -> None:
        """Write the missing catalog lines with ``request.model``, one plain call per tool."""
        for spec in self._missing_summaries(request):
            try:
                self._store_summary(spec, request.model.invoke(self._summary_messages(spec)))
            except Exception:
                self._store_fallback(spec)

    async def _aprime_summaries(self, request: ModelRequest) -> None:
        """Async twin of :meth:`_prime_summaries`, at most ``_SUMMARY_CONCURRENCY`` calls in flight."""
        missing = self._missing_summaries(request)
        if not missing:
            return
        gate = asyncio.Semaphore(_SUMMARY_CONCURRENCY)

        async def one(spec: ToolSpec) -> None:
            async with gate:
                try:
                    self._store_summary(spec, await request.model.ainvoke(self._summary_messages(spec)))
                except Exception:
                    self._store_fallback(spec)

        await asyncio.gather(*(one(spec) for spec in missing))

    def _rewrite(self, request: ModelRequest) -> ModelRequest:
        """Return a copy of ``request`` carrying only the callable tools, the catalog and the folded messages.

        Args:
            request: The request as assembled by the agent. Read only.

        Returns:
            A new request, or ``request`` itself when there is nothing to hide — no tool is bound, or the
            disclosure tools are not among those that are, which is the case where the model would have
            no way to load a hidden schema.
        """
        bound = list(request.tools or ())
        specs = _specs_of(bound)
        catalog_names = {spec["name"] for spec in specs}
        if not PLUGIN_TOOL_NAMES <= catalog_names:
            return request

        self._ensure_index(specs)

        messages = list(request.messages or ())
        # The cycle and the renewals are read off the PERSISTED history, not off ``request.messages``: a
        # middleware wrapping this one (the context graph) may have projected the call's messages down,
        # and counting cycles on a projection puts the current cycle behind the one ``get_tool_details``
        # recorded from state -- the load then never becomes callable and the model reloads forever.
        # The Strands plugin reads ``event_loop_metrics.cycle_count``, which no projection touches either.
        history = list((request.state or {}).get("messages") or messages)
        cycle = _cycle(history)
        loaded = request.state.get("loaded_tools") or {}
        active = _active_names(
            _last_used(history, loaded, cycle), cycle, self._ttl_cycles, self._always_available, catalog_names
        )

        overrides: dict[str, Any] = {
            "tools": _keep_active(bound, active),
            "messages": _fold_messages(messages, active),
        }
        block = build_catalog(specs, self._catalog_chars, active_tool_names=active, cache=self._summaries)
        if block:
            overrides["system_message"] = _with_catalog(request.system_message, block)

        logger.debug(
            "tool disclosure applied | cycle=<%d> | tools=<%d/%d> | catalog_tokens=<%d>",
            cycle,
            len(overrides["tools"]),
            len(bound),
            estimate_tokens(block),
        )
        return request.override(**overrides)

    def _ensure_index(self, specs: Sequence[ToolSpec]) -> None:
        """Build the search index when the bound tools changed since the last build.

        Not at construction time: what it covers are the specifications of a call, and the first rewrite
        is what has them. Tools can also arrive at runtime, so the ``(name, description)`` pairs are kept
        as a fingerprint and compared on every call. The fingerprint is written only after the build
        returns, so a build that raises is retried on the next call.

        Args:
            specs: Specifications of the bound tools.
        """
        fingerprint = frozenset((spec["name"], spec.get("description") or "") for spec in specs)
        if self._fingerprint == fingerprint:
            return
        self._index.build(list(specs))
        self._fingerprint = fingerprint

    # ---------------------------------------------------------------------------------------------
    # The guard: a call to a name the model could not see does not run.
    # ---------------------------------------------------------------------------------------------

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        """Cancel a call to a tool whose schema the model could not see, otherwise run it.

        A guessed call is a name the model read in the catalog and called without loading it first.
        Whether the schema was visible is recomputed for the cycle the model chose on — the cycle of the
        ``AIMessage`` that issued this call — so a ``get_tool_details`` that ran in the same batch cannot
        make the guess look sanctioned, and a tool that expired since is not mistaken for a guess either.

        Nothing is loaded on the model's behalf: a recovery that loaded the tool would teach the model
        that calling a catalog name directly works.

        Exempt: the two disclosure tools, anything in ``always_available``, and a tool with no required
        parameter, which is callable with no arguments and so cannot have been guessed at.

        Args:
            request: The tool call as assembled by the agent. Read only.
            handler: Runs the tool.

        Returns:
            A ``ToolMessage`` carrying the cancellation, or whatever ``handler`` returns.
        """
        cancellation = self._cancellation(request)
        return cancellation if cancellation is not None else handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        """Async twin of :meth:`wrap_tool_call`. The decision does no I/O, so it is shared."""
        cancellation = self._cancellation(request)
        return cancellation if cancellation is not None else await handler(request)

    def _cancellation(self, request: ToolCallRequest) -> ToolMessage | None:
        """Return the ``ToolMessage`` that cancels a guessed call, or ``None`` to let the call run.

        Any failure here returns ``None``: a guard that cannot decide lets the call through, because
        cancelling a legitimate call is the worse outcome of the two.

        Args:
            request: The tool call as assembled by the agent. Read only.

        Returns:
            The cancellation, or ``None``.
        """
        try:
            name = request.tool_call.get("name") or ""
            if not name or name in PLUGIN_TOOL_NAMES or name in self._always_available:
                return None

            bound = list(getattr(request.runtime, "tools", None) or ())
            specs = {spec["name"]: spec for spec in _specs_of(bound)}
            spec = specs.get(name)
            # A name nothing bound has is not this middleware's to judge: the agent answers it already.
            if spec is None:
                return None

            messages = list((request.state or {}).get("messages") or ())
            # The AIMessage carrying this call is already in state, so the cycle the model chose on is
            # one behind the cycle the next call will run on.
            chose_on = max(_cycle(messages) - 1, 0)
            loaded = (request.state or {}).get("loaded_tools") or {}
            active = _active_names(
                _last_used(messages, loaded, chose_on), chose_on, self._ttl_cycles, self._always_available, set(specs)
            )
            if name in active:
                return None

            # Arguments or not, the model could not have known the parameters, so either way the call is
            # a guess. A tool that needs none is exempt: an empty call to it is a legitimate call.
            if not _requires_parameters(spec):
                return None

            logger.info("guessed tool call cancelled | tool=<%s> | cycle=<%d>", name, chose_on)
            self.premature_cancellations += 1
            return ToolMessage(
                content=_PREMATURE_CALL_MESSAGE.format(name=name),
                tool_call_id=request.tool_call.get("id") or "",
                name=name,
                status="error",
            )
        except Exception:
            logger.warning("guessed-call guard failed | letting the call through", exc_info=True)
            return None

    # ---------------------------------------------------------------------------------------------
    # The two tools.
    # ---------------------------------------------------------------------------------------------

    def _build_tools(self) -> list[BaseTool]:
        """Build ``find_tools`` and ``get_tool_details``, bound to this instance.

        Closures rather than methods: the ``@tool`` decorator derives the model-facing schema from the
        signature, and a bound ``self`` in it would surface as a parameter the model is asked to fill.

        Returns:
            The two tools, in the order the catalog rule names them.
        """
        middleware = self

        @tool(FIND_TOOLS_NAME)
        def find_tools(need: str, runtime: ToolRuntime) -> str:
            """Search for tools that can do what you need, when no name in the tool catalog fits.

            This only finds tools; it does not load them. It answers with matching tool names and one
            line about each. To use any of them, call `get_tool_details` with their names, then call them.

            If a name in the catalog already fits what you need, skip this and call `get_tool_details`
            directly.

            Args:
                need: What you are trying to do, described in your own words. A capability, not a tool
                    name — "list the transactions of an investment account" works better than a guess at
                    what the tool might be called.
                runtime: Injected by the framework. Not user-facing.

            Returns:
                The matching tool names with a one-line summary of each, or guidance to describe the need
                or to reword it when there is nothing to list.
            """
            return middleware._search(need, runtime)

        @tool(GET_TOOL_DETAILS_NAME)
        def get_tool_details(names: list[str], runtime: ToolRuntime) -> Command:
            """Load the full parameters of one or more tools from the catalog, so you can call them.

            Pass every tool you are about to need in one call. They arrive complete in your tool list on
            your next call. A tool left unused for a few calls is unloaded; to call it after that, call
            this again with its name.

            Args:
                names: Exact tool names, as written in the catalog or in a `find_tools` result.
                runtime: Injected by the framework. Not user-facing.

            Returns:
                A state update recording what was loaded, and a message naming it plus any requested
                name that is not a tool.
            """
            return middleware._load(names, runtime)

        return [find_tools, get_tool_details]

    def _search(self, need: str, runtime: ToolRuntime) -> str:
        """Rank the bound tools against ``need`` and report the matches by name and summary.

        Nothing is loaded here: loading is ``get_tool_details``' one job, so the model always takes the
        same path to a schema whether it started from the catalog or from a search.

        Args:
            need: The need as the model described it.
            runtime: Tool runtime, read for the bound tool list.

        Returns:
            The matches, or guidance when the need was blank, nothing matched, or the search failed.
        """
        # A blank need cannot rank anything, so the search is not attempted at all.
        if not need or not need.strip():
            return _EMPTY_NEED_GUIDANCE

        specs = {spec["name"]: spec for spec in _specs_of(list(getattr(runtime, "tools", None) or ()))}
        self._ensure_index(list(specs.values()))

        try:
            matches: Sequence[ToolMatch] = self._index.search(need, self._top_k)  # type: ignore[assignment]
        except Exception:
            logger.warning("tool search failed | returning guidance to the model", exc_info=True)
            return _SEARCH_FAILED_GUIDANCE

        lines = [
            f"- {match.name}: {self._short_description(specs[match.name])}"
            for match in matches
            if match.name in specs and match.name not in PLUGIN_TOOL_NAMES
        ]
        logger.info("tool search outcome | need=<%s> | found=<%d>", need, len(lines))
        if not lines:
            return _NO_MATCH_GUIDANCE
        return "\n".join([_MATCHES_HEADER, *lines])

    def _load(self, names: object, runtime: ToolRuntime) -> Command:
        """Record the named tools as loaded on the current cycle, and report what was loaded.

        The specifications themselves are NOT in the answer: they travel in the next call's tool list,
        because a tool result is resident in the history while a tool list is per call and forgettable.

        Args:
            names: Names as the model passed them. A bare string and duplicates are tolerated: a model
                loading one tool does not always wrap it in a list.
            runtime: Tool runtime, read for the bound tool list, the state and the call id.

        Returns:
            A ``Command`` updating ``loaded_tools`` and appending the answer to the messages.
        """
        requested = [names] if isinstance(names, str) else list(names or ())  # type: ignore[arg-type]
        wanted = list(dict.fromkeys(n.strip() for n in requested if isinstance(n, str) and n.strip()))

        specs = {spec["name"]: spec for spec in _specs_of(list(getattr(runtime, "tools", None) or ()))}
        state = getattr(runtime, "state", None) or {}
        # The AIMessage carrying this call is already in state, so its cycle is one behind the next call's.
        cycle = max(_cycle(list(state.get("messages") or ())) - 1, 0)

        loaded: dict[str, int] = {}
        lines: list[str] = []
        unknown: list[str] = []
        for name in wanted:
            spec = specs.get(name)
            if spec is None or name in PLUGIN_TOOL_NAMES:
                unknown.append(name)
                continue
            loaded[name] = cycle
            lines.append(f"- {name}: {self._short_description(spec)}")

        if not wanted:
            answer = _DETAILS_EMPTY_GUIDANCE
        else:
            parts = [_DETAILS_LOADED_HEADER, *lines] if lines else []
            if unknown:
                parts.append(_DETAILS_UNKNOWN.format(names=", ".join(unknown)))
            answer = "\n".join(parts) or _DETAILS_EMPTY_GUIDANCE

        logger.info("tools loaded | loaded=<%d> | unknown=<%s>", len(loaded), ", ".join(unknown))
        answer_message = ToolMessage(
            content=answer,
            tool_call_id=getattr(runtime, "tool_call_id", None) or "",
            name=GET_TOOL_DETAILS_NAME,
        )
        update: dict[str, Any] = {"messages": [answer_message]}
        if loaded:
            update["loaded_tools"] = loaded
        return Command(update=update)

    def _short_description(self, spec: ToolSpec) -> str:
        """Return ``spec``'s catalog line, computing and caching it on first ask.

        Falls back to the default limit when the catalog is suppressed: ``catalog_chars=None`` drops the
        catalog from the prompt, it does not mean a search result should carry a full description.

        Args:
            spec: Specification to summarize. Left unmodified.

        Returns:
            The one-line description of the tool.
        """
        limit = DEFAULT_CATALOG_CHARS if self._catalog_chars is None else self._catalog_chars
        return self._summaries.line(spec, limit)
