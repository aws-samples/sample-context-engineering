"""Progressive tool disclosure: the projection sent on each model call.

Every call carries, in ``tool_specs``, only the tools that are callable on it: the two plugin tools
(``find_tools`` and ``get_tool_details``), the always-available ones and the ones loaded by
``get_tool_details`` and not yet used. A loaded tool is released once it has returned, so ``tool_specs``
goes back to the mandatory set instead of growing with every tool the conversation has touched. Every
other tool reaches the model as one line of a catalog in the system prompt -- its name and a summary of
its description, at most ``catalog_chars`` characters long.

A ``toolUse`` left in the history for a tool that is no longer in ``tool_specs`` is accepted by the
Bedrock Converse API (checked on Claude Haiku 4.5, Claude Opus 4.8, GLM 4.7 Flash and Qwen3 Next), so
releasing a tool never breaks the protocol.

The summary is written by a model, once per tool, and cached for the life of the plugin instance, so
the catalog is byte-stable across calls. A description that already fits the limit is used verbatim and
costs no call. Truncation at a sentence or word boundary is the fallback when a summary cannot be
produced, never the primary path.

Token counts in the logs are estimated from character counts rather than measured, with the same
four-characters-per-token heuristic ``ContextOffloader`` uses for preview slicing.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import weakref
from collections.abc import Awaitable, Callable, Container, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, TypeAlias

from strands.hooks.events import AfterToolCallEvent, BeforeToolCallEvent
from strands.plugins import Plugin, hook
from strands.tools.decorator import tool
from strands.types.content import Messages, SystemPrompt
from strands.types.tools import ToolContext, ToolSpec

from ._compat import InvokeModelStage
from .index import LexicalToolIndex, ToolIndex, ToolMatch

if TYPE_CHECKING:
    from strands.agent.agent import Agent
    from strands.models.model import Model

    from ._compat import InvokeModelContext

logger = logging.getLogger(__name__)

FIND_TOOLS_NAME = "find_tools"
"""Name of the search tool. Also the name the projection looks for to decide it can project at all."""

GET_TOOL_DETAILS_NAME = "get_tool_details"
"""Name of the loading tool: the one call that puts full specifications into the next projection.

Not ``get_details``: a bare verb-noun that generic is a name a domain tool can already hold, and a
collision would silently shadow one of the two in the registry."""

_PLUGIN_TOOL_NAMES = frozenset({FIND_TOOLS_NAME, GET_TOOL_DETAILS_NAME})
"""Both plugin tools. Carried in full on every projected call and never cancelled by the guard."""

_DEFAULT_CATALOG_CHARS = 80
"""Summary limit per catalog line, in characters. About twenty tokens, the budget that measured ~96% of
the resident saving at a low risk."""

_SUMMARY_CONCURRENCY = 8
"""Summaries requested at once when an index build meets many new tools. Bounds the burst, not the total."""

_SUMMARY_SYSTEM_PROMPT = """\
You write catalog lines for tools. You are given one tool's name and description. Reply with a single \
summary of what the tool does, at most {max_chars} characters, in the language of the description. Keep \
what tells this tool apart from similar ones: the object it acts on and what it returns. No preamble, no \
quotes, no tool name, no trailing period needed."""
"""Instruction of the default summarizer. Asks for the discriminating content, which is what the model
reading the catalog needs to decide whether to load the tool."""

ToolSummarizer: TypeAlias = "Callable[[ToolSpec, int], str | Awaitable[str]]"
"""Summarizer: receives a registered specification and the character limit, returns the summary.

May be sync or async. What it returns is clamped to the limit afterwards, so a summarizer that overruns
cannot overrun the catalog; one that raises or returns nothing falls back to truncation."""

_DEFAULT_TTL_CYCLES = 5
"""Cycles an exposure survives after its last use."""

_DEFAULT_TOP_K = 3
"""How many tools one search exposes."""

_MATCHES_HEADER = (
    f"Tools that match. Nothing is loaded yet: call `{GET_TOOL_DETAILS_NAME}` with the names you want, then call them."
)
"""Opens the search result. A search only finds; loading is the other tool's job, and this says so."""

_DETAILS_LOADED_HEADER = (
    "Loaded. These tools are callable with their full parameters on your next call. Each one is unloaded "
    f"again after it returns; to call it again later, call `{GET_TOOL_DETAILS_NAME}` again:"
)
"""Opens the loading result. The specification itself travels in ``tool_specs``, not in this text: a tool
result is resident in the history, a projection is per call and forgettable."""

_DETAILS_UNKNOWN = "Not a tool, ignored: {names}. Use names from the catalog or from `" + FIND_TOOLS_NAME + "`."
"""Line naming the requested names the registry does not have."""

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
failure is the plugin's to log, not the model's to reason about."""

_PREMATURE_CALL_MESSAGE = (
    "'{name}' did not run: it is not loaded, so its parameters are unknown to you. Call `"
    + GET_TOOL_DETAILS_NAME
    + "` with [\"{name}\"] first, then call '{name}' with its real parameters."
)
"""Cancellation message of a premature call. Names the tool and points at ``get_tool_details``. The guard
does NOT load the tool on the model's behalf: a recovery that loads it would teach the model that calling
a catalog name directly works, which is the very shortcut the catalog rule forbids."""

_CHARS_PER_TOKEN = 4
"""Approximate characters per token — same heuristic ContextOffloader uses for preview slicing."""

_ELLIPSIS = "..."
"""Marks a description as cut. Counts against the budget like any other character."""

_SENTENCE_ENDINGS = ".!?"
"""Characters that end a sentence when followed by whitespace or by the end of the text."""


def _estimate_tokens(text: str) -> int:
    """Estimate the token count of ``text`` from its character count.

    Rounds up, matching ``_heuristic_estimate_text`` in ``models/model.py``: a non-empty text never
    estimates as zero tokens.

    Args:
        text: Text to estimate.

    Returns:
        Estimated token count.
    """
    return math.ceil(len(text) / _CHARS_PER_TOKEN)


_CATALOG_PROMPT_HEADER = """\
# Tools available on request

The tools listed below are NOT in your tool list, and you MUST NOT call them directly: their parameters
are not loaded, and a direct call is rejected without running.

To use any of them, always follow these steps:
1. Call `{get_tool_details}` with the names you need, as a list, in one call.
2. On your next call they are in your tool list with their full parameters. Call them from there.
3. Each tool is unloaded again after it returns. To call it again later, repeat step 1.

If no name below fits what you need, call `{find_tools}` with the need in your own words, then go to
step 1 with the names it returns.

The tools that ARE in your tool list for this call you call directly.

"""
"""Preamble of the system-prompt catalog: the rule, stated where the model reads the names.

The names are not in ``tool_specs`` at all, so nothing asserts they are callable, and the rule that
governs them arrives in the same block. The common path is catalog -> ``get_tool_details`` -> call;
``find_tools`` is the fallback for a need the model cannot map to a listed name.
"""


def _catalog_prompt_block(
    incoming: Sequence[ToolSpec],
    full_spec_names: Container[str],
    summaries: Mapping[str, str],
) -> str:
    """Render the catalog as one system-prompt block: the rule, then ``- name: summary``.

    Every tool that is not getting a full specification on this call is listed, and NOTHING else --
    the projection and this block partition the registry between them, so no name appears in both and
    no name is missing from both.

    Args:
        incoming: Specifications received in ``context.tool_specs``, in arrival order. Read only.
        full_spec_names: Names that ARE carrying a full specification on this call, and so must not be
            listed here.
        summaries: Catalog line of each tool, by name. A name without one is listed by name only.

    Returns:
        The block, or ``""`` when every incoming tool is already carrying a full specification -- an
        empty catalog must add nothing to the prompt rather than a header promising a list.
    """
    lines = []
    for spec in incoming:
        name = spec["name"]
        if name in full_spec_names:
            continue
        summary = summaries.get(name, "")
        lines.append(f"- {name}: {summary}" if summary else f"- {name}")
    if not lines:
        return ""

    header = _CATALOG_PROMPT_HEADER.format(find_tools=FIND_TOOLS_NAME, get_tool_details=GET_TOOL_DETAILS_NAME)
    return header + "\n".join(lines)


def _clamp_summary(text: object, max_chars: int) -> str:
    """Normalize a summarizer's answer to one line of at most ``max_chars`` characters.

    Whitespace is collapsed and wrapping quotes are dropped, because a model asked for a bare line still
    answers with one sometimes. An answer over the limit is cut at a boundary rather than rejected: it is
    still a summary, and the cut is the same one the fallback would make.

    Args:
        text: What the summarizer returned. Anything that is not a string is treated as empty.
        max_chars: Character limit.

    Returns:
        The clamped line, or ``""`` when there was nothing usable.
    """
    if not isinstance(text, str):
        return ""
    line = " ".join(text.split()).strip("\"'`")
    return _truncate_description(line, max_chars) if line else ""


async def _summarize(spec: ToolSpec, max_chars: int, summarizer: ToolSummarizer) -> tuple[str, bool]:
    """Produce the catalog line of ``spec``: verbatim, summarized, or truncated, in that order of preference.

    A description that already fits the limit is the best possible summary of itself and costs no call.
    A longer one goes to ``summarizer``. A summarizer that raises or answers nothing falls back to the
    boundary truncation, so no tool is ever left without a line and no failure escapes the projection.

    Args:
        spec: Full specification as registered. Left unmodified.
        max_chars: Character limit of the line.
        summarizer: What writes the summary.

    Returns:
        The line, and whether ``summarizer`` was actually called for it.
    """
    description = spec.get("description") or ""
    if len(description) <= max_chars:
        return " ".join(description.split()), False

    try:
        answered = summarizer(spec, max_chars)
        if inspect.isawaitable(answered):
            answered = await answered
        line = _clamp_summary(answered, max_chars)
    except Exception:
        logger.warning("tool summary failed | tool=<%s> | falling back to truncation", spec["name"], exc_info=True)
        line = ""

    return (line or _truncate_description(description, max_chars)), True


def _model_summarizer(model: Model, usage: dict[str, int]) -> ToolSummarizer:
    """Build the default summarizer: one plain call to ``model`` per tool, no tools, no history.

    The call goes to ``model.stream`` directly rather than through an agent, so it passes through no
    agent middleware -- this plugin's projection included -- and cannot recurse into itself. Its token
    usage is added to ``usage``, because a summary is an auxiliary cost of the strategy and has to be
    visible next to what it saves.

    Args:
        model: Model to summarize with -- by default the agent's own.
        usage: Counters to accumulate into: ``calls``, ``inputTokens``, ``outputTokens``.

    Returns:
        An async summarizer.
    """

    async def summarize(spec: ToolSpec, max_chars: int) -> str:
        messages: Messages = [
            {"role": "user", "content": [{"text": f"Tool: {spec['name']}\n\nDescription:\n{spec['description']}"}]}
        ]
        parts: list[str] = []
        usage["calls"] = usage.get("calls", 0) + 1
        async for event in model.stream(messages, None, _SUMMARY_SYSTEM_PROMPT.format(max_chars=max_chars)):
            delta = event.get("contentBlockDelta", {}).get("delta", {})
            if "text" in delta:
                parts.append(delta["text"])
            reported: dict[str, Any] = dict(event.get("metadata", {}).get("usage") or {})
            for key in ("inputTokens", "outputTokens"):
                if key in reported:
                    usage[key] = usage.get(key, 0) + int(reported[key])
        return "".join(parts)

    return summarize


def _append_to_system_prompt(system_prompt: SystemPrompt, block: str) -> SystemPrompt:
    """Append ``block`` to ``system_prompt``, preserving whichever of its three shapes arrived.

    ``SystemPrompt`` is ``str | list[SystemContentBlock] | None``. A list is extended with one new text
    block rather than flattened to a string, because a caller using the list form is usually doing so to
    place cache checkpoints between blocks, and collapsing it would move them.

    Appending rather than prepending is deliberate: the caller's own prompt keeps the opening position,
    and on a provider that caches by prefix the operator's text stays at a stable offset.

    Args:
        system_prompt: The prompt as received on the invocation context. Not mutated -- the list form
            is copied.
        block: Text to append. An empty block returns ``system_prompt`` unchanged, by identity.

    Returns:
        The extended prompt, in the shape it arrived in.
    """
    if not block:
        return system_prompt
    if system_prompt is None:
        return block
    if isinstance(system_prompt, str):
        return f"{system_prompt}\n\n{block}"
    return [*system_prompt, {"text": block}]


def _requires_parameters(spec: ToolSpec) -> bool:
    """Report whether ``spec`` declares at least one required parameter.

    Only the ``required`` list decides. A tool whose parameters are all optional is callable with no
    arguments, so an empty call to it is a legitimate call and not the symptom of a missing schema.

    Args:
        spec: Full specification as registered in the ``ToolRegistry``. Left unmodified — only read.

    Returns:
        ``True`` when the schema lists at least one required parameter; ``False`` when it lists none,
        and for any schema shape this cannot read.
    """
    # Typed as object so the runtime guard against malformed data is not read as unreachable:
    # ToolSpec declares inputSchema as a dict, but this reads specs that may arrive malformed.
    input_schema: object = spec.get("inputSchema")
    if not isinstance(input_schema, dict):
        return False

    # inputSchema arrives wrapped as {"json": {...}}; tolerate an unwrapped schema as well.
    root = input_schema.get("json", input_schema)
    if not isinstance(root, dict):
        return False

    required = root.get("required")
    return isinstance(required, list) and len(required) > 0


def _truncate_description(text: str, max_chars: int) -> str:
    """Cut ``text`` to at most ``max_chars`` characters, preferring a sentence/word boundary.

    The FALLBACK of a catalog line, used only when a summary could not be produced, and the clamp applied
    to a summary that overran. The cut is attempted at the last sentence boundary that fits, then at the
    last word boundary, and only by character count when no word boundary fits — the case of a single
    word longer than the whole limit, where any boundary-based cut would return nothing.

    A cut at a sentence boundary reads as a whole and carries no ellipse; a cut mid-sentence gets
    one, and the ellipse counts against the limit like any other character.

    Args:
        text: Text to cut.
        max_chars: Character limit. Must be at least ``1``.

    Returns:
        A prefix of ``text``, possibly followed by an ellipse, at most ``max_chars`` characters long.
        Returns ``text`` unchanged when it already fits, which covers the empty text.
    """
    if len(text) <= max_chars:
        return text

    # A sentence boundary needs no ellipse to read as a whole, so it gets the full budget.
    cut = _last_sentence_end(text, max_chars)
    if cut > 0:
        return text[:cut]

    budget = max_chars - len(_ELLIPSIS)
    if budget > 0:
        cut = _last_word_end(text, budget)
        if cut > 0:
            return text[:cut].rstrip() + _ELLIPSIS

    # No sentence and no word boundary fits the budget; cut by character count at the limit.
    return text[:max_chars]


def _last_sentence_end(text: str, budget: int) -> int:
    """Find the end of the last sentence of ``text`` that fits ``budget`` characters.

    A sentence ends at a terminator followed by whitespace or by the end of ``text``, so that a
    period inside ``v1.2`` or ``e.g.`` is not mistaken for one.

    Args:
        text: Text to scan.
        budget: Maximum number of characters the result may span.

    Returns:
        Number of characters to keep, terminator included, or ``-1`` when no sentence ends within
        the budget.
    """
    for i in range(min(budget, len(text)) - 1, -1, -1):
        if text[i] in _SENTENCE_ENDINGS and (i + 1 >= len(text) or text[i + 1].isspace()):
            return i + 1
    return -1


def _last_word_end(text: str, budget: int) -> int:
    """Find the end of the last whole word of ``text`` that fits ``budget`` characters.

    Args:
        text: Text to scan.
        budget: Maximum number of characters the result may span.

    Returns:
        Number of characters to keep, trailing whitespace excluded from the word itself, or ``-1``
        when no word ends within the budget.
    """
    # A word may end exactly at the budget: the character just past it decides, not the budget.
    for i in range(min(budget, len(text) - 1), -1, -1):
        if text[i].isspace():
            return i
    return -1


def _validate_positive_int(value: object, parameter: str) -> None:
    """Reject anything that is not an integer greater than or equal to ``1``.

    ``bool`` is rejected explicitly: it passes as an integer in Python, and ``ttl_cycles=True``
    silently meaning "one cycle" is the kind of configuration that looks like it works. A float is
    rejected as well, including ``5.0``: the value is counted in cycles and compared to a cycle
    counter, so a non-integer has no meaning here.

    The parameter is typed ``object`` so the checks run on what the caller actually passed rather
    than on what the annotation promised — a wrong type is exactly the case this exists to catch.

    Args:
        value: Value received by the constructor.
        parameter: Name of the parameter, for the message.

    Raises:
        ValueError: When ``value`` is not an ``int`` of at least ``1``.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{parameter}=<{value!r}> | must be an integer greater than or equal to 1")


def _validate_catalog_chars(catalog_chars: object) -> None:
    """Reject anything that is neither ``None`` nor an integer greater than or equal to ``1``.

    ``None`` suppresses the catalog altogether, which is a supported configuration; ``0`` is not, as
    a limit of zero characters would emit catalog lines nothing fits in.

    Args:
        catalog_chars: Value received by the constructor.

    Raises:
        ValueError: When ``catalog_chars`` is neither ``None`` nor an ``int`` of at least ``1``.
    """
    if catalog_chars is None:
        return
    if isinstance(catalog_chars, bool) or not isinstance(catalog_chars, int) or catalog_chars < 1:
        raise ValueError(f"catalog_chars=<{catalog_chars!r}> | must be None or an integer greater than or equal to 1")


def _validate_summarizer(summarizer: object) -> None:
    """Reject anything that is neither ``None`` nor callable.

    Args:
        summarizer: Value received by the constructor.

    Raises:
        ValueError: When ``summarizer`` is not ``None`` and is not callable.
    """
    if summarizer is None or callable(summarizer):
        return
    raise ValueError(f"summarizer=<{summarizer!r}> | must be None or a callable taking (spec, max_chars)")


def _validate_always_available(always_available: object) -> None:
    """Reject anything that is not a sequence of non-empty strings.

    A bare string is rejected even though it is a sequence of strings: ``always_available="current_time"``
    would silently configure one name per character, so the mistake is caught rather than honored.

    Args:
        always_available: Value received by the constructor.

    Raises:
        ValueError: When ``always_available`` is not a sequence, is a string, or holds anything other
            than strings of length greater than zero.
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

    The protocol is checked by member rather than by ``isinstance``: :class:`ToolIndex` is a
    structural protocol, so any object carrying both operations is a valid implementation, and a
    duplicate used in tests never has to inherit from anything.

    Args:
        index: Value received by the constructor.

    Raises:
        ValueError: When ``index`` is not ``None`` and lacks a callable ``build`` or ``search``.
    """
    if index is None:
        return
    for member in ("build", "search"):
        if not callable(getattr(index, member, None)):
            raise ValueError(f"index=<{index!r}> | must expose a callable '{member}' member")


@dataclass
class _DisclosureState:
    """Per-agent disclosure state: which tools are exposed, and against which registry.

    Exposure is a per-call decision, not a durable fact, so this state never reaches ``agent.state``
    or any session storage. Losing it on a process restart costs one search, and the worst case is
    today's behavior without the plugin.

    The counters live here for the same reason the exposures do: what they measure is a session,
    and a session is an agent. They are what tells whether the disclosure pays off on a given load —
    a high search count for the same need argues for preloading, a high cancellation count argues
    the catalog lines are not informative enough, and ``summary_usage`` is the auxiliary cost the
    catalog was bought with.

    Attributes:
        exposed: Tool name to the cycle count of its load. A fresh state has zero exposures.
        projected: Names carried in full by the last projection -- what the model could actually see
            when it chose its calls. The guard reads this, not ``exposed``, so a tool released between
            two parallel calls of the same assistant message is never mistaken for a guessed call.
        consumed: Loaded tools that have returned since the last projection. The next projection
            releases them from ``exposed``, which is what keeps ``tool_specs`` from growing.
        fingerprint: ``(name, description)`` pairs as of the last index build, or ``None`` when the
            index has not been built yet — the value that makes the first projection build it.
        searches: Cycles this session spent searching: one per ``find_tools`` invocation.
        loads: Cycles this session spent loading: one per ``get_tool_details`` invocation.
        premature_cancellations: Calls this session cancelled for a schema that was not loaded.
        summary_usage: Calls and tokens the default summarizer spent on this agent's behalf. Empty when
            every description fit the limit, or when a custom summarizer was supplied.
    """

    exposed: dict[str, int] = field(default_factory=dict)
    projected: frozenset[str] = frozenset()
    consumed: set[str] = field(default_factory=set)
    fingerprint: frozenset[tuple[str, str]] | None = None
    searches: int = 0
    loads: int = 0
    premature_cancellations: int = 0
    summary_usage: dict[str, int] = field(default_factory=dict)


_DisclosureStates: TypeAlias = "weakref.WeakKeyDictionary[Agent, _DisclosureState]"
"""Per-agent state map. Weak keys so one plugin instance can serve many agents without keeping any
of them alive: the state is dropped along with the agent it belongs to."""


def _new_disclosure_states() -> _DisclosureStates:
    """Build an empty per-agent state map, to be held by the plugin instance.

    Returns:
        An empty ``WeakKeyDictionary`` keyed by agent.
    """
    return weakref.WeakKeyDictionary()


def _state_for(states: _DisclosureStates, agent: Agent) -> _DisclosureState:
    """Return ``agent``'s disclosure state, creating it on first access.

    Each agent gets its own state object, so creating or updating one agent's state leaves every
    other agent served by the same plugin instance untouched.

    Args:
        states: Per-agent state map held by the plugin instance.
        agent: Agent whose state is wanted.

    Returns:
        The state associated with ``agent``: an existing one, or a fresh state with zero exposures
        and no fingerprint.
    """
    state = states.get(agent)
    if state is None:
        state = _DisclosureState()
        states[agent] = state
    return state


def _expire(state: _DisclosureState, cycle: int, ttl_cycles: int) -> None:
    """Release every tool that has returned since the last projection, and every idle load.

    Two ways out of ``exposed``. The main one is use: a loaded tool that has returned is released on
    the next projection, so ``tool_specs`` goes back to the mandatory set once the work is done. The
    backstop is age: a tool loaded and never called is released after ``ttl_cycles`` cycles, measured
    against the cycle counter only, so a slow provider call never ages a load.

    Only the exposure map is touched. The ``ToolRegistry`` and ``agent.tool_names`` are left alone:
    releasing a tool withdraws a schema from the next projection, it does not unregister it.

    Args:
        state: Disclosure state of the agent. Mutated in place.
        cycle: Current cycle counter, ``agent.event_loop_metrics.cycle_count``.
        ttl_cycles: Cycles a load that was never used survives. At least ``1``.
    """
    for name in state.consumed:
        state.exposed.pop(name, None)
    state.consumed.clear()
    # Materialize the names first: the map is mutated while the decision is applied.
    for name in [n for n, loaded in state.exposed.items() if cycle - loaded > ttl_cycles]:
        del state.exposed[name]


def _renew(state: _DisclosureState, name: str, cycle: int) -> None:
    """Record ``cycle`` as ``name``'s last use, exposing it if it was not exposed yet.

    Renewal and first exposure are the same write: a tool used again this cycle and a tool just
    matched by search both end up with the current cycle as their last use. That is what keeps a
    repeatedly used tool's schema resident without a second search.

    Args:
        state: Disclosure state of the agent. Mutated in place.
        name: Tool name to expose or renew.
        cycle: Current cycle counter, ``agent.event_loop_metrics.cycle_count``.
    """
    state.exposed[name] = cycle


def _instrument(emit: Callable[[], None]) -> None:
    """Run one instrumentation step, swallowing whatever it raises.

    Observability is not part of the contract of any operation here: a projection, a search or a
    pre-call decision has to come out the same whether its log went out or not. So every counter
    update and every log call goes through this, and the operation around it never learns that one
    failed.

    The failure is swallowed rather than logged. A logger that raises is precisely the case this
    exists to cover, so reaching for it in the handler would reintroduce what was just guarded.

    Args:
        emit: The instrumentation step: a counter update, a log call, or both.
    """
    try:
        emit()
    except Exception:
        # Deliberately silent: see above.
        pass


def _log_projection(specs: Sequence[ToolSpec]) -> None:
    """Log the size of a projection, in specifications and in estimated tokens.

    The token count is estimated over the serialized projection, which is the closest stand-in for
    what the provider is actually charged for, using the same four-characters-per-token heuristic the
    catalog budget uses. Two calls' counts are therefore comparable to each other, and to the count
    of the same call without the plugin.

    Args:
        specs: The projected specifications.
    """
    logger.debug(
        "projection produced | specs=<%d> | estimated_tokens=<%d>",
        len(specs),
        _estimate_tokens(json.dumps(specs, default=str)),
    )


def _record_search(state: _DisclosureState) -> None:
    """Count one cycle spent searching, and log the session total.

    Args:
        state: Disclosure state of the agent. Its search counter is the only field written.
    """
    state.searches += 1
    logger.info("tool search invoked | searches_this_session=<%d>", state.searches)


def _log_search_outcome(need: str, exposed: Sequence[str]) -> None:
    """Log the need a search received and the names it exposed.

    Need and outcome go out together on purpose: repeated invocations for the same need are what
    tells a search that answered from one that sent the model back to reword, and that reading is
    only available when both sides of the invocation sit in one record.

    Args:
        need: The need as received, including the blank need of an unattempted search.
        exposed: Names exposed by this invocation, empty when nothing was.
    """
    logger.info("tool search outcome | need=<%s> | exposed=<%s>", need, ", ".join(exposed))


def _record_premature_cancellation(state: _DisclosureState, name: str) -> None:
    """Count one call cancelled for a schema that was not loaded, and log the tool.

    Args:
        state: Disclosure state of the agent. Its cancellation counter is the only field written.
        name: Name of the cancelled tool.
    """
    state.premature_cancellations += 1
    logger.info(
        "premature tool call cancelled | tool=<%s> | cancellations_this_session=<%d>",
        name,
        state.premature_cancellations,
    )


def _should_passthrough(
    incoming_names: Iterable[str],
    registry_names: Container[str],
) -> bool:
    """Decide whether the incoming call must be left exactly as it arrived.

    Two cases, both structural — neither reads a mode flag off the context:

    - A name arrives that the registry does not have. Forced structured output swaps ``tool_specs``
      for a synthetic spec that was never registered, and projecting over it would break the mode.
      Any name outside the registry is treated the same way: the projection has no full specification
      to emit for it, and dropping it would strand the caller.
    - A plugin tool is not in the call. ``init_agent`` returns before the ``_PluginRegistry`` registers
      the vended tools, so the first calls can legitimately arrive without them. Without
      ``get_tool_details`` the model has no way to load a hidden schema, and without ``find_tools`` no
      way to find one, so there is nothing to hide.

    Args:
        incoming_names: Tool names received in ``context.tool_specs``, in arrival order.
        registry_names: Names registered in the agent's ``ToolRegistry``. Membership is all that is
            asked of it.

    Returns:
        ``True`` when the caller must return the invocation context unchanged, by object identity;
        ``False`` when the projection applies.
    """
    found: set[str] = set()
    for name in incoming_names:
        if name not in registry_names:
            return True
        if name in _PLUGIN_TOOL_NAMES:
            found.add(name)

    return found != _PLUGIN_TOOL_NAMES


def _compose_projection(
    incoming: Sequence[ToolSpec],
    exposed: Container[str],
    always_available: Container[str],
) -> list[ToolSpec]:
    """Build the projection as the union of three blocks of full specifications, each name at most once.

    The blocks are visited in a fixed order — the plugin tools, ``always_available`` and the tools
    currently loaded. Every name that reaches the projection carries its full, verbatim specification:
    there is no reduced form in ``tool_specs``. Whatever is left goes to the system-prompt catalog
    instead. The plugin tools come first, which is what makes the projection non-empty on every
    projected path.

    Tools the history already used are NOT kept: a ``toolUse`` whose tool is absent from ``tool_specs``
    is accepted by the provider, and keeping them is what made ``tool_specs`` grow with every tool the
    conversation had touched.

    Inside a block, iteration follows ``incoming`` rather than the block's own container. Two calls
    with the same disclosure state and the same configuration then produce the same list in the same
    order, which keeps the provider's prompt cache from being invalidated by a reordering alone.

    Args:
        incoming: Specifications received in ``context.tool_specs``, in arrival order. Left unmodified.
        exposed: Names currently loaded, after release has been applied.
        always_available: Names configured to carry their full specification on every call.

    Returns:
        The projected specifications. Every name is a name of ``incoming`` and appears once. A name
        configured in ``always_available`` but absent from ``incoming`` is simply omitted.
    """
    projected: list[ToolSpec] = []
    seen: set[str] = set()

    for block in (_PLUGIN_TOOL_NAMES, always_available, exposed):
        for spec in incoming:
            name = spec["name"]
            if name in block and name not in seen:
                seen.add(name)
                projected.append(spec)

    return projected


def _project(
    context: InvokeModelContext,
    exposed: Container[str],
    always_available: Container[str],
    summaries: Mapping[str, str] | None,
) -> InvokeModelContext:
    """Return ``context`` with ``tool_specs`` replaced by the projection and the catalog in the prompt.

    A new context object rather than a mutation of the received one. Two fields are written, and only
    two: ``tool_specs`` carries ONLY the tools that are callable on this call, and ``system_prompt``
    carries one line for every other tool. The retained history, the ``ToolRegistry`` and every other
    field are carried over untouched — the projection changes what a call is told about, not what the
    agent has.

    Args:
        context: Invocation context received by the ``InvokeModelStage.Input`` handler.
        exposed: Names currently loaded, after release has been applied.
        always_available: Names configured to carry their full specification on every call.
        summaries: Catalog line of each tool, by name, or ``None`` to add no catalog at all.

    Returns:
        A new invocation context whose ``tool_specs`` is the projection, and whose ``system_prompt``
        carries the catalog block unless ``summaries`` is ``None`` or nothing is left to list.
    """
    projected = _compose_projection(context.tool_specs, exposed, always_available)

    if summaries is None:
        return replace(context, tool_specs=projected)

    block = _catalog_prompt_block(context.tool_specs, {spec["name"] for spec in projected}, summaries)
    return replace(
        context,
        tool_specs=projected,
        system_prompt=_append_to_system_prompt(context.system_prompt, block),
    )


class ProgressiveToolDisclosure(Plugin):
    """Send a catalog in the system prompt plus two small tools on each call, instead of every full schema.

    Every registered tool stays in the ``ToolRegistry`` and stays callable. What changes is the
    projection: ``tool_specs`` carries only the tools that are callable on the call -- ``find_tools``,
    ``get_tool_details``, the tools configured as always available and the tools loaded and not yet
    used -- and every other tool is one line of a catalog appended to the system prompt: its name and a
    summary of its description.

    The flow is catalog -> ``get_tool_details([names])`` -> call. A loaded tool is released once it has
    returned, so ``tool_specs`` goes back to the mandatory set after each use; calling it again means
    loading it again. ``find_tools`` stays for a need the model cannot map to a listed name: it searches
    and lists matches, and loading them is still ``get_tool_details``' job. A tool loaded and never
    called is released after ``ttl_cycles`` cycles.

    No failure here leaves the agent without tool specifications. A summary that cannot be produced
    falls back to a boundary truncation, a search that raises returns guidance, and a failure on the
    projection path degrades to the specifications received, which is today's behaviour without the
    plugin.

    Example:
        ```python
        from strands import Agent
        from strands_progressive_tool_disclosure import ProgressiveToolDisclosure

        agent = Agent(tools=[...], plugins=[ProgressiveToolDisclosure(catalog_chars=80)])
        ```
    """

    name = "strands:progressive-tool-disclosure"

    def __init__(
        self,
        *,
        catalog_chars: int | None = _DEFAULT_CATALOG_CHARS,
        summarizer: ToolSummarizer | None = None,
        ttl_cycles: int = _DEFAULT_TTL_CYCLES,
        always_available: Sequence[str] = (),
        index: ToolIndex | None = None,
        top_k: int = _DEFAULT_TOP_K,
    ) -> None:
        """Fix the configuration of the instance. Nothing is indexed or summarized and no call goes out here.

        Args:
            catalog_chars: Character limit of one catalog line's summary, or ``None`` to add no catalog
                at all and leave the two plugin tools' descriptions as the only hint that other tools
                exist. That is the cheapest configuration and the one with the least to go on: with no
                name to recognize, the model may well answer from what it knows instead of searching.
            summarizer: What writes a catalog line when a description does not fit ``catalog_chars``.
                Receives ``(spec, max_chars)`` and returns the summary, sync or async. ``None`` uses the
                agent's own model, one plain call per tool, once per plugin instance; its token usage is
                reported in the agent's disclosure state as ``summary_usage``. A description that
                already fits is used verbatim and never reaches the summarizer, and a summarizer that
                fails falls back to truncation at a sentence or word boundary.
            ttl_cycles: Cycles a tool loaded by ``get_tool_details`` and never called survives. A tool
                that is called is released as soon as it returns, whatever this says.
            always_available: Names that carry their full specification on every call, skipping the
                discovery cycle.
            index: Search implementation behind ``find_tools``. Defaults to :class:`LexicalToolIndex`,
                which needs no network.
            top_k: How many tools one search lists.

        Raises:
            ValueError: When any parameter is outside its accepted values. Every check runs before
                any state is set up, so a construction that fails leaves no handler, hook or tool
                registered on any agent.
        """
        _validate_catalog_chars(catalog_chars)
        _validate_summarizer(summarizer)
        _validate_positive_int(ttl_cycles, "ttl_cycles")
        _validate_positive_int(top_k, "top_k")
        _validate_always_available(always_available)
        _validate_index(index)

        self._catalog_chars = catalog_chars
        self._summarizer = summarizer
        self._ttl_cycles = ttl_cycles
        # A tuple, so the sequence the caller keeps cannot change the configuration after the fact.
        self._always_available = tuple(always_available)
        self._top_k = top_k
        # The index is only instantiated here, never built: building reads the specifications of a
        # call, which the first projection is what has.
        self._index: ToolIndex = LexicalToolIndex() if index is None else index
        # Keyed by (name, description): a tool re-registered with a different description gets a new
        # line, and the same description is never summarized twice, whichever agent asked first.
        self._summaries: dict[tuple[str, str], str] = {}
        self._states = _new_disclosure_states()
        super().__init__()

    def init_agent(self, agent: Agent) -> None:
        """Register the projection handler on the agent's ``InvokeModelStage`` input phase.

        One handler per agent, and nothing else: ``Plugin`` auto-registers ``@hook`` and ``@tool``
        members, but not middleware, so this is the whole hook-up.

        A single instance may be registered on several agents. The handler is the same bound method
        on each, but the disclosure state it reads is keyed by the agent of the call, so exposures
        never cross over. Summaries are shared, because a summary depends on the description alone.

        The plugin tools and the pre-call hook are registered by the ``_PluginRegistry`` *after* this
        returns, so the handler cannot assume they are in the registry on its first calls —
        :func:`_should_passthrough` is what covers that window.

        Args:
            agent: Agent being initialized.
        """
        agent._middleware_registry.add_middleware(InvokeModelStage.Input, self._projection_handler)

    async def _projection_handler(self, context: InvokeModelContext) -> InvokeModelContext:
        """Rewrite ``context.tool_specs`` and append the catalog to ``context.system_prompt``, for this call.

        The only place either field is ever rewritten. Any failure on this path degrades to the context
        received, unchanged: the call goes out with the full ``tool_specs``, which is exactly today's
        behaviour without the plugin. Nothing escapes to the stage, and no failure state is kept: the
        very next model call attempts the projection again.

        Args:
            context: Invocation context received from the stage.

        Returns:
            A new context carrying the projection, or ``context`` itself when the projection does not
            apply or fails.
        """
        try:
            agent = context.agent
            registry = agent.tool_registry.registry

            if _should_passthrough((spec["name"] for spec in context.tool_specs), registry):
                return context

            state = _state_for(self._states, agent)
            _expire(state, agent.event_loop_metrics.cycle_count, self._ttl_cycles)

            await self._ensure_index(state, context.tool_specs, agent)

            projected = _project(
                context,
                state.exposed,
                self._always_available,
                None if self._catalog_chars is None else self._summaries_for(context.tool_specs),
            )
            state.projected = frozenset(spec["name"] for spec in projected.tool_specs)
            _instrument(lambda: _log_projection(projected.tool_specs))
            return projected
        except Exception:
            logger.warning("projection failed | passing the received context through unchanged", exc_info=True)
            return context

    async def _ensure_index(self, state: _DisclosureState, specs: Sequence[ToolSpec], agent: Agent) -> None:
        """Build the index and the missing summaries when the incoming tool names changed.

        Neither can happen at construction time: what they cover are the specifications of a call, and
        the first projection is what has them. MCP tools and ``register_dynamic_tool`` can arrive at
        runtime, and a tool can be re-registered with a new description, so the set of incoming
        ``(name, description)`` pairs is kept as a fingerprint and compared on every projection.

        Summaries are only requested for tools that do not have one yet, concurrently and bounded by
        :data:`_SUMMARY_CONCURRENCY`. They are written before the fingerprint, and the fingerprint only
        after the index build returns, so a build that raises is retried on the next projection.

        Args:
            state: Disclosure state of the agent. Its fingerprint and summary usage are written.
            specs: Specifications received in ``context.tool_specs``.
            agent: Agent of the call, whose model the default summarizer uses.
        """
        fingerprint = frozenset((spec["name"], spec.get("description") or "") for spec in specs)
        if state.fingerprint == fingerprint:
            return

        if self._catalog_chars is not None:
            await self._summarize_missing(specs, self._catalog_chars, agent, state)

        # The index may be a network-backed implementation, so build is allowed to be awaitable.
        built = self._index.build(list(specs))
        if inspect.isawaitable(built):
            await built

        state.fingerprint = fingerprint

    async def _summarize_missing(
        self, specs: Sequence[ToolSpec], max_chars: int, agent: Agent, state: _DisclosureState
    ) -> None:
        """Fill the summary cache for every catalog-eligible spec that has no line yet.

        Args:
            specs: Specifications received in ``context.tool_specs``.
            max_chars: Character limit of a line.
            agent: Agent of the call.
            state: Disclosure state of the agent; the default summarizer accounts its usage here.
        """
        missing = [
            spec
            for spec in specs
            if spec["name"] not in _PLUGIN_TOOL_NAMES
            and (spec["name"], spec.get("description") or "") not in self._summaries
        ]
        if not missing:
            return

        summarizer = self._summarizer or _model_summarizer(agent.model, state.summary_usage)
        gate = asyncio.Semaphore(_SUMMARY_CONCURRENCY)

        async def one(spec: ToolSpec) -> None:
            async with gate:
                line, _ = await _summarize(spec, max_chars, summarizer)
            self._summaries[(spec["name"], spec.get("description") or "")] = line

        await asyncio.gather(*(one(spec) for spec in missing))

    def _summaries_for(self, specs: Sequence[ToolSpec]) -> dict[str, str]:
        """Return the cached catalog line of each spec, by name. A spec without one is left out.

        Args:
            specs: Specifications received in ``context.tool_specs``.

        Returns:
            Name to catalog line.
        """
        lines: dict[str, str] = {}
        for spec in specs:
            line = self._summaries.get((spec["name"], spec.get("description") or ""))
            if line is not None:
                lines[spec["name"]] = line
        return lines

    @tool(context=True)
    async def find_tools(self, need: str, tool_context: ToolContext) -> str:
        """Search for tools that can do what you need, when no name in the tool catalog fits.

        This only finds tools; it does not load them. It answers with matching tool names and one line
        about each. To use any of them, call `get_tool_details` with their names, then call them.

        If a name in the catalog already fits what you need, skip this and call `get_tool_details`
        directly.

        Args:
            need: What you are trying to do, described in your own words. A capability, not a tool
                name — "list the transactions of an investment account" works better than a guess at
                what the tool might be called.
            tool_context: Injected by the framework. Not user-facing.

        Returns:
            The matching tool names with a one-line summary of each, or guidance to describe the need
            or to reword it when there is nothing to list.
        """
        agent = tool_context.agent
        state = _state_for(self._states, agent)

        # Every invocation counts: a blank need and a failed search each cost the cycle just the same.
        _instrument(lambda: _record_search(state))

        # A blank need cannot rank anything, so the search is not attempted at all.
        if not need.strip():
            _instrument(lambda: _log_search_outcome(need, ()))
            return _EMPTY_NEED_GUIDANCE

        try:
            # The index may be a network-backed implementation, so search is allowed to be awaitable.
            found = self._index.search(need, self._top_k)
            matches: Sequence[ToolMatch] = await found if inspect.isawaitable(found) else found
        except Exception:
            logger.warning("tool search failed | returning guidance to the model", exc_info=True)
            _instrument(lambda: _log_search_outcome(need, ()))
            return _SEARCH_FAILED_GUIDANCE

        registry = agent.tool_registry.registry
        names: list[str] = []
        lines: list[str] = []
        for match in matches:
            registered = registry.get(match.name)
            # A match the registry does not have has nothing to load and no description to report.
            if registered is None or match.name in _PLUGIN_TOOL_NAMES:
                continue
            names.append(match.name)
            lines.append(f"- {match.name}: {self._short_description(registered.tool_spec)}")

        _instrument(lambda: _log_search_outcome(need, names))

        if not lines:
            return _NO_MATCH_GUIDANCE

        # Names and summaries only: nothing is exposed here. Loading is get_tool_details' one job, so
        # the model always takes the same path to a schema whether it started from the catalog or here.
        return "\n".join([_MATCHES_HEADER, *lines])

    @tool(context=True)
    async def get_tool_details(self, names: list[str], tool_context: ToolContext) -> str:
        """Load the full parameters of one or more tools from the catalog, so you can call them.

        Pass every tool you are about to need in one call. They arrive complete in your tool list on
        your next call. Each one is unloaded again after it returns, so to call a tool again later,
        call this again with its name.

        Args:
            names: Exact tool names, as written in the catalog or in a `find_tools` result.
            tool_context: Injected by the framework. Not user-facing.

        Returns:
            The tools that were loaded, and any requested name that is not a tool.
        """
        agent = tool_context.agent
        state = _state_for(self._states, agent)
        cycle = agent.event_loop_metrics.cycle_count
        _instrument(lambda: setattr(state, "loads", state.loads + 1))

        # Tolerate a bare string and duplicates: a model that loads one tool may not wrap it in a list.
        received: object = names
        requested = [received] if isinstance(received, str) else list(names or ())
        wanted = list(dict.fromkeys(n.strip() for n in requested if isinstance(n, str) and n.strip()))
        if not wanted:
            return _DETAILS_EMPTY_GUIDANCE

        registry = agent.tool_registry.registry
        lines: list[str] = []
        unknown: list[str] = []
        for name in wanted:
            registered = registry.get(name)
            if registered is None:
                unknown.append(name)
                continue
            _renew(state, name, cycle)
            lines.append(f"- {name}: {self._short_description(registered.tool_spec)}")

        _instrument(lambda: logger.info("tools loaded | loaded=<%s> | unknown=<%s>", len(lines), ", ".join(unknown)))

        parts = [_DETAILS_LOADED_HEADER, *lines] if lines else []
        if unknown:
            parts.append(_DETAILS_UNKNOWN.format(names=", ".join(unknown)))
        return "\n".join(parts)

    @hook  # type: ignore[call-overload]  # sync hook method; the @hook overloads only infer async
    def _on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
        """Cancel a call to a tool whose schema the model could not see.

        A premature call is a name the model read in the catalog and called without loading it first.
        Whether the schema was visible is read off ``state.projected`` -- the names the last projection
        actually carried -- rather than off ``exposed``, which a parallel call of the same assistant
        message may already have released.

        The call is cancelled with a message pointing at ``get_tool_details``, and nothing is loaded on
        the model's behalf: a recovery that loaded the tool would teach the model that calling a catalog
        name directly works.

        Exempt: anything in ``always_available``, the two plugin tools, and a tool with no required
        parameter, which is callable empty. A name the registry does not have is left alone entirely.

        Args:
            event: The pre-call event. Only ``cancel_tool`` is ever written.
        """
        name = event.tool_use["name"]
        agent = event.agent
        registry = agent.tool_registry.registry

        if name not in registry or name in _PLUGIN_TOOL_NAMES or name in self._always_available:
            return

        state = _state_for(self._states, agent)
        if name in state.projected:
            return

        # Arguments or not, the model could not have known the parameters, so either way the call is a
        # guess: invented arguments against a permissive tool are the worse outcome, a confidently wrong
        # answer nothing in the run marks as suspect.
        if _requires_parameters(registry[name].tool_spec):
            event.cancel_tool = _PREMATURE_CALL_MESSAGE.format(name=name)
            _instrument(lambda: _record_premature_cancellation(state, name))

    @hook  # type: ignore[call-overload]  # sync hook method; the @hook overloads only infer async
    def _on_after_tool_call(self, event: AfterToolCallEvent) -> None:
        """Mark a loaded tool that has returned, so the next projection releases it.

        The release waits for the next projection instead of happening here: two parallel calls of the
        same tool in one assistant message both have to find it loaded, and the model's next call is the
        first one that no longer needs it. A cancelled call ran nothing, so it releases nothing.

        Args:
            event: The post-call event. Read only.
        """
        if event.cancel_message is not None:
            return
        name = event.tool_use["name"]
        if name in _PLUGIN_TOOL_NAMES or name in self._always_available:
            return
        state = _state_for(self._states, event.agent)
        if name in state.exposed:
            state.consumed.add(name)

    def _short_description(self, spec: ToolSpec) -> str:
        """Return ``spec``'s catalog line: the cached summary, or a truncation when there is none.

        Falls back to the default limit when the catalog is suppressed: ``catalog_chars=None`` drops the
        catalog from the prompt, it does not mean a search result should carry a full description.

        Args:
            spec: Full specification as registered in the ``ToolRegistry``. Left unmodified.

        Returns:
            The one-line description of the tool.
        """
        cached = self._summaries.get((spec["name"], spec.get("description") or ""))
        if cached is not None:
            return cached
        limit = _DEFAULT_CATALOG_CHARS if self._catalog_chars is None else self._catalog_chars
        return _truncate_description(" ".join((spec.get("description") or "").split()), limit)
