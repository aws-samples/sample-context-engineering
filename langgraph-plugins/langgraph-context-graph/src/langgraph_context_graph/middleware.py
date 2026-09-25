"""Practice D on LangGraph: one middleware that projects the call's message list through the context graph.

What the Strands plugin spreads across three hooks and a middleware stage, LangChain v1 admits in two
places. :meth:`ContextGraphMiddleware.wrap_model_call` is the delivery surface: it converts the call's
messages to the neutral shape, hands them to :func:`context_core.graph.project` -- which closes the last
turn into a Card, scores the graph against this turn's question and hands out the body budget -- and sends
the projected list to the provider with ``request.override(messages=…)``.

:meth:`ContextGraphMiddleware.wrap_tool_call` is the artifact surface, the ``AfterToolCallEvent`` analog:
after a tool has run it records the return in the conversation's reference store and derives the artifact
Cards of any reference the return names, so ``expand_artifact`` has both an address to be asked for and
content to answer with. Two properties of the Strands hook are kept: the Card holds the **address** and the
store holds the block, so nothing in the graph rots when the content changes; and a return naming no
reference registers no artifact Card, which is the ordinary nothing-offloaded path rather than a
degradation.

The three optional host symbols :mod:`context_core.graph.store` bridges to are registered here, at the
binding boundary, because the core names no framework module of its own -- see
:func:`_register_host_symbols`.

Three properties are carried over from the Strands plugin unchanged, because they are the practice rather
than the wiring:

**The persisted history is never touched.** ``override`` is per call, so ``state["messages"]`` comes out of
a projection exactly as it went in. Nothing is deleted, which is why a Card the choice collapsed can be
raised back up: the messages it addresses are still there.

**A full pass is the identity.** ``expand_threshold=0.0`` makes the core return the *received* neutral list
by object identity; this middleware reads that identity and then calls the handler with the **original**
request, so no ``override`` is applied at all and the provider sees a call byte-for-byte identical to the
one it would see without the middleware. That is the "is this a graph problem or a pre-existing one" probe.

**Nothing derives a Card with a model.** The only remote call in any configuration is the matcher's
embedding round, one per turn, and the default matcher is built on first need so construction reaches no
network.

The graph itself travels in the agent state under ``context_graph`` (see :class:`ContextGraphState`): every
field of :class:`~context_core.graph.GraphState` is plain data, so the state *is* the serialized form and
survives a checkpointer without a codec. It is written back twice on purpose -- into ``request.state`` for
the retrieval tools of this same turn, and through a ``Command`` on the response so the checkpointer keeps
it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import warnings
from collections.abc import Coroutine, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from numbers import Real
from types import MappingProxyType
from typing import Any, TypeVar

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool, tool
from typing_extensions import NotRequired

from context_core.graph import CardChoice, GraphState, SimilarityMatcher, Thresholds, TurnChoice, project
from context_core.graph import store as core_store
from context_core.graph.cards import derive_and_register_artifacts
from context_core.graph.describe import normalize
from context_core.graph.scoring import record_reuse, titles_in_turn_order
from context_core.graph.store import (
    InMemoryReferenceStore,
    absent_message,
    estimate_tokens,
    non_textual_message,
    read_artifact,
    record_references,
    resolve_artifact,
    unknown_message,
)

from ._adapter import to_langchain_list, to_neutral_list, tool_message_to_result_block
from ._compat import (
    AgentMiddleware,
    AgentState,
    Command,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    ToolRuntime,
)

__all__ = ["ContextGraphMiddleware", "ContextGraphState"]

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


def _register_host_symbols() -> None:
    """Register this binding's host symbols with :data:`context_core.graph.store.HOST_SYMBOLS`.

    The core names no framework module of its own, so the optional bridges it reads are looked up through a
    registry a binding fills at its boundary. Called once at import, with ``setdefault``, so a caller who
    registered its own pair first keeps it.

    - ``"search_content"`` -> :func:`context_core.relevance.search._search_content`. Without it every
      ``line_range``/``pattern`` request degrades to the prose "targeted reads are unavailable", which is the
      only reason this registration exists: whole reads never needed the helper.
    - ``"extract_text"`` is deliberately **not** registered: the core publishes no neutral text-recovery
      helper (``store._extract_text`` is the consumer of the symbol, not a candidate for it), and the
      fallback -- a bare ``str`` is read as the text it plainly is -- is exactly what this binding stores.
    - ``"context_manager"`` is left empty: there is no LangGraph equivalent of the Strands ``ContextManager``
      Stash, so a reference this binding did not record itself resolves to prose naming the miss.
    """
    core_store.HOST_SYMBOLS.setdefault("search_content", ("context_core.relevance.search", "_search_content"))


_register_host_symbols()

_STATE_KEY = "context_graph"
"""Agent-state key the serialized graph travels under."""

_DEFAULT_EXPAND_THRESHOLD = 0.55
_DEFAULT_COLLAPSE_FLOOR = 0.45
_DEFAULT_DESCRIPTION_TOKENS = 100
_DEFAULT_TAGS_PER_CARD = 5
_DEFAULT_NEIGHBORS_PER_CANDIDATE = 3
_DEFAULT_BODY_BUDGET: int | None = None
_DEFAULT_MIN_CARDS = 3
_DEFAULT_LINK_THRESHOLD = 0.50
_DEFAULT_REUSE_TTL_CYCLES = 5
_DEFAULT_MAX_RETRIEVAL_CYCLES = 8
_DEFAULT_RARITY_WEIGHT = 0.70

_MAX_CANDIDATES = 5
"""Candidates ``find_context`` returns at most: a search that answers with the whole graph has re-injected
the very thing the graph collapsed."""

_PRUNING_MARKERS = ("summariz", "summaris", "prun", "trim", "compact")
"""Substrings of a middleware name that mark it as one that rewrites the persisted message list.

The LangGraph analog of the ``NullConversationManager`` precondition, matched by name because a middleware
that returns ``RemoveMessages`` declares nothing this could be checked by. ``ContextEditingMiddleware`` is
deliberately absent: it edits the call through ``override`` like this one, so it removes nothing from
``state["messages"]`` and breaks no recovery."""

_PRUNING_WARNING = (
    "context_graph=<pruning_middleware> | middleware=<{middleware}> prunes or summarizes the persisted message list | "
    "a middleware that rewrites state['messages'] removes messages this middleware only meant to fold, so raising a "
    "Card's Resolution back up then recovers nothing | drop it, or accept that a collapsed turn may be unrecoverable"
)
"""The one wiring-time notice, the analog of the Strands plugin's ``NullConversationManager`` warning.

Developer-facing, so it goes out through ``warnings.warn`` rather than the logger: it is about how the
middleware is being wired, not about something that went wrong while it ran. It degrades and never blocks
-- the warned agent is wired exactly like an unwarned one -- and the other middleware is left alone: not
removed, not reordered, not reconfigured."""

_EXHAUSTED = (
    "{tool} | this turn has already spent its {spent} retrieval calls | no further recovery is available on this turn: "
    "answer from what the summary and the messages already give you, and state plainly which part you could not verify"
)
"""Refusal once a turn hits ``max_retrieval_cycles``.

Worded as an instruction rather than an error because the failure mode it exists to stop is a model that
keeps asking: every retrieval miss on these tools answers with text, so "not found" reads as "try
differently"."""

_EXPAND_ARTIFACT_DESCRIPTION = """\
Read a stored artifact that an earlier turn of THIS conversation referred to by address.

Use this for a reference that appeared in the conversation as a placeholder standing in for content
too large to keep -- an image, a document, an export. The reference is the address that placeholder
carried.

If another tool told you it had replaced a tool result with a preview and handed you a reference,
that reference belongs to that tool, not to this one: use the tool that minted it. This one resolves
only addresses this plugin recorded, and answers by naming the miss when handed any other.

Prefer a line range or a pattern: without either, the whole artifact comes back and costs its
full token count again.

Args:
    reference: The artifact reference, copied as it was shown to you.
    runtime: Injected by the framework. Not user-facing.
    line_range: ``{"start": int, "end": int}`` to read only those lines.
    pattern: Return only the lines matching this pattern.

Returns:
    The requested part of the artifact, or an error naming what was missing."""
"""What the model reads about ``expand_artifact``: the Strands plugin's docstring verbatim, ``tool_context``
renamed to ``runtime``, which is the parameter LangChain injects the call's context through.

Held as a constant rather than written as the tool body's docstring because the tool carries two bodies --
one sync, one async -- and the text the model sees must not depend on which of them a run reaches."""


class ContextGraphState(AgentState):
    """Agent state plus the serialized context graph.

    ``NotRequired`` because a first call has no graph yet, which the core reads as a fresh one: a
    conversation with no Card is a full pass, so an agent whose state carries nothing here is delivered to
    exactly as one without the middleware.
    """

    context_graph: NotRequired[GraphState]
    """The graph carried over from the previous call. Plain data, so no codec is involved."""


def _validate_ratio(value: object, parameter: str) -> None:
    """Reject anything that is not a finite real number in the closed range ``[0.0, 1.0]``.

    ``bool`` is rejected explicitly: it passes as a number in Python, and ``True`` silently meaning ``1.0``
    is configuration that looks like it works.

    Raises:
        ValueError: When ``value`` is a bool, not a real number, not finite, or out of range.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{parameter}=<{value!r}> | must be a finite real number in the closed range 0.0 to 1.0")


def _validate_count(value: object, parameter: str, *, floor: int = 1) -> None:
    """Reject anything that is not an integer greater than or equal to ``floor``.

    Raises:
        ValueError: When ``value`` is a bool, not an ``int``, or below ``floor``.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < floor:
        raise ValueError(f"{parameter}=<{value!r}> | must be an integer greater than or equal to {floor}")


def _validate_optional_count(value: object, parameter: str) -> None:
    """Reject anything that is neither ``None`` nor an integer greater than or equal to ``1``.

    ``None`` is the explicit opt-out -- no ceiling -- so it has to stay distinguishable from a caller who
    passed nothing.

    Raises:
        ValueError: When ``value`` is neither ``None`` nor an ``int`` of at least ``1``.
    """
    if value is None:
        return
    _validate_count(value, parameter)


def _validate_matcher(matcher: object) -> None:
    """Reject anything that is neither ``None`` nor an object exposing a callable ``score``.

    Checked by member, not by ``isinstance``: the matcher contract is structural, so any object carrying
    the operation is a valid implementation -- which is what lets a test pass a mock and reach no network.

    Raises:
        ValueError: When ``matcher`` is not ``None`` and lacks a callable ``score``.
    """
    if matcher is None:
        return
    if not callable(getattr(matcher, "score", None)):
        raise ValueError(f"matcher=<{matcher!r}> | must expose a callable 'score' member")


class ContextGraphMiddleware(AgentMiddleware):
    """Project an agent's short-term memory as a graph of Cards, once per model call.

    The graph derives a Card per closed turn by deterministic scan and decides a Resolution per Card --
    Title, Description or Full Content -- against this turn's question. Only the projected list reaches the
    provider; ``state["messages"]`` is left whole, so ``expand_card`` and ``find_context`` can raise a Card
    the choice collapsed and have something to raise it to.

    Do **not** co-install a summarization or pruning middleware that rewrites ``state["messages"]``. That is
    the LangGraph form of the ``NullConversationManager`` precondition: a middleware that physically removes
    a message removes what this one only meant to fold, and no Resolution recovers it. Pass the agent's
    middleware list as ``middleware=`` and one notice is emitted at construction when it holds such a
    middleware; the wiring is not otherwise changed.

    Args:
        expand_threshold: Note at or above which a Card is Full Content, budget permitting. Defaults to
            ``0.55``. ``0.0`` is the regression short circuit: every Card goes at Full Content and the
            provider sees the call it would see with no middleware installed at all.
        collapse_floor: Note below which a Card keeps only its Title. Defaults to ``0.45``. Must not exceed
            ``expand_threshold``.
        description_tokens: Token ceiling of a Description. Defaults to ``100``.
        tags_per_card: How many identifiers define a Card. Defaults to ``5``.
        neighbors_per_candidate: How many ``similar`` neighbours ``find_context`` lists under each candidate.
            Defaults to ``3``. It answers a question the ranking cannot: candidates are scored against the
            *question* and never against each other, while the edge already holds that relation. ``0`` lists
            none.
        body_budget: Token ceiling across Cards in Full Content, or ``None`` for no ceiling. Defaults to
            ``None``.
        min_cards: Below this many Cards the choice is skipped entirely, before the matcher is reached.
            Defaults to ``3``.
        link_threshold: Similarity at or above which two Cards link. Defaults to ``0.50``.
        reuse_ttl_cycles: Turns a fed-back note survives. Defaults to ``5``. ``0`` discards it at the end of
            the turn that fed it back.
        max_retrieval_cycles: Retrieval calls one turn may spend before the tools refuse and tell the model
            to answer from what it has. Defaults to ``8``. ``None`` restores unbounded retrieval.
        include_artifact_tool: Register ``expand_artifact``. Defaults to ``True``. Beside a middleware that
            offloads tool results -- typically ``RelevanceFilterMiddleware`` -- pass that middleware's
            ``stash`` so both retrieval tools read the same content; without it each ships a tool over a store
            the other cannot read. ``expand_card`` and ``find_context`` are unaffected and have no switch: they
            reach back into the conversation's own turns, which is a job no offloader does.
        rarity_weight: How much rarity counts when Tags are selected. Defaults to ``0.70``.
        stash: Second layer for ``expand_artifact``, asked for a reference this binding's own store does not
            hold -- anything with an awaitable ``retrieve(reference)`` answering text. Pass
            ``RelevanceFilterMiddleware.stash`` so the ``[ref: mem_N_...]`` the filter mints resolves here
            too; this is the role the ``ContextManager`` Stash plays for the Strands plugin. ``None`` (the
            default) keeps the own store as the only layer, as in a Strands agent with no manager.
        matcher: Similarity matcher, or ``None`` for the default embedding matcher. Checked by member, so an
            implementation inherits from nothing, and resolved on first need, so construction opens no
            client.
        middleware: The agent's middleware list, for the one wiring-time notice. Optional: passing nothing
            says nothing about the wiring and warns about nothing.

    Raises:
        ValueError: On any invalid argument, naming the parameter and what it accepts.

    Example:
        ```python
        from langchain.agents import create_agent
        from langgraph_context_graph import ContextGraphMiddleware

        graph = ContextGraphMiddleware()
        agent = create_agent(model="...", tools=[...], middleware=[graph])
        ```
    """

    state_schema = ContextGraphState

    def __init__(
        self,
        *,
        expand_threshold: float = _DEFAULT_EXPAND_THRESHOLD,
        collapse_floor: float = _DEFAULT_COLLAPSE_FLOOR,
        description_tokens: int = _DEFAULT_DESCRIPTION_TOKENS,
        tags_per_card: int = _DEFAULT_TAGS_PER_CARD,
        neighbors_per_candidate: int = _DEFAULT_NEIGHBORS_PER_CANDIDATE,
        body_budget: int | None = _DEFAULT_BODY_BUDGET,
        min_cards: int = _DEFAULT_MIN_CARDS,
        link_threshold: float = _DEFAULT_LINK_THRESHOLD,
        reuse_ttl_cycles: int = _DEFAULT_REUSE_TTL_CYCLES,
        max_retrieval_cycles: int | None = _DEFAULT_MAX_RETRIEVAL_CYCLES,
        rarity_weight: float = _DEFAULT_RARITY_WEIGHT,
        include_artifact_tool: bool = True,
        stash: object | None = None,
        matcher: SimilarityMatcher | None = None,
        middleware: Iterable[object] | None = None,
    ) -> None:
        """Validate the configuration, fix it for the lifetime of the instance, and build the tools.

        Every check runs before the first attribute is assigned, so a ``ValueError`` leaves an instance that
        was never handed to an agent. Nothing is built here beyond plain attributes and the tool objects: no
        network call, no model client, no matcher, and no reference store -- the store of a conversation is
        created the first time a tool call is seen on it.
        """
        _validate_ratio(expand_threshold, "expand_threshold")
        _validate_ratio(collapse_floor, "collapse_floor")
        _validate_ratio(link_threshold, "link_threshold")
        _validate_ratio(rarity_weight, "rarity_weight")
        # Checked after both are known to be ratios: a floor above the ceiling leaves the middle resolution
        # unreachable, so the ladder would have two steps while the configuration says three.
        if float(collapse_floor) > float(expand_threshold):
            raise ValueError(
                f"collapse_floor=<{collapse_floor!r}> | must be less than or equal to "
                f"expand_threshold=<{expand_threshold!r}>"
            )
        _validate_count(description_tokens, "description_tokens")
        _validate_count(tags_per_card, "tags_per_card")
        _validate_count(neighbors_per_candidate, "neighbors_per_candidate", floor=0)
        _validate_count(reuse_ttl_cycles, "reuse_ttl_cycles", floor=0)
        _validate_count(min_cards, "min_cards")
        _validate_optional_count(body_budget, "body_budget")
        _validate_optional_count(max_retrieval_cycles, "max_retrieval_cycles")
        if not isinstance(include_artifact_tool, bool):
            raise ValueError(f"include_artifact_tool=<{include_artifact_tool!r}> | must be True or False")
        if stash is not None and not callable(getattr(stash, "retrieve", None)):
            raise ValueError(f"stash=<{stash!r}> | must be None or have a retrieve(reference) method")
        _validate_matcher(matcher)

        super().__init__()

        self._stash = stash
        self._collapse_floor = float(collapse_floor)
        self._neighbors_per_candidate = int(neighbors_per_candidate)
        self._body_budget = body_budget
        self._reuse_ttl_cycles = int(reuse_ttl_cycles)
        self._max_retrieval_cycles = max_retrieval_cycles
        self._include_artifact_tool = include_artifact_tool
        self._matcher = matcher
        self._resolved_matcher: SimilarityMatcher | None = None

        # One reference store per conversation, held on the instance and never persisted -- the same shape
        # the Strands plugin's weakly keyed per-agent map has, and for the same reason: a reference
        # discovered in one conversation means nothing in another. It is deliberately NOT agent state.
        # LangGraph deep-copies every state update as it applies it (which is what ``_persistable`` exists
        # for), so a store in the state would be copied on every superstep and checkpointed at its full
        # size, and the blocks are the one part of this plugin that is content rather than addresses. A
        # process-local store therefore trades durability across processes -- which the Strands store does
        # not have either -- for not paying for the content twice. Keyed by the thread id when the run has
        # one; a run without a checkpointer has none and shares the default entry, which is exactly one
        # conversation's worth of tool calls.
        self._stores: dict[str, InMemoryReferenceStore] = {}

        self.tools: Sequence[BaseTool] = self._build_tools()
        self._thresholds = Thresholds(
            expand_threshold=float(expand_threshold),
            collapse_floor=float(collapse_floor),
            description_tokens=int(description_tokens),
            tags_per_card=int(tags_per_card),
            rarity_weight=float(rarity_weight),
            min_cards=int(min_cards),
            link_threshold=float(link_threshold),
            reuse_ttl_cycles=int(reuse_ttl_cycles),
            # Named by the final block's guidance, so the model is told about the tools it actually has.
            retrieval_tools=tuple(each.name for each in self.tools),
        )

        self._warn_on_pruning_middleware(middleware)

    # ---- the one engagement point -----------------------------------------------------------------

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Any,
    ) -> Any:
        """Project this call's messages through the graph, then let the call go out.

        Order inside the projection is the order the Strands hooks imposed -- close the last turn into a
        Card, score the graph, apply the removal and fold the descriptions -- and it is
        :func:`context_core.graph.project` that keeps it, not this method.

        On a full pass the core returns the received neutral list *by identity*, and this method then calls
        the handler with the **original** request: no ``override``, so the call is identical to the
        no-middleware one rather than merely equal to it.

        The updated graph is written twice. Into ``request.state``, so ``expand_card`` and ``find_context``
        called out of this very turn see the graph the choice was taken from; and through a ``Command`` on
        the way out, which is the write the checkpointer keeps. Neither write touches ``messages``.

        Args:
            request: The model request. Its ``messages`` are read, never mutated.
            handler: Callback that runs the model call.

        Returns:
            The handler's response, carrying the graph-state update.
        """
        prior = self._state_of(request.state)
        neutral = to_neutral_list(list(request.messages))

        projected, new_state = project(
            neutral,
            state=prior,
            matcher=self._matcher_for(),
            body_budget=self._body_budget,
            thresholds=self._thresholds,
        )

        persisted = self._persistable(new_state)
        self._write_back(request.state, persisted)

        # Identity, not equality: the core's own regression short circuit, read here as "change nothing".
        call = request if projected is neutral else request.override(messages=to_langchain_list(projected))
        return self._with_state_update(handler(call), persisted)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Any,
    ) -> Any:
        """Async twin of :meth:`wrap_model_call`, for an agent run under ``ainvoke``/``astream``.

        LangChain does not bridge a sync ``wrap_model_call`` to an async run — it raises
        ``NotImplementedError`` — so a stack that also carries an async-only middleware (the relevance
        filter's ``awrap_tool_call``) forces the whole run async and needs this twin to exist. The
        projection (:func:`context_core.graph.project`) is pure and synchronous, so the only difference
        from the sync hook is that the model handler is awaited; the state writes and the response wrapping
        are the same synchronous helpers.

        Args:
            request: The model request. Its ``messages`` are read, never mutated.
            handler: Async callback that runs the model call.

        Returns:
            The handler's response, carrying the graph-state update.
        """
        prior = self._state_of(request.state)
        neutral = to_neutral_list(list(request.messages))

        projected, new_state = project(
            neutral,
            state=prior,
            matcher=self._matcher_for(),
            body_budget=self._body_budget,
            thresholds=self._thresholds,
        )

        persisted = self._persistable(new_state)
        self._write_back(request.state, persisted)

        call = request if projected is neutral else request.override(messages=to_langchain_list(projected))
        return self._with_state_update(await handler(call), persisted)

    # ---- the artifact surface: one tool call's return ------------------------------------------------

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Any,
    ) -> Any:
        """Let the tool run, then record what it returned as an addressable artifact.

        The ``AfterToolCallEvent`` analog, at the same point in the lifecycle: the recording happens after
        the handler has answered and never before, so what is recorded is the return an offloader behind
        this middleware has already had its say on.

        The handler's own answer is returned **untouched**, never wrapped in a ``Command``. A ``Command``
        would carry the artifact Cards into the checkpoint, but it would also make an outer offloader's
        ``isinstance(result, ToolMessage)`` guard skip the result entirely -- silently turning that plugin
        off. The Cards therefore travel on the graph object already in ``request.state``, which is the same
        object the next projection reads, and the store they address is on this instance.

        Args:
            request: The tool call, read for the call, the state and the runtime.
            handler: Runs the tool. Called exactly once.

        Returns:
            Whatever the handler returned, unchanged.
        """
        result = handler(request)
        self._record_artifacts(request, result)
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Any,
    ) -> Any:
        """Async twin of :meth:`wrap_tool_call`, for a run under ``ainvoke``/``astream``.

        LangChain does not bridge the two -- a sync ``wrap_tool_call`` raises ``NotImplementedError`` under
        an async run -- so both exist. The recording itself reaches no network and is shared.

        Args:
            request: The tool call, read for the call, the state and the runtime.
            handler: Async callable that runs the tool. Awaited exactly once.

        Returns:
            Whatever the handler returned, unchanged.
        """
        result = await handler(request)
        self._record_artifacts(request, result)
        return result

    def _record_artifacts(self, request: ToolCallRequest, result: Any) -> None:
        """Store this return's blocks and register the artifact Cards of any reference it names.

        Two writes, in the order the Strands hook makes them plus one this binding can make and Strands
        cannot:

        1. **The return's blocks go into the conversation's store**, keyed ``<tool_call_id>_<index>`` -- the
           same key format the relevance filter hands its own store. Strands records a *name with nothing
           behind it* here, because by the time its hook runs the offloader has replaced the content and
           there is no block to pair; ``expand_artifact`` then falls through to the ``ContextManager`` Stash
           for the content itself. This binding has no Stash to fall through to, so a reference that
           resolves to nothing would make the tool answerable in no configuration at all. Wrapping the tool
           is what makes the difference: the return is in hand here, so it is stored rather than named.
        2. **The artifact Cards are derived from the return**, exactly as the Strands hook derives them: off
           the references the placeholder text names, holding the address and never a copy of the content.
           A return naming none registers none, which is the ordinary nothing-offloaded path.
        3. **Those placeholder references are noted in the store**, names only, as Strands records them.

        Failures do not propagate and nothing is half-written: the graph keeps no artifact Card and the tool
        result the model receives is unaffected either way.

        Args:
            request: The tool call, read for the call's name, the state and the runtime.
            result: Whatever the handler answered with.
        """
        if not isinstance(result, ToolMessage):
            # A ``Command`` is a state update and a cancelled call carries no return: nothing to address.
            # This is also the Strands guard on a failed call, which carries an exception where a result
            # would be.
            return

        name = str((request.tool_call or {}).get("name") or "")
        if name in {each.name for each in self.tools}:
            # This middleware's own retrieval answers are not tool returns to be addressed: a whole read
            # echoes the artifact's entire text, so storing it would keep a second copy of the same content
            # under a second reference, and carding it would derive an artifact Card off our own answer.
            return

        try:
            block = tool_message_to_result_block(result)["toolResult"]
            store = self._store_for(_thread_of(getattr(request, "runtime", None)))
            texts = _stored_texts(block)
            record_references(store, [f"{result.tool_call_id}_{index}" for index in range(len(texts))], texts)

            state = self._state_of(getattr(request, "state", None))
            if state is None:
                # Nothing has projected yet, so there is no graph to card onto. The blocks are stored all
                # the same, since the reference that addresses them is minted from the call and not from
                # the graph.
                return

            messages = to_neutral_list(list(_state_messages(getattr(request, "state", None))))
            cards = derive_and_register_artifacts(
                state,
                messages,
                block,
                name,
                state.turn,
                description_tokens=self._thresholds.description_tokens,
                tags_per_card=self._thresholds.tags_per_card,
                rarity_weight=self._thresholds.rarity_weight,
            )
            record_references(store, [card.reference for card in cards if card.reference])
        except Exception:
            logger.warning("artifact card derivation failed | the graph keeps none", exc_info=True)

    def _store_for(self, thread: str) -> InMemoryReferenceStore:
        """Return the reference store of one conversation, creating it on first use.

        Args:
            thread: The thread id, or ``""`` for a run that has none.

        Returns:
            The store. Always present, even on a conversation where nothing is ever read back, in which
            case it simply holds what the tools returned and is dropped with this instance.
        """
        store = self._stores.get(thread)
        if store is None:
            store = InMemoryReferenceStore()
            self._stores[thread] = store
        return store

    # ---- state -------------------------------------------------------------------------------------

    @staticmethod
    def _state_of(state: Any) -> GraphState | None:
        """Return the graph carried in ``state``, or ``None`` for a conversation that has none yet.

        Read defensively: the state is a plain ``TypedDict`` at runtime, and a caller assembling a request
        by hand may hand over something that is merely mapping-like.
        """
        try:
            carried = state.get(_STATE_KEY) if state is not None else None
        except AttributeError:
            return None
        return carried if isinstance(carried, GraphState) else None

    @staticmethod
    def _persistable(state: GraphState) -> GraphState:
        """Make ``state`` copyable by LangGraph, and return it.

        The graph state is plain data with one exception: :class:`~context_core.graph.TurnChoice` holds its
        ``by_title`` mapping in a ``MappingProxyType``, which is what keeps the turn's decision unwritable
        for the whole turn -- and which neither ``copy.deepcopy`` nor ``pickle`` can carry. LangGraph copies
        every state update as it applies it, so a proxy reaching the agent state fails the *superstep*, not
        just the persistence.

        The mapping is therefore flattened to a plain dict on the way into the state. Nothing about the
        freeze is lost where it matters: the choice is recomputed by the next projection rather than read
        back and extended, so what the proxy protects against -- a decision shifting mid-turn -- cannot
        happen here. Mutated in place on purpose: the only states reaching this method are ones this
        binding just produced (a fresh state out of the projection) or already mutates by design (the state
        a retrieval tool elevated).
        """
        if type(state.choice.by_title) is not dict:
            state.choice = TurnChoice(
                by_title=dict(state.choice.by_title),
                full_pass=state.choice.full_pass,
                selected=state.choice.selected,
            )
        return state

    @staticmethod
    def _write_back(state: Any, new_state: GraphState) -> None:
        """Put ``new_state`` into ``state`` so this turn's retrieval tools read the graph just computed.

        Best effort by nature: this is the in-memory dict of the model node, and the durable write is the
        ``Command`` on the response. A state that refuses the assignment costs the tools of *this* turn
        their fresh graph and nothing else, so it is logged rather than raised.
        """
        try:
            state[_STATE_KEY] = new_state
        except Exception:  # pragma: no cover - a mapping that refuses assignment
            logger.debug("graph state not writable in place | the response command still carries it", exc_info=True)

    @staticmethod
    def _with_state_update(response: Any, new_state: GraphState) -> Any:
        """Attach the graph-state update to ``response``, whatever shape the handler answered in.

        ``wrap_model_call`` may be answered with a :class:`ModelResponse`, a bare ``AIMessage`` or an
        :class:`ExtendedModelResponse` an inner middleware already built. Only ``context_graph`` is ever
        written, so a command another middleware put there keeps its own keys.
        """
        if isinstance(response, ExtendedModelResponse):
            command = response.command
            if command is not None and isinstance(getattr(command, "update", None), dict):
                command.update[_STATE_KEY] = new_state
                return response
            return ExtendedModelResponse(
                model_response=response.model_response,
                command=Command(update={_STATE_KEY: new_state}),
            )

        if isinstance(response, AIMessage):
            response = ModelResponse(result=[response])

        if isinstance(response, ModelResponse):
            return ExtendedModelResponse(
                model_response=response,
                command=Command(update={_STATE_KEY: new_state}),
            )

        # An unrecognised shape is passed through untouched: the in-place write already happened, and
        # inventing a response here would cost the call.
        logger.debug("unexpected model response shape=<%s> | graph state not persisted", type(response).__name__)
        return response

    def _matcher_for(self) -> SimilarityMatcher:
        """Return the configured matcher, building the default one on first need.

        Resolved once per instance, so a rebuilt default is never two different matchers -- and never built
        at all when the caller supplied one, which is what lets a test reach no network.
        """
        if self._matcher is not None:
            return self._matcher

        if self._resolved_matcher is None:
            from context_core.graph import EmbeddingSimilarityMatcher

            # Its client is built lazily too, so this construction is still free of I/O.
            self._resolved_matcher = EmbeddingSimilarityMatcher()

        return self._resolved_matcher

    # ---- the wiring-time notice ---------------------------------------------------------------------

    def _warn_on_pruning_middleware(self, middleware: Iterable[object] | None) -> None:
        """Emit the one precondition notice when ``middleware`` holds a pruning or summarizing middleware.

        At most one notice per construction, naming the first offender: the risk is the same whichever of
        them rewrites the list, and a second line says nothing new. Passing no list warns about nothing --
        silence here is "the wiring was not described", not "the wiring is safe".
        """
        for candidate in middleware or ():
            if candidate is self:
                continue
            name = self._name_of(candidate)
            if any(marker in name.casefold() for marker in _PRUNING_MARKERS):
                warnings.warn(_PRUNING_WARNING.format(middleware=name), stacklevel=3)
                return

    @staticmethod
    def _name_of(candidate: object) -> str:
        """Return the name to match a candidate middleware by: its own ``name``, else its class name."""
        published = getattr(candidate, "name", None)
        return published if isinstance(published, str) and published else type(candidate).__name__

    # ---- the two retrieval tools -------------------------------------------------------------------

    def _build_tools(self) -> list[BaseTool]:
        """Build the retrieval tools as ordinary LangChain tools, in the Strands registration order.

        Closures over ``self`` rather than bound methods, so the schema the model sees carries the tool's
        own arguments and nothing else. All of them read and write the graph through ``ToolRuntime.state``
        and return a ``Command``, which is how a tool persists a state change in LangGraph: the elevation
        and the fed-back note have to outlive the tool call to be read by the next projection.

        ``expand_artifact`` is built with **two** bodies, one sync and one async, because LangChain bridges
        neither direction: a coroutine-only tool raises ``NotImplementedError`` under ``invoke`` and a
        sync-only one would run in a worker thread under ``ainvoke``. Its resolution is the only awaitable
        step in this package -- the store's ``retrieve`` is async by contract, since a foreign store's read
        may cross a process boundary -- so the sync body drives that one coroutine to completion itself.
        """

        @tool("expand_card")
        def expand_card(titles: list[str], runtime: ToolRuntime) -> Command:
            """Bring back the full content of one or more earlier turns, by their titles.

            Earlier turns may reach you as a title and a short description instead of their messages. When
            a description is not enough to answer, call this with the titles exactly as they were shown and
            those turns arrive in full for the rest of this turn.

            Ask for every turn you need in ONE call: a list of titles costs one retrieval where the same
            titles one at a time cost one each, and each extra call grows the conversation you are about to
            reason over.

            Args:
                titles: Titles of the turns you want back, copied as they were shown to you.

            Returns:
                Confirmation that the turns will arrive in full, or an error naming the title asked for.
            """
            state = self._graph_of(runtime)
            return self._answer(runtime, state, self.expand_card(state, titles))

        @tool("find_context")
        def find_context(need: str, runtime: ToolRuntime, tag: str | None = None) -> Command:
            """Find earlier turns of this conversation that match what you need, described in your words.

            Use this when you suspect the conversation already covered something but you cannot see it in
            what reached you. Describe the need, not a title.

            Args:
                need: What you are looking for, in your own words.
                tag: Restrict the search to turns carrying this tag.

            Returns:
                Up to five candidate turns with their title, tags and description, or an empty result
                naming the need received.
            """
            state = self._graph_of(runtime)
            return self._answer(runtime, state, self.find_context(state, need, tag))

        built: list[BaseTool] = [expand_card]
        if self._include_artifact_tool:
            built.append(self._build_artifact_tool())
        built.append(find_context)
        return built

    def _build_artifact_tool(self) -> BaseTool:
        """Build ``expand_artifact``, whose description is :data:`_EXPAND_ARTIFACT_DESCRIPTION` verbatim.

        Built only when ``include_artifact_tool`` is true, so an excluded tool is never registered rather
        than registered and then removed. The guidance block reads the registered set on every render, so
        excluding the tool also stops it being advertised, without a second switch: advertising a tool the
        agent does not hold is what sent a measured run after something it could not call.
        """

        def expand_artifact(
            reference: str,
            runtime: ToolRuntime,
            line_range: dict[str, int] | None = None,
            pattern: str | None = None,
        ) -> Command:
            """Sync body. The model reads ``_EXPAND_ARTIFACT_DESCRIPTION``, not this line."""
            state = self._graph_of(runtime)
            store = self._store_for(_thread_of(runtime))
            answer = _driven(self.expand_artifact(state, store, reference, line_range, pattern))
            return self._answer(runtime, state, answer)

        async def aexpand_artifact(
            reference: str,
            runtime: ToolRuntime,
            line_range: dict[str, int] | None = None,
            pattern: str | None = None,
        ) -> Command:
            """Async body. The model reads ``_EXPAND_ARTIFACT_DESCRIPTION``, not this line."""
            state = self._graph_of(runtime)
            store = self._store_for(_thread_of(runtime))
            answer = await self.expand_artifact(state, store, reference, line_range, pattern)
            return self._answer(runtime, state, answer)

        return StructuredTool.from_function(
            func=expand_artifact,
            coroutine=aexpand_artifact,
            name="expand_artifact",
            description=_EXPAND_ARTIFACT_DESCRIPTION,
        )

    def _graph_of(self, runtime: ToolRuntime) -> GraphState:
        """Return the graph a tool call operates on: the one in the agent state, or a fresh one.

        A fresh graph is the honest answer for a tool reached before any projection ran: it holds no Card,
        so every title misses and the model is told so, rather than being answered from a graph that does
        not describe this conversation.
        """
        return self._state_of(getattr(runtime, "state", None)) or GraphState()

    @staticmethod
    def _answer(runtime: ToolRuntime, state: GraphState, text: str) -> Command:
        """Wrap a tool's answer and the graph it changed into the state update LangGraph persists."""
        return Command(
            update={
                _STATE_KEY: ContextGraphMiddleware._persistable(state),
                "messages": [ToolMessage(content=text, tool_call_id=getattr(runtime, "tool_call_id", "") or "")],
            }
        )

    def expand_card(self, state: GraphState, titles: Sequence[str] | str) -> str:
        """Raise every Subject Card in ``titles`` to Full Content for the remainder of the turn.

        Both axes, dialogue *and* evidence. The elevation rewrites the frozen choice rather than adding a
        field to the state, so it ends with the turn: the next projection recomputes the choice from the
        graph, and the fed-back note is what carries the request across that boundary. A full pass is left
        as it is, since every Card is already at Full Content and a ``by_title`` entry would flip
        ``full_pass`` to false and cost the delivery its identity short circuit.

        Published rather than private because it is the tool's whole body, and a body reachable without a
        ``ToolRuntime`` is a body that can be tested without an agent.

        Args:
            state: The graph. ``choice``, ``reuse`` and ``retrieval_cycles`` are mutated; ``cards`` read
                only.
            titles: Titles the model asked for, as they were shown to it. A bare string is accepted as a
                single title: the schema says array, and a model that sends the scalar anyway should be
                answered rather than corrected.

        Returns:
            Confirmation naming the turns that will arrive whole, an error naming the titles that matched
            nothing, or the refusal when the turn's retrieval budget is spent.
        """
        refusal = self._exhausted("expand_card", state)
        if refusal is not None:
            return refusal

        state.retrieval_cycles += 1

        wanted = [titles] if isinstance(titles, str) else list(titles)
        if not wanted:
            return "expand_card | no title given | pass the titles you need, copied exactly as they were shown to you"

        found: list[str] = []
        missing: list[str] = []
        for title in wanted:
            card = state.cards.get(title)
            if card is None or card.kind != "subject":
                missing.append(title)
            else:
                found.append(title)

        if found and not state.choice.full_pass:
            state.choice = TurnChoice(
                by_title=MappingProxyType(
                    {
                        **state.choice.by_title,
                        **{title: CardChoice(dialogue="full", evidence="full") for title in found},
                    }
                ),
                full_pass=False,
                selected=state.choice.selected,
            )

        for title in found:
            record_reuse(state, title, state.turn, reuse_ttl_cycles=self._reuse_ttl_cycles)

        if not found:
            return (
                f"expand_card | no earlier turn of this conversation is titled {_quoted(missing)} | "
                "copy a title exactly as it was shown to you, or use find_context to describe what you need"
            )

        confirmation = (
            f"expand_card | {_quoted(found)} arrives in full for the rest of this turn, its messages and its tool "
            "results together"
        )
        if missing:
            confirmation += f" | no turn is titled {_quoted(missing)}, so nothing was raised for it"

        return confirmation

    async def expand_artifact(
        self,
        state: GraphState,
        store: InMemoryReferenceStore,
        reference: str,
        line_range: dict[str, int] | None = None,
        pattern: str | None = None,
    ) -> str:
        """Read the artifact behind ``reference``, whole or in part, through the store resolution order.

        The read is delegated and never reimplemented: :func:`~context_core.graph.store.resolve_artifact`
        consults the store, and :func:`~context_core.graph.store.read_artifact` bounds a targeted read. This
        module opens no file, resolves no path and builds no URI.

        No Resolution changes on any path, success included. The content asked for is in this answer, and
        what crosses into the next turn is the fed-back note on the artifact's Card.

        Asynchronous because the store contract is: ``retrieve`` may cross a process boundary. The sync tool
        body drives this coroutine to completion itself, since the default in-memory store never suspends.

        Args:
            state: The graph. Its ``reuse`` and ``retrieval_cycles`` are mutated.
            store: The conversation's reference store. Read only.
            reference: The artifact reference, as it was shown to the model.
            line_range: ``{"start": int, "end": int}``, 1-indexed and inclusive, or ``None``.
            pattern: Regex or keyword to keep only matching lines, or ``None``.

        Returns:
            The requested part of the artifact, or an error naming what was missing -- no storage at all, an
            unknown reference, non-textual content, or a line range outside the content -- with nothing
            recorded and no Resolution changed.
        """
        refusal = self._exhausted("expand_artifact", state)
        if refusal is not None:
            return refusal

        state.retrieval_cycles += 1

        # ``agent=None``: the core's own second-layer discovery is the Strands ``ContextManager`` Stash, which
        # LangGraph has no equivalent of. The second layer is therefore the explicit ``stash`` -- typically the
        # relevance filter's store -- and with none a reference the own store does not hold is ``"absent"``.
        resolved = await resolve_artifact(store, None, reference, stash=self._stash)
        if resolved.outcome == "absent":
            return absent_message(reference)
        if resolved.outcome == "unknown":
            return unknown_message(reference)
        if resolved.outcome != "text" or resolved.text is None:
            return non_textual_message(reference)

        if line_range is None and pattern is None:
            answer = _whole_artifact(reference, resolved.text)
        else:
            span = _span_of(line_range)
            if line_range is not None and span is None:
                return (
                    f"expand_artifact | line_range=<{line_range!r}> is not a pair of integers | pass "
                    '{"start": <int>, "end": <int>}, 1-indexed and inclusive'
                )
            try:
                answer = read_artifact(resolved.text, line_range=span, pattern=pattern)
            except ValueError as error:
                return f"expand_artifact | reference '{reference}' | {error}"

        title = _artifact_title(state, reference)
        if title is not None:
            record_reuse(state, title, state.turn, reuse_ttl_cycles=self._reuse_ttl_cycles)

        return answer

    def find_context(self, state: GraphState, need: str, tag: str | None = None) -> str:
        """Score every candidate Card's Description against ``need`` and answer with the best five.

        Scored over the same index the turn choice uses and no other: one ``score`` call against the
        resolved matcher, whose vectors come from its own cache. No index is built here. ``collapse_floor``
        is the bar rather than ``expand_threshold``, being the note below which the graph decided a Card was
        not worth a Description, so a candidate clearing it is one the graph did not dismiss.

        Args:
            state: The graph. ``reuse`` and ``retrieval_cycles`` are mutated; ``cards`` read only.
            need: What the model is looking for, in its own words.
            tag: Restrict candidates to Cards carrying this tag, compared in normalized form. ``None``
                leaves every Card a candidate.

        Returns:
            At most five candidates with their title, tags and description, or an empty result naming the
            ``need`` received, in which case no fed-back note was recorded and no Resolution changed.
        """
        refusal = self._exhausted("find_context", state)
        if refusal is not None:
            return refusal

        state.retrieval_cycles += 1

        if not need.strip():
            # An empty need scores every Description against nothing: a ranking of noise, not an answer.
            return _nothing_found(need, tag)

        titles = titles_in_turn_order(state)
        if tag is not None:
            wanted = normalize(tag)
            titles = tuple(title for title in titles if wanted and wanted in state.cards[title].tags)

        similarities = self._similarities(state, titles, need)
        if similarities is None:
            return _nothing_found(need, tag)

        passing = [title for title in titles if similarities[title] >= self._collapse_floor]
        passing.sort(key=lambda title: (-similarities[title], state.cards[title].turn, title))
        chosen = passing[:_MAX_CANDIDATES]

        if not chosen:
            return _nothing_found(need, tag)

        for title in chosen:
            record_reuse(state, title, state.turn, reuse_ttl_cycles=self._reuse_ttl_cycles)

        return _render_candidates(state, need, chosen, self._neighbors_per_candidate)

    def _exhausted(self, name: str, state: GraphState) -> str | None:
        """Return the refusal when this turn has no retrieval budget left, or ``None`` while it has.

        Checked *before* the call's own counter increment, so a ceiling of ``n`` admits exactly ``n`` calls.
        """
        if self._max_retrieval_cycles is None or state.retrieval_cycles < self._max_retrieval_cycles:
            return None
        return _EXHAUSTED.format(tool=name, spent=state.retrieval_cycles)

    def _similarities(self, state: GraphState, titles: tuple[str, ...], need: str) -> dict[str, float] | None:
        """One similarity per candidate, or ``None`` when the matcher was unusable.

        The matcher is contractually non-raising, so anything it does raise reads here as "no candidate",
        never as an exception the model has to interpret.
        """
        if not titles:
            return None

        descriptions = tuple(state.cards[title].description for title in titles)
        try:
            scores = self._matcher_for().score(need, descriptions)
            if len(scores) != len(descriptions):
                # Covers the empty answer too: with at least one candidate, empty is a length mismatch.
                raise ValueError(f"similarity count=<{len(scores)}> | expected=<{len(descriptions)}>")
            return {title: float(scores[index]) for index, title in enumerate(titles)}
        except Exception:
            logger.debug("find_context similarity unavailable for %d candidate(s)", len(descriptions), exc_info=True)
            return None


def _quoted(titles: Sequence[str]) -> str:
    """Render titles for a message, quoted and comma separated."""
    return ", ".join(f"'{title}'" for title in titles)


# ---- the artifact path: what a tool return is stored as, and how it is read back -------------------


def _state_messages(state: Any) -> list[Any]:
    """Read ``messages`` out of an agent state that may be a dict, a model, or nothing at all.

    Read defensively for the same reason ``_state_of`` is: a caller assembling a tool-call request by hand
    may hand over something that is merely mapping-like, and the artifact path must not be the reason a tool
    call fails.
    """
    if state is None:
        return []
    if isinstance(state, dict):
        return list(state.get("messages") or ())
    return list(getattr(state, "messages", None) or ())


def _thread_of(runtime: object) -> str:
    """Return the thread id of the run, or ``""`` when it has none.

    The conversation's identity as far as LangGraph states it: ``config["configurable"]["thread_id"]``, which
    a run with a checkpointer always carries and a bare ``invoke`` does not. Absent, every call of the
    process shares one store, which is exactly one conversation's worth of returns.
    """
    config = getattr(runtime, "config", None)
    configurable = config.get("configurable") if isinstance(config, dict) else None
    thread = configurable.get("thread_id") if isinstance(configurable, dict) else None
    return thread if isinstance(thread, str) and thread else ""


def _stored_texts(result_block: dict[str, Any]) -> list[str]:
    """Render the storable sub-blocks of one tool return, in order, as the text each is stored as.

    Text and JSON blocks only, which is what the relevance filter stores too and what a targeted read can
    act on: anything else has no text to search and would be reported as non-textual anyway. JSON is
    rendered the way that filter renders it, so the same return stored by either plugin reads the same.

    Index alignment is the contract of the reference: block ``n`` of the return is ``<tool_call_id>_<n>``.
    """
    texts: list[str] = []
    for block in result_block.get("content") or ():
        if not isinstance(block, dict):
            continue
        if isinstance(block.get("text"), str):
            texts.append(block["text"])
        elif "json" in block:
            try:
                texts.append(json.dumps(block["json"], indent=2))
            except (TypeError, ValueError):
                # A JSON block that does not serialize is content this store cannot address; the return
                # itself is unaffected, and the index of the blocks after it is not shifted by skipping it.
                texts.append("")
    return texts


def _driven(coroutine: Coroutine[Any, Any, _T]) -> _T:
    """Run ``coroutine`` to completion from synchronous code.

    The artifact read is the one awaitable step in this package, and a tool invoked through ``invoke`` has to
    answer without one. ``asyncio.run`` is the whole of it on the ordinary path -- LangGraph's sync tool node
    runs in a thread with no loop of its own. The worker-thread branch covers the caller who invokes the sync
    tool from inside a running loop, where ``asyncio.run`` refuses.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coroutine).result()


def _whole_artifact(reference: str, text: str) -> str:
    """The whole artifact, verbatim, with the cost of having asked for it whole stated in the answer.

    The notice is part of the contract: without it the cheapest request to write is also the most expensive
    to serve, and the model has no way to know. The text itself is untouched -- no truncation, no
    reformatting -- so the answer contains it character for character.
    """
    notice = (
        f"expand_artifact | whole artifact '{reference}' | this call re-injects the artifact's entire "
        f"token count, about {estimate_tokens(text)} tokens, and it stays in the conversation for the rest of the "
        "turn | next time pass line_range or pattern to read only the part you need"
    )
    return f"{notice}\n\n{text}"


def _span_of(line_range: dict[str, int] | None) -> tuple[int, int] | None:
    """The line range as the pair ``read_artifact`` reads, or ``None`` when it is unusable.

    The caller tells an absent range from a malformed one, since it already knows whether one was supplied.
    """
    if line_range is None:
        return None
    try:
        return (int(line_range["start"]), int(line_range["end"]))
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def _artifact_title(state: GraphState, reference: str) -> str | None:
    """Title of the artifact Card addressing ``reference``, or ``None`` when the graph holds none.

    A reference the graph never carded is still a reference the store can read -- the return may have named
    none, the store holding it under the key minted from the call -- so the read succeeds with no Card for
    the note to land on.
    """
    for title in titles_in_turn_order(state):
        card = state.cards[title]
        if card.kind == "artifact" and card.reference == reference:
            return title
    return None


def _nothing_found(need: str, tag: str | None) -> str:
    """The empty result, naming the ``need`` received and the tag it was narrowed by.

    Naming both lets the model tell "nothing in this conversation is about that" from "nothing carrying that
    tag is about that", and only the second has an obvious next move.
    """
    narrowed = f", among the turns tagged '{tag}'" if tag is not None else ""
    return (
        f"find_context | nothing in this conversation matches '{need}'{narrowed} | "
        "the titles already in front of you are the whole conversation, so what you need was "
        "either never discussed or is in a turn you can name directly with expand_card"
    )


def _similar_neighbors(state: GraphState, title: str, exclude: frozenset[str], limit: int) -> list[tuple[str, float]]:
    """The ``similar`` neighbours of ``title``, strongest first, excluding anything in ``exclude``.

    The only reader of the ``similar`` edge: it is measured on the write path and stored with its similarity
    as the weight, propagates no note by design, and without this traversal is paid for and read by nothing.

    Ordered by weight and then by title, never by turn: a neighbour is offered because it is *related* to a
    candidate, and recency is already what the candidate ordering carries.
    """
    if limit <= 0:
        return []

    neighbors = [
        (link.target, link.weight)
        for link in state.links.get(title, ())
        if link.kind == "similar" and link.target not in exclude and link.target in state.cards
    ]
    neighbors.sort(key=lambda pair: (-pair[1], pair[0]))
    return neighbors[:limit]


def _render_candidates(state: GraphState, need: str, chosen: list[str], neighbors_per_candidate: int) -> str:
    """Render the chosen candidates: title, tags and description each.

    The Description goes in full rather than trimmed, being already bounded by ``description_tokens`` at
    derivation. Neighbours are not extra candidates and are not scored against the need: they are the
    graph's own statement that two turns discuss related things, which the ranking cannot see, since it
    compares each Description to the QUESTION and never to another Description.
    """
    lines = [f"find_context | {len(chosen)} earlier turn(s) match '{need}', best first:"]
    # Every chosen title is excluded from every neighbourhood: a candidate is already being rendered in
    # full, so offering it again as somebody's neighbour would spend tokens to say nothing.
    already = frozenset(chosen)
    for title in chosen:
        card = state.cards[title]
        lines.append(f"- title: {title}")
        if card.tags:
            lines.append(f"  tags: {', '.join(card.tags)}")
        for fragment in card.description.splitlines():
            if fragment.strip():
                lines.append(f"  {fragment}")
        neighbors = _similar_neighbors(state, title, already, neighbors_per_candidate)
        if neighbors:
            rendered = ", ".join(f"{neighbor} ({weight:.2f})" for neighbor, weight in neighbors)
            lines.append(f"  related turns: {rendered}")
    lines.append("call expand_card with one of these titles to bring that turn back in full")
    return "\n".join(lines)
