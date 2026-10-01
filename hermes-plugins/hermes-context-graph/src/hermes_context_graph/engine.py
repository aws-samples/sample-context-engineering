"""The ``ContextGraphEngine``: Practice D on Hermes's synchronous ``ContextEngine`` surface.

Practice D projects the conversation's closed turns into a graph of Cards and renders each turn at the
resolution this question needs (Full Content / Description / Title), while leaving the persisted history
untouched — a mis-cut costs one recovery call, never a lost fact. Recovery tools raise a folded turn or
artifact back: ``expand_card``, ``expand_artifact``, ``find_context``.

All graph logic lives in ``context_core.graph``; this engine is the Hermes seam wiring, the engine-held
``GraphState`` and reference store, the sync driver for the async store, and the tool dispatch.

Mapping onto the Hermes ABC:

- ``on_turn_complete(messages)`` — store each tool return as an addressable artifact and derive its
  artifact Cards (the LangGraph ``wrap_tool_call`` ``_record_artifacts`` analog).
- ``select_context(request_messages)`` — run ``context_core.graph.project`` over the closed turns and
  return the projected list; request-only, persisted history untouched; fail-open.
- ``get_tool_schemas()`` / ``handle_tool_call()`` — ``expand_card`` / ``expand_artifact`` /
  ``find_context``.

Single-select bonus: because D *replaces* Hermes's default lossy ``context_compressor``, the "nothing
may delete from the history behind the graph" precondition holds structurally — there is no co-active
summarizer to fight (unlike LangGraph, where it can only warn).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
from collections.abc import Awaitable, Sequence
from types import MappingProxyType
from typing import Any, Dict, List, Optional

from context_core.graph import (
    CardChoice,
    GraphState,
    SimilarityMatcher,
    Thresholds,
    TurnChoice,
    project,
)
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

from ._adapter import hermes_to_neutral_list, neutral_to_hermes_list
from ._base import BaseEngine

__all__ = ["ContextGraphEngine", "register"]

logger = logging.getLogger(__name__)

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

_EXPAND_CARD = "expand_card"
_EXPAND_ARTIFACT = "expand_artifact"
_FIND_CONTEXT = "find_context"
_TOOL_NAMES = frozenset({_EXPAND_CARD, _EXPAND_ARTIFACT, _FIND_CONTEXT})
_EXHAUSTED = "{tool} | this turn's retrieval budget is spent ({spent} calls) | answer from what you have"


def _run_to_completion(coroutine: Awaitable[Any]) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)  # type: ignore[arg-type]
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coroutine).result()


class ContextGraphEngine(BaseEngine):
    """Context-graph projection on Hermes's ``ContextEngine`` surface.

    Construction is inert: no matcher and no reference store is built here (the store is created on the
    first tool return, the matcher on first need), so an engine that never fires costs nothing.

    Args mirror the LangGraph middleware (see its README for the semantics): ``expand_threshold``,
    ``collapse_floor``, ``description_tokens``, ``tags_per_card``, ``neighbors_per_candidate``,
    ``body_budget``, ``min_cards``, ``link_threshold``, ``reuse_ttl_cycles``, ``max_retrieval_cycles``,
    ``rarity_weight``, ``include_artifact_tool``, ``stash`` (second resolution layer for
    ``expand_artifact``, e.g. the relevance filter's store), ``matcher``.
    """

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
    ) -> None:
        if float(collapse_floor) > float(expand_threshold):
            raise ValueError("collapse_floor must be <= expand_threshold")
        if stash is not None and not callable(getattr(stash, "retrieve", None)):
            raise ValueError("stash must be None or have a retrieve(reference) method")
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
        self._store = InMemoryReferenceStore()
        # Engine-held, not Hermes persisted state (process-local, same rationale as the LangGraph store).
        self._state: GraphState | None = None
        self._thresholds = Thresholds(
            expand_threshold=float(expand_threshold),
            collapse_floor=float(collapse_floor),
            description_tokens=int(description_tokens),
            tags_per_card=int(tags_per_card),
            rarity_weight=float(rarity_weight),
            min_cards=int(min_cards),
            link_threshold=float(link_threshold),
            reuse_ttl_cycles=int(reuse_ttl_cycles),
            retrieval_tools=tuple(sorted(self._tool_names())),
        )

    @property
    def name(self) -> str:
        return "context-graph"

    def _tool_names(self) -> "frozenset[str]":
        return _TOOL_NAMES if self._include_artifact_tool else (_TOOL_NAMES - {_EXPAND_ARTIFACT})

    def _matcher_for(self) -> SimilarityMatcher:
        if self._matcher is not None:
            return self._matcher
        if self._resolved_matcher is None:
            from context_core.graph import EmbeddingSimilarityMatcher

            self._resolved_matcher = EmbeddingSimilarityMatcher()
        return self._resolved_matcher

    # ------------------------------------------------------------------ select_context (project)

    def select_context(
        self,
        request_messages: List[Dict[str, Any]],
        *,
        conversation_messages: List[Dict[str, Any]] = None,  # type: ignore[assignment]
        incoming_message: Dict[str, Any] = None,  # type: ignore[assignment]
        budget_tokens: int = 0,
    ) -> Optional[List[Dict[str, Any]]]:
        """Project this call's messages through the graph; request-only; fail-open."""
        try:
            neutral = hermes_to_neutral_list(list(request_messages))
            projected, new_state = project(
                neutral,
                state=self._state,
                matcher=self._matcher_for(),
                body_budget=self._body_budget,
                thresholds=self._thresholds,
            )
            self._state = _persistable(new_state)
            if projected is neutral:  # core's identity short circuit = "change nothing".
                return None
            return neutral_to_hermes_list(list(projected))
        except Exception:  # noqa: BLE001 — fail open.
            logger.debug("context-graph select_context failed; leaving request unchanged", exc_info=True)
            return None

    # ------------------------------------------------------------------ on_turn_complete (record)

    def on_turn_complete(self, messages: List[Dict[str, Any]], usage: Dict[str, Any] = None, **kwargs: Any) -> None:  # type: ignore[assignment]
        """Store each tool return as an addressable artifact and derive its artifact Cards."""
        try:
            neutral = hermes_to_neutral_list(list(messages))
            for position, message in enumerate(messages):
                if not isinstance(message, dict) or message.get("role") != "tool":
                    continue
                name = _producing_tool_name(messages, message.get("tool_call_id", ""))
                if name in self._tool_names():
                    continue  # our own retrieval answers are not artifacts to re-store.
                tool_call_id = message.get("tool_call_id", "")
                text, _ = _stringify(message.get("content"))
                if not tool_call_id or not text:
                    continue
                record_references(self._store, [f"{tool_call_id}_0"], [text])
                if self._state is None:
                    continue
                block = {"toolUseId": tool_call_id, "status": "success", "content": [{"text": text}]}
                cards = derive_and_register_artifacts(
                    self._state,
                    neutral,
                    block,
                    name,
                    self._state.turn,
                    description_tokens=self._thresholds.description_tokens,
                    tags_per_card=self._thresholds.tags_per_card,
                    rarity_weight=self._thresholds.rarity_weight,
                )
                record_references(self._store, [c.reference for c in cards if c.reference])
        except Exception:  # noqa: BLE001
            logger.debug("context-graph on_turn_complete failed", exc_info=True)

    # ------------------------------------------------------------------ tools

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        schemas = [_expand_card_schema(), _find_context_schema()]
        if self._include_artifact_tool:
            schemas.insert(1, _expand_artifact_schema())
        return schemas

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs: Any) -> str:
        try:
            if self._state is None:
                return json.dumps({"result": f"{name} | no earlier turns to retrieve yet"})
            if name == _EXPAND_CARD:
                return json.dumps({"result": self.expand_card(self._state, args.get("titles", []))})
            if name == _EXPAND_ARTIFACT and self._include_artifact_tool:
                return json.dumps({"result": _run_to_completion(self.expand_artifact(
                    self._state, args.get("reference", ""), args.get("line_range"), args.get("pattern")))})
            if name == _FIND_CONTEXT:
                return json.dumps({"result": self.find_context(self._state, args.get("need", ""), args.get("tag"))})
        except Exception as error:  # noqa: BLE001
            logger.debug("context-graph tool %s failed", name, exc_info=True)
            return json.dumps({"error": f"{name} failed: {error}"})
        return json.dumps({"error": f"Unknown context engine tool: {name}"})

    # ---- tool bodies (published so they are testable without an agent) ----

    def expand_card(self, state: GraphState, titles: Sequence[str] | str) -> str:
        refusal = self._exhausted(_EXPAND_CARD, state)
        if refusal is not None:
            return refusal
        state.retrieval_cycles += 1
        wanted = [titles] if isinstance(titles, str) else list(titles)
        if not wanted:
            return "expand_card | no title given | pass the titles you need, copied exactly as shown"
        found, missing = [], []
        for title in wanted:
            card = state.cards.get(title)
            (found if card is not None and card.kind == "subject" else missing).append(title)
        if found and not state.choice.full_pass:
            state.choice = TurnChoice(
                by_title=MappingProxyType({
                    **state.choice.by_title,
                    **{t: CardChoice(dialogue="full", evidence="full") for t in found},
                }),
                full_pass=False,
                selected=state.choice.selected,
            )
        for title in found:
            record_reuse(state, title, state.turn, reuse_ttl_cycles=self._reuse_ttl_cycles)
        if not found:
            return f"expand_card | no earlier turn is titled {_quoted(missing)} | copy a title exactly, or use find_context"
        confirmation = f"expand_card | {_quoted(found)} arrives in full for the rest of this turn"
        if missing:
            confirmation += f" | no turn is titled {_quoted(missing)}"
        return confirmation

    async def expand_artifact(
        self, state: GraphState, reference: str, line_range: dict[str, int] | None = None, pattern: str | None = None
    ) -> str:
        refusal = self._exhausted(_EXPAND_ARTIFACT, state)
        if refusal is not None:
            return refusal
        state.retrieval_cycles += 1
        resolved = await resolve_artifact(self._store, None, reference, stash=self._stash)
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
                return f"expand_artifact | line_range={line_range!r} is not a pair of integers"
            try:
                answer = read_artifact(resolved.text, line_range=span, pattern=pattern)
            except ValueError as error:
                return f"expand_artifact | reference '{reference}' | {error}"
        title = _artifact_title(state, reference)
        if title is not None:
            record_reuse(state, title, state.turn, reuse_ttl_cycles=self._reuse_ttl_cycles)
        return answer

    def find_context(self, state: GraphState, need: str, tag: str | None = None) -> str:
        refusal = self._exhausted(_FIND_CONTEXT, state)
        if refusal is not None:
            return refusal
        state.retrieval_cycles += 1
        if not need.strip():
            return _nothing_found(need, tag)
        titles = titles_in_turn_order(state)
        if tag is not None:
            wanted = normalize(tag)
            titles = tuple(t for t in titles if wanted and wanted in state.cards[t].tags)
        similarities = self._similarities(state, titles, need)
        if similarities is None:
            return _nothing_found(need, tag)
        passing = [t for t in titles if similarities[t] >= self._collapse_floor]
        passing.sort(key=lambda t: (-similarities[t], state.cards[t].turn, t))
        chosen = passing[:_MAX_CANDIDATES]
        if not chosen:
            return _nothing_found(need, tag)
        for title in chosen:
            record_reuse(state, title, state.turn, reuse_ttl_cycles=self._reuse_ttl_cycles)
        return _render_candidates(state, need, chosen, self._neighbors_per_candidate)

    def _exhausted(self, name: str, state: GraphState) -> str | None:
        if self._max_retrieval_cycles is None or state.retrieval_cycles < self._max_retrieval_cycles:
            return None
        return _EXHAUSTED.format(tool=name, spent=state.retrieval_cycles)

    def _similarities(self, state: GraphState, titles: tuple[str, ...], need: str) -> dict[str, float] | None:
        if not titles:
            return None
        descriptions = tuple(state.cards[t].description for t in titles)
        try:
            scores = self._matcher_for().score(need, descriptions)
            if len(scores) != len(descriptions):
                raise ValueError("similarity count mismatch")
            return {t: float(scores[i]) for i, t in enumerate(titles)}
        except Exception:  # noqa: BLE001
            logger.debug("find_context similarity unavailable", exc_info=True)
            return None

    def on_session_reset(self) -> None:
        super().on_session_reset()
        self._state = None
        self._store = InMemoryReferenceStore()


# ---- module helpers (pure; mirror the LangGraph binding) ----


def _persistable(state: GraphState) -> GraphState:
    """Flatten the frozen ``by_title`` proxy to a plain dict so the state can be carried/copied."""
    if type(state.choice.by_title) is not dict:
        state.choice = TurnChoice(
            by_title=dict(state.choice.by_title),
            full_pass=state.choice.full_pass,
            selected=state.choice.selected,
        )
    return state


def _producing_tool_name(messages: List[Dict[str, Any]], tool_call_id: str) -> str:
    """Find the tool name that produced ``tool_call_id`` by scanning assistant tool_calls."""
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                if isinstance(call, dict) and call.get("id") == tool_call_id:
                    return (call.get("function") or {}).get("name", "")
    return ""


def _stringify(content: Any) -> tuple[str, bool]:
    if content is None:
        return "", False
    if isinstance(content, str):
        return content, False
    if isinstance(content, (dict, list)):
        try:
            return json.dumps(content, indent=2), True
        except (TypeError, ValueError):
            return str(content), False
    return str(content), False


def _quoted(titles: Sequence[str]) -> str:
    return ", ".join(f"'{t}'" for t in titles)


def _whole_artifact(reference: str, text: str) -> str:
    notice = (
        f"expand_artifact | whole artifact '{reference}' | this re-injects about {estimate_tokens(text)} "
        "tokens for the rest of the turn | next time pass line_range or pattern for only the part you need"
    )
    return f"{notice}\n\n{text}"


def _span_of(line_range: dict[str, int] | None) -> tuple[int, int] | None:
    if line_range is None:
        return None
    try:
        return (int(line_range["start"]), int(line_range["end"]))
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def _artifact_title(state: GraphState, reference: str) -> str | None:
    for title in titles_in_turn_order(state):
        card = state.cards[title]
        if card.kind == "artifact" and card.reference == reference:
            return title
    return None


def _nothing_found(need: str, tag: str | None) -> str:
    narrowed = f", among the turns tagged '{tag}'" if tag is not None else ""
    return (
        f"find_context | nothing in this conversation matches '{need}'{narrowed} | the titles already in "
        "front of you are the whole conversation, so name a turn directly with expand_card"
    )


def _similar_neighbors(state: GraphState, title: str, exclude: "frozenset[str]", limit: int) -> list[tuple[str, float]]:
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
    lines = [f"find_context | {len(chosen)} earlier turn(s) match '{need}', best first:"]
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
            rendered = ", ".join(f"{n} ({w:.2f})" for n, w in neighbors)
            lines.append(f"  related turns: {rendered}")
    lines.append("call expand_card with one of these titles to bring that turn back in full")
    return "\n".join(lines)


def _expand_card_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": _EXPAND_CARD,
            "description": (
                "Bring an earlier turn of this conversation back in full for the rest of this turn. Pass "
                "the title(s) exactly as shown in the projected context."
            ),
            "parameters": {
                "type": "object",
                "properties": {"titles": {"type": "array", "items": {"type": "string"}, "description": "Turn titles."}},
                "required": ["titles"],
            },
        },
    }


def _expand_artifact_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": _EXPAND_ARTIFACT,
            "description": (
                "Read the full (or a part of a) tool-result artifact behind a reference shown in the "
                "projected context. Pass line_range or pattern to read only the part you need."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reference": {"type": "string", "description": "Artifact reference as shown."},
                    "line_range": {
                        "type": "object",
                        "properties": {"start": {"type": "integer"}, "end": {"type": "integer"}},
                        "description": "1-indexed inclusive span.",
                    },
                    "pattern": {"type": "string", "description": "Regex/keyword to keep only matching lines."},
                },
                "required": ["reference"],
            },
        },
    }


def _find_context_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": _FIND_CONTEXT,
            "description": (
                "Describe what you are looking for and get the earlier turns of this conversation that "
                "match best, so you can expand_card the right one."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "need": {"type": "string", "description": "What you are looking for, in your own words."},
                    "tag": {"type": "string", "description": "Optional: restrict to turns carrying this tag."},
                },
                "required": ["need"],
            },
        },
    }


def register(ctx: Any) -> None:
    """Hermes plugin entry point: register the single context-graph engine."""
    ctx.register_context_engine(ContextGraphEngine())
