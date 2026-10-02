"""The ``RelevanceFilterEngine``: Practice A on Hermes's synchronous ``ContextEngine`` surface.

Practice A rewrites an oversized tool result into a marker + disclaimer + verbatim relevance preview
(+ a ``[ref]`` token) before it reaches the model, keeps the full result in a store, and exposes
``rf_retrieve_all_context`` to read it back.

Mapping the LangGraph middleware hooks onto the Hermes ABC:

- LangGraph ``awrap_tool_call`` (per-result) has **no** Hermes analog — Hermes does not hand the engine
  each tool result at production time. Instead ``on_turn_complete(messages)`` sees the finished turn's
  tool messages; A detects oversized results there, stores them, and records the rewrite.
- LangGraph ``wrap_model_call`` → Hermes ``select_context(request_messages)``: build a request-only copy
  with oversized results replaced by their marker/disclaimer/preview and closed ``rf_retrieve_all_context``
  exchanges dropped. Request-only, fail-open — exactly the ABC contract.
- ``get_tool_schemas`` / ``handle_tool_call`` → the ``rf_retrieve_all_context`` tool.

Every decision about *content* (chunking, scoring, preview assembly, search) is delegated to
``context_core.relevance``; what lives here is the Hermes seam wiring, the marker/disclaimer text, the
sync driver for the async core, and the history surgery.

The token gate is the Strands default heuristic (``ceil(chars/4)`` text, ``ceil(json chars/2)`` JSON),
recomputed here: Hermes's sync engine exposes no model handle for native token counting (Requirement
2.6), the same compromise the LangGraph binding documents.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import math
from collections.abc import Awaitable
from typing import Any, Dict, List, Optional

from context_core.message import latest_user_text
from context_core.relevance import (
    BedrockReranker,
    InMemoryStore,
    PreviewStats,
    RelevancePreview,
    Reranker,
    RerankerError,
    Store,
)
from context_core.relevance.preview import _assemble_preview, _chunk_text
from context_core.relevance.search import _is_searchable_content, _search_content

from ._adapter import hermes_to_neutral_list
from ._base import BaseEngine

__all__ = ["RelevanceFilterEngine", "register"]

logger = logging.getLogger(__name__)

_DEFAULT_MAX_RESULT_TOKENS = 8_000
_DEFAULT_RELEVANCE_THRESHOLD = 0.5
_DEFAULT_CHUNK_TOKENS = 2_500
_DEFAULT_PREVIEW_TOKENS = 1_000
_DEFAULT_CONTEXT_LINES = 5
_CHARS_PER_TOKEN = 4
_MAX_QUERY_CHARS = 2_000
_RETRIEVAL_TOOL_NAME = "rf_retrieve_all_context"


def _run_to_completion(coroutine: Awaitable[Any]) -> Any:
    """Drive an async coroutine to completion from Hermes's synchronous call path."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)  # type: ignore[arg-type]
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coroutine).result()


def _approximate_tokens(text: str, is_json: bool) -> int:
    """Strands default token heuristic: ceil(chars/4) for text, ceil(json chars/2) for JSON."""
    if is_json:
        return math.ceil(len(text) / 2)
    return math.ceil(len(text) / _CHARS_PER_TOKEN)


def _disclaimer(token_count: int, stats: PreviewStats, reference: str | None, retrieval_tool: str) -> str:
    """Tell the model it sees an excerpt, how much it covers, and how to recover the rest."""
    spans = ", ".join(f"{s}-{e}" if s != e else f"{s}" for s, e in stats.shown_spans)
    head = (
        f"[Filtered: this is an EXCERPT, not the whole result | original: {stats.total_lines:,} lines, "
        f"{stats.total_chunks} chunks | shown: {len(stats.shown)} chunk(s), lines {spans or 'none'}]\n"
        "An answer that needs every row -- a maximum, minimum, total, count, average, ranking or any "
        "comparison across the whole result -- cannot be computed from this excerpt."
    )
    if not reference:
        return f"{head} Say that the result was filtered instead of computing it from the excerpt."
    return (
        f'{head} For such an answer, call `{retrieval_tool}` with reference "{reference}" and either a '
        f"`pattern` (regex) matching only the rows you need, or `max_chunks`/`max_tokens` large enough "
        f"for the whole result ({stats.total_chunks} chunks, ~{token_count:,} tokens). What you retrieve "
        "is removed from the conversation once you have answered, so state the figures you relied on."
    )


class RelevanceFilterEngine(BaseEngine):
    """Relevance-filter oversized tool results on Hermes's ``ContextEngine`` surface.

    Construction is inert: no reranker and no AWS client is built here (the default
    :class:`~context_core.relevance.BedrockReranker` is lazy), so an engine that never fires costs
    nothing and needs no credentials merely to be selected.

    Args:
        max_result_tokens: Filter only results whose estimated token count exceeds this. Default 8000.
        config: Preview tuning (``reranker``, ``relevance_threshold``, ``chunk_tokens``,
            ``preview_tokens``, ``summarize_overflow``); every key optional.
        include_retrieval_tool: Store raw content and register ``rf_retrieve_all_context``. Default True.
        store: Backend for raw sub-blocks (only the retrieval tool reads it). When None and the tool is
            on, an ``InMemoryStore`` is built.
    """

    def __init__(
        self,
        *,
        max_result_tokens: int = _DEFAULT_MAX_RESULT_TOKENS,
        config: Dict[str, Any] | None = None,
        include_retrieval_tool: bool = True,
        store: Store | None = None,
    ) -> None:
        if max_result_tokens <= 0:
            raise ValueError("max_result_tokens must be positive")
        super().__init__()
        self._max_result_tokens = max_result_tokens
        self._config: Dict[str, Any] = config or {}
        self._include_retrieval_tool = include_retrieval_tool
        self._store: Store | None = store if store is not None or not include_retrieval_tool else InMemoryStore()
        self._preview: RelevancePreview | None = None
        self.retrieval_tool_name = _RETRIEVAL_TOOL_NAME
        # tool_call_id -> (marker_text, reference) for results already filtered this session.
        self._rewrites: Dict[str, tuple[str, Optional[str]]] = {}
        self._rankings: Dict[str, tuple[int, ...]] = {}

    @property
    def name(self) -> str:
        return "relevance-filter"

    @property
    def stash(self) -> "Optional[_RelevanceStash]":
        """Read-only view of this filter's store, so another practice can resolve a ``[ref]`` it minted.

        Used by ``hermes-all-three`` to let the context graph read filter references through
        ``cg_expand_artifact``. ``None`` when nothing is stored.
        """
        return None if self._store is None else _RelevanceStash(self._store)

    # ------------------------------------------------------------------ preview / query

    def _resolve_preview(self) -> RelevancePreview:
        if self._preview is None:
            reranker: Reranker = self._config.get("reranker") or BedrockReranker()
            self._preview = RelevancePreview(
                reranker,
                relevance_threshold=self._config.get("relevance_threshold", _DEFAULT_RELEVANCE_THRESHOLD),
                chunk_tokens=self._config.get("chunk_tokens", _DEFAULT_CHUNK_TOKENS),
                preview_tokens=self._config.get("preview_tokens", _DEFAULT_PREVIEW_TOKENS),
                summarize_overflow=self._config.get("summarize_overflow", False),
            )
        return self._preview

    def _build_query(self, messages: List[Dict[str, Any]]) -> str:
        user_text = latest_user_text(hermes_to_neutral_list(messages))
        return (user_text[-_MAX_QUERY_CHARS:] or "{}")

    # ------------------------------------------------------------------ on_turn_complete (detect+store)

    def on_turn_complete(self, messages: List[Dict[str, Any]], usage: Dict[str, Any] = None, **kwargs: Any) -> None:  # type: ignore[assignment]
        """Detect oversized tool results in the finished turn, store them, and record the rewrite."""
        try:
            query = self._build_query(list(messages))
            for message in messages:
                if not isinstance(message, dict) or message.get("role") != "tool":
                    continue
                tool_call_id = message.get("tool_call_id", "")
                if not tool_call_id or tool_call_id in self._rewrites:
                    continue
                text, is_json = _stringify_content(message.get("content"))
                if not text:
                    continue
                if _approximate_tokens(text, is_json) <= self._max_result_tokens:
                    continue
                self._rewrites[tool_call_id] = _run_to_completion(
                    self._build_rewrite(tool_call_id, text, is_json, query)
                )
        except Exception:  # noqa: BLE001 — indexing is best-effort; never break the turn.
            logger.debug("relevance-filter on_turn_complete failed", exc_info=True)

    async def _build_rewrite(self, tool_call_id: str, text: str, is_json: bool, query: str) -> tuple[str, Optional[str]]:
        """Store the raw result and build the marker/disclaimer/preview. Returns (marker, reference)."""
        token_count = _approximate_tokens(text, is_json)
        reference: Optional[str] = None
        if self._include_retrieval_tool and self._store is not None:
            try:
                content_type = "application/json" if is_json else "text/plain"
                reference = await self._store.store(f"{tool_call_id}_0", text.encode("utf-8"), content_type)
            except Exception:  # noqa: BLE001
                logger.warning("relevance-filter failed to store %s; no reference emitted", tool_call_id, exc_info=True)
                reference = None
        try:
            preview, stats = await self._resolve_preview().build_with_stats(text, query)
        except RerankerError:
            logger.debug("relevance scoring failed for %s; keeping original", tool_call_id)
            raise
        if reference is not None:
            self._rankings[reference] = stats.ranking
        disclaimer = _disclaimer(token_count, stats, reference, self.retrieval_tool_name)
        marker = f"[Relevance: tool result, ~{token_count:,} tokens]\n{disclaimer}\n\n{preview}"
        if reference is not None:
            marker = f"{marker}\n\n[ref: {reference}]"
        return marker, reference

    # ------------------------------------------------------------------ select_context (apply)

    def select_context(
        self,
        request_messages: List[Dict[str, Any]],
        *,
        conversation_messages: List[Dict[str, Any]] = None,  # type: ignore[assignment]
        incoming_message: Dict[str, Any] = None,  # type: ignore[assignment]
        budget_tokens: int = 0,
    ) -> Optional[List[Dict[str, Any]]]:
        """Return a request-only copy with oversized results replaced and closed retrievals dropped."""
        try:
            if not self._rewrites and not self._include_retrieval_tool:
                return None
            selected: List[Dict[str, Any]] = []
            changed = False
            for message in request_messages:
                if isinstance(message, dict) and message.get("role") == "tool":
                    rewrite = self._rewrites.get(message.get("tool_call_id", ""))
                    if rewrite is not None:
                        copy_msg = dict(message)
                        copy_msg["content"] = rewrite[0]
                        selected.append(copy_msg)
                        changed = True
                        continue
                selected.append(message)
            dropped = _drop_tool_exchanges(selected, self.retrieval_tool_name)
            if dropped is not None:
                selected = dropped
                changed = True
            return selected if changed else None
        except Exception:  # noqa: BLE001 — fail open: unmodified request.
            logger.debug("relevance-filter select_context failed; leaving request unchanged", exc_info=True)
            return None

    # ------------------------------------------------------------------ tools

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        if not self._include_retrieval_tool:
            return []
        return [_retrieve_all_context_schema(self.retrieval_tool_name)]

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs: Any) -> str:
        if name != self.retrieval_tool_name:
            return json.dumps({"error": f"Unknown context engine tool: {name}"})
        try:
            return _run_to_completion(self._retrieve(**_retrieval_args(args)))
        except ValueError as error:
            return json.dumps({"error": str(error)})
        except Exception as error:  # noqa: BLE001
            logger.debug("rf_retrieve_all_context failed", exc_info=True)
            return json.dumps({"error": f"retrieval failed: {error}"})

    async def _retrieve(
        self,
        reference: str,
        *,
        pattern: Optional[str] = None,
        line_range: Optional[Dict[str, int]] = None,
        context_lines: Optional[int] = None,
        max_chunks: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        for key, value in (("max_chunks", max_chunks), ("max_tokens", max_tokens)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"{key} must be an integer >= 1, got {value!r}")
        if self._store is None:
            raise ValueError(f"reference not found: {reference}")
        try:
            content_bytes, content_type = await self._store.retrieve(reference)
        except KeyError as error:
            raise ValueError(f"reference not found: {reference}") from error

        if all(option is None for option in (pattern, line_range, context_lines, max_chunks, max_tokens)):
            return json.dumps({"content": content_bytes.decode("utf-8", errors="replace")})
        if not _is_searchable_content(content_type):
            raise ValueError(
                f"cannot search binary content ({content_type}); omit all read options for the full content"
            )
        text = content_bytes.decode("utf-8")
        ctx_lines = context_lines if context_lines is not None else _DEFAULT_CONTEXT_LINES
        max_chars = (max_tokens or self._max_result_tokens) * _CHARS_PER_TOKEN

        if line_range is None and pattern is None:
            if max_chunks is not None:
                return json.dumps({"content": self._read_chunks(reference, text, max_chunks, max_chars)})
            if context_lines is None:
                whole = _search_content(text, line_range=(1, max(1, text.count("\n") + 1)), max_chars=max_chars)
                return json.dumps({"content": whole})

        span: Optional[tuple[int, int]] = None
        if line_range is not None:
            span = (int(line_range["start"]), int(line_range["end"]))
            pattern = None
        elif pattern is None:
            span = (1, max(1, ctx_lines))
        rendered = _search_content(text, pattern=pattern, line_range=span, context_lines=ctx_lines, max_chars=max_chars)
        return json.dumps({"content": rendered})

    def _read_chunks(self, reference: str, text: str, max_chunks: int, max_chars: int) -> str:
        chunks = _chunk_text(text, self._config.get("chunk_tokens", _DEFAULT_CHUNK_TOKENS))
        if not chunks:
            return "Content is empty (0 lines)."
        ranking = self._rankings.get(reference)
        if ranking is None or len(ranking) != len(chunks):
            ranking = tuple(range(len(chunks)))
            order = "document order"
        else:
            order = "most relevant first"
        chosen = sorted((chunks[i] for i in ranking[:max_chunks]), key=lambda c: c.index)
        body = _assemble_preview(chunks, chosen, max_chars)
        header = (
            f"[{len(chosen)} of {len(chunks)} chunks ({order}), lines 1-{chunks[-1].end_line} in total, "
            f"limit ~{max_chars // _CHARS_PER_TOKEN:,} tokens]"
        )
        return f"{header}\n{body}"

    def on_session_reset(self) -> None:
        super().on_session_reset()
        self._rewrites.clear()
        self._rankings.clear()


class _RelevanceStash:
    """``retrieve(reference) -> text | None`` view over the filter's store, for cross-practice resolution."""

    def __init__(self, store: Store) -> None:
        self._store = store

    async def retrieve(self, reference: str) -> Optional[str]:
        content_bytes, content_type = await self._store.retrieve(reference)
        if content_type.startswith("text/") or content_type == "application/json":
            return content_bytes.decode("utf-8")
        return None


def _stringify_content(content: Any) -> tuple[str, bool]:
    """Render a Hermes tool-message content to (text, is_json) for the gate and the preview."""
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


def _tool_call_names(message: Dict[str, Any]) -> Dict[str, str]:
    """Map ``tool_call_id -> tool name`` for an assistant message's tool calls."""
    out: Dict[str, str] = {}
    for call in message.get("tool_calls") or []:
        if isinstance(call, dict) and call.get("id"):
            out[call["id"]] = (call.get("function") or {}).get("name", "")
    return out


def _drop_tool_exchanges(messages: List[Dict[str, Any]], tool_name: str) -> Optional[List[Dict[str, Any]]]:
    """Drop closed exchanges of ``tool_name`` (assistant tool_call + its tool reply). None if none."""
    answered = {m.get("tool_call_id") for m in messages if isinstance(m, dict) and m.get("role") == "tool"}
    drop_ids: set[str] = set()
    drop_positions: set[int] = set()
    for position, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        names = _tool_call_names(message)
        ids = {cid for cid, nm in names.items() if nm == tool_name}
        if not ids or not ids <= answered:
            continue
        drop_ids |= ids
        if ids == set(names):
            drop_positions.add(position)
    if not drop_ids:
        return None
    kept: List[Dict[str, Any]] = []
    for position, message in enumerate(messages):
        if position in drop_positions:
            continue
        if isinstance(message, dict) and message.get("role") == "tool" and message.get("tool_call_id") in drop_ids:
            continue
        if isinstance(message, dict) and message.get("role") == "assistant" and _tool_call_names(message):
            remaining = [c for c in (message.get("tool_calls") or []) if c.get("id") not in drop_ids]
            if len(remaining) != len(message.get("tool_calls") or []):
                copy_msg = dict(message)
                copy_msg["tool_calls"] = remaining
                kept.append(copy_msg)
                continue
        kept.append(message)
    return kept


def _retrieval_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """Pick the known ``rf_retrieve_all_context`` arguments out of the raw tool-call args."""
    reference = args.get("reference")
    if not isinstance(reference, str) or not reference:
        raise ValueError("reference is required")
    return {
        "reference": reference,
        "pattern": args.get("pattern"),
        "line_range": args.get("line_range"),
        "context_lines": args.get("context_lines"),
        "max_chunks": args.get("max_chunks"),
        "max_tokens": args.get("max_tokens"),
    }


def _retrieve_all_context_schema(name: str) -> Dict[str, Any]:
    """OpenAI function-tool schema for ``rf_retrieve_all_context``."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": (
                "Load the WHOLE of a tool result that the relevance filter cut down to an excerpt. Use it "
                "ONLY when the answer needs every row of that result (a maximum, minimum, total, count, "
                "average, ranking or a comparison across all of it). Pass the reference from the filtered "
                "block, with a `pattern` matching only the rows you need, or `max_chunks`/`max_tokens` "
                "large enough for all of it. What you retrieve is removed from the conversation once you "
                "answer, so state the figures you relied on."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reference": {"type": "string", "description": "Reference from the filtered block, e.g. 'mem_1_tool-123_0'."},
                    "pattern": {"type": "string", "description": "Regex/keyword; returns only matching lines with context."},
                    "line_range": {
                        "type": "object",
                        "properties": {"start": {"type": "integer"}, "end": {"type": "integer"}},
                        "description": "1-indexed inclusive span; takes precedence over pattern.",
                    },
                    "context_lines": {"type": "integer", "description": "Lines before/after each match (default 5)."},
                    "max_chunks": {"type": "integer", "description": "Number of most-relevant chunks, document order."},
                    "max_tokens": {"type": "integer", "description": "Approximate response size budget, in tokens."},
                },
                "required": ["reference"],
            },
        },
    }


def register(ctx: Any) -> None:
    """Hermes plugin entry point: register the single relevance-filter engine."""
    ctx.register_context_engine(RelevanceFilterEngine())
