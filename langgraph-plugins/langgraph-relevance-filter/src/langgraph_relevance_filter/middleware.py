"""The ``RelevanceFilterMiddleware``: configuration, the two LangGraph hooks, and the retrieval tool.

Attaches to the LangChain v1 ``create_agent`` middleware surface only (verified against ``langchain``
1.4.2). Two hooks carry Practice A:

- ``awrap_tool_call`` is the ``AfterToolCallEvent`` analog. It runs the tool through ``handler``, guards
  the returned :class:`~langchain_core.messages.ToolMessage` through six checks, then stores the raw
  sub-blocks and rewrites the message content into the marker, a disclaimer carrying the processing
  metadata, a verbatim budget-bounded preview, and a reference token.
- ``after_agent`` is the ``AfterInvocationEvent`` analog. It removes the *closed*
  ``retrieve_all_context`` exchanges from ``state["messages"]`` once the run ends.

The ``retrieve_all_context`` tool is shipped on the middleware's ``tools`` attribute and reads those
references back by span, pattern, chunk count or token budget.

Every decision this module makes about *content* is delegated to :mod:`context_core.relevance`; what
lives here is attachment, the LangChain↔neutral adaptation (through :mod:`._adapter`), the marker and
disclaimer text, and the history surgery. Two things are deliberately native rather than core:

- **The token count is the Strands default heuristic, recomputed here.** The Strands plugin wraps the
  result as a message and calls ``model.count_tokens``; unless a provider opts into native counting,
  that is ``ceil(chars / 4)`` per text item and ``ceil(json chars / 2)`` per JSON item.
  ``wrap_tool_call`` exposes no model (``request`` carries the tool, the state and the runtime), so the
  same heuristic is applied to the neutral blocks and the gate flips at the same size. A Strands model
  configured with ``use_native_token_count`` would count exactly; this binding cannot.
- **The delegation guard reads ``tool.return_direct``.** LangGraph has no ``_AgentAsTool``; the flag
  that carries the same meaning — this result becomes the final answer, so no later model call could
  retrieve what the filter cut — is ``BaseTool.return_direct``.
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import inspect
import json
import logging
import math
from collections.abc import Awaitable, Sequence
from typing import Any, Protocol, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, RemoveMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.graph.message import REMOVE_ALL_MESSAGES

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

# Private core helpers: the chunker and the gap-marker assembler. ``retrieve_all_context`` renders the
# N most relevant chunks with the ranking the filter already computed, which is the same chunking and
# the same assembly the preview used -- re-deriving either here would be a second implementation of
# core logic. The core does not re-export them, so they are imported from their module.
from context_core.relevance.preview import _assemble_preview, _chunk_text
from context_core.relevance.search import _is_searchable_content, _search_content

from ._adapter import to_neutral_list, tool_message_to_result_block
from ._compat import AgentMiddleware

__all__ = [
    "LineRange",
    "RelevanceConfig",
    "RelevanceFilterMiddleware",
    "ShouldFilter",
]

logger = logging.getLogger(__name__)

_DEFAULT_MAX_RESULT_TOKENS = 8_000
"""Default token threshold above which a textual tool result is relevance-filtered."""

_DEFAULT_RELEVANCE_THRESHOLD = 0.5
"""Default minimum score for a chunk to be eligible for the preview."""

_DEFAULT_CHUNK_TOKENS = 2_500
"""Default scoring granularity — the approximate token budget of one scored chunk."""

_DEFAULT_PREVIEW_TOKENS = 1_000
"""Default approximate token budget of the whole preview."""

_DEFAULT_CONTEXT_LINES = 5
"""Lines kept before and after each pattern match when the caller names none, as in ``grep -C 5``."""

_CHARS_PER_TOKEN = 4
"""Approximate characters per token: the conversion used to bound a retrieval response, and — unlike
in the Strands plugin, which has a tokenizer at the hook — also to size the result for the gate."""

_MAX_QUERY_CHARS = 2_000
"""Cap on the scoring query. Rerankers charge per query length, and a query longer than this adds
context without sharpening the ranking."""

_RETRIEVAL_TOOL_NAME = "retrieve_all_context"
"""Registered name of the retrieval tool. Every guard and the cleanup match on this one constant."""

_NON_SCORABLE_BLOCK_TYPES = frozenset({"image", "image_url", "audio", "video", "file", "document"})
"""LangChain content-block ``type`` values that carry no scorable text. Such a block is left verbatim
after the filtered block, the way the Strands plugin keeps non-textual sub-blocks."""


def _run_to_completion(coroutine: Awaitable[Any]) -> Any:
    """Run ``coroutine`` to completion from sync code, whether or not this thread has a running loop.

    Args:
        coroutine: The awaitable to drive.

    Returns:
        Its result. Its exception, if any, propagates.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)  # type: ignore[arg-type]
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coroutine).result()


def _approximate_tokens(blocks: Sequence[Any]) -> int:
    """Estimate the token count of a tool result the way the Strands plugin's gate does.

    The Strands plugin wraps the result as a message and calls ``model.count_tokens``, whose default
    (the base ``Model`` heuristic, used unless a provider opts into native counting) is
    ``ceil(chars / 4)`` per text item and ``ceil(len(json.dumps(obj)) / 2)`` per JSON item, with binary
    items not counted. ``wrap_tool_call`` exposes no model, so that same heuristic is applied here to
    the neutral blocks, and the gate flips at the same size as the Strands one.

    Args:
        blocks: Neutral ``text`` / ``json`` blocks of the result, plus raw passthrough parts (not counted).

    Returns:
        The estimated token count, never negative.
    """
    total = 0
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if isinstance(block.get("text"), str):
            total += math.ceil(len(block["text"]) / _CHARS_PER_TOKEN)
        elif "json" in block:
            try:
                total += math.ceil(len(json.dumps(block["json"])) / 2)
            except (TypeError, ValueError):
                pass
    return total


def _disclaimer(token_count: int, stats: PreviewStats, references: Sequence[str], retrieval_tool: str) -> str:
    """Tell the model that it sees an excerpt, how much of the result it covers, and what to do about it.

    A relevance excerpt answers "where does the result mention X" and cannot answer anything that needs
    every row: a maximum, a total, a count. Without being told, a model computes the aggregate over the
    excerpt and is confidently wrong, or distrusts the sample and refuses. So the processing metadata
    travels with the excerpt, and with the retrieval tool registered the text names the call and the
    budgets that reach the rest -- a pattern for the rows to aggregate, or chunks and tokens enough for
    all of it.

    Args:
        token_count: Estimated tokens of the original result.
        stats: What the selection did.
        references: References issued for the stored content; empty when the retrieval tool is off.
        retrieval_tool: Registered name of the retrieval tool.

    Returns:
        The disclaimer, a few lines of plain text.
    """
    spans = ", ".join(f"{start}-{end}" if start != end else f"{start}" for start, end in stats.shown_spans)
    head = (
        f"[Filtered: this is an EXCERPT, not the whole result | original: {stats.total_lines:,} lines, "
        f"{stats.total_chunks} chunks | shown: {len(stats.shown)} chunk(s), lines {spans or 'none'}]\n"
        "An answer that needs every row -- a maximum, minimum, total, count, average, ranking or any "
        "comparison across the whole result -- cannot be computed from this excerpt."
    )
    if not references:
        return f"{head} Say that the result was filtered instead of computing it from the excerpt."

    named = f'reference "{references[0]}"' if len(references) == 1 else f"references {', '.join(references)}"
    return (
        f"{head} For such an answer, call `{retrieval_tool}` with {named} and either a `pattern` (regex) "
        f"that matches only the rows you need, or `max_chunks`/`max_tokens` large enough for the whole "
        f"result ({stats.total_chunks} chunks, ~{token_count:,} tokens). What you retrieve is removed from "
        "the conversation once you have answered, so state the figures you relied on in the answer."
    )


def _tool_call_ids(message: BaseMessage, tool_name: str) -> set[str]:
    """Return the ids of ``message``'s tool calls that name ``tool_name``."""
    if not isinstance(message, AIMessage):
        return set()
    return {call["id"] for call in message.tool_calls or () if call.get("name") == tool_name and call.get("id")}


def _drop_tool_exchanges(messages: Sequence[BaseMessage], tool_name: str) -> list[BaseMessage] | None:
    """Return ``messages`` without the closed exchanges of ``tool_name``, or None when none are removed.

    The LangChain analog of the Strands ``_drop_tool_exchanges``: an exchange is a tool call on an
    ``AIMessage`` and the ``ToolMessage`` answering it. An ``AIMessage`` whose tool calls are *all*
    ``tool_name`` goes whole, together with the answering ``ToolMessage``\\ s -- its interim text goes with
    it, as in Strands. An ``AIMessage`` that mixes them with other tool calls keeps its text and loses
    only the ``tool_name`` calls. An unclosed call -- a run that ended on an error -- is left as it is,
    so the history never holds a tool call with no result.

    Args:
        messages: The conversation, oldest first. Read only.
        tool_name: The tool whose exchanges are removed.

    Returns:
        A new list without those exchanges, or None when nothing matched.
    """
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}

    dropped_ids: set[str] = set()
    dropped_positions: set[int] = set()
    for position, message in enumerate(messages):
        ids = _tool_call_ids(message, tool_name)
        if not ids or not ids <= answered:
            continue
        dropped_ids |= ids
        if len(ids) == len(message.tool_calls or ()):  # type: ignore[union-attr]  # AIMessage by construction
            dropped_positions.add(position)

    if not dropped_ids:
        return None

    kept: list[BaseMessage] = []
    for position, message in enumerate(messages):
        if position in dropped_positions:
            continue
        if isinstance(message, ToolMessage):
            if message.tool_call_id in dropped_ids:
                continue
            kept.append(message)
            continue
        if isinstance(message, AIMessage) and _tool_call_ids(message, tool_name):
            remaining = [call for call in message.tool_calls if call.get("id") not in dropped_ids]
            kept.append(message.model_copy(update={"tool_calls": remaining}))
            continue
        kept.append(message)
    return kept


class RelevanceConfig(TypedDict, total=False):
    """Tuning for the relevance preview.

    All keys are optional; the middleware reads each with ``dict.get`` and the default documented here,
    so a partial config is as valid as a full one and an omitted key never has to be spelled out.

    Attributes:
        reranker: Scorer used to rank chunks against the query. Defaults to a lazily built
            ``BedrockReranker``, so an install that supplies its own scorer never constructs an AWS
            client.
        relevance_threshold: Minimum score, in ``[0.0, 1.0]``, for a chunk to enter the preview.
            Defaults to ``0.5``.
        chunk_tokens: Scoring granularity — the maximum size of one scored chunk. Defaults to ``2500``.
        preview_tokens: Budget for what stays visible in context. Defaults to ``1000``.
        summarize_overflow: Reserved. Accepted and stored but not acted upon; while off, the preview is
            composed exclusively of verbatim source substrings and gap markers. Defaults to ``False``.
    """

    reranker: Reranker
    relevance_threshold: float
    chunk_tokens: int
    preview_tokens: int
    summarize_overflow: bool


class ShouldFilter(Protocol):
    """Callback protocol for deciding whether an oversized tool result should be filtered."""

    def __call__(self, tool_name: str, token_count: int, **kwargs: Any) -> bool | Awaitable[bool]:
        """Return True to filter, False to keep the result in context. May be sync or async.

        Args:
            tool_name: Name of the tool that produced the result.
            token_count: Estimated token count of the result.
            **kwargs: Reserved for future parameters. Implementations should accept ``**kwargs`` for
                forward compatibility.
        """
        ...


class LineRange(TypedDict):
    """A span of lines to retrieve (1-indexed, inclusive)."""

    start: int
    end: int


class _RelevanceStash:
    """Read-only view of the filter's store in the shape a reference resolver reads: ``retrieve -> text``.

    The store answers ``(bytes, content_type)``; a resolver wants the text. Text and JSON decode to a string,
    anything else (images, files) to ``None``, which the resolver reports as non-textual content.
    """

    def __init__(self, store: Store) -> None:
        self._store = store

    async def retrieve(self, reference: str) -> str | None:
        content_bytes, content_type = await self._store.retrieve(reference)
        if content_type.startswith("text/") or content_type == "application/json":
            return content_bytes.decode("utf-8")
        return None


class RelevanceFilterMiddleware(AgentMiddleware):
    """Relevance-filter oversized tool results on the LangChain v1 middleware surface.

    When a tool result exceeds ``max_result_tokens``, its raw sub-blocks are written to ``store`` and
    the in-context ``ToolMessage`` is replaced with a marker, a verbatim relevance preview scored
    against the query in progress, and a reference token the model can pass to
    ``retrieve_all_context``. Selection is verbatim — chosen chunks reach the model
    character-for-character — so numeric, monetary and tabular content stays exact.

    Construction is inert: no ``Reranker``, no preview builder and no AWS client is created here. The
    reranker is built on the first scoring call, so a middleware that never fires costs nothing and
    needs no AWS credentials or region merely to be declared.

    Asynchrony: the core's preview and every ``Store`` are async. The tool-call hook ships both twins --
    ``awrap_tool_call``, and ``wrap_tool_call``, which drives the same pipeline to completion -- so the agent
    runs under ``invoke`` or ``ainvoke``. The end-of-run cleanup is on ``after_agent`` and ``aafter_agent``.

    Args:
        store: Optional backend for the raw sub-blocks, used only by the retrieval tool. Filtering
            never depends on it. When None and ``include_retrieval_tool`` is True, an
            ``InMemoryStore`` is created here; otherwise no store exists.
        max_result_tokens: Filter only results whose estimated token count exceeds this threshold.
            Defaults to ``8_000``. The estimate is the Strands default heuristic — see the module docstring.
        config: Preview tuning (reranker, threshold, chunk and preview budgets). Every key is
            optional; see :class:`RelevanceConfig` for the defaults.
        include_retrieval_tool: Whether to store the raw content and register the
            ``retrieve_all_context`` tool that reads it back. Defaults to True. The tool is for the one
            question an excerpt cannot answer -- one that needs every row -- and its exchanges are
            removed from the history when the run ends, so a retrieval is paid for in the turn that
            asked for it and not on every later call. Off, no raw sub-block is stored, no reference
            token is emitted, and ``after_agent`` does nothing -- filtering itself is unchanged.
        should_filter: Callback deciding whether a specific oversized result is filtered. Called only
            once the result is over threshold. Defaults to None (every oversized result is filtered).

    Example:
        ```python
        from langchain.agents import create_agent
        from langgraph_relevance_filter import RelevanceFilterMiddleware

        agent = create_agent(
            model="bedrock:anthropic.claude-sonnet-4-20250514-v1:0",
            tools=[query_ledger],
            middleware=[RelevanceFilterMiddleware(max_result_tokens=8_000)],
        )
        result = await agent.ainvoke({"messages": [{"role": "user", "content": "..."}]})
        ```
    """

    def __init__(
        self,
        *,
        store: Store | None = None,
        max_result_tokens: int = _DEFAULT_MAX_RESULT_TOKENS,
        config: RelevanceConfig | None = None,
        include_retrieval_tool: bool = True,
        should_filter: ShouldFilter | None = None,
    ) -> None:
        """Initialize the middleware without building any scoring or network dependency.

        Args:
            store: Optional backend for the raw sub-blocks. When None, an ``InMemoryStore`` is created
                only if the retrieval tool is enabled.
            max_result_tokens: Filter only results above this estimated token count.
            config: Preview tuning; read key by key with the documented defaults at use time, so a
                partial config is as valid as a full one.
            include_retrieval_tool: Store the raw content and register ``retrieve_all_context``.
                Defaults to True. False also suppresses the store write, the reference token and the
                end-of-run cleanup: with no tool to resolve it, a reference would be a promise nothing
                can keep.
            should_filter: Callback ``(tool_name, token_count, **kwargs) -> bool``, sync or async.

        Raises:
            ValueError: If ``max_result_tokens`` is not positive.
        """
        if max_result_tokens <= 0:
            raise ValueError("max_result_tokens must be positive")

        super().__init__()

        self._max_result_tokens = max_result_tokens
        self._config: RelevanceConfig = config if config is not None else {}
        self._include_retrieval_tool = include_retrieval_tool
        self._should_filter = should_filter
        # The store exists only to serve the retrieval tool. Filtering never needs it, so a default
        # store is built only when something can read it back.
        self._store: Store | None = store if store is not None or not include_retrieval_tool else InMemoryStore()
        # Built on first use, never here: constructing a reranker would reach for AWS credentials.
        self._preview: RelevancePreview | None = None
        # Chunk ranking of each stored reference, by descending relevance, so ``retrieve_all_context``
        # can hand back more chunks in relevance order without scoring the text again.
        self._rankings: dict[str, tuple[int, ...]] = {}
        self.retrieval_tool_name = _RETRIEVAL_TOOL_NAME
        # LangGraph reads ``tools`` at compile time; an empty sequence registers nothing.
        self.tools: Sequence[BaseTool] = [self._build_retrieval_tool()] if include_retrieval_tool else []

    @property
    def stash(self) -> _RelevanceStash | None:
        """This filter's store, readable by another plugin's reference resolution.

        Pass it as ``ContextGraphMiddleware(stash=...)`` so a ``[ref: mem_N_...]`` the filter minted resolves
        through ``expand_artifact`` too. It plays the role the Strands ``ContextManager`` Stash plays for the
        Strands graph plugin, which LangGraph has no equivalent of. ``None`` when nothing is stored.
        """
        return None if self._store is None else _RelevanceStash(self._store)

    # ------------------------------------------------------------------ preview / query

    def _resolve_preview(self) -> RelevancePreview:
        """Return the preview builder, constructing it on the first call.

        Deferring construction is what keeps ``__init__`` inert: the default
        :class:`~context_core.relevance.BedrockReranker` opens an AWS client, so a middleware that
        never fires never needs credentials or a region. The builder is created once and reused, which
        also makes its ``search_units`` counter a session total.

        The config is read key by key with the documented defaults and is otherwise unvalidated — a
        nonsensical value surfaces where it is used rather than at construction.

        Returns:
            The memoized preview builder.
        """
        if self._preview is None:
            reranker = self._config.get("reranker") or BedrockReranker()
            self._preview = RelevancePreview(
                reranker,
                relevance_threshold=self._config.get("relevance_threshold", _DEFAULT_RELEVANCE_THRESHOLD),
                chunk_tokens=self._config.get("chunk_tokens", _DEFAULT_CHUNK_TOKENS),
                preview_tokens=self._config.get("preview_tokens", _DEFAULT_PREVIEW_TOKENS),
                summarize_overflow=self._config.get("summarize_overflow", False),
            )
        return self._preview

    def _build_query(self, state: Any, tool_call: dict[str, Any]) -> str:
        """Compose the scoring query from the latest human message and the tool-call arguments.

        Both signals matter: the question states the goal, the arguments state the sub-goal of this
        particular call, which is what keeps selection on target in a multi-tool loop. The tool *name*
        is deliberately left out — it names the capability, not what is being looked for, and it biases
        the ranking toward chunks that echo the name.

        The cap keeps the **tail**, so a long question cannot push the arguments out of the query.
        Arguments that alone exceed the cap are kept as a head instead, since their leading keys are
        the identifying part. When the arguments cannot be serialized the question tail stands alone,
        falling back to the literal ``"{}"`` so the reranker never receives an empty query.

        Args:
            state: The agent state, read for ``messages`` only.
            tool_call: This call's ``{"name", "args", "id"}`` dict.

        Returns:
            The scoring query, between 1 and ``_MAX_QUERY_CHARS`` characters.
        """
        messages = _state_messages(state)
        user_text = latest_user_text(to_neutral_list(list(messages)))

        try:
            serialized = json.dumps(tool_call.get("args", {}))
        except (TypeError, ValueError):
            return user_text[-_MAX_QUERY_CHARS:] or "{}"

        if len(serialized) >= _MAX_QUERY_CHARS:
            return serialized[:_MAX_QUERY_CHARS]

        query = f"{user_text}\n{serialized}" if user_text else serialized
        return query[-_MAX_QUERY_CHARS:]

    # ------------------------------------------------------------------ retrieval tool

    def _build_retrieval_tool(self) -> BaseTool:
        """Build the ``retrieve_all_context`` tool bound to this middleware's store.

        The tool is built per instance rather than declared as a class attribute because it closes over
        the store and the chunk rankings of *this* middleware, so two filters on one agent (different
        thresholds per tool, say) never read each other's references.

        Returns:
            The LangChain tool, named :data:`_RETRIEVAL_TOOL_NAME`.
        """

        @tool(_RETRIEVAL_TOOL_NAME)
        async def retrieve_all_context(
            reference: str,
            pattern: str | None = None,
            line_range: LineRange | None = None,
            context_lines: int | None = None,
            max_chunks: int | None = None,
            max_tokens: int | None = None,
        ) -> str | dict:
            """Load the WHOLE of a tool result that the relevance filter cut down to an excerpt.

            Use it ONLY when the answer needs every row of that result -- a maximum, minimum, total,
            count, average, ranking or a comparison across all of it -- which the excerpt cannot answer.
            Do NOT use it to read a passage the excerpt already shows, or when the excerpt answers the
            question.

            When a tool result was too large to keep in context, its raw content was stored and the
            result was rewritten into an excerpt, a disclaimer with its size, and a reference. Pass that
            reference here, with a ``pattern`` that matches only the rows you need to aggregate (the
            cheapest way), or ``max_chunks``/``max_tokens`` large enough for all of it.

            Returns:
              - With line_range: exactly that span of lines, with line numbers
              - With pattern: only the matching lines, with line numbers and surrounding context. Best
                for aggregates: a pattern matching the rows to aggregate returns just those rows
              - With max_chunks: the max_chunks most relevant chunks, in document order, with markers
                for the lines left out. A max_chunks at least the result's chunk count returns all of it
              - Without any of these: the full original content, cut at max_tokens when given

            What this tool returns is removed from the conversation once the answer is given, so state
            the figures you relied on in the answer itself.

            Constraints:
              - pattern/line_range/context_lines/max_chunks/max_tokens only work on text content. For
                binary content, omit them.
              - Line numbers are 1-indexed and are the same ones the excerpt's gap markers report, so a
                "[... N lines omitted ...]" marker can be turned straight into a follow-up line_range.
              - Precedence: line_range, then pattern, then max_chunks.
              - max_tokens bounds the response in every mode. Without it, line_range, pattern and
                max_chunks answer within the filter's own size threshold, and a full read is unbounded.

            Examples:
              {"reference": "mem_1_tool-123_0", "pattern": "error"} -> matches with 5 lines of context
              {"reference": "mem_1_tool-123_0", "pattern": "REFUND", "context_lines": 0,
               "max_tokens": 20000} -> every matching row, no context, up to ~20k tokens
              {"reference": "mem_1_tool-123_0", "max_chunks": 3} -> the 3 most relevant chunks
              {"reference": "mem_1_tool-123_0", "max_chunks": 1000, "max_tokens": 50000} -> everything
              {"reference": "mem_1_tool-123_0", "line_range": {"start": 10, "end": 25}} -> lines 10-25

            Args:
                reference: The reference string from the filtered block (e.g. "mem_1_tool-123_0").
                pattern: Regex or keyword to grep for. Returns only matching lines with context.
                line_range: Return only this span of lines; a dict with 1-indexed inclusive ``start``
                    and ``end`` keys. Takes precedence over ``pattern``.
                context_lines: Lines before AND after each match, like ``grep -C``. Defaults to 5.
                max_chunks: Number of chunks to return, most relevant first, rendered in document
                    order. At least 1.
                max_tokens: Approximate size limit of the response, in tokens. At least 1.

            Raises:
                ValueError: If the reference is unknown, ``max_chunks`` or ``max_tokens`` is below 1,
                    the content is binary and any read option was supplied, or ``line_range`` *starts*
                    outside the content. An ``end`` past the last line is **not** an error: the span is
                    clamped, the way ``sed -n 'start,$p'`` behaves.
            """
            return await self._retrieve(
                reference,
                pattern=pattern,
                line_range=line_range,
                context_lines=context_lines,
                max_chunks=max_chunks,
                max_tokens=max_tokens,
            )

        return retrieve_all_context

    async def _retrieve(
        self,
        reference: str,
        *,
        pattern: str | None = None,
        line_range: LineRange | None = None,
        context_lines: int | None = None,
        max_chunks: int | None = None,
        max_tokens: int | None = None,
    ) -> str | dict:
        """Resolve a reference and render it according to the read options.

        The body of ``retrieve_all_context``, kept on the class so it is unit-testable without going
        through the tool's argument schema. The option contract is the tool's docstring.

        Args:
            reference: The reference issued when the result was filtered.
            pattern: Regex or keyword to grep for.
            line_range: 1-indexed inclusive span, taking precedence over ``pattern``.
            context_lines: Lines before and after each match. Defaults to 5.
            max_chunks: Number of chunks to render, most relevant first.
            max_tokens: Approximate response budget, in tokens.

        Returns:
            The rendered text, or a content dict for a non-textual full read.

        Raises:
            ValueError: As documented on the tool.
        """
        for name, value in (("max_chunks", max_chunks), ("max_tokens", max_tokens)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be an integer >= 1, got {value!r}")

        store = self._store
        if store is None:  # No store configured: nothing was ever stored, so nothing resolves.
            raise ValueError(f"reference not found: {reference}")

        try:
            content_bytes, content_type = await store.retrieve(reference)
        except KeyError as error:
            raise ValueError(f"reference not found: {reference}") from error

        options = (pattern, line_range, context_lines, max_chunks, max_tokens)
        if all(option is None for option in options):
            return _decode_full_content(content_bytes, content_type, reference)

        if not _is_searchable_content(content_type):
            raise ValueError(
                f"cannot search binary content ({content_type}). "
                "Omit pattern/line_range/context_lines/max_chunks/max_tokens to retrieve the full content."
            )

        text = content_bytes.decode("utf-8")
        ctx_lines = context_lines if context_lines is not None else _DEFAULT_CONTEXT_LINES
        # By default a read is bounded by the same threshold that made the result oversized, so reading
        # content back never costs more than keeping the original would have. max_tokens is the model's
        # explicit choice to pay for more.
        max_chars = (max_tokens or self._max_result_tokens) * _CHARS_PER_TOKEN

        if line_range is None and pattern is None:
            if max_chunks is not None:
                return self._read_chunks(reference, text, max_chunks, max_chars)
            if context_lines is None:
                # max_tokens alone: the whole content, cut to that budget.
                return _search_content(text, line_range=(1, max(1, text.count("\n") + 1)), max_chars=max_chars)

        span: tuple[int, int] | None = None
        if line_range is not None:
            span = (int(line_range["start"]), int(line_range["end"]))
            # A span is an exact request, so it answers on its own: the pattern is dropped rather than
            # applied within the span, which would return matches instead of the lines asked for.
            pattern = None
        elif pattern is None:
            # context_lines alone names no match to center on, so it reads the head of the content.
            span = (1, max(1, ctx_lines))

        return _search_content(text, pattern=pattern, line_range=span, context_lines=ctx_lines, max_chars=max_chars)

    def _read_chunks(self, reference: str, text: str, max_chunks: int, max_chars: int) -> str:
        """Render the ``max_chunks`` most relevant chunks of ``text`` in document order.

        The ranking is the one the filter computed when it built the excerpt, so no second scoring call
        is made. A reference with no recorded ranking -- several sub-blocks scored together, or a store
        that outlived this process -- is read in document order instead.

        Args:
            reference: The reference the text came from, the key of its ranking.
            text: The stored text.
            max_chunks: How many chunks to render.
            max_chars: Size limit of the rendering, gap markers included.

        Returns:
            A header naming what was returned, then the chunks with a marker for every omitted span.
        """
        chunks = _chunk_text(text, self._config.get("chunk_tokens", _DEFAULT_CHUNK_TOKENS))
        if not chunks:
            return "Content is empty (0 lines)."

        ranking = self._rankings.get(reference)
        if ranking is None or len(ranking) != len(chunks):
            ranking = tuple(range(len(chunks)))
            order = "document order"
        else:
            order = "most relevant first"

        chosen = sorted((chunks[i] for i in ranking[:max_chunks]), key=lambda chunk: chunk.index)
        body = _assemble_preview(chunks, chosen, max_chars)
        header = (
            f"[{len(chosen)} of {len(chunks)} chunks ({order}), lines 1-{chunks[-1].end_line} in total, "
            f"limit ~{max_chars // _CHARS_PER_TOKEN:,} tokens]"
        )
        return f"{header}\n{body}"

    # ------------------------------------------------------------------ hooks

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        """Relevance-filter an oversized textual tool result before it enters the state.

        Six guards run in order, each an early return that hands back the handler's result untouched,
        so a result that is not a ``ToolMessage``, self-produced, delegated, small, vetoed or
        non-textual is never scored. Only past all six is the result rewritten (and, if enabled,
        stored).

        Args:
            request: The tool call; ``tool_call``, ``tool`` and ``state`` are read.
            handler: Runs the tool. Called exactly once.

        Returns:
            The handler's own result, or a copy of its ``ToolMessage`` whose content is the marker,
            the disclaimer, the verbatim preview and the reference token.
        """
        result = await handler(request)
        return await self._process_result(request, result)

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        """Sync twin of :meth:`awrap_tool_call`, so the filter also runs under ``agent.invoke``.

        LangChain does not bridge a sync run to an async-only hook, while the Strands plugin runs under
        both a sync and an async agent call. The tool runs synchronously; the filtering pipeline (whose
        reranker and store are async) runs to completion on a private event loop -- on this thread when
        none is running here, on a short-lived worker thread otherwise.

        Args:
            request: The tool call; ``tool_call``, ``tool`` and ``state`` are read.
            handler: Runs the tool. Called exactly once.

        Returns:
            Same as :meth:`awrap_tool_call`.
        """
        result = handler(request)
        return _run_to_completion(self._process_result(request, result))

    async def _process_result(self, request: Any, result: Any) -> Any:
        """The six guards and the rewrite, shared by both hooks. See :meth:`awrap_tool_call`."""

        # (1) Only a ToolMessage carries content to filter. A ``Command`` is a state update, and the
        # Strands analog of this guard -- a cancelled call -- likewise has no result to rewrite.
        if not isinstance(result, ToolMessage):
            return result

        tool_call = dict(request.tool_call or {})
        tool_name = tool_call.get("name", "")

        # (2) Recursion guard: the retrieval tool's own (possibly large) output must stay intact,
        # otherwise retrieving content would re-filter it.
        if tool_name == self.retrieval_tool_name:
            return result

        # (3) A delegation result becomes the final user-facing answer, and no later model call could
        # retrieve what the filter cut. ``return_direct`` is LangChain's expression of exactly that.
        if getattr(request.tool, "return_direct", False):
            return result

        scorable, passthrough = _split_content_blocks(result)

        # (4) Size gate, with the Strands heuristic -- see _approximate_tokens. Passthrough parts are
        # binary and, as in Strands, not counted.
        token_count = _approximate_tokens(scorable)
        if token_count <= self._max_result_tokens:
            return result

        # (5) Caller veto, consulted only for a result that is already over threshold. A callback that
        # raises fails open -- filtering is the safe default, since the alternative is letting an
        # oversized result through on a bug in caller code.
        if self._should_filter is not None:
            try:
                verdict = self._should_filter(tool_name, token_count)
                if inspect.isawaitable(verdict):
                    verdict = await verdict
                if not verdict:
                    return result
            except Exception:
                logger.warning(
                    "tool_use_id=<%s> | should_filter callback failed, filtering anyway",
                    result.tool_call_id,
                    exc_info=True,
                )

        # (6) Only text and JSON sub-blocks can be relevance-scored. Empty text blocks add no content,
        # so a result made solely of those (or of binaries) is left alone.
        full_text = _rendered_text(scorable)
        if not full_text:
            return result

        return await self._filter_and_rewrite(request, result, token_count, full_text, scorable, passthrough)

    async def _store_raw(self, tool_call_id: str, scorable: list[dict[str, Any]]) -> list[str] | None:
        """Optionally store the raw scorable sub-blocks so ``retrieve_all_context`` can read them back.

        This is an add-on to filtering, never a precondition of it. With the retrieval tool off, or no
        store configured, nothing is stored and an empty list is returned.

        Args:
            tool_call_id: Id of the call whose result is stored, the stem of every reference.
            scorable: The neutral text/json blocks the preview stands in for.

        Returns:
            The references issued (empty when storage is off), or None if a write failed -- the caller
            then keeps the original result rather than hand out a reference that names nothing.
        """
        store = self._store
        if not self._include_retrieval_tool or store is None:
            return []

        references: list[str] = []
        try:
            for index, block in enumerate(scorable):
                if "text" in block:
                    raw, content_type = block["text"].encode("utf-8"), "text/plain"
                else:
                    raw, content_type = json.dumps(block["json"], indent=2).encode("utf-8"), "application/json"
                references.append(await store.store(f"{tool_call_id}_{index}", raw, content_type))
        except Exception:
            logger.warning(
                "tool_use_id=<%s> | failed to store tool result, keeping original",
                tool_call_id,
                exc_info=True,
            )
            return None
        return references

    async def _filter_and_rewrite(
        self,
        request: Any,
        result: ToolMessage,
        token_count: int,
        full_text: str,
        scorable: list[dict[str, Any]],
        passthrough: list[Any],
    ) -> ToolMessage:
        """Score the result against the question and return the rewritten ``ToolMessage``.

        Reached only past every guard in :meth:`awrap_tool_call`, with a result that is oversized and
        carries scorable text. Filtering (chunk, rerank, select, assemble) does not depend on the
        store. Storage is an optional step, run first only when enabled, so a reference token is never
        handed out for content that is not already there.

        Args:
            request: The tool call, read for the state and the arguments that shape the query.
            result: The handler's ``ToolMessage``. Never mutated; a copy is returned.
            token_count: Estimated token count of the original result, shown in the marker.
            full_text: Concatenated text and JSON sub-block content to score and preview.
            scorable: The neutral blocks ``full_text`` was rendered from.
            passthrough: The non-scorable content parts, kept verbatim after the filtered block.

        Returns:
            A copy of ``result`` carrying the marker, or ``result`` itself when storing or scoring
            failed.
        """
        tool_call_id = result.tool_call_id

        references = await self._store_raw(tool_call_id, scorable)
        if references is None:
            return result

        query = self._build_query(request.state, dict(request.tool_call or {}))
        try:
            preview, stats = await self._resolve_preview().build_with_stats(full_text, query)
        except RerankerError:
            # Abstaining is safer than a positional slice: a wrong cut drops the passage the question
            # needed. The result stays as it came, and is never scored a second time.
            logger.debug("tool_use_id=<%s> | relevance scoring failed, keeping original", tool_call_id)
            return result

        # The ranking indexes chunks of ``full_text``, so it is only valid for a reference holding
        # exactly that text: one scorable sub-block, stored whole.
        if len(references) == 1 and len(scorable) == 1:
            self._rankings[references[0]] = stats.ranking

        disclaimer = _disclaimer(token_count, stats, references, self.retrieval_tool_name)
        marker = f"[Relevance: tool result, ~{token_count:,} tokens]\n{disclaimer}\n\n{preview}"
        if references:
            token = f"[ref: {references[0]}]" if len(references) == 1 else f"[refs: {', '.join(references)}]"
            marker = f"{marker}\n\n{token}"

        logger.debug(
            "tool_use_id=<%s>, refs=<%d>, tokens=<%d> | tool result relevance-filtered",
            tool_call_id,
            len(references),
            token_count,
        )

        # The textual sub-blocks are what the preview stands in for, so only the non-textual ones
        # survive, verbatim and in their original order, after the filtered block.
        content: Any = marker if not passthrough else [{"type": "text", "text": marker}, *passthrough]
        return result.model_copy(update={"content": content})

    def after_agent(self, state: Any, runtime: Any = None) -> dict[str, Any] | None:
        """Remove this middleware's ``retrieve_all_context`` exchanges from the state once the run ends.

        The ``AfterInvocationEvent`` analog. A retrieval is for the answer in flight. Kept, its result
        -- possibly the whole of a large tool result, since the model may ask for all of it -- would
        ride along on every later call and be carded by a context graph as evidence of the turn. What
        stays is the excerpt with its reference, so the content can still be retrieved again later, and
        the answer, which carries the figures the retrieval served.

        The update is expressed as ``RemoveMessage(id=REMOVE_ALL_MESSAGES)`` followed by the reduced
        list, because the ``messages`` reducer *merges* a returned list rather than replacing it: a
        plain list could never drop a message. This is the only place this middleware writes persisted
        state, and it only ever removes its own closed retrieval exchanges.

        Args:
            state: The finished agent state; ``messages`` is read.
            runtime: The runtime context. Unused.

        Returns:
            The state update, or None when nothing is removed.
        """
        if not self._include_retrieval_tool:
            return None
        messages = list(_state_messages(state))
        kept = _drop_tool_exchanges(messages, self.retrieval_tool_name)
        if kept is None:
            return None
        logger.debug("messages=<%d->%d> | retrieval exchanges removed from the state", len(messages), len(kept))
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *kept]}

    async def aafter_agent(self, state: Any, runtime: Any = None) -> dict[str, Any] | None:
        """Async form of :meth:`after_agent`. The cleanup is pure list surgery, so it delegates."""
        return self.after_agent(state, runtime)


def _state_messages(state: Any) -> Sequence[BaseMessage]:
    """Read ``messages`` out of an agent state that may be a dict or a model.

    Args:
        state: The agent state, or None.

    Returns:
        The messages, oldest first; empty when the state carries none.
    """
    if state is None:
        return []
    if isinstance(state, dict):
        return state.get("messages") or []
    return getattr(state, "messages", None) or []


def _split_content_blocks(message: ToolMessage) -> tuple[list[dict[str, Any]], list[Any]]:
    """Split a ``ToolMessage``'s content into scorable neutral blocks and verbatim passthrough parts.

    The adapter renders the message as a neutral ``toolResult`` block, so the split is over neutral
    ``{"text": ...}`` / ``{"json": ...}`` blocks. A ``json`` block that is really a non-textual
    LangChain content part (an image, a file, ...) carries nothing to score, so it is handed back as a
    passthrough part in its original form.

    Args:
        message: The tool result to split.

    Returns:
        The scorable neutral blocks, and the original content parts to keep verbatim.
    """
    block = tool_message_to_result_block(message)
    scorable: list[dict[str, Any]] = []
    passthrough: list[Any] = []
    for neutral in block["toolResult"]["content"]:
        if "text" in neutral:
            scorable.append(neutral)
            continue
        payload = neutral.get("json")
        if isinstance(payload, dict) and payload.get("type") in _NON_SCORABLE_BLOCK_TYPES:
            passthrough.append(payload)
        else:
            scorable.append(neutral)
    return scorable, passthrough


def _rendered_text(blocks: Sequence[Any]) -> str:
    """Render neutral blocks (or raw content parts) to the text the filter scores and sizes.

    Args:
        blocks: Neutral ``text``/``json`` blocks, or raw passthrough parts.

    Returns:
        The blocks' text joined by line breaks; JSON is pretty-printed, as in the Strands plugin.
    """
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, dict) and block.get("text"):
            parts.append(block["text"])
        elif isinstance(block, dict) and "json" in block:
            parts.append(json.dumps(block["json"], indent=2))
        else:
            parts.append(json.dumps(block, indent=2, default=str))
    return "\n".join(part for part in parts if part)


def _decode_full_content(content_bytes: bytes, content_type: str, reference: str) -> str | dict:
    """Decode stored content back into its native shape for a full read.

    Args:
        content_bytes: The raw stored bytes.
        content_type: The MIME type recorded alongside them.
        reference: The reference the bytes came from, used to name a document part.

    Returns:
        Text as a string, and anything else as a LangChain content part dict.
    """
    if content_type.startswith("text/"):
        return content_bytes.decode("utf-8")

    if content_type == "application/json":
        return json.loads(content_bytes)

    if content_type.startswith("image/"):
        return {"type": "image", "mime_type": content_type, "base64": base64.b64encode(content_bytes).decode("ascii")}

    if content_type.startswith("application/"):
        return {
            "type": "file",
            "mime_type": content_type,
            "name": reference,
            "base64": base64.b64encode(content_bytes).decode("ascii"),
        }

    # Unknown type: a lossy decode still shows the model something rather than failing the read.
    return content_bytes.decode("utf-8", errors="replace")
