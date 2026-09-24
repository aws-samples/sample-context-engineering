"""The ``RelevanceFilter`` plugin: configuration, the ``AfterToolCallEvent`` hook, and the retrieval tool.

Attaches to the SDK's public extension surface only. The hook guards a tool result through the
cancelled-call, own-tool, delegation, size, ``should_filter``, and emptiness checks, then stores the raw
sub-blocks and rewrites ``event.result`` into the marker plus a verbatim, budget-bounded preview plus
reference tokens. The ``retrieve_context`` ``@tool`` reads those references back by span or pattern.
"""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Awaitable, Sequence
from typing import TYPE_CHECKING, Any, Protocol, TypedDict

# Imported at runtime, not under TYPE_CHECKING: @hook resolves the handler's type hints at decoration
# time to infer which event it subscribes to.
from strands import tool
from strands.hooks.events import AfterToolCallEvent
from strands.plugins import Plugin, hook
from strands.types.tools import ToolContext, ToolResult, ToolResultContent

from .preview import RelevancePreview
from .reranker import BedrockReranker, Reranker, RerankerError
from .search import _is_searchable_content, _search_content
from .store import InMemoryStore, Store

if TYPE_CHECKING:
    from strands.agent.agent import Agent
    from strands.types.content import Message

try:
    from strands.agent._agent_as_tool import _AgentAsTool
except ImportError:  # pragma: no cover - the SDK may rename or move this private class.
    # Without the class there is no way to recognize a delegation tool, so that guard degrades to
    # the caller's own ``should_filter`` (or is documented as their responsibility).
    _AgentAsTool = None  # type: ignore[assignment,misc]

__all__ = [
    "LineRange",
    "RelevanceConfig",
    "RelevanceFilter",
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
"""Approximate characters per token, the conversion used to bound a retrieval response."""

_MAX_QUERY_CHARS = 2_000
"""Cap on the scoring query. Rerankers charge per query length, and a query longer than this adds
context without sharpening the ranking."""


def _latest_question(messages: Sequence[Message]) -> str:
    """Return the newest user message's text, or ``""`` when the history holds none.

    A turn carrying only a tool result is not a question, so only ``user`` messages with at least one
    ``text`` block count. Multiple text blocks in that message are joined by a line break.

    Args:
        messages: The conversation history, oldest first.

    Returns:
        The newest user question, or ``""``.
    """
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        texts = [block["text"] for block in message.get("content", []) if block.get("text")]
        if texts:
            return "\n".join(texts)
    return ""


class RelevanceConfig(TypedDict, total=False):
    """Tuning for the relevance preview.

    All keys are optional; the plugin reads each with ``dict.get`` and the default documented here, so
    a partial config is as valid as a full one and an omitted key never has to be spelled out.

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


class RelevanceFilter(Plugin):
    """Relevance-filter oversized tool results on the public ``AfterToolCallEvent`` hook.

    When a tool result exceeds ``max_result_tokens``, its raw sub-blocks are written to ``store`` and
    the in-context result is replaced with a marker, a verbatim relevance preview scored against the
    query in progress, and reference tokens the model can pass to ``retrieve_context``. Selection is
    verbatim — chosen chunks reach the model character-for-character — so numeric, monetary, and
    tabular content stays exact.

    Construction is inert: no ``Reranker``, no preview builder, and no AWS client is created here. The
    reranker is built on the first scoring call, so a filter that never fires costs nothing and needs
    no AWS credentials or region merely to be declared.

    Args:
        store: Optional backend for the raw sub-blocks, used only by the retrieval tool. Filtering
            never depends on it. When None and ``include_retrieval_tool`` is True, a per-agent
            ``InMemoryStore`` is created during initialization; otherwise no store exists.
        max_result_tokens: Filter only results whose estimated token count exceeds this threshold.
            Defaults to ``8_000``.
        config: Preview tuning (reranker, threshold, chunk and preview budgets). Every key is
            optional; see :class:`RelevanceConfig` for the defaults.
        include_retrieval_tool: Whether to register the ``retrieve_context`` tool so the model can
            read the raw content back. Defaults to False: the filter's job ends at the tool result,
            and a retrieval cycle's own result becomes a conversation message that rides along on
            every later call, so reading a cut chunk back is paid for repeatedly. Off, no raw
            sub-block is stored and no reference token is emitted -- nothing could resolve one.
        should_filter: Callback deciding whether a specific oversized result is filtered. Called only
            once the result is over threshold. Defaults to None (every oversized result is filtered).

    Example:
        ```python
        from strands import Agent
        from strands_relevance_filter import RelevanceFilter

        agent = Agent(plugins=[RelevanceFilter(max_result_tokens=8_000)])
        ```
    """

    name = "strands-community:relevance-filter"
    """Plugin name, following the ``vendor:plugin`` convention. Overridable on a subclass so two
    filters (different thresholds per tool, say) can coexist on one agent."""

    def __init__(
        self,
        *,
        store: Store | None = None,
        max_result_tokens: int = _DEFAULT_MAX_RESULT_TOKENS,
        config: RelevanceConfig | None = None,
        include_retrieval_tool: bool = False,
        should_filter: ShouldFilter | None = None,
    ) -> None:
        """Initialize the plugin without building any scoring or network dependency.

        Args:
            store: Optional backend for the raw sub-blocks. When None, an ``InMemoryStore`` is
                created per agent during initialization only if the retrieval tool is enabled.
            max_result_tokens: Filter only results above this estimated token count.
            config: Preview tuning; read key by key with the documented defaults at use time, so a
                partial config is as valid as a full one.
            include_retrieval_tool: Register the ``retrieve_context`` tool. Defaults to False, which
                also suppresses the store write and the reference token: with no tool to resolve it,
                a reference would be a promise nothing can keep.
            should_filter: Callback ``(tool_name, token_count, **kwargs) -> bool``, sync or async.

        Raises:
            ValueError: If ``max_result_tokens`` is not positive.
        """
        if max_result_tokens <= 0:
            raise ValueError("max_result_tokens must be positive")

        self._store: Store | None = store
        self._max_result_tokens = max_result_tokens
        self._config: RelevanceConfig = config if config is not None else {}
        self._include_retrieval_tool = include_retrieval_tool
        self._should_filter = should_filter
        # Built on first use, never here: constructing a reranker would reach for AWS credentials.
        self._preview: RelevancePreview | None = None
        # The base scans this instance for @hook and @tool methods, so it runs last.
        super().__init__()

    def init_agent(self, agent: Agent) -> None:
        """Bind the store to this agent and conditionally drop the retrieval tool.

        The store is resolved on the first call and kept for the instance lifetime, so one filter
        should not be shared across agents with differing stores. An ``InMemoryStore`` is claimed by
        the agent it first sees, because its eviction cycle belongs to a single agent loop.

        Args:
            agent: The agent this plugin instance is being attached to.
        """
        # The store is optional: it exists only to serve the retrieval tool. Filtering never needs it,
        # so a default store is built only when something can read it back.
        if self._store is None and self._include_retrieval_tool:
            self._store = InMemoryStore()
        if isinstance(self._store, InMemoryStore):
            self._store._bind(id(agent))
        if not self._include_retrieval_tool:
            # Drop the auto-discovered retrieval tool, matched by tool_name rather than a literal.
            retrieval_tool_name = self.retrieve_context.tool_name
            self._tools = [t for t in self._tools if t.tool_name != retrieval_tool_name]

    def _resolve_preview(self) -> RelevancePreview:
        """Return the preview builder, constructing it on the first call.

        Deferring construction is what keeps ``__init__`` inert: the default
        :class:`~strands_relevance_filter.reranker.BedrockReranker` opens an AWS client, so a filter
        that never fires never needs credentials or a region. The builder is created once and reused,
        which also makes its ``search_units`` counter a session total.

        The config is read key by key with the documented defaults and is otherwise unvalidated — a
        nonsensical value surfaces where it is used rather than at construction (``chunk_tokens < 1``,
        for instance, raises ``ValueError`` from chunking on the first filtered result).

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

    def _build_query(self, event: AfterToolCallEvent) -> str:
        """Compose the scoring query from the latest user question and the tool-call arguments.

        The hook receives ``event.tool_use`` directly, so the arguments are read from it rather than
        walked out of history; only the question comes from ``event.agent.messages``. Both signals
        matter: the question states the goal, the arguments state the sub-goal of this particular call,
        which is what keeps selection on target in a multi-tool loop. The tool *name* is deliberately
        left out — it names the capability, not what is being looked for, and it biases the ranking
        toward chunks that echo the name.

        The cap keeps the **tail**, so a long question cannot push the arguments out of the query.
        Arguments that alone exceed the cap are kept as a head instead, since their leading keys are
        the identifying part. When the arguments cannot be serialized the question tail stands alone,
        falling back to the literal ``"{}"`` so the reranker never receives an empty query.

        Args:
            event: The event carrying the tool call and the agent whose history holds the question.

        Returns:
            The scoring query, between 1 and ``_MAX_QUERY_CHARS`` characters.
        """
        user_text = _latest_question(event.agent.messages)

        try:
            serialized = json.dumps(event.tool_use.get("input", {}))
        except (TypeError, ValueError):
            return user_text[-_MAX_QUERY_CHARS:] or "{}"

        if len(serialized) >= _MAX_QUERY_CHARS:
            return serialized[:_MAX_QUERY_CHARS]

        query = f"{user_text}\n{serialized}" if user_text else serialized
        return query[-_MAX_QUERY_CHARS:]

    @tool(context=True)
    async def retrieve_context(
        self,
        reference: str,
        tool_context: ToolContext,
        pattern: str | None = None,
        line_range: LineRange | None = None,
        context_lines: int | None = None,
    ) -> dict | str:
        """Read back content that the relevance filter replaced with a preview.

        When a tool result was too large to keep in context, its raw content was stored and the result
        was rewritten into a preview plus a reference. Use this tool with that reference to reach the
        parts the preview left out.

        Returns:
          - With line_range: exactly that span of lines, with line numbers
          - With pattern: only the matching lines, with line numbers and surrounding context
          - Without pattern/line_range/context_lines: the full original content (use sparingly — this
            re-injects every token the filter removed)

        Constraints:
          - pattern/line_range/context_lines only work on text content. For binary content, omit them.
          - Line numbers are 1-indexed and are the same ones the preview's gap markers report, so a
            "[... N lines omitted ...]" marker can be turned straight into a follow-up line_range.
          - A valid line_range wins over a pattern: the span is returned and the pattern is ignored.

        Examples:
          {"reference": "mem_1_tool-123_0", "pattern": "error"} -> matches with 5 lines of context
          {"reference": "mem_1_tool-123_0", "pattern": "error|warning", "context_lines": 3} -> regex
          {"reference": "mem_1_tool-123_0", "line_range": {"start": 10, "end": 25}} -> lines 10-25

        Args:
            reference: The reference string from the filtered block (e.g. "mem_1_tool-123_0").
            tool_context: Injected by the framework. Not user-facing.
            pattern: Regex or keyword to grep for. Returns only matching lines with context — never the
                full content.
            line_range: Return only this span of lines; a dict with 1-indexed inclusive ``start`` and
                ``end`` keys. Takes precedence over ``pattern``.
            context_lines: Lines before AND after each match, like ``grep -C``. Defaults to 5.

        Raises:
            ValueError: If the reference is unknown, the content is binary and
                ``pattern``/``line_range``/``context_lines`` were supplied, or ``line_range`` *starts*
                outside the content — its ``start`` is below 1, above ``end``, or past the last line.
                An ``end`` past the last line is **not** an error: the span is clamped to the content,
                the way ``sed -n 'start,$p'`` behaves, so asking for more lines than exist returns
                everything from ``start`` onwards.
        """
        store = self._store
        if store is None:  # No store configured: nothing was ever stored, so nothing resolves.
            raise ValueError(f"reference not found: {reference}")

        try:
            content_bytes, content_type = await store.retrieve(reference)
        except KeyError as error:
            raise ValueError(f"reference not found: {reference}") from error

        if pattern is None and line_range is None and context_lines is None:
            return self._decode_full_content(content_bytes, content_type, reference)

        if not _is_searchable_content(content_type):
            raise ValueError(
                f"cannot search binary content ({content_type}). "
                "Omit pattern/line_range/context_lines to retrieve the full content."
            )

        text = content_bytes.decode("utf-8")
        ctx_lines = context_lines if context_lines is not None else _DEFAULT_CONTEXT_LINES
        # A retrieval response is bounded by the same threshold that made the result oversized, so
        # reading content back can never cost more context than keeping the original would have.
        max_chars = self._max_result_tokens * _CHARS_PER_TOKEN

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

    @staticmethod
    def _decode_full_content(content_bytes: bytes, content_type: str, reference: str) -> dict | str:
        """Decode stored content back into its native shape for a full read.

        Args:
            content_bytes: The raw stored bytes.
            content_type: The MIME type recorded alongside them.
            reference: The reference the bytes came from, used to name a document block.

        Returns:
            Text as a string, and anything else as a tool-result dict carrying the native block.
        """
        if content_type.startswith("text/"):
            return content_bytes.decode("utf-8")

        if content_type == "application/json":
            return {"status": "success", "content": [{"json": json.loads(content_bytes)}]}

        if content_type.startswith("image/"):
            img_format = content_type.split("/")[-1]
            return {
                "status": "success",
                "content": [{"image": {"format": img_format, "source": {"bytes": content_bytes}}}],
            }

        if content_type.startswith("application/"):
            doc_format = content_type.split("/")[-1]
            doc_block = {"format": doc_format, "name": reference, "source": {"bytes": content_bytes}}
            return {"status": "success", "content": [{"document": doc_block}]}

        # Unknown type: a lossy decode still shows the model something rather than failing the read.
        return content_bytes.decode("utf-8", errors="replace")

    @hook  # type: ignore[call-overload]  # bound method; the @hook overloads describe a one-arg callback
    async def _on_after_tool_call(self, event: AfterToolCallEvent) -> None:
        """Relevance-filter an oversized textual tool result before it enters the conversation.

        Six guards run in order, each an early return that leaves ``event.result`` untouched, so a
        result that is cancelled, self-produced, delegated, small, vetoed, or non-textual is never
        scored. Only past all six is the result filtered and rewritten (and, if enabled, stored).

        Args:
            event: The completed tool call. Only ``result`` is ever written.
        """
        # (1) A cancelled call carries no result worth filtering.
        if event.cancel_message is not None:
            return

        # (2) Recursion guard: the retrieval tool's own (possibly large) output must stay intact,
        # otherwise retrieving content would re-filter it. Matched by tool_name, never a literal.
        if event.tool_use.get("name") == self.retrieve_context.tool_name:
            return

        # (3) A delegation result becomes the final user-facing answer, and no later model call could
        # retrieve what the filter cut.
        if _AgentAsTool is not None and isinstance(event.selected_tool, _AgentAsTool) and event.selected_tool.delegate:
            return

        result = event.result
        tool_use_id = event.tool_use["toolUseId"]

        # (4) Size gate. Wrapping the result as a message is what lets the model's own tokenizer
        # count it, so the threshold matches what the result would actually cost in context.
        tool_result_message: Message = {"role": "user", "content": [{"toolResult": result}]}
        token_count = await event.agent.model.count_tokens([tool_result_message])
        if token_count <= self._max_result_tokens:
            return

        # (5) Caller veto, consulted only for a result that is already over threshold. A callback that
        # raises fails open — filtering is the safe default, since the alternative is letting an
        # oversized result through on a bug in caller code.
        if self._should_filter is not None:
            try:
                verdict = self._should_filter(event.tool_use.get("name", ""), token_count)
                if inspect.isawaitable(verdict):
                    verdict = await verdict
                if not verdict:
                    return
            except Exception:
                logger.warning(
                    "tool_use_id=<%s> | should_filter callback failed, filtering anyway",
                    tool_use_id,
                    exc_info=True,
                )

        # (6) Only text and JSON sub-blocks can be relevance-scored. Empty text blocks add no content,
        # so a result made solely of those (or of binaries) is left alone.
        text_parts = [
            block["text"] if block.get("text") else json.dumps(block["json"], indent=2)
            for block in result["content"]
            if block.get("text") or "json" in block
        ]
        full_text = "\n".join(text_parts)
        if not full_text:
            return

        await self._filter_and_rewrite(event, token_count, full_text)

    async def _store_raw(self, event: AfterToolCallEvent) -> list[str] | None:
        """Optionally store the raw scorable sub-blocks so ``retrieve_context`` can read them back.

        This is an add-on to filtering, never a precondition of it. With the retrieval tool off, or
        no store configured, nothing is stored and an empty list is returned.

        Args:
            event: The completed tool call whose raw sub-blocks are stored.

        Returns:
            The references issued (empty when storage is off), or None if a write failed -- the caller
            then keeps the original result rather than hand out a reference that names nothing.
        """
        store = self._store
        if not self._include_retrieval_tool or store is None:
            return []

        tool_use_id = event.tool_use["toolUseId"]
        references: list[str] = []
        try:
            for index, block in enumerate(event.result["content"]):
                # Only the scorable sub-blocks are stored: they are the ones the preview replaces,
                # so they are the only ones the model can still need to read back.
                if block.get("text"):
                    raw, content_type = block["text"].encode("utf-8"), "text/plain"
                elif "json" in block:
                    raw, content_type = json.dumps(block["json"], indent=2).encode("utf-8"), "application/json"
                else:
                    continue
                references.append(await store.store(f"{tool_use_id}_{index}", raw, content_type))
        except Exception:
            logger.warning(
                "tool_use_id=<%s> | failed to store tool result, keeping original",
                tool_use_id,
                exc_info=True,
            )
            return None
        return references

    async def _filter_and_rewrite(self, event: AfterToolCallEvent, token_count: int, full_text: str) -> None:
        """Score the result against the question and rewrite ``event.result`` into marker + preview.

        Reached only past every guard in :meth:`_on_after_tool_call`, with a result that is oversized
        and carries scorable text. Filtering (chunk, rerank, select, assemble) does not depend on the
        store. Storage is an optional step, run first only when enabled, so a reference token is never
        handed out for content that is not already there.

        Args:
            event: The completed tool call whose ``result`` is rewritten in place.
            token_count: Estimated token count of the original result, shown in the marker.
            full_text: Concatenated text and JSON sub-block content to score and preview.
        """
        result = event.result
        content = result["content"]
        tool_use_id = event.tool_use["toolUseId"]

        references = await self._store_raw(event)
        if references is None:
            return

        query = self._build_query(event)
        try:
            preview = await self._resolve_preview().build(full_text, query)
        except RerankerError:
            # Abstaining is safer than a positional slice: a wrong cut drops the passage the
            # question needed. The result stays as it came, and is never scored a second time.
            logger.debug("tool_use_id=<%s> | relevance scoring failed, keeping original", tool_use_id)
            return

        marker = f"[Relevance: tool result, ~{token_count:,} tokens]\n\n{preview}"
        if references:
            token = f"[ref: {references[0]}]" if len(references) == 1 else f"[refs: {', '.join(references)}]"
            marker = f"{marker}\n\n{token}"

        logger.debug(
            "tool_use_id=<%s>, refs=<%d>, tokens=<%d> | tool result relevance-filtered",
            tool_use_id,
            len(references),
            token_count,
        )

        # The textual sub-blocks are what the preview stands in for, so only the non-textual ones
        # survive, verbatim and in their original order, after the filtered block.
        new_content: list[ToolResultContent] = [ToolResultContent(text=marker)]
        new_content.extend(block for block in content if not block.get("text") and "json" not in block)

        event.result = ToolResult(
            toolUseId=result["toolUseId"],
            status=result["status"],
            content=new_content,
        )
