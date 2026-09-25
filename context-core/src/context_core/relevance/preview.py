"""Deterministic preview primitives: chunking, protected-content detection, selection, and assembly.

Holds the pure, network-free core of the filter — the frozen ``Chunk``, the chunker, the
protected-content detector, score validation, threshold-and-budget selection, gap-marker assembly, and
the ``RelevancePreview`` that composes them. Nothing here performs I/O; the only remote call of the
pipeline sits behind the ``Reranker`` protocol in :mod:`context_core.relevance.reranker`.

Framework-agnostic: every function here takes and returns plain strings and dataclasses, so a binding
extracts the text of a neutral tool-result block (see :mod:`context_core.message`) and hands it over.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .reranker import Reranker, RerankerError

# ``RerankerError`` is re-exported here because it is the failure type of this module's score
# validation: a caller catching a scoring failure imports it from either module.
__all__ = ["Chunk", "PreviewStats", "RelevancePreview", "RerankerError"]

_CHARS_PER_TOKEN = 4
"""Approximate characters per token, the conversion used for every budget in this module."""


@dataclass(frozen=True)
class Chunk:
    """A contiguous, verbatim slice of the original text.

    Attributes:
        index: Position in the chunk sequence, 0-based.
        text: Exact substring of the source text, never normalized.
        start_line: 1-indexed, inclusive line where the chunk starts.
        end_line: 1-indexed, inclusive line where the chunk ends.
    """

    index: int
    text: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class PreviewStats:
    """What one preview selection did, as reported to the model and reused by retrieval.

    Attributes:
        total_chunks: Chunks the text was split into.
        total_lines: Lines of the source text.
        shown: The chunks the preview renders, in ascending index order. The last one may be
            truncated by the preview budget.
        ranking: Every chunk index, ordered by descending relevance score (ties by index).
    """

    total_chunks: int
    total_lines: int
    shown: tuple[Chunk, ...]
    ranking: tuple[int, ...]

    @property
    def shown_spans(self) -> list[tuple[int, int]]:
        """Line spans the preview covers, adjacent chunks merged, 1-indexed and inclusive."""
        spans: list[tuple[int, int]] = []
        for chunk in self.shown:
            if spans and chunk.start_line == spans[-1][1] + 1:
                spans[-1] = (spans[-1][0], chunk.end_line)
            else:
                spans.append((chunk.start_line, chunk.end_line))
        return spans


def _split_keeping_newlines(text: str) -> list[str]:
    r"""Split ``text`` on ``"\n"``, keeping the separator on the line it terminates."""
    parts = text.split("\n")
    return [part + "\n" for part in parts[:-1]] + [parts[-1]]


def _chunk_text(text: str, chunk_tokens: int) -> list[Chunk]:
    """Split ``text`` into verbatim chunks bounded by a character budget.

    Boundaries fall right after a newline, which stays with the chunk it terminates.
    A single line longer than the budget is cut by character so the algorithm does not
    degenerate into one giant chunk (the minified JSON case); every fragment of such a
    line inherits the line's own ``start_line`` and ``end_line``.

    Args:
        text: The source text. An empty string yields an empty list.
        chunk_tokens: Approximate token budget per chunk; must be greater than zero.

    Returns:
        Chunks with contiguous 0-based indices whose concatenation reproduces ``text``
        character by character.

    Raises:
        ValueError: If ``chunk_tokens`` is less than 1.
    """
    if chunk_tokens < 1:
        raise ValueError(f"chunk_tokens must be >= 1, got {chunk_tokens}")

    if not text:
        return []

    max_chars = chunk_tokens * _CHARS_PER_TOKEN
    chunks: list[Chunk] = []

    def emit(chunk_text: str, start_line: int, end_line: int) -> None:
        if not chunk_text:
            return
        chunks.append(Chunk(index=len(chunks), text=chunk_text, start_line=start_line, end_line=end_line))

    buffer: list[str] = []
    buffer_len = 0
    first_line = 1
    line_no = 1

    for line in _split_keeping_newlines(text):
        # A line longer than the budget cannot fit any chunk: cut it by character.
        if len(line) > max_chars:
            if buffer_len:
                emit("".join(buffer), first_line, line_no - 1)
            buffer = []
            buffer_len = 0
            for start in range(0, len(line), max_chars):
                emit(line[start : start + max_chars], line_no, line_no)
            first_line = line_no + 1
            line_no += 1
            continue

        if buffer_len + len(line) > max_chars and buffer_len:
            emit("".join(buffer), first_line, line_no - 1)
            buffer = []
            buffer_len = 0
            first_line = line_no

        buffer.append(line)
        buffer_len += len(line)
        line_no += 1

    if buffer_len:
        emit("".join(buffer), first_line, line_no - 1)

    return chunks


def _truncate_at_line_boundary(text: str, max_chars: int) -> str:
    """Cut ``text`` down to ``max_chars`` characters, preferring a line boundary.

    Keeps the leading complete lines up to the last boundary that fits. When not even
    the first line fits, falls back to a character count cut so the caller still gets
    the most content the budget allows.

    Args:
        text: The source text. The return value is always a prefix of it.
        max_chars: Maximum number of characters to keep. Values below 1 yield ``""``.

    Returns:
        A prefix of ``text`` with at most ``max_chars`` characters.
    """
    if max_chars < 1:
        return ""

    if len(text) <= max_chars:
        return text

    kept = 0
    for line in _split_keeping_newlines(text):
        if kept + len(line) > max_chars:
            break
        kept += len(line)

    # No line boundary fits the budget: cut by character count instead.
    return text[:kept] if kept else text[:max_chars]


_PROTECTED_CONTENT_PATTERN = re.compile(
    "|".join(
        (
            r"\d[.,]\d",  # decimal or thousand separator between digits
            r"R\$|[$€]|USD|BRL",  # currency symbols and codes
            r"\|[^|\n]*\||\t[^\t\n]*\t",  # two or more table delimiters on the same line
            r"\d{3}",  # three or more consecutive digits
        )
    )
)
"""Patterns that mark text as protected content.

Constant, never built from user input, so there is no ReDoS surface here.
"""


def _has_protected_content(text: str) -> bool:
    """Report whether ``text`` holds numeric, monetary or tabular content.

    Protected content must reach the model verbatim, so this detector gates any
    paraphrasing step. A false positive is cheap (the text passes through unchanged);
    a false negative would let a summarizer rewrite a number, so the patterns err on
    the permissive side.

    Deterministic and pure: no I/O, no state, no mutation of ``text``.

    Args:
        text: The text to classify. An empty string is never protected.

    Returns:
        ``True`` when at least one line carries a decimal or thousand separator between
        digits, a ``R$``, ``$``, ``€``, ``USD`` or ``BRL`` marker, two or more ``|`` or
        tab characters, or a run of three or more digits.
    """
    return _PROTECTED_CONTENT_PATTERN.search(text) is not None


def _validate_scores(scores: object, chunk_count: int) -> list[float]:
    """Check a reranker result against the scoring contract before it is used.

    A third-party reranker may violate the contract, and a bad score list would
    silently corrupt the selection: a wrong length misaligns scores from chunks, and a
    value outside ``[0.0, 1.0]`` breaks the threshold comparison. Failing here turns
    the violation into the same ``RerankerError`` a transport failure raises, so the
    caller leaves the block unchanged through a single path.

    Booleans are rejected: ``True`` would pass as ``1.0`` and hide a broken scorer.

    Args:
        scores: The value returned by :meth:`Reranker.score`.
        chunk_count: Number of chunks submitted for scoring.

    Returns:
        The scores as a new list of floats, safe to align with the chunks by index.

    Raises:
        RerankerError: If ``scores`` is not a sequence of exactly ``chunk_count``
            finite numbers within ``[0.0, 1.0]``.
    """
    if not isinstance(scores, (list, tuple)):
        raise RerankerError(f"reranker must return a list of scores, got {type(scores).__name__}")

    if len(scores) != chunk_count:
        raise RerankerError(f"reranker returned {len(scores)} scores for {chunk_count} chunks")

    validated: list[float] = []
    for position, score in enumerate(scores):
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise RerankerError(f"score at index {position} is not a number: {score!r}")
        if not math.isfinite(score):
            raise RerankerError(f"score at index {position} is not finite: {score!r}")
        if not 0.0 <= score <= 1.0:
            raise RerankerError(f"score at index {position} is outside [0.0, 1.0]: {score!r}")
        validated.append(float(score))

    return validated


def _select_chunks(
    chunks: list[Chunk],
    scores: list[float],
    threshold: float,
    budget_chars: int,
) -> list[Chunk]:
    """Pick the chunks that best fill the preview budget.

    Candidates are the chunks scoring at or above ``threshold``. They are visited by
    descending score, ties broken by ascending index, and every candidate is visited:
    a smaller, lower-scored chunk may still fit after a large one was skipped.

    Two guards keep the selection non-empty, because an empty preview leaves the model
    blind about what the tool returned:

    - No chunk reaches the threshold: the highest-scored chunk is selected anyway.
    - The highest-scored candidate alone exceeds the budget: it is selected and the
      truncation is delegated to the assembly step.

    Args:
        chunks: Chunks in ascending index order, as produced by :func:`_chunk_text`.
        scores: Scores aligned by position to ``chunks``.
        threshold: Minimum score for a chunk to be eligible.
        budget_chars: Maximum total characters of the selection.

    Returns:
        The selected chunks in ascending index order, each at most once. Empty only when
        ``chunks`` is empty. The total length stays within ``budget_chars`` except when a
        single chunk is returned by one of the guards.

    Raises:
        ValueError: If ``scores`` and ``chunks`` have different lengths.
    """
    if len(scores) != len(chunks):
        raise ValueError(f"scores must align with chunks, got {len(scores)} scores for {len(chunks)} chunks")

    if not chunks:
        return []

    scored = list(zip(chunks, scores, strict=True))

    def by_descending_score(pair: tuple[Chunk, float]) -> tuple[float, int]:
        chunk, score = pair
        return (-score, chunk.index)

    candidates = sorted((pair for pair in scored if pair[1] >= threshold), key=by_descending_score)

    # Empty-selection guard: nothing reached the threshold, keep the best chunk anyway.
    if not candidates:
        return [min(scored, key=by_descending_score)[0]]

    selected: list[Chunk] = []
    used_chars = 0
    for chunk, _score in candidates:
        # No early break: a shorter, lower-scored candidate may still fit the budget.
        if used_chars + len(chunk.text) <= budget_chars:
            selected.append(chunk)
            used_chars += len(chunk.text)

    # The highest-scored candidate alone busts the budget; assembly truncates it.
    if not selected:
        return [candidates[0][0]]

    return sorted(selected, key=lambda chunk: chunk.index)


_GAP_MARKER = "\n[... {n} lines omitted ...]\n"
"""Template for the only preview segment that does not come from the source text."""


def _format_gap_marker(omitted_lines: int) -> str:
    """Render a gap marker for ``omitted_lines`` raw lines left out of the preview."""
    return _GAP_MARKER.format(n=omitted_lines)


def _trailing_gap_marker(chunk: Chunk, kept: str, total_lines: int) -> str:
    """Build the closing marker for a truncated last chunk.

    The line that got cut in the middle counts as omitted: the model needs to fetch it
    again to see it whole.

    One consequence worth knowing, because the number can look wrong: a truncation falling
    inside the *first* line of the last chunk makes ``first_unshown_line`` that line itself,
    so the marker reports every line of the source as omitted — "29 lines omitted" on a
    29-line text — even though a fragment of line 1 was rendered. That is the rule above
    applied consistently rather than an off-by-one: the partially shown line is not readable
    in full, so it is counted as missing. The count is therefore a safe lower bound on what a
    follow-up ``line_range`` should ask for, never an exact tally of untouched lines.

    Args:
        chunk: The chunk that was truncated.
        kept: The prefix of ``chunk.text`` that made it into the preview.
        total_lines: Line count of the source text, 1-indexed.

    Returns:
        The marker, or ``""`` when nothing is left out after the truncation point.
    """
    first_unshown_line = chunk.start_line + kept.count("\n")
    omitted = total_lines - first_unshown_line + 1
    return _format_gap_marker(omitted) if omitted > 0 else ""


def _assemble_preview(chunks: list[Chunk], selected: list[Chunk], budget_chars: int) -> str:
    """Concatenate the selected chunks verbatim, flagging every omission.

    Chunks are joined in ascending index order and each one is a character-by-character
    substring of the source text. Gap markers are the only inserted content, and they
    count against ``budget_chars`` like any other character.

    When the budget runs out, only the *last* chunk of the preview is truncated, at a
    line boundary when one fits and by character count otherwise. Earlier chunks stay
    intact, so the preview never carries a silent hole. Two edge cases:

    - Not even one character is left for a chunk: the chunk is dropped and its lines are
      folded into the marker that precedes it, yielding a single merged marker.
    - The last chunk is truncated: a closing marker reports the lines left out from the
      truncation point to the end of the source.

    Args:
        chunks: The full chunk sequence, used to derive the source line count.
        selected: The chunks to render, in ascending index order, a subset of ``chunks``.
        budget_chars: Maximum length of the returned preview, markers included.

    Returns:
        The assembled preview, at most ``budget_chars`` characters long. Empty only when
        there is nothing to render or the budget cannot hold a single character.
    """
    if not chunks or not selected or budget_chars < 1:
        return ""

    total_lines = chunks[-1].end_line
    parts: list[str] = []
    used = 0
    previous: Chunk | None = None

    for chunk in selected:
        gap_lines = 0
        if previous is not None and chunk.index > previous.index + 1:
            gap_lines = chunk.start_line - previous.end_line - 1

        marker = _format_gap_marker(gap_lines) if gap_lines > 0 else ""
        if marker and used + len(marker) <= budget_chars:
            parts.append(marker)
            used += len(marker)
        else:
            marker = ""  # Did not fit, so nothing was appended.

        remaining = budget_chars - used

        if len(chunk.text) <= remaining:
            parts.append(chunk.text)  # Verbatim.
            used += len(chunk.text)
            previous = chunk
            continue

        # The budget dies on this chunk, so it is the last one of the preview.
        if remaining < 1:
            # Fold the dropped chunk and its preceding marker into a single marker.
            if marker:
                parts.pop()
                used -= len(marker)
            merged = _format_gap_marker(gap_lines + total_lines - chunk.start_line + 1)
            if used + len(merged) <= budget_chars:
                parts.append(merged)
            break

        kept = _truncate_at_line_boundary(chunk.text, remaining)
        tail = _trailing_gap_marker(chunk, kept, total_lines)
        if tail and used + len(kept) + len(tail) > budget_chars:
            # Make room for the closing marker; it counts inside the budget.
            shrunk = _truncate_at_line_boundary(chunk.text, remaining - len(tail))
            if shrunk:
                kept = shrunk
                tail = _trailing_gap_marker(chunk, kept, total_lines)
            if used + len(kept) + len(tail) > budget_chars:
                tail = ""  # Source content wins over the marker.
        parts.append(kept)
        if tail:
            parts.append(tail)
        break

    return "".join(parts)


class RelevancePreview:
    """Build a preview that keeps the parts of a tool result the query asks about.

    Composes the four steps of the filtering pipeline: chunking, scoring, selection and
    assembly. Only the scoring step reaches outside this module, through the injected
    :class:`~context_core.relevance.reranker.Reranker`.

    Args:
        reranker: Scorer used to rank chunks against the query.
        relevance_threshold: Minimum score, in ``[0.0, 1.0]``, for a chunk to be
            eligible for the preview.
        chunk_tokens: Approximate token budget per chunk, the scoring granularity.
        preview_tokens: Approximate token budget of the whole preview.
        summarize_overflow: Reserved for the opt-in summarization of a truncated
            preview. Accepted and stored, not yet acted upon.

    Attributes:
        search_units: Number of scoring batches submitted since construction, the
            billable unit of the reranker. Monotonic, never reset: the instance lives
            as long as the binding that owns it, so this is the session total.
    """

    def __init__(
        self,
        reranker: Reranker,
        *,
        relevance_threshold: float,
        chunk_tokens: int,
        preview_tokens: int,
        summarize_overflow: bool = False,
    ) -> None:
        """Initialize the preview builder. See the class docstring for the arguments."""
        self._reranker = reranker
        self._relevance_threshold = relevance_threshold
        self._chunk_tokens = chunk_tokens
        self._preview_tokens = preview_tokens
        self._summarize_overflow = summarize_overflow
        self.search_units = 0

    async def build(self, text: str, query: str) -> str:
        """Return a verbatim, budget-bounded preview of ``text`` for ``query``.

        Two shortcuts skip scoring entirely, because there is nothing to choose from:
        an empty text, and a text that already fits the preview budget as a single
        chunk. Both leave the reranker untouched, so no search unit is spent.

        Args:
            text: The concatenated text of the stored blocks.
            query: The scoring query. Must hold at least one non-whitespace character
                when scoring is needed.

        Returns:
            The preview, at most ``preview_tokens * 4`` characters long. Every segment
            that is not a gap marker is an exact substring of ``text``. Empty only when
            ``text`` is empty.

        Raises:
            RerankerError: Propagated from the reranker, or raised here when the
                returned score list violates the scoring contract. Either way the
                caller leaves the block unchanged through one path, and
                ``text`` is left untouched.
        """
        preview, _stats = await self.build_with_stats(text, query)
        return preview

    async def build_with_stats(self, text: str, query: str) -> tuple[str, PreviewStats]:
        """Return the preview of :meth:`build` together with what the selection did.

        The statistics are what the binding's disclaimer reports to the model -- how much of the
        result it is looking at -- and the chunk ranking is what a retrieval tool reuses to hand
        back more chunks in relevance order without scoring the text a second time.

        Args:
            text: The concatenated text of the stored blocks.
            query: The scoring query, as for :meth:`build`.

        Returns:
            The preview and its :class:`PreviewStats`.

        Raises:
            RerankerError: As for :meth:`build`.
        """
        budget_chars = self._preview_tokens * _CHARS_PER_TOKEN
        chunks = _chunk_text(text, self._chunk_tokens)

        if not chunks:
            return "", PreviewStats(total_chunks=0, total_lines=0, shown=(), ranking=())

        total_lines = chunks[-1].end_line
        # The whole text already fits: scoring could only pick it, at the cost of a call.
        if len(chunks) == 1 and len(text) <= budget_chars:
            return text, PreviewStats(total_chunks=1, total_lines=total_lines, shown=(chunks[0],), ranking=(0,))

        # Counted before the call, not after: the batches are billed once submitted, so a
        # failure halfway through still consumed what it sent.
        self.search_units += math.ceil(len(chunks) / max(1, self._reranker.max_sources_per_query))

        raw_scores = await self._reranker.score(query, [chunk.text for chunk in chunks])
        # A contract violation must not reach the selection: it would misalign scores
        # from chunks or break the threshold comparison.
        scores = _validate_scores(raw_scores, len(chunks))
        selected = _select_chunks(chunks, scores, self._relevance_threshold, budget_chars)
        ranking = tuple(sorted(range(len(chunks)), key=lambda i: (-scores[i], i)))
        stats = PreviewStats(total_chunks=len(chunks), total_lines=total_lines, shown=tuple(selected), ranking=ranking)
        return _assemble_preview(chunks, selected, budget_chars), stats
