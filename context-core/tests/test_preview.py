"""Tests for the pure preview primitives in ``context_core.relevance.preview``.

Everything here is network-free: chunking, protected-content detection, score
validation, threshold-and-budget selection, gap-marker assembly, and the
``RelevancePreview`` orchestration with a local fake reranker.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from context_core.relevance.preview import (
    Chunk,
    RelevancePreview,
    RerankerError,
)
from context_core.relevance.preview import (
    _assemble_preview,
    _chunk_text,
    _has_protected_content,
    _select_chunks,
    _validate_scores,
)

_CHARS_PER_TOKEN = 4
_GAP_MARKER_RE = "[... "  # substring present in every gap marker


# --------------------------------------------------------------------------- #
# Local fake reranker — no shared conftest, so each test module owns its double.
# --------------------------------------------------------------------------- #
class FakeReranker:
    """Deterministic scorer for the preview pipeline.

    Returns one score per chunk from ``scores_by_chunk``, keyed by the chunk
    text; a chunk absent from the map scores ``default``. Optionally raises a
    ``RerankerError`` to exercise the propagation path. Records the queries and
    chunk batches it was called with, so a test can assert scoring was skipped.
    """

    def __init__(
        self,
        scores_by_chunk: dict[str, float] | None = None,
        *,
        default: float = 0.0,
        raise_error: bool = False,
        max_sources_per_query: int = 100,
    ) -> None:
        self._scores_by_chunk = scores_by_chunk or {}
        self._default = default
        self._raise_error = raise_error
        self.max_sources_per_query = max_sources_per_query
        self.calls: list[tuple[str, list[str]]] = []

    async def score(self, query: str, chunks: list[str]) -> list[float]:
        self.calls.append((query, list(chunks)))
        if self._raise_error:
            raise RerankerError("injected reranker failure")
        return [self._scores_by_chunk.get(chunk, self._default) for chunk in chunks]


# --------------------------------------------------------------------------- #
# _chunk_text
# --------------------------------------------------------------------------- #
def test_chunk_text_empty_returns_empty_list():
    assert _chunk_text("", chunk_tokens=10) == []


def test_chunk_text_single_short_line():
    chunks = _chunk_text("hello", chunk_tokens=10)
    assert len(chunks) == 1
    assert chunks[0].text == "hello"
    assert chunks[0].index == 0
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 1


def test_chunk_text_long_line_cut_by_character():
    # chunk_tokens=2 -> max_chars=8; a 20-char single line must be split by char.
    line = "x" * 20
    chunks = _chunk_text(line, chunk_tokens=2)
    max_chars = 2 * _CHARS_PER_TOKEN
    assert len(chunks) == math.ceil(20 / max_chars)
    assert all(len(c.text) <= max_chars for c in chunks)
    assert "".join(c.text for c in chunks) == line
    # Every fragment inherits the line's own start/end line number.
    assert all(c.start_line == 1 and c.end_line == 1 for c in chunks)


def test_chunk_text_invalid_chunk_tokens_raises():
    with pytest.raises(ValueError):
        _chunk_text("anything", chunk_tokens=0)
    with pytest.raises(ValueError):
        _chunk_text("anything", chunk_tokens=-3)


@settings(max_examples=200)
@given(
    text=st.text(max_size=400),
    chunk_tokens=st.integers(min_value=1, max_value=30),
)
def test_chunk_text_concatenation_reproduces_source_and_lines_non_decreasing(
    text: str, chunk_tokens: int
):
    chunks = _chunk_text(text, chunk_tokens=chunk_tokens)
    assert "".join(c.text for c in chunks) == text
    # Indices are contiguous 0-based.
    assert [c.index for c in chunks] == list(range(len(chunks)))
    # start_line/end_line are non-decreasing across the sequence.
    prev_start = 0
    prev_end = 0
    for c in chunks:
        assert c.start_line >= prev_start
        assert c.end_line >= prev_end
        prev_start = c.start_line
        prev_end = c.end_line


# --------------------------------------------------------------------------- #
# _has_protected_content
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text",
    [
        "the price is 3.14 dollars",       # decimal separator between digits
        "value 1,000 total",               # thousand separator between digits
        "R$ 50 total",                     # currency marker
        "cost $99 today",                  # currency symbol
        "amount €10",                      # currency symbol
        "paid USD amount",                 # currency code
        "col a | col b | col c",           # two-or-more pipes on a line
        "a\tb\tc",                         # two-or-more tabs on a line
        "order 123 shipped",               # three consecutive digits
    ],
)
def test_has_protected_content_positive(text: str):
    assert _has_protected_content(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "",
        "just plain prose with no numbers",
        "a single pipe | here only",       # only one pipe -> not tabular
        "one\ttab only",                   # only one tab -> not tabular
        "there were 12 apples",            # only two digits, no separator/currency
    ],
)
def test_has_protected_content_negative(text: str):
    assert _has_protected_content(text) is False


# --------------------------------------------------------------------------- #
# _validate_scores
# --------------------------------------------------------------------------- #
def test_validate_scores_rejects_non_sequence():
    with pytest.raises(RerankerError):
        _validate_scores("not a sequence", 1)
    with pytest.raises(RerankerError):
        _validate_scores({"a": 1}, 1)


def test_validate_scores_rejects_wrong_length():
    with pytest.raises(RerankerError):
        _validate_scores([0.1, 0.2], 3)


def test_validate_scores_rejects_bool():
    with pytest.raises(RerankerError):
        _validate_scores([True], 1)


def test_validate_scores_rejects_nan():
    with pytest.raises(RerankerError):
        _validate_scores([float("nan")], 1)


def test_validate_scores_rejects_out_of_range():
    with pytest.raises(RerankerError):
        _validate_scores([1.5], 1)
    with pytest.raises(RerankerError):
        _validate_scores([-0.1], 1)


def test_validate_scores_accepts_valid_list():
    result = _validate_scores([0.0, 0.5, 1.0], 3)
    assert result == [0.0, 0.5, 1.0]
    assert all(isinstance(s, float) for s in result)


def test_validate_scores_accepts_tuple():
    assert _validate_scores((0.0, 1.0), 2) == [0.0, 1.0]


# --------------------------------------------------------------------------- #
# _select_chunks
# --------------------------------------------------------------------------- #
def _mk_chunks(texts: list[str]) -> list[Chunk]:
    chunks = []
    line = 1
    for i, t in enumerate(texts):
        n = t.count("\n")
        chunks.append(Chunk(index=i, text=t, start_line=line, end_line=line + n))
        line += n + 1
    return chunks


def test_select_chunks_only_at_or_above_threshold_eligible():
    chunks = _mk_chunks(["aaaa", "bbbb", "cccc"])
    scores = [0.9, 0.2, 0.8]  # only 0 and 2 clear a 0.5 threshold
    selected = _select_chunks(chunks, scores, threshold=0.5, budget_chars=1000)
    assert [c.index for c in selected] == [0, 2]


def test_select_chunks_respects_budget_and_ascending_order():
    chunks = _mk_chunks(["aaaaa", "bbbbb", "ccccc"])  # 5 chars each
    scores = [0.9, 0.8, 0.7]
    # Budget only fits two of the 5-char chunks.
    selected = _select_chunks(chunks, scores, threshold=0.0, budget_chars=10)
    assert sum(len(c.text) for c in selected) <= 10
    assert [c.index for c in selected] == sorted(c.index for c in selected)


def test_select_chunks_never_empty_when_nothing_reaches_threshold():
    chunks = _mk_chunks(["aaaa", "bbbb"])
    scores = [0.1, 0.2]
    selected = _select_chunks(chunks, scores, threshold=0.99, budget_chars=1000)
    assert len(selected) == 1
    # Best-scored chunk is kept.
    assert selected[0].index == 1


def test_select_chunks_single_oversized_candidate_guard():
    chunks = _mk_chunks(["x" * 100])
    scores = [1.0]
    selected = _select_chunks(chunks, scores, threshold=0.5, budget_chars=10)
    assert len(selected) == 1
    assert selected[0].index == 0


def test_select_chunks_empty_chunks_returns_empty():
    assert _select_chunks([], [], threshold=0.5, budget_chars=100) == []


def test_select_chunks_mismatched_lengths_raises():
    chunks = _mk_chunks(["aaaa", "bbbb"])
    with pytest.raises(ValueError):
        _select_chunks(chunks, [0.5], threshold=0.0, budget_chars=100)


# --------------------------------------------------------------------------- #
# _assemble_preview
# --------------------------------------------------------------------------- #
def test_assemble_preview_segments_are_exact_substrings():
    text = "line1\nline2\nline3\nline4\nline5"
    chunks = _chunk_text(text, chunk_tokens=2)  # forces several chunks
    # Select a discontinuous subset to force a gap marker.
    selected = [chunks[0], chunks[-1]]
    out = _assemble_preview(chunks, selected, budget_chars=1000)
    # Strip markers, then verify remaining segments are substrings of the source.
    for segment in out.split("\n[... "):
        # The tail of a marker is "N lines omitted ...]\n<rest>"; recover the source
        # part by splitting off the marker close.
        pieces = segment.split("...]\n")
        candidate = pieces[-1]
        if candidate and candidate in out:
            # Each non-marker run must appear verbatim in the source.
            for line in candidate.splitlines():
                if line and not line.endswith("lines omitted "):
                    assert line in text


def test_assemble_preview_gap_marker_reports_omitted_lines():
    # Six one-line chunks, correctly numbered by the real chunker (chunk_tokens=1
    # -> max_chars=4, so each short line becomes its own chunk).
    text = "l1\nl2\nl3\nl4\nl5\nl6"
    chunks = _chunk_text(text, chunk_tokens=1)
    assert len(chunks) == 6
    selected = [chunks[0], chunks[5]]  # keep line 1 and line 6, skip lines 2..5
    out = _assemble_preview(chunks, selected, budget_chars=1000)
    assert _GAP_MARKER_RE in out
    assert "4 lines omitted" in out


def test_assemble_preview_never_exceeds_budget():
    text = "\n".join(f"line{i}" for i in range(1, 40))
    chunks = _chunk_text(text, chunk_tokens=3)
    selected = list(chunks)
    budget = 40
    out = _assemble_preview(chunks, selected, budget_chars=budget)
    assert len(out) <= budget


def test_assemble_preview_closing_marker_counts_a_partly_shown_line_as_omitted():
    """A truncation inside the FIRST line of the last chunk reports every source line as omitted.

    Not an off-by-one: the rule is that a line the reader cannot see in full counts as missing, and
    when the cut lands inside line 1 that makes line 1 itself the first unshown line. The count is a
    safe lower bound for a follow-up ``line_range`` -- it never under-reports what is still needed --
    so it can exceed the number of completely untouched lines.
    """
    # One chunk spanning the whole text, and a budget that dies part-way through its first line.
    text = "\n".join(f"line{i}" for i in range(1, 30))
    chunks = _chunk_text(text, chunk_tokens=1_000)
    assert len(chunks) == 1
    assert chunks[0].end_line == 29

    out = _assemble_preview(chunks, [chunks[0]], budget_chars=30)

    # A fragment of line 1 did reach the preview -- verbatim, as a prefix of the source -- and the
    # marker still claims all 29 lines, because that fragment is not the whole line.
    rendered = out.split("\n", 1)[0]
    assert rendered
    assert text.startswith(rendered)
    assert rendered != "line1"  # a fragment, not the complete line
    assert "29 lines omitted" in out


def test_assemble_preview_truncated_final_chunk_gets_closing_marker():
    # One big chunk spanning many lines; budget cuts inside it at a line boundary,
    # so lines after the cut are omitted and a closing marker must report them.
    text = "\n".join(f"line{i}" for i in range(1, 30))
    chunks = _chunk_text(text, chunk_tokens=1000)  # whole text in a single chunk
    assert len(chunks) == 1
    selected = list(chunks)
    budget = 30  # far below the full text, forces truncation of the only chunk
    out = _assemble_preview(chunks, selected, budget_chars=budget)
    assert len(out) <= budget
    assert "lines omitted" in out


def test_assemble_preview_empty_when_nothing_to_render():
    assert _assemble_preview([], [], budget_chars=100) == ""
    chunks = _mk_chunks(["a"])
    assert _assemble_preview(chunks, [], budget_chars=100) == ""
    assert _assemble_preview(chunks, chunks, budget_chars=0) == ""


# --------------------------------------------------------------------------- #
# RelevancePreview.build
# --------------------------------------------------------------------------- #
def _build(reranker: FakeReranker, **kwargs) -> RelevancePreview:
    defaults = dict(
        relevance_threshold=0.5,
        chunk_tokens=4,
        preview_tokens=100,
    )
    defaults.update(kwargs)
    return RelevancePreview(reranker, **defaults)


@pytest.mark.asyncio
async def test_build_empty_text_returns_empty_without_scoring():
    reranker = FakeReranker()
    preview = _build(reranker)
    result = await preview.build("", "query")
    assert result == ""
    assert reranker.calls == []
    assert preview.search_units == 0


@pytest.mark.asyncio
async def test_build_fitting_single_chunk_returned_verbatim_no_scoring():
    reranker = FakeReranker()
    # Small text, large preview budget, chunk_tokens big enough for one chunk.
    text = "short text that fits"
    preview = _build(reranker, chunk_tokens=100, preview_tokens=100)
    result = await preview.build(text, "query")
    assert result == text
    assert reranker.calls == []
    assert preview.search_units == 0


@pytest.mark.asyncio
async def test_build_full_path_selects_and_assembles():
    text = "\n".join(f"line number {i}" for i in range(1, 40))
    # Force many small chunks and a small preview budget so scoring is needed.
    reranker = FakeReranker(default=1.0)  # everything relevant
    preview = _build(reranker, chunk_tokens=3, preview_tokens=20, relevance_threshold=0.5)
    result = await preview.build(text, "query")
    assert result != ""
    assert result != text
    assert len(result) <= 20 * _CHARS_PER_TOKEN
    assert len(reranker.calls) == 1
    assert preview.search_units >= 1


@pytest.mark.asyncio
async def test_build_reranker_error_propagates():
    text = "\n".join(f"line {i}" for i in range(1, 40))
    reranker = FakeReranker(raise_error=True)
    preview = _build(reranker, chunk_tokens=3, preview_tokens=20)
    with pytest.raises(RerankerError):
        await preview.build(text, "query")
