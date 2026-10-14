"""Tests for the host-neutral read of a preserved tool result (:mod:`context_core.relevance.read`)."""

from __future__ import annotations

import pytest

from context_core.relevance import ReadRequestError, has_read_options, read_content

ROWS = "\n".join(f"2026-05-{(i % 28) + 1:02d},{'REFUND' if i % 7 == 0 else 'PIX'},row {i}" for i in range(1, 101))
KW = {"default_max_tokens": 4000, "chunk_tokens": 25}


def test_a_read_without_options_returns_the_original_verbatim() -> None:
    assert read_content(ROWS, **KW) == ROWS
    assert not has_read_options()
    assert has_read_options(pattern="x")


def test_line_range_returns_exactly_that_span_with_line_numbers() -> None:
    out = read_content(ROWS, line_range={"start": 10, "end": 12}, **KW)
    assert out.startswith("[Lines 10-12 of 100]")
    assert "|" in out and "row 10" in out and "row 12" in out and "row 13" not in out and "row 9\n" not in out


def test_line_range_end_past_the_content_is_clamped_but_a_start_past_it_is_an_error() -> None:
    out = read_content(ROWS, line_range={"start": 99, "end": 5000}, **KW)
    assert out.startswith("[Lines 99-100 of 100]")
    with pytest.raises(ReadRequestError, match="beyond content length"):
        read_content(ROWS, line_range={"start": 101, "end": 102}, **KW)
    with pytest.raises(ReadRequestError, match="must be >= 1"):
        read_content(ROWS, line_range={"start": 0, "end": 3}, **KW)
    with pytest.raises(ReadRequestError, match="must be <="):
        read_content(ROWS, line_range={"start": 9, "end": 3}, **KW)


def test_pattern_returns_only_matching_rows_with_context_and_a_match_header() -> None:
    out = read_content(ROWS, pattern="REFUND", context_lines=0, **KW)
    assert out.startswith("[14 matches for /REFUND/]")
    assert out.count("> ") == 14 and ",row 7\n" in out and ",row 8\n" not in out


def test_default_context_is_five_lines_around_each_match() -> None:
    out = read_content(ROWS, pattern="row 50$", **KW)
    assert "row 45" in out and "row 55" in out and "row 56" not in out and "row 44" not in out


def test_an_invalid_regex_falls_back_to_a_literal_match_and_no_match_is_reported() -> None:
    assert "[1 match for /row (1/]" in read_content("row (1\nother", pattern="row (1", context_lines=0, **KW)
    assert read_content(ROWS, pattern="NOPE", **KW).startswith("No matches found for pattern 'NOPE'")


def test_line_range_wins_over_pattern_and_pattern_wins_over_max_chunks() -> None:
    by_range = read_content(ROWS, line_range={"start": 1, "end": 2}, pattern="REFUND", **KW)
    assert by_range.startswith("[Lines 1-2 of 100]")
    by_pattern = read_content(ROWS, pattern="REFUND", max_chunks=1, context_lines=0, **KW)
    assert by_pattern.startswith("[14 matches")


def test_max_chunks_follows_the_recorded_ranking_and_renders_in_document_order() -> None:
    from context_core.relevance.preview import _chunk_text

    chunks = _chunk_text(ROWS, KW["chunk_tokens"])
    assert len(chunks) >= 3
    ranking = tuple(reversed(range(len(chunks))))  # last chunk is the most relevant
    out = read_content(ROWS, max_chunks=1, ranking=ranking, **KW)
    assert out.startswith(f"[1 of {len(chunks)} chunks (most relevant first)")
    assert chunks[-1].text.splitlines()[0] in out and chunks[0].text.splitlines()[0] not in out
    unranked = read_content(ROWS, max_chunks=1, **KW)
    assert "(document order)" in unranked and chunks[0].text.splitlines()[0] in unranked


def test_max_tokens_alone_reads_everything_cut_to_the_budget() -> None:
    out = read_content(ROWS, max_tokens=20, **KW)
    assert "output truncated" in out and len(out) < len(ROWS)
    assert "row 100" in read_content(ROWS, max_tokens=50_000, **KW)


def test_every_option_bearing_read_is_bounded_by_the_default_threshold() -> None:
    small = {"default_max_tokens": 10, "chunk_tokens": 25}
    assert "output truncated" in read_content(ROWS, pattern="PIX", **small)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"max_tokens": 0}, "max_tokens must be an integer >= 1"),
        ({"max_chunks": True}, "max_chunks must be an integer >= 1"),
        ({"max_chunks": "3"}, "max_chunks must be an integer >= 1"),
        ({"context_lines": "2"}, "context_lines must be an integer"),
        ({"pattern": 5}, "pattern must be a string"),
        ({"line_range": "1-3"}, "line_range must be an object"),
        ({"line_range": {"start": "1", "end": 3}}, "line_range.start must be an integer"),
        ({"line_range": {"start": 1}}, "line_range.end must be an integer"),
    ],
)
def test_malformed_options_are_correctable_errors_that_never_quote_content(kwargs: dict, message: str) -> None:
    with pytest.raises(ReadRequestError, match=message) as excinfo:
        read_content(ROWS, **kwargs, **KW)
    assert "REFUND" not in str(excinfo.value)


def test_context_lines_alone_reads_the_head_of_the_content() -> None:
    out = read_content(ROWS, context_lines=3, **KW)
    assert out.startswith("[Lines 1-3 of 100]") and "row 3" in out and "row 4" not in out


def test_empty_content_is_reported_not_crashed() -> None:
    assert read_content("", pattern="x", **KW) == "Content is empty (0 lines)."
