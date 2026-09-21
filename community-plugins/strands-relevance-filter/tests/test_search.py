"""Tests for the pure search helpers in ``strands_relevance_filter.search``.

Both ``_is_searchable_content`` and ``_search_content`` are network-free.
"""

from __future__ import annotations

import pytest

from strands_relevance_filter.search import _is_searchable_content, _search_content


# --------------------------------------------------------------------------- #
# _is_searchable_content
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "content_type",
    [
        "text/plain",
        "text/markdown",
        "text/html",
        "application/json",
        "application/xml",
        "application/yaml",
    ],
)
def test_is_searchable_content_searchable(content_type: str):
    assert _is_searchable_content(content_type) is True


@pytest.mark.parametrize(
    "content_type",
    [
        "image/png",
        "image/jpeg",
        "application/octet-stream",
    ],
)
def test_is_searchable_content_not_searchable(content_type: str):
    assert _is_searchable_content(content_type) is False


# --------------------------------------------------------------------------- #
# _search_content — line range
# --------------------------------------------------------------------------- #
_SAMPLE = "alpha\nbravo\ncharlie\ndelta\necho"


def test_search_content_line_range_returns_inclusive_lines_with_numbers():
    out = _search_content(_SAMPLE, line_range=(2, 4))
    # 1-indexed inclusive: lines 2..4 = bravo, charlie, delta.
    assert "bravo" in out
    assert "charlie" in out
    assert "delta" in out
    assert "alpha" not in out
    assert "echo" not in out
    # Line numbers are present.
    assert "2|" in out
    assert "3|" in out
    assert "4|" in out


def test_search_content_line_range_outside_content_raises():
    with pytest.raises(ValueError):
        _search_content(_SAMPLE, line_range=(10, 20))
    with pytest.raises(ValueError):
        _search_content(_SAMPLE, line_range=(0, 2))
    with pytest.raises(ValueError):
        _search_content(_SAMPLE, line_range=(3, 1))


# --------------------------------------------------------------------------- #
# _search_content — pattern
# --------------------------------------------------------------------------- #
def test_search_content_pattern_returns_matches_with_context():
    text = "\n".join(f"line{i}" for i in range(1, 11))  # line1..line10
    out = _search_content(text, pattern="line5", context_lines=1)
    assert "line5" in out
    # context_lines=1 -> the neighbours are shown.
    assert "line4" in out
    assert "line6" in out
    # Lines far outside the context window are not.
    assert "line1" not in out
    assert "line10" not in out


def test_search_content_regex_pattern_works():
    text = "apple\nbanana\ncherry\ndate"
    out = _search_content(text, pattern=r"^b.*a$", context_lines=0)
    assert "banana" in out
    assert "apple" not in out
    assert "cherry" not in out


def test_search_content_pattern_no_match_reported_not_raised():
    out = _search_content(_SAMPLE, pattern="zzzznotpresent")
    assert "No matches found" in out
    assert "zzzznotpresent" in out


def test_search_content_max_chars_truncates_with_message():
    text = "\n".join(f"match{i}" for i in range(1, 200))
    out = _search_content(text, pattern="match", context_lines=0, max_chars=100)
    assert "output truncated, narrow your search" in out


def test_search_content_line_range_max_chars_truncates_with_message():
    text = "\n".join(f"line{i}" for i in range(1, 200))
    out = _search_content(text, line_range=(1, 199), max_chars=100)
    assert "output truncated, narrow your range" in out


def test_search_content_empty_content_reported():
    assert "empty" in _search_content("").lower()
