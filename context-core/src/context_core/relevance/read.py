"""Read a stored tool result back: whole, by line span, by pattern, or by relevance rank.

A relevance preview shows an excerpt and keeps the integral original aside. When the excerpt cannot answer a
question (a maximum, a total, a count over every row) the model must be able to ask for the rest, and the
cheapest correct way is to ask for only what it needs: the rows a pattern matches, an exact line span, the
next most relevant chunks, or the whole thing under an explicit token budget. This module is that read, pure
and host-neutral: the host adapter authorizes the owner and fetches the preserved text, then calls in here.

The semantics, precedence and messages follow the reference retrieval tool of the Strands relevance filter,
so the same figures mean the same thing in every host:

* precedence is ``line_range``, then ``pattern``, then ``max_chunks``;
* a read without options returns the whole original, verbatim;
* every option-bearing read is bounded by ``max_tokens`` when given and otherwise by the filter's own size
  threshold, so reading a slice back never costs more than keeping the original would have;
* line numbers are 1-indexed and are the numbers the preview's ``[... N lines omitted ...]`` markers report.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .preview import _assemble_preview, _chunk_text
from .search import _search_content

__all__ = ["DEFAULT_CONTEXT_LINES", "ReadRequestError", "has_read_options", "read_content"]

DEFAULT_CONTEXT_LINES = 5
"""Lines of context before and after each pattern match, like ``grep -C``."""

_CHARS_PER_TOKEN = 4


class ReadRequestError(ValueError):
    """A read request the model can correct: the message says how, and never quotes the content."""


def has_read_options(
    *,
    pattern: Any = None,
    line_range: Any = None,
    context_lines: Any = None,
    max_chunks: Any = None,
    max_tokens: Any = None,
) -> bool:
    """Return whether any read option was supplied (``None`` means "not supplied")."""
    return any(option is not None for option in (pattern, line_range, context_lines, max_chunks, max_tokens))


def _positive_int(name: str, value: Any) -> int | None:
    """Validate an optional integer option that must be at least 1."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ReadRequestError(f"{name} must be an integer >= 1, got {value!r}")
    return value


def _span(line_range: Any) -> tuple[int, int]:
    """Validate a ``{"start": int, "end": int}`` mapping and return it as a tuple."""
    if not isinstance(line_range, Mapping):
        raise ReadRequestError("line_range must be an object with integer 'start' and 'end'")
    start, end = line_range.get("start"), line_range.get("end")
    for name, value in (("start", start), ("end", end)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ReadRequestError(f"line_range.{name} must be an integer, got {value!r}")
    return int(start), int(end)


def read_content(
    text: str,
    *,
    pattern: str | None = None,
    line_range: Mapping[str, Any] | None = None,
    context_lines: int | None = None,
    max_chunks: int | None = None,
    max_tokens: int | None = None,
    default_max_tokens: int,
    chunk_tokens: int,
    ranking: tuple[int, ...] | None = None,
) -> str:
    """Return the requested view of the stored ``text``.

    Args:
        text: The integral original, already authorized by the caller.
        pattern: Regex or keyword to grep for; an invalid or risky regex falls back to a literal match.
        line_range: ``{"start", "end"}``, 1-indexed and inclusive; an ``end`` past the last line is
            clamped. Takes precedence over ``pattern``.
        context_lines: Lines before and after each match (default 5). Alone, it reads the head.
        max_chunks: How many chunks to return, most relevant first, rendered in document order.
        max_tokens: Approximate size limit of the response in tokens, in every option-bearing mode.
        default_max_tokens: The bound used when ``max_tokens`` is absent (the filter's size threshold).
        chunk_tokens: The chunk size the preview used, so ``max_chunks`` splits the text the same way.
        ranking: Chunk indices by descending relevance, recorded when the preview was built; without it
            ``max_chunks`` reads in document order.

    Returns:
        The original text for a read without options, otherwise the formatted slice.

    Raises:
        ReadRequestError: A malformed option, or a ``line_range`` that starts outside the content.
    """
    max_chunks = _positive_int("max_chunks", max_chunks)
    max_tokens = _positive_int("max_tokens", max_tokens)
    if context_lines is not None and (isinstance(context_lines, bool) or not isinstance(context_lines, int)):
        raise ReadRequestError(f"context_lines must be an integer, got {context_lines!r}")
    if pattern is not None and not isinstance(pattern, str):
        raise ReadRequestError("pattern must be a string")

    if not has_read_options(
        pattern=pattern, line_range=line_range, context_lines=context_lines, max_chunks=max_chunks, max_tokens=max_tokens
    ):
        return text

    ctx_lines = context_lines if context_lines is not None else DEFAULT_CONTEXT_LINES
    max_chars = (max_tokens or default_max_tokens) * _CHARS_PER_TOKEN

    try:
        if line_range is None and pattern is None:
            if max_chunks is not None:
                return _read_chunks(text, max_chunks, max_chars, chunk_tokens, ranking)
            if context_lines is None:
                # max_tokens alone: the whole content, cut to that budget.
                return _search_content(text, line_range=(1, max(1, text.count("\n") + 1)), max_chars=max_chars)

        span: tuple[int, int] | None = None
        if line_range is not None:
            span = _span(line_range)
            # A span is an exact request, so it answers on its own: the pattern is dropped rather than
            # applied within the span, which would return matches instead of the lines asked for.
            pattern = None
        elif pattern is None:
            # context_lines alone names no match to center on, so it reads the head of the content.
            span = (1, max(1, ctx_lines))

        return _search_content(text, pattern=pattern, line_range=span, context_lines=ctx_lines, max_chars=max_chars)
    except ReadRequestError:
        raise
    except ValueError as error:
        raise ReadRequestError(str(error)) from error


def _read_chunks(
    text: str, max_chunks: int, max_chars: int, chunk_tokens: int, ranking: tuple[int, ...] | None
) -> str:
    """Render the ``max_chunks`` most relevant chunks of ``text`` in document order.

    The ranking is the one computed when the preview was built, so no second scoring call is made. A
    reference with no usable ranking is read in document order instead, and the header says so.
    """
    chunks = _chunk_text(text, chunk_tokens)
    if not chunks:
        return "Content is empty (0 lines)."

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
