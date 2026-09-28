"""The lean catalog and the exchange fold — the pure half of progressive tool disclosure.

Progressive tool disclosure sends, on each model call, only the tools that are *callable* on that call;
every other registered tool reaches the model as one line of a catalog in the system prompt — its name
and a summary of its description, at most ``catalog_chars`` characters long. In the messages that call
sends, an exchange of a tool the call does not carry is folded to one sentence — ``The tool X was
called and the result was: Y`` — so the model keeps the evidence without a call shape to repeat.

This module holds the decision logic of both halves, over the neutral message/tool shape
(:mod:`context_core.message` and :class:`context_core.disclosure.tool_index.ToolSpec`). What stays in a
framework layer is everything that touches a framework object: registering the two tools, rewriting the
request, holding per-agent TTL state, and calling a *model* to write a summary. A model-written summary
is injected here through :meth:`SummaryCache.prime`, so the cache — and therefore the catalog — stays
byte-stable across calls whether the summary came from a model or from the truncation fallback.

Budget
------
``catalog_chars`` is counted in CHARACTERS. A description that already fits is used verbatim and costs
nothing; a longer one is cut at a sentence boundary, else at a word boundary with an ellipse, else at the
character limit. Truncation is the FALLBACK for a summary that could not be produced, and the clamp
applied to one that overran — never a rewrite of the text.

No framework import, no I/O, no model call anywhere in this module.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Container, Iterable, Mapping, Sequence
from typing import Any

from ..message import NeutralBlock, NeutralMessage
from .tool_index import ToolSpec

__all__ = [
    "CATALOG_PROMPT_HEADER",
    "DEFAULT_CATALOG_CHARS",
    "FIND_TOOLS_NAME",
    "GET_TOOL_DETAILS_NAME",
    "PLUGIN_TOOL_NAMES",
    "SummaryCache",
    "build_catalog",
    "catalog_prompt_block",
    "clamp_summary",
    "estimate_tokens",
    "fold_closed_exchanges",
    "fold_note",
    "pairs_intact",
    "summary_line",
    "truncate_description",
]

logger = logging.getLogger(__name__)

FIND_TOOLS_NAME = "find_tools"
"""Name of the search tool. Also the name a projection looks for to decide it can project at all."""

GET_TOOL_DETAILS_NAME = "get_tool_details"
"""Name of the loading tool: the one call that puts full specifications into the next projection.

Not ``get_details``: a bare verb-noun that generic is a name a domain tool can already hold, and a
collision would silently shadow one of the two in the registry."""

PLUGIN_TOOL_NAMES = frozenset({FIND_TOOLS_NAME, GET_TOOL_DETAILS_NAME})
"""Both disclosure tools. Carried in full on every projected call, and folded away with no sentence:
they matter on the call right after them and are dead weight past it."""

DEFAULT_CATALOG_CHARS = 80
"""Summary limit per catalog line, in characters. About twenty tokens, the budget that measured ~96% of
the resident saving at a low risk."""

_CHARS_PER_TOKEN = 4
"""Approximate characters per token — the same heuristic the relevance preview slicing uses."""

_ELLIPSIS = "..."
"""Marks a description as cut. Counts against the budget like any other character."""

_SENTENCE_ENDINGS = ".!?"
"""Characters that end a sentence when followed by whitespace or by the end of the text."""

CATALOG_PROMPT_HEADER = """\
# Tools available on request

The tools listed below are NOT in your tool list, and you MUST NOT call them directly: their parameters
are not loaded, and a direct call is rejected without running.

To use any of them, always follow these steps:
1. Call `{get_tool_details}` with the names you need, as a list, in one call.
2. On your next call they are in your tool list with their full parameters. Call them from there.
3. A tool left unused for a few calls is unloaded again. If a call to it is rejected, repeat step 1.

If no name below fits what you need, call `{find_tools}` with the need in your own words, then go to
step 1 with the names it returns.

The tools that ARE in your tool list for this call you call directly.

"""
"""Preamble of the system-prompt catalog: the rule, stated where the model reads the names.

The names are not in the call's tool list at all, so nothing asserts they are callable, and the rule that
governs them arrives in the same block. The common path is catalog -> ``get_tool_details`` -> call;
``find_tools`` is the fallback for a need the model cannot map to a listed name.
"""


def estimate_tokens(text: str) -> int:
    """Estimate the token count of ``text`` from its character count.

    Rounds up, so a non-empty text never estimates as zero tokens.

    Args:
        text: Text to estimate.

    Returns:
        Estimated token count.
    """
    return math.ceil(len(text) / _CHARS_PER_TOKEN)


# --------------------------------------------------------------------------------------------------
# The budgeted line: verbatim when it fits, else a boundary cut.
# --------------------------------------------------------------------------------------------------


def truncate_description(text: str, max_chars: int) -> str:
    """Cut ``text`` to at most ``max_chars`` characters, preferring a sentence/word boundary.

    The FALLBACK of a catalog line, used only when a summary could not be produced, and the clamp applied
    to a summary that overran. The cut is attempted at the last sentence boundary that fits, then at the
    last word boundary, and only by character count when no word boundary fits — the case of a single
    word longer than the whole limit, where any boundary-based cut would return nothing.

    A cut at a sentence boundary reads as a whole and carries no ellipse; a cut mid-sentence gets
    one, and the ellipse counts against the limit like any other character.

    Args:
        text: Text to cut.
        max_chars: Character limit. Must be at least ``1``.

    Returns:
        A prefix of ``text``, possibly followed by an ellipse, at most ``max_chars`` characters long.
        Returns ``text`` unchanged when it already fits, which covers the empty text.
    """
    if len(text) <= max_chars:
        return text

    # A sentence boundary needs no ellipse to read as a whole, so it gets the full budget.
    cut = _last_sentence_end(text, max_chars)
    if cut > 0:
        return text[:cut]

    budget = max_chars - len(_ELLIPSIS)
    if budget > 0:
        cut = _last_word_end(text, budget)
        if cut > 0:
            return text[:cut].rstrip() + _ELLIPSIS

    # No sentence and no word boundary fits the budget; cut by character count at the limit.
    return text[:max_chars]


def _last_sentence_end(text: str, budget: int) -> int:
    """Find the end of the last sentence of ``text`` that fits ``budget`` characters.

    A sentence ends at a terminator followed by whitespace or by the end of ``text``, so that a
    period inside ``v1.2`` or ``e.g.`` is not mistaken for one.

    Args:
        text: Text to scan.
        budget: Maximum number of characters the result may span.

    Returns:
        Number of characters to keep, terminator included, or ``-1`` when no sentence ends within
        the budget.
    """
    for i in range(min(budget, len(text)) - 1, -1, -1):
        if text[i] in _SENTENCE_ENDINGS and (i + 1 >= len(text) or text[i + 1].isspace()):
            return i + 1
    return -1


def _last_word_end(text: str, budget: int) -> int:
    """Find the end of the last whole word of ``text`` that fits ``budget`` characters.

    Args:
        text: Text to scan.
        budget: Maximum number of characters the result may span.

    Returns:
        Number of characters to keep, trailing whitespace excluded from the word itself, or ``-1``
        when no word ends within the budget.
    """
    # A word may end exactly at the budget: the character just past it decides, not the budget.
    for i in range(min(budget, len(text) - 1), -1, -1):
        if text[i].isspace():
            return i
    return -1


def clamp_summary(text: object, max_chars: int) -> str:
    """Normalize a summarizer's answer to one line of at most ``max_chars`` characters.

    Whitespace is collapsed and wrapping quotes are dropped, because a model asked for a bare line still
    answers with one sometimes. An answer over the limit is cut at a boundary rather than rejected: it is
    still a summary, and the cut is the same one the fallback would make.

    Args:
        text: What the summarizer returned. Anything that is not a string is treated as empty.
        max_chars: Character limit.

    Returns:
        The clamped line, or ``""`` when there was nothing usable.
    """
    if not isinstance(text, str):
        return ""
    line = " ".join(text.split()).strip("\"'`")
    return truncate_description(line, max_chars) if line else ""


def summary_line(spec: ToolSpec, max_chars: int) -> str:
    """Produce the catalog line of ``spec`` with no model involved: verbatim, else truncated.

    A description that already fits the limit is the best possible summary of itself and costs no call;
    a longer one falls back to the boundary truncation. A model-written summary for the longer case is a
    framework-layer concern — it reaches the catalog through :meth:`SummaryCache.prime`.

    Args:
        spec: Tool specification. Left unmodified.
        max_chars: Character limit of the line.

    Returns:
        The line. ``""`` when the specification carries no description.
    """
    description = spec.get("description") or ""
    if len(description) <= max_chars:
        return " ".join(description.split())
    return truncate_description(description, max_chars)


class SummaryCache:
    """Catalog lines, computed once per ``(name, description)`` pair and kept.

    Keyed by ``(name, description)``: a tool re-registered with a different description gets a new line,
    and the same description is never summarized twice. That is what keeps the catalog byte-stable across
    calls, which matters because the block sits in the system prompt a provider may be caching by prefix.

    A framework layer that has a model writes the expensive lines itself and hands them over with
    :meth:`prime`; everything it does not prime falls back to :func:`summary_line`.
    """

    def __init__(self, summarizer: Callable[[ToolSpec, int], str] | None = None) -> None:
        """Create an empty cache.

        Args:
            summarizer: Optional synchronous summarizer, receiving ``(spec, max_chars)``. Its answer is
                clamped to the limit, and a summarizer that raises or answers nothing falls back to
                :func:`summary_line`, so no tool is ever left without a line. ``None`` uses the fallback
                for every tool.
        """
        self._summarizer = summarizer
        self._lines: dict[tuple[str, str], str] = {}

    @staticmethod
    def key(spec: ToolSpec) -> tuple[str, str]:
        """Return the cache key of ``spec``: its name and its description verbatim."""
        return (spec.get("name") or "", spec.get("description") or "")

    def prime(self, name: str, description: str, line: str, max_chars: int = DEFAULT_CATALOG_CHARS) -> str:
        """Store a line written elsewhere — by a model, or by an operator — for ``(name, description)``.

        Args:
            name: Tool name.
            description: The description the line summarizes, verbatim.
            line: The line to store. Normalized and clamped to ``max_chars`` like any summary.
            max_chars: Character limit to clamp to.

        Returns:
            The line as stored.
        """
        clamped = clamp_summary(line, max_chars)
        self._lines[(name, description)] = clamped
        return clamped

    def get(self, spec: ToolSpec) -> str | None:
        """Return ``spec``'s cached line, or ``None`` when it has none yet."""
        return self._lines.get(self.key(spec))

    def line(self, spec: ToolSpec, max_chars: int = DEFAULT_CATALOG_CHARS) -> str:
        """Return ``spec``'s catalog line, computing and caching it on first ask.

        Args:
            spec: Tool specification. Left unmodified.
            max_chars: Character limit of the line.

        Returns:
            The cached line.
        """
        key = self.key(spec)
        cached = self._lines.get(key)
        if cached is not None:
            return cached

        line = ""
        description = spec.get("description") or ""
        if self._summarizer is not None and len(description) > max_chars:
            try:
                line = clamp_summary(self._summarizer(spec, max_chars), max_chars)
            except Exception:
                logger.warning(
                    "tool summary failed | tool=<%s> | falling back to truncation", key[0], exc_info=True
                )
                line = ""

        self._lines[key] = line or summary_line(spec, max_chars)
        return self._lines[key]

    def lines_for(self, specs: Sequence[ToolSpec], max_chars: int = DEFAULT_CATALOG_CHARS) -> dict[str, str]:
        """Return the catalog line of each spec, by name, computing the missing ones.

        Args:
            specs: Tool specifications. Left unmodified.
            max_chars: Character limit of a line.

        Returns:
            Name to catalog line.
        """
        return {spec["name"]: self.line(spec, max_chars) for spec in specs if spec.get("name")}

    def __len__(self) -> int:
        """Number of distinct ``(name, description)`` pairs that have a line."""
        return len(self._lines)


# --------------------------------------------------------------------------------------------------
# The block: the rule, then one ``- name: summary`` line per tool not callable on this call.
# --------------------------------------------------------------------------------------------------


def catalog_prompt_block(
    tool_specs: Sequence[ToolSpec],
    active_tool_names: Container[str],
    summaries: Mapping[str, str],
) -> str:
    """Render the catalog as one system-prompt block: the rule, then ``- name: summary``.

    Every tool that is not getting a full specification on this call is listed, and NOTHING else — the
    active tool list and this block partition the registry between them, so no name appears in both and
    no name is missing from both.

    Args:
        tool_specs: Every registered specification, in arrival order. Read only.
        active_tool_names: Names that ARE carrying a full specification on this call, and so must not be
            listed here.
        summaries: Catalog line of each tool, by name. A name without one is listed by name only.

    Returns:
        The block, or ``""`` when every tool is already carrying a full specification — an empty catalog
        must add nothing to the prompt rather than a header promising a list.
    """
    lines = []
    for spec in tool_specs:
        name = spec["name"]
        if name in active_tool_names:
            continue
        summary = summaries.get(name, "")
        lines.append(f"- {name}: {summary}" if summary else f"- {name}")
    if not lines:
        return ""

    header = CATALOG_PROMPT_HEADER.format(find_tools=FIND_TOOLS_NAME, get_tool_details=GET_TOOL_DETAILS_NAME)
    return header + "\n".join(lines)


def build_catalog(
    tool_specs: Sequence[ToolSpec],
    catalog_chars: int | None = DEFAULT_CATALOG_CHARS,
    *,
    active_tool_names: Container[str] = (),
    summaries: Mapping[str, str] | None = None,
    cache: SummaryCache | None = None,
) -> str:
    """Build the catalog block for one model call.

    The one entry point a framework layer needs: it derives the line of every tool that is not callable
    on this call — from ``summaries``, else from ``cache``, else from :func:`summary_line` — and renders
    the block. ``catalog_chars=None`` suppresses the catalog entirely, which is a supported configuration:
    the two disclosure tools' own descriptions are then the only hint that other tools exist.

    Args:
        tool_specs: Every registered specification, in arrival order. Read only.
        catalog_chars: Character limit of one line's summary, or ``None`` to add no catalog at all.
        active_tool_names: Names carrying a full specification on this call. They are not listed.
        summaries: Pre-computed lines by name, taking precedence over ``cache``. A name absent here falls
            through to ``cache``.
        cache: Where a computed line is kept, so a description is summarized once per instance. A fresh
            cache is used when none is passed, which makes a single call self-contained.

    Returns:
        The block, or ``""`` when the catalog is suppressed or nothing is left to list.
    """
    if catalog_chars is None:
        return ""

    store = cache if cache is not None else SummaryCache()
    lines: dict[str, str] = {}
    for spec in tool_specs:
        name = spec.get("name") or ""
        if not name or name in active_tool_names:
            continue
        given = None if summaries is None else summaries.get(name)
        lines[name] = given if given is not None else store.line(spec, catalog_chars)

    return catalog_prompt_block(tool_specs, active_tool_names, lines)


# --------------------------------------------------------------------------------------------------
# The fold: a closed exchange of a tool this call cannot call becomes one sentence.
# --------------------------------------------------------------------------------------------------


def fold_note(name: str, result: Mapping[str, Any]) -> str:
    """Render a tool result as the one sentence that replaces its call in the history.

    Args:
        name: Name of the tool that was called.
        result: The ``toolResult`` mapping. Its text and JSON parts are rendered; other parts are kept
            apart by the caller.

    Returns:
        ``The tool X was called and the result was: Y``, or ``... and failed with: Y`` for an error.
    """
    parts: list[str] = []
    for block in result.get("content") or ():
        if "text" in block:
            parts.append(str(block["text"]))
        elif "json" in block:
            parts.append(json.dumps(block["json"], ensure_ascii=False))
    outcome = "failed with" if result.get("status") == "error" else "the result was"
    return f"The tool {name} was called and {outcome}: {' '.join(parts)}"


def _current_turn_start(messages: Sequence[NeutralMessage]) -> int:
    """Return the index of the user message that opened the turn in flight.

    That is the last user message carrying no ``toolResult``: everything from it on is the agent's
    current work, including the tool loop and its latest assistant message. Providers that return
    reasoning (``reasoningContent``) require the latest assistant message to arrive unmodified, so the
    whole turn in flight is off limits to the fold.

    Args:
        messages: Messages of the call.

    Returns:
        The index, or ``0`` when there is none.
    """
    for position in range(len(messages) - 1, -1, -1):
        message = messages[position]
        if message.get("role") == "user" and not any("toolResult" in b for b in message.get("content") or ()):
            return position
    return 0


def _without_reasoning(content: Sequence[NeutralBlock]) -> list[NeutralBlock]:
    """Drop ``reasoningContent`` blocks: a rewritten message can no longer carry a valid signature.

    Only ever applied to messages of CLOSED turns, whose reasoning providers accept being omitted.
    """
    return [block for block in content if "reasoningContent" not in block]


def _results_first(content: Sequence[NeutralBlock]) -> list[NeutralBlock]:
    """Move ``toolResult`` blocks to the front, keeping order otherwise.

    A provider rejects a user message where text precedes the ``toolResult`` answering the previous
    assistant message (``tool_use ids were found without tool_result blocks immediately after``), and a
    fold sentence or a merge can put text there.
    """
    return [b for b in content if "toolResult" in b] + [b for b in content if "toolResult" not in b]


def _tidy(message: NeutralMessage, content: Sequence[NeutralBlock]) -> NeutralMessage:
    """Return ``message`` with ``content`` fixed up for its role: no stale reasoning, results first."""
    fixed = _without_reasoning(content) if message.get("role") == "assistant" else _results_first(content)
    return {**message, "content": fixed}


def pairs_intact(messages: Sequence[NeutralMessage]) -> bool:
    """Report whether every ``toolUse`` is answered by a ``toolResult`` in the very next message.

    The fold's own invariant, checked on its output: a case it did not foresee then costs the saving on
    that call, never the call itself.
    """
    for position, message in enumerate(messages[:-1]):
        if message.get("role") != "assistant":
            continue
        use_ids = {b["toolUse"].get("toolUseId") for b in message.get("content") or () if "toolUse" in b}
        if not use_ids:
            continue
        answers = messages[position + 1]
        result_ids = {b["toolResult"].get("toolUseId") for b in answers.get("content") or () if "toolResult" in b}
        if answers.get("role") != "user" or not use_ids <= result_ids:
            return False
    return True


def fold_closed_exchanges(
    messages: Sequence[NeutralMessage],
    active_tool_names: Container[str],
    *,
    drop_names: Iterable[str] = PLUGIN_TOOL_NAMES,
) -> Sequence[NeutralMessage]:
    """Return the call's messages with every exchange the model cannot repeat folded out of tool form.

    A ``toolUse`` with its arguments is a template: seen next to a success, the model repeats it, and if
    the tool is not callable on this call that repeat is a call to a tool it cannot see. So for every
    exchange of a tool NOT in ``active_tool_names``, the ``toolUse`` block goes and its ``toolResult``
    becomes a plain sentence — ``The tool X was called and the result was: Y`` — which keeps the evidence
    and drops the call shape. Image or document parts of the result are kept as they are.

    The disclosure tools' own exchanges (``drop_names``) go entirely, with no sentence: they matter on
    the call right after them and are dead weight past it.

    Only closed turns are folded. The first message and the turn in flight are returned as the very same
    objects, which is what keeps a reasoning model's latest assistant message intact. Inside the folded
    span, emptied messages are dropped and same-role neighbours are merged, so the roles keep alternating
    and every remaining ``toolResult`` still follows its ``toolUse``; an assistant message that is
    rewritten or merged loses its ``reasoningContent``. The caller's list is never mutated.

    Args:
        messages: Messages of the call. Read only.
        active_tool_names: Names carried in full in this call's tool list.
        drop_names: Names whose exchanges are removed outright rather than folded to a sentence.

    Returns:
        ``messages`` itself when nothing is folded; otherwise a new list.
    """
    dropped = frozenset(drop_names)
    boundary = _current_turn_start(messages)
    if boundary <= 1:
        return messages

    uses: dict[str, str] = {}
    results: set[str] = set()
    for message in messages[1:boundary]:
        for block in message.get("content") or ():
            if (use := block.get("toolUse")) and (use_id := use.get("toolUseId")):
                name = use.get("name") or ""
                if name in dropped or name not in active_tool_names:
                    uses[use_id] = name
            elif (result := block.get("toolResult")) and (result_id := result.get("toolUseId")):
                results.add(result_id)
    fold = {use_id: name for use_id, name in uses.items() if use_id in results}
    if not fold:
        return messages

    rebuilt: list[NeutralMessage] = [messages[0]]
    for message in messages[1:boundary]:
        content: list[NeutralBlock] = []
        changed = False
        for block in message.get("content") or ():
            use, result = block.get("toolUse"), block.get("toolResult")
            if use and use.get("toolUseId") in fold:
                changed = True
                continue
            if result and (folded_name := fold.get(result.get("toolUseId") or "")) is not None:
                changed = True
                if folded_name not in dropped:
                    content.append({"text": fold_note(folded_name, result)})
                    content.extend(b for b in result.get("content") or () if "text" not in b and "json" not in b)
                continue
            content.append(block)
        if not changed:
            rebuilt.append(message)
            continue
        if message.get("role") == "assistant":
            content = _without_reasoning(content)
        if content:
            rebuilt.append(_tidy(message, content))

    merged: list[NeutralMessage] = []
    for message in rebuilt:
        if merged and merged[-1].get("role") == message.get("role"):
            previous = merged[-1]
            merged[-1] = _tidy(previous, [*(previous.get("content") or ()), *(message.get("content") or ())])
        else:
            merged.append(message)

    # The turn in flight opens with a user message. If the folded span now ends with one too, its
    # content goes in front of the opening message instead, never into the turn's assistant messages.
    current = list(messages[boundary:])
    if merged[-1].get("role") == current[0].get("role"):
        tail = merged.pop()
        opening = _tidy(current[0], [*(tail.get("content") or ()), *(current[0].get("content") or ())])
        current = [opening, *current[1:]]
    folded = [*merged, *current]
    # The last message may be an assistant toolUse still waiting for its result, so it is not checked.
    if not pairs_intact(folded[:-1]) and pairs_intact(list(messages[:-1])):
        logger.warning("fold broke a toolUse/toolResult pair | sending the messages unfolded")
        return messages
    return folded
