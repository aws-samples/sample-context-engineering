"""The two Card retrieval tools of practice D, as host-neutral functions over the graph state.

``expand_card`` asks by Title and ``find_context`` asks by Description. Both are what a wrong automatic choice leads to:
the graph decides a Resolution per Card from a Note, the Note is a guess, and a wrong guess leaves the model with a Title
and an explicit invitation to ask.

These are the bodies the Strands plugin keeps in ``tools.py``, reproduced here without any framework import so a host
binding that has no Stash and no artifact store (a coding-agent host whose tool results carry no references) can offer
the same two tools over the same state. ``expand_artifact`` is deliberately absent: it resolves only references some
offloader minted, and a host without one has nothing for it to read.

Shared properties, all inherited from the reference implementation:

- Nothing here calls a language model or touches the message history. The history is the host's.
- Every failure is a return value, never an exception: an unknown Title, an empty need and a spent budget all come back as
  prose naming what was missing, because a raise would report the *tool* broken rather than the *request*.
- Success records the fed-back Note and an error records nothing, by not reaching :func:`~.scoring.record_reuse`.
- The retrieval cycle counter is incremented on invocation, not on success: a cycle spent on a request that came back
  empty was still spent.
- ``find_context`` scores through the matcher the Turn Choice uses, whose vectors come from the per-process cache keyed
  by ``(purpose, text)``, so an unchanged Description costs nothing on this path.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Container, Sequence
from types import MappingProxyType

from .describe import normalize
from .matcher import SimilarityMatcher
from .scoring import record_reuse, titles_in_turn_order
from .state import CardChoice, TurnChoice, _GraphState

__all__ = [
    "EXPAND_CARD_DESCRIPTION",
    "EXPAND_CARD_RESULT_DESCRIPTION",
    "FIND_CONTEXT_DESCRIPTION",
    "expand_card",
    "find_context",
    "recall_card",
]

logger = logging.getLogger(__name__)

EXPAND_CARD_DESCRIPTION = (
    "Bring back the full content of one or more earlier turns, by their titles. Earlier turns may reach you as a "
    "title and a short description instead of their messages. When a description is not enough to answer, call this "
    "with the titles exactly as they were shown and those turns arrive in full for the rest of this turn. Ask for "
    "every turn you need in ONE call: a list of titles costs one retrieval where the same titles one at a time cost "
    "one each, and each extra call grows the conversation you are about to reason over."
)
"""Model-facing description of ``expand_card``; the text the Strands plugin shows its model."""

EXPAND_CARD_RESULT_DESCRIPTION = (
    "Bring back the full content of one or more earlier turns, by their titles. Earlier turns may reach you as a "
    "title and a short description instead of their messages. When a description is not enough to answer, call this "
    "with the titles exactly as they were shown and the messages of those turns arrive in the result of this call. "
    "Ask for every turn you need in ONE call: a list of titles costs one retrieval where the same titles one at a "
    "time cost one each, and each extra call grows the conversation you are about to reason over."
)
"""Model-facing description of ``expand_card`` for a host that delivers a recalled turn as the tool result (the in-place
mode of the OpenCode binding) instead of putting it back where it was. Same text as :data:`EXPAND_CARD_DESCRIPTION` up to
the sentence that says where the content arrives, which is the one thing that differs."""

FIND_CONTEXT_DESCRIPTION = (
    "Find earlier turns of this conversation that match what you need, described in your words. Use this when you "
    "suspect the conversation already covered something but you cannot see it in what reached you. Describe the need, "
    "not a title."
)
"""Model-facing description of ``find_context``; the text the Strands plugin shows its model."""

_EXHAUSTED = (
    "{tool} | this turn has already spent its {spent} retrieval calls | no further recovery is available on this turn: "
    "answer from what the summary and the messages already give you, and state plainly which part you could not verify"
)
"""Refusal once a turn hits ``max_retrieval_cycles``: worded as an instruction, because the failure mode it stops is a
model that keeps asking."""

_MAX_CANDIDATES = 5
"""Candidates :func:`find_context` returns at most. Capped rather than "as many as clear the floor": a search that answers
with the whole graph has re-injected the very thing the graph collapsed."""


def _exhausted(tool: str, state: _GraphState, max_retrieval_cycles: int | None) -> str | None:
    """Return the refusal when this turn has no retrieval budget left, or ``None`` while it has.

    Args:
        tool: Name of the calling tool, for the message.
        state: Graph state. Read only here.
        max_retrieval_cycles: The ceiling, or ``None`` for unbounded retrieval.

    Returns:
        The refusal text, or ``None`` when the call may proceed.
    """
    if max_retrieval_cycles is None or state.retrieval_cycles < max_retrieval_cycles:
        return None
    return _EXHAUSTED.format(tool=tool, spent=state.retrieval_cycles)


def _quoted(titles: Sequence[str]) -> str:
    """Render titles for a message, quoted and comma separated."""
    return ", ".join(f"'{title}'" for title in titles)


def expand_card(
    state: _GraphState,
    titles: Sequence[str] | str,
    *,
    cycle: int,
    reuse_ttl_cycles: int,
    max_retrieval_cycles: int | None = None,
) -> str:
    """Raise every Subject Card in ``titles`` to Full Content, and feed the request back into the next turn's Note.

    Both axes, dialogue and evidence. The elevation rewrites the frozen choice rather than adding a field to the state;
    the fed-back Note carries the request across the next projection. A full pass is left as it is, since every Card is
    already at Full Content and a ``by_title`` entry would flip ``full_pass`` to false.

    Args:
        state: Graph state. ``choice``, ``reuse`` and ``retrieval_cycles`` are mutated; ``cards`` is read only.
        titles: Titles of the Cards the model asked for, as they were shown to it. A bare string is accepted as one title.
        cycle: Current cycle counter (the projection ordinal).
        reuse_ttl_cycles: Cycles the fed-back Note survives.
        max_retrieval_cycles: Retrieval calls this turn may spend, or ``None`` for unbounded.

    Returns:
        Confirmation naming the turns that will arrive whole, an error naming the Titles that matched nothing, or the
        refusal when the turn's retrieval budget is spent.
    """
    refusal = _exhausted("expand_card", state, max_retrieval_cycles)
    if refusal is not None:
        return refusal

    state.retrieval_cycles += 1

    wanted = [titles] if isinstance(titles, str) else list(titles)
    if not wanted:
        return "expand_card | no title given | pass the titles you need, copied exactly as they were shown to you"

    found: list[str] = []
    missing: list[str] = []
    for title in wanted:
        card = state.cards.get(title)
        if card is None or card.kind != "subject":
            missing.append(title)
        else:
            found.append(title)

    if found and not state.choice.full_pass:
        state.choice = TurnChoice(
            by_title=MappingProxyType(
                {**state.choice.by_title, **{title: CardChoice(dialogue="full", evidence="full") for title in found}}
            ),
            full_pass=False,
            selected=state.choice.selected,
        )

    for title in found:
        record_reuse(state, title, cycle, reuse_ttl_cycles=reuse_ttl_cycles)

    if not found:
        return (
            f"expand_card | no earlier turn of this conversation is titled {_quoted(missing)} | "
            "copy a title exactly as it was shown to you, or use find_context to describe what you need"
        )

    confirmation = (
        f"expand_card | {_quoted(found)} arrives in full for the rest of this turn, its messages and its tool results "
        "together"
    )
    if missing:
        confirmation += f" | no turn is titled {_quoted(missing)}, so nothing was raised for it"
    return confirmation


_TITLE_QUOTES = "'\"`\u2018\u2019\u201c\u201d"
"""Quote marks a model puts around a title it copies from a summary, which are not part of the title."""


def _resolve_title(state: _GraphState, raw: str) -> str | None:
    """The Title of the subject Card ``raw`` names, or ``None``. Exact first, then without surrounding quotes and blanks."""
    for candidate in (raw, raw.strip().strip(_TITLE_QUOTES).strip()):
        card = state.cards.get(candidate)
        if card is not None and card.kind == "subject":
            return candidate
    return None


_MAX_RESULT_LINES = 1_500
"""Lines one recall may return in all. OpenCode saves a tool result of more than 2,000 lines or 50 KB to a file and shows the
model a preview, and a preview is not a recall: the cap keeps the answer below both limits."""


def _fit(text: str, share: int, max_lines: int) -> str:
    """``text`` within ``share`` characters and ``max_lines`` lines, cut at a line end, and saying that it was cut."""
    lines = text.split("\n")
    if len(text) <= share and len(lines) <= max_lines:
        return text
    kept: list[str] = []
    used = 0
    for line in lines[:max_lines]:
        if kept and used + len(line) + 1 > share:
            break
        kept.append(line)
        used += len(line) + 1
    shown = "\n".join(kept)[:share]
    return (
        f"{shown}\n[... {len(text) - len(shown)} more characters of this turn were left out to keep the result small; "
        "ask for fewer turns in one call to get more of each]"
    )


def recall_card(
    state: _GraphState,
    titles: Sequence[str] | str,
    *,
    collapsed: Collection[str],
    content_of: Callable[[str], str | None],
    max_chars: int,
    max_retrieval_cycles: int | None = None,
) -> str:
    """Return the content of the named turns as the result of the call: the recall of a host that collapses turns in place.

    :func:`expand_card` raises a Card for the rest of the turn and lets the next request put the messages back where they
    were. A host whose provider caches the request prefix cannot afford that: putting an old turn back changes the request
    early, and everything after it is written to the cache again. This function therefore changes nothing -- not the frozen
    choice, not the Note, not the request -- and answers with the content itself, which lands at the end of the request
    where it costs what any tool result costs.

    The retrieval budget is the same one ``expand_card`` spends: per user turn, shared with ``find_context``, counted on
    invocation.

    Args:
        state: Graph state. ``retrieval_cycles`` is mutated; ``cards`` is read only.
        titles: Titles as the summaries showed them. A bare string is accepted as one title.
        collapsed: Titles whose messages are currently replaced by a summary in the request. A Card outside it has its
            messages in the conversation already, and recalling them would only repeat them.
        content_of: Reads one collapsed turn's content by Title, or ``None`` when it is no longer available.
        max_chars: Characters the whole result may carry, shared equally by the turns asked for.
        max_retrieval_cycles: Retrieval calls this turn may spend, or ``None`` for unbounded.

    Returns:
        The content of the turns that were collapsed, with a note for any title that matched nothing or was not collapsed;
        or the refusal when the turn's budget is spent.
    """
    refusal = _exhausted("expand_card", state, max_retrieval_cycles)
    if refusal is not None:
        return refusal

    state.retrieval_cycles += 1

    wanted = [titles] if isinstance(titles, str) else list(titles)
    if not wanted:
        return "expand_card | no title given | pass the titles you need, copied exactly as they were shown to you"

    found: list[str] = []
    whole: list[str] = []
    missing: list[str] = []
    for raw in wanted:
        title = _resolve_title(state, raw)
        if title is None:
            missing.append(raw)
        elif title not in collapsed:
            if title not in whole:
                whole.append(title)
        elif title not in found:
            found.append(title)

    sections: list[tuple[str, str]] = []
    for title in found:
        content = content_of(title)
        if content is None or not content.strip():
            missing.append(title)
        else:
            sections.append((title, content))
    found = [title for title, _ in sections]

    if not sections:
        if missing:
            return (
                f"expand_card | no earlier turn of this conversation is titled {_quoted(missing)} | "
                "copy a title exactly as it was shown to you, or use find_context to describe what you need"
            )
        return (
            f"expand_card | {_quoted(whole)} {_is_are(whole)} still in the conversation above in full, so there is "
            "nothing to bring back | read it where it is"
        )

    share = max(1, max_chars // len(sections))
    line_share = max(40, _MAX_RESULT_LINES // len(sections))
    lines = [f"expand_card | {_quoted(found)} below, in full: the messages that its summary stood for"]
    for title, content in sections:
        lines.append(f"\n=== {title} ===\n{_fit(content, share, line_share)}")
    if whole:
        lines.append(
            f"\n{_quoted(whole)} {_is_are(whole)} still in the conversation above in full, so nothing was added for it"
        )
    if missing:
        lines.append(f"\nno turn is titled {_quoted(missing)}, so nothing was returned for it")
    return "\n".join(lines)


def _is_are(titles: Sequence[str]) -> str:
    """``is`` for one title and ``are`` for several, for a sentence the model reads."""
    return "is" if len(titles) == 1 else "are"


def find_context(
    state: _GraphState,
    need: str,
    tag: str | None = None,
    *,
    matcher: SimilarityMatcher | None,
    collapse_floor: float,
    cycle: int,
    reuse_ttl_cycles: int,
    max_retrieval_cycles: int | None = None,
    neighbors_per_candidate: int = 0,
) -> str:
    """Score every candidate Card's Description against ``need`` and answer with the best five.

    ``collapse_floor`` is the bar rather than ``expand_threshold``: it is the Note below which the graph decided a Card was
    not worth a Description, so a candidate clearing it is one the graph did not dismiss.

    Args:
        state: Graph state. ``reuse`` and ``retrieval_cycles`` are mutated; ``cards`` is read only.
        need: What the model is looking for, in its own words.
        tag: Restrict candidates to Cards carrying this Tag, compared in normalized form.
        matcher: The similarity matcher the Turn Choice uses, or ``None`` when scoring is unavailable (which answers as
            "nothing found" rather than raising).
        collapse_floor: Similarity below which a candidate is not returned at all.
        cycle: Current cycle counter (the projection ordinal).
        reuse_ttl_cycles: Cycles the fed-back Note survives.
        max_retrieval_cycles: Retrieval calls this turn may spend, or ``None`` for unbounded.
        neighbors_per_candidate: Most ``similar`` neighbours to list under each candidate; ``0`` lists none.

    Returns:
        At most five candidates with their Title, Tags and Description, or an empty result naming the ``need`` received,
        in which case no fed-back Note was recorded and no Resolution changed.
    """
    refusal = _exhausted("find_context", state, max_retrieval_cycles)
    if refusal is not None:
        return refusal

    state.retrieval_cycles += 1

    if not need.strip():
        return _nothing_found(need, tag)

    titles = titles_in_turn_order(state)
    if tag is not None:
        wanted = normalize(tag)
        titles = tuple(title for title in titles if wanted and wanted in state.cards[title].tags)

    similarities = _similarities(state, titles, need, matcher)
    if similarities is None:
        return _nothing_found(need, tag)

    passing = [title for title in titles if similarities[title] >= collapse_floor]
    passing.sort(key=lambda title: (-similarities[title], state.cards[title].turn, title))
    chosen = passing[:_MAX_CANDIDATES]
    if not chosen:
        return _nothing_found(need, tag)

    for title in chosen:
        record_reuse(state, title, cycle, reuse_ttl_cycles=reuse_ttl_cycles)

    return _render_candidates(state, need, chosen, neighbors_per_candidate)


def _similarities(
    state: _GraphState,
    titles: tuple[str, ...],
    need: str,
    matcher: SimilarityMatcher | None,
) -> dict[str, float] | None:
    """One similarity per candidate, or ``None`` when the matcher was missing or unusable."""
    if not titles or matcher is None:
        return None

    descriptions = tuple(state.cards[title].description for title in titles)
    try:
        scores = matcher.score(need, descriptions)
        if len(scores) != len(descriptions):
            raise ValueError(f"similarity count=<{len(scores)}> | expected=<{len(descriptions)}>")
        return {title: float(scores[index]) for index, title in enumerate(titles)}
    except Exception:
        logger.debug("find_context similarity unavailable for %d candidate(s)", len(descriptions), exc_info=True)
        return None


def _nothing_found(need: str, tag: str | None) -> str:
    """The empty result, naming the ``need`` received and the Tag it was narrowed by."""
    narrowed = f", among the turns tagged '{tag}'" if tag is not None else ""
    return (
        f"find_context | nothing in this conversation matches '{need}'{narrowed} | "
        "the titles already in front of you are the whole conversation, so what you need was "
        "either never discussed or is in a turn you can name directly with expand_card"
    )


def _similar_neighbors(state: _GraphState, title: str, exclude: Container[str], limit: int) -> list[tuple[str, float]]:
    """The ``similar`` neighbours of ``title``, strongest first, excluding anything in ``exclude``."""
    if limit <= 0:
        return []

    neighbors = [
        (link.target, link.weight)
        for link in state.links.get(title, ())
        if link.kind == "similar" and link.target not in exclude and link.target in state.cards
    ]
    neighbors.sort(key=lambda pair: (-pair[1], pair[0]))
    return neighbors[:limit]


def _render_candidates(state: _GraphState, need: str, chosen: list[str], neighbors_per_candidate: int = 0) -> str:
    """Render the chosen candidates: Title, Tags and Description each."""
    lines = [f"find_context | {len(chosen)} earlier turn(s) match '{need}', best first:"]
    already = set(chosen)
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
            rendered = ", ".join(f"{neighbor} ({weight:.2f})" for neighbor, weight in neighbors)
            lines.append(f"  related turns: {rendered}")
    lines.append("call expand_card with one of these titles to bring that turn back in full")
    return "\n".join(lines)
