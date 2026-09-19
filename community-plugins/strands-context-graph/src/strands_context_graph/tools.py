"""The three retrieval tools: what makes a wrong automatic choice cost a cycle instead of an answer.

The graph decides a Resolution per Card from a Note, and the Note is a guess. A wrong guess leaves the model not with a
worse answer but with a Title, an explicit invitation to ask. These three are what the invitation leads to:

- :func:`expand_card` asks by Title, reads the graph, and raises both axes for the rest of the turn.
- :func:`expand_artifact` asks by reference, reads the plugin's own store and then the optional Stash bridge, and raises
  nothing: the content comes back inline.
- :func:`find_context` asks by Description in the model's own words, reads the same vector index the Turn Choice scores
  against, and raises nothing.

Five properties are shared by all three. Nothing here calls a language model or touches ``agent.messages``: none holds a
handle on the model, none holds a mutable reference to the history (Requirement 12.13). Every failure is a return value,
never an exception — an unknown Title, an unknown reference, non-textual content and an absent store all come back as
prose naming what was missing (Requirements 12.3, 12.6, 12.7, 12.12, 16.8), because a raise would report the *tool*
broken rather than the *request*. Success records the fed-back Note and an error records nothing, by not reaching
:func:`~.scoring.record_reuse` on the error paths rather than by a branch inside it (Requirement 13.8); the elevation
:func:`expand_card` performs lasts the rest of the turn, while the fed-back Note carries the request into the turn
*after* it (Requirement 12.14). The retrieval cycle counter is incremented on invocation, not on success
(Requirement 17.8): a cycle spent on a request that came back empty was still spent. And no index is built here —
:func:`find_context` scores through the matcher the Turn Choice uses (Requirement 12.10), whose vectors come from the
per-process cache keyed by ``(purpose, text)``, so an unchanged Description costs nothing on this path.

The reads themselves are delegated and never reimplemented: :mod:`.store` owns the resolution order, the bound of a
targeted read and the prose of every miss, so this module decides *what* was asked and hands the answer back.
"""

from __future__ import annotations

import logging
from types import MappingProxyType
from typing import TYPE_CHECKING

from .describe import normalize
from .scoring import record_reuse, titles_in_turn_order
from .state import CardChoice, TurnChoice, _GraphState
from .store import (
    ReferenceStore,
    absent_message,
    estimate_tokens,
    non_textual_message,
    read_artifact,
    resolve_artifact,
    unknown_message,
)

if TYPE_CHECKING:
    from .matcher import SimilarityMatcher

__all__ = ["expand_artifact", "expand_card", "find_context"]

logger = logging.getLogger(__name__)

_MAX_CANDIDATES = 5
"""Candidates :func:`find_context` returns at most (Requirement 12.9). Capped rather than "as many as clear the floor":
a search that answers with the whole graph has re-injected the very thing the graph collapsed."""


# ---- expand_card ------------------------------------------------------------------------------


def expand_card(
    state: _GraphState,
    title: str,
    *,
    cycle: int,
    reuse_ttl_cycles: int,
) -> str:
    """Raise the Subject Card titled ``title`` to Full Content for the remainder of the turn.

    Both axes, dialogue *and* evidence (Requirement 12.2). The elevation rewrites the frozen choice rather than adding a
    field to the state, so it ends with the turn: the next ``BeforeInvocationEvent`` recomputes the choice from the
    graph, and the fed-back Note carries the request across that boundary (Requirement 12.14). A full pass is left as
    is, since every Card is already at Full Content and a ``by_title`` entry would flip ``full_pass`` to false and cost
    the delivery its identity short circuit.

    Args:
        state: Graph state of the agent. ``choice``, ``reuse`` and ``retrieval_cycles`` are mutated; ``cards`` read
            only.
        title: Title of the Card the model asked for, as it was shown to it.
        cycle: Current cycle counter, ``agent.event_loop_metrics.cycle_count``.
        reuse_ttl_cycles: Cycles the fed-back Note survives.

    Returns:
        Confirmation that the turn will arrive whole, or an error naming the Title asked for, in which case no
        Resolution changed and no fed-back Note was recorded (Requirement 12.3).
    """
    state.retrieval_cycles += 1

    card = state.cards.get(title)
    if card is None or card.kind != "subject":
        return (
            f"expand_card | no earlier turn of this conversation is titled '{title}' | "
            "copy a title exactly as it was shown to you, or use find_context to describe what you need"
        )

    if not state.choice.full_pass:
        state.choice = TurnChoice(
            by_title=MappingProxyType({**state.choice.by_title, title: CardChoice(dialogue="full", evidence="full")}),
            full_pass=False,
            selected=state.choice.selected,
        )

    record_reuse(state, title, cycle, reuse_ttl_cycles=reuse_ttl_cycles)

    return (
        f"expand_card | '{title}' arrives in full for the rest of this turn, its messages and its tool results together"
    )


# ---- expand_artifact --------------------------------------------------------------------------


async def expand_artifact(
    state: _GraphState,
    store: ReferenceStore,
    agent: object,
    reference: str,
    line_range: dict[str, int] | None = None,
    pattern: str | None = None,
    *,
    cycle: int,
    reuse_ttl_cycles: int,
) -> str:
    """Read the artifact behind ``reference``, whole or in part, through the store resolution order.

    The read is delegated and never reimplemented: :func:`~.store.resolve_artifact` consults the plugin's own store
    first and the optional ``ContextManager`` Stash second (Requirements 15.2, 15.3), and
    :func:`~.store.read_artifact` bounds a targeted read by ``_MAX_RESULT_TOKENS`` (Requirement 12.4). This module
    opens no file, resolves no path and builds no URI.

    No Resolution changes on any path, success included. The content asked for is in this answer, and what crosses into
    the next turn is the fed-back Note on the artifact's Card.

    Args:
        state: Graph state of the agent. Its ``reuse`` and ``retrieval_cycles`` are mutated.
        store: The plugin's own reference store for this agent. Read only.
        agent: The agent of the call, for the optional Stash bridge. Read only, and typed loosely because the tool
            reaches here from a ``ToolContext`` whose ``agent`` is typed for backwards compatibility.
        reference: The artifact reference, as it was shown to the model.
        line_range: ``{"start": int, "end": int}``, 1-indexed and inclusive, or ``None``.
        pattern: Regex or keyword to keep only matching lines, or ``None``.
        cycle: Current cycle counter, ``agent.event_loop_metrics.cycle_count``.
        reuse_ttl_cycles: Cycles the fed-back Note survives.

    Returns:
        The requested part of the artifact, or an error naming what was missing — no storage at all, an unknown
        reference, non-textual content, or a line range outside the content — with nothing recorded and no Resolution
        changed (Requirements 12.6, 12.7).
    """
    state.retrieval_cycles += 1

    resolved = await resolve_artifact(store, agent, reference)
    if resolved.outcome == "absent":
        return absent_message(reference)
    if resolved.outcome == "unknown":
        return unknown_message(reference)
    if resolved.outcome != "text" or resolved.text is None:
        return non_textual_message(reference)

    if line_range is None and pattern is None:
        answer = _whole_artifact(reference, resolved.text)
    else:
        span = _span_of(line_range)
        if line_range is not None and span is None:
            return (
                f"expand_artifact | line_range=<{line_range!r}> is not a pair of integers | pass "
                '{"start": <int>, "end": <int>}, 1-indexed and inclusive'
            )
        try:
            answer = read_artifact(resolved.text, line_range=span, pattern=pattern)
        except ValueError as error:
            return f"expand_artifact | reference '{reference}' | {error}"

    title = _artifact_title(state, reference)
    if title is not None:
        record_reuse(state, title, cycle, reuse_ttl_cycles=reuse_ttl_cycles)

    return answer


def _whole_artifact(reference: str, text: str) -> str:
    """The whole artifact, verbatim, with the cost of having asked for it whole stated in the answer.

    The notice is part of the contract (Requirement 12.5): without it the cheapest request to write is also the most
    expensive to serve, and the model has no way to know. The text itself is untouched — no truncation, no reformatting
    — so the answer contains it character for character (Requirement 15.5).

    Args:
        reference: The reference read.
        text: The artifact's whole text.

    Returns:
        The notice followed by the content.
    """
    notice = (
        f"expand_artifact | whole artifact '{reference}' | this call re-injects the artifact's entire "
        f"token count, about {estimate_tokens(text)} tokens, and it stays in the conversation for the rest of the "
        "turn | next time pass line_range or pattern to read only the part you need"
    )
    return f"{notice}\n\n{text}"


def _span_of(line_range: dict[str, int] | None) -> tuple[int, int] | None:
    """The line range as the pair :func:`~.store.read_artifact` reads, or ``None`` when it is unusable.

    Args:
        line_range: The mapping the model supplied, or ``None``.

    Returns:
        ``(start, end)``, or ``None`` for an absent or malformed range. The caller tells the two apart, since it already
        knows whether a range was supplied.
    """
    if line_range is None:
        return None
    try:
        return (int(line_range["start"]), int(line_range["end"]))
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def _artifact_title(state: _GraphState, reference: str) -> str | None:
    """Title of the artifact Card addressing ``reference``, or ``None`` when the graph holds none.

    A reference the graph never carded is still a reference the store can read — the ``AfterToolCallEvent`` fast path
    may have missed it, or the rebuild scan may not have run — so the read succeeds with no Card for the Note to land
    on.

    Args:
        state: The graph state. Read only.
        reference: The reference that was read.

    Returns:
        The title, or ``None``.
    """
    for title in titles_in_turn_order(state):
        card = state.cards[title]
        if card.kind == "artifact" and card.reference == reference:
            return title
    return None


# ---- find_context -----------------------------------------------------------------------------


def find_context(
    state: _GraphState,
    need: str,
    tag: str | None = None,
    *,
    matcher: SimilarityMatcher,
    collapse_floor: float,
    cycle: int,
    reuse_ttl_cycles: int,
) -> str:
    """Score every candidate Card's Description against ``need`` and answer with the best five.

    Scored over the index the Turn Choice already uses and no other (Requirement 12.8): one ``score`` call against the
    matcher the plugin resolved, whose vectors come from the per-process cache keyed by ``(purpose, text)``. No index is
    built here (Requirement 12.10). ``collapse_floor`` is the bar rather than ``expand_threshold``, being the Note below
    which the graph decided a Card was not worth a Description, so a candidate clearing it is one the graph did not
    dismiss.

    Args:
        state: Graph state of the agent. ``reuse`` and ``retrieval_cycles`` are mutated; ``cards`` read only.
        need: What the model is looking for, in its own words.
        tag: Restrict candidates to Cards carrying this Tag, compared in normalized form (Requirement 12.10). ``None``
            leaves every Card a candidate.
        matcher: The similarity matcher the Turn Choice uses.
        collapse_floor: Similarity below which a candidate is not returned at all.
        cycle: Current cycle counter, ``agent.event_loop_metrics.cycle_count``.
        reuse_ttl_cycles: Cycles the fed-back Note survives.

    Returns:
        At most five candidates with their Title, Tags and Description (Requirement 12.9), or an empty result naming the
        ``need`` received, in which case no fed-back Note was recorded and no Resolution changed (Requirement 12.12).
    """
    state.retrieval_cycles += 1

    if not need.strip():
        # An empty need scores every Description against nothing, which is a ranking of noise rather than an answer.
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

    return _render_candidates(state, need, chosen)


def _similarities(
    state: _GraphState,
    titles: tuple[str, ...],
    need: str,
    matcher: SimilarityMatcher,
) -> dict[str, float] | None:
    """One similarity per candidate, or ``None`` when the matcher was unusable.

    Same failure rule as :func:`~.scoring._score`: the matcher is contractually non-raising, so anything it does raise
    reads here as "no candidate", never as an exception the model must interpret.

    Args:
        state: The graph state. Read only.
        titles: Candidate titles, already in fixed turn order.
        need: The text to score against.
        matcher: The similarity matcher. Invoked at most once.

    Returns:
        Title to similarity, or ``None``.
    """
    if not titles:
        return None

    descriptions = tuple(state.cards[title].description for title in titles)
    try:
        scores = matcher.score(need, descriptions)
        if len(scores) != len(descriptions):
            # Covers the empty answer too: with at least one candidate, empty is a length mismatch.
            raise ValueError(f"similarity count=<{len(scores)}> | expected=<{len(descriptions)}>")
        return {title: float(scores[index]) for index, title in enumerate(titles)}
    except Exception:
        logger.debug("find_context similarity unavailable for %d candidate(s)", len(descriptions), exc_info=True)
        return None


def _nothing_found(need: str, tag: str | None) -> str:
    """The empty result, naming the ``need`` received and the Tag it was narrowed by.

    Naming both lets the model tell "nothing in this conversation is about that" from "nothing carrying that tag is
    about that", and only the second has an obvious next move.

    Args:
        need: The need as received.
        tag: The tag as received, or ``None``.

    Returns:
        The message.
    """
    narrowed = f", among the turns tagged '{tag}'" if tag is not None else ""
    return (
        f"find_context | nothing in this conversation matches '{need}'{narrowed} | "
        "the titles already in front of you are the whole conversation, so what you need was "
        "either never discussed or is in a turn you can name directly with expand_card"
    )


def _render_candidates(state: _GraphState, need: str, chosen: list[str]) -> str:
    """Render the chosen candidates: Title, Tags and Description each (Requirement 12.9).

    The Description is rendered in full rather than trimmed, being already bounded by ``description_tokens`` at
    derivation.

    Args:
        state: The graph state. Read only.
        need: The need as received, quoted back so the answer stands on its own.
        chosen: Titles to render, already ordered and already capped.

    Returns:
        The rendered answer.
    """
    lines = [f"find_context | {len(chosen)} earlier turn(s) match '{need}', best first:"]
    for title in chosen:
        card = state.cards[title]
        lines.append(f"- title: {title}")
        if card.tags:
            lines.append(f"  tags: {', '.join(card.tags)}")
        for fragment in card.description.splitlines():
            if fragment.strip():
                lines.append(f"  {fragment}")
    lines.append("call expand_card with one of these titles to bring that turn back in full")
    return "\n".join(lines)
