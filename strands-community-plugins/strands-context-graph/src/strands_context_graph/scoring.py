"""The Note, and the choice it decides: the two passes, the warm-up short circuit, and the distribution.

Two passes per turn. Pass 1 is everything from outside the graph — the similarity between the turn's question and each
Card's Description, plus the Fed-Back Note of an explicit model request — and enters with no factor of its own
(Requirements 8.1, 13.2). Pass 2 reads a frozen ``base`` and writes ``note``, so no write is read within the same pass:
propagation is exactly one jump (Requirement 8.3) and confluent. Titles are walked in ascending turn order, so every sum
happens in the same sequence on every run over the same state, which is what makes two runs over the same state,
question and similarities produce the same choice Card by Card (Requirement 8.14).

A tool name is not a Card and holds no Note of its own; it is the axis along which two Cards that used the same tool
reach each other in one Card-to-Card step. Summing per hub and then subtracting each Card's own contribution, so a Card
never propagates Note to itself through its own tool, keeps the cost at ``O(Cards + Links)`` rather than at degree
squared.

The ``similar`` Link is walked by no pass here. It answers whether a manual search can reach a Card, not whether a Card
is pertinent to this question, and it inherits no Note (Requirement 8.4): its weight carries a cosine of 0.5 to 0.8, so
one such edge would hand over more than the artifact edge while being the least informative channel of the four.

The warm-up short circuit runs before anything is scored, so no embedding is paid for a decision the size of the graph
already made (Requirement 8.12); the first turn, with no Card at all, falls into it for free (Requirement 8.13).
``expand_threshold == 0.0`` is the documented off switch, a fixed full pass rather than every Note clearing the
threshold.

Failing open is the whole failure policy: an exception, a timeout, an empty sequence, a length not matching the number
of Descriptions, and a non-sequence answer all end as Full Content everywhere, one debug log carrying ``exc_info``, and
no exception reaching the agent loop. Malformed answers are raised inside the guarded block so the single log carries a
real traceback.

Resolution steps down by budget, never by verdict. :func:`distribute` walks the Cards in descending Note handing out
Full Content until the body budget runs out; a Card whose Note cleared ``expand_threshold`` but no longer fits steps one
rung down, to Description, never to Title (Requirement 8.10). No Card is dropped from the returned mapping: the choice's
domain is the whole set of Cards.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType

from .matcher import SimilarityMatcher
from .state import Card, CardChoice, Resolution, TurnChoice, _GraphState

__all__ = [
    "compute_notes",
    "distribute",
    "expire_reuse",
    "full_pass_choice",
    "record_reuse",
    "titles_in_turn_order",
    "warm_up_choice",
]

logger = logging.getLogger(__name__)

_DECAY = 0.5
"""The neighbourhood decay factor, the third term of the propagation product (Requirement 8.2). One factor for every
Link kind the Note inherits along, and applied nowhere else: the first pass carries no factor. Not calibrated, not
exposed."""

_W_TOOL = 0.6
"""Weight of the tool-hub axis: two Cards that used the same tool reach each other through it."""

_W_ARTIFACT = 0.6
"""Weight of the artifact Link, from the Card that cited a reference to the artifact's Card."""

_W_PREVIOUS = 0.4
"""Weight of the previous-turn Link, from a Card to the Card of the turn before it."""

_STRUCTURAL_WEIGHTS: Mapping[str, float] = MappingProxyType(
    {
        "follows": _W_PREVIOUS,
        "artifact": _W_ARTIFACT,
    }
)
"""Factor per Link kind the Note inherits along, for the Card-to-Card edges. Private, and neither weight is calibrated.

Two kinds are absent, which is Requirement 8.2's "exclusively" spelled out. ``tool`` targets a tool name rather than a
Card, so it is no Note destination and is walked separately by :func:`_spread_over_tool_hubs`. ``similar`` targets a
Card but inherits no Note at all (Requirement 8.4): it is an edge a manual search traverses.
"""

_TOKENS_PER_MESSAGE = 250
"""Fallback estimate of the token cost of one addressed message, used when no cost table is supplied.

A Card holds addresses and never content (see ``state.py``), so it cannot measure its own parts: the text lives in
``agent.messages``. The caller holding the messages passes ``costs``; without it the count of durable identities stands
in for the size. Coarse by design: a wrong estimate spends budget and never breaks a call."""


_REUSE_BONUS = 1.0
"""The fed-back Note a retrieval tool grants, undecayed. Whole on the cycle of the grant, so it clears any admissible
``expand_threshold`` on the turn the model asked: the request the model made is not a hint to be outvoted."""


def record_reuse(state: _GraphState, title: str, cycle: int, *, reuse_ttl_cycles: int) -> None:
    """Grant ``title`` the fed-back Note, restarting its countdown at ``reuse_ttl_cycles``.

    Called only when one of the three retrieval tools completed successfully; a completion with an error records nothing
    (Requirement 13.8), by the caller not reaching here rather than by a branch inside. Granting and renewing are the
    same write, so a Card already holding a fed-back Note has its countdown restarted (Requirement 13.5).
    ``reuse_ttl_cycles == 0`` writes nothing: the fed-back Note is only ever read across turns (Requirement 13.6), and
    the elevation for the rest of the current turn is :func:`~.tools.expand_card`'s own doing, not this map's.

    Args:
        state: Graph state of the agent. Mutated in place, and the only place the fed-back Note ever lives
            (Requirement 13.7).
        title: Title of the Card the model asked for.
        cycle: Current cycle counter, ``agent.event_loop_metrics.cycle_count``.
        reuse_ttl_cycles: Cycles the fed-back Note survives. At least ``0``.
    """
    if reuse_ttl_cycles == 0:
        return

    state.reuse[title] = (_REUSE_BONUS, cycle + reuse_ttl_cycles)


def expire_reuse(state: _GraphState, cycle: int) -> None:
    """Drop every fed-back Note the cycle counter has carried past its expiry.

    Age is measured against the cycle counter and against nothing else (Requirement 13.3): no wall clock and no message
    count, so a slow provider call or a burst of messages inside one cycle never ages a Note. A Note granted at cycle
    ``C`` with ``reuse_ttl_cycles = k`` stored ``C + k``, so it stands while the counter is at most that value and is
    removed the first time the counter passes it (Requirement 13.4). Removed outright, whole until then: the bonus is
    the model's explicit request, and a request that only fades would still be in the state long after it stopped
    mattering.

    Called immediately before the choice reads the map and nowhere else, which is the per-cycle ageing without a hook of
    its own: the choice is the only place the fed-back Note is ever summed (Requirement 13.2), and the decision here
    reads the stored expiry rather than counting down, so running it twice within one cycle changes nothing.

    Args:
        state: Graph state of the agent. Mutated in place, and the only place the fed-back Note ever lives
            (Requirement 13.7).
        cycle: Current cycle counter, ``agent.event_loop_metrics.cycle_count``.
    """
    # Materialized first: the map is mutated while the decision is applied.
    for title, (_bonus, expiry_cycle) in list(state.reuse.items()):
        if cycle > expiry_cycle:
            del state.reuse[title]


def full_pass_choice() -> TurnChoice:
    """Build the short circuit: every Card keeps Full Content.

    Returns:
        A frozen, empty choice with ``full_pass`` set. The delivery handler reading it returns the received context by
        object identity, so the assembled context is identical field by field to the one produced without the plugin.
    """
    return TurnChoice(by_title=MappingProxyType({}), full_pass=True)


def warm_up_choice(
    state: _GraphState,
    *,
    expand_threshold: float,
    min_cards: int,
) -> TurnChoice | None:
    """Decide whether the whole choice is skipped, without ever reaching the matcher.

    Called before anything is scored, so no embedding is paid for a decision already settled by the size of the graph
    (Requirement 8.12). An agent's first turn holds no Card, so it lands here and presents Full Content for everything
    (Requirement 8.13).

    Args:
        state: The graph state. Read only.
        expand_threshold: Note at or above which a Card is Full Content. ``0.0`` is the off switch.
        min_cards: Below this many Cards the choice is skipped entirely.

    Returns:
        A full-pass choice when the choice must be skipped, or ``None`` when scoring should proceed.
    """
    if expand_threshold == 0.0 or len(state.cards) < min_cards:
        return full_pass_choice()
    return None


def titles_in_turn_order(state: _GraphState) -> tuple[str, ...]:
    """Order the Titles by ascending turn, with the Title itself breaking ties.

    The Descriptions reach the matcher in this order, the similarities come back aligned to it, and every later sum
    walks it unchanged, which is where determinism comes from (Requirement 8.14).

    Args:
        state: The graph state. Read only.

    Returns:
        Every Card Title, ordered by ``(turn, title)``.
    """
    return tuple(sorted(state.cards, key=lambda title: (state.cards[title].turn, title)))


def compute_notes(
    state: _GraphState,
    question: str,
    matcher: SimilarityMatcher,
) -> Mapping[str, float]:
    """Compute the Note per Card: the first pass, then the propagation on top of it.

    The caller has already short-circuited the warm-up cases, so reaching here means the graph is worth scoring. Mutates
    neither ``state`` nor the sequence of Descriptions handed to the matcher: what crosses that boundary is a fresh
    tuple, so its elements, order and size are ours to guarantee.

    Args:
        state: The graph state. Read only.
        question: The turn's question, embedded under the query purpose by the matcher.
        matcher: The similarity matcher. Invoked exactly once.

    Returns:
        One Note per Card, keyed by Title, never below the similarity that Card was scored with. Empty when the matcher
        failed or answered malformed, which the caller reads as "score nothing, send everything".
    """
    titles = titles_in_turn_order(state)
    descriptions = tuple(state.cards[title].description for title in titles)

    similarities = _score(question, descriptions, matcher)
    if similarities is None:
        return {}

    # ---- Pass 1: the similarity, then the Fed-Back Note. No factor on either. ----
    base = {title: similarities[index] for index, title in enumerate(titles)}

    for title, (bonus, _expiry_cycle) in state.reuse.items():
        # Added after the first pass and before the propagation (Requirement 13.2). The bonus is read as stored: the
        # aging of the stored value, and the expiry cycle beside it, belong to whoever writes this map.
        if title in base:
            base[title] += bonus

    # ---- Pass 2: the propagation, exactly one jump, over a frozen `base`. ----
    return _propagate(state, titles, base)


def _propagate(
    state: _GraphState,
    titles: Sequence[str],
    base: Mapping[str, float],
) -> dict[str, float]:
    """Spread the first pass over the Links, exactly one jump, and only ever adding.

    ``base`` is frozen throughout: every read comes out of it and every write goes into ``note``, so no write is read
    within the same pass. The jump is therefore single and the result confluent, so the two Link families may be walked
    in either order.

    Args:
        state: The graph state. Read only.
        titles: Every Card Title in fixed turn order.
        base: The first pass, complete. Read only.

    Returns:
        The Note per Card, with ``note[title] >= base[title]`` for every Card.
    """
    note = dict(base)
    _spread_over_card_edges(state, titles, base, note)
    _spread_over_tool_hubs(state, titles, base, note)
    return note


def _spread_over_card_edges(
    state: _GraphState,
    titles: Sequence[str],
    base: Mapping[str, float],
    note: dict[str, float],
) -> None:
    """Add ``source Note x Link weight x decay`` along the artifact and previous-turn edges, from source to target.

    A Link pointing at a Title the graph no longer holds is skipped rather than raising: Notes are computed per turn and
    Links are rebuilt by scan, so a dangling target is a stale edge, not a corrupt state.

    Args:
        state: The graph state. Read only.
        titles: Every Card Title in fixed turn order.
        base: The frozen first pass. Read only.
        note: Accumulator, mutated in place.
    """
    for title in titles:
        inherited = base[title] * _DECAY
        for link in state.links.get(title, ()):
            weight = _STRUCTURAL_WEIGHTS.get(link.kind)
            # ``None`` covers the two kinds the Note does not inherit along: ``tool``, whose target is a tool name and
            # never a Note destination, and ``similar``, which a manual search walks instead. See
            # ``_STRUCTURAL_WEIGHTS``.
            if weight is None or link.target not in note:
                continue
            note[link.target] += inherited * link.weight * weight


def _spread_over_tool_hubs(
    state: _GraphState,
    titles: Sequence[str],
    base: Mapping[str, float],
    note: dict[str, float],
) -> None:
    """Add the same product across the tool-name hubs, in ``O(Cards + Links)`` rather than in degree squared.

    Two walks instead of one nested walk: sum the frozen base per hub, then hand each Card the hub total minus its own
    contribution, which keeps a Card from propagating Note to itself through its own tool. The subtrahend is one addend
    of a sum of non-negative terms, so the difference is never negative and the propagation only adds.

    Args:
        state: The graph state. Read only.
        titles: Every Card Title in fixed turn order.
        base: The frozen first pass. Read only.
        note: Accumulator, mutated in place.
    """
    hub_total: dict[str, float] = {}
    for title in titles:
        for link in state.links.get(title, ()):
            if link.kind == "tool":
                hub_total[link.target] = hub_total.get(link.target, 0.0) + base[title]

    for title in titles:
        for link in state.links.get(title, ()):
            if link.kind == "tool":
                others = hub_total[link.target] - base[title]
                note[title] += others * link.weight * _W_TOOL * _DECAY


def distribute(
    notes: Mapping[str, float],
    state: _GraphState,
    *,
    expand_threshold: float,
    collapse_floor: float,
    body_budget: int | None,
    costs: Mapping[tuple[str, str], int] | None = None,
) -> TurnChoice:
    """Hand out the body budget in descending Note, and let the Resolution step down only by budget.

    Two axes, decided independently (Requirement 7.1), reading different things:

    - Dialogue, three rungs, decided by the Note. At or above ``expand_threshold`` it is Full Content when it fits the
      remaining budget and Description when it does not — one rung down, never Title (Requirements 8.7, 8.10). Between
      the two thresholds it is Description; below ``collapse_floor`` it is Title (Requirements 8.8, 8.9). With
      ``body_budget`` at ``None`` there is no ceiling to miss, so everything at or above the threshold is Full Content
      (Requirement 8.11).
    - Evidence, two rungs, decided by message order and never by the Note. Every pair consumed means the numbers already
      reached the assistant's own text, leaving only the numeric lines to preserve: Description. An unconsumed pair is
      work in progress and travels whole (Requirements 7.3, 7.4, 7.5). The classification is read off ``ToolPair``,
      which ``cards`` derived by message order alone, so deciding it here runs no model call and no embedding call
      (Requirement 7.6).

    A Card of the Turn in Progress is Full Content on both axes whatever its Note (Requirement 7.2). A subject Card
    normally closes at a turn boundary and so cannot be one, which makes this the guard for a write half that registered
    a Card for the turn now running.

    An artifact Card never reaches Full Content here, whatever its Note: its content is the reference store's, reached
    with ``expand_artifact``, and it addresses no message, so Full Content would be a rung with nothing on it. It is
    short-circuited before the debit, so the budget it would have spent is left for a Card that can use it.

    Args:
        notes: One Note per Card, as returned by :func:`compute_notes`. A missing Title reads as ``0.0``, so an empty
            mapping — the matcher having failed — decides every Card by the thresholds alone.
        state: The graph state. Read only.
        expand_threshold: Note at or above which the Dialogue is Full Content, budget permitting.
        collapse_floor: Note below which the Dialogue is Title only.
        body_budget: Token ceiling across the parts in Full Content, or ``None`` for no ceiling at all.
        costs: Estimated token cost per ``(title, part)``, where ``part`` is ``"dialogue"`` or ``"evidence"``. When
            omitted, the count of addressed messages stands in for the size.

    Returns:
        The turn choice, with ``full_pass`` false and ``by_title`` frozen. Its domain is the whole set of Cards: every
        Card is decided, and none is dropped from the mapping.
    """
    remaining = body_budget
    decided: dict[str, CardChoice] = {}

    for title in _titles_by_descending_note(state, notes):
        card = state.cards[title]
        value = notes.get(title, 0.0)
        is_artifact = card.kind == "artifact"
        # The turn ordinal counts closed turns, so a subject Card at or beyond it stands for the Turn in Progress. An
        # artifact Card carries the ordinal of the turn its content was offloaded in and is excluded by kind.
        in_progress = not is_artifact and card.turn >= state.turn

        # ---- Dialogue axis: three rungs, decided by the Note. ----
        dialogue: Resolution
        if in_progress or value >= expand_threshold:
            cost = _part_cost(card, "dialogue", costs)
            if is_artifact:
                dialogue = "description"
            elif remaining is None or cost <= remaining or in_progress:
                dialogue = "full"
                remaining = remaining if remaining is None else max(0, remaining - cost)
            else:
                # Budget exhausted: one rung down, never to Title.
                dialogue = "description"
        elif value >= collapse_floor:
            dialogue = "description"
        else:
            dialogue = "title"

        # ---- Evidence axis: two rungs, decided by message order, never by the Note. ----
        evidence: Resolution
        if not in_progress and (is_artifact or all(pair.consumed for pair in card.pairs)):
            evidence = "description"
        else:
            evidence = "full"
            if remaining is not None:
                # The floor wins over the arithmetic: an unconsumed pair travels whole regardless, so what would have
                # gone negative is clamped instead of denied.
                remaining = max(0, remaining - _part_cost(card, "evidence", costs))

        decided[title] = CardChoice(dialogue=dialogue, evidence=evidence)

    return TurnChoice(by_title=MappingProxyType(decided), full_pass=False)


def _titles_by_descending_note(state: _GraphState, notes: Mapping[str, float]) -> tuple[str, ...]:
    """Order the Titles by descending Note, with the turn ordinal and then the Title breaking ties.

    Total, and ties broken by two values that cannot collide, so the order the budget is spent in is the same on every
    run over the same state and the same Notes (Requirement 8.14).

    Args:
        state: The graph state. Read only.
        notes: One Note per Card. A missing Title reads as ``0.0``.

    Returns:
        Every Card Title, in the order the budget is handed out.
    """
    return tuple(
        sorted(
            state.cards,
            key=lambda title: (-notes.get(title, 0.0), state.cards[title].turn, title),
        )
    )


def _part_cost(card: Card, part: str, costs: Mapping[tuple[str, str], int] | None) -> int:
    """Estimate the token cost of one part of a Card in Full Content.

    Args:
        card: The Card the part belongs to.
        part: ``"dialogue"`` or ``"evidence"``.
        costs: The caller's cost table, or ``None`` to fall back on the count of addressed messages.

    Returns:
        The estimate, never negative.
    """
    if costs is not None:
        return max(0, costs.get((card.title, part), 0))
    if part == "dialogue":
        return len(card.dialogue_ids) * _TOKENS_PER_MESSAGE
    unconsumed = sum(len(pair.tracking_ids) for pair in card.pairs if not pair.consumed)
    return unconsumed * _TOKENS_PER_MESSAGE


def _score(
    question: str,
    descriptions: Sequence[str],
    matcher: SimilarityMatcher,
) -> tuple[float, ...] | None:
    """Score the Descriptions, or fail open with exactly one debug log carrying ``exc_info``.

    Malformed answers are raised inside the guarded block: one exit, one log, one traceback, whether the matcher raised
    or answered badly.

    Args:
        question: The turn's question.
        descriptions: The Cards' Descriptions, already in fixed order.
        matcher: The similarity matcher.

    Returns:
        One clamped similarity per Description, or ``None`` when the matcher was unusable.
    """
    try:
        similarities = matcher.score(question, descriptions)
        if len(similarities) != len(descriptions):
            # Covers the empty sequence too: with at least one Card, empty is a length mismatch.
            raise ValueError(f"similarity count=<{len(similarities)}> | expected=<{len(descriptions)}>")
        return tuple(_clamp(value) for value in similarities)
    except Exception:
        logger.debug(
            "graph similarity unavailable for %d description(s) | falling back to full content",
            len(descriptions),
            exc_info=True,
        )
        return None


def _clamp(value: object) -> float:
    """Clamp a similarity into ``[0.0, 1.0]``, mapping ``nan`` to ``0.0``.

    Notes must be totally ordered for the distribution to be deterministic, and ``nan`` is the one float that is not.

    Args:
        value: The similarity as answered. A non-numeric value raises, which the caller reads as a matcher failure.

    Returns:
        The similarity, within the closed interval.
    """
    number = float(value)  # type: ignore[arg-type]
    if math.isnan(number):
        return 0.0
    return min(1.0, max(0.0, number))
