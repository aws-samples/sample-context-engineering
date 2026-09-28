"""The neutral entry point: one function that turns a message list into the projected one.

This is the framework-agnostic recreation of what the Strands plugin spreads across three event hooks and one
middleware stage. The event wiring is **not** lifted -- there is no hook, no stage, no registry and no agent here.
What is lifted is the *order* those hooks imposed, which is the part that carries behaviour:

1. **The write half** (was ``MessageAddedEvent``). A turn is closed by what comes after it, so the Card of the last
   closed turn is derived off the messages themselves -- Title, Description, Tags and Links together, with no model,
   no disk and no network (Requirement 3.10). A state holding no Card in front of a history that already has a closed
   boundary is the restore case, and one scan derives the whole graph (:func:`~.cards.rebuild_into`, Requirements
   14.5, 14.6, 14.8).
2. **The read half** (was ``BeforeInvocationEvent``). The fed-back note is aged, the graph is scored against the
   turn's question in one embedding round, the description vector index is filled from the vectors that round already
   paid for, and the body budget is handed out in descending note (Requirements 10.4, 11.3, 13.2, 13.4).
3. **The delivery** (was the ``InvokeModelStage.Input`` handler). The removal drops the collapsed identities from the
   call's own message list and the compaction folds the Descriptions of what actually left into the last user message.
   They are one step because Requirement 16.2 does not admit the state in between: a removal applied with no block
   folded is a call that lost content and says nothing about the loss. Any failure therefore returns the **received**
   message list by object identity, with one warning (Requirement 16.3).

Two properties survive the move intact.

**Nothing is mutated.** The received list, the dicts inside it and the received state are all read only. The projected
list is new (or the received one itself, by identity, on a full pass) and the returned state is a copy.

**A full pass is the identity.** ``expand_threshold=0.0`` makes :func:`~.scoring.warm_up_choice` return the full-pass
choice before the matcher is ever reached, and a full-pass choice returns the received list *itself*, so the messages
a caller gets back are the messages it handed in -- the same object, not merely an equal one. That is the regression
short circuit of Requirement 1.11, machine-checked in ``tests/test_projection.py``.

**What this module does NOT carry**, and where it went instead: the SDK's ``_create_injection_middleware`` fold, the
``dynamic_trailing_blocks`` counter it maintains, the ``MiddlewareRegistry._handlers`` reordering, the per-agent
``WeakKeyDictionary`` lookup and the ``contextvars`` hand-off between the handler and its render callback were all
mechanism for delivering *inside* an event loop. A single synchronous function needs none of them: the fold is one
local call on a list (:func:`_fold_into_last_user_message`, the SDK primitive's behaviour reimplemented verbatim), and
ordering against a memory manager is the binding's problem, not the core's.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..message import NeutralMessage
from .cards import closed_turn_ranges, derive_and_register, link_newly_measurable, rebuild_into, turn_ranges
from .compaction import render_final_block
from .matcher import SimilarityMatcher
from .removal import apply_removal
from .scoring import (
    compute_notes,
    distribute,
    expire_reuse,
    full_pass_choice,
    titles_in_turn_order,
    warm_up_choice,
)
from .state import TurnChoice, _GraphState

__all__ = ["Thresholds", "current_turn_ids", "project"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Thresholds:
    """Everything the projection decides under, in one frozen record.

    The defaults are the plugin's constructor defaults, so a caller passing nothing gets the plugin's behaviour.

    Attributes:
        expand_threshold: Note at or above which a Card is Full Content, budget permitting. ``0.0`` is the regression
            short circuit: every Card goes at Full Content and the projection returns its input by identity.
        collapse_floor: Note below which a Card keeps only its Title. Must not exceed ``expand_threshold``.
        description_tokens: Token ceiling of a Description, and of one Card's entry in the final block.
        tags_per_card: How many identifiers define a Card.
        rarity_weight: How much rarity counts when Tags are selected.
        min_cards: Below this many Cards the choice is skipped entirely, before the matcher is reached.
        link_threshold: Similarity at or above which two Cards link.
        reuse_ttl_cycles: Cycles a fed-back note survives. ``0`` discards it at the end of the turn that fed it back.
        retrieval_tools: Names of the retrieval tools the host can actually call, named by the final block's guidance.
            Empty means the block says there is nothing to call with.
    """

    expand_threshold: float = 0.55
    collapse_floor: float = 0.45
    description_tokens: int = 100
    tags_per_card: int = 5
    rarity_weight: float = 0.70
    min_cards: int = 3
    link_threshold: float = 0.50
    reuse_ttl_cycles: int = 5
    retrieval_tools: tuple[str, ...] = field(default_factory=tuple)


_DEFAULTS = Thresholds()


def project(
    messages: Sequence[NeutralMessage],
    state: _GraphState | None = None,
    matcher: SimilarityMatcher | None = None,
    body_budget: int | None = None,
    thresholds: Thresholds = _DEFAULTS,
) -> tuple[Sequence[NeutralMessage], _GraphState]:
    """Project ``messages`` through the graph, and return them with the updated graph state.

    One call does the write half, the read half and the delivery, in that order, which is the order the three event
    hooks imposed. Neither ``messages`` nor ``state`` is modified: the state is deep-copied on the way in, so a caller
    may hand the same state to two projections and get two independent results.

    Args:
        messages: The conversation, oldest first, in the neutral shape of :mod:`context_core.message`. The trailing
            turn is the open one -- the question the model is answering -- and is excluded from every Resolution
            decision. Read only.
        state: The graph state carried over from the previous call, or ``None`` to start from a fresh graph. Read only;
            every field is plain data, so this same value is the graph's serialized form.
        matcher: The similarity matcher scoring the Descriptions against the question. ``None`` means no scoring is
            possible, which reads as "score nothing, send everything" and not as "nothing is relevant": the choice
            degrades to the full pass. The default :class:`~.matcher.EmbeddingSimilarityMatcher` is **not** built here,
            so this function reaches no network of its own.
        body_budget: Token ceiling across the Cards in Full Content, or ``None`` for no ceiling.
        thresholds: What the projection decides under. See :class:`Thresholds`.

    Returns:
        ``(projected_messages, new_state)``. On a full pass, on an empty removal request and on any failure of the
        delivery, ``projected_messages`` **is** the received ``messages`` object, so the assembled context is identical
        to the one produced without the feature (Requirements 9.8, 1.11, 16.3). Otherwise it is a new list carrying the
        removal and the final block. ``new_state`` is always a new object.
    """
    new_state = _copy_state(state) if state is not None else _GraphState()

    _close_turns(new_state, messages, thresholds)

    # The clock the fed-back note is aged by. The plugin reads ``agent.event_loop_metrics.cycle_count``; with no agent
    # to read, the turn ordinal is the neutral equivalent -- monotonic, incremented once per projection, and never
    # advanced by a burst of messages inside one turn (Requirement 13.3).
    expire_reuse(new_state, new_state.turn)
    new_state.choice = _compute_choice(new_state, messages, matcher, body_budget, thresholds)
    # After the increment the ordinal names the turn now opening, whose boundary the next projection will see.
    new_state.turn += 1
    # Counted per turn, so it starts each turn at zero.
    new_state.retrieval_cycles = 0

    return _deliver(new_state, messages, thresholds), new_state


def _copy_state(state: _GraphState) -> _GraphState:
    """Return an independent copy of ``state``, so the received one is never written to.

    A container-level copy is enough and ``copy.deepcopy`` is wrong here: every value the state holds is already
    immutable -- ``Card`` and ``Link`` are frozen dataclasses, the identities and vectors are tuples, ``referenced`` is
    a frozenset and ``TurnChoice`` is frozen around a ``MappingProxyType`` (which ``deepcopy`` cannot pickle at all). So
    copying the four mutable dicts is exactly the isolation the caller needs, and it costs no traversal of the Cards.

    Args:
        state: The state handed to the projection. Read only.

    Returns:
        A new state sharing no mutable container with ``state``.
    """
    return _GraphState(
        cards=dict(state.cards),
        links={title: list(edges) for title, edges in state.links.items()},
        choice=state.choice,
        reuse=dict(state.reuse),
        turn=state.turn,
        vectors=dict(state.vectors),
        retrieval_cycles=state.retrieval_cycles,
        referenced=state.referenced,
    )


def current_turn_ids(messages: Sequence[NeutralMessage]) -> frozenset[str]:
    """Durable identities of the turn in progress: the trailing range of ``messages``.

    :func:`~.cards.turn_ranges`' last range is the open turn by definition, so this is the slice
    :func:`~.cards.closed_turn_ranges` excludes, read from the other side. The removal subtracts it, which is what
    keeps the question the model is answering out of every Resolution decision.

    Args:
        messages: The call's message list. Read only.

    Returns:
        The identities of the open turn. Empty when no message opens a turn.
    """
    ranges = turn_ranges(messages)
    if not ranges:
        return frozenset()
    start, stop = ranges[-1]
    return frozenset(identity for message in messages[start:stop] if (identity := message.get("tracking_id")))


# ---- the write half: the turn boundary ------------------------------------------------------------


def _close_turns(state: _GraphState, messages: Sequence[NeutralMessage], thresholds: Thresholds) -> None:
    """Derive the Card of the last closed turn, or the whole graph by scan.

    A state holding no Card in front of a conversation that has at least one closed boundary is a process that
    inherited the history, and one scan derives the whole graph from the messages -- the same Cards, Links and Tags
    the turn-by-turn path would have built, with no serialized format to read and nothing remote to reach
    (Requirements 14.5, 14.8). Cards reference only messages present in the list, so a message that vanished is
    referenced by nothing and therefore takes part in no Resolution decision, without a raise (Requirement 14.7).

    A late write half is a turn with no Card yet, which projects Full Content -- the behaviour without the feature --
    and per-Card failures are absorbed by :func:`~.cards.derive_and_register`, which logs one warning and registers
    nothing (Requirements 16.4, 16.5).

    Args:
        state: The graph state to fill. Left as it was when the derivation failed.
        messages: The conversation. Read only.
        thresholds: The values a Card is derived under.
    """
    try:
        closed = closed_turn_ranges(messages)
        if not closed:
            # Nothing has closed yet: no turn to card, and nothing to rebuild from.
            return

        if not state.cards:
            # The restore case, and equally the first closed boundary of a fresh graph, where the scan and the
            # turn-by-turn step derive the same single Card and the two paths' turn ordinals stay aligned from here.
            rebuild_into(state, messages, **_card_config(thresholds))
            return

        start, stop = closed[-1]
        turn_ids = _identities_in(messages[start:stop])
        if not turn_ids:
            # A turn whose messages all lack a Durable Identity: messages without a Card, projected whole.
            return

        # The ordinal is the position of the boundary among the closed turns, not ``state.turn``, so a turn's Card is
        # the same whether it was derived turn by turn or by one scan (Requirement 14.3).
        derive_and_register(state, messages, turn_ids, len(closed) - 1, **_card_config(thresholds))
    except Exception:
        logger.warning("closing the turn's card failed | the turn's messages go whole", exc_info=True)


def _card_config(thresholds: Thresholds) -> dict[str, Any]:
    """Return the values a Card is derived under, shared by the turn-by-turn step and by the rebuild scan.

    One source for both, which is what makes them comparable: the scan is only equal to the turn-by-turn graph if it
    ran under the same ceilings and the same link threshold (Requirement 14.5).

    Args:
        thresholds: What the projection decides under.

    Returns:
        The keyword arguments both derivation entry points take.
    """
    return {
        "description_tokens": thresholds.description_tokens,
        "tags_per_card": thresholds.tags_per_card,
        "rarity_weight": thresholds.rarity_weight,
        "link_threshold": thresholds.link_threshold,
    }


def _identities_in(messages: Sequence[NeutralMessage]) -> tuple[str, ...]:
    """Return the Durable Identities of ``messages``, in order and without duplicates.

    The same shape the rebuild scan hands :func:`~.cards.derive_and_register`, which is what makes the turn-by-turn
    construction and the scan produce the same Card for the same turn (Requirements 14.3, 14.5). It is also the one
    place a vanished message drops out of the graph: an identity is collected only from a message present in the list,
    so a Card can never reference one that is not (Requirement 14.7).

    Args:
        messages: The messages of one turn, in order. Read only.

    Returns:
        The identities. A message carrying none contributes nothing: it is a message without a Card.
    """
    return tuple(dict.fromkeys(identity for message in messages if (identity := message.get("tracking_id"))))


# ---- the read half: the turn choice ---------------------------------------------------------------


def _compute_choice(
    state: _GraphState,
    messages: Sequence[NeutralMessage],
    matcher: SimilarityMatcher | None,
    body_budget: int | None,
    thresholds: Thresholds,
) -> TurnChoice:
    """Score the graph against the turn's question, fill the vector index, and hand out the body budget.

    Any failure at any step degrades to the full pass -- the behaviour without the feature -- with exactly one warning
    carrying ``exc_info`` and no failure state kept, so the next projection computes a choice again (Requirement 16.1).

    Args:
        state: The graph state. Read, except for ``vectors``, which this fills from the embedding the scoring round
            already paid for.
        messages: The conversation, for the question. Read only.
        matcher: The matcher, or ``None``.
        body_budget: Token ceiling across the Cards in Full Content, or ``None``.
        thresholds: What the projection decides under.

    Returns:
        The frozen choice of this turn.
    """
    try:
        skipped = warm_up_choice(
            state,
            expand_threshold=thresholds.expand_threshold,
            min_cards=thresholds.min_cards,
        )
        if skipped is not None:
            # Below ``min_cards``, and at ``expand_threshold=0.0``, the only possible decision is "send it all", so no
            # embedding is paid for a decision the size of the graph already made (Requirements 8.12, 11.3).
            return skipped

        if matcher is None:
            # No matcher is "score nothing, send everything", the same reading a failed matcher gets below.
            return full_pass_choice()

        question = _question_of(messages)
        # The one embedding round of the turn: the question as the query, the Descriptions as the documents (Req. 10.4).
        notes = compute_notes(state, question, matcher)
        if not notes:
            # The matcher failed or answered malformed, which reads as "score nothing, send everything" and not as
            # "nothing is relevant".
            return full_pass_choice()

        # Only on this path, and only after the scoring round: the vectors are already in the matcher's own cache under
        # the document purpose, so filling the index sends nothing. Above the ``min_cards`` short circuit it would have
        # sent a request for a decision that was never taken, which Requirement 11.3 forbids.
        _cache_description_vectors(state, matcher, link_threshold=thresholds.link_threshold)

        return distribute(
            notes,
            state,
            expand_threshold=thresholds.expand_threshold,
            collapse_floor=thresholds.collapse_floor,
            body_budget=body_budget,
        )
    except Exception:
        logger.warning("turn choice failed | the whole conversation goes at full content", exc_info=True)
        return full_pass_choice()


def _question_of(messages: Sequence[NeutralMessage]) -> str:
    """Return the text the turn's Cards are scored against: the texts of the last user message.

    Args:
        messages: The messages to read the question from, in order. Read only.

    Returns:
        The texts of the last user message, joined. Empty when no user message carries text, which scores every Card
        against nothing and therefore leaves the whole graph at its floor.
    """
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        texts = [
            block["text"]
            for block in message.get("content") or ()
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ]
        if texts:
            return " ".join(texts)

    return ""


def _cache_description_vectors(state: _GraphState, matcher: SimilarityMatcher, *, link_threshold: float) -> None:
    """Fill ``state.vectors`` from the Descriptions the scoring round just embedded.

    Without this the index stays empty, and an empty index is not a slow path but a missing feature: the similarity
    Link is measured from this index alone, by a step Requirement 3.2 keeps free of network calls, so an unfilled index
    makes every pair unmeasurable and the ``similar`` Link never forms at all. The three structural Link kinds still
    form, which is why a graph with no ``similar`` edge looks like a working graph in the counters.

    Two properties keep it free. It runs only where :func:`_compute_choice` has already scored, so the document vectors
    sit in the matcher's cache; and ``vectors`` is optional on the protocol, so a matcher that does not publish one
    leaves the index as it was rather than failing the turn.

    Stale Titles are dropped rather than left to accumulate: a Description that changed makes its entry unusable
    anyway, and a Title no longer in the graph will not be asked about again.

    Args:
        state: The graph state. ``vectors`` is replaced; nothing else is touched.
        matcher: The matcher that just scored, and therefore already holds these vectors.
        link_threshold: Similarity at or above which two Cards link.
    """
    published = getattr(matcher, "vectors", None)
    if published is None:
        return

    titles = titles_in_turn_order(state)
    descriptions = tuple(state.cards[title].description for title in titles)
    try:
        vectors = published(descriptions)
    except Exception:
        # Same posture as the scoring round: an unusable index costs Links, never the turn.
        logger.debug("graph description vectors unavailable for %d card(s)", len(titles), exc_info=True)
        return

    if len(vectors) != len(titles):
        # Includes the empty answer the matcher returns when the embedding was unavailable. A partial index would pair
        # vectors with the wrong Titles, which is worse than no index.
        return

    filled = {
        title: (description, tuple(vector))
        for title, description, vector in zip(titles, descriptions, vectors, strict=True)
    }
    # A Title whose entry is unchanged has already been measured against everything; one that is new, or whose
    # Description was rewritten, is what the second pass is for.
    newly_measurable = [title for title, entry in filled.items() if state.vectors.get(title) != entry]
    state.vectors = filled

    if newly_measurable:
        link_newly_measurable(state, newly_measurable, link_threshold=link_threshold)


# ---- the delivery: removal and compaction, as one step --------------------------------------------


def _deliver(
    state: _GraphState,
    messages: Sequence[NeutralMessage],
    thresholds: Thresholds,
) -> Sequence[NeutralMessage]:
    """Apply the removal and the compaction, as one step that either happens or does not.

    Requirement 16.2 does not admit the state in between, so a failure anywhere returns the **received** list by object
    identity rather than a list carrying the removal with nothing folded (Requirement 16.3).

    Args:
        state: The graph state, holding the Cards and the frozen choice. Read only.
        messages: The call's message list. Read only, never modified in place.
        thresholds: What the projection decides under, for the entry ceiling and the guidance.

    Returns:
        A new list carrying the removal and the final block, or the received list itself -- by object identity on a
        full pass, on an empty request, and on any failure.
    """
    try:
        # A fresh state is a full pass, so a conversation with no Card behaves as it does without the feature.
        if state.choice.full_pass:
            return messages

        removed, requested = apply_removal(messages, state, state.choice, current_turn_ids(messages))
        if not requested:
            # Nothing to drop, so nothing to describe: the same list object.
            return messages

        final_block = render_final_block(
            removed,
            state,
            requested,
            description_tokens=thresholds.description_tokens,
            retrieval_tools=thresholds.retrieval_tools,
        )
        if final_block is None:
            # The removal happened and no part contributed a fragment: nothing to fold, so the removed list goes as it
            # is. Not the intermediate state Requirement 16.2 forbids -- there is no description to lose.
            return removed

        folded, _ = _fold_into_last_user_message(removed, final_block)
        return folded
    except Exception:
        logger.warning("delivery failed | passing the received messages through unchanged", exc_info=True)
        return messages


def _fold_into_last_user_message(
    messages: Sequence[NeutralMessage],
    text: str,
) -> tuple[list[NeutralMessage], int]:
    """Fold ``text`` into the most recent ``user`` message as a text block, returning a NEW list.

    The behaviour of the SDK's own injection primitive, reimplemented so the core carries no framework import. Folding
    into the existing user message (rather than inserting a standalone message) keeps role alternation valid in both
    chat and the autonomous tool loop.

    The text is always **appended**. A tool result has to stay the first content block in the turn that answers a tool
    use, and a trailing run is the only placement a provider can keep out of its cached prefix -- text ahead of the
    stable conversation would invalidate the cache from the first block onward.

    The input list and its messages are never mutated. When there is no ``user`` message, a copy of the input list is
    returned unchanged.

    Args:
        messages: The conversation to fold into. Read only.
        text: The text to fold into the most recent user message.

    Returns:
        The folded conversation and how many trailing blocks of its last message are per-call content: 1 when the fold
        landed on the last message, 0 otherwise (including when there is no user message to fold into).
    """
    target_index = -1
    for index in range(len(messages) - 1, -1, -1):
        if messages[index]["role"] == "user":
            target_index = index
            break
    if target_index < 0:
        return list(messages), 0

    target = messages[target_index]
    # Some providers concatenate adjacent text blocks, which would run this onto the user's own words.
    separator = "\n\n" if target["content"] and not text.startswith("\n") else ""
    injected: NeutralMessage = {"text": f"{separator}{text}"}
    content = [*target["content"], injected]

    folded: NeutralMessage = {"role": target["role"], "content": content}
    for carried in ("metadata", "tracking_id"):
        # ``tracking_id`` travels with the message: dropping it here would make the folded message invisible to the
        # next projection's scan, which reads Cards off exactly these identities.
        if carried in target:
            folded[carried] = target[carried]

    result = list(messages)
    result[target_index] = folded
    # A provider only places its cache point in the last message, so a fold elsewhere reports nothing.
    return result, (1 if target_index == len(messages) - 1 else 0)
