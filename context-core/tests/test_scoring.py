"""Unit tests for the two-pass Note and the distribution across the three resolutions.

The matcher is a table here, never an embedding: every similarity is chosen so the propagation product is checkable by
hand, which is what lets a test assert the exact Note rather than an inequality. The arithmetic is asserted with
``pytest.approx`` because the weights are floats; the ordering, the thresholds and the budget are asserted exactly.
"""

from collections.abc import Sequence

import pytest

from context_core.graph.scoring import (
    _DECAY,
    _REUSE_BONUS,
    _W_ARTIFACT,
    _W_PREVIOUS,
    _W_TOOL,
    compute_notes,
    distribute,
    expire_reuse,
    full_pass_choice,
    record_reuse,
    titles_in_turn_order,
    warm_up_choice,
)
from context_core.graph.state import Card, Link, ToolPair, _GraphState


class TableMatcher:
    """A matcher answering from a table, counting its invocations."""

    def __init__(self, table: dict[str, float], *, default: float = 0.0) -> None:
        """Answer ``table[description]`` per Description, falling back on ``default``."""
        self.table = table
        self.default = default
        self.calls = 0
        self.seen: list[tuple[str, tuple[str, ...]]] = []

    def score(self, question: str, descriptions: Sequence[str]) -> Sequence[float]:
        """Return one similarity per Description, in the order received."""
        self.calls += 1
        self.seen.append((question, tuple(descriptions)))
        return [self.table.get(description, self.default) for description in descriptions]


class BrokenMatcher:
    """A matcher that raises, standing in for a transport failure or a timeout."""

    def score(self, question: str, descriptions: Sequence[str]) -> Sequence[float]:
        """Raise, which the caller must read as "score nothing, send everything"."""
        raise RuntimeError("embedding unavailable")


class ShortMatcher:
    """A matcher answering fewer values than it was given Descriptions."""

    def score(self, question: str, descriptions: Sequence[str]) -> Sequence[float]:
        """Return one value short of the count asked for."""
        return [0.9] * max(0, len(descriptions) - 1)


def pair(*, consumed: bool, tool_use_id: str = "tu-1") -> ToolPair:
    """Build a tool pair whose consumption state is the point of the test."""
    return ToolPair(tool_use_id=tool_use_id, tool_name="calculator", tracking_ids=("a", "b"), consumed=consumed)


def card(
    title: str,
    *,
    turn: int = 0,
    kind: str = "subject",
    description: str | None = None,
    dialogue_ids: tuple[str, ...] = ("m-1",),
    pairs: tuple[ToolPair, ...] = (),
) -> Card:
    """Build a Card carrying only the fields ``scoring`` reads."""
    return Card(
        title=title,
        kind=kind,  # type: ignore[arg-type]
        turn=turn,
        dialogue_ids=dialogue_ids,
        evidence_ids=("e-1",) if pairs else (),
        pairs=pairs,
        tool_names=frozenset(),
        references=(),
        numeric_lines=(),
        tags=(),
        description=description if description is not None else f"description of {title}",
        reference=title if kind == "artifact" else None,
    )


def graph(*cards: Card, links: dict[str, list[Link]] | None = None, turn: int | None = None) -> _GraphState:
    """Assemble a state around ``cards``, with the turn ordinal past the last closed turn by default."""
    state = _GraphState()
    for entry in cards:
        state.cards[entry.title] = entry
    state.links.update(links or {})
    state.turn = turn if turn is not None else max((entry.turn for entry in cards), default=-1) + 1
    return state


# ---- The warm-up short circuit (Requirements 8.12, 8.13) ----


def test_full_pass_choice_is_frozen_and_empty() -> None:
    choice = full_pass_choice()

    assert choice.full_pass is True
    assert dict(choice.by_title) == {}
    with pytest.raises(TypeError):
        choice.by_title["anything"] = None  # type: ignore[index]


def test_below_min_cards_skips_the_choice() -> None:
    state = graph(card("first", turn=0), card("second", turn=1))

    choice = warm_up_choice(state, expand_threshold=0.55, min_cards=3)

    assert choice is not None
    assert choice.full_pass is True


def test_first_turn_holds_no_card_and_skips_the_choice() -> None:
    choice = warm_up_choice(graph(), expand_threshold=0.55, min_cards=3)

    assert choice is not None
    assert choice.full_pass is True


def test_zero_expand_threshold_is_the_off_switch() -> None:
    state = graph(card("a", turn=0), card("b", turn=1), card("c", turn=2))

    choice = warm_up_choice(state, expand_threshold=0.0, min_cards=3)

    assert choice is not None
    assert choice.full_pass is True


def test_at_min_cards_the_choice_proceeds() -> None:
    state = graph(card("a", turn=0), card("b", turn=1), card("c", turn=2))

    assert warm_up_choice(state, expand_threshold=0.55, min_cards=3) is None


def test_the_warm_up_never_reaches_the_matcher() -> None:
    matcher = TableMatcher({})
    state = graph(card("only", turn=0))

    assert warm_up_choice(state, expand_threshold=0.55, min_cards=3) is not None
    assert matcher.calls == 0


# ---- Ordering and determinism (Requirement 8.14) ----


def test_titles_are_ordered_by_turn_then_title() -> None:
    state = graph(card("zeta", turn=0), card("alpha", turn=2), card("beta", turn=2))

    assert titles_in_turn_order(state) == ("zeta", "alpha", "beta")


def test_descriptions_reach_the_matcher_in_turn_order() -> None:
    state = graph(card("second", turn=1), card("first", turn=0))
    matcher = TableMatcher({})

    compute_notes(state, "the question", matcher)

    assert matcher.seen == [("the question", ("description of first", "description of second"))]


def test_the_same_state_and_similarities_produce_the_same_choice() -> None:
    state = graph(
        card("a", turn=0),
        card("b", turn=1, pairs=(pair(consumed=True),)),
        card("c", turn=2),
        links={"b": [Link(kind="follows", target="a", weight=1.0)]},
    )
    table = {"description of a": 0.2, "description of b": 0.6, "description of c": 0.9}

    first = distribute(
        compute_notes(state, "q", TableMatcher(table)),
        state,
        expand_threshold=0.55,
        collapse_floor=0.45,
        body_budget=700,
    )
    second = distribute(
        compute_notes(state, "q", TableMatcher(table)),
        state,
        expand_threshold=0.55,
        collapse_floor=0.45,
        body_budget=700,
    )

    assert dict(first.by_title) == dict(second.by_title)


# ---- The first pass, and the Fed-Back Note (Requirements 8.1, 13.2) ----


def test_the_first_pass_is_the_similarity_itself() -> None:
    state = graph(card("a", turn=0), card("b", turn=1))
    table = {"description of a": 0.3, "description of b": 0.8}

    notes = compute_notes(state, "q", TableMatcher(table))

    assert notes == {"a": pytest.approx(0.3), "b": pytest.approx(0.8)}


def test_similarities_are_clamped_into_the_unit_interval() -> None:
    state = graph(card("low", turn=0), card("high", turn=1), card("nan", turn=2))
    table = {"description of low": -3.0, "description of high": 4.0, "description of nan": float("nan")}

    notes = compute_notes(state, "q", TableMatcher(table))

    assert notes == {"low": pytest.approx(0.0), "high": pytest.approx(1.0), "nan": pytest.approx(0.0)}


def test_the_fed_back_note_is_added_before_the_propagation() -> None:
    state = graph(
        card("a", turn=0),
        card("b", turn=1),
        links={"b": [Link(kind="follows", target="a", weight=1.0)]},
    )
    state.reuse["b"] = (1.0, 99)
    table = {"description of a": 0.1, "description of b": 0.2}

    notes = compute_notes(state, "q", TableMatcher(table))

    # The bonus is part of what `b` propagates, so `a` inherits from 1.2 and not from 0.2.
    assert notes["b"] == pytest.approx(1.2)
    assert notes["a"] == pytest.approx(0.1 + 1.2 * _DECAY * _W_PREVIOUS)


def test_a_fed_back_note_for_an_absent_card_is_ignored() -> None:
    state = graph(card("a", turn=0))
    state.reuse["vanished"] = (1.0, 99)

    notes = compute_notes(state, "q", TableMatcher({"description of a": 0.4}))

    assert notes == {"a": pytest.approx(0.4)}


# ---- The fed-back Note's TTL aging (Requirements 13.1, 13.3 to 13.8) ----


def test_a_grant_stores_the_whole_bonus_and_its_expiry_cycle() -> None:
    state = graph(card("a", turn=0))

    record_reuse(state, "a", 4, reuse_ttl_cycles=3)

    assert state.reuse == {"a": (_REUSE_BONUS, 7)}


def test_a_zero_ttl_grant_writes_nothing() -> None:
    state = graph(card("a", turn=0))

    record_reuse(state, "a", 4, reuse_ttl_cycles=0)

    assert state.reuse == {}


def test_a_repeat_request_restarts_the_countdown() -> None:
    state = graph(card("a", turn=0))

    record_reuse(state, "a", 4, reuse_ttl_cycles=3)
    record_reuse(state, "a", 6, reuse_ttl_cycles=3)

    assert state.reuse == {"a": (_REUSE_BONUS, 9)}


@pytest.mark.parametrize("cycle", [4, 5, 6, 7])
def test_the_note_stands_whole_up_to_its_expiry_cycle(cycle: int) -> None:
    state = graph(card("a", turn=0))
    record_reuse(state, "a", 4, reuse_ttl_cycles=3)

    expire_reuse(state, cycle)

    # Granted at cycle 4 with a TTL of 3: present while the counter is at most 7, and undecayed throughout.
    assert state.reuse == {"a": (_REUSE_BONUS, 7)}


def test_the_note_is_removed_once_the_counter_passes_its_expiry() -> None:
    state = graph(card("a", turn=0))
    record_reuse(state, "a", 4, reuse_ttl_cycles=3)

    expire_reuse(state, 8)

    assert state.reuse == {}


def test_aging_decides_each_card_on_its_own_expiry() -> None:
    state = graph(card("old", turn=0), card("fresh", turn=1))
    record_reuse(state, "old", 1, reuse_ttl_cycles=2)
    record_reuse(state, "fresh", 5, reuse_ttl_cycles=2)

    expire_reuse(state, 6)

    assert state.reuse == {"fresh": (_REUSE_BONUS, 7)}


def test_aging_twice_within_one_cycle_changes_nothing() -> None:
    state = graph(card("a", turn=0))
    record_reuse(state, "a", 4, reuse_ttl_cycles=3)

    expire_reuse(state, 6)
    expire_reuse(state, 6)

    assert state.reuse == {"a": (_REUSE_BONUS, 7)}


def test_aging_an_empty_map_is_a_no_op() -> None:
    state = graph(card("a", turn=0))

    expire_reuse(state, 99)

    assert state.reuse == {}


def test_an_expired_note_no_longer_reaches_the_note() -> None:
    state = graph(card("a", turn=0), card("b", turn=1))
    record_reuse(state, "b", 1, reuse_ttl_cycles=1)
    expire_reuse(state, 3)

    notes = compute_notes(state, "q", TableMatcher({"description of a": 0.1, "description of b": 0.2}))

    assert notes == {"a": pytest.approx(0.1), "b": pytest.approx(0.2)}


# ---- The propagation (Requirements 8.2, 8.3, 8.4) ----


def test_the_previous_turn_link_carries_its_own_weight() -> None:
    state = graph(
        card("a", turn=0),
        card("b", turn=1),
        links={"b": [Link(kind="follows", target="a", weight=1.0)]},
    )
    table = {"description of a": 0.4, "description of b": 0.8}

    notes = compute_notes(state, "q", TableMatcher(table))

    assert notes["a"] == pytest.approx(0.4 + 0.8 * _DECAY * _W_PREVIOUS)
    assert notes["b"] == pytest.approx(0.8)


def test_the_artifact_link_carries_its_own_weight() -> None:
    state = graph(
        card("a", turn=0),
        card("ref-1", turn=0, kind="artifact"),
        links={"a": [Link(kind="artifact", target="ref-1", weight=1.0)]},
    )
    table = {"description of a": 0.6, "description of ref-1": 0.1}

    notes = compute_notes(state, "q", TableMatcher(table))

    assert notes["ref-1"] == pytest.approx(0.1 + 0.6 * _DECAY * _W_ARTIFACT)


def test_two_cards_reach_each_other_through_their_shared_tool() -> None:
    state = graph(
        card("a", turn=0),
        card("b", turn=1),
        links={
            "a": [Link(kind="tool", target="calculator", weight=1.0)],
            "b": [Link(kind="tool", target="calculator", weight=1.0)],
        },
    )
    table = {"description of a": 0.4, "description of b": 0.8}

    notes = compute_notes(state, "q", TableMatcher(table))

    # Each Card receives the hub total minus its own contribution, so neither propagates to itself.
    assert notes["a"] == pytest.approx(0.4 + 0.8 * _W_TOOL * _DECAY)
    assert notes["b"] == pytest.approx(0.8 + 0.4 * _W_TOOL * _DECAY)


def test_a_card_alone_on_a_tool_hub_gains_nothing_from_it() -> None:
    state = graph(
        card("a", turn=0),
        links={"a": [Link(kind="tool", target="calculator", weight=1.0)]},
    )

    notes = compute_notes(state, "q", TableMatcher({"description of a": 0.5}))

    assert notes["a"] == pytest.approx(0.5)


def test_the_similarity_link_inherits_no_note() -> None:
    state = graph(
        card("a", turn=0),
        card("b", turn=1),
        links={"b": [Link(kind="similar", target="a", weight=0.8)]},
    )
    table = {"description of a": 0.2, "description of b": 0.9}

    notes = compute_notes(state, "q", TableMatcher(table))

    assert notes["a"] == pytest.approx(0.2)


def test_the_propagation_runs_in_exactly_one_jump() -> None:
    state = graph(
        card("a", turn=0),
        card("b", turn=1),
        card("c", turn=2),
        links={
            "b": [Link(kind="follows", target="a", weight=1.0)],
            "c": [Link(kind="follows", target="b", weight=1.0)],
        },
    )
    table = {"description of a": 0.0, "description of b": 0.0, "description of c": 1.0}

    notes = compute_notes(state, "q", TableMatcher(table))

    assert notes["b"] == pytest.approx(1.0 * _DECAY * _W_PREVIOUS)
    # `a` inherits from `b`'s *first pass*, which is zero: `c` does not reach it in the same turn.
    assert notes["a"] == pytest.approx(0.0)


def test_a_link_to_a_vanished_card_is_skipped() -> None:
    state = graph(
        card("a", turn=0),
        links={"a": [Link(kind="follows", target="gone", weight=1.0)]},
    )

    assert compute_notes(state, "q", TableMatcher({"description of a": 0.5})) == {"a": pytest.approx(0.5)}


def test_the_propagation_only_ever_adds() -> None:
    state = graph(
        card("a", turn=0),
        card("b", turn=1),
        links={
            "a": [Link(kind="tool", target="calculator", weight=1.0)],
            "b": [Link(kind="tool", target="calculator", weight=1.0), Link(kind="follows", target="a", weight=1.0)],
        },
    )
    table = {"description of a": 0.4, "description of b": 0.1}

    notes = compute_notes(state, "q", TableMatcher(table))

    assert notes["a"] >= 0.4
    assert notes["b"] >= 0.1


# ---- Failing open (the matcher contract) ----


def test_a_raising_matcher_scores_nothing() -> None:
    state = graph(card("a", turn=0), card("b", turn=1))

    assert compute_notes(state, "q", BrokenMatcher()) == {}


def test_a_short_answer_scores_nothing() -> None:
    state = graph(card("a", turn=0), card("b", turn=1))

    assert compute_notes(state, "q", ShortMatcher()) == {}


def test_no_notes_still_decides_every_card() -> None:
    state = graph(card("a", turn=0), card("b", turn=1))

    choice = distribute({}, state, expand_threshold=0.55, collapse_floor=0.45, body_budget=None)

    assert set(choice.by_title) == {"a", "b"}


# ---- The distribution (Requirements 7.1, 8.7 to 8.11) ----


def test_a_note_above_the_threshold_with_no_ceiling_is_full_content() -> None:
    state = graph(card("a", turn=0), card("b", turn=1))

    choice = distribute(
        {"a": 0.9, "b": 0.55},
        state,
        expand_threshold=0.55,
        collapse_floor=0.45,
        body_budget=None,
    )

    assert choice.by_title["a"].dialogue == "full"
    assert choice.by_title["b"].dialogue == "full"
    assert choice.full_pass is False


def test_a_note_between_the_thresholds_is_description() -> None:
    state = graph(card("a", turn=0))

    choice = distribute({"a": 0.45}, state, expand_threshold=0.55, collapse_floor=0.45, body_budget=None)

    assert choice.by_title["a"].dialogue == "description"


def test_a_note_below_the_floor_is_title() -> None:
    state = graph(card("a", turn=0))

    choice = distribute({"a": 0.44}, state, expand_threshold=0.55, collapse_floor=0.45, body_budget=None)

    assert choice.by_title["a"].dialogue == "title"


def test_the_budget_is_spent_in_descending_note_and_steps_the_rest_down() -> None:
    state = graph(card("rich", turn=0), card("poor", turn=1))

    # One dialogue message each, so the table prices them equally and only one fits.
    choice = distribute(
        {"rich": 0.9, "poor": 0.8},
        state,
        expand_threshold=0.55,
        collapse_floor=0.45,
        body_budget=100,
        costs={("rich", "dialogue"): 100, ("poor", "dialogue"): 100},
    )

    assert choice.by_title["rich"].dialogue == "full"
    # One rung down, never to Title.
    assert choice.by_title["poor"].dialogue == "description"


def test_the_fallback_estimate_prices_a_card_by_its_message_count() -> None:
    state = graph(card("a", turn=0, dialogue_ids=("m-1", "m-2")))

    fits = distribute({"a": 0.9}, state, expand_threshold=0.55, collapse_floor=0.45, body_budget=500)
    misses = distribute({"a": 0.9}, state, expand_threshold=0.55, collapse_floor=0.45, body_budget=499)

    assert fits.by_title["a"].dialogue == "full"
    assert misses.by_title["a"].dialogue == "description"


def test_every_card_is_decided_and_none_is_dropped() -> None:
    state = graph(card("a", turn=0), card("b", turn=1), card("c", turn=2))

    choice = distribute(
        {"a": 0.9, "b": 0.5, "c": 0.1},
        state,
        expand_threshold=0.55,
        collapse_floor=0.45,
        body_budget=None,
    )

    assert set(choice.by_title) == {"a", "b", "c"}
    with pytest.raises(TypeError):
        choice.by_title["a"] = None  # type: ignore[index]


def test_an_artifact_card_never_reaches_full_content() -> None:
    state = graph(card("ref-1", turn=0, kind="artifact"), turn=1)

    choice = distribute({"ref-1": 1.0}, state, expand_threshold=0.55, collapse_floor=0.45, body_budget=None)

    assert choice.by_title["ref-1"].dialogue == "description"
    assert choice.by_title["ref-1"].evidence == "description"


# ---- The Evidence axis (Requirements 7.1 to 7.5) ----


def test_consumed_evidence_is_description_while_the_dialogue_stays_full() -> None:
    state = graph(card("a", turn=0, pairs=(pair(consumed=True),)), turn=1)

    choice = distribute({"a": 0.9}, state, expand_threshold=0.55, collapse_floor=0.45, body_budget=None)

    assert choice.by_title["a"].dialogue == "full"
    assert choice.by_title["a"].evidence == "description"


def test_unconsumed_evidence_travels_whole_while_the_dialogue_collapses() -> None:
    state = graph(card("a", turn=0, pairs=(pair(consumed=False),)), turn=1)

    choice = distribute({"a": 0.0}, state, expand_threshold=0.55, collapse_floor=0.45, body_budget=None)

    assert choice.by_title["a"].dialogue == "title"
    assert choice.by_title["a"].evidence == "full"


def test_one_unconsumed_pair_keeps_the_whole_evidence() -> None:
    state = graph(
        card("a", turn=0, pairs=(pair(consumed=True, tool_use_id="tu-1"), pair(consumed=False, tool_use_id="tu-2"))),
        turn=1,
    )

    choice = distribute({"a": 0.5}, state, expand_threshold=0.55, collapse_floor=0.45, body_budget=None)

    assert choice.by_title["a"].evidence == "full"


def test_a_card_with_no_evidence_at_all_is_description() -> None:
    state = graph(card("a", turn=0), turn=1)

    choice = distribute({"a": 0.0}, state, expand_threshold=0.55, collapse_floor=0.45, body_budget=None)

    assert choice.by_title["a"].evidence == "description"


def test_unconsumed_evidence_is_never_denied_by_an_exhausted_budget() -> None:
    state = graph(card("a", turn=0, pairs=(pair(consumed=False),)), turn=1)

    choice = distribute(
        {"a": 0.9},
        state,
        expand_threshold=0.55,
        collapse_floor=0.45,
        body_budget=1,
        costs={("a", "dialogue"): 1, ("a", "evidence"): 10_000},
    )

    assert choice.by_title["a"].evidence == "full"


# ---- The Turn in Progress (Requirement 7.2) ----


def test_a_card_of_the_turn_in_progress_is_full_content_on_both_axes() -> None:
    state = graph(card("running", turn=3, pairs=(pair(consumed=True),)), turn=3)

    choice = distribute(
        {"running": 0.0},
        state,
        expand_threshold=0.55,
        collapse_floor=0.45,
        body_budget=1,
        costs={("running", "dialogue"): 10_000},
    )

    assert choice.by_title["running"].dialogue == "full"
    assert choice.by_title["running"].evidence == "full"


def test_a_closed_card_is_decided_by_its_note() -> None:
    state = graph(card("closed", turn=2), turn=3)

    choice = distribute({"closed": 0.0}, state, expand_threshold=0.55, collapse_floor=0.45, body_budget=None)

    assert choice.by_title["closed"].dialogue == "title"
