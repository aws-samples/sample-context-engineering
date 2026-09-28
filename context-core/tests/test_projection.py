"""Unit tests for the neutral projection entry point.

``project`` is not a lift of the Strands ``Projection`` class -- that one is event glue -- but it must reproduce the
order and the outcomes those three hooks imposed. These tests pin the claims that survive the move:

- **the identity** (Requirements 1.11, 2.20): ``expand_threshold=0.0`` and ``collapse_floor=0.0`` project every Card at
  Full Content, and the list that comes out **is** the list that went in, by object identity, with the matcher never
  reached;
- the write half derives one Card per closed turn off the neutral messages alone;
- the delivery drops the collapsed identities and folds the Descriptions of what actually left into the last user
  message, as one step;
- nothing is mutated: neither the received list, nor the messages in it, nor the received state.

Every matcher here is a stub. No embedding call, no network, no framework import.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from types import MappingProxyType

from context_core.graph.projection import Thresholds, _deliver, current_turn_ids, project
from context_core.graph.state import CardChoice, TurnChoice, _GraphState


class UnusedMatcher:
    """A matcher that fails the test if it is ever reached.

    ``expand_threshold=0.0`` is decided by configuration, so no embedding round is owed on any turn.
    """

    def __init__(self) -> None:
        self.calls = 0

    def score(self, question: str, descriptions: Sequence[str]) -> Sequence[float]:
        self.calls += 1
        raise AssertionError("the full-pass switch must decide before the matcher is reached")


class TableMatcher:
    """Scores a Description by a lookup on a substring of it, so a test names the note it wants."""

    def __init__(self, table: dict[str, float], default: float = 0.0) -> None:
        self._table = table
        self._default = default
        self.calls = 0

    def score(self, question: str, descriptions: Sequence[str]) -> Sequence[float]:
        self.calls += 1
        scores = []
        for description in descriptions:
            score = self._default
            for needle, value in self._table.items():
                if needle in description:
                    score = value
                    break
            scores.append(score)
        return scores


class BrokenMatcher:
    """A matcher that raises, which must read as "score nothing, send everything"."""

    def score(self, question: str, descriptions: Sequence[str]) -> Sequence[float]:
        raise RuntimeError("no embedding available")


def user(text: str, tracking_id: str) -> dict:
    """A plain user ask: a turn boundary, since it carries no tool result."""
    return {"role": "user", "content": [{"text": text}], "tracking_id": tracking_id}


def assistant(text: str, tracking_id: str) -> dict:
    """An assistant answer carrying text, which consumes the tool pairs before it."""
    return {"role": "assistant", "content": [{"text": text}], "tracking_id": tracking_id}


def tool_use(name: str, tool_use_id: str, tracking_id: str) -> dict:
    """An assistant message asking for a tool call."""
    return {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": tool_use_id, "name": name, "input": {}}}],
        "tracking_id": tracking_id,
    }


def tool_result(text: str, tool_use_id: str, tracking_id: str) -> dict:
    """A user message carrying the tool's return, which is evidence and no turn boundary."""
    return {
        "role": "user",
        "content": [{"toolResult": {"toolUseId": tool_use_id, "status": "success", "content": [{"text": text}]}}],
        "tracking_id": tracking_id,
    }


def conversation() -> list[dict]:
    """Four turns, three of them closed, each with a tool pair and a numeric line in its evidence."""
    return [
        user("What did the Frankfurt region cost last month?", "t1"),
        tool_use("billing", "tu-1", "t2"),
        tool_result("region: eu-central-1\ntotal: $1,204.00", "tu-1", "t3"),
        assistant("Frankfurt came to $1,204.00 last month.", "t4"),
        user("And how many instances were running in Ireland?", "t5"),
        tool_use("inventory", "tu-2", "t6"),
        tool_result("region: eu-west-1\ninstances: 42", "tu-2", "t7"),
        assistant("Ireland had 42 instances.", "t8"),
        user("Which of the two migrations finished first?", "t9"),
        tool_use("timeline", "tu-3", "t10"),
        tool_result("migration alpha: 2024-03-01\nmigration beta: 2024-05-17", "tu-3", "t11"),
        assistant("Alpha finished first, on 2024-03-01.", "t12"),
        user("Summarise the Ireland instance count again.", "t13"),
    ]


# ---- the identity ---------------------------------------------------------------------------------


def test_expand_threshold_zero_returns_the_received_list_by_identity() -> None:
    messages = conversation()
    matcher = UnusedMatcher()

    projected, state = project(
        messages,
        None,
        matcher,
        None,
        Thresholds(expand_threshold=0.0, collapse_floor=0.0),
    )

    assert projected is messages
    assert matcher.calls == 0
    assert state.choice.full_pass is True


def test_expand_threshold_zero_is_the_identity_on_a_carried_over_state() -> None:
    """The second projection sees a graph full of Cards and must still return its input untouched."""
    messages = conversation()
    thresholds = Thresholds(expand_threshold=0.0, collapse_floor=0.0)

    _, first = project(messages, None, UnusedMatcher(), None, thresholds)
    assert first.cards, "the write half must have derived Cards for the closed turns"

    projected, second = project(messages, first, UnusedMatcher(), None, thresholds)

    assert projected is messages
    assert second.choice.full_pass is True
    assert set(second.cards) == set(first.cards)


def test_a_fresh_graph_with_no_closed_turn_is_the_identity() -> None:
    messages = [user("First question of the conversation.", "t1")]

    projected, state = project(messages, None, TableMatcher({}), None, Thresholds())

    assert projected is messages
    assert state.cards == {}
    assert state.choice.full_pass is True


def test_below_min_cards_the_matcher_is_never_reached() -> None:
    messages = conversation()
    matcher = UnusedMatcher()

    projected, _ = project(messages, None, matcher, None, Thresholds(min_cards=99))

    assert projected is messages
    assert matcher.calls == 0


# ---- the write half -------------------------------------------------------------------------------


def test_the_write_half_derives_one_card_per_closed_turn() -> None:
    messages = conversation()

    _, state = project(messages, None, TableMatcher({}), None, Thresholds())

    # Three closed turns; the trailing user ask is the open turn and has no Card yet.
    assert len(state.cards) == 3
    titles = [card.title for card in sorted(state.cards.values(), key=lambda card: card.turn)]
    assert titles[0].startswith("What did the Frankfurt region cost")
    assert "Summarise the Ireland instance count again." not in state.cards


def test_a_card_holds_numeric_lines_copied_verbatim() -> None:
    messages = conversation()

    _, state = project(messages, None, TableMatcher({}), None, Thresholds())

    numeric = [line for card in state.cards.values() for line in card.numeric_lines]
    assert "total: $1,204.00" in numeric
    assert "instances: 42" in numeric


def test_the_turn_ordinal_advances_once_per_projection() -> None:
    messages = conversation()

    _, first = project(messages, None, TableMatcher({}), None, Thresholds())
    _, second = project(messages, first, TableMatcher({}), None, Thresholds())

    assert second.turn == first.turn + 1
    assert second.retrieval_cycles == 0


# ---- the delivery ---------------------------------------------------------------------------------


def test_a_collapsing_choice_drops_messages_and_folds_a_final_block() -> None:
    messages = conversation()
    # Only the Ireland turn stays relevant; the other two collapse.
    matcher = TableMatcher({"how many instances were running in Ireland": 0.9}, default=0.1)

    projected, state = project(messages, None, matcher, None, Thresholds())

    assert projected is not messages
    assert len(projected) < len(messages)
    assert state.choice.full_pass is False
    folded = projected[-1]
    assert folded["role"] == "user"
    assert "<collapsed_turns>" in folded["content"][-1]["text"]


def test_the_open_turn_is_never_removed() -> None:
    messages = conversation()
    matcher = TableMatcher({"how many instances were running in Ireland": 0.9}, default=0.1)

    projected, _ = project(messages, None, matcher, None, Thresholds())

    identities = {message.get("tracking_id") for message in projected}
    assert current_turn_ids(messages) <= identities


def test_the_fold_carries_the_target_messages_durable_identity() -> None:
    """A folded message that lost its identity would be invisible to the next projection's scan."""
    messages = conversation()
    matcher = TableMatcher({"how many instances were running in Ireland": 0.9}, default=0.1)

    projected, _ = project(messages, None, matcher, None, Thresholds())

    assert projected[-1]["tracking_id"] == "t13"


def test_a_broken_matcher_sends_everything() -> None:
    messages = conversation()

    projected, state = project(messages, None, BrokenMatcher(), None, Thresholds())

    assert projected is messages
    assert state.choice.full_pass is True


def test_no_matcher_sends_everything() -> None:
    messages = conversation()

    projected, state = project(messages, None, None, None, Thresholds())

    assert projected is messages
    assert state.choice.full_pass is True


def test_a_choice_that_collapses_nothing_returns_the_received_list() -> None:
    """Every Card at full on both parts with ``full_pass`` false: the removal asks for nothing.

    The delivery is exercised directly, because the scoring cannot be talked into producing this choice -- a note high
    enough to keep every part at ``full`` also selects every Card, which is the full-pass path one step earlier. The
    claim under test is the delivery's own: an empty removal request returns the received list by identity.
    """
    messages = conversation()
    _, state = project(messages, None, TableMatcher({}), None, Thresholds(expand_threshold=0.0))
    state.choice = TurnChoice(
        MappingProxyType({title: CardChoice("full", "full") for title in state.cards}),
        full_pass=False,
    )

    projected = _deliver(state, messages, Thresholds())

    assert projected is messages


# ---- nothing is mutated ---------------------------------------------------------------------------


def test_the_received_list_and_state_are_never_mutated() -> None:
    messages = conversation()
    before_messages = copy.deepcopy(messages)
    state = _GraphState()
    before_cards, before_turn, before_full_pass = dict(state.cards), state.turn, state.choice.full_pass
    matcher = TableMatcher({"how many instances were running in Ireland": 0.9}, default=0.1)

    projected, new_state = project(messages, state, matcher, None, Thresholds())

    assert messages == before_messages
    assert state.cards == before_cards
    assert state.turn == before_turn
    assert state.choice.full_pass is before_full_pass
    assert new_state is not state
    assert projected is not None


def test_two_projections_of_one_state_are_independent() -> None:
    messages = conversation()
    _, shared = project(messages, None, TableMatcher({}), None, Thresholds(expand_threshold=0.0))

    _, left = project(messages, shared, TableMatcher({}), None, Thresholds())
    _, right = project(messages, shared, TableMatcher({}), None, Thresholds())

    left.cards.clear()
    assert right.cards, "the second projection must not share the first one's card dict"
    assert shared.cards, "the received state must not share either"


def test_body_budget_is_honoured() -> None:
    """A budget of zero tokens leaves no Card in Full Content, so the call collapses everything it can."""
    messages = conversation()
    matcher = TableMatcher({}, default=0.9)

    projected, state = project(messages, None, matcher, 0, Thresholds())

    assert state.choice.full_pass is False
    assert len(projected) < len(messages)
