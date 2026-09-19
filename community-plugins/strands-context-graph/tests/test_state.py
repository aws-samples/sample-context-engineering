"""Unit tests for the per-agent Graph State."""

import gc
import weakref

import pytest

from strands_context_graph.state import Card, CardChoice, Link, ToolPair, TurnChoice, _GraphState


class _FakeAgent:
    """Stand-in for an Agent: the state map only needs a weak-referenceable object as key."""


def test_fresh_state_presents_no_card_no_link_turn_zero() -> None:
    state = _GraphState()

    assert state.cards == {}
    assert state.links == {}
    assert state.turn == 0
    assert state.retrieval_cycles == 0
    assert state.reuse == {}
    assert state.referenced == frozenset()


def test_fresh_state_is_a_full_pass() -> None:
    state = _GraphState()

    assert state.choice.full_pass is True
    assert dict(state.choice.by_title) == {}
    assert state.choice.selected is None


def test_frozen_choice_mapping_rejects_mutation() -> None:
    state = _GraphState()

    with pytest.raises(TypeError):
        state.choice.by_title["anything"] = CardChoice("full", "full")  # type: ignore[index]


def test_state_is_independent_per_agent() -> None:
    states: weakref.WeakKeyDictionary[_FakeAgent, _GraphState] = weakref.WeakKeyDictionary()
    first, second = _FakeAgent(), _FakeAgent()

    states[first] = _GraphState()
    states[second] = _GraphState()
    states[first].turn = 7
    states[first].links["a"] = [Link("follows", "b", 1.0)]

    assert states[second].turn == 0
    assert states[second].links == {}


def test_state_is_discarded_with_its_agent() -> None:
    states: weakref.WeakKeyDictionary[_FakeAgent, _GraphState] = weakref.WeakKeyDictionary()
    agent = _FakeAgent()
    states[agent] = _GraphState()

    del agent
    gc.collect()

    assert len(states) == 0


def test_card_holds_addresses_never_content() -> None:
    card = Card(
        title="Compare the two quotes",
        kind="subject",
        turn=3,
        dialogue_ids=("t1", "t2"),
        evidence_ids=("t3",),
        pairs=(ToolPair("tu-1", "http_request", ("t3", "t4"), consumed=True),),
        tool_names=frozenset({"http_request"}),
        references=("mem_1_tu-1_0",),
        numeric_lines=("total: $1,204.00",),
        tags=("http_request",),
        description="Compare the two quotes | http_request x1",
    )

    assert not hasattr(card, "content")
    assert card.reference is None
    assert card.content_type is None
    assert card.size_bytes is None


def test_cards_and_choices_are_frozen() -> None:
    choice = TurnChoice({"a": CardChoice("full", "full")}, full_pass=False)

    with pytest.raises(AttributeError):
        choice.full_pass = True  # type: ignore[misc]
