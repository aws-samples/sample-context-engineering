"""The two Card retrieval tools of practice D, as host-neutral functions over the graph state.

``expand_card`` and ``find_context`` are ports of the Strands plugin's tool bodies without the framework: every failure
is a returned sentence, never an exception, and a request that found nothing records nothing.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from context_core.graph import GraphState, Thresholds, project
from context_core.graph.retrieval import (
    EXPAND_CARD_DESCRIPTION,
    FIND_CONTEXT_DESCRIPTION,
    expand_card,
    find_context,
)

TOPICS = ("alpha", "bravo", "charlie", "delta", "echo")


class KeywordMatcher:
    """1.0 when the question names the topic a Description is about, 0.1 otherwise."""

    def score(self, question: str, descriptions: Sequence[str]) -> list[float]:
        words = {word.strip(".,:?!").lower() for word in question.split()}
        return [
            1.0 if any(topic in words and topic in text.lower() for topic in TOPICS) else 0.1 for text in descriptions
        ]


def _messages(turns: int) -> list[dict]:
    messages: list[dict] = []
    for index in range(turns):
        topic = TOPICS[index]
        messages += [
            {
                "role": "user",
                "content": [{"text": f"{topic} report: what is the total for {topic}?"}],
                "tracking_id": f"u{index}",
            },
            {
                "role": "assistant",
                "content": [
                    {"text": f"reading {topic}"},
                    {"toolUse": {"toolUseId": f"c{index}", "name": "read_docs", "input": {}}},
                ],
                "tracking_id": f"t{index}",
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": f"c{index}",
                            "status": "success",
                            "content": [{"text": f"{topic} total: {index}00.50"}],
                        }
                    }
                ],
                "tracking_id": f"t{index}",
            },
            {
                "role": "assistant",
                "content": [{"text": f"the {topic} total is {index}00.50"}],
                "tracking_id": f"f{index}",
            },
        ]
    messages.append(
        {"role": "user", "content": [{"text": "alpha again: restate the alpha total"}], "tracking_id": f"u{turns}"}
    )
    return messages


def _state(turns: int = 5) -> GraphState:
    _, state = project(_messages(turns), GraphState(), matcher=KeywordMatcher(), thresholds=Thresholds())
    assert len(state.cards) == turns
    return state


def _title(state: GraphState, topic: str) -> str:
    return next(title for title in state.cards if title.startswith(topic))


def test_the_model_facing_descriptions_name_the_tool_contract():
    assert "titles" in EXPAND_CARD_DESCRIPTION and "ONE call" in EXPAND_CARD_DESCRIPTION
    assert "your words" in FIND_CONTEXT_DESCRIPTION


def test_expand_card_raises_the_resolution_and_feeds_the_note_back():
    state = _state()
    title = _title(state, "bravo")
    answer = expand_card(state, [title], cycle=7, reuse_ttl_cycles=5)
    assert "arrives in full" in answer
    assert state.choice.by_title[title].dialogue == "full" and state.choice.by_title[title].evidence == "full"
    assert state.reuse[title] == (1.0, 12)
    assert state.retrieval_cycles == 1


def test_expand_card_accepts_one_bare_title():
    state = _state()
    assert "arrives in full" in expand_card(state, _title(state, "charlie"), cycle=0, reuse_ttl_cycles=5)


def test_expand_card_names_the_titles_that_matched_nothing_and_records_nothing_for_them():
    state = _state()
    title = _title(state, "bravo")
    answer = expand_card(state, [title, "no such turn"], cycle=0, reuse_ttl_cycles=5)
    assert "arrives in full" in answer and "'no such turn'" in answer
    assert set(state.reuse) == {title}


def test_expand_card_with_nothing_found_changes_nothing_but_still_spends_a_cycle():
    state = _state()
    before = state.choice
    answer = expand_card(state, ["nope"], cycle=0, reuse_ttl_cycles=5)
    assert "no earlier turn" in answer and "find_context" in answer
    assert state.choice is before and state.reuse == {}
    assert state.retrieval_cycles == 1


def test_expand_card_with_no_titles_asks_for_them():
    state = _state()
    assert "no title given" in expand_card(state, [], cycle=0, reuse_ttl_cycles=5)


def test_expand_card_on_a_full_pass_leaves_the_choice_alone():
    _, state = project(_messages(2), GraphState(), matcher=KeywordMatcher(), thresholds=Thresholds())
    before = state.choice
    assert before.full_pass
    expand_card(state, [next(iter(state.cards))], cycle=0, reuse_ttl_cycles=5)
    assert state.choice is before


def test_the_retrieval_budget_refuses_once_spent_and_the_refusal_is_an_instruction():
    state = _state()
    title = _title(state, "bravo")
    for _ in range(2):
        assert "arrives in full" in expand_card(state, [title], cycle=0, reuse_ttl_cycles=5, max_retrieval_cycles=2)
    refusal = expand_card(state, [title], cycle=0, reuse_ttl_cycles=5, max_retrieval_cycles=2)
    assert "already spent its 2 retrieval calls" in refusal and "answer from what" in refusal
    assert state.retrieval_cycles == 2, "a refusal spends nothing"


def test_a_zero_ttl_records_no_fed_back_note():
    state = _state()
    expand_card(state, [_title(state, "bravo")], cycle=0, reuse_ttl_cycles=0)
    assert state.reuse == {}


def test_find_context_answers_the_best_matches_and_feeds_them_back():
    state = _state()
    answer = find_context(state, "bravo", matcher=KeywordMatcher(), collapse_floor=0.45, cycle=3, reuse_ttl_cycles=5)
    assert answer.startswith("find_context | 1 earlier turn(s) match 'bravo'")
    assert f"- title: {_title(state, 'bravo')}" in answer
    assert set(state.reuse) == {_title(state, "bravo")}


def test_find_context_caps_the_candidates_at_five():
    state = _state(5)

    class Everything:
        def score(self, question: str, descriptions: Sequence[str]) -> list[float]:
            return [0.9] * len(descriptions)

    answer = find_context(state, "anything", matcher=Everything(), collapse_floor=0.45, cycle=0, reuse_ttl_cycles=5)
    assert answer.count("- title:") == 5


@pytest.mark.parametrize("need", ["", "   "])
def test_find_context_with_an_empty_need_finds_nothing(need: str):
    state = _state()
    answer = find_context(state, need, matcher=KeywordMatcher(), collapse_floor=0.45, cycle=0, reuse_ttl_cycles=5)
    assert "nothing in this conversation matches" in answer and state.reuse == {}


def test_find_context_without_a_matcher_or_with_a_broken_one_finds_nothing_and_does_not_raise():
    state = _state()

    class WrongLength:
        def score(self, question: str, descriptions: Sequence[str]) -> list[float]:
            return [1.0]

    class Raises:
        def score(self, question: str, descriptions: Sequence[str]) -> list[float]:
            raise RuntimeError("unreachable")

    for matcher in (None, WrongLength(), Raises()):
        answer = find_context(state, "bravo", matcher=matcher, collapse_floor=0.45, cycle=0, reuse_ttl_cycles=5)
        assert "nothing in this conversation matches" in answer
    assert state.reuse == {}


def test_find_context_below_the_floor_returns_nothing():
    state = _state()
    answer = find_context(state, "zulu", matcher=KeywordMatcher(), collapse_floor=0.45, cycle=0, reuse_ttl_cycles=5)
    assert "nothing in this conversation matches 'zulu'" in answer


def test_find_context_narrowed_by_a_tag_that_no_card_carries_finds_nothing():
    state = _state()
    answer = find_context(
        state, "bravo", "no-such-tag", matcher=KeywordMatcher(), collapse_floor=0.45, cycle=0, reuse_ttl_cycles=5
    )
    assert "among the turns tagged 'no-such-tag'" in answer


def test_find_context_obeys_the_same_retrieval_budget_as_expand_card():
    state = _state()
    expand_card(state, [_title(state, "bravo")], cycle=0, reuse_ttl_cycles=5, max_retrieval_cycles=1)
    answer = find_context(
        state,
        "bravo",
        matcher=KeywordMatcher(),
        collapse_floor=0.45,
        cycle=0,
        reuse_ttl_cycles=5,
        max_retrieval_cycles=1,
    )
    assert "already spent its 1 retrieval calls" in answer
