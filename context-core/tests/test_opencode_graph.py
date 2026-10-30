"""Offline tests for Practice D on OpenCode: the transcript conversion, the projection binding and the bridge requests.

No Bedrock call is made. The similarity matcher is the one seam the graph takes from outside, so these tests hand it a
deterministic keyword matcher; the real embedding model is exercised only by the live smoke and the campaign. Everything
else -- the Cards, the choice, the removal and the final block -- is the real core.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from context_core import opencode_graph
from context_core.graph import GraphState, Thresholds, project
from context_core.graph.retrieval import (
    EXPAND_CARD_DESCRIPTION,
    EXPAND_CARD_RESULT_DESCRIPTION,
    FIND_CONTEXT_DESCRIPTION,
)
from context_core.opencode_bridge import BridgeConfig, BridgeConfigError, OpenCodeBridge
from context_core.opencode_graph import RETRIEVAL_TOOLS, GraphConfig, GraphConfigError, OpenCodeGraph, to_neutral

TOPICS = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot")


class KeywordMatcher:
    """Scores 1.0 when the question names the topic a Description is about, 0.1 otherwise. Counts its calls."""

    def __init__(self) -> None:
        self.calls = 0

    def score(self, question: str, descriptions: Sequence[str]) -> list[float]:
        self.calls += 1
        words = {word.strip(".,:?!").lower() for word in question.split()}
        return [
            1.0 if any(topic in words and topic in description.lower() for topic in TOPICS) else 0.1
            for description in descriptions
        ]


class BrokenMatcher:
    """The shape of an unreachable embedding model: an empty answer, never an exception."""

    def score(self, question: str, descriptions: Sequence[str]) -> list[float]:
        return []


def _tool_part(call_id: str, output: str, *, status: str = "completed") -> dict[str, Any]:
    part: dict[str, Any] = {
        "type": "tool",
        "tool": "read_docs",
        "callID": call_id,
        "status": status,
        "input": {"q": call_id},
    }
    if status == "completed":
        part["output"] = output
    elif status == "error":
        part["error"] = output
    return part


def _turn(index: int, question: str | None = None, *, with_answer: bool = True) -> list[dict[str, Any]]:
    """One user turn: the question, an assistant step that calls a tool, and the assistant's final answer."""
    topic = TOPICS[index]
    messages: list[dict[str, Any]] = [
        {
            "id": f"msg_{index}u",
            "role": "user",
            "parts": [{"type": "text", "text": question or f"{topic} report: what is the total for {topic}?"}],
        }
    ]
    if with_answer:
        messages.append(
            {
                "id": f"msg_{index}t",
                "role": "assistant",
                "parts": [
                    {"type": "text", "text": f"Reading the {topic} document."},
                    _tool_part(f"call_{index}", f"{topic} ledger\ntotal: {1000 + index}.50\nitems: {index + 10}"),
                ],
            }
        )
        messages.append(
            {
                "id": f"msg_{index}f",
                "role": "assistant",
                "parts": [
                    {"type": "text", "text": f"The {topic} total is {1000 + index}.50 across {index + 10} items."}
                ],
            }
        )
    return messages


def _transcript(turns: int, *, open_question: str | None = None) -> list[dict[str, Any]]:
    """``turns`` finished turns, then a final user message opening the turn the model is about to answer."""
    messages: list[dict[str, Any]] = []
    for index in range(turns):
        messages.extend(_turn(index))
    messages.extend(_turn(turns, open_question, with_answer=False))
    return messages


def _config(**overrides: Any) -> GraphConfig:
    return GraphConfig(enabled=True, **overrides)


def _graph(matcher: Any = None, **overrides: Any) -> tuple[OpenCodeGraph, Any]:
    chosen = matcher if matcher is not None else KeywordMatcher()
    return OpenCodeGraph(_config(**overrides), matcher_factory=lambda: chosen), chosen


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #


def test_graph_is_off_by_default_and_reads_no_other_variable():
    config = GraphConfig.from_env({"CONTEXT_GRAPH_EXPAND_THRESHOLD": "not-a-number"})
    assert config == GraphConfig()
    assert config.enabled is False


@pytest.mark.parametrize("value", ["on", "1", "true", "YES"])
def test_graph_flag_accepts_on_values(value: str):
    assert GraphConfig.from_env({"CONTEXT_GRAPH": value}).enabled is True


def test_graph_flag_rejects_an_unknown_value():
    with pytest.raises(GraphConfigError):
        GraphConfig.from_env({"CONTEXT_GRAPH": "maybe"})


def test_graph_defaults_are_the_benchmark_tuning():
    config = GraphConfig.from_env({"CONTEXT_GRAPH": "on"})
    assert (config.expand_threshold, config.collapse_floor, config.link_threshold) == (0.62, 0.45, 0.50)
    assert (config.description_tokens, config.body_budget, config.max_retrieval_cycles) == (100, 40_000, 8)
    assert (config.reuse_ttl_cycles, config.tags_per_card, config.min_cards, config.neighbors_per_candidate) == (
        5,
        5,
        3,
        0,
    )
    assert config.rarity_weight == 0.70
    assert config.embed_model_id == "cohere.embed-multilingual-v3"


def test_graph_overrides_and_unbounded_values():
    config = GraphConfig.from_env(
        {
            "CONTEXT_GRAPH": "on",
            "CONTEXT_GRAPH_EXPAND_THRESHOLD": "0.7",
            "CONTEXT_GRAPH_BODY_BUDGET": "none",
            "CONTEXT_GRAPH_MAX_RETRIEVAL_CYCLES": "none",
            "CONTEXT_GRAPH_EMBED_MODEL_ID": "other.model",
        }
    )
    assert config.expand_threshold == 0.7
    assert config.body_budget is None
    assert config.max_retrieval_cycles is None
    assert config.embed_model_id == "other.model"


@pytest.mark.parametrize(
    "env",
    [
        {"CONTEXT_GRAPH_COLLAPSE_FLOOR": "0.9", "CONTEXT_GRAPH_EXPAND_THRESHOLD": "0.5"},
        {"CONTEXT_GRAPH_EXPAND_THRESHOLD": "1.5"},
        {"CONTEXT_GRAPH_MIN_CARDS": "0"},
        {"CONTEXT_GRAPH_DESCRIPTION_TOKENS": "many"},
        {"CONTEXT_GRAPH_REUSE_TTL": "-1"},
    ],
)
def test_graph_rejects_malformed_or_inconsistent_values(env: dict[str, str]):
    with pytest.raises(GraphConfigError):
        GraphConfig.from_env({"CONTEXT_GRAPH": "on", **env})


def test_bridge_config_maps_a_graph_error_to_its_own_error():
    with pytest.raises(BridgeConfigError):
        BridgeConfig.from_env({"AWS_PROFILE": "p", "CONTEXT_GRAPH": "on", "CONTEXT_GRAPH_MIN_CARDS": "0"})


def test_thresholds_name_exactly_the_two_registered_tools():
    assert _config().thresholds().retrieval_tools == ("expand_card", "find_context")


# --------------------------------------------------------------------------- #
# Transcript -> neutral                                                       #
# --------------------------------------------------------------------------- #


def test_every_neutral_message_carries_its_opencode_id_as_tracking_id():
    neutral = to_neutral(_transcript(2))
    assert neutral
    assert all(isinstance(message.get("tracking_id"), str) and message["content"] for message in neutral)


def test_an_assistant_message_with_a_tool_becomes_two_neutral_messages_under_one_id():
    neutral = to_neutral(_turn(0))
    shape = [(message["role"], message["tracking_id"]) for message in neutral]
    assert shape == [("user", "msg_0u"), ("assistant", "msg_0t"), ("user", "msg_0t"), ("assistant", "msg_0f")]
    use = next(block["toolUse"] for block in neutral[1]["content"] if "toolUse" in block)
    result = neutral[2]["content"][0]["toolResult"]
    assert use["toolUseId"] == result["toolUseId"] == "call_0"
    assert result["status"] == "success"
    assert "total: 1000.50" in result["content"][0]["text"]


def test_a_tool_result_message_is_not_a_turn_boundary():
    from context_core.graph import is_turn_boundary

    neutral = to_neutral(_turn(0))
    assert [is_turn_boundary(message) for message in neutral] == [True, False, False, False]


def test_pending_and_running_tool_calls_are_left_out_so_no_pair_is_ever_split():
    transcript = [
        {"id": "m1", "role": "user", "parts": [{"type": "text", "text": "go"}]},
        {
            "id": "m2",
            "role": "assistant",
            "parts": [
                _tool_part("done", "ok"),
                {"type": "tool", "tool": "bash", "callID": "later", "status": "running", "input": {}},
            ],
        },
    ]
    neutral = to_neutral(transcript)
    uses = [block["toolUse"]["toolUseId"] for message in neutral for block in message["content"] if "toolUse" in block]
    results = [
        block["toolResult"]["toolUseId"] for message in neutral for block in message["content"] if "toolResult" in block
    ]
    assert uses == results == ["done"]


def test_an_errored_tool_call_is_an_error_result():
    transcript = [
        {"id": "m1", "role": "user", "parts": [{"type": "text", "text": "go"}]},
        {"id": "m2", "role": "assistant", "parts": [_tool_part("c1", "boom", status="error")]},
    ]
    result = to_neutral(transcript)[2]["content"][0]["toolResult"]
    assert result["status"] == "error"
    assert result["content"] == [{"text": "boom"}]


def test_synthetic_and_ignored_text_is_not_the_users_words_and_a_textless_user_message_opens_no_turn():
    transcript = [
        {"id": "m1", "role": "user", "parts": [{"type": "text", "text": "plumbing", "synthetic": True}]},
        {
            "id": "m2",
            "role": "user",
            "parts": [
                {"type": "text", "text": "real", "ignored": False},
                {"type": "text", "text": "skip", "ignored": True},
            ],
        },
        {"id": "m3", "role": "user", "parts": [{"type": "file", "url": "x"}]},
    ]
    neutral = to_neutral(transcript)
    assert [(message["tracking_id"], message["content"]) for message in neutral] == [("m2", [{"text": "real"}])]


def test_malformed_entries_are_skipped_not_raised():
    assert (
        to_neutral([None, {"id": 3, "role": "user", "parts": []}, {"id": "x", "role": "system", "parts": []}, "text"])
        == []
    )  # type: ignore[list-item]


def test_reasoning_only_assistant_message_has_no_neutral_form():
    transcript = [{"id": "m1", "role": "assistant", "parts": [{"type": "reasoning", "text": "thinking"}]}]
    assert to_neutral(transcript) == []


# --------------------------------------------------------------------------- #
# Projection                                                                  #
# --------------------------------------------------------------------------- #


def _surviving(transcript: list[dict[str, Any]], response: dict[str, Any]) -> list[dict[str, Any]]:
    removed = set(response.get("removed_ids", ()))
    return [message for message in transcript if message["id"] not in removed]


def test_fewer_cards_than_min_cards_is_a_full_pass():
    graph, matcher = _graph()
    response = graph.project("s", _transcript(2, open_question="alpha again"), scoring_allowed=True)
    assert response["operation"] == "unchanged"
    assert response["stats"]["cards"] == 2
    assert matcher.calls == 0, "below min_cards the matcher must not be reached"


def test_off_means_skipped_and_nothing_is_built():
    graph = OpenCodeGraph(GraphConfig(), matcher_factory=lambda: pytest.fail("no matcher while off"))
    assert graph.project("s", _transcript(5), scoring_allowed=True) == {
        "operation": "skipped",
        "reason": "graph_disabled",
    }


def test_collapse_removes_other_topics_and_folds_one_block_naming_the_gap_and_the_tools():
    graph, matcher = _graph()
    transcript = _transcript(5, open_question="alpha follow-up: restate the alpha total")
    response = graph.project("s", transcript, scoring_allowed=True)

    assert response["operation"] == "projected"
    removed = set(response["removed_ids"])
    # The relevant turn keeps its dialogue whole. Its tool exchange goes too: the evidence axis is decided by message
    # order, and a consumed pair (an answer followed it) is carried by its numeric lines in the block instead.
    assert not removed & {"msg_0u", "msg_0f"}
    assert "msg_0t" in removed
    # Every other closed turn loses its dialogue and its tool exchange.
    for other in range(1, 5):
        assert {f"msg_{other}u", f"msg_{other}t", f"msg_{other}f"} <= removed
    # The first user message and the question being answered are never removed.
    assert "msg_0u" not in removed and "msg_5u" not in removed
    block = response["final_block"]
    assert block.startswith("<collapsed_turns>")
    assert "expand_card" in block and "find_context" in block
    assert "expand_artifact" not in block, "OpenCode has no artifact store, so the guidance must not offer the tool"
    assert "bravo report" in block
    assert matcher.calls == 1
    assert response["stats"]["collapsed_cards"] >= 1
    assert response["stats"]["removed"] == len(removed)


def test_no_tool_pair_is_split_by_the_removal():
    graph, _ = _graph()
    transcript = _transcript(5, open_question="alpha follow-up: restate the alpha total")
    response = graph.project("s", transcript, scoring_allowed=True)
    neutral = to_neutral(_surviving(transcript, response))
    uses = {block["toolUse"]["toolUseId"] for m in neutral for block in m["content"] if "toolUse" in block}
    results = {block["toolResult"]["toolUseId"] for m in neutral for block in m["content"] if "toolResult" in block}
    assert uses == results


def test_scoring_needs_admission_without_it_the_graph_never_reaches_the_matcher():
    graph, matcher = _graph()
    response = graph.project("s", _transcript(5, open_question="alpha follow-up"), scoring_allowed=False)
    assert response["operation"] == "unchanged"
    assert matcher.calls == 0
    assert response["stats"]["scored"] is False


def test_the_factory_is_called_once_and_only_when_scoring_is_allowed():
    built: list[int] = []

    def factory() -> KeywordMatcher:
        built.append(1)
        return KeywordMatcher()

    graph = OpenCodeGraph(_config(), matcher_factory=factory)
    graph.project("no-admission", _transcript(5), scoring_allowed=False)
    assert built == []
    graph.project("admitted-1", _transcript(5), scoring_allowed=True)
    graph.project("admitted-2", _transcript(5), scoring_allowed=True)
    assert built == [1], "one matcher serves every session, so its vector cache is shared"


def test_a_matcher_that_cannot_score_is_a_full_pass_and_is_counted_not_hidden():
    graph, _ = _graph(BrokenMatcher())
    response = graph.project("s", _transcript(5, open_question="alpha follow-up"), scoring_allowed=True)
    assert response["operation"] == "unchanged"
    assert graph.status()["totals"]["scoring_unavailable"] == 1


def test_a_factory_that_raises_degrades_to_no_scoring():
    def factory() -> KeywordMatcher:
        raise RuntimeError("no credentials")

    graph = OpenCodeGraph(_config(), matcher_factory=factory)
    assert graph.project("s", _transcript(5), scoring_allowed=True)["operation"] == "unchanged"
    assert graph.project("s", _transcript(5), scoring_allowed=True)["operation"] == "unchanged"


def test_full_pass_at_expand_threshold_zero_is_the_identity():
    graph, matcher = _graph(expand_threshold=0.0, collapse_floor=0.0)
    response = graph.project("s", _transcript(5, open_question="alpha follow-up"), scoring_allowed=True)
    assert response == {"operation": "unchanged", "stats": response["stats"]}
    assert "removed_ids" not in response and matcher.calls == 0


def test_a_failure_inside_the_projection_answers_error_and_leaves_the_host_alone(monkeypatch: pytest.MonkeyPatch):
    graph, _ = _graph()

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("core failure")

    monkeypatch.setattr(opencode_graph, "project", boom)
    response = graph.project("s", _transcript(5), scoring_allowed=True)
    assert response == {"operation": "error", "code": "GRAPH_ERROR"}
    assert graph.status()["totals"]["errors"] == 1


@pytest.mark.parametrize("bad", [None, "text", 3, {"a": 1}])
def test_a_transcript_that_is_not_a_list_is_skipped(bad: Any):
    graph, _ = _graph()
    assert graph.project("s", bad, scoring_allowed=True) == {"operation": "skipped", "reason": "no_messages"}


def test_the_same_transcript_projects_the_same_way_in_a_fresh_process():
    transcript = _transcript(5, open_question="alpha follow-up: restate the alpha total")
    first, _ = _graph()
    second, _ = _graph()
    one = first.project("s", transcript, scoring_allowed=True)
    two = second.project("another-session-id", transcript, scoring_allowed=True)
    assert one["removed_ids"] == two["removed_ids"]
    assert one["final_block"] == two["final_block"]


def test_a_second_call_in_the_same_turn_projects_identically_so_the_prefix_stays_cacheable():
    graph, _ = _graph()
    transcript = _transcript(5, open_question="alpha follow-up: restate the alpha total")
    first = graph.project("s", transcript, scoring_allowed=True)
    # The tool loop grows the open turn; the closed turns' projection must not move.
    transcript += [
        {
            "id": "msg_5t",
            "role": "assistant",
            "parts": [{"type": "text", "text": "checking"}, _tool_part("call_5", "alpha ledger\ntotal: 1000.50")],
        },
    ]
    second = graph.project("s", transcript, scoring_allowed=True)
    assert second["removed_ids"] == first["removed_ids"]
    assert second["final_block"] == first["final_block"]


def test_later_calls_of_a_turn_deliver_the_frozen_choice_and_never_score_again():
    graph, matcher = _graph()
    transcript = _transcript(5, open_question="alpha follow-up: restate the alpha total")
    first = graph.project("s", transcript, scoring_allowed=True)
    turn_after_first = first["stats"]["turn"]
    for step in range(3):
        transcript.append(
            {
                "id": f"msg_loop{step}",
                "role": "assistant",
                "parts": [_tool_part(f"loop{step}", f"alpha ledger\ntotal: {step}")],
            }
        )
        later = graph.project("s", transcript, scoring_allowed=True)
        assert later["stats"]["mode"] == "delivered"
        assert later["removed_ids"] == first["removed_ids"]
        assert later["stats"]["turn"] == turn_after_first + step + 1, "every model call is one cycle of the Note clock"
    assert matcher.calls == 1
    totals = graph.status()["totals"]
    assert (totals["choices"], totals["deliveries"], totals["projections"]) == (1, 3, 4)


def test_a_new_user_turn_takes_a_new_choice():
    graph, matcher = _graph()
    transcript = _transcript(5, open_question="alpha follow-up: restate the alpha total")
    graph.project("s", transcript, scoring_allowed=True)
    transcript += [
        {"id": "msg_5f", "role": "assistant", "parts": [{"type": "text", "text": "done"}]},
        {"id": "msg_6u", "role": "user", "parts": [{"type": "text", "text": "charlie follow-up: the charlie total"}]},
    ]
    response = graph.project("s", transcript, scoring_allowed=True)
    assert response["stats"]["mode"] == "chosen"
    assert matcher.calls == 2
    assert {"msg_2u", "msg_2f"}.isdisjoint(response["removed_ids"]), "the newly relevant turn comes back"


def test_deliver_is_the_delivery_half_of_project_and_never_mutates():
    from context_core.graph import deliver

    neutral = to_neutral(_transcript(5, open_question="alpha follow-up: restate the alpha total"))
    projected, state = project(neutral, GraphState(), matcher=KeywordMatcher(), thresholds=_config().thresholds())
    assert projected is not neutral
    before = (dict(state.cards), state.turn, state.choice)
    again = deliver(neutral, state, _config().thresholds())
    assert again == projected
    assert (dict(state.cards), state.turn, state.choice) == before
    assert deliver(neutral, GraphState(), _config().thresholds()) is neutral


def test_card_ids_are_stable_scoped_to_the_session_and_payload_free():
    graph, _ = _graph()
    transcript = _transcript(5, open_question="alpha follow-up: restate the alpha total")
    graph.project("s1", transcript, scoring_allowed=True)
    graph.project("s2", transcript, scoring_allowed=True)
    first, second = graph.card_sources("s1"), graph.card_sources("s2")
    ids_first, ids_second = {card["card_id"] for card in first}, {card["card_id"] for card in second}
    assert len(first) == len(second) == 5
    assert not ids_first & ids_second, "two sessions must never share a Card id, even for the same words"
    assert all(re.fullmatch(r"c_[0-9a-f]{16}", identity) for identity in ids_first)
    assert [card["sources"] for card in first] == [card["sources"] for card in second]
    assert "alpha report" not in json.dumps(first), "the evidence names Cards by digest, never by the user's words"
    assert graph.card_sources("never-seen") == []


def test_the_sources_of_a_card_are_the_host_message_ids_of_its_turn():
    graph, _ = _graph()
    graph.project("s", _transcript(5, open_question="alpha follow-up"), scoring_allowed=True)
    by_first_source = {card["sources"][0]: card["sources"] for card in graph.card_sources("s")}
    assert by_first_source["msg_1u"] == ["msg_1u", "msg_1f", "msg_1t"]


def test_the_chosen_projection_carries_card_sources_and_a_delivery_does_not():
    graph, _ = _graph()
    transcript = _transcript(5, open_question="alpha follow-up: restate the alpha total")
    chosen = graph.project("s", transcript, scoring_allowed=True)
    assert len(chosen["stats"]["card_sources"]) == 5
    transcript.append({"id": "msg_loop", "role": "assistant", "parts": [_tool_part("loop", "alpha ledger\ntotal: 1")]})
    delivered = graph.project("s", transcript, scoring_allowed=True)
    assert delivered["stats"]["mode"] == "delivered" and "card_sources" not in delivered["stats"]


def _repeating_transcript(repeats: int = 4) -> list[dict[str, Any]]:
    """Turns that open with the same words but ask about different accounts, as the CE-60 scenario does."""
    messages: list[dict[str, Any]] = []
    for index in range(repeats):
        messages.append(
            {
                "id": f"msg_r{index}u",
                "role": "user",
                "parts": [
                    {
                        "type": "text",
                        "text": f"Please review the current state of the main investment portfolio held in account {1000 + index}.",
                    }
                ],
            }
        )
        messages.append(
            {
                "id": f"msg_r{index}t",
                "role": "assistant",
                "parts": [_tool_part(f"rcall_{index}", f"account {1000 + index}\nbalance: {500 + index}.25")],
            }
        )
        messages.append(
            {
                "id": f"msg_r{index}f",
                "role": "assistant",
                "parts": [{"type": "text", "text": f"It is {500 + index}.25."}],
            }
        )
    messages.append(
        {"id": "msg_ropen", "role": "user", "parts": [{"type": "text", "text": "And now something else entirely"}]}
    )
    return messages


def test_by_default_a_repeated_opening_takes_over_the_earlier_turns_card():
    # The reference behaviour, kept as the default: the Card is named by the opening words, so the later turn replaces the earlier.
    graph, _ = _graph()
    graph.project("s", _repeating_transcript(4), scoring_allowed=True)
    cards = graph.card_sources("s")
    assert len(cards) == 1
    assert cards[0]["sources"] == ["msg_r3u", "msg_r3f", "msg_r3t"], "only the last of the four turns has a Card"


def test_with_unique_titles_every_turn_has_its_own_card_and_the_ids_do_not_move():
    graph, _ = _graph(unique_titles=True)
    transcript = _repeating_transcript(4)
    graph.project("s", transcript[:-1] + [transcript[-1]], scoring_allowed=True)
    cards = graph.card_sources("s")
    assert [card["sources"] for card in cards] == [[f"msg_r{i}u", f"msg_r{i}f", f"msg_r{i}t"] for i in range(4)]
    ids_before = [card["card_id"] for card in cards]

    # The conversation goes on: the ids and sources of the turns that were there do not change.
    transcript += [
        {
            "id": "msg_rnext",
            "role": "user",
            "parts": [
                {
                    "type": "text",
                    "text": "Please review the current state of the main investment portfolio held in account 9999.",
                }
            ],
        }
    ]
    graph.project("s", transcript, scoring_allowed=True)
    after = {card["card_id"]: card["sources"] for card in graph.card_sources("s")}
    for card_id_, card in zip(ids_before, cards, strict=True):
        assert after[card_id_] == card["sources"]
    assert len(after) == 5


def test_unique_titles_read_the_second_occurrence_as_a_numbered_title_and_leave_the_first_alone():
    transcript = _repeating_transcript(3)
    plain = to_neutral(transcript)
    marked = to_neutral(transcript, unique_titles=True)
    first_user = [m for m in marked if m["role"] == "user" and m["tracking_id"].endswith("u")]
    assert first_user[0]["content"][0]["text"] == plain[0]["content"][0]["text"], "the first occurrence is untouched"
    assert first_user[1]["content"][0]["text"].startswith("(#2) Please review the current state")
    assert first_user[2]["content"][0]["text"].startswith("(#3) Please review the current state")
    # Only the graph's copy carries the marker: the wire transcript is never edited.
    assert "(#2)" not in json.dumps(transcript)


def _titles_of_block(response: dict[str, Any]) -> list[str]:
    return [line[2:] for line in (response.get("final_block") or "").splitlines() if line.startswith("- ")]


def test_a_numbered_title_is_shown_as_such_and_can_be_asked_back_by_that_name():
    graph, _ = _graph(unique_titles=True)
    transcript = _repeating_transcript(4)
    response = graph.project("s", transcript, scoring_allowed=True)
    titles = _titles_of_block(response)
    numbered = [title for title in titles if title.startswith("(#2) ")]
    assert len(titles) == 4 and len(numbered) == 1, titles

    assert "arrives in full" in graph.expand_card("s", numbered)
    restored = graph.project("s", transcript, scoring_allowed=True)
    assert {"msg_r1u", "msg_r1t", "msg_r1f"}.isdisjoint(restored.get("removed_ids", ())), "the second turn came back"
    assert {"msg_r0t", "msg_r2t"} <= set(restored["removed_ids"]), "and only that one"


def test_unique_titles_is_off_unless_asked_for_and_is_a_flag():
    assert GraphConfig().unique_titles is False
    assert GraphConfig.from_env({"CONTEXT_GRAPH": "on"}).unique_titles is False
    assert GraphConfig.from_env({"CONTEXT_GRAPH": "on", "CONTEXT_GRAPH_UNIQUE_TITLES": "on"}).unique_titles is True
    assert GraphConfig.from_env({"CONTEXT_GRAPH": "on"}).summary()["unique_titles"] is False


def test_sessions_do_not_share_a_graph():
    graph, _ = _graph()
    graph.project("parent", _transcript(5, open_question="alpha follow-up"), scoring_allowed=True)
    child = graph.project("child", _transcript(1, open_question="alpha, one turn only"), scoring_allowed=True)
    assert child["operation"] == "unchanged"
    status = graph.status()["sessions"]
    assert status["parent"]["cards"] == 5 and status["child"]["cards"] == 1


def test_compaction_that_removes_old_messages_rebuilds_instead_of_failing():
    graph, _ = _graph()
    long = _transcript(5, open_question="alpha follow-up: restate the alpha total")
    assert graph.project("s", long, scoring_allowed=True)["operation"] == "projected"

    # OpenCode compacted the session: the first turns are gone and the rest has new shape.
    compacted = long[9:]
    response = graph.project("s", compacted, scoring_allowed=True)
    assert response["operation"] in {"projected", "unchanged"}
    assert graph.status()["sessions"]["s"]["resets"] == 1
    removed = set(response.get("removed_ids", ()))
    assert removed <= {message["id"] for message in compacted}


def test_discard_forgets_the_session():
    graph, _ = _graph()
    graph.project("s", _transcript(5), scoring_allowed=True)
    assert graph.discard("s") is True
    assert graph.discard("s") is False
    assert graph.status()["sessions"] == {}


def test_status_is_payload_free_and_says_nothing_was_written_to_the_host():
    graph, _ = _graph()
    graph.project("s", _transcript(5, open_question="alpha follow-up"), scoring_allowed=True)
    text = json.dumps(graph.status())
    assert "ledger" not in text and "alpha report" not in text
    assert graph.status()["durable_writes"] == 0


# --------------------------------------------------------------------------- #
# The two retrieval tools                                                     #
# --------------------------------------------------------------------------- #


def _collapsed_graph(**overrides: Any) -> tuple[OpenCodeGraph, list[dict[str, Any]], dict[str, Any]]:
    graph, _ = _graph(**overrides)
    transcript = _transcript(5, open_question="alpha follow-up: restate the alpha total")
    response = graph.project("s", transcript, scoring_allowed=True)
    assert response["operation"] == "projected"
    return graph, transcript, response


def test_expand_card_brings_the_turn_back_on_the_next_projection():
    graph, transcript, first = _collapsed_graph()
    assert {"msg_1t", "msg_1f"} & set(first["removed_ids"])
    title = next(line[2:] for line in first["final_block"].splitlines() if line.startswith("- ") and "bravo" in line)

    answer = graph.expand_card("s", [title])
    assert "arrives in full" in answer

    # The next model call of the same turn: the requested turn comes back whole, tool results included.
    second = graph.project("s", transcript, scoring_allowed=True)
    assert not {"msg_1u", "msg_1t", "msg_1f"} & set(second.get("removed_ids", ())), (
        "the requested turn must come back whole"
    )
    assert "bravo report" not in (second.get("final_block") or "")
    assert second["stats"]["mode"] == "delivered"

    # The next user turn recomputes the choice: the request lives on as a Note (the dialogue stays), the tool results
    # return to Description, which is what "for the rest of this turn" promises.
    transcript += [
        {"id": "msg_5f", "role": "assistant", "parts": [{"type": "text", "text": "The alpha total is 1000.50."}]},
        {"id": "msg_6u", "role": "user", "parts": [{"type": "text", "text": "alpha again: and the alpha count?"}]},
    ]
    third = graph.project("s", transcript, scoring_allowed=True)
    assert third["stats"]["mode"] == "chosen"
    removed = set(third["removed_ids"])
    assert not {"msg_1u", "msg_1f"} & removed
    assert "msg_1t" in removed


def test_expand_card_with_an_unknown_title_says_so_and_changes_nothing():
    graph, transcript, first = _collapsed_graph()
    answer = graph.expand_card("s", ["no such turn"])
    assert "no earlier turn" in answer
    assert graph.project("s", transcript, scoring_allowed=True)["removed_ids"] == first["removed_ids"]


def test_expand_card_before_any_projection_has_nothing_to_bring_back():
    graph, _ = _graph()
    assert "nothing to bring back" in graph.expand_card("never-seen", ["x"])


def test_the_retrieval_budget_is_per_user_turn_and_survives_projections_inside_it():
    graph, transcript, first = _collapsed_graph(max_retrieval_cycles=2)
    title = next(line[2:] for line in first["final_block"].splitlines() if line.startswith("- ") and "bravo" in line)

    assert "arrives in full" in graph.expand_card("s", [title])
    graph.project(
        "s", transcript, scoring_allowed=True
    )  # a model call inside the same turn: must not refund the budget
    assert "arrives in full" in graph.expand_card("s", [title])
    refusal = graph.expand_card("s", [title])
    assert "already spent its 2 retrieval calls" in refusal

    # A new user turn gets a fresh budget.
    transcript += [
        {"id": "msg_5t", "role": "assistant", "parts": [{"type": "text", "text": "ok"}]},
        {"id": "msg_6u", "role": "user", "parts": [{"type": "text", "text": "charlie question"}]},
    ]
    graph.project("s", transcript, scoring_allowed=True)
    assert "arrives in full" in graph.expand_card("s", [title])


def test_find_context_needs_admission_to_score_and_never_raises():
    graph, transcript, _ = _collapsed_graph()
    with_scoring = graph.find_context("s", "bravo", scoring_allowed=True)
    assert with_scoring.startswith("find_context |")
    without = graph.find_context("s", "bravo", scoring_allowed=False)
    assert "nothing in this conversation matches" in without


def test_find_context_lists_matching_turns_by_title():
    graph, transcript, _ = _collapsed_graph()
    result = graph.find_context("s", "bravo", scoring_allowed=True)
    assert "bravo report" in result
    assert "call expand_card" in result


# --------------------------------------------------------------------------- #
# Through the bridge                                                          #
# --------------------------------------------------------------------------- #

_AGENT_MODEL = "anthropic/claude-x"
_BASE_ENV = {
    "AWS_PROFILE": "test-profile",
    "AWS_REGION": "us-west-2",
    "CONTEXT_OPENCODE_TOOLS": "read",
    "CONTEXT_EXECUTION_ALLOWED_MODELS": _AGENT_MODEL,
    "CONTEXT_EXECUTION_MODEL_ID": _AGENT_MODEL,
}


def _bridge(**env: str) -> tuple[OpenCodeBridge, KeywordMatcher]:
    bridge = OpenCodeBridge(BridgeConfig.from_env({**_BASE_ENV, **env}))
    matcher = KeywordMatcher()
    bridge._graph._matcher_factory = lambda: matcher
    return bridge, matcher


def _admit(bridge: OpenCodeBridge) -> str:
    response = bridge.handle(
        {
            "type": "admit",
            "id": "adm",
            "execution_mode": "live",
            "approval_id": "appr-1",
            "actor_class": "maintainer",
            "budget_ceiling": 5.0,
            "budget_unit": "USD",
            "nonce": "nonce-graph",
            "preflight_requested": True,
        }
    )
    assert response["operation"] == "admitted"
    return str(response["admission_id"])


def test_a_bridge_with_the_graph_off_answers_skipped_and_startup_is_unchanged():
    bridge, matcher = _bridge()
    started = bridge.handle({"type": "startup", "id": "s"})
    assert "graph" not in started
    response = bridge.handle({"type": "graph_project", "id": "g", "session_id": "s", "messages": _transcript(5)})
    assert response["operation"] == "skipped" and response["reason"] == "graph_disabled"
    assert matcher.calls == 0
    assert bridge.handle({"type": "graph_status", "id": "gs"})["enabled"] is False


def test_startup_publishes_the_graph_tuning_only_when_it_is_on():
    bridge, _ = _bridge(CONTEXT_GRAPH="on")
    started = bridge.handle({"type": "startup", "id": "s"})
    assert started["graph"]["enabled"] is True
    assert started["graph"]["expand_threshold"] == 0.62
    assert started["graph"]["embed_model_id"] == "cohere.embed-multilingual-v3"


def test_graph_project_without_a_live_admission_never_scores():
    bridge, matcher = _bridge(CONTEXT_GRAPH="on")
    response = bridge.handle(
        {
            "type": "graph_project",
            "id": "g",
            "session_id": "s",
            "messages": _transcript(5, open_question="alpha follow-up"),
        }
    )
    assert response["operation"] == "unchanged"
    assert matcher.calls == 0


def test_graph_project_with_a_live_admission_collapses_and_answers_the_ids():
    bridge, matcher = _bridge(CONTEXT_GRAPH="on")
    admission = _admit(bridge)
    response = bridge.handle(
        {
            "type": "graph_project",
            "id": "g",
            "session_id": "s",
            "admission_id": admission,
            "messages": _transcript(5, open_question="alpha follow-up: restate the alpha total"),
        }
    )
    assert response["operation"] == "projected"
    assert response["removed_ids"] and response["final_block"].startswith("<collapsed_turns>")
    assert matcher.calls == 1
    assert json.loads(json.dumps(response)) == response, "the answer must survive the JSONL codec"


def test_an_offline_admission_does_not_open_scoring():
    bridge, matcher = _bridge(CONTEXT_GRAPH="on")
    offline = bridge.handle({"type": "admit", "id": "a", "execution_mode": "offline"})
    response = bridge.handle(
        {
            "type": "graph_project",
            "id": "g",
            "session_id": "s",
            "admission_id": offline["admission_id"],
            "messages": _transcript(5, open_question="alpha follow-up"),
        }
    )
    assert response["operation"] == "unchanged" and matcher.calls == 0


@pytest.mark.parametrize(
    "request_extra",
    [{}, {"session_id": ""}, {"session_id": "s", "messages": "nope"}],
)
def test_graph_project_with_an_unusable_request_is_skipped(request_extra: dict[str, Any]):
    bridge, _ = _bridge(CONTEXT_GRAPH="on")
    response = bridge.handle({"type": "graph_project", "id": "g", **request_extra})
    assert response["operation"] == "skipped"


def test_the_two_tools_answer_text_through_the_bridge_and_shutdown_forgets_the_graph():
    bridge, _ = _bridge(CONTEXT_GRAPH="on")
    admission = _admit(bridge)
    projected = bridge.handle(
        {
            "type": "graph_project",
            "id": "g",
            "session_id": "s",
            "admission_id": admission,
            "messages": _transcript(5, open_question="alpha follow-up: restate the alpha total"),
        }
    )
    title = next(
        line[2:] for line in projected["final_block"].splitlines() if line.startswith("- ") and "bravo" in line
    )

    expanded = bridge.handle({"type": "graph_expand_card", "id": "e", "session_id": "s", "titles": [title]})
    assert expanded["operation"] == "graph_result" and "arrives in full" in expanded["text"]
    found = bridge.handle(
        {"type": "graph_find_context", "id": "f", "session_id": "s", "admission_id": admission, "need": "bravo"}
    )
    assert found["operation"] == "graph_result" and "bravo report" in found["text"]

    status = bridge.handle({"type": "graph_status", "id": "gs"})
    assert status["totals"]["engaged"] == 1 and status["totals"]["expand_calls"] == 1 and status["durable_writes"] == 0

    assert bridge.handle({"type": "shutdown", "id": "sd", "session_id": "s"})["operation"] == "discarded"
    assert bridge.handle({"type": "graph_status", "id": "gs2"})["sessions"] == {}


def test_a_tool_title_that_is_not_a_string_is_ignored():
    bridge, _ = _bridge(CONTEXT_GRAPH="on")
    response = bridge.handle({"type": "graph_expand_card", "id": "e", "session_id": "s", "titles": [1, None, "x"]})
    assert response["operation"] == "graph_result"


def test_the_plugin_never_needs_the_bridge_to_hold_a_credential_for_the_graph_off_path():
    # Constructing the bridge with the graph on resolves no AWS configuration: the matcher is built on first scoring.
    bridge = OpenCodeBridge(BridgeConfig.from_env({**_BASE_ENV, "CONTEXT_GRAPH": "on"}))
    assert bridge._graph._matcher is None


def test_core_projection_is_still_the_identity_on_the_neutral_list_it_was_given():
    neutral = to_neutral(_transcript(5))
    projected, state = project(neutral, GraphState(), matcher=None, thresholds=Thresholds())
    assert projected is neutral
    assert len(state.cards) == 5


# --------------------------------------------------------------------------- #
# The TypeScript side must say what the Python side says                      #
# --------------------------------------------------------------------------- #

_GRAPH_TS = Path(__file__).resolve().parents[2] / ".opencode" / "lib" / "graph.ts"


def _ts_string_constant(source: str, name: str) -> str:
    match = re.search(rf"export const {name} =\s*(.*?);\n", source, re.S)
    assert match, f"{name} not found in graph.ts"
    return "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', match.group(1)))


def test_the_plugin_offers_the_model_the_same_tool_descriptions_the_core_publishes():
    source = _GRAPH_TS.read_text()
    assert _ts_string_constant(source, "EXPAND_CARD_DESCRIPTION") == EXPAND_CARD_DESCRIPTION
    assert _ts_string_constant(source, "EXPAND_CARD_RESULT_DESCRIPTION") == EXPAND_CARD_RESULT_DESCRIPTION
    assert _ts_string_constant(source, "FIND_CONTEXT_DESCRIPTION") == FIND_CONTEXT_DESCRIPTION


def test_the_plugin_registers_the_tool_names_the_final_block_guidance_names():
    source = _GRAPH_TS.read_text()
    names = (_ts_string_constant(source, "EXPAND_CARD_TOOL"), _ts_string_constant(source, "FIND_CONTEXT_TOOL"))
    assert names == RETRIEVAL_TOOLS
