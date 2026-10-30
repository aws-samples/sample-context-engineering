"""Offline tests for the ``inplace`` policy of Practice D on OpenCode.

``inplace`` collapses each old closed turn where it is: the assistant side of the turn becomes one short message made of the
turn's Card, the user's question stays as written, and the decision is a pure function of the transcript. These tests pin the
properties that make it cheap behind a prefix cache and safe to apply: the plan only ever grows at its end, it needs no
score and no network, it leaves a turn whole whenever anything about the turn is in doubt, and a recall changes nothing about
the request.

No Bedrock call is made. The matcher is only ever reached by ``find_context``, and these tests hand it a keyword matcher.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest
from context_core.graph.retrieval import EXPAND_CARD_RESULT_DESCRIPTION
from context_core.opencode_bridge import BridgeConfig, BridgeConfigError, OpenCodeBridge
from context_core.opencode_graph import (
    COLLAPSE_NOTE,
    RETRIEVAL_TOOLS,
    GraphConfig,
    GraphConfigError,
    OpenCodeGraph,
    plan_inplace,
    to_neutral,
    wire_turn_ranges,
)

TOPICS = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel")


class KeywordMatcher:
    """Scores 1.0 when the need names the topic a Description is about, 0.1 otherwise. Counts its calls."""

    def __init__(self) -> None:
        self.calls = 0

    def score(self, question: str, descriptions: Sequence[str]) -> list[float]:
        self.calls += 1
        words = {word.strip(".,:?!").lower() for word in question.split()}
        return [
            1.0 if any(topic in words and topic in description.lower() for topic in TOPICS) else 0.1
            for description in descriptions
        ]


def _tool(call_id: str, output: str, *, name: str = "read_docs") -> dict[str, Any]:
    return {
        "type": "tool",
        "tool": name,
        "callID": call_id,
        "status": "completed",
        "input": {"q": call_id},
        "output": output,
    }


def _question(index: int) -> str:
    return f"{TOPICS[index]} report: what is the total for {TOPICS[index]}?"


def _turn(index: int, *, answered: bool = True, question: str | None = None) -> list[dict[str, Any]]:
    """A user turn: the question, one assistant step that reads a document, and the assistant's final answer."""
    topic = TOPICS[index]
    messages: list[dict[str, Any]] = [
        {"id": f"u{index}", "role": "user", "parts": [{"type": "text", "text": question or _question(index)}]}
    ]
    if answered:
        messages.append(
            {
                "id": f"a{index}",
                "role": "assistant",
                "parts": [
                    {"type": "text", "text": f"Reading the {topic} document."},
                    _tool(f"call_{index}", f"{topic} ledger\ntotal: {1000 + index}.50\nitems: {index + 10}"),
                ],
            }
        )
        messages.append(
            {
                "id": f"f{index}",
                "role": "assistant",
                "parts": [
                    {"type": "text", "text": f"The {topic} total is {1000 + index}.50 across {index + 10} items."}
                ],
            }
        )
    return messages


_UNANSWERED = {"id": "uX", "role": "user", "parts": [{"type": "text", "text": "a question nobody answered"}]}


def _transcript(closed: int) -> list[dict[str, Any]]:
    """``closed`` finished turns, then the user message that opens the turn the model is about to answer."""
    messages: list[dict[str, Any]] = []
    for index in range(closed):
        messages.extend(_turn(index))
    messages.extend(_turn(closed, answered=False))
    return messages


def _graph(*, matcher: Any = None, **overrides: Any) -> tuple[OpenCodeGraph, Any]:
    """A graph under the whole-turn policy (keep the newest closed turn as it was, no dialogue tier) unless told otherwise."""
    chosen = matcher if matcher is not None else KeywordMatcher()
    settings = {"keep_recent": 1, "keep_dialogue": 0, **overrides}
    config = GraphConfig(enabled=True, mode="inplace", **settings)
    return OpenCodeGraph(config, matcher_factory=lambda: chosen), chosen


def _plan(response: dict[str, Any]) -> list[tuple[str, str]]:
    return [(item["at_id"], item["text"]) for item in response.get("collapses", [])]


def _sent(prefix: list[dict[str, Any]], response: dict[str, Any]) -> list[dict[str, Any]]:
    """What the plugin sends the model for ``prefix`` once it has applied ``response``: summaries where the turns were."""
    removed = set(response.get("removed_ids", ()))
    summaries = {item["at_id"]: item["text"] for item in response.get("collapses", [])}
    sent: list[dict[str, Any]] = []
    for message in prefix:
        if message["id"] not in removed:
            sent.append(message)
        elif message["id"] in summaries:
            parts = [{"type": "text", "text": summaries[message["id"]]}]
            sent.append({"id": f"summary-of-{message['id']}", "role": "assistant", "parts": parts})
    return sent


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #


def test_the_default_policy_is_the_reference_one_and_inplace_is_opt_in():
    config = GraphConfig.from_env({"CONTEXT_GRAPH": "on"})
    assert (config.mode, config.keep_recent, config.keep_dialogue, config.recall_max_chars) == (
        "reference",
        0,
        0,
        40_000,
    )
    assert config.names_cards_by_turn is False

    chosen = GraphConfig.from_env(
        {
            "CONTEXT_GRAPH": "on",
            "CONTEXT_GRAPH_MODE": "inplace",
            "CONTEXT_GRAPH_KEEP_RECENT": "2",
            "CONTEXT_GRAPH_KEEP_DIALOGUE": "3",
            "CONTEXT_GRAPH_RECALL_MAX_CHARS": "5000",
        }
    )
    assert (chosen.mode, chosen.keep_recent, chosen.keep_dialogue, chosen.recall_max_chars) == ("inplace", 2, 3, 5000)
    assert chosen.names_cards_by_turn is True, "in place every turn needs a Card of its own"
    assert chosen.summary()["mode"] == "inplace" and chosen.summary()["keep_recent"] == 2
    assert chosen.summary()["keep_dialogue"] == 3


@pytest.mark.parametrize(
    "env",
    [
        {"CONTEXT_GRAPH_MODE": "sticky"},
        {"CONTEXT_GRAPH_KEEP_RECENT": "-1"},
        {"CONTEXT_GRAPH_KEEP_RECENT": "one"},
        {"CONTEXT_GRAPH_KEEP_DIALOGUE": "-1"},
        {"CONTEXT_GRAPH_RECALL_MAX_CHARS": "10"},
    ],
)
def test_malformed_inplace_settings_are_refused_by_name(env: dict[str, str]):
    with pytest.raises(GraphConfigError):
        GraphConfig.from_env({"CONTEXT_GRAPH": "on", **env})


def test_inplace_settings_are_not_read_while_the_graph_is_off():
    assert GraphConfig.from_env({"CONTEXT_GRAPH_MODE": "sticky"}) == GraphConfig()


def test_the_bridge_refuses_a_bad_mode_with_its_own_error():
    with pytest.raises(BridgeConfigError):
        BridgeConfig.from_env({"AWS_PROFILE": "p", "CONTEXT_GRAPH": "on", "CONTEXT_GRAPH_MODE": "sticky"})


# --------------------------------------------------------------------------- #
# The plan                                                                    #
# --------------------------------------------------------------------------- #


def test_wire_turns_are_the_turns_the_graph_derives_cards_for():
    transcript = _transcript(3)
    assert wire_turn_ranges(transcript) == ((0, 3), (3, 6), (6, 9), (9, 10))

    transcript.insert(
        1, {"id": "plumbing", "role": "user", "parts": [{"type": "text", "text": "x", "synthetic": True}]}
    )
    assert [start for start, _ in wire_turn_ranges(transcript)] == [0, 4, 7, 10], "synthetic text opens no turn"
    assert wire_turn_ranges([]) == ()
    assert wire_turn_ranges([{"id": "a", "role": "assistant", "parts": []}]) == ()


@pytest.mark.parametrize(("keep", "collapsed"), [(0, 5), (1, 4), (2, 3), (4, 1), (5, 0), (10**6, 0)])
def test_every_closed_turn_but_the_last_keep_recent_collapses(keep: int, collapsed: int):
    graph, _ = _graph(keep_recent=keep)
    response = graph.project("s", _transcript(5), scoring_allowed=True)

    if collapsed == 0:
        assert response["operation"] == "unchanged"
        return
    assert response["operation"] == "projected"
    assert [item["at_id"] for item in response["collapses"]] == [f"a{index}" for index in range(collapsed)]
    assert response["removed_ids"] == [
        identity for index in range(collapsed) for identity in (f"a{index}", f"f{index}")
    ]
    assert response["final_block"] is None, "nothing is appended at the end of the request"


def test_a_conversation_with_one_closed_turn_is_left_alone_when_one_turn_is_kept():
    graph, _ = _graph()
    assert graph.project("s", _transcript(1), scoring_allowed=True)["operation"] == "unchanged"
    assert graph.project("s", _transcript(2), scoring_allowed=True)["collapses"][0]["at_id"] == "a0"


def test_the_users_questions_and_the_turn_in_progress_are_never_removed():
    graph, _ = _graph(keep_recent=0)
    transcript = _transcript(5)
    response = graph.project("s", transcript, scoring_allowed=True)
    removed = set(response["removed_ids"])
    assert not {message["id"] for message in transcript if message["role"] == "user"} & removed
    assert "u5" not in removed and len(removed) == 10


def test_a_summary_is_the_cards_description_with_its_numbers_copied_literally():
    graph, _ = _graph()
    response = graph.project("s", _transcript(3), scoring_allowed=True)
    first, second = (item["text"] for item in response["collapses"])

    assert first.startswith(COLLAPSE_NOTE)
    assert (
        first.endswith("</collapsed_turn>") and "<collapsed_turn>\nalpha report: what is the total for alpha?" in first
    )
    for line in ("tools: read_docs (1)", "total: 1000.50", "items: 10", "The alpha total is 1000.50 across 10 items."):
        assert line in first
    assert COLLAPSE_NOTE not in second, "the explanation is said once, in the earliest summary"
    assert second.startswith("<collapsed_turn>\nbravo report") and "total: 1001.50" in second
    assert "Reading the alpha document" not in first, "the assistant's narration is not part of the summary"


def test_no_score_no_network_and_no_matcher_while_projecting():
    graph = OpenCodeGraph(
        GraphConfig(enabled=True, mode="inplace"), matcher_factory=lambda: pytest.fail("a matcher was built")
    )
    for turns in range(1, 7):
        graph.project("s", _transcript(turns), scoring_allowed=True)
    status = graph.status()
    assert status["matcher_ready"] is False and status["durable_writes"] == 0
    assert status["totals"]["scoring_unavailable"] == 0


def test_the_plan_is_a_pure_function_of_the_transcript():
    transcript = _transcript(6)
    answers = []
    for _ in range(2):
        graph, _ = _graph()
        answers.append(graph.project("s", transcript, scoring_allowed=False))
    assert answers[0] == answers[1]

    # A process that inherited the conversation (a restart) rebuilds the Cards from the messages and says the same thing.
    warm, _ = _graph()
    for turns in range(1, 7):
        last = warm.project("s", _transcript(turns), scoring_allowed=True)
    assert _plan(last) == _plan(answers[0]) and last["removed_ids"] == answers[0]["removed_ids"]


def test_the_plan_only_grows_at_its_end_so_the_start_of_the_request_never_changes():
    graph, _ = _graph()
    previous: list[tuple[str, str]] = []
    for turns in range(1, 8):
        current = _plan(graph.project("s", _transcript(turns), scoring_allowed=True))
        assert current[: len(previous)] == previous, f"the collapses already made changed at {turns} closed turns"
        assert len(current) >= len(previous)
        previous = current
    assert len(previous) == 6


def test_later_model_calls_of_a_turn_say_the_same_thing_and_reuse_the_cards():
    graph, _ = _graph()
    transcript = _transcript(4)
    first = graph.project("s", transcript, scoring_allowed=True)
    assert first["stats"]["mode"] == "chosen" and "card_sources" in first["stats"]

    # The tool loop of the open turn: the model called a tool, and the host asks again.
    transcript.append(
        {"id": "a4", "role": "assistant", "parts": [{"type": "text", "text": "Looking."}, _tool("call_4", "total: 5")]}
    )
    second = graph.project("s", transcript, scoring_allowed=True)
    assert second["stats"]["mode"] == "delivered" and "card_sources" not in second["stats"]
    assert _plan(second) == _plan(first) and second["removed_ids"] == first["removed_ids"]
    assert graph.status()["totals"]["choices"] == 1 and graph.status()["totals"]["deliveries"] == 1


def test_the_answer_is_plain_data_the_jsonl_codec_carries():
    graph, _ = _graph()
    response = graph.project("s", _transcript(4), scoring_allowed=True)
    assert json.loads(json.dumps(response)) == response
    assert set(response) == {"operation", "removed_ids", "final_block", "collapses", "stats"}
    assert response["stats"]["policy"] == "inplace" and response["stats"]["collapsed_cards"] == 3
    assert response["stats"]["scored"] is False and response["stats"]["choice_full_pass"] is False


def test_a_repeated_opening_gets_a_card_and_a_summary_of_its_own_per_turn():
    transcript: list[dict[str, Any]] = []
    for index in range(4):
        transcript += _turn(index, question=f"Please summarise the portfolio for account number {index}")
    transcript += _turn(4, answered=False, question="Please summarise the portfolio for account number 4")

    graph, _ = _graph()
    response = graph.project("s", transcript, scoring_allowed=True)
    assert [item["at_id"] for item in response["collapses"]] == ["a0", "a1", "a2"]
    assert len({card["card_id"] for card in response["stats"]["card_sources"]}) == 4
    bodies = [item["text"].split("<collapsed_turn>\n", 1)[1] for item in response["collapses"]]
    assert [body.split("\n", 1)[0] for body in bodies] == [
        "Please summarise the portfolio for account",
        "(#2) Please summarise the portfolio for account",
        "(#3) Please summarise the portfolio for account",
    ]


# --------------------------------------------------------------------------- #
# A turn collapses only when nothing about it is in doubt                     #
# --------------------------------------------------------------------------- #


def test_a_turn_with_a_user_role_message_inside_it_stays_whole():
    transcript = _transcript(4)
    plumbing = {"id": "x1", "role": "user", "parts": [{"type": "text", "text": "reminder", "synthetic": True}]}
    transcript.insert(4, plumbing)  # inside turn 1, between its question and the assistant's step
    graph, _ = _graph()
    response = graph.project("s", transcript, scoring_allowed=True)
    assert [item["at_id"] for item in response["collapses"]] == ["a0", "a2"], "turn 1 holds a message not understood"
    assert not {"a1", "f1", "x1"} & set(response["removed_ids"])


def test_a_turn_the_host_never_answered_stays_whole():
    transcript = [_UNANSWERED, *_transcript(3)]
    graph, _ = _graph()
    response = graph.project("s", transcript, scoring_allowed=True)
    assert [item["at_id"] for item in response["collapses"]] == ["a0", "a1"], (
        "the unanswered turn has nothing to replace"
    )
    assert "uX" not in response["removed_ids"]


def test_a_compaction_summary_is_never_collapsed():
    transcript = _transcript(4)
    transcript[2]["summary"] = True  # the host's own summary of what came before: the only copy of it
    graph, _ = _graph()
    response = graph.project("s", transcript, scoring_allowed=True)
    assert [item["at_id"] for item in response["collapses"]] == ["a1", "a2"]
    assert not {"a0", "f0"} & set(response["removed_ids"])


def test_a_turn_with_a_repeated_or_missing_message_id_stays_whole():
    for broken in ({"id": "a0"}, {"id": ""}):
        transcript = _transcript(4)
        transcript[2] = {**transcript[2], **broken}  # the answer of turn 0 now repeats a0's id, or has none
        graph, _ = _graph()
        response = graph.project("s", transcript, scoring_allowed=True)
        assert [item["at_id"] for item in response["collapses"]] == ["a1", "a2"]
        assert "" not in response["removed_ids"]
        assert len(response["removed_ids"]) == len(set(response["removed_ids"]))


def test_an_ineligible_turn_still_counts_towards_the_turns_kept():
    transcript = [_UNANSWERED, *_transcript(2)]  # closed: the unanswered turn, 0 and 1 -- then the open turn
    graph, _ = _graph(keep_recent=2)
    response = graph.project("s", transcript, scoring_allowed=True)
    assert response["operation"] == "unchanged", "the unanswered turn cannot collapse, and 0 and 1 are the two kept"


def test_the_plan_names_only_turns_whose_cards_point_inside_their_own_range():
    graph, _ = _graph()
    transcript = _transcript(4)
    graph.project("s", transcript, scoring_allowed=True)
    cards = graph._sessions["s"].state.cards  # type: ignore[union-attr]
    # A stale Card (one that points at a message outside the turn) is no reason to rewrite the turn.
    stale = {title: card for title, card in cards.items()}
    first = next(iter(stale))
    from dataclasses import replace

    stale[first] = replace(stale[first], evidence_ids=(*stale[first].evidence_ids, "elsewhere"))
    plan = plan_inplace(transcript, stale, 1)
    assert [item.at_id for item in plan] == ["a1", "a2"]


# --------------------------------------------------------------------------- #
# The retrieval exchanges are not part of the turn they were made in          #
# --------------------------------------------------------------------------- #


def test_what_a_recall_returned_does_not_describe_the_turn_that_asked_for_it():
    recall = {"id": "r2", "role": "assistant", "parts": [_tool("call_r", "total: 777.00", name="expand_card")]}
    transcript = _transcript(4)
    transcript[8:8] = [recall]  # inside turn 2: after its tool step, before its answer

    assert RETRIEVAL_TOOLS == ("expand_card", "find_context")
    plain = to_neutral(transcript)
    skipped = to_neutral(transcript, skip_tools=RETRIEVAL_TOOLS)
    assert any(message["tracking_id"] == "r2" for message in plain)
    assert all(message["tracking_id"] != "r2" for message in skipped)

    graph, _ = _graph(keep_recent=0)
    response = graph.project("s", transcript, scoring_allowed=True)
    summaries = [item["text"] for item in response["collapses"]]
    assert not any("777.00" in text for text in summaries)
    assert "r2" in response["removed_ids"], "the exchange still leaves the request with the rest of its turn"


# --------------------------------------------------------------------------- #
# Recall                                                                      #
# --------------------------------------------------------------------------- #


def _projected(**overrides: Any) -> tuple[OpenCodeGraph, list[dict[str, Any]], dict[str, Any]]:
    graph, _ = _graph(**overrides)
    transcript = _transcript(4)
    return graph, transcript, graph.project("s", transcript, scoring_allowed=True)


def test_expand_card_returns_the_content_of_the_collapsed_turn_as_its_result():
    graph, _, _ = _projected()
    answer = graph.expand_card("s", [_question(0)])
    assert answer.startswith("expand_card |") and f"'{_question(0)}' below, in full" in answer
    for text in (
        "Reading the alpha document.",
        '[tool read_docs {"q":"call_0"}]',
        "alpha ledger\ntotal: 1000.50\nitems: 10",
        "The alpha total is 1000.50 across 10 items.",
    ):
        assert text in answer
    assert "bravo" not in answer, "only the turn that was asked for"


def test_a_recall_changes_nothing_about_the_request():
    graph, transcript, first = _projected()
    graph.expand_card("s", [_question(0), _question(1)])
    again = graph.project("s", transcript, scoring_allowed=True)
    assert _plan(again) == _plan(first) and again["removed_ids"] == first["removed_ids"]
    assert again["final_block"] is None


def test_several_titles_come_back_in_one_result_in_the_order_asked():
    graph, _, _ = _projected()
    answer = graph.expand_card("s", [_question(1), _question(0)])
    assert answer.index("=== " + _question(1)) < answer.index("=== " + _question(0))
    assert "total: 1001.50" in answer and "total: 1000.50" in answer


def test_a_title_copied_with_quotes_still_finds_its_turn():
    graph, _, _ = _projected()
    assert "total: 1000.50" in graph.expand_card("s", [f' "{_question(0)}" '])


def test_a_turn_that_is_still_whole_in_the_request_is_not_repeated():
    graph, _, _ = _projected()  # turns 0, 1 and 2 are collapsed; turn 3, the last closed one, is kept
    answer = graph.expand_card("s", [_question(3)])
    assert "is still in the conversation above in full" in answer
    assert "total: 1003.50" not in answer


def test_a_mix_of_collapsed_whole_and_unknown_titles_answers_each():
    graph, _, _ = _projected()
    answer = graph.expand_card("s", [_question(0), _question(3), "no such turn"])
    assert "total: 1000.50" in answer
    assert "is still in the conversation above in full, so nothing was added for it" in answer
    assert "no turn is titled 'no such turn'" in answer


def test_an_unknown_title_says_so_and_a_blank_request_asks_for_titles():
    graph, _, _ = _projected()
    assert "no earlier turn of this conversation is titled 'nope'" in graph.expand_card("s", ["nope"])
    assert "no title given" in graph.expand_card("s", [])


def test_expand_card_before_any_projection_has_nothing_to_bring_back():
    graph, _ = _graph()
    assert "nothing to bring back" in graph.expand_card("never-seen", ["x"])


def test_a_recall_is_capped_and_says_what_it_left_out():
    transcript = _transcript(4)
    transcript[1]["parts"][1]["output"] = "x" * 20_000 + "\ntotal: 9"  # the tool result of turn 0
    graph, _ = _graph(recall_max_chars=2_000)
    graph.project("s", transcript, scoring_allowed=True)

    answer = graph.expand_card("s", [_question(0)])
    assert len(answer) < 3_000
    assert "more characters of this turn were left out" in answer

    both = graph.expand_card("s", [_question(0), _question(1)])
    assert len(both) < 3_000, "the cap is shared by the turns asked for, not given to each"
    assert "total: 1001.50" in both, "a short turn is returned whole next to a long one"


def test_the_retrieval_budget_is_per_user_turn_and_shared_by_both_tools():
    graph, transcript, _ = _projected(max_retrieval_cycles=2)
    assert "below, in full" in graph.expand_card("s", [_question(0)])
    graph.project("s", transcript, scoring_allowed=True)  # another model call of the same turn: no refund
    assert graph.find_context("s", "alpha", scoring_allowed=True).startswith("find_context |")
    refusal = graph.expand_card("s", [_question(0)])
    assert "already spent its 2 retrieval calls" in refusal

    transcript += [
        {"id": "f4", "role": "assistant", "parts": [{"type": "text", "text": "ok"}]},
        {"id": "u5", "role": "user", "parts": [{"type": "text", "text": "charlie again"}]},
    ]
    graph.project("s", transcript, scoring_allowed=True)
    assert "below, in full" in graph.expand_card("s", [_question(0)]), "a new user turn has a new budget"


def test_find_context_still_searches_the_descriptions_and_needs_admission_to_score():
    graph, matcher = _graph()
    graph.project("s", _transcript(4), scoring_allowed=True)
    assert matcher.calls == 0

    found = graph.find_context("s", "bravo", scoring_allowed=True)
    assert "bravo report" in found and "call expand_card" in found and matcher.calls == 1
    assert "nothing in this conversation matches" in graph.find_context("s", "bravo", scoring_allowed=False)


def test_the_graph_forgets_a_session_that_is_discarded_and_a_recall_then_has_nothing():
    graph, _, _ = _projected()
    assert graph.discard("s") is True
    assert "nothing to bring back" in graph.expand_card("s", [_question(0)])


# --------------------------------------------------------------------------- #
# The host compacted or reverted the session                                  #
# --------------------------------------------------------------------------- #


def test_when_the_host_drops_early_messages_the_graph_starts_over_from_what_is_left():
    graph, _ = _graph()
    graph.project("s", _transcript(5), scoring_allowed=True)
    remaining = _transcript(5)[6:]  # the first two turns are gone
    response = graph.project("s", remaining, scoring_allowed=True)

    assert response["stats"]["resets"] == 1
    present = {message["id"] for message in remaining}
    assert set(response["removed_ids"]) <= present
    assert [item["at_id"] for item in response["collapses"]] == ["a2", "a3"]
    assert COLLAPSE_NOTE in response["collapses"][0]["text"], "the note moves to what is now the earliest summary"
    assert "no earlier turn" in graph.expand_card("s", [_question(0)]), "the Card of a turn that is gone is gone"


# --------------------------------------------------------------------------- #
# Counters, and the other policy is untouched                                 #
# --------------------------------------------------------------------------- #


def test_the_counters_say_what_was_done_and_never_what_was_seen():
    graph, _ = _graph()
    transcript = _transcript(4)
    graph.project("s", transcript, scoring_allowed=True)
    graph.expand_card("s", [_question(0)])
    status = graph.status()
    assert status["config"]["mode"] == "inplace"
    totals = status["totals"]
    assert totals["engaged"] == 1 and totals["removed_messages"] == 6 and totals["expand_calls"] == 1
    assert totals["block_chars"] > 0 and status["sessions"]["s"]["cards"] == 4
    assert "alpha" not in json.dumps(status)


def test_the_reference_policy_still_folds_a_block_and_never_returns_collapses():
    transcript = _transcript(5)
    transcript[-1]["parts"][0]["text"] = "alpha follow-up: restate the alpha total"
    matcher = KeywordMatcher()
    graph = OpenCodeGraph(GraphConfig(enabled=True), matcher_factory=lambda: matcher)
    response = graph.project("s", transcript, scoring_allowed=True)
    assert response["operation"] == "projected" and "collapses" not in response
    assert response["final_block"].startswith("<collapsed_turns>") and matcher.calls == 1


def test_the_recall_description_differs_from_the_reference_one_only_in_where_the_content_arrives():
    from context_core.graph.retrieval import EXPAND_CARD_DESCRIPTION

    assert "arrive in the result of this call" in EXPAND_CARD_RESULT_DESCRIPTION
    assert "for the rest of this turn" in EXPAND_CARD_DESCRIPTION
    assert "for the rest of this turn" not in EXPAND_CARD_RESULT_DESCRIPTION
    shared = "Ask for every turn you need in ONE call"
    assert shared in EXPAND_CARD_DESCRIPTION and shared in EXPAND_CARD_RESULT_DESCRIPTION


# --------------------------------------------------------------------------- #
# Through the bridge                                                          #
# --------------------------------------------------------------------------- #

_BASE_ENV = {
    "AWS_PROFILE": "test-profile",
    "AWS_REGION": "us-west-2",
    "CONTEXT_OPENCODE_TOOLS": "read",
    "CONTEXT_EXECUTION_ALLOWED_MODELS": "anthropic/claude-x",
    "CONTEXT_EXECUTION_MODEL_ID": "anthropic/claude-x",
}


def _bridge(**env: str) -> tuple[OpenCodeBridge, KeywordMatcher]:
    bridge = OpenCodeBridge(BridgeConfig.from_env({**_BASE_ENV, **env}))
    matcher = KeywordMatcher()
    bridge._graph._matcher_factory = lambda: matcher
    return bridge, matcher


def test_the_bridge_publishes_the_policy_and_needs_no_admission_to_collapse():
    bridge, matcher = _bridge(CONTEXT_GRAPH="on", CONTEXT_GRAPH_MODE="inplace")
    assert bridge.handle({"type": "startup", "id": "s"})["graph"]["mode"] == "inplace"

    response = bridge.handle({"type": "graph_project", "id": "g", "session_id": "s", "messages": _transcript(4)})
    # The defaults: a turn is final the moment it closes, so all four closed turns are one summary each.
    assert response["operation"] == "projected" and [item["at_id"] for item in response["collapses"]] == [
        "a0",
        "a1",
        "a2",
        "a3",
    ]
    assert response["removed_ids"] == ["a0", "f0", "a1", "f1", "a2", "f2", "a3", "f3"]
    assert matcher.calls == 0, "deciding what to collapse scores nothing, so it needs no admission"
    assert json.loads(json.dumps(response)) == response


def test_the_recall_tool_answers_through_the_bridge_and_shutdown_forgets_the_graph():
    bridge, _ = _bridge(CONTEXT_GRAPH="on", CONTEXT_GRAPH_MODE="inplace")
    bridge.handle({"type": "graph_project", "id": "g", "session_id": "s", "messages": _transcript(4)})

    expanded = bridge.handle({"type": "graph_expand_card", "id": "e", "session_id": "s", "titles": [_question(0)]})
    assert expanded["operation"] == "graph_result" and "total: 1000.50" in expanded["text"]
    assert bridge.handle({"type": "graph_status", "id": "gs"})["totals"]["expand_calls"] == 1

    assert bridge.handle({"type": "shutdown", "id": "sd", "session_id": "s"})["operation"] == "discarded"
    assert bridge.handle({"type": "graph_status", "id": "gs2"})["sessions"] == {}


# --------------------------------------------------------------------------- #
# The dialogue tier: tool calls collapse at once, the question and answers stay    #
# --------------------------------------------------------------------------- #


def test_the_newest_closed_turn_keeps_its_question_and_its_answer_while_its_tool_calls_become_a_summary():
    graph, _ = _graph(keep_recent=0, keep_dialogue=1)
    response = graph.project("s", _transcript(3), scoring_allowed=True)

    assert [item["at_id"] for item in response["collapses"]] == ["a0", "a1", "a2"]
    assert response["removed_ids"] == ["a0", "f0", "a1", "f1", "a2"], "turn 2 keeps its answer f2"
    sent = _sent(_transcript(3), response)
    assert [message["id"] for message in sent if not message["id"].startswith("summary-of-")] == [
        "u0",
        "u1",
        "u2",
        "f2",
        "u3",
    ]
    assert [message["id"] for message in sent[-3:]] == ["summary-of-a2", "f2", "u3"], (
        "the summary stands where the tool step was"
    )


def test_a_summary_has_the_same_bytes_in_the_dialogue_tier_and_after_it():
    graph, _ = _graph(keep_recent=0, keep_dialogue=1)
    young = _plan(graph.project("s", _transcript(3), scoring_allowed=True))
    older = _plan(graph.project("s", _transcript(4), scoring_allowed=True))
    assert young[-1][1] == older[2][1], "the summary of turn 2 does not change when its answer is collapsed"
    assert older[: len(young) - 1] == young[:-1], "nothing before the newest change moves"


@pytest.mark.parametrize(
    ("keep", "dialogue", "whole", "dialogue_only", "complete"),
    [
        (0, 0, 0, 0, 5),
        (0, 1, 0, 1, 4),
        (0, 2, 0, 2, 3),
        (1, 1, 1, 1, 3),
        (2, 2, 2, 2, 1),
        (3, 3, 3, 2, 0),
        (10**6, 3, 5, 0, 0),
    ],
)
def test_ages_decide_the_state_of_each_closed_turn(
    keep: int, dialogue: int, whole: int, dialogue_only: int, complete: int
):
    graph, _ = _graph(keep_recent=keep, keep_dialogue=dialogue)
    response = graph.project("s", _transcript(5), scoring_allowed=True)
    removed = set(response.get("removed_ids", ()))
    states = []
    for index in range(5):  # oldest first
        gone = {name for name in (f"a{index}", f"f{index}") if name in removed}
        states.append("whole" if not gone else "dialogue" if gone == {f"a{index}"} else "complete")
    assert states.count("whole") == whole
    assert states.count("dialogue") == dialogue_only
    assert states.count("complete") == complete
    assert states == sorted(states, key={"complete": 0, "dialogue": 1, "whole": 2}.get), "older turns are never fresher"


def test_a_turn_that_called_no_tool_has_nothing_to_collapse_until_it_is_old_enough():
    transcript = _transcript(4)
    transcript[4:6] = [  # turn 1 becomes pure chat: its question, then one text answer
        {"id": "a1", "role": "assistant", "parts": [{"type": "text", "text": "Just a chat answer with 7 items."}]}
    ]
    graph, _ = _graph(keep_recent=0, keep_dialogue=3)
    response = graph.project("s", transcript, scoring_allowed=True)
    assert [item["at_id"] for item in response["collapses"]] == ["a0", "a2", "a3"], "turn 1 has no tool step to replace"
    assert "a1" not in response["removed_ids"]

    old, _ = _graph(keep_recent=0, keep_dialogue=0)
    assert [item["at_id"] for item in old.project("s", transcript, scoring_allowed=True)["collapses"]][1] == "a1"


def test_recalling_a_turn_in_the_dialogue_tier_returns_its_tool_calls_and_not_the_answer_that_is_still_there():
    graph, _ = _graph(keep_recent=0, keep_dialogue=1)
    graph.project("s", _transcript(3), scoring_allowed=True)
    answer = graph.expand_card("s", [_question(2)])
    assert '[tool read_docs {"q":"call_2"}]' in answer and "total: 1002.50" in answer
    assert "The charlie total is 1002.50 across 12 items." not in answer, "the answer stayed in the conversation"


def test_the_dialogue_tier_only_keeps_what_the_whole_turn_policy_would_have_removed_from_the_turn_after_the_kept_ones():
    old, _ = _graph(keep_recent=1, keep_dialogue=0)
    new, _ = _graph(keep_recent=1, keep_dialogue=1)
    transcript = _transcript(4)
    a, b = old.project("s", transcript, scoring_allowed=True), new.project("s", transcript, scoring_allowed=True)
    assert (
        [item["at_id"] for item in a["collapses"]] == [item["at_id"] for item in b["collapses"]] == ["a0", "a1", "a2"]
    )
    assert a["removed_ids"] == ["a0", "f0", "a1", "f1", "a2", "f2"]
    assert b["removed_ids"] == ["a0", "f0", "a1", "f1", "a2"], "turn 2 keeps its answer; nothing else differs"
    assert _plan(a) == _plan(b), "and the summaries are the same bytes"
