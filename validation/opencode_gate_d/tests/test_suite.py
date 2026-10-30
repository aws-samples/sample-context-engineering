"""The Gate D contract suite on the real binding, over synthetic transcripts. No network, no host."""

from __future__ import annotations

import json

import pytest
from context_core import opencode_graph
from context_core.opencode_graph import GraphConfig

from opencode_gate_d import __main__ as cli
from opencode_gate_d.contracts import REQUIRED_SCENARIOS
from opencode_gate_d.evaluator import evaluate
from opencode_gate_d.suite import run_suite, synthetic_wire

WAIVER = {"by": "owner", "requirement": "FR5.1/BR5.1", "reason": "measure D on OpenCode anyway"}


@pytest.fixture(scope="module")
def result():
    return run_suite({"synthetic": synthetic_wire(10)}, gate_a_evidence_id="gate-a-v6-no_go")


def test_the_binding_passes_every_scenario_on_a_synthetic_conversation(result):
    assert result.failures == []
    decision = evaluate(result.requests, gate_a_outcome="no_go", waiver=WAIVER, scenario_failures=result.failures)
    assert decision.outcome == "go", decision.reasons
    assert decision.capability_state == "experimental"
    assert set(REQUIRED_SCENARIOS) <= set(decision.scenarios_covered)


def test_every_scenario_produced_requests_and_the_graph_really_engaged(result):
    for scenario in REQUIRED_SCENARIOS:
        assert result.counts.get(scenario, 0) > 0, scenario
    assert any(request.projected for request in result.requests)
    assert result.counts["full_pass"] == len(result.requests)


def test_the_durable_transcript_is_the_same_before_and_after_every_request(result):
    assert all(
        r.transcript_before_digest == r.transcript_after_digest and r.canonical_equality for r in result.requests
    )


def test_the_full_pass_twin_reproduces_the_baseline_input_for_every_request(result):
    assert all(r.full_pass_digest == r.baseline_model_input_digest for r in result.requests)


def test_evidence_holds_digests_and_ids_and_not_a_word_of_the_conversation(result):
    text = json.dumps([r.to_dict() for r in result.requests])
    for word in ("ledger", "balance", "alpha", "bravo", "total"):
        assert word not in text


def test_sessions_never_share_a_card_id(result):
    owner: dict[str, str] = {}
    for request in result.requests:
        for card_id, _ in request.card_ids_and_sources:
            assert owner.setdefault(card_id, request.stream_id) == request.stream_id


def test_a_binding_that_leaks_state_between_sessions_is_caught(monkeypatch):
    """Break isolation on purpose: the suite must say so rather than pass."""
    from context_core import opencode_graph

    original = opencode_graph.OpenCodeGraph._project

    def leaky(self, session_id, session, transcript, *, scoring_allowed):
        # Every session is served from one shared session record: the sessions step on each other.
        shared = self._sessions.setdefault("__shared__", session)
        return original(self, session_id, shared, transcript, scoring_allowed=scoring_allowed)

    monkeypatch.setattr(opencode_graph.OpenCodeGraph, "_project", leaky)
    broken = run_suite({"synthetic": synthetic_wire(10)}, gate_a_evidence_id="gate-a-v6-no_go")
    assert broken.failures, "a binding that shares state across sessions must fail the suite"
    decision = evaluate(broken.requests, gate_a_outcome="no_go", waiver=WAIVER, scenario_failures=broken.failures)
    assert decision.outcome == "no_go"


def test_a_binding_that_mutates_the_transcript_is_caught(monkeypatch):
    from context_core import opencode_graph

    original = opencode_graph.to_neutral

    def mutating(transcript, **kwargs):
        if transcript:
            transcript[0]["parts"].append({"type": "text", "text": "tampered"})
        return original(transcript, **kwargs)

    monkeypatch.setattr(opencode_graph, "to_neutral", mutating)
    broken = run_suite({"synthetic": synthetic_wire(6)}, gate_a_evidence_id="gate-a-v6-no_go")
    assert "binding_mutated_the_transcript" in broken.failures
    assert (
        evaluate(broken.requests, gate_a_outcome="no_go", waiver=WAIVER, scenario_failures=broken.failures).outcome
        == "no_go"
    )


def test_cli_suite_then_decide(tmp_path):
    suite_path = tmp_path / "suite.json"
    assert cli.main(["suite", "--synthetic-turns", "8", "--out", str(suite_path)]) == 0
    decision_path = tmp_path / "decision.json"
    code = cli.main(
        [
            "decide",
            "--evidence",
            str(suite_path),
            "--gate-a-outcome",
            "no_go",
            "--waiver-by",
            "owner",
            "--waiver-reason",
            "measure D anyway",
            "--out",
            str(decision_path),
        ]
    )
    decision = json.loads(decision_path.read_text())
    assert code == 0 and decision["outcome"] == "go" and decision["precondition"] == "waived_by_owner"
    assert decision["capability_state"] == "experimental" and decision["gate_a_reclassified"] is False
    # And without the waiver the very same evidence stays unsupported.
    assert (
        cli.main(
            ["decide", "--evidence", str(suite_path), "--gate-a-outcome", "no_go", "--out", str(tmp_path / "d2.json")]
        )
        == 1
    )
    assert json.loads((tmp_path / "d2.json").read_text())["outcome"] == "unsupported"


# --------------------------------------------------------------------------- #
# The in-place policy: the same five scenarios, the same decision rule        #
# --------------------------------------------------------------------------- #

INPLACE = GraphConfig(enabled=True, mode="inplace")  # the defaults: a turn is final the moment it closes


@pytest.fixture(scope="module")
def inplace_result():
    return run_suite({"synthetic": synthetic_wire(10)}, gate_a_evidence_id="gate-a-v6-no_go", config=INPLACE)


def test_the_inplace_binding_passes_every_scenario(inplace_result):
    assert inplace_result.failures == []
    decision = evaluate(
        inplace_result.requests, gate_a_outcome="no_go", waiver=WAIVER, scenario_failures=inplace_result.failures
    )
    assert decision.outcome == "go", decision.reasons
    assert decision.capability_state == "experimental"
    assert set(REQUIRED_SCENARIOS) <= set(decision.scenarios_covered)


def test_the_inplace_policy_really_engaged_and_never_appended_anything(inplace_result):
    for scenario in REQUIRED_SCENARIOS:
        assert inplace_result.counts.get(scenario, 0) > 0, scenario
    assert any(request.projected for request in inplace_result.requests)
    assert all(r.transcript_before_digest == r.transcript_after_digest for r in inplace_result.requests)
    assert all(r.full_pass_digest == r.baseline_model_input_digest for r in inplace_result.requests)


def test_inplace_evidence_holds_digests_and_ids_and_not_a_word_of_the_conversation(inplace_result):
    text = json.dumps([r.to_dict() for r in inplace_result.requests])
    for word in ("ledger", "balance", "alpha", "bravo", "total", "collapsed_turn"):
        assert word not in text


def test_the_model_input_of_an_inplace_answer_is_the_conversation_with_summaries_where_the_turns_were():
    from context_core.opencode_graph import OpenCodeGraph

    from opencode_gate_d.suite import model_input

    wire = synthetic_wire(4)
    graph = OpenCodeGraph(INPLACE, matcher_factory=lambda: None)
    # Everything up to and including the question of the fourth turn: three closed turns, each one summary.
    prefix = wire[: next(index for index, message in enumerate(wire) if message["id"] == "u003") + 1]
    answer = graph.project("s", prefix, scoring_allowed=False)
    sent = model_input(prefix, answer)
    roles = [message["role"] for message in sent]
    assert roles[0] == "user" and roles[1] == "assistant" and roles[-1] == "user"
    assert [m["id"] for m in sent if m["role"] == "user"] == ["u000", "u001", "u002", "u003"]
    assert sum(1 for m in sent if m["id"].startswith("graph-collapse-")) == 3
    assert "f002" not in [m["id"] for m in sent], "the answer of the newest closed turn is part of its summary now"
    assert len(sent) < len(prefix)


def test_a_binding_that_removes_a_user_message_in_an_inplace_answer_is_caught(monkeypatch):
    original = opencode_graph.plan_inplace

    def wrong(transcript, cards, keep_recent, keep_dialogue=0):
        plan = original(transcript, cards, keep_recent, keep_dialogue)
        if plan:
            # A collapse that also takes out the question of its turn.
            question = next(m["id"] for m in transcript if m["role"] == "user")
            first = plan[0]
            plan[0] = type(first)(first.at_id, (*first.removed_ids, question), first.text, first.title)
        return plan

    monkeypatch.setattr(opencode_graph, "plan_inplace", wrong)
    broken = run_suite({"synthetic": synthetic_wire(8)}, gate_a_evidence_id="gate-a-v6-no_go", config=INPLACE)
    assert "inplace_removed_a_message_that_is_not_the_assistants" in broken.failures
    decision = evaluate(broken.requests, gate_a_outcome="no_go", waiver=WAIVER, scenario_failures=broken.failures)
    assert decision.outcome == "no_go"


def test_a_binding_whose_recall_moves_the_request_is_caught(monkeypatch):
    """A recall that changed what the next request carries would rewrite the cache from there on: the suite must see it."""
    original = opencode_graph.plan_inplace
    recalled = {"yes": False}

    def moving(transcript, cards, keep_recent, keep_dialogue=0):
        plan = original(transcript, cards, keep_recent, keep_dialogue)
        return plan[:-1] if recalled["yes"] and plan else plan

    expand = opencode_graph.OpenCodeGraph.expand_card

    def noting(self, session_id, titles):
        # Only the recovery scenario's own recall flips it: the subagent scenario recalls too, long before.
        if session_id.endswith("/recovery"):
            recalled["yes"] = True
        return expand(self, session_id, titles)

    monkeypatch.setattr(opencode_graph, "plan_inplace", moving)
    monkeypatch.setattr(opencode_graph.OpenCodeGraph, "expand_card", noting)
    broken = run_suite({"synthetic": synthetic_wire(8)}, gate_a_evidence_id="gate-a-v6-no_go", config=INPLACE)
    assert "recovery_moved_the_request" in broken.failures


def test_a_recall_that_returns_nothing_is_caught(monkeypatch):
    monkeypatch.setattr(
        opencode_graph.OpenCodeGraph, "expand_card", lambda self, session_id, titles: "expand_card | nothing"
    )
    broken = run_suite({"synthetic": synthetic_wire(8)}, gate_a_evidence_id="gate-a-v6-no_go", config=INPLACE)
    assert "recovery_expand_card_did_not_return_the_turn" in broken.failures


def test_cli_suite_inplace_then_decide(tmp_path):
    suite_path = tmp_path / "suite.json"
    assert cli.main(["suite", "--mode", "inplace", "--synthetic-turns", "8", "--out", str(suite_path)]) == 0
    body = json.loads(suite_path.read_text())
    assert body["mode"] == "inplace" and (body["keep_recent"], body["keep_dialogue"]) == (0, 0)
    assert body["failures"] == []
    dialogue = tmp_path / "dialogue.json"
    args = ["suite", "--mode", "inplace", "--keep-recent", "1", "--keep-dialogue", "1", "--synthetic-turns", "8"]
    assert cli.main([*args, "--out", str(dialogue)]) == 0
    body = json.loads(dialogue.read_text())
    assert (body["keep_recent"], body["keep_dialogue"], body["failures"]) == (1, 1, [])
    decision_path = tmp_path / "decision.json"
    code = cli.main(
        [
            "decide",
            "--evidence",
            str(suite_path),
            "--gate-a-outcome",
            "no_go",
            "--waiver-by",
            "owner",
            "--waiver-reason",
            "measure D anyway",
            "--out",
            str(decision_path),
        ]
    )
    decision = json.loads(decision_path.read_text())
    assert code == 0 and decision["outcome"] == "go" and decision["capability_state"] == "experimental"
    assert decision["gate_a_reclassified"] is False
