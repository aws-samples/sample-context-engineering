"""The Gate D evidence contract and its evaluator, on hand-built records."""

from __future__ import annotations

from dataclasses import replace

import pytest

from opencode_gate_d.contracts import (
    REQUIRED_SCENARIOS,
    GateDRequestEvidence,
    absence_digest,
    digest_is_well_formed,
)
from opencode_gate_d.evaluator import evaluate

H = "a" * 64
OTHER = "b" * 64
WAIVER = {"by": "owner", "requirement": "FR5.1/BR5.1", "reason": "measure D on OpenCode anyway"}


def record(index: int = 1, **overrides) -> GateDRequestEvidence:
    base = GateDRequestEvidence(
        evidence_id=f"e{index}",
        gate_a_evidence_id="gate-a-v6-no_go",
        request_id=f"s:{index}",
        stream_id="s",
        transcript_before_digest=H,
        transcript_after_digest=H,
        card_ids_and_sources=(("c_1", ("m1", "m2")),),
        scenarios=("tool_loop", "full_pass"),
        full_pass_digest=OTHER,
        baseline_model_input_digest=OTHER,
        canonical_equality=True,
        absence_digest=absence_digest(),
        projected=True,
    )
    return replace(base, **overrides)


def full_set() -> list[GateDRequestEvidence]:
    """One clean record per required scenario, in two streams."""
    return [
        record(1, scenarios=("tool_loop", "full_pass")),
        record(2, scenarios=("subagent", "full_pass"), stream_id="child", card_ids_and_sources=(("c_9", ("x1",)),)),
        record(3, scenarios=("compact", "full_pass")),
        record(4, scenarios=("resume", "full_pass")),
        record(5, scenarios=("recovery", "full_pass")),
    ]


def test_a_complete_clean_set_is_go_but_only_experimental_under_a_waiver():
    decision = evaluate(full_set(), gate_a_outcome="no_go", waiver=WAIVER)
    assert decision.outcome == "go"
    assert decision.precondition == "waived_by_owner"
    assert decision.capability_state == "experimental", "a waived precondition can never make D supported"
    assert decision.waiver == WAIVER
    assert decision.to_dict()["gate_a_reclassified"] is False
    assert all(decision.checks.values())


def test_with_a_gate_a_go_a_clean_set_makes_d_supported():
    decision = evaluate(full_set(), gate_a_outcome="go")
    assert (decision.outcome, decision.precondition, decision.capability_state) == ("go", "gate_a_go", "supported")


@pytest.mark.parametrize("waiver", [None, {}, {"by": "owner"}, {"by": "", "requirement": "x", "reason": "y"}])
def test_without_go_or_an_explicit_waiver_d_stays_unsupported_and_nothing_is_evaluated(waiver):
    decision = evaluate(full_set(), gate_a_outcome="no_go", waiver=waiver)
    assert decision.outcome == "unsupported"
    assert decision.precondition == "unmet"
    assert decision.reasons == ("CAPABILITY_UNSUPPORTED",)
    assert decision.requests_evaluated == 0


def test_a_missing_scenario_is_no_go_and_names_it():
    records = [r for r in full_set() if "compact" not in r.scenarios]
    decision = evaluate(records, gate_a_outcome="no_go", waiver=WAIVER)
    assert decision.outcome == "no_go" and decision.capability_state == "unsupported"
    assert "SCENARIO_MISSING:compact" in decision.reasons


def test_the_durable_transcript_changing_is_no_go():
    records = full_set()
    records[0] = replace(records[0], transcript_after_digest=OTHER)
    assert "TRANSCRIPT_DURABLY_CHANGED" in evaluate(records, gate_a_outcome="no_go", waiver=WAIVER).reasons
    records[0] = replace(records[0], transcript_after_digest=H, canonical_equality=False)
    assert "TRANSCRIPT_DURABLY_CHANGED" in evaluate(records, gate_a_outcome="no_go", waiver=WAIVER).reasons


def test_the_digest_framing_is_not_part_of_equality():
    records = full_set()
    records[0] = replace(records[0], transcript_before_digest="sha256:" + H, transcript_after_digest=H)
    assert evaluate(records, gate_a_outcome="no_go", waiver=WAIVER).outcome == "go"


def test_a_request_with_a_missing_field_is_incomplete_evidence():
    records = full_set()
    records[1] = replace(records[1], transcript_before_digest="")
    assert "EVIDENCE_INCOMPLETE" in evaluate(records, gate_a_outcome="no_go", waiver=WAIVER).reasons
    assert "transcript_before_digest" in records[1].defects()[0]


def test_a_card_without_a_source_and_a_request_without_cards_are_both_no_go():
    records = full_set()
    records[0] = replace(records[0], card_ids_and_sources=(("c_1", ()),))
    assert "EVIDENCE_INCOMPLETE" in evaluate(records, gate_a_outcome="no_go", waiver=WAIVER).reasons
    records[0] = replace(records[0], card_ids_and_sources=())
    assert "CARD_SOURCE_MISSING" in evaluate(records, gate_a_outcome="no_go", waiver=WAIVER).reasons


def test_a_request_before_the_first_closed_turn_says_so_and_is_allowed_no_cards():
    records = full_set() + [record(6, card_ids_and_sources=(), scenarios=("live", "pre_card"), full_pass_digest=None)]
    assert evaluate(records, gate_a_outcome="no_go", waiver=WAIVER).outcome == "go"


def test_a_card_id_whose_sources_change_is_unstable_and_a_shared_id_collides():
    records = full_set()
    records.append(record(6, card_ids_and_sources=(("c_1", ("m1", "m2", "m3")),)))
    assert "CARD_ID_UNSTABLE" in evaluate(records, gate_a_outcome="no_go", waiver=WAIVER).reasons
    records = full_set()
    records[1] = replace(records[1], card_ids_and_sources=(("c_1", ("m1", "m2")),))  # same id, another stream
    assert "CARD_ID_COLLISION" in evaluate(records, gate_a_outcome="no_go", waiver=WAIVER).reasons


def test_a_compaction_is_a_defined_transition_for_card_identity_and_only_within_its_own_stream():
    records = full_set()
    # The same id points at other messages after a compaction: allowed, because the epoch moved.
    records.append(record(6, card_ids_and_sources=(("c_1", ("m7", "m8")),), epoch=1))
    assert evaluate(records, gate_a_outcome="no_go", waiver=WAIVER).outcome == "go"
    # The same change inside one epoch is not.
    records.append(record(7, card_ids_and_sources=(("c_1", ("m9",)),), epoch=1))
    assert "CARD_ID_UNSTABLE" in evaluate(records, gate_a_outcome="no_go", waiver=WAIVER).reasons
    # And an epoch never lets two streams share an id.
    other = full_set() + [record(6, stream_id="child", card_ids_and_sources=(("c_1", ("m1", "m2")),), epoch=1)]
    assert "CARD_ID_COLLISION" in evaluate(other, gate_a_outcome="no_go", waiver=WAIVER).reasons


def test_a_full_pass_that_differs_from_the_baseline_is_no_go_and_one_that_never_ran_is_missing():
    records = full_set()
    records[0] = replace(records[0], full_pass_digest=H)
    assert "FULL_PASS_DIVERGES" in evaluate(records, gate_a_outcome="no_go", waiver=WAIVER).reasons
    none = [replace(r, scenarios=tuple(s for s in r.scenarios if s != "full_pass"), full_pass_digest=None) for r in full_set()]
    assert "FULL_PASS_MISSING" in evaluate(none, gate_a_outcome="no_go", waiver=WAIVER).reasons


def test_any_durable_effect_is_no_go():
    records = full_set()
    records[0] = replace(
        records[0],
        absence_digest=absence_digest({"graph_messages_in_store": 1, "durable_writes": 0, "stored_transcript_changed": False}),
    )
    assert "DURABLE_EFFECT_DETECTED" in evaluate(records, gate_a_outcome="no_go", waiver=WAIVER).reasons


def test_a_graph_that_never_engaged_proves_nothing():
    records = [replace(r, projected=False) for r in full_set()]
    assert "NEVER_ENGAGED" in evaluate(records, gate_a_outcome="no_go", waiver=WAIVER).reasons


def test_a_scenario_failure_is_a_reason_and_no_evidence_is_no_go():
    decision = evaluate(full_set(), gate_a_outcome="no_go", waiver=WAIVER, scenario_failures=["resume_projects_differently"])
    assert decision.outcome == "no_go" and "SCENARIO_FAILED:resume_projects_differently" in decision.reasons
    assert "EVIDENCE_MISSING" in evaluate([], gate_a_outcome="no_go", waiver=WAIVER).reasons


def test_records_round_trip_and_carry_no_free_text():
    original = record(1)
    assert GateDRequestEvidence.from_dict(original.to_dict()) == original
    assert digest_is_well_formed(original.absence_digest) and not digest_is_well_formed("abc")
    assert set(REQUIRED_SCENARIOS) == {"tool_loop", "subagent", "compact", "resume", "recovery"}
