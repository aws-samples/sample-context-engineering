"""Offline tests for the CE-60 host bridge (:mod:`opencode_gate_a.host_driver`).

Everything here is offline and deterministic: no OpenCode process, no network, no AWS, no model call.
The tests cover turn resolution (and its fail-closed guard), the 30-eligible oracle mapping, the
60-turn finalize pipeline, the no-payload-persistence guarantee, the live-manifest binding, and the
Gate A outcome never being ``go`` while retrieval is host-not-exposed.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest
from _ce60_factories import build_stat_plan
from opencode_gate_a import host_driver, runner
from opencode_gate_a.contracts import (
    CORRECTNESS_ELIGIBLE_COUNT,
    TURN_COUNT,
    ContractViolation,
)


def _turn_ids() -> list[str]:
    """Return the 60 canonical CE-60 turn ids from the scenario."""
    return [t["turn_id"] for t in host_driver.emit_turns()["turns"]]


def _observations(turn_ids: list[str], *, model_input: int) -> list[dict[str, Any]]:
    """Build 60 successful observations with a fixed model-input reading and empty answers.

    Empty answers score as not materially correct, which keeps correctness deterministic (0/30) across
    arms without needing real model text.

    Args:
        turn_ids: The 60 turn ids.
        model_input: The model-input token value for every turn.

    Returns:
        The observation list.
    """
    return [
        {
            "turn_id": turn_id,
            "host_signal": "success",
            "matcher_passed": True,
            "answer_text": "",
            "tool_calls": [],
            "model_input": model_input,
            "latency_ms": 12.0,
            "retrieval_status": "host_not_exposed",
        }
        for turn_id in turn_ids
    ]


def _finalize(tmp_path: Any, arm: str, *, model_input: int, canary: bool = False) -> dict[str, Any]:
    """Run :func:`finalize` for one offline arm, optionally seeding a payload canary."""
    turn_ids = _turn_ids()
    observations = _observations(turn_ids, model_input=model_input)
    if canary:
        observations[0]["answer_text"] = "SECRET-ANSWER-CANARY"
    return host_driver.finalize(
        {
            "mode": "offline",
            "arm": arm,
            "run_id": f"run-{arm}",
            "host": "opencode",
            "host_version": "1.18.34",
            "model_id": "anthropic/claude-x",
            "output_dir": str(tmp_path / arm),
            "turns": observations,
        }
    )


# --- Turn resolution -------------------------------------------------------------------------------


def test_emit_turns_resolves_sixty_from_scenario() -> None:
    result = host_driver.emit_turns()
    assert len(result["turns"]) == TURN_COUNT
    assert len({t["turn_id"] for t in result["turns"]}) == TURN_COUNT
    assert result["source"] == "scenario"


def test_emit_turns_reads_an_explicit_file(tmp_path: Any) -> None:
    turns = [{"turn_id": f"t{i:02d}", "prompt": f"p{i}"} for i in range(TURN_COUNT)]
    path = tmp_path / "turns.json"
    path.write_text(json.dumps({"turns": turns}), encoding="utf-8")
    result = host_driver.emit_turns(str(path))
    assert len(result["turns"]) == TURN_COUNT
    assert result["source"].startswith("turns-json:")


def test_emit_turns_fails_closed_on_wrong_count(tmp_path: Any) -> None:
    turns = [{"turn_id": f"t{i:02d}", "prompt": f"p{i}"} for i in range(TURN_COUNT - 1)]
    path = tmp_path / "turns.json"
    path.write_text(json.dumps(turns), encoding="utf-8")
    with pytest.raises(host_driver.TurnsUnavailable):
        host_driver.emit_turns(str(path))


def test_emit_turns_fails_closed_without_a_source(tmp_path: Any) -> None:
    bogus_root = str(tmp_path / "no-scenario-here")
    with pytest.raises(host_driver.TurnsUnavailable):
        host_driver.emit_turns(scenario_root=bogus_root)


# --- Oracle / catalog ------------------------------------------------------------------------------


def test_correctness_eligibility_is_exactly_thirty() -> None:
    eligible = host_driver.correctness_eligibility(_turn_ids())
    assert len(eligible) == CORRECTNESS_ELIGIBLE_COUNT


def test_build_ce60_catalog_is_valid_and_digest_bound() -> None:
    turn_ids = _turn_ids()
    eligible = host_driver.correctness_eligibility(turn_ids)
    catalog = host_driver.build_ce60_catalog(turn_ids, eligible)
    assert len(catalog.turn_definitions) == TURN_COUNT
    assert catalog.digest == catalog.content_digest()


# --- Finalize --------------------------------------------------------------------------------------


def test_finalize_writes_payload_free_artifacts(tmp_path: Any) -> None:
    evidence = _finalize(tmp_path, "baseline", model_input=1000, canary=True)
    assert evidence["turns_completed"] == TURN_COUNT
    validity = {m["metric_name"]: m["status"] for m in evidence["metric_validity"]}
    assert validity["completion"] == "valid"
    assert validity["correctness"] == "valid"
    assert validity["model_input"] == "valid"
    # OpenCode exposes no CE-60 retrieval count; it must be explicit, not a fabricated zero.
    assert validity["retrieval"] == "host_not_exposed"

    out_dir = tmp_path / "baseline"
    written = sorted(os.listdir(out_dir))
    assert written == [
        "baseline-catalog.json",
        "baseline-evaluation.json",
        "baseline-evidence.json",
        "baseline-manifest.json",
        "baseline-sanitized.json",
    ]
    # The raw answer canary must never reach any written artefact.
    for name in written:
        text = (out_dir / name).read_text(encoding="utf-8")
        assert "SECRET-ANSWER-CANARY" not in text


def test_finalize_never_truncates(tmp_path: Any) -> None:
    turn_ids = _turn_ids()
    observations = _observations(turn_ids, model_input=1000)[:-1]  # 59 turns
    with pytest.raises(ContractViolation) as exc:
        host_driver.finalize(
            {
                "mode": "offline",
                "arm": "baseline",
                "run_id": "run-x",
                "host": "opencode",
                "host_version": "1.18.34",
                "model_id": "anthropic/claude-x",
                "output_dir": str(tmp_path / "x"),
                "turns": observations,
            }
        )
    assert exc.value.code == "RESULT_NOT_SANITIZED"


def test_finalize_live_binds_admission_and_plan(tmp_path: Any) -> None:
    turn_ids = _turn_ids()
    observations = _observations(turn_ids, model_input=1000)
    evidence = host_driver.finalize(
        {
            "mode": "live",
            "arm": "A",
            "run_id": "run-live-A",
            "host": "opencode",
            "host_version": "1.18.34",
            "model_id": "anthropic/claude-x",
            "output_dir": str(tmp_path / "live"),
            "turns": observations,
            "live": {
                "execution_admission_id": "appr-1",
                "admission_policy_digest": "sha256:" + ("a" * 64),
                "budget_ceiling_units": 10.0,
                "budget_currency_or_unit": "USD",
                "price_provenance": "operator-declared:USD:appr-1",
                "stat_plan": runner._stat_plan_to_dict(build_stat_plan()),
            },
        }
    )
    # A live run is evaluated as a replay of itself; the live admission is bound in a separate
    # provenance manifest.
    assert evidence["requested_mode"] == "live"
    assert evidence["evaluation_mode"] == "replay"
    eval_manifest = runner.manifest_from_dict(
        json.loads((tmp_path / "live" / "A-manifest.json").read_text(encoding="utf-8"))
    )
    assert eval_manifest.execution_mode == "replay"
    assert eval_manifest.backend_id == "local-fake"

    live_manifest = runner.manifest_from_dict(
        json.loads((tmp_path / "live" / "A-live-manifest.json").read_text(encoding="utf-8"))
    )
    assert live_manifest.execution_mode == "live"
    assert live_manifest.backend_id == "remote-explicit"
    assert live_manifest.execution_admission_id == "appr-1"
    assert live_manifest.stat_plan is not None
    assert evidence["live_manifest_digest"] == live_manifest.manifest_digest


def test_finalize_live_fails_closed_without_admission(tmp_path: Any) -> None:
    turn_ids = _turn_ids()
    observations = _observations(turn_ids, model_input=1000)
    with pytest.raises(ContractViolation):
        host_driver.finalize(
            {
                "mode": "live",
                "arm": "A",
                "run_id": "run-live-A",
                "host": "opencode",
                "host_version": "1.18.34",
                "model_id": "anthropic/claude-x",
                "output_dir": str(tmp_path / "live2"),
                "turns": observations,
                "live": {"budget_ceiling_units": 10.0},
            }
        )


# --- Gate A never a false go -----------------------------------------------------------------------


def test_gate_a_is_never_go_while_retrieval_is_host_not_exposed(tmp_path: Any) -> None:
    _finalize(tmp_path, "baseline", model_input=1000)
    _finalize(tmp_path, "A", model_input=700)  # 30% model-input reduction, so only retrieval blocks

    decision = runner.decide_gate_a_files(
        baseline_manifest_path=str(tmp_path / "baseline" / "baseline-manifest.json"),
        baseline_eval_path=str(tmp_path / "baseline" / "baseline-evaluation.json"),
        treatment_manifest_path=str(tmp_path / "A" / "A-manifest.json"),
        treatment_eval_path=str(tmp_path / "A" / "A-evaluation.json"),
        output_path=str(tmp_path / "gate-a-decision.json"),
    )
    assert decision.outcome != "go"
    assert "GATE_RETRIEVAL_INVALID" in decision.reasons


# --- CLI -------------------------------------------------------------------------------------------


def test_cli_emit_turns_returns_zero(capsys: Any) -> None:
    code = host_driver.main(["emit-turns"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert len(out["turns"]) == TURN_COUNT


# --- Retrieval instrumentation mapping (scenario-free) --------------------------------------------


def test_metric_block_reports_retrieval_counts_when_exposed() -> None:
    block = host_driver._metric_block(
        model_input=100, latency_ms=5.0, retrieval_attempts=3, retrieval_successes=2, correctness=None
    )
    assert block["retrieval"] == {"attempts": 3, "successes": 2}


def test_metric_block_marks_unmeasured_exposed_retrieval_as_collection_failed() -> None:
    block = host_driver._metric_block(
        model_input=None,
        latency_ms=None,
        retrieval_attempts=None,
        retrieval_successes=None,
        retrieval_status="collection_failed",
        correctness=None,
    )
    assert block["retrieval"] == {"status": "collection_failed"}



    block = host_driver._metric_block(
        model_input=100, latency_ms=5.0, retrieval_attempts=None, retrieval_successes=None, correctness=None
    )
    assert block["retrieval"] == {"status": "host_not_exposed"}


def test_metric_block_is_host_not_exposed_when_only_one_side_present() -> None:
    # A half-reading is never a fabricated count: both attempts and successes are required.
    block = host_driver._metric_block(
        model_input=None, latency_ms=None, retrieval_attempts=3, retrieval_successes=None, correctness=None
    )
    assert block["retrieval"] == {"status": "host_not_exposed"}


def test_build_raw_run_maps_retrieval_counts_without_the_oracle() -> None:
    # eligible is empty, so no turn is scored and the correctness oracle (scenario) is never imported.
    turn_ids = [f"t{i:02d}" for i in range(TURN_COUNT)]
    observations = [
        {
            "turn_id": tid,
            "host_signal": "success",
            "matcher_passed": True,
            "model_input": 100,
            "latency_ms": 1.0,
            "retrieval_attempts": 1,
            "retrieval_successes": 1,
        }
        for tid in turn_ids
    ]
    raw = host_driver._build_raw_run(
        run_id="r",
        observations=observations,
        eligible=frozenset(),
        host_evidence={},
        scenario_root=None,
    )
    assert all(t["metrics"]["retrieval"] == {"attempts": 1, "successes": 1} for t in raw["turns"])


def test_finalize_gate_a_can_go_once_retrieval_is_exposed(tmp_path: Any, monkeypatch: Any) -> None:
    # Prove the end state the instrumentation unlocks: with retrieval exposed and exact (0 failures) in
    # arm A, a 30% model-input drop and no correctness/completion regression, Gate A returns ``go``.
    # The correctness oracle (scenario) is stubbed so the whole test is offline and deterministic.
    turn_ids = [f"t{i:02d}" for i in range(TURN_COUNT)]
    eligible = frozenset(turn_ids[:CORRECTNESS_ELIGIBLE_COUNT])
    monkeypatch.setattr(host_driver, "correctness_eligibility", lambda ids, scenario_root=None: eligible)
    monkeypatch.setattr(host_driver, "score_correctness", lambda tid, text, scenario_root=None: True)

    def observations(*, model_input: int, retrieval: bool) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for tid in turn_ids:
            row: dict[str, Any] = {
                "turn_id": tid,
                "host_signal": "success",
                "matcher_passed": True,
                "answer_text": "",
                "model_input": model_input,
                "latency_ms": 10.0,
            }
            if retrieval:
                row["retrieval_attempts"] = 1
                row["retrieval_successes"] = 1
            rows.append(row)
        return rows

    def run(arm: str, *, model_input: int, retrieval: bool) -> None:
        host_driver.finalize(
            {
                "mode": "offline",
                "arm": arm,
                "run_id": f"run-{arm}",
                "host": "opencode",
                "host_version": "1.18.34",
                "model_id": "anthropic/claude-x",
                "output_dir": str(tmp_path / arm),
                "turns": observations(model_input=model_input, retrieval=retrieval),
            }
        )

    # Baseline: no retrieval exposed, heavier model input.
    run("baseline", model_input=1000, retrieval=False)
    # Arm A: retrieval exposed and exact, 30% model-input reduction.
    run("A", model_input=700, retrieval=True)

    decision = runner.decide_gate_a_files(
        baseline_manifest_path=str(tmp_path / "baseline" / "baseline-manifest.json"),
        baseline_eval_path=str(tmp_path / "baseline" / "baseline-evaluation.json"),
        treatment_manifest_path=str(tmp_path / "A" / "A-manifest.json"),
        treatment_eval_path=str(tmp_path / "A" / "A-evaluation.json"),
        output_path=str(tmp_path / "gate-a-decision.json"),
    )
    assert decision.outcome == "go"
    assert "GATE_RETRIEVAL_INVALID" not in decision.reasons


def test_finalize_carries_sanitized_metrics_summary_into_result_and_manifest(tmp_path: Any) -> None:
    turn_ids = _turn_ids()
    observations = _observations(turn_ids, model_input=900)
    metrics = {
        "metrics_schema_version": 1,
        "metrics_status": "exposed",
        "transform_count": 4,
        "skipped_count": 3,
        "previewed_count": 1,
        "baseline_count": 0,
        "refused_count": 0,
        "filtered_count": 1,
        "input_bytes": 4000,
        "output_bytes": 1000,
        "source_already_truncated_count": 0,
        "fallback_count": 0,
        "rerank_attempts": 1,
        "rerank_successes": 1,
        "rerank_failures": 0,
        "retrieval_attempts": 2,
        "retrieval_successes": 2,
        "metrics_write_failed": False,
        "tool_classes": "list_accounts,read_aws_documentation",
        "prompt": "SECRET-PROMPT-CANARY",
    }
    host_driver.finalize(
        {
            "mode": "offline",
            "arm": "A",
            "run_id": "run-metrics",
            "host": "opencode",
            "host_version": "1.18.34",
            "model_id": "anthropic/claude-x",
            "output_dir": str(tmp_path / "A"),
            "turns": observations,
            "metrics_summary": metrics,
        }
    )
    result = json.loads((tmp_path / "A" / "A-sanitized.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "A" / "A-manifest.json").read_text(encoding="utf-8"))
    assert result["host_evidence"]["metrics_status"] == "exposed"
    assert result["host_evidence"]["previewed_count"] == 1
    assert result["host_evidence"]["retrieval_attempts"] == 2
    assert result["host_evidence"]["tool_spec_tokens_status"] == "host_not_exposed"
    assert manifest["host_evidence"]["input_bytes"] == 4000
    serialized = (tmp_path / "A" / "A-sanitized.json").read_text(encoding="utf-8")
    assert "SECRET-PROMPT-CANARY" not in serialized


def test_finalize_persists_bounded_observability_without_payload(tmp_path: Any) -> None:
    """The host bridge records session diagnostics and rejects payload-shaped fields."""
    turn_ids = _turn_ids()
    observations = _observations(turn_ids, model_input=900)
    summary = {
        "observability_schema_version": 1,
        "message_count_max": 8,
        "message_count_median": 4.5,
        "tool_call_count_total": 4,
        "tool_call_count_max": 2,
        "mcp_tool_call_count_total": 1,
        "timeout_count": 1,
        "timeout_turn_ids": ["F002"],
        "tool_names": ["read", "mcp_ce60-fixture_search"],
        "session_compacted_observed_count": 1,
        "prompt": "SECRET-PROMPT-CANARY",
        "response": "SECRET-RESPONSE-CANARY",
        "arguments": "SECRET-ARGS-CANARY",
        "payload": {"value": "SECRET-PAYLOAD-CANARY"},
    }
    host_driver.finalize(
        {
            "mode": "offline",
            "arm": "A",
            "run_id": "run-observability",
            "host": "opencode",
            "host_version": "1.18.34",
            "model_id": "anthropic/claude-x",
            "output_dir": str(tmp_path / "A"),
            "turns": observations,
            "observability_summary": summary,
        }
    )

    result_path = tmp_path / "A" / "A-sanitized.json"
    manifest_path = tmp_path / "A" / "A-manifest.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for evidence in (result["host_evidence"], manifest["host_evidence"]):
        assert evidence["observability"]["message_count_max"] == 8
        assert evidence["observability"]["message_count_median"] == 4.5
        assert evidence["observability"]["tool_call_count_total"] == 4
        assert evidence["observability"]["mcp_tool_call_count_total"] == 1
        assert evidence["observability"]["timeout_count"] == 1
        assert evidence["observability"]["timeout_turn_ids"] == ["F002"]
        assert evidence["observability"]["tool_names"] == ["read", "mcp_ce60-fixture_search"]
    serialized = result_path.read_text(encoding="utf-8") + manifest_path.read_text(encoding="utf-8")
    for canary in (
        "SECRET-PROMPT-CANARY",
        "SECRET-RESPONSE-CANARY",
        "SECRET-ARGS-CANARY",
        "SECRET-PAYLOAD-CANARY",
    ):
        assert canary not in serialized


def test_finalize_rejects_unbounded_observability_list(tmp_path: Any) -> None:
    """The host bridge fails closed when the supplied diagnostic list exceeds its bound."""
    with pytest.raises(ContractViolation) as exc:
        host_driver.finalize(
            {
                "mode": "offline",
                "arm": "A",
                "run_id": "run-observability-invalid",
                "host": "opencode",
                "host_version": "1.18.34",
                "model_id": "anthropic/claude-x",
                "output_dir": str(tmp_path / "invalid"),
                "turns": _observations(_turn_ids(), model_input=900),
                "observability_summary": {"tool_names": ["read"] * 513},
            }
        )
    assert exc.value.code == "RESULT_NOT_SANITIZED"


def test_finalize_persists_schema_two_delta_observability(tmp_path: Any) -> None:
    """Schema 2 keeps final snapshots/deltas and compaction state without payload fields."""
    turn_ids = _turn_ids()
    observations = _observations(turn_ids, model_input=900)
    summary = {
        "observability_schema_version": 2,
        "telemetry_status": "partial",
        "session_message_count_final": 7,
        "session_tool_call_count_final": 6,
        "session_mcp_tool_call_count_final": 4,
        "delta_message_count_sum": None,
        "delta_tool_call_count_sum": None,
        "delta_mcp_tool_call_count_sum": None,
        "delta_message_count_max": 3,
        "delta_tool_call_count_max": 2,
        "delta_mcp_tool_call_count_max": 2,
        "snapshot_gap_turn_ids": ["F002-filler"],
        "snapshot_regression_turn_ids": [],
        "context_drop_turn_ids": ["F002-filler"],
        "timeout_count": 1,
        "timeout_turn_ids": ["F002-filler"],
        "tool_names": ["read"],
        "compaction_status": "unavailable",
        "compaction_reason": "unexplained_context_drop",
        "compaction_observed_count": 0,
        "prompt": "SECRET-SCHEMA2-CANARY",
    }
    host_driver.finalize(
        {
            "mode": "offline",
            "arm": "A",
            "run_id": "run-observability-v2",
            "host": "opencode",
            "host_version": "1.18.34",
            "model_id": "anthropic/claude-x",
            "output_dir": str(tmp_path / "A"),
            "turns": observations,
            "observability_summary": summary,
        }
    )
    result_path = tmp_path / "A" / "A-sanitized.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    evidence = result["host_evidence"]["observability"]
    assert evidence["observability_schema_version"] == 2
    assert evidence["session_message_count_final"] == 7
    assert evidence["snapshot_gap_turn_ids"] == ["F002-filler"]
    assert evidence["compaction_status"] == "unavailable"
    assert "SECRET-SCHEMA2-CANARY" not in result_path.read_text(encoding="utf-8")


def test_finalize_persists_timeout_state_and_declared_tool_policy(tmp_path: Any) -> None:
    """Timeout state lists and the arm-identical tool policy reach the manifest; payloads do not."""
    turn_ids = _turn_ids()
    observations = _observations(turn_ids, model_input=900)
    summary = {
        "observability_schema_version": 2,
        "telemetry_status": "ok",
        "timeout_count": 1,
        "timeout_turn_ids": ["F002-filler"],
        "timeout_classifications": ["blocked_question"],
        "timeout_last_tool_names": ["question"],
        "timeout_last_tool_statuses": ["running"],
        "timeout_assistant_states": ["incomplete"],
        "timeout_abort_statuses": ["ok"],
        "timeout_idle_statuses": ["idle"],
        "session_not_idle_turn_ids": [],
        "prompt": "SECRET-TIMEOUT-CANARY",
    }
    host_driver.finalize(
        {
            "mode": "offline",
            "arm": "baseline",
            "run_id": "run-timeout-state",
            "host": "opencode",
            "host_version": "1.18.34",
            "model_id": "anthropic/claude-x",
            "output_dir": str(tmp_path / "baseline"),
            "turns": observations,
            "observability_summary": summary,
            "disabled_tools": "question,webfetch",
            "removed_env_keys": "OPENCODE_CONFIG_CONTENT",
        }
    )
    manifest_path = tmp_path / "baseline" / "baseline-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evidence = manifest["host_evidence"]
    assert evidence["disabled_tools"] == "question,webfetch"
    assert evidence["removed_env_keys"] == "OPENCODE_CONFIG_CONTENT"
    assert evidence["observability"]["timeout_classifications"] == ["blocked_question"]
    assert evidence["observability"]["timeout_last_tool_names"] == ["question"]
    assert "SECRET-TIMEOUT-CANARY" not in manifest_path.read_text(encoding="utf-8")


def test_finalize_drops_a_tool_policy_that_is_not_identifier_only(tmp_path: Any) -> None:
    turn_ids = _turn_ids()
    host_driver.finalize(
        {
            "mode": "offline",
            "arm": "baseline",
            "run_id": "run-bad-policy",
            "host": "opencode",
            "host_version": "1.18.34",
            "model_id": "anthropic/claude-x",
            "output_dir": str(tmp_path / "baseline"),
            "turns": _observations(turn_ids, model_input=900),
            "disabled_tools": "question; rm -rf /",
        }
    )
    manifest = json.loads((tmp_path / "baseline" / "baseline-manifest.json").read_text(encoding="utf-8"))
    assert "disabled_tools" not in manifest["host_evidence"]


def test_finalize_persists_structured_error_class_without_message(tmp_path: Any) -> None:
    """A non-timeout provider error keeps its class, HTTP status and retryable flag, never its message."""
    turn_ids = _turn_ids()
    summary = {
        "observability_schema_version": 2,
        "telemetry_status": "ok",
        "error_turn_ids": ["F015-filler"],
        "error_codes": ["APIError"],
        "error_status_codes": ["503"],
        "error_retryable_flags": ["retryable"],
        "error_message": "SECRET-PROVIDER-MESSAGE-CANARY",
    }
    host_driver.finalize(
        {
            "mode": "offline",
            "arm": "baseline",
            "run_id": "run-error-class",
            "host": "opencode",
            "host_version": "1.18.34",
            "model_id": "anthropic/claude-x",
            "output_dir": str(tmp_path / "baseline"),
            "turns": _observations(turn_ids, model_input=900),
            "observability_summary": summary,
        }
    )
    path = tmp_path / "baseline" / "baseline-manifest.json"
    observability = json.loads(path.read_text(encoding="utf-8"))["host_evidence"]["observability"]
    assert observability["error_codes"] == ["APIError"]
    assert observability["error_status_codes"] == ["503"]
    assert observability["error_retryable_flags"] == ["retryable"]
    assert "SECRET-PROVIDER-MESSAGE-CANARY" not in path.read_text(encoding="utf-8")
