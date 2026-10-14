"""Tests for the deterministic offline evaluator, replay, and the isolated same-host Gate A decision."""

from __future__ import annotations

import pytest
from _ce60_factories import build_catalog, build_manifest, raw_run
from opencode_gate_a.contracts import ContractViolation, EvaluationResult, RunManifest
from opencode_gate_a.evaluator import decide_gate_a, evaluate, replay
from opencode_gate_a.sanitize import sanitize_run

_CATALOG = build_catalog()


def _make_eval(
    *,
    run_id: str,
    arm: str = "baseline",
    host_version: str = "1.4.0",
    model_id: str = "model-x",
    model_input: int = 100,
    correctness: float = 1.0,
    retrieval: tuple[int, int] = (1, 1),
    error_turns: int = 0,
    break_model_input: bool = False,
) -> tuple[RunManifest, EvaluationResult]:
    """Build a manifest and its evaluation for one arm, offline.

    Args:
        run_id: The run id.
        arm: The experiment arm.
        host_version: The host version.
        model_id: The model identity.
        model_input: Model-input tokens per turn.
        correctness: Correctness score per eligible turn.
        retrieval: ``(attempts, successes)`` per turn.
        error_turns: Number of leading turns forced to a host error (reduces completion).
        break_model_input: When true, blank one turn's model_input to invalidate the run metric.

    Returns:
        ``(manifest, evaluation)``.
    """
    manifest = build_manifest(
        run_id=run_id,
        arm=arm,
        host_version=host_version,
        model_id=model_id,
        catalog=_CATALOG,
    )
    run = raw_run(
        run_id=run_id,
        model_input=model_input,
        correctness=correctness,
        retrieval=retrieval,
    )
    for i in range(error_turns):
        run["turns"][i]["host_signal"] = "error"
    if break_model_input:
        run["turns"][0]["metrics"]["model_input"] = {}
    result = sanitize_run(run, manifest_digest=manifest.manifest_digest, catalog=_CATALOG)
    return manifest, evaluate(result, manifest=manifest, catalog=_CATALOG)


class TestEvaluate:
    """Core evaluation and preconditions."""

    def test_metric_matrix_is_derived(self) -> None:
        """A clean run produces full completion, correctness and model-input aggregates."""
        _, ev = _make_eval(run_id="run-1")
        assert ev.turns_attempted == 60
        assert ev.turns_completed == 60
        assert ev.correctness_denominator == 30
        assert ev.correctness_numerator == 30
        assert ev.model_input_tokens_median == 100.0
        assert ev.retrieval_failures == 0

    def test_manifest_result_mismatch_is_rejected(self) -> None:
        """A result bound to a different manifest digest fails closed before any metric."""
        manifest = build_manifest(run_id="run-1", catalog=_CATALOG)
        result = sanitize_run(
            raw_run(run_id="run-1"),
            manifest_digest="sha256:" + ("f" * 64),
            catalog=_CATALOG,
        )
        with pytest.raises(ContractViolation) as exc:
            evaluate(result, manifest=manifest, catalog=_CATALOG)
        assert exc.value.code == "RESULT_NOT_SANITIZED"

    def test_catalog_digest_mismatch_is_rejected(self) -> None:
        """A catalog whose digest differs from the manifest binding fails closed."""
        manifest = build_manifest(run_id="run-1", catalog=_CATALOG)
        result = sanitize_run(
            raw_run(run_id="run-1"),
            manifest_digest=manifest.manifest_digest,
            catalog=_CATALOG,
        )
        other_catalog = build_catalog()  # equal content, but we tamper the manifest below
        tampered = build_manifest(run_id="run-1", catalog=other_catalog)
        object.__setattr__(tampered, "terminal_catalog_digest", "sha256:" + ("9" * 64))
        with pytest.raises(ContractViolation) as exc:
            evaluate(result, manifest=tampered, catalog=other_catalog)
        assert exc.value.code == "TERMINAL_CATALOG_INVALID"

    def test_live_manifest_is_refused_offline(self) -> None:
        """The offline evaluator refuses a live manifest."""
        manifest = build_manifest(run_id="run-1", catalog=_CATALOG)
        object.__setattr__(manifest, "execution_mode", "live")
        result = sanitize_run(
            raw_run(run_id="run-1"),
            manifest_digest=manifest.manifest_digest,
            catalog=_CATALOG,
        )
        with pytest.raises(ContractViolation) as exc:
            evaluate(result, manifest=manifest, catalog=_CATALOG)
        assert exc.value.code == "ADMISSION_POLICY_DENIED"


class TestReplay:
    """Offline replay determinism."""

    def test_replay_is_deterministic(self) -> None:
        """Evaluating the same sanitized run twice yields an identical digest (NFR1.1)."""
        manifest, ev = _make_eval(run_id="run-1")
        run = raw_run(run_id="run-1")
        result = sanitize_run(run, manifest_digest=manifest.manifest_digest, catalog=_CATALOG)
        again = replay(result, manifest=manifest, catalog=_CATALOG, expected=ev)
        assert again.evaluation_digest == ev.evaluation_digest

    def test_replay_rejects_digest_drift(self) -> None:
        """A replay whose recomputed digest differs from the expected one fails closed."""
        manifest, _ev = _make_eval(run_id="run-1", model_input=100)
        # A different run yields a different evaluation and digest.
        _, other = _make_eval(run_id="run-1", model_input=50)
        run = raw_run(run_id="run-1", model_input=100)
        result = sanitize_run(run, manifest_digest=manifest.manifest_digest, catalog=_CATALOG)
        with pytest.raises(ContractViolation) as exc:
            replay(result, manifest=manifest, catalog=_CATALOG, expected=other)
        assert exc.value.code == "REPLAY_DIGEST_MISMATCH"


class TestGateADecision:
    """The isolated same-host Gate A ladder."""

    def test_go_when_isolated_same_host_and_thresholds_pass(self) -> None:
        """baseline vs A, same host/model, completion flat, correctness flat, >=20% input drop -> go."""
        base_m, base = _make_eval(run_id="run-base", arm="baseline", model_input=100)
        treat_m, treat = _make_eval(run_id="run-a", arm="A", model_input=70)
        decision = decide_gate_a(
            base, treat, baseline_manifest=base_m, treatment_manifest=treat_m
        )
        assert decision.outcome == "go"
        assert decision.reasons == ("GATE_PASS",)

    def test_a_plus_d_is_not_evaluable(self) -> None:
        """An A_plus_D treatment arm is never eligible for Gate A (BR8.1)."""
        base_m, base = _make_eval(run_id="run-base", arm="baseline", model_input=100)
        treat_m, treat = _make_eval(run_id="run-ad", arm="A_plus_D", model_input=70)
        decision = decide_gate_a(
            base, treat, baseline_manifest=base_m, treatment_manifest=treat_m
        )
        assert decision.outcome == "not_evaluable"
        assert "GATE_INELIGIBLE_ARM" in decision.reasons

    def test_native_b_is_not_evaluable(self) -> None:
        """A native_B treatment arm is never eligible for Gate A."""
        base_m, base = _make_eval(run_id="run-base", arm="baseline", model_input=100)
        treat_m, treat = _make_eval(run_id="run-b", arm="native_B", model_input=70)
        decision = decide_gate_a(
            base, treat, baseline_manifest=base_m, treatment_manifest=treat_m
        )
        assert decision.outcome == "not_evaluable"
        assert "GATE_INELIGIBLE_ARM" in decision.reasons

    def test_host_mismatch_is_not_evaluable(self) -> None:
        """A different host version between arms is not evaluable (BR7.6)."""
        base_m, base = _make_eval(run_id="run-base", arm="baseline", host_version="1.4.0")
        treat_m, treat = _make_eval(
            run_id="run-a", arm="A", host_version="2.0.0", model_input=70
        )
        decision = decide_gate_a(
            base, treat, baseline_manifest=base_m, treatment_manifest=treat_m
        )
        assert decision.outcome == "not_evaluable"
        assert "GATE_HOST_MISMATCH" in decision.reasons

    def test_model_mismatch_is_not_evaluable(self) -> None:
        """A different model identity between arms is not evaluable (BR7.6)."""
        base_m, base = _make_eval(run_id="run-base", arm="baseline", model_id="model-x")
        treat_m, treat = _make_eval(
            run_id="run-a", arm="A", model_id="model-y", model_input=70
        )
        decision = decide_gate_a(
            base, treat, baseline_manifest=base_m, treatment_manifest=treat_m
        )
        assert decision.outcome == "not_evaluable"
        assert "GATE_MODEL_MISMATCH" in decision.reasons

    def test_essential_metric_invalid_is_not_evaluable(self) -> None:
        """A treatment arm with an invalid model-input metric is not evaluable, never zeroed."""
        base_m, base = _make_eval(run_id="run-base", arm="baseline", model_input=100)
        treat_m, treat = _make_eval(
            run_id="run-a", arm="A", model_input=70, break_model_input=True
        )
        decision = decide_gate_a(
            base, treat, baseline_manifest=base_m, treatment_manifest=treat_m
        )
        assert decision.outcome == "not_evaluable"
        assert "GATE_METRIC_NOT_EVALUABLE" in decision.reasons

    def test_insufficient_input_reduction_is_no_go(self) -> None:
        """A model-input drop below 20% is a no_go (BR8.4)."""
        base_m, base = _make_eval(run_id="run-base", arm="baseline", model_input=100)
        treat_m, treat = _make_eval(run_id="run-a", arm="A", model_input=90)
        decision = decide_gate_a(
            base, treat, baseline_manifest=base_m, treatment_manifest=treat_m
        )
        assert decision.outcome == "no_go"
        assert "GATE_MODEL_INPUT_INSUFFICIENT" in decision.reasons

    def test_correctness_regression_is_no_go(self) -> None:
        """A correctness drop beyond 3 p.p. is a no_go (BR8.3)."""
        base_m, base = _make_eval(run_id="run-base", arm="baseline", correctness=1.0)
        treat_m, treat = _make_eval(
            run_id="run-a", arm="A", model_input=70, correctness=0.0
        )
        decision = decide_gate_a(
            base, treat, baseline_manifest=base_m, treatment_manifest=treat_m
        )
        assert decision.outcome == "no_go"
        assert "GATE_CORRECTNESS_REGRESSION" in decision.reasons

    def test_completion_regression_is_no_go(self) -> None:
        """A drop in completed turns is a no_go (BR8.2)."""
        base_m, base = _make_eval(run_id="run-base", arm="baseline", error_turns=0)
        treat_m, treat = _make_eval(
            run_id="run-a", arm="A", model_input=70, error_turns=5
        )
        decision = decide_gate_a(
            base, treat, baseline_manifest=base_m, treatment_manifest=treat_m
        )
        assert decision.outcome == "no_go"
        assert "GATE_COMPLETION_REGRESSION" in decision.reasons

    def test_retrieval_not_exact_is_no_go(self) -> None:
        """A treatment arm with a retrieval failure is a no_go (BR8.5)."""
        base_m, base = _make_eval(run_id="run-base", arm="baseline", retrieval=(1, 1))
        treat_m, treat = _make_eval(
            run_id="run-a", arm="A", model_input=70, retrieval=(2, 1)
        )
        decision = decide_gate_a(
            base, treat, baseline_manifest=base_m, treatment_manifest=treat_m
        )
        assert decision.outcome == "no_go"
        assert "GATE_RETRIEVAL_INVALID" in decision.reasons
