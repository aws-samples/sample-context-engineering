"""The deterministic offline evaluator and the isolated same-host Gate A decision.

:func:`evaluate` is a pure, offline function: it re-verifies the manifest and result digests, the
sanitisation status, the assignment count and the catalog binding, then derives the per-run metric
matrix (completion_60, correctness_30, model input, retrieval, latency, errors, fallbacks) together with
one run-scope :class:`MetricValidity` per metric. The matrix is hashed into ``evaluation_digest``, so an
identical replay produces an identical digest (NFR1.1). No network, no AWS, no provider, no credential.

:func:`replay` runs :func:`evaluate` again and, when given a prior result, rejects any digest drift.

:func:`decide_gate_a` consolidates a baseline and an OpenCode-A evaluation into a :class:`GateDecision`.
``go`` is returned only when the comparison is a same-host, same-model, isolated baseline-vs-A pair with
every essential metric valid and every threshold met; a missing or incompatible input becomes
``not_evaluable`` and a valid comparison that fails a threshold becomes ``no_go`` — never a silent zero
(BR8).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .contracts import (
    CORRECTNESS_MAX_DROP_PP,
    MODEL_INPUT_MIN_REDUCTION,
    TURN_COUNT,
    ContractViolation,
    EvaluationResult,
    GateDecision,
    MetricValidity,
    RunManifest,
    SanitizedRunResult,
    SanitizedTurnResult,
    TerminalOutcomeCatalog,
)

__all__ = [
    "decide_gate_a",
    "evaluate",
    "replay",
]

# Metrics whose run-scope validity must be ``valid`` in both arms for Gate A to be evaluable (BR8.5).
_ESSENTIAL_METRICS = ("completion", "correctness", "model_input")


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    """Return the nearest-rank percentile of an already-sorted, non-empty sequence.

    Nearest-rank is used (rather than interpolation) so the result is a deterministic member of the data
    set regardless of platform float behaviour.

    Args:
        sorted_values: The values in ascending order; must be non-empty.
        fraction: The percentile as a fraction in [0, 1].

    Returns:
        The percentile value.
    """
    n = len(sorted_values)
    rank = max(1, math.ceil(fraction * n))
    return float(sorted_values[rank - 1])


def _median(sorted_values: Sequence[float]) -> float:
    """Return the median of an already-sorted, non-empty sequence.

    Args:
        sorted_values: The values in ascending order; must be non-empty.

    Returns:
        The median value (mean of the two middle values for an even count).
    """
    n = len(sorted_values)
    mid = n // 2
    if n % 2 == 1:
        return float(sorted_values[mid])
    return (float(sorted_values[mid - 1]) + float(sorted_values[mid])) / 2.0


def _reduce_metric_status(statuses: Sequence[str]) -> str:
    """Reduce turn-scope statuses for one metric into a single run-scope status (BR7.5).

    Only applicable turns are considered. Any ``collection_failed`` dominates and invalidates the run for
    that metric; otherwise ``host_not_exposed`` or ``redacted_by_policy`` invalidate claims; an all-empty
    (fully not-applicable) set yields ``not_applicable``; otherwise ``valid``.

    Args:
        statuses: The turn-scope statuses for applicable turns (``not_applicable`` already excluded).

    Returns:
        The run-scope status.
    """
    if not statuses:
        return "not_applicable"
    if any(s == "collection_failed" for s in statuses):
        return "collection_failed"
    if any(s == "host_not_exposed" for s in statuses):
        return "host_not_exposed"
    if any(s == "redacted_by_policy" for s in statuses):
        return "redacted_by_policy"
    return "valid"


def _turn_statuses(turns: Sequence[SanitizedTurnResult], metric_name: str) -> list[str]:
    """Collect the turn-scope statuses for ``metric_name`` across turns, excluding not-applicable.

    Args:
        turns: The sanitized turns.
        metric_name: The metric to collect.

    Returns:
        The list of statuses for turns where the metric is applicable.
    """
    out: list[str] = []
    for t in turns:
        mv = t.validity_for(metric_name)
        if mv is None or mv.status == "not_applicable":
            continue
        out.append(mv.status)
    return out


def _run_validity(run_id: str, metric_name: str, status: str) -> MetricValidity:
    """Build a run-scope :class:`MetricValidity`.

    Args:
        run_id: The run id.
        metric_name: The metric name.
        status: The reduced run-scope status.

    Returns:
        The run-scope validity record.
    """
    reason = None if status in ("valid", "not_applicable") else "METRIC_INVALID"
    return MetricValidity(
        scope="run",
        run_id=run_id,
        metric_name=metric_name,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        turn_id=None,
        reason_code=reason,
    )


def _verify_preconditions(
    result: SanitizedRunResult,
    manifest: RunManifest,
    catalog: TerminalOutcomeCatalog,
) -> None:
    """Verify every evaluator precondition fail-closed before any metric is derived.

    Args:
        result: The sanitized run result.
        manifest: The run manifest.
        catalog: The terminal outcome catalog.

    Raises:
        ContractViolation: ``ADMISSION_POLICY_DENIED`` for a live manifest (the offline evaluator never
            evaluates live); ``RESULT_NOT_SANITIZED`` for a sanitation/identity/digest mismatch;
            ``TERMINAL_CATALOG_INVALID`` when the catalog digest does not match the manifest.
    """
    if manifest.execution_mode == "live":
        raise ContractViolation(
            "ADMISSION_POLICY_DENIED", "the offline evaluator does not evaluate live runs"
        )
    if result.sanitization_status != "passed":
        raise ContractViolation(
            "RESULT_NOT_SANITIZED", "result sanitization_status is not passed"
        )
    if result.run_id != manifest.run_id:
        raise ContractViolation("RESULT_NOT_SANITIZED", "result run_id does not match the manifest")
    if result.manifest_digest != manifest.manifest_digest:
        raise ContractViolation(
            "RESULT_NOT_SANITIZED", "result manifest_digest does not match the manifest"
        )
    if result.result_digest != result.compute_digest():
        raise ContractViolation("RESULT_NOT_SANITIZED", "result_digest does not match the content")
    if catalog.digest != manifest.terminal_catalog_digest:
        raise ContractViolation(
            "TERMINAL_CATALOG_INVALID", "catalog digest does not match the manifest binding"
        )
    manifest_turn_ids = {a.turn_id for a in manifest.turn_assignments}
    result_turn_ids = {t.turn_id for t in result.turn_results}
    if manifest_turn_ids != result_turn_ids:
        raise ContractViolation(
            "RESULT_NOT_SANITIZED", "result turn ids do not match the manifest assignments"
        )
    catalog_turn_ids = {d.turn_id for d in catalog.turn_definitions}
    if result_turn_ids != catalog_turn_ids:
        raise ContractViolation(
            "TERMINAL_CATALOG_INVALID", "result turn ids do not match the catalog"
        )


def evaluate(
    result: SanitizedRunResult,
    *,
    manifest: RunManifest,
    catalog: TerminalOutcomeCatalog,
) -> EvaluationResult:
    """Evaluate a sanitized offline/replay run into a deterministic :class:`EvaluationResult`.

    Args:
        result: The sanitized run result (the only accepted input shape).
        manifest: The run manifest the result is bound to.
        catalog: The terminal outcome catalog that classified the turns.

    Returns:
        The evaluation result with the full metric matrix, run-scope validity and a stable digest.

    Raises:
        ContractViolation: If any precondition in :func:`_verify_preconditions` fails.
    """
    _verify_preconditions(result, manifest, catalog)

    turns = result.turn_results
    run_id = result.run_id

    turns_completed = sum(1 for t in turns if t.completion_status == "completed")
    error_count = sum(1 for t in turns if t.error_code is not None)
    fallback_count = sum(1 for t in turns if t.fallback_status == "declared_fallback")

    # Run-scope validity per metric, reduced from the turn-scope records.
    validity_by_metric = {
        name: _run_validity(run_id, name, _reduce_metric_status(_turn_statuses(turns, name)))
        for name in ("completion", "correctness", "model_input", "retrieval", "latency")
    }

    correctness_numerator, correctness_denominator = _correctness(turns, validity_by_metric)
    mi_total, mi_median = _model_input(turns, validity_by_metric)
    rt_attempts, rt_successes, rt_failures = _retrieval(turns, validity_by_metric)
    lat_p50, lat_p95 = _latency(turns, validity_by_metric)

    metric_validity = tuple(validity_by_metric[name] for name in sorted(validity_by_metric))

    # Compute the digest over the matrix, then construct the frozen result (which re-verifies it).
    provisional = EvaluationResult.__new__(EvaluationResult)
    object.__setattr__(provisional, "run_id", run_id)
    object.__setattr__(provisional, "manifest_digest", manifest.manifest_digest)
    object.__setattr__(provisional, "arm", manifest.arm)
    object.__setattr__(provisional, "turns_attempted", TURN_COUNT)
    object.__setattr__(provisional, "turns_completed", turns_completed)
    object.__setattr__(provisional, "error_count", error_count)
    object.__setattr__(provisional, "fallback_count", fallback_count)
    object.__setattr__(provisional, "metric_validity", metric_validity)
    object.__setattr__(provisional, "correctness_numerator", correctness_numerator)
    object.__setattr__(provisional, "correctness_denominator", correctness_denominator)
    object.__setattr__(provisional, "model_input_tokens_total", mi_total)
    object.__setattr__(provisional, "model_input_tokens_median", mi_median)
    object.__setattr__(provisional, "retrieval_attempts", rt_attempts)
    object.__setattr__(provisional, "retrieval_successes", rt_successes)
    object.__setattr__(provisional, "retrieval_failures", rt_failures)
    object.__setattr__(provisional, "latency_p50_ms", lat_p50)
    object.__setattr__(provisional, "latency_p95_ms", lat_p95)
    evaluation_digest = provisional.compute_digest()
    object.__setattr__(provisional, "evaluation_digest", evaluation_digest)
    provisional.__post_init__()
    return provisional


def _correctness(
    turns: Sequence[SanitizedTurnResult],
    validity: dict[str, MetricValidity],
) -> tuple[int | None, int | None]:
    """Compute the correctness numerator/denominator over valid eligible turns (BR7.4).

    Args:
        turns: The sanitized turns.
        validity: The run-scope validity by metric.

    Returns:
        ``(numerator, denominator)``, or ``(None, None)`` when the run-scope correctness is not valid.
    """
    if validity["correctness"].status != "valid":
        return None, None
    numerator = 0
    denominator = 0
    for t in turns:
        if not t.correctness_eligible:
            continue
        mv = t.validity_for("correctness")
        if mv is None or mv.status != "valid" or t.correctness_score is None:
            continue
        denominator += 1
        if t.correctness_score >= 1.0:
            numerator += 1
    return numerator, denominator


def _model_input(
    turns: Sequence[SanitizedTurnResult],
    validity: dict[str, MetricValidity],
) -> tuple[int | None, float | None]:
    """Compute total and median model-input tokens over valid turns.

    Args:
        turns: The sanitized turns.
        validity: The run-scope validity by metric.

    Returns:
        ``(total, median)``, or ``(None, None)`` when the run-scope model_input is not valid.
    """
    if validity["model_input"].status != "valid":
        return None, None
    values = [t.model_input_tokens for t in turns if _metric_valid(t, "model_input")]
    if not values:
        return None, None
    ordered = sorted(values)
    return sum(ordered), _median(ordered)


def _retrieval(
    turns: Sequence[SanitizedTurnResult],
    validity: dict[str, MetricValidity],
) -> tuple[int | None, int | None, int | None]:
    """Compute retrieval attempts/successes/failures over valid turns.

    Args:
        turns: The sanitized turns.
        validity: The run-scope validity by metric.

    Returns:
        ``(attempts, successes, failures)``, or ``(None, None, None)`` when retrieval is not valid.
    """
    if validity["retrieval"].status != "valid":
        return None, None, None
    attempts = 0
    successes = 0
    seen = False
    for t in turns:
        mv = t.validity_for("retrieval")
        if mv is None or mv.status != "valid":
            continue
        if t.retrieval_attempts is None or t.retrieval_successes is None:
            continue
        seen = True
        attempts += t.retrieval_attempts
        successes += t.retrieval_successes
    if not seen:
        return None, None, None
    return attempts, successes, attempts - successes


def _latency(
    turns: Sequence[SanitizedTurnResult],
    validity: dict[str, MetricValidity],
) -> tuple[float | None, float | None]:
    """Compute p50/p95 latency over valid turns.

    Args:
        turns: The sanitized turns.
        validity: The run-scope validity by metric.

    Returns:
        ``(p50, p95)``, or ``(None, None)`` when latency is not valid.
    """
    if validity["latency"].status != "valid":
        return None, None
    values = [t.latency_ms for t in turns if _metric_valid(t, "latency")]
    if not values:
        return None, None
    ordered = sorted(values)
    return _percentile(ordered, 0.50), _percentile(ordered, 0.95)


def _metric_valid(turn: SanitizedTurnResult, metric_name: str) -> bool:
    """Return whether ``turn`` carries a valid turn-scope validity for ``metric_name``.

    Args:
        turn: The sanitized turn.
        metric_name: The metric to check.

    Returns:
        ``True`` when the turn's validity for the metric exists and is ``valid``.
    """
    mv = turn.validity_for(metric_name)
    return mv is not None and mv.status == "valid"


def replay(
    result: SanitizedRunResult,
    *,
    manifest: RunManifest,
    catalog: TerminalOutcomeCatalog,
    expected: EvaluationResult | None = None,
) -> EvaluationResult:
    """Re-evaluate a run offline and, when ``expected`` is given, reject any digest drift (NFR1.1).

    Args:
        result: The sanitized run result.
        manifest: The run manifest.
        catalog: The terminal outcome catalog.
        expected: A prior evaluation to compare against, if any.

    Returns:
        The freshly computed evaluation result.

    Raises:
        ContractViolation: ``REPLAY_DIGEST_MISMATCH`` if ``expected`` is given and the recomputed digest
            differs; plus any precondition failure from :func:`evaluate`.
    """
    recomputed = evaluate(result, manifest=manifest, catalog=catalog)
    if expected is not None and recomputed.evaluation_digest != expected.evaluation_digest:
        raise ContractViolation(
            "REPLAY_DIGEST_MISMATCH", "replay produced a different evaluation digest"
        )
    return recomputed


def _eligibility_reasons(
    baseline: EvaluationResult,
    treatment: EvaluationResult,
    baseline_manifest: RunManifest,
    treatment_manifest: RunManifest,
) -> list[str]:
    """Collect the Gate A eligibility failures (arm isolation, host and model) (BR8.1, BR7.6).

    Args:
        baseline: The baseline evaluation.
        treatment: The OpenCode A evaluation.
        baseline_manifest: The baseline manifest.
        treatment_manifest: The treatment manifest.

    Returns:
        A list of stable reason codes; empty when the comparison is eligible.
    """
    reasons: list[str] = []
    # Only an isolated baseline-vs-A pair is eligible; A_plus_D and native_B are already excluded by
    # requiring the treatment arm to be exactly "A" (BR8.1).
    if baseline.arm != "baseline" or treatment.arm != "A":
        reasons.append("GATE_INELIGIBLE_ARM")
    if (
        baseline_manifest.host != treatment_manifest.host
        or baseline_manifest.host_version != treatment_manifest.host_version
    ):
        reasons.append("GATE_HOST_MISMATCH")
    if baseline_manifest.model_id != treatment_manifest.model_id:
        reasons.append("GATE_MODEL_MISMATCH")
    return reasons


def _essential_valid(baseline: EvaluationResult, treatment: EvaluationResult) -> bool:
    """Return whether every essential metric is run-scope valid in both arms (BR8.5).

    Args:
        baseline: The baseline evaluation.
        treatment: The treatment evaluation.

    Returns:
        ``True`` when completion, correctness and model_input are all valid in both arms.
    """
    for ev in (baseline, treatment):
        for name in _ESSENTIAL_METRICS:
            mv = ev.run_validity_for(name)
            if mv is None or mv.status != "valid":
                return False
    return True


def _threshold_reasons(baseline: EvaluationResult, treatment: EvaluationResult) -> list[str]:
    """Collect threshold and evidence failures for an eligible, essential-valid comparison (BR8.2-BR8.5).

    Args:
        baseline: The baseline evaluation.
        treatment: The treatment evaluation.

    Returns:
        A list of stable ``no_go`` reason codes; empty when every threshold and evidence check passes.
    """
    reasons: list[str] = []

    # BR8.2 — completion must not regress.
    if treatment.turns_completed < baseline.turns_completed:
        reasons.append("GATE_COMPLETION_REGRESSION")

    # BR8.3 — correctness may drop at most 3 percentage points.
    base_rate = _rate(baseline.correctness_numerator, baseline.correctness_denominator)
    treat_rate = _rate(treatment.correctness_numerator, treatment.correctness_denominator)
    if base_rate is None or treat_rate is None:
        reasons.append("GATE_METRIC_NOT_EVALUABLE")
    elif (base_rate - treat_rate) * 100.0 > CORRECTNESS_MAX_DROP_PP:
        reasons.append("GATE_CORRECTNESS_REGRESSION")

    # BR8.4 — median model-input tokens must drop by at least 20%.
    base_mi = baseline.model_input_tokens_median
    treat_mi = treatment.model_input_tokens_median
    if base_mi is None or treat_mi is None or base_mi <= 0:
        reasons.append("GATE_METRIC_NOT_EVALUABLE")
    elif (base_mi - treat_mi) / base_mi < MODEL_INPUT_MIN_REDUCTION:
        reasons.append("GATE_MODEL_INPUT_INSUFFICIENT")

    # BR8.5 — retrieval must be present, valid and exact (zero failures) in the treatment arm.
    rt = treatment.run_validity_for("retrieval")
    if (
        rt is None
        or rt.status != "valid"
        or treatment.retrieval_failures is None
        or treatment.retrieval_failures != 0
    ):
        reasons.append("GATE_RETRIEVAL_INVALID")
    return reasons


def _rate(numerator: int | None, denominator: int | None) -> float | None:
    """Return ``numerator / denominator`` as a rate, or ``None`` when either is unavailable or zero.

    Args:
        numerator: The numerator, or ``None``.
        denominator: The denominator, or ``None``.

    Returns:
        The rate in [0, 1], or ``None``.
    """
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def decide_gate_a(
    baseline: EvaluationResult,
    treatment: EvaluationResult,
    *,
    baseline_manifest: RunManifest,
    treatment_manifest: RunManifest,
) -> GateDecision:
    """Decide Gate A for the isolated OpenCode A arm against a same-host baseline (BR8).

    The outcome ladder is strict and non-compensating:

    * ineligible arm composition, host or model mismatch -> ``not_evaluable``;
    * an essential metric (completion, correctness, model input) not valid in either arm ->
      ``not_evaluable``;
    * a valid, eligible comparison that fails any threshold or the retrieval/isolation evidence ->
      ``no_go``;
    * only an eligible comparison with every essential metric valid and every threshold met -> ``go``.

    Args:
        baseline: The baseline (A/B/D off) evaluation.
        treatment: The OpenCode A evaluation.
        baseline_manifest: The baseline manifest.
        treatment_manifest: The treatment manifest.

    Returns:
        The :class:`GateDecision`, carrying stable reason codes and the two evaluation digests.
    """
    evidence = (baseline.evaluation_digest, treatment.evaluation_digest)

    eligibility = _eligibility_reasons(baseline, treatment, baseline_manifest, treatment_manifest)
    if eligibility:
        return GateDecision(
            outcome="not_evaluable",
            reasons=tuple(eligibility),
            evidence_digests=evidence,
        )

    if not _essential_valid(baseline, treatment):
        return GateDecision(
            outcome="not_evaluable",
            reasons=("GATE_METRIC_NOT_EVALUABLE",),
            evidence_digests=evidence,
        )

    reasons = _threshold_reasons(baseline, treatment)
    if reasons:
        # Deduplicate while preserving order so the reason list is stable.
        seen: list[str] = []
        for r in reasons:
            if r not in seen:
                seen.append(r)
        outcome = "not_evaluable" if seen == ["GATE_METRIC_NOT_EVALUABLE"] else "no_go"
        return GateDecision(
            outcome=outcome,  # type: ignore[arg-type]
            reasons=tuple(seen),
            evidence_digests=evidence,
        )

    return GateDecision(
        outcome="go",
        reasons=("GATE_PASS",),
        evidence_digests=evidence,
    )
