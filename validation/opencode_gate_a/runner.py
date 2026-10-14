"""Offline orchestration for the CE-60 Gate A layer: assignments, manifests, files, and a safe CLI.

This module is the thin, side-effect-light shell around the pure offline layer
(:mod:`contracts`, :mod:`sanitize`, :mod:`evaluator`). It builds the canonical 60-turn assignment
set, assembles fail-closed :class:`~opencode_gate_a.contracts.RunManifest` objects for the pilot and
full plans, drives sanitisation and evaluation over files, and renders the isolated same-host Gate A
decision — all without the network, AWS, a provider or a live host.

Three safety properties are load-bearing:

* **Never truncate.** :func:`build_assignments` yields exactly 60 assignments or raises; it never
  silently drops or fabricates turns. When no explicit turn ids are given and the CE-60 scenario is not
  importable, it fails closed and asks the caller for the ids.
* **Never persist a payload.** :func:`sanitize_file` and :func:`evaluate_file` write only the
  payload-free sanitized/evaluation projections produced by the pure layer; a raw prompt, response,
  tool payload, credential or log never reaches an output file.
* **Never start a live run by accident.** The CLI performs no model/host execution by default, and any
  ``--execute-live`` request fails closed with the stable code
  :data:`LIVE_EXECUTION_REQUIRES_HOST_DRIVER` rather than beginning an incomplete benchmark.

Everything here is stdlib-only. Digests are delegated to :mod:`canonical`; no value in a manifest is
invented — identity, admission, budget, price and statistical-plan fields come from the caller and are
validated by the existing contracts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from typing import Any, Final

from .canonical import digest
from .contracts import (
    ARMS,
    BACKEND_IDS,
    EXECUTION_MODES,
    TURN_COUNT,
    ContractViolation,
    EvaluationResult,
    GateDecision,
    MetricValidity,
    RunManifest,
    SanitizedRunResult,
    SanitizedTurnResult,
    StatPlan,
    TerminalOutcomeCatalog,
    TerminalTurnDefinition,
    TurnAssignment,
)
from .evaluator import decide_gate_a, evaluate
from .sanitize import sanitize_run

__all__ = [
    "LIVE_EXECUTION_REQUIRES_HOST_DRIVER",
    "SCOPES",
    "LiveExecutionUnavailable",
    "build_assignments",
    "build_manifest",
    "catalog_from_dict",
    "catalog_to_dict",
    "decide_gate_a_files",
    "evaluate_file",
    "load_catalog",
    "load_ce60_turn_ids",
    "main",
    "manifest_from_dict",
    "manifest_to_dict",
    "sanitize_file",
    "write_catalog",
]

LIVE_EXECUTION_REQUIRES_HOST_DRIVER: Final[str] = "LIVE_EXECUTION_REQUIRES_HOST_DRIVER"
"""Stable code the runner emits instead of starting a live run: no host driver ships in this layer."""

SCOPES: Final[frozenset[str]] = frozenset({"pilot", "full"})
"""The two run plans. Neither changes the 60-turn manifest shape; the label travels for traceability."""

_DEFAULT_FAULT_PROFILE_ID: Final[str] = "ce60-offline-no-fault"
"""The offline default fault profile: no fault injection. Its digest is derived, not invented."""


class LiveExecutionUnavailable(RuntimeError):
    """Raised when a live execution is requested but no host driver exists in this layer.

    Attributes:
        code: Always :data:`LIVE_EXECUTION_REQUIRES_HOST_DRIVER`, so callers branch on the code.
    """

    def __init__(self, message: str = LIVE_EXECUTION_REQUIRES_HOST_DRIVER) -> None:
        """Wrap the stable live-unavailable code.

        Args:
            message: The message; defaults to the stable code itself.
        """
        super().__init__(message)
        self.code = LIVE_EXECUTION_REQUIRES_HOST_DRIVER


# --- Turn-id resolution and assignments ------------------------------------------------------------


def load_ce60_turn_ids() -> list[str] | None:
    """Best-effort load of the 60 canonical CE-60 turn ids from the community scenario, if importable.

    The scenario module lives in a sibling validation package and pulls heavy, optional dependencies
    (an agent framework) through its tool suite. This function therefore imports it defensively: any
    failure — module not found, dependency missing, or a turn count other than 60 — yields ``None`` so
    the caller falls back to explicit ids rather than guessing.

    Returns:
        The 60 unique turn-id strings in script order, or ``None`` when the scenario cannot provide
        exactly 60 unique ids.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    plugin_root = os.path.normpath(os.path.join(here, "..", "community-plugin-A-B-D"))
    if not os.path.isdir(plugin_root):
        return None
    added = plugin_root not in sys.path
    if added:
        sys.path.insert(0, plugin_root)
    try:
        from src import scenario  # type: ignore[import-not-found]

        ids = [turn.label for turn in scenario.turns(total=TURN_COUNT)]
    except Exception:  # noqa: BLE001 - defensive: any import/attr/dep failure means fall back to explicit ids
        return None
    finally:
        if added and plugin_root in sys.path:
            sys.path.remove(plugin_root)
    if len(ids) != TURN_COUNT or len(set(ids)) != TURN_COUNT:
        return None
    return ids


def build_assignments(
    turn_ids: Sequence[str] | None = None,
    *,
    order_seed: str,
    fault_profile_id: str = _DEFAULT_FAULT_PROFILE_ID,
    fault_profile_digest: str | None = None,
) -> tuple[TurnAssignment, ...]:
    """Build exactly 60 canonical, deterministic turn assignments — or fail closed.

    Turn ids are resolved from, in order: the explicit ``turn_ids`` argument, then the importable CE-60
    scenario. If neither yields exactly 60 unique ids the function raises; it never truncates a longer
    list, pads a shorter one, or fabricates ids. Each per-turn ``seed`` is derived deterministically
    from ``order_seed`` and the turn id, so the same inputs always produce the same assignment digests.

    Args:
        turn_ids: The 60 explicit turn ids, or ``None`` to resolve them from the CE-60 scenario.
        order_seed: The counterbalanced execution order seed; must be a non-empty string.
        fault_profile_id: The fault profile applied to every turn. Defaults to the offline no-fault
            profile.
        fault_profile_digest: The digest pinning the fault profile bytes. When ``None`` it is derived
            deterministically from ``fault_profile_id``.

    Returns:
        A tuple of exactly 60 :class:`TurnAssignment`, ordered by the resolved turn ids.

    Raises:
        ContractViolation: ``RUN_MANIFEST_INVALID`` when ``order_seed`` is empty, when turn ids cannot
            be resolved, or when the resolved ids are not exactly 60 unique non-empty strings.
    """
    if not (isinstance(order_seed, str) and order_seed.strip()):
        raise ContractViolation("RUN_MANIFEST_INVALID", "order_seed must be a non-empty string")

    resolved: list[str] | None
    if turn_ids is not None:
        resolved = list(turn_ids)
    else:
        resolved = load_ce60_turn_ids()
        if resolved is None:
            raise ContractViolation(
                "RUN_MANIFEST_INVALID",
                "CE-60 turn ids are unavailable; pass turn_ids explicitly (never truncate)",
            )

    if len(resolved) != TURN_COUNT:
        raise ContractViolation(
            "RUN_MANIFEST_INVALID",
            f"expected exactly {TURN_COUNT} turn ids, got {len(resolved)} (never truncate)",
        )
    if any(not (isinstance(tid, str) and tid.strip()) for tid in resolved):
        raise ContractViolation(
            "RUN_MANIFEST_INVALID", "every turn id must be a non-empty string"
        )
    if len(set(resolved)) != TURN_COUNT:
        raise ContractViolation(
            "RUN_MANIFEST_INVALID", "turn ids must be unique across all 60 turns"
        )

    fault_digest = (
        fault_profile_digest
        if fault_profile_digest is not None
        else digest(["fault-profile", fault_profile_id])
    )
    return tuple(
        TurnAssignment(
            turn_id=turn_id,
            seed=digest(["seed", order_seed, turn_id]),
            fault_profile_id=fault_profile_id,
            fault_profile_digest=fault_digest,
        )
        for turn_id in resolved
    )


# --- Manifest assembly -----------------------------------------------------------------------------


def build_manifest(
    *,
    run_id: str,
    arm: str,
    host: str,
    host_version: str,
    model_id: str,
    capability_manifest_digest: str,
    fixture_catalog_digest: str,
    terminal_catalog_digest: str,
    order_seed: str,
    execution_mode: str = "offline",
    backend_id: str = "local-fake",
    scope: str = "pilot",
    turn_ids: Sequence[str] | None = None,
    turn_assignments: Sequence[TurnAssignment] | None = None,
    budget_status: str = "absent",
    sample_plan_status: str = "absent",
    execution_admission_id: str | None = None,
    admission_policy_digest: str | None = None,
    budget_ceiling_units: float | None = None,
    budget_currency_or_unit: str | None = None,
    price_provenance: str | None = None,
    stat_plan: StatPlan | None = None,
) -> RunManifest:
    """Assemble a fail-closed :class:`RunManifest` for the pilot or full plan.

    The function does not invent any provenance value: it threads the caller's identity, catalog
    digests and (for live) admission/budget/price/stat-plan fields straight into
    :meth:`RunManifest.build`, where the existing contract enforces every invariant. Offline and replay
    manifests are local-only by construction (the contract rejects a remote backend or any live-only
    field); a live manifest requires an approved admission, a configured budget with a price provenance
    and an approved :class:`StatPlan`, exactly as the contract demands.

    Args:
        run_id: The opaque run id.
        arm: The experiment arm (``baseline``/``A``/``A_plus_D``/``native_B``).
        host: The host under test.
        host_version: The host version string.
        model_id: The model identity.
        capability_manifest_digest: Digest of the capability manifest.
        fixture_catalog_digest: Digest of the fixture catalog.
        terminal_catalog_digest: Digest of the terminal outcome catalog.
        order_seed: The counterbalanced order seed used to derive per-turn seeds.
        execution_mode: ``offline``, ``replay`` or ``live``.
        backend_id: The backend; local for offline/replay, ``remote-explicit`` for live.
        scope: The run plan, ``pilot`` or ``full`` (traceability only; the manifest shape is identical).
        turn_ids: Explicit turn ids, used when ``turn_assignments`` is not given.
        turn_assignments: Pre-built assignments; when ``None`` they are derived from ``turn_ids`` or the
            CE-60 scenario via :func:`build_assignments`.
        budget_status: The budget status.
        sample_plan_status: The sample-plan status.
        execution_admission_id: The live admission id.
        admission_policy_digest: The live admission policy digest.
        budget_ceiling_units: The live budget ceiling.
        budget_currency_or_unit: The live budget unit.
        price_provenance: The live price provenance.
        stat_plan: The approved live :class:`StatPlan`.

    Returns:
        A valid :class:`RunManifest`.

    Raises:
        ContractViolation: ``RUN_MANIFEST_INVALID`` for an unknown scope or any contract failure
            surfaced by :meth:`RunManifest.build`.
    """
    if scope not in SCOPES:
        raise ContractViolation(
            "RUN_MANIFEST_INVALID", f"unknown scope: {scope!r} (expected one of {sorted(SCOPES)})"
        )
    assignments = (
        tuple(turn_assignments)
        if turn_assignments is not None
        else build_assignments(turn_ids, order_seed=order_seed)
    )
    return RunManifest.build(
        run_id=run_id,
        host=host,
        host_version=host_version,
        model_id=model_id,
        backend_id=backend_id,  # type: ignore[arg-type]
        arm=arm,  # type: ignore[arg-type]
        capability_manifest_digest=capability_manifest_digest,
        fixture_catalog_digest=fixture_catalog_digest,
        terminal_catalog_digest=terminal_catalog_digest,
        execution_mode=execution_mode,  # type: ignore[arg-type]
        order_seed=order_seed,
        turn_assignments=assignments,
        budget_status=budget_status,  # type: ignore[arg-type]
        sample_plan_status=sample_plan_status,  # type: ignore[arg-type]
        execution_admission_id=execution_admission_id,
        admission_policy_digest=admission_policy_digest,
        budget_ceiling_units=budget_ceiling_units,
        budget_currency_or_unit=budget_currency_or_unit,
        price_provenance=price_provenance,
        stat_plan=stat_plan,
    )


# --- (De)serialisation: catalog, stat plan, manifest, sanitized result, evaluation ----------------


def catalog_to_dict(catalog: TerminalOutcomeCatalog) -> dict[str, Any]:
    """Project a terminal catalog to a JSON-safe dict, digest included.

    Args:
        catalog: The catalog.

    Returns:
        A dict carrying the catalog id, version, ordered turn definitions and the pinning digest.
    """
    body = catalog.content_binding()
    body["digest"] = catalog.digest
    return body


def catalog_from_dict(data: dict[str, Any]) -> TerminalOutcomeCatalog:
    """Rebuild a terminal catalog from :func:`catalog_to_dict`, re-verifying its digest.

    Args:
        data: The serialized catalog.

    Returns:
        The reconstructed, digest-verified :class:`TerminalOutcomeCatalog`.

    Raises:
        ContractViolation: ``TERMINAL_CATALOG_INVALID`` when the stored digest does not match the
            rebuilt content.
    """
    definitions = [
        TerminalTurnDefinition(
            turn_id=d["turn_id"],
            terminal_matcher=d["terminal_matcher"],
            accepted_states=tuple(d["accepted_states"]),
            fallback_policy=d["fallback_policy"],
            correctness_eligible=bool(d["correctness_eligible"]),
        )
        for d in data["turn_definitions"]
    ]
    catalog = TerminalOutcomeCatalog.build(
        terminal_catalog_id=data["terminal_catalog_id"],
        version=data["version"],
        turn_definitions=definitions,
    )
    stored = data.get("digest")
    if stored is not None and stored != catalog.digest:
        raise ContractViolation(
            "TERMINAL_CATALOG_INVALID", "stored catalog digest does not match its content"
        )
    return catalog


def _stat_plan_to_dict(plan: StatPlan) -> dict[str, Any]:
    """Project a stat plan to a JSON-safe dict with its digest."""
    body = plan.content_binding()
    body["digest"] = plan.digest
    return body


def _stat_plan_from_dict(data: dict[str, Any]) -> StatPlan:
    """Rebuild a stat plan, re-verifying its digest.

    Raises:
        ContractViolation: ``STAT_PLAN_INVALID`` when the stored digest does not match the content.
    """
    plan = StatPlan.build(
        confidence_level=float(data["confidence_level"]),
        paired_unit=data["paired_unit"],
        method=data["method"],
        decision_rule=data["decision_rule"],
        minimum_paired_repetitions=int(data["minimum_paired_repetitions"]),
        approval_id=data["approval_id"],
        approval_actor=data["approval_actor"],
    )
    stored = data.get("digest")
    if stored is not None and stored != plan.digest:
        raise ContractViolation(
            "STAT_PLAN_INVALID", "stored stat plan digest does not match its content"
        )
    return plan


def manifest_to_dict(
    manifest: RunManifest,
    *,
    scope: str | None = None,
    host_evidence: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Project a manifest to a JSON-safe dict for persistence.

    The ``scope`` label, when given, is written as a sibling of the contract fields and is deliberately
    not part of the manifest digest binding — it is traceability metadata, not provenance. ``host_evidence``
    is likewise an additive, payload-free observation block and is not used to alter the manifest digest.

    Args:
        manifest: The manifest.
        scope: The optional run plan label to record alongside the manifest.

    Returns:
        A JSON-safe dict carrying every manifest field, including the assignments and the digest.
    """
    body: dict[str, Any] = {
        "run_id": manifest.run_id,
        "host": manifest.host,
        "host_version": manifest.host_version,
        "model_id": manifest.model_id,
        "backend_id": manifest.backend_id,
        "arm": manifest.arm,
        "capability_manifest_digest": manifest.capability_manifest_digest,
        "fixture_catalog_digest": manifest.fixture_catalog_digest,
        "terminal_catalog_digest": manifest.terminal_catalog_digest,
        "execution_mode": manifest.execution_mode,
        "order_seed": manifest.order_seed,
        "turn_assignments": [
            {
                "turn_id": a.turn_id,
                "seed": a.seed,
                "fault_profile_id": a.fault_profile_id,
                "fault_profile_digest": a.fault_profile_digest,
            }
            for a in manifest.turn_assignments
        ],
        "budget_status": manifest.budget_status,
        "sample_plan_status": manifest.sample_plan_status,
        "execution_admission_id": manifest.execution_admission_id,
        "admission_policy_digest": manifest.admission_policy_digest,
        "budget_ceiling_units": manifest.budget_ceiling_units,
        "budget_currency_or_unit": manifest.budget_currency_or_unit,
        "price_provenance": manifest.price_provenance,
        "stat_plan": None if manifest.stat_plan is None else _stat_plan_to_dict(manifest.stat_plan),
        "manifest_digest": manifest.manifest_digest,
    }
    if scope is not None:
        body["scope"] = scope
    if host_evidence is not None:
        body["host_evidence"] = dict(host_evidence)
    return body


def manifest_from_dict(data: dict[str, Any]) -> RunManifest:
    """Rebuild a manifest from :func:`manifest_to_dict`, re-verifying its digest.

    The manifest is reconstructed through :meth:`RunManifest.build`, which recomputes the assignment and
    manifest digests from the stored content. The stored ``manifest_digest`` is then compared to the
    recomputed one, so any post-hoc edit to a provenance field (including an assignment seed) is caught.

    Args:
        data: The serialized manifest.

    Returns:
        The reconstructed, digest-verified :class:`RunManifest`.

    Raises:
        ContractViolation: ``RUN_MANIFEST_INVALID`` when the stored digest does not match the rebuilt
            content, plus any contract failure surfaced during the rebuild.
    """
    assignments = tuple(
        TurnAssignment(
            turn_id=a["turn_id"],
            seed=a["seed"],
            fault_profile_id=a["fault_profile_id"],
            fault_profile_digest=a["fault_profile_digest"],
        )
        for a in data["turn_assignments"]
    )
    stat_plan_data = data.get("stat_plan")
    manifest = RunManifest.build(
        run_id=data["run_id"],
        host=data["host"],
        host_version=data["host_version"],
        model_id=data["model_id"],
        backend_id=data["backend_id"],
        arm=data["arm"],
        capability_manifest_digest=data["capability_manifest_digest"],
        fixture_catalog_digest=data["fixture_catalog_digest"],
        terminal_catalog_digest=data["terminal_catalog_digest"],
        execution_mode=data["execution_mode"],
        order_seed=data["order_seed"],
        turn_assignments=assignments,
        budget_status=data.get("budget_status", "absent"),
        sample_plan_status=data.get("sample_plan_status", "absent"),
        execution_admission_id=data.get("execution_admission_id"),
        admission_policy_digest=data.get("admission_policy_digest"),
        budget_ceiling_units=data.get("budget_ceiling_units"),
        budget_currency_or_unit=data.get("budget_currency_or_unit"),
        price_provenance=data.get("price_provenance"),
        stat_plan=None if stat_plan_data is None else _stat_plan_from_dict(stat_plan_data),
    )
    stored = data.get("manifest_digest")
    if stored is not None and stored != manifest.manifest_digest:
        raise ContractViolation(
            "RUN_MANIFEST_INVALID", "stored manifest_digest does not match the manifest content"
        )
    return manifest


def _metric_validity_from_dict(data: dict[str, Any]) -> MetricValidity:
    """Rebuild a :class:`MetricValidity` from its canonical dict."""
    return MetricValidity(
        scope=data["scope"],
        run_id=data["run_id"],
        metric_name=data["metric_name"],
        status=data["status"],
        turn_id=data.get("turn_id"),
        reason_code=data.get("reason_code"),
    )


def _result_to_dict(result: SanitizedRunResult) -> dict[str, Any]:
    """Project a sanitized run result to a JSON-safe, payload-free dict.

    Only the sanitized projection is emitted (the same fields the digest covers), so no raw prompt,
    response, tool payload, credential or log can appear in the file.

    Args:
        result: The sanitized run result.

    Returns:
        A payload-free dict carrying the run identity, host evidence, the 60 turn projections and the
        result digest.
    """
    body = result.binding()
    body["result_digest"] = result.result_digest
    return body


def _result_from_dict(data: dict[str, Any]) -> SanitizedRunResult:
    """Rebuild a sanitized run result from :func:`_result_to_dict`, re-verifying its digest.

    Args:
        data: The serialized sanitized result.

    Returns:
        The reconstructed, digest-verified :class:`SanitizedRunResult`.

    Raises:
        ContractViolation: ``RESULT_NOT_SANITIZED`` when the stored digest does not match the content.
    """
    turns = tuple(
        SanitizedTurnResult(
            turn_id=t["turn_id"],
            terminal_state=t["terminal_state"],
            completion_status=t["completion_status"],
            correctness_eligible=bool(t["correctness_eligible"]),
            metric_validity=tuple(
                _metric_validity_from_dict(mv) for mv in t["metric_validity"]
            ),
            correctness_score=t.get("correctness_score"),
            model_input_tokens=t.get("model_input_tokens"),
            retrieval_attempts=t.get("retrieval_attempts"),
            retrieval_successes=t.get("retrieval_successes"),
            latency_ms=t.get("latency_ms"),
            error_code=t.get("error_code"),
            fallback_status=t.get("fallback_status"),
        )
        for t in data["turn_results"]
    )
    return SanitizedRunResult(
        run_id=data["run_id"],
        manifest_digest=data["manifest_digest"],
        turn_results=turns,
        host_evidence=dict(data.get("host_evidence", {})),
        sanitization_status=data["sanitization_status"],
        result_digest=data["result_digest"],
    )


def _evaluation_to_dict(ev: EvaluationResult) -> dict[str, Any]:
    """Project an evaluation to a JSON-safe dict with its digest."""
    body = ev.matrix_binding()
    body["evaluation_digest"] = ev.evaluation_digest
    return body


def _evaluation_from_dict(data: dict[str, Any]) -> EvaluationResult:
    """Rebuild an evaluation from :func:`_evaluation_to_dict`, re-verifying its digest.

    Args:
        data: The serialized evaluation.

    Returns:
        The reconstructed, digest-verified :class:`EvaluationResult`.

    Raises:
        ContractViolation: ``REPLAY_DIGEST_MISMATCH`` when the stored digest does not match the content.
    """
    return EvaluationResult(
        run_id=data["run_id"],
        manifest_digest=data["manifest_digest"],
        arm=data["arm"],
        turns_attempted=int(data["turns_attempted"]),
        turns_completed=int(data["turns_completed"]),
        error_count=int(data["error_count"]),
        fallback_count=int(data["fallback_count"]),
        metric_validity=tuple(
            _metric_validity_from_dict(mv) for mv in data["metric_validity"]
        ),
        evaluation_digest=data["evaluation_digest"],
        correctness_numerator=data.get("correctness_numerator"),
        correctness_denominator=data.get("correctness_denominator"),
        model_input_tokens_total=data.get("model_input_tokens_total"),
        model_input_tokens_median=data.get("model_input_tokens_median"),
        retrieval_attempts=data.get("retrieval_attempts"),
        retrieval_successes=data.get("retrieval_successes"),
        retrieval_failures=data.get("retrieval_failures"),
        latency_p50_ms=data.get("latency_p50_ms"),
        latency_p95_ms=data.get("latency_p95_ms"),
    )


# --- File I/O helpers ------------------------------------------------------------------------------


def _read_json(path: str) -> Any:
    """Read and parse a JSON file.

    Args:
        path: The file path.

    Returns:
        The parsed JSON value.
    """
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str, data: Any) -> None:
    """Write ``data`` as pretty JSON, creating parent directories as needed.

    Args:
        path: The output path.
        data: The JSON-safe value to write.
    """
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_catalog(catalog: TerminalOutcomeCatalog, path: str) -> None:
    """Persist a terminal catalog as JSON.

    Args:
        catalog: The catalog.
        path: The output path.
    """
    _write_json(path, catalog_to_dict(catalog))


def load_catalog(path: str) -> TerminalOutcomeCatalog:
    """Load and digest-verify a terminal catalog from JSON.

    Args:
        path: The catalog file path.

    Returns:
        The reconstructed :class:`TerminalOutcomeCatalog`.
    """
    return catalog_from_dict(_read_json(path))


# --- File-level operations -------------------------------------------------------------------------


def sanitize_file(
    raw_path: str,
    manifest_path: str,
    output_path: str,
    *,
    catalog_path: str,
) -> SanitizedRunResult:
    """Sanitize a raw run file into a payload-free sanitized-result file.

    The raw file is untrusted; it may carry prompts, responses, tool payloads, credentials or logs.
    This function hands it to the pure :func:`~opencode_gate_a.sanitize.sanitize_run`, which drops every
    forbidden field, and writes only the resulting payload-free projection. The raw bytes are never
    copied to ``output_path``.

    Args:
        raw_path: Path to the untrusted raw run JSON.
        manifest_path: Path to the manifest the result binds to.
        output_path: Path to write the sanitized result JSON.
        catalog_path: Path to the terminal outcome catalog used to classify the turns.

    Returns:
        The sanitized run result.

    Raises:
        ContractViolation: Any failure surfaced by sanitisation or by the digest-verified reloads.
    """
    raw_run = _read_json(raw_path)
    manifest = manifest_from_dict(_read_json(manifest_path))
    catalog = load_catalog(catalog_path)
    result = sanitize_run(
        raw_run, manifest_digest=manifest.manifest_digest, catalog=catalog
    )
    _write_json(output_path, _result_to_dict(result))
    return result


def evaluate_file(
    sanitized_path: str,
    manifest_path: str,
    catalog_path: str,
    output_path: str,
) -> EvaluationResult:
    """Evaluate a sanitized-result file into a deterministic evaluation file.

    Args:
        sanitized_path: Path to the sanitized result JSON (written by :func:`sanitize_file`).
        manifest_path: Path to the manifest the result binds to.
        catalog_path: Path to the terminal outcome catalog.
        output_path: Path to write the evaluation JSON.

    Returns:
        The evaluation result.

    Raises:
        ContractViolation: Any precondition or digest failure surfaced by the evaluator or the reloads.
    """
    result = _result_from_dict(_read_json(sanitized_path))
    manifest = manifest_from_dict(_read_json(manifest_path))
    catalog = load_catalog(catalog_path)
    ev = evaluate(result, manifest=manifest, catalog=catalog)
    _write_json(output_path, _evaluation_to_dict(ev))
    return ev


def decide_gate_a_files(
    *,
    baseline_manifest_path: str,
    baseline_eval_path: str,
    treatment_manifest_path: str,
    treatment_eval_path: str,
    output_path: str | None = None,
) -> GateDecision:
    """Decide Gate A from a baseline and a treatment evaluation file pair.

    Args:
        baseline_manifest_path: Path to the baseline manifest.
        baseline_eval_path: Path to the baseline evaluation.
        treatment_manifest_path: Path to the OpenCode-A treatment manifest.
        treatment_eval_path: Path to the OpenCode-A treatment evaluation.
        output_path: Optional path to write the decision JSON.

    Returns:
        The :class:`GateDecision`, carrying stable reason codes and the two evaluation evidence digests.
    """
    baseline_manifest = manifest_from_dict(_read_json(baseline_manifest_path))
    treatment_manifest = manifest_from_dict(_read_json(treatment_manifest_path))
    baseline_eval = _evaluation_from_dict(_read_json(baseline_eval_path))
    treatment_eval = _evaluation_from_dict(_read_json(treatment_eval_path))
    decision = decide_gate_a(
        baseline_eval,
        treatment_eval,
        baseline_manifest=baseline_manifest,
        treatment_manifest=treatment_manifest,
    )
    if output_path is not None:
        _write_json(
            output_path,
            {
                "gate_name": decision.gate_name,
                "outcome": decision.outcome,
                "reasons": list(decision.reasons),
                "evidence_digests": list(decision.evidence_digests),
            },
        )
    return decision


# --- CLI -------------------------------------------------------------------------------------------


def _guard_live(args: argparse.Namespace) -> None:
    """Enforce the live-execution guard fail-closed.

    A ``--execute-live`` request must carry explicit confirmation and the admission and stat-plan paths,
    and even when fully specified this layer refuses to start a run because no host driver ships here.

    Args:
        args: The parsed CLI arguments.

    Raises:
        SystemExit: With a message when confirmation or the required paths are missing.
        LiveExecutionUnavailable: Always, when a fully specified live execution is requested.
    """
    if not getattr(args, "execute_live", False):
        return
    if not getattr(args, "i_understand_live_costs", False):
        raise SystemExit(
            "live execution requires explicit confirmation via --i-understand-live-costs"
        )
    if not getattr(args, "admission_path", None) or not getattr(args, "stat_plan_path", None):
        raise SystemExit(
            "live execution requires --admission-path and --stat-plan-path"
        )
    raise LiveExecutionUnavailable()


def _cmd_manifest(args: argparse.Namespace) -> int:
    """Handle the ``manifest`` subcommand: build and persist a manifest, offline by default."""
    _guard_live(args)

    turn_ids: list[str] | None = None
    if args.turn_ids_file:
        with open(args.turn_ids_file, encoding="utf-8") as handle:
            turn_ids = [line.strip() for line in handle if line.strip()]

    catalog = load_catalog(args.catalog)
    scope = "full" if args.full else "pilot"

    stat_plan = None
    if args.stat_plan_path:
        stat_plan = _stat_plan_from_dict(_read_json(args.stat_plan_path))

    manifest = build_manifest(
        run_id=args.run_id,
        arm=args.arm,
        host=args.host,
        host_version=args.host_version,
        model_id=args.model_id,
        capability_manifest_digest=args.capability_digest,
        fixture_catalog_digest=args.fixture_digest,
        terminal_catalog_digest=catalog.digest,
        order_seed=args.order_seed,
        execution_mode=args.execution_mode,
        backend_id=args.backend,
        scope=scope,
        turn_ids=turn_ids,
        budget_status=args.budget_status,
        sample_plan_status=args.sample_plan_status,
        execution_admission_id=args.execution_admission_id,
        admission_policy_digest=args.admission_policy_digest,
        budget_ceiling_units=args.budget_ceiling_units,
        budget_currency_or_unit=args.budget_unit,
        price_provenance=args.price_provenance,
        stat_plan=stat_plan,
    )
    _write_json(args.out, manifest_to_dict(manifest, scope=scope))
    print(f"wrote manifest {manifest.run_id} ({scope}) -> {args.out}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    """Handle the ``evaluate`` subcommand: sanitize a raw run then evaluate it, offline."""
    _guard_live(args)
    sanitize_file(
        args.raw,
        args.manifest,
        args.out_sanitized,
        catalog_path=args.catalog,
    )
    ev = evaluate_file(
        args.out_sanitized,
        args.manifest,
        args.catalog,
        args.out_evaluation,
    )
    print(
        f"evaluated {ev.run_id}: completed={ev.turns_completed}/{ev.turns_attempted} "
        f"-> {args.out_evaluation}"
    )
    return 0


def _cmd_gate_a(args: argparse.Namespace) -> int:
    """Handle the ``gate-a`` subcommand: decide Gate A from a baseline/treatment evaluation pair."""
    decision = decide_gate_a_files(
        baseline_manifest_path=args.baseline_manifest,
        baseline_eval_path=args.baseline_eval,
        treatment_manifest_path=args.treatment_manifest,
        treatment_eval_path=args.treatment_eval,
        output_path=args.out,
    )
    print(f"Gate A: {decision.outcome} reasons={list(decision.reasons)}")
    return 0


def _add_live_flags(parser: argparse.ArgumentParser) -> None:
    """Attach the shared, fail-closed live-execution flags to ``parser``."""
    parser.add_argument(
        "--execute-live",
        action="store_true",
        help="Request a live run. Fails closed: no host driver ships in this layer.",
    )
    parser.add_argument(
        "--i-understand-live-costs",
        action="store_true",
        help="Explicit confirmation required alongside --execute-live.",
    )
    parser.add_argument("--admission-path", default=None, help="Live execution admission file path.")
    parser.add_argument("--stat-plan-path", default=None, help="Approved stat plan file path.")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the three safe subcommands.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="opencode_gate_a.runner",
        description="Offline CE-60 Gate A runner: manifest / evaluate / gate-a (no live run).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # manifest
    m = sub.add_parser("manifest", help="Build and persist a run manifest (offline/replay/live-data).")
    m.add_argument("--run-id", required=True)
    m.add_argument("--arm", required=True, choices=sorted(ARMS))
    m.add_argument("--host", required=True)
    m.add_argument("--host-version", required=True)
    m.add_argument("--model-id", required=True)
    m.add_argument("--backend", default="local-fake", choices=sorted(BACKEND_IDS))
    m.add_argument("--execution-mode", default="offline", choices=sorted(EXECUTION_MODES))
    m.add_argument("--capability-digest", required=True)
    m.add_argument("--fixture-digest", required=True)
    m.add_argument("--catalog", required=True, help="Terminal catalog JSON; its digest is bound.")
    m.add_argument("--order-seed", required=True)
    m.add_argument("--turn-ids-file", default=None, help="Newline-separated 60 turn ids.")
    m.add_argument("--budget-status", default="absent")
    m.add_argument("--sample-plan-status", default="absent")
    m.add_argument("--execution-admission-id", default=None)
    m.add_argument("--admission-policy-digest", default=None)
    m.add_argument("--budget-ceiling-units", type=float, default=None)
    m.add_argument("--budget-unit", default=None)
    m.add_argument("--price-provenance", default=None)
    scope_group = m.add_mutually_exclusive_group()
    scope_group.add_argument("--pilot", action="store_true", help="Pilot plan (default).")
    scope_group.add_argument("--full", action="store_true", help="Full plan.")
    m.add_argument("--out", required=True)
    _add_live_flags(m)
    m.set_defaults(func=_cmd_manifest)

    # evaluate
    e = sub.add_parser("evaluate", help="Sanitize a raw run then evaluate it, offline.")
    e.add_argument("--raw", required=True, help="Untrusted raw run JSON.")
    e.add_argument("--manifest", required=True)
    e.add_argument("--catalog", required=True)
    e.add_argument("--out-sanitized", required=True)
    e.add_argument("--out-evaluation", required=True)
    scope_group_e = e.add_mutually_exclusive_group()
    scope_group_e.add_argument("--pilot", action="store_true")
    scope_group_e.add_argument("--full", action="store_true")
    _add_live_flags(e)
    e.set_defaults(func=_cmd_evaluate)

    # gate-a
    g = sub.add_parser("gate-a", help="Decide Gate A from a baseline/treatment evaluation pair.")
    g.add_argument("--baseline-manifest", required=True)
    g.add_argument("--baseline-eval", required=True)
    g.add_argument("--treatment-manifest", required=True)
    g.add_argument("--treatment-eval", required=True)
    g.add_argument("--out", default=None)
    g.set_defaults(func=_cmd_gate_a)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI.

    The CLI performs no model or host execution. A ``--execute-live`` request fails closed with the
    stable code :data:`LIVE_EXECUTION_REQUIRES_HOST_DRIVER` printed to stderr.

    Args:
        argv: The argument vector, or ``None`` to read ``sys.argv``.

    Returns:
        A process exit code: ``0`` on success, ``2`` for a usage/validation error, ``3`` for the
        fail-closed live guard, ``4`` for a contract violation.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except LiveExecutionUnavailable as exc:
        print(exc.code, file=sys.stderr)
        return 3
    except ContractViolation as exc:
        print(str(exc), file=sys.stderr)
        return 4
    except SystemExit as exc:
        if isinstance(exc.code, int):
            return exc.code
        print(str(exc.code), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
