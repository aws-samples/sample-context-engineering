"""Deterministic test factories for the offline Gate A layer.

These builders construct a valid 60-turn CE-60 catalog, a matching ``RunManifest`` and a raw run mapping
whose numbers the tests can shape per metric. Everything here is in-memory and offline: no network, no
provider, no AWS. The helpers never create a fake provider — they only build input *data*.
"""

from __future__ import annotations

from typing import Any

from opencode_gate_a.contracts import (
    CORRECTNESS_ELIGIBLE_COUNT,
    TURN_COUNT,
    RunManifest,
    StatPlan,
    TerminalOutcomeCatalog,
    TerminalTurnDefinition,
    TurnAssignment,
)

TURN_IDS = tuple(f"t{index:02d}" for index in range(TURN_COUNT))
ELIGIBLE_IDS = frozenset(TURN_IDS[:CORRECTNESS_ELIGIBLE_COUNT])


def build_catalog() -> TerminalOutcomeCatalog:
    """Build a valid 60-turn catalog with exactly 30 correctness-eligible turns.

    Returns:
        A valid :class:`TerminalOutcomeCatalog`.
    """
    definitions = [
        TerminalTurnDefinition(
            turn_id=turn_id,
            terminal_matcher="deterministic_predicate",
            accepted_states=("completed_expected",),
            fallback_policy="not_allowed",
            correctness_eligible=turn_id in ELIGIBLE_IDS,
        )
        for turn_id in TURN_IDS
    ]
    return TerminalOutcomeCatalog.build(
        terminal_catalog_id="cat-ce60",
        version="1.0",
        turn_definitions=definitions,
    )


def build_assignments() -> tuple[TurnAssignment, ...]:
    """Build the 60 deterministic turn assignments.

    Returns:
        A tuple of 60 :class:`TurnAssignment`.
    """
    return tuple(
        TurnAssignment(
            turn_id=turn_id,
            seed=f"seed-{turn_id}",
            fault_profile_id="fault-none",
            fault_profile_digest="sha256:" + ("0" * 64),
        )
        for turn_id in TURN_IDS
    )


def build_manifest(
    *,
    run_id: str = "run-baseline",
    arm: str = "baseline",
    host: str = "opencode",
    host_version: str = "1.4.0",
    model_id: str = "model-x",
    execution_mode: str = "offline",
    backend_id: str = "local-fake",
    catalog: TerminalOutcomeCatalog | None = None,
) -> RunManifest:
    """Build a valid offline/replay manifest bound to ``catalog``.

    Args:
        run_id: The run id.
        arm: The experiment arm.
        host: The host under test.
        host_version: The host version.
        model_id: The model identity.
        execution_mode: ``offline`` or ``replay``.
        backend_id: A local backend id.
        catalog: The catalog whose digest to bind; a fresh one is built when omitted.

    Returns:
        A valid :class:`RunManifest`.
    """
    cat = catalog if catalog is not None else build_catalog()
    return RunManifest.build(
        run_id=run_id,
        host=host,
        host_version=host_version,
        model_id=model_id,
        backend_id=backend_id,  # type: ignore[arg-type]
        arm=arm,  # type: ignore[arg-type]
        capability_manifest_digest="sha256:" + ("1" * 64),
        fixture_catalog_digest="sha256:" + ("2" * 64),
        terminal_catalog_digest=cat.digest,
        execution_mode=execution_mode,  # type: ignore[arg-type]
        order_seed="order-1",
        turn_assignments=build_assignments(),
        budget_status="not_applicable",
        sample_plan_status="not_applicable",
    )


def build_stat_plan() -> StatPlan:
    """Build a valid approved stat plan for live-manifest tests.

    Returns:
        A valid :class:`StatPlan`.
    """
    return StatPlan.build(
        confidence_level=0.95,
        paired_unit="per_turn",
        method="paired-bootstrap-v1",
        decision_rule="noninferiority-v1",
        minimum_paired_repetitions=5,
        approval_id="appr-1",
        approval_actor="reviewer-1",
    )


def raw_turn(
    turn_id: str,
    *,
    host_signal: str = "success",
    matcher_passed: bool = True,
    model_input: int | None = 100,
    correctness: float | None = 1.0,
    retrieval: tuple[int, int] | None = (1, 1),
    latency: float | None = 10.0,
    model_input_status: str | None = None,
    retrieval_status: str | None = None,
    include_forbidden: bool = False,
) -> dict[str, Any]:
    """Build one raw, untrusted turn mapping with shapeable metrics.

    Args:
        turn_id: The turn id.
        host_signal: The observed host signal.
        matcher_passed: Whether the catalog matcher accepted the content.
        model_input: Model-input token value, or ``None`` to omit.
        correctness: Correctness value (eligible turns only), or ``None`` to omit.
        retrieval: ``(attempts, successes)`` or ``None`` to omit.
        latency: Latency value, or ``None`` to omit.
        model_input_status: An explicit model_input validity status, if any.
        retrieval_status: An explicit retrieval validity status, if any.
        include_forbidden: When true, inject forbidden payload keys to exercise sanitisation.

    Returns:
        A raw turn mapping suitable for :func:`opencode_gate_a.sanitize.sanitize_turn`.
    """
    metrics: dict[str, Any] = {}
    if model_input is not None or model_input_status is not None:
        entry: dict[str, Any] = {}
        if model_input is not None:
            entry["value"] = model_input
        if model_input_status is not None:
            entry["status"] = model_input_status
        metrics["model_input"] = entry
    if correctness is not None:
        metrics["correctness"] = {"value": correctness}
    if retrieval is not None or retrieval_status is not None:
        rentry: dict[str, Any] = {}
        if retrieval is not None:
            rentry["attempts"] = retrieval[0]
            rentry["successes"] = retrieval[1]
        if retrieval_status is not None:
            rentry["status"] = retrieval_status
        metrics["retrieval"] = rentry
    if latency is not None:
        metrics["latency"] = {"value": latency}

    turn: dict[str, Any] = {
        "turn_id": turn_id,
        "host_signal": host_signal,
        "matcher_passed": matcher_passed,
        "metrics": metrics,
    }
    if include_forbidden:
        turn["prompt"] = "SECRET-PROMPT-CANARY"
        turn["response"] = "SECRET-RESPONSE-CANARY"
        turn["tool_payload"] = {"body": "SECRET-PAYLOAD-CANARY"}
        turn["credentials"] = "AKIA-SECRET-CANARY"
        turn["logs"] = ["SECRET-LOG-CANARY"]
    return turn


def raw_run(
    *,
    run_id: str = "run-baseline",
    model_input: int = 100,
    correctness: float = 1.0,
    retrieval: tuple[int, int] = (1, 1),
    include_forbidden: bool = False,
) -> dict[str, Any]:
    """Build a raw run mapping of exactly 60 uniform turns.

    Args:
        run_id: The run id.
        model_input: Model-input tokens for every turn.
        correctness: Correctness score for every eligible turn.
        retrieval: ``(attempts, successes)`` for every turn.
        include_forbidden: When true, inject forbidden keys into every turn and the host evidence.

    Returns:
        A raw run mapping suitable for :func:`opencode_gate_a.sanitize.sanitize_run`.
    """
    turns = [
        raw_turn(
            turn_id,
            model_input=model_input,
            correctness=correctness if turn_id in ELIGIBLE_IDS else None,
            retrieval=retrieval,
            include_forbidden=include_forbidden,
        )
        for turn_id in TURN_IDS
    ]
    host_evidence: dict[str, Any] = {"host": "opencode", "host_version": "1.4.0"}
    if include_forbidden:
        host_evidence["credentials"] = "AKIA-SECRET-CANARY"
        host_evidence["logs"] = "SECRET-LOG-CANARY"
    return {"run_id": run_id, "turns": turns, "host_evidence": host_evidence}
