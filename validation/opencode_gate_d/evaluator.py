"""The Gate D decision: pure checks over request evidence, with the precondition made explicit.

The functional design says practice D stays ``unsupported`` until Gate A is ``go`` and then needs its own gate. Gate A was
decided on OpenCode and is ``no_go``. The owner asked for D to be built and measured anyway and waived the precondition
(FR5.1 / BR5.1). The evaluator therefore takes the precondition as an input and records how it was met:

* ``gate_a_go``        -- Gate A is ``go``; a passing Gate D can make the capability ``supported``.
* ``waived_by_owner``  -- Gate A is not ``go`` and an explicit waiver is on record; a passing Gate D makes the capability
                          ``experimental`` and no better, and says so in the decision.
* ``unmet``            -- neither: the outcome is ``unsupported`` and nothing else is evaluated.

No Gate D result reclassifies Gate A; the decision says ``gate_a_reclassified: false`` to make that visible.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .contracts import (
    REQUIRED_SCENARIOS,
    GateDDecision,
    GateDRequestEvidence,
    all_scenarios,
    streams,
)

__all__ = ["REQUIRED_WAIVER_KEYS", "evaluate"]

REQUIRED_WAIVER_KEYS = ("by", "requirement", "reason")
"""What a waiver must state. A waiver without a named author, requirement and reason is not a waiver."""


def _waiver_is_explicit(waiver: Mapping[str, str] | None) -> bool:
    return bool(waiver) and all(isinstance(waiver.get(key), str) and waiver[key].strip() for key in REQUIRED_WAIVER_KEYS)  # type: ignore[union-attr]


def evaluate(
    requests: Sequence[GateDRequestEvidence],
    *,
    gate_a_outcome: str,
    waiver: Mapping[str, str] | None = None,
    scenario_failures: Sequence[str] = (),
) -> GateDDecision:
    """Decide Gate D from request evidence.

    Args:
        requests: One record per model request, in the order they happened within each stream.
        gate_a_outcome: The Gate A outcome on record (``go``, ``no_go``, ``not_evaluable``, ``pending``).
        waiver: The owner's explicit waiver of the Gate A precondition, or ``None``.
        scenario_failures: Failures a scenario reported by itself, such as a Card left without a source after a compaction.
            Each one is a ``no_go`` reason.

    Returns:
        The decision, with every check and its reasons.
    """
    if gate_a_outcome == "go":
        precondition = "gate_a_go"
    elif _waiver_is_explicit(waiver):
        precondition = "waived_by_owner"
    else:
        precondition = "unmet"

    covered = all_scenarios(requests)
    if precondition == "unmet":
        return GateDDecision(
            outcome="unsupported",
            capability_state="unsupported",
            precondition="unmet",
            reasons=("CAPABILITY_UNSUPPORTED",),
            checks={"precondition": False},
            requests_evaluated=0,
            scenarios_covered=covered,
            gate_a_outcome=gate_a_outcome,
        )

    reasons: list[str] = []
    checks: dict[str, bool] = {"precondition": True}

    def check(name: str, passed: bool, reason: str) -> None:
        checks[name] = passed
        if not passed:
            reasons.append(reason)

    check("evidence_present", bool(requests), "EVIDENCE_MISSING")

    defects = [(request.request_id, problem) for request in requests for problem in request.defects()]
    check("records_well_formed", not defects, "EVIDENCE_INCOMPLETE")

    check(
        "request_only_transcript",
        all(request.transcripts_equal() and request.canonical_equality for request in requests),
        "TRANSCRIPT_DURABLY_CHANGED",
    )

    missing = [name for name in REQUIRED_SCENARIOS if name not in covered]
    check("scenarios_covered", not missing, "SCENARIO_MISSING:" + ",".join(missing) if missing else "")

    check(
        "cards_have_sources",
        all(request.card_ids_and_sources for request in requests if "pre_card" not in request.scenarios),
        "CARD_SOURCE_MISSING",
    )

    # FR5.3: an id keeps its sources for as long as it exists (within one epoch: a compaction is a defined transition), and no
    # id is shared between sessions.
    unstable = False
    collision = False
    owner: dict[str, str] = {}
    for (stream_id, _epoch), records in streams(requests).items():
        seen: dict[str, tuple[str, ...]] = {}
        for record in records:
            for card_id, sources in record.card_ids_and_sources:
                if card_id in seen and seen[card_id] != sources:
                    unstable = True
                seen[card_id] = sources
                if owner.setdefault(card_id, stream_id) != stream_id:
                    collision = True
    check("card_ids_stable", not unstable, "CARD_ID_UNSTABLE")
    check("card_ids_not_shared_between_sessions", not collision, "CARD_ID_COLLISION")

    full_pass = [request for request in requests if "full_pass" in request.scenarios]
    check("full_pass_present", bool(full_pass), "FULL_PASS_MISSING")
    check("full_pass_equals_baseline", all(request.full_pass_matches_baseline() for request in full_pass), "FULL_PASS_DIVERGES")

    check("no_extra_durable_effect", all(request.absence_is_clean() for request in requests), "DURABLE_EFFECT_DETECTED")
    check("projection_engaged", any(request.projected for request in requests), "NEVER_ENGAGED")

    check("scenarios_reported_no_failure", not scenario_failures, "SCENARIO_FAILED:" + ",".join(scenario_failures))

    reasons = [reason for reason in reasons if reason]
    passed = not reasons
    if not passed:
        outcome, state = "no_go", "unsupported"
    elif precondition == "gate_a_go":
        outcome, state = "go", "supported"
    else:
        # The contract holds, but the precondition was waived rather than met: experimental, never supported.
        outcome, state = "go", "experimental"
    return GateDDecision(
        outcome=outcome,  # type: ignore[arg-type]
        capability_state=state,  # type: ignore[arg-type]
        precondition=precondition,  # type: ignore[arg-type]
        reasons=tuple(reasons),
        checks=checks,
        requests_evaluated=len(requests),
        scenarios_covered=covered,
        gate_a_outcome=gate_a_outcome,
        waiver=dict(waiver or {}) if precondition == "waived_by_owner" else {},
    )
