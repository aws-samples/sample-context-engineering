"""Frozen, canonicalisable contracts for the offline Gate A layer, with fail-closed validation.

This module models U2's provenance and evaluation entities as immutable (``frozen=True``) dataclasses
whose ``__post_init__`` rejects any inconsistent value at construction time — nothing is silently
coerced, defaulted or zeroed (BR6, BR7, BR8, BR10; NFR1-NFR9).

The entities are:

* :class:`TurnAssignment` — the deterministic ``(turn_id, seed, fault_profile_id, fault_profile_digest)``
  binding for one of the 60 turns.
* :class:`StatPlan` — the closed statistical precondition for ``live`` and for any ``go`` carrying a
  statistical claim (NFR8.3).
* :class:`TerminalTurnDefinition` / :class:`TerminalOutcomeCatalog` — the canonical, immutable terminal
  classification used by :mod:`sanitize` and :mod:`evaluator` (BR6.3-BR6.5).
* :class:`RunManifest` — the immutable identity, admission and provenance of a run; it fixes exactly 60
  assignments and binds every digest into ``manifest_digest`` (BR7.1, NFR4.2-NFR4.3).
* :class:`MetricValidity` — the explicit per-metric, per-scope validity record (NFR1.3).
* :class:`SanitizedTurnResult` / :class:`SanitizedRunResult` — the sanitized, payload-free results that
  are the only input the evaluator accepts (BR7.2, NFR4.1).
* :class:`EvaluationResult` — the deterministic per-run metric matrix and its digest (BR7.5).
* :class:`GateDecision` — the isolated same-host Gate A go/no-go/not-evaluable decision (BR8).

Everything here is stdlib-only: no network, no AWS, no agent framework, no provider. Digests use
:mod:`canonical`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

from .canonical import CANONICAL_SPEC_VERSION, digest

__all__ = [
    "ACCEPTED_STATES",
    "ARMS",
    "BACKEND_IDS",
    "BUDGET_STATUSES",
    "COMPLETION_STATUSES",
    "COMPLETION_TERMINALS",
    "EXECUTION_MODES",
    "FALLBACK_POLICIES",
    "GATE_OUTCOMES",
    "LOCAL_BACKENDS",
    "MATCHERS",
    "METRIC_NAMES",
    "METRIC_SCOPES",
    "METRIC_STATUSES",
    "REASON_CODES",
    "TERMINAL_STATES",
    "TURN_COUNT",
    "ContractViolation",
    "EvaluationResult",
    "GateDecision",
    "MetricValidity",
    "RunManifest",
    "SanitizedRunResult",
    "SanitizedTurnResult",
    "StatPlan",
    "TerminalOutcomeCatalog",
    "TerminalTurnDefinition",
    "TurnAssignment",
]

TURN_COUNT: Final[int] = 60
"""The CE-60 benchmark has exactly 60 turns; nothing with a different count is admitted."""

CORRECTNESS_ELIGIBLE_COUNT: Final[int] = 30
"""Exactly 30 of the 60 turns are correctness-eligible in the approved catalog (BR6.5)."""

ExecutionMode = Literal["offline", "live", "replay"]
BackendId = Literal["local-fake", "local-scorer", "remote-explicit"]
Arm = Literal["baseline", "A", "A_plus_D", "native_B"]
BudgetStatus = Literal["absent", "configured", "not_applicable"]
SamplePlanStatus = Literal["approved", "absent", "not_applicable"]
TerminalState = Literal[
    "completed_expected",
    "completed_with_fallback",
    "refused_expected",
    "invalid_terminal",
    "unrecovered_error",
    "timed_out",
]
CompletionStatus = Literal["completed", "not_completed"]
AcceptedState = Literal["completed_expected", "completed_with_fallback", "refused_expected"]
TerminalMatcher = Literal[
    "canonical_content_equality",
    "canonical_schema_validation",
    "deterministic_predicate",
]
FallbackPolicy = Literal["not_allowed", "declared_fallback_only"]
MetricScope = Literal["turn", "run"]
MetricStatus = Literal[
    "valid",
    "host_not_exposed",
    "not_applicable",
    "collection_failed",
    "redacted_by_policy",
]
MetricName = Literal["completion", "correctness", "model_input", "retrieval", "latency"]
GateOutcome = Literal["go", "no_go", "not_evaluable"]

EXECUTION_MODES: Final[frozenset[str]] = frozenset({"offline", "live", "replay"})
BACKEND_IDS: Final[frozenset[str]] = frozenset({"local-fake", "local-scorer", "remote-explicit"})
LOCAL_BACKENDS: Final[frozenset[str]] = frozenset({"local-fake", "local-scorer"})
ARMS: Final[frozenset[str]] = frozenset({"baseline", "A", "A_plus_D", "native_B"})
BUDGET_STATUSES: Final[frozenset[str]] = frozenset({"absent", "configured", "not_applicable"})
SAMPLE_PLAN_STATUSES: Final[frozenset[str]] = frozenset({"approved", "absent", "not_applicable"})
TERMINAL_STATES: Final[frozenset[str]] = frozenset(
    {
        "completed_expected",
        "completed_with_fallback",
        "refused_expected",
        "invalid_terminal",
        "unrecovered_error",
        "timed_out",
    }
)
COMPLETION_STATUSES: Final[frozenset[str]] = frozenset({"completed", "not_completed"})
# The terminal states that count as a completed task for the completion_60 metric (BR7.3).
COMPLETION_TERMINALS: Final[frozenset[str]] = frozenset(
    {"completed_expected", "completed_with_fallback"}
)
ACCEPTED_STATES: Final[frozenset[str]] = frozenset(
    {"completed_expected", "completed_with_fallback", "refused_expected"}
)
MATCHERS: Final[frozenset[str]] = frozenset(
    {"canonical_content_equality", "canonical_schema_validation", "deterministic_predicate"}
)
FALLBACK_POLICIES: Final[frozenset[str]] = frozenset({"not_allowed", "declared_fallback_only"})
METRIC_SCOPES: Final[frozenset[str]] = frozenset({"turn", "run"})
METRIC_STATUSES: Final[frozenset[str]] = frozenset(
    {"valid", "host_not_exposed", "not_applicable", "collection_failed", "redacted_by_policy"}
)
METRIC_NAMES: Final[frozenset[str]] = frozenset(
    {"completion", "correctness", "model_input", "retrieval", "latency"}
)
GATE_OUTCOMES: Final[frozenset[str]] = frozenset({"go", "no_go", "not_evaluable"})

# The closed set of stable reason/violation codes this layer emits. A refusal or decision never carries a
# free-text explanation; it carries one or more of these codes (NFR9.2).
REASON_CODES: Final[frozenset[str]] = frozenset(
    {
        # Contract / provenance violations (fail-closed construction and evaluator preconditions).
        "FIXTURE_DIGEST_MISMATCH",
        "TERMINAL_CATALOG_INVALID",
        "ORACLE_MISMATCH",
        "RUN_MANIFEST_INVALID",
        "RESULT_NOT_SANITIZED",
        "METRIC_INVALID",
        "REPLAY_DIGEST_MISMATCH",
        "COMPARISON_INVALID",
        "ADMISSION_POLICY_DENIED",
        "SAMPLE_PLAN_INVALID",
        "STAT_PLAN_INVALID",
        # Gate A decision reasons.
        "GATE_INELIGIBLE_ARM",
        "GATE_HOST_MISMATCH",
        "GATE_MODEL_MISMATCH",
        "GATE_METRIC_NOT_EVALUABLE",
        "GATE_COMPLETION_REGRESSION",
        "GATE_CORRECTNESS_REGRESSION",
        "GATE_MODEL_INPUT_INSUFFICIENT",
        "GATE_RETRIEVAL_INVALID",
        "GATE_ISOLATION_INVALID",
        "GATE_PASS",
    }
)

# Gate A thresholds, taken verbatim from BR8.2-BR8.4. These are spec constants, not configuration.
CORRECTNESS_MAX_DROP_PP: Final[float] = 3.0
"""Correctness may drop at most 3 percentage points (BR8.3)."""

MODEL_INPUT_MIN_REDUCTION: Final[float] = 0.20
"""Median model-input tokens must drop by at least 20% (BR8.4)."""


class ContractViolation(ValueError):
    """A fail-closed contract violation carrying a stable reason code.

    Raised by every ``__post_init__`` and by the evaluator preconditions. The ``code`` is always one of
    :data:`REASON_CODES`, so a caller can branch on the code without parsing a message.

    Attributes:
        code: The stable reason code.
    """

    def __init__(self, code: str, message: str) -> None:
        """Wrap a stable ``code`` and a human-readable ``message``.

        Args:
            code: A stable reason code; must be in :data:`REASON_CODES`.
            message: A short description for logs and test assertions (never leaked to a payload).

        Raises:
            ValueError: If ``code`` is not a known reason code — an unknown code would itself be an
                invalid refusal, so this is a programming error.
        """
        if code not in REASON_CODES:
            raise ValueError(f"unknown reason code: {code!r}")
        super().__init__(f"{code}: {message}")
        self.code = code


def _require(condition: bool, code: str, message: str) -> None:
    """Raise :class:`ContractViolation` with ``code`` when ``condition`` is false.

    Args:
        condition: The invariant that must hold.
        code: The stable reason code to raise on failure.
        message: A short description of the failure.

    Raises:
        ContractViolation: If ``condition`` is false.
    """
    if not condition:
        raise ContractViolation(code, message)


def _is_nonempty_str(value: object) -> bool:
    """Return whether ``value`` is a non-empty, non-whitespace string.

    Args:
        value: The value to check.

    Returns:
        ``True`` if ``value`` is a ``str`` with at least one non-whitespace character.
    """
    return isinstance(value, str) and value.strip() != ""


@dataclass(frozen=True)
class TurnAssignment:
    """The deterministic execution assignment for one of the 60 turns (BR6.6).

    Attributes:
        turn_id: The canonical turn id.
        seed: The deterministic seed used for this turn's fixture and fault.
        fault_profile_id: The fault profile applied to this turn.
        fault_profile_digest: The digest pinning the exact fault profile bytes.
    """

    turn_id: str
    seed: str
    fault_profile_id: str
    fault_profile_digest: str

    def __post_init__(self) -> None:
        """Reject an assignment missing any required field (BR6.6).

        Raises:
            ContractViolation: ``RUN_MANIFEST_INVALID`` if any field is empty.
        """
        for name in ("turn_id", "seed", "fault_profile_id", "fault_profile_digest"):
            _require(
                _is_nonempty_str(getattr(self, name)),
                "RUN_MANIFEST_INVALID",
                f"turn assignment field {name!r} must be a non-empty string",
            )

    def seed_pair(self) -> list[str]:
        """Return the canonical ``[turn_id, seed]`` pair for the seed-assignment digest.

        Returns:
            The two-element list used to build ``seed_assignment_digest``.
        """
        return [self.turn_id, self.seed]

    def fault_triplet(self) -> list[str]:
        """Return the canonical ``[turn_id, fault_profile_id, fault_profile_digest]`` triplet.

        Returns:
            The three-element list used to build ``fault_assignment_digest``.
        """
        return [self.turn_id, self.fault_profile_id, self.fault_profile_digest]


def _seed_assignment_digest(assignments: Sequence[TurnAssignment]) -> str:
    """Compute the seed-assignment digest over assignments sorted by ``turn_id``.

    Args:
        assignments: The turn assignments.

    Returns:
        The canonical digest of the sorted ``[turn_id, seed]`` list.
    """
    pairs = [a.seed_pair() for a in sorted(assignments, key=lambda a: a.turn_id)]
    return digest(pairs)


def _fault_assignment_digest(assignments: Sequence[TurnAssignment]) -> str:
    """Compute the fault-assignment digest over assignments sorted by ``turn_id``.

    Args:
        assignments: The turn assignments.

    Returns:
        The canonical digest of the sorted ``[turn_id, fault_profile_id, fault_profile_digest]`` list.
    """
    triplets = [a.fault_triplet() for a in sorted(assignments, key=lambda a: a.turn_id)]
    return digest(triplets)


@dataclass(frozen=True)
class StatPlan:
    """The closed statistical precondition for ``live`` and for any ``go`` carrying a statistical claim.

    All content fields are required and non-empty, and ``digest`` must equal the recomputed digest of the
    content so the plan cannot be altered after approval (NFR8.3, BR10.3).

    Attributes:
        confidence_level: The approved confidence level in the open interval (0, 1).
        paired_unit: The paired comparison unit (for example ``"per_turn"``).
        method: The versioned statistical method identifier.
        decision_rule: The versioned decision rule identifier.
        minimum_paired_repetitions: The approved minimum number of paired repetitions (positive).
        approval_id: The approval identifier.
        approval_actor: The actor who approved the plan.
        digest: The digest binding the content fields above.
    """

    confidence_level: float
    paired_unit: str
    method: str
    decision_rule: str
    minimum_paired_repetitions: int
    approval_id: str
    approval_actor: str
    digest: str

    def __post_init__(self) -> None:
        """Reject an incomplete, non-finite or digest-mismatched plan fail-closed.

        Raises:
            ContractViolation: ``STAT_PLAN_INVALID`` if any field is missing, out of range or the digest
                does not match the recomputed content digest.
        """
        _require(
            isinstance(self.confidence_level, (int, float))
            and 0.0 < float(self.confidence_level) < 1.0,
            "STAT_PLAN_INVALID",
            "confidence_level must be a float in the open interval (0, 1)",
        )
        for name in ("paired_unit", "method", "decision_rule", "approval_id", "approval_actor"):
            _require(
                _is_nonempty_str(getattr(self, name)),
                "STAT_PLAN_INVALID",
                f"stat plan field {name!r} must be a non-empty string",
            )
        _require(
            isinstance(self.minimum_paired_repetitions, int)
            and self.minimum_paired_repetitions > 0,
            "STAT_PLAN_INVALID",
            "minimum_paired_repetitions must be a positive integer",
        )
        _require(
            self.digest == self.content_digest(),
            "STAT_PLAN_INVALID",
            "stat plan digest does not match its content",
        )

    def content_binding(self) -> dict[str, object]:
        """Return the canonical content binding the digest covers.

        Returns:
            A dict of the content fields (everything except ``digest``).
        """
        return {
            "confidence_level": float(self.confidence_level),
            "paired_unit": self.paired_unit,
            "method": self.method,
            "decision_rule": self.decision_rule,
            "minimum_paired_repetitions": self.minimum_paired_repetitions,
            "approval_id": self.approval_id,
            "approval_actor": self.approval_actor,
        }

    def content_digest(self) -> str:
        """Return the digest of :meth:`content_binding`.

        Returns:
            The canonical digest of the content fields.
        """
        return digest(self.content_binding())

    @classmethod
    def build(
        cls,
        *,
        confidence_level: float,
        paired_unit: str,
        method: str,
        decision_rule: str,
        minimum_paired_repetitions: int,
        approval_id: str,
        approval_actor: str,
    ) -> StatPlan:
        """Build a plan, computing the content digest so the result is valid by construction.

        Args:
            confidence_level: Confidence level in (0, 1).
            paired_unit: The paired comparison unit.
            method: The versioned method identifier.
            decision_rule: The versioned decision rule identifier.
            minimum_paired_repetitions: The approved minimum repetitions.
            approval_id: The approval identifier.
            approval_actor: The approving actor.

        Returns:
            A valid :class:`StatPlan`.
        """
        content = {
            "confidence_level": float(confidence_level),
            "paired_unit": paired_unit,
            "method": method,
            "decision_rule": decision_rule,
            "minimum_paired_repetitions": minimum_paired_repetitions,
            "approval_id": approval_id,
            "approval_actor": approval_actor,
        }
        return cls(
            confidence_level=float(confidence_level),
            paired_unit=paired_unit,
            method=method,
            decision_rule=decision_rule,
            minimum_paired_repetitions=minimum_paired_repetitions,
            approval_id=approval_id,
            approval_actor=approval_actor,
            digest=digest(content),
        )


@dataclass(frozen=True)
class TerminalTurnDefinition:
    """The canonical terminal matcher and accepted states for one turn (BR6.4-BR6.5).

    Attributes:
        turn_id: The canonical turn id.
        terminal_matcher: The deterministic matcher mechanism.
        accepted_states: The non-empty subset of accepted terminal states.
        fallback_policy: Whether a declared fallback is permitted.
        correctness_eligible: Whether this turn contributes to the correctness denominator.
    """

    turn_id: str
    terminal_matcher: TerminalMatcher
    accepted_states: tuple[AcceptedState, ...]
    fallback_policy: FallbackPolicy
    correctness_eligible: bool

    def __post_init__(self) -> None:
        """Reject an invalid matcher, empty/unknown accepted states or unknown fallback policy.

        Raises:
            ContractViolation: ``TERMINAL_CATALOG_INVALID`` on any violation.
        """
        _require(
            _is_nonempty_str(self.turn_id),
            "TERMINAL_CATALOG_INVALID",
            "turn_id must be a non-empty string",
        )
        _require(
            self.terminal_matcher in MATCHERS,
            "TERMINAL_CATALOG_INVALID",
            f"unknown terminal matcher: {self.terminal_matcher!r}",
        )
        _require(
            len(self.accepted_states) > 0,
            "TERMINAL_CATALOG_INVALID",
            "accepted_states must not be empty",
        )
        for state in self.accepted_states:
            _require(
                state in ACCEPTED_STATES,
                "TERMINAL_CATALOG_INVALID",
                f"unknown accepted state: {state!r}",
            )
        _require(
            self.fallback_policy in FALLBACK_POLICIES,
            "TERMINAL_CATALOG_INVALID",
            f"unknown fallback policy: {self.fallback_policy!r}",
        )
        if "completed_with_fallback" in self.accepted_states:
            _require(
                self.fallback_policy == "declared_fallback_only",
                "TERMINAL_CATALOG_INVALID",
                "completed_with_fallback requires declared_fallback_only policy",
            )


@dataclass(frozen=True)
class TerminalOutcomeCatalog:
    """A versioned, immutable catalog of exactly 60 turn definitions, 30 correctness-eligible (BR6.3).

    Attributes:
        terminal_catalog_id: The opaque catalog id.
        version: The catalog version string.
        turn_definitions: Exactly 60 unique :class:`TerminalTurnDefinition`.
        digest: The digest pinning the catalog bytes.
    """

    terminal_catalog_id: str
    version: str
    turn_definitions: tuple[TerminalTurnDefinition, ...]
    digest: str

    def __post_init__(self) -> None:
        """Reject a catalog that is not exactly 60 unique turns with exactly 30 eligible, or mis-digested.

        Raises:
            ContractViolation: ``TERMINAL_CATALOG_INVALID`` for count/uniqueness/digest failures;
                ``ORACLE_MISMATCH`` when the correctness-eligible count is not 30.
        """
        _require(
            _is_nonempty_str(self.terminal_catalog_id) and _is_nonempty_str(self.version),
            "TERMINAL_CATALOG_INVALID",
            "catalog id and version must be non-empty strings",
        )
        _require(
            len(self.turn_definitions) == TURN_COUNT,
            "TERMINAL_CATALOG_INVALID",
            f"catalog must contain exactly {TURN_COUNT} turn definitions",
        )
        ids = [d.turn_id for d in self.turn_definitions]
        _require(
            len(set(ids)) == TURN_COUNT,
            "TERMINAL_CATALOG_INVALID",
            "catalog turn_ids must be unique",
        )
        eligible = sum(1 for d in self.turn_definitions if d.correctness_eligible)
        _require(
            eligible == CORRECTNESS_ELIGIBLE_COUNT,
            "ORACLE_MISMATCH",
            f"catalog must have exactly {CORRECTNESS_ELIGIBLE_COUNT} correctness-eligible turns",
        )
        _require(
            self.digest == self.content_digest(),
            "TERMINAL_CATALOG_INVALID",
            "terminal catalog digest does not match its content",
        )

    def content_binding(self) -> dict[str, object]:
        """Return the canonical binding the catalog digest covers.

        Returns:
            A dict of catalog id, version and the ordered turn definitions.
        """
        return {
            "terminal_catalog_id": self.terminal_catalog_id,
            "version": self.version,
            "turn_definitions": [
                {
                    "turn_id": d.turn_id,
                    "terminal_matcher": d.terminal_matcher,
                    "accepted_states": list(d.accepted_states),
                    "fallback_policy": d.fallback_policy,
                    "correctness_eligible": d.correctness_eligible,
                }
                for d in sorted(self.turn_definitions, key=lambda d: d.turn_id)
            ],
        }

    def content_digest(self) -> str:
        """Return the digest of :meth:`content_binding`.

        Returns:
            The canonical digest of the catalog content.
        """
        return digest(self.content_binding())

    @classmethod
    def build(
        cls,
        *,
        terminal_catalog_id: str,
        version: str,
        turn_definitions: Sequence[TerminalTurnDefinition],
    ) -> TerminalOutcomeCatalog:
        """Build a catalog, computing its content digest so the result is valid by construction.

        Args:
            terminal_catalog_id: The opaque catalog id.
            version: The catalog version string.
            turn_definitions: The 60 turn definitions.

        Returns:
            A valid :class:`TerminalOutcomeCatalog`.
        """
        definitions = tuple(turn_definitions)
        partial = cls.__new__(cls)
        object.__setattr__(partial, "terminal_catalog_id", terminal_catalog_id)
        object.__setattr__(partial, "version", version)
        object.__setattr__(partial, "turn_definitions", definitions)
        object.__setattr__(partial, "digest", "")
        content_digest = partial.content_digest()
        object.__setattr__(partial, "digest", content_digest)
        partial.__post_init__()
        return partial

    def definition(self, turn_id: str) -> TerminalTurnDefinition:
        """Return the definition for ``turn_id``.

        Args:
            turn_id: The canonical turn id.

        Returns:
            The matching :class:`TerminalTurnDefinition`.

        Raises:
            ContractViolation: ``TERMINAL_CATALOG_INVALID`` if the turn is not in the catalog.
        """
        for d in self.turn_definitions:
            if d.turn_id == turn_id:
                return d
        raise ContractViolation("TERMINAL_CATALOG_INVALID", f"turn {turn_id!r} not in catalog")


@dataclass(frozen=True)
class RunManifest:
    """The immutable identity, admission and provenance of a run (BR7.1, BR7.7, NFR4.2-NFR4.3).

    A manifest fixes exactly 60 turn assignments, the order seed, the two assignment digests, host /
    version / model / backend / arm, the capability / fixture / terminal catalog digests, and the
    execution-mode-specific admission, budget, price and sample-plan fields. ``manifest_digest`` is the
    digest of :meth:`binding`, so any post-hoc change produces a different digest and the replay rejects
    it. Build one with :meth:`build` so the digests are correct by construction.

    Attributes:
        run_id: The opaque run id.
        host: The host under test.
        host_version: The host version string.
        model_id: The model identity.
        backend_id: The backend; local for offline/replay, ``remote-explicit`` for live.
        arm: The experiment arm.
        capability_manifest_digest: Digest of the capability manifest.
        fixture_catalog_digest: Digest of the fixture catalog.
        terminal_catalog_digest: Digest of the terminal outcome catalog.
        execution_mode: ``offline``, ``live`` or ``replay``.
        order_seed: The counterbalanced execution order seed.
        turn_assignments: Exactly 60 :class:`TurnAssignment`, one per unique turn id.
        seed_assignment_digest: Digest of the canonical ``[turn_id, seed]`` list.
        fault_assignment_digest: Digest of the canonical fault triplet list.
        execution_admission_id: The admission id for live; ``None`` for offline/replay.
        admission_policy_digest: The admission policy digest for live; ``None`` otherwise.
        budget_status: The budget status.
        budget_ceiling_units: The budget ceiling for live; ``None`` otherwise.
        budget_currency_or_unit: The budget unit for live; ``None`` otherwise.
        price_provenance: The price provenance for live; ``None`` otherwise.
        sample_plan_status: The sample-plan status.
        stat_plan: The approved :class:`StatPlan` for live; ``None`` otherwise.
        manifest_digest: The digest of :meth:`binding`.
    """

    run_id: str
    host: str
    host_version: str
    model_id: str
    backend_id: BackendId
    arm: Arm
    capability_manifest_digest: str
    fixture_catalog_digest: str
    terminal_catalog_digest: str
    execution_mode: ExecutionMode
    order_seed: str
    turn_assignments: tuple[TurnAssignment, ...]
    seed_assignment_digest: str
    fault_assignment_digest: str
    budget_status: BudgetStatus
    sample_plan_status: SamplePlanStatus
    manifest_digest: str
    execution_admission_id: str | None = None
    admission_policy_digest: str | None = None
    budget_ceiling_units: float | None = None
    budget_currency_or_unit: str | None = None
    price_provenance: str | None = None
    stat_plan: StatPlan | None = None

    def __post_init__(self) -> None:
        """Enforce every manifest invariant fail-closed.

        Raises:
            ContractViolation: ``RUN_MANIFEST_INVALID`` for identity/assignment/digest failures;
                ``ADMISSION_POLICY_DENIED`` for offline/replay carrying live-only fields;
                ``SAMPLE_PLAN_INVALID`` for a live manifest missing an approved plan.
        """
        self._validate_identity()
        self._validate_assignments()
        self._validate_mode()
        _require(
            self.manifest_digest == self.compute_digest(),
            "RUN_MANIFEST_INVALID",
            "manifest_digest does not match the manifest binding",
        )

    def _validate_identity(self) -> None:
        """Validate the identity, enum and digest-shaped fields.

        Raises:
            ContractViolation: ``RUN_MANIFEST_INVALID`` on any violation.
        """
        for name in (
            "run_id",
            "host",
            "host_version",
            "model_id",
            "order_seed",
            "capability_manifest_digest",
            "fixture_catalog_digest",
            "terminal_catalog_digest",
            "seed_assignment_digest",
            "fault_assignment_digest",
        ):
            _require(
                _is_nonempty_str(getattr(self, name)),
                "RUN_MANIFEST_INVALID",
                f"manifest field {name!r} must be a non-empty string",
            )
        _require(
            self.backend_id in BACKEND_IDS,
            "RUN_MANIFEST_INVALID",
            f"unknown backend_id: {self.backend_id!r}",
        )
        _require(self.arm in ARMS, "RUN_MANIFEST_INVALID", f"unknown arm: {self.arm!r}")
        _require(
            self.execution_mode in EXECUTION_MODES,
            "RUN_MANIFEST_INVALID",
            f"unknown execution_mode: {self.execution_mode!r}",
        )
        _require(
            self.budget_status in BUDGET_STATUSES,
            "RUN_MANIFEST_INVALID",
            f"unknown budget_status: {self.budget_status!r}",
        )
        _require(
            self.sample_plan_status in SAMPLE_PLAN_STATUSES,
            "RUN_MANIFEST_INVALID",
            f"unknown sample_plan_status: {self.sample_plan_status!r}",
        )

    def _validate_assignments(self) -> None:
        """Validate the 60 assignments and the two assignment digests (BR6.6, BR7.1).

        Raises:
            ContractViolation: ``RUN_MANIFEST_INVALID`` on count, uniqueness or digest mismatch.
        """
        _require(
            len(self.turn_assignments) == TURN_COUNT,
            "RUN_MANIFEST_INVALID",
            f"manifest must contain exactly {TURN_COUNT} turn assignments",
        )
        ids = [a.turn_id for a in self.turn_assignments]
        _require(
            len(set(ids)) == TURN_COUNT,
            "RUN_MANIFEST_INVALID",
            "turn assignment turn_ids must be unique",
        )
        _require(
            self.seed_assignment_digest == _seed_assignment_digest(self.turn_assignments),
            "RUN_MANIFEST_INVALID",
            "seed_assignment_digest does not match the assignments",
        )
        _require(
            self.fault_assignment_digest == _fault_assignment_digest(self.turn_assignments),
            "RUN_MANIFEST_INVALID",
            "fault_assignment_digest does not match the assignments",
        )

    def _validate_mode(self) -> None:
        """Validate the execution-mode-specific admission, budget, price and sample-plan fields.

        Raises:
            ContractViolation: ``ADMISSION_POLICY_DENIED`` when offline/replay carry live-only fields;
                ``RUN_MANIFEST_INVALID``/``SAMPLE_PLAN_INVALID`` when live is incomplete.
        """
        if self.execution_mode in ("offline", "replay"):
            self._validate_non_live()
        else:
            self._validate_live()

    def _validate_non_live(self) -> None:
        """Validate the offline/replay invariants: local backend, no credentials, no live-only fields.

        Raises:
            ContractViolation: ``ADMISSION_POLICY_DENIED`` on any live-only field present.
        """
        _require(
            self.backend_id in LOCAL_BACKENDS,
            "ADMISSION_POLICY_DENIED",
            f"{self.execution_mode} must use a local backend, not {self.backend_id!r}",
        )
        _require(
            self.execution_admission_id is None and self.admission_policy_digest is None,
            "ADMISSION_POLICY_DENIED",
            f"{self.execution_mode} must not carry an execution admission",
        )
        _require(
            self.budget_status in ("absent", "not_applicable"),
            "ADMISSION_POLICY_DENIED",
            f"{self.execution_mode} budget_status must be absent or not_applicable",
        )
        _require(
            self.budget_ceiling_units is None
            and self.budget_currency_or_unit is None
            and self.price_provenance is None,
            "ADMISSION_POLICY_DENIED",
            f"{self.execution_mode} must not carry budget or price fields",
        )
        _require(
            self.sample_plan_status in ("absent", "not_applicable") and self.stat_plan is None,
            "ADMISSION_POLICY_DENIED",
            f"{self.execution_mode} must not carry an approved sample plan",
        )

    def _validate_live(self) -> None:
        """Validate the live invariants: admission, budget, price and an approved, matching plan.

        Raises:
            ContractViolation: ``ADMISSION_POLICY_DENIED`` / ``RUN_MANIFEST_INVALID`` /
                ``SAMPLE_PLAN_INVALID`` when any live precondition is missing or incoherent.
        """
        _require(
            self.backend_id == "remote-explicit",
            "ADMISSION_POLICY_DENIED",
            "live requires backend_id=remote-explicit",
        )
        _require(
            _is_nonempty_str(self.execution_admission_id)
            and _is_nonempty_str(self.admission_policy_digest),
            "ADMISSION_POLICY_DENIED",
            "live requires an execution admission id and policy digest",
        )
        _require(
            self.budget_status == "configured",
            "ADMISSION_POLICY_DENIED",
            "live requires budget_status=configured",
        )
        _require(
            isinstance(self.budget_ceiling_units, (int, float))
            and self.budget_ceiling_units is not None
            and self.budget_ceiling_units >= 0
            and _is_nonempty_str(self.budget_currency_or_unit)
            and _is_nonempty_str(self.price_provenance),
            "ADMISSION_POLICY_DENIED",
            "live requires a non-negative budget ceiling, a unit and price provenance",
        )
        _require(
            self.sample_plan_status == "approved" and self.stat_plan is not None,
            "SAMPLE_PLAN_INVALID",
            "live requires an approved stat plan",
        )

    def binding(self) -> dict[str, object]:
        """Return the canonical binding ``manifest_digest`` covers (NFR4.3).

        Returns:
            A dict of every immutable provenance field, including both assignment digests, the catalog
            digests, the order seed, the admission/price fields and the sample-plan digest.
        """
        return {
            "run_id": self.run_id,
            "host": self.host,
            "host_version": self.host_version,
            "model_id": self.model_id,
            "backend_id": self.backend_id,
            "arm": self.arm,
            "capability_manifest_digest": self.capability_manifest_digest,
            "fixture_catalog_digest": self.fixture_catalog_digest,
            "terminal_catalog_digest": self.terminal_catalog_digest,
            "execution_mode": self.execution_mode,
            "order_seed": self.order_seed,
            "seed_assignment_digest": self.seed_assignment_digest,
            "fault_assignment_digest": self.fault_assignment_digest,
            "execution_admission_id": self.execution_admission_id,
            "admission_policy_digest": self.admission_policy_digest,
            "budget_status": self.budget_status,
            "budget_ceiling_units": self.budget_ceiling_units,
            "budget_currency_or_unit": self.budget_currency_or_unit,
            "price_provenance": self.price_provenance,
            "sample_plan_status": self.sample_plan_status,
            "stat_plan_digest": None if self.stat_plan is None else self.stat_plan.digest,
            "canonical_spec_version": CANONICAL_SPEC_VERSION,
        }

    def compute_digest(self) -> str:
        """Return the digest of :meth:`binding`.

        Returns:
            The canonical digest of the manifest binding.
        """
        return digest(self.binding())

    @classmethod
    def build(
        cls,
        *,
        run_id: str,
        host: str,
        host_version: str,
        model_id: str,
        backend_id: BackendId,
        arm: Arm,
        capability_manifest_digest: str,
        fixture_catalog_digest: str,
        terminal_catalog_digest: str,
        execution_mode: ExecutionMode,
        order_seed: str,
        turn_assignments: Sequence[TurnAssignment],
        budget_status: BudgetStatus = "absent",
        sample_plan_status: SamplePlanStatus = "absent",
        execution_admission_id: str | None = None,
        admission_policy_digest: str | None = None,
        budget_ceiling_units: float | None = None,
        budget_currency_or_unit: str | None = None,
        price_provenance: str | None = None,
        stat_plan: StatPlan | None = None,
    ) -> RunManifest:
        """Build a manifest, computing the assignment digests and ``manifest_digest`` by construction.

        Args:
            run_id: The opaque run id.
            host: The host under test.
            host_version: The host version string.
            model_id: The model identity.
            backend_id: The backend id.
            arm: The experiment arm.
            capability_manifest_digest: Digest of the capability manifest.
            fixture_catalog_digest: Digest of the fixture catalog.
            terminal_catalog_digest: Digest of the terminal outcome catalog.
            execution_mode: The execution mode.
            order_seed: The order seed.
            turn_assignments: The 60 turn assignments.
            budget_status: The budget status. Defaults to ``"absent"``.
            sample_plan_status: The sample-plan status. Defaults to ``"absent"``.
            execution_admission_id: The admission id (live only).
            admission_policy_digest: The admission policy digest (live only).
            budget_ceiling_units: The budget ceiling (live only).
            budget_currency_or_unit: The budget unit (live only).
            price_provenance: The price provenance (live only).
            stat_plan: The approved stat plan (live only).

        Returns:
            A valid :class:`RunManifest`.
        """
        assignments = tuple(turn_assignments)
        seed_digest = _seed_assignment_digest(assignments)
        fault_digest = _fault_assignment_digest(assignments)
        partial = cls.__new__(cls)
        object.__setattr__(partial, "run_id", run_id)
        object.__setattr__(partial, "host", host)
        object.__setattr__(partial, "host_version", host_version)
        object.__setattr__(partial, "model_id", model_id)
        object.__setattr__(partial, "backend_id", backend_id)
        object.__setattr__(partial, "arm", arm)
        object.__setattr__(partial, "capability_manifest_digest", capability_manifest_digest)
        object.__setattr__(partial, "fixture_catalog_digest", fixture_catalog_digest)
        object.__setattr__(partial, "terminal_catalog_digest", terminal_catalog_digest)
        object.__setattr__(partial, "execution_mode", execution_mode)
        object.__setattr__(partial, "order_seed", order_seed)
        object.__setattr__(partial, "turn_assignments", assignments)
        object.__setattr__(partial, "seed_assignment_digest", seed_digest)
        object.__setattr__(partial, "fault_assignment_digest", fault_digest)
        object.__setattr__(partial, "budget_status", budget_status)
        object.__setattr__(partial, "sample_plan_status", sample_plan_status)
        object.__setattr__(partial, "execution_admission_id", execution_admission_id)
        object.__setattr__(partial, "admission_policy_digest", admission_policy_digest)
        object.__setattr__(partial, "budget_ceiling_units", budget_ceiling_units)
        object.__setattr__(partial, "budget_currency_or_unit", budget_currency_or_unit)
        object.__setattr__(partial, "price_provenance", price_provenance)
        object.__setattr__(partial, "stat_plan", stat_plan)
        manifest_digest = partial.compute_digest()
        object.__setattr__(partial, "manifest_digest", manifest_digest)
        partial.__post_init__()
        return partial


@dataclass(frozen=True)
class MetricValidity:
    """An explicit per-metric, per-scope validity record (NFR1.3, BR7.5).

    Attributes:
        scope: ``"turn"`` or ``"run"``.
        run_id: The run this validity belongs to.
        metric_name: The metric this record covers.
        status: The validity status.
        turn_id: The turn id for ``scope="turn"``; ``None`` for ``scope="run"``.
        reason_code: An optional stable reason code for a non-valid status.
    """

    scope: MetricScope
    run_id: str
    metric_name: MetricName
    status: MetricStatus
    turn_id: str | None = None
    reason_code: str | None = None

    def __post_init__(self) -> None:
        """Reject an unknown scope/metric/status or a scope/turn-id mismatch.

        Raises:
            ContractViolation: ``METRIC_INVALID`` on any violation.
        """
        _require(self.scope in METRIC_SCOPES, "METRIC_INVALID", f"unknown scope: {self.scope!r}")
        _require(
            _is_nonempty_str(self.run_id),
            "METRIC_INVALID",
            "metric validity run_id must be a non-empty string",
        )
        _require(
            self.metric_name in METRIC_NAMES,
            "METRIC_INVALID",
            f"unknown metric_name: {self.metric_name!r}",
        )
        _require(
            self.status in METRIC_STATUSES,
            "METRIC_INVALID",
            f"unknown metric status: {self.status!r}",
        )
        if self.scope == "turn":
            _require(
                _is_nonempty_str(self.turn_id),
                "METRIC_INVALID",
                "turn-scoped metric validity requires a turn_id",
            )
        else:
            _require(
                self.turn_id is None,
                "METRIC_INVALID",
                "run-scoped metric validity must not carry a turn_id",
            )
        if self.reason_code is not None:
            _require(
                self.reason_code in REASON_CODES,
                "METRIC_INVALID",
                f"unknown reason_code: {self.reason_code!r}",
            )

    def as_dict(self) -> dict[str, object]:
        """Return the canonical dict used in digest bindings.

        Returns:
            A dict of all fields.
        """
        return {
            "scope": self.scope,
            "run_id": self.run_id,
            "metric_name": self.metric_name,
            "status": self.status,
            "turn_id": self.turn_id,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True)
class SanitizedTurnResult:
    """A sanitized, payload-free per-turn result with explicit metric validity (BR7.2, NFR4.1).

    The field set is closed and carries no prompt, raw response, tool payload, credential or log — the
    sanitizer in :mod:`sanitize` only ever populates these fields. Any unavailable numeric metric is
    ``None`` and must be explained by a matching :class:`MetricValidity`, never silently zeroed.

    Attributes:
        turn_id: The canonical turn id.
        terminal_state: The catalog-classified terminal state.
        completion_status: Whether the turn completed, derived from the terminal state (BR7.3).
        correctness_eligible: Whether the turn is correctness-eligible.
        correctness_score: The correctness score for eligible turns, or ``None``.
        model_input_tokens: Model-input tokens, or ``None`` when unavailable.
        retrieval_attempts: Retrieval attempts, or ``None`` when unavailable.
        retrieval_successes: Retrieval successes, or ``None`` when unavailable.
        latency_ms: Latency in milliseconds, or ``None`` when unavailable.
        error_code: A stable error code, or ``None``.
        fallback_status: The fallback status, or ``None``.
        metric_validity: One :class:`MetricValidity` per observed metric at ``scope="turn"``.
    """

    turn_id: str
    terminal_state: TerminalState
    completion_status: CompletionStatus
    correctness_eligible: bool
    metric_validity: tuple[MetricValidity, ...]
    correctness_score: float | None = None
    model_input_tokens: int | None = None
    retrieval_attempts: int | None = None
    retrieval_successes: int | None = None
    latency_ms: float | None = None
    error_code: str | None = None
    fallback_status: str | None = None

    def __post_init__(self) -> None:
        """Reject unknown enums, scope/turn mismatches and negative counts fail-closed.

        Raises:
            ContractViolation: ``RESULT_NOT_SANITIZED`` on any violation.
        """
        _require(
            _is_nonempty_str(self.turn_id),
            "RESULT_NOT_SANITIZED",
            "turn result turn_id must be a non-empty string",
        )
        _require(
            self.terminal_state in TERMINAL_STATES,
            "RESULT_NOT_SANITIZED",
            f"unknown terminal_state: {self.terminal_state!r}",
        )
        _require(
            self.completion_status in COMPLETION_STATUSES,
            "RESULT_NOT_SANITIZED",
            f"unknown completion_status: {self.completion_status!r}",
        )
        expected = (
            "completed"
            if self.terminal_state in COMPLETION_TERMINALS
            else "not_completed"
        )
        _require(
            self.completion_status == expected,
            "RESULT_NOT_SANITIZED",
            "completion_status must be derived from terminal_state",
        )
        if self.retrieval_attempts is not None:
            _require(
                self.retrieval_attempts >= 0,
                "RESULT_NOT_SANITIZED",
                "retrieval_attempts must be non-negative",
            )
        if self.retrieval_successes is not None:
            _require(
                self.retrieval_successes >= 0,
                "RESULT_NOT_SANITIZED",
                "retrieval_successes must be non-negative",
            )
        if self.retrieval_attempts is not None and self.retrieval_successes is not None:
            _require(
                self.retrieval_successes <= self.retrieval_attempts,
                "RESULT_NOT_SANITIZED",
                "retrieval_successes cannot exceed retrieval_attempts",
            )
        if self.model_input_tokens is not None:
            _require(
                self.model_input_tokens >= 0,
                "RESULT_NOT_SANITIZED",
                "model_input_tokens must be non-negative",
            )
        if self.latency_ms is not None:
            _require(
                self.latency_ms >= 0,
                "RESULT_NOT_SANITIZED",
                "latency_ms must be non-negative",
            )
        if not self.correctness_eligible:
            _require(
                self.correctness_score is None,
                "RESULT_NOT_SANITIZED",
                "non-eligible turns must not carry a correctness_score",
            )
        for mv in self.metric_validity:
            _require(
                mv.scope == "turn" and mv.turn_id == self.turn_id,
                "RESULT_NOT_SANITIZED",
                "turn metric validity must be turn-scoped for this turn",
            )

    def validity_for(self, metric_name: str) -> MetricValidity | None:
        """Return the validity record for ``metric_name``, if present.

        Args:
            metric_name: The metric to look up.

        Returns:
            The matching :class:`MetricValidity`, or ``None`` when absent.
        """
        for mv in self.metric_validity:
            if mv.metric_name == metric_name:
                return mv
        return None

    def as_dict(self) -> dict[str, object]:
        """Return the canonical dict used in the result digest binding.

        Returns:
            A dict of every sanitized field, with metric validity expanded.
        """
        return {
            "turn_id": self.turn_id,
            "terminal_state": self.terminal_state,
            "completion_status": self.completion_status,
            "correctness_eligible": self.correctness_eligible,
            "correctness_score": self.correctness_score,
            "model_input_tokens": self.model_input_tokens,
            "retrieval_attempts": self.retrieval_attempts,
            "retrieval_successes": self.retrieval_successes,
            "latency_ms": self.latency_ms,
            "error_code": self.error_code,
            "fallback_status": self.fallback_status,
            "metric_validity": [
                mv.as_dict()
                for mv in sorted(self.metric_validity, key=lambda m: m.metric_name)
            ],
        }


@dataclass(frozen=True)
class SanitizedRunResult:
    """A sanitized, immutable set of exactly 60 turn results — the only input the evaluator accepts.

    Attributes:
        run_id: The run id, matching the manifest.
        manifest_digest: The manifest digest this result is bound to.
        turn_results: Exactly 60 :class:`SanitizedTurnResult`.
        host_evidence: Sanitized, payload-free host evidence (digests, versions, flags only).
        sanitization_status: ``"passed"`` or ``"failed"``.
        result_digest: The digest pinning the sanitized result bytes.
    """

    run_id: str
    manifest_digest: str
    turn_results: tuple[SanitizedTurnResult, ...]
    host_evidence: Mapping[str, object]
    sanitization_status: Literal["passed", "failed"]
    result_digest: str

    def __post_init__(self) -> None:
        """Reject a result that is not exactly 60 unique turns or whose digest does not match.

        Raises:
            ContractViolation: ``RESULT_NOT_SANITIZED`` on any violation.
        """
        _require(
            _is_nonempty_str(self.run_id) and _is_nonempty_str(self.manifest_digest),
            "RESULT_NOT_SANITIZED",
            "run result run_id and manifest_digest must be non-empty strings",
        )
        _require(
            self.sanitization_status in ("passed", "failed"),
            "RESULT_NOT_SANITIZED",
            f"unknown sanitization_status: {self.sanitization_status!r}",
        )
        _require(
            len(self.turn_results) == TURN_COUNT,
            "RESULT_NOT_SANITIZED",
            f"run result must contain exactly {TURN_COUNT} turn results",
        )
        ids = [t.turn_id for t in self.turn_results]
        _require(
            len(set(ids)) == TURN_COUNT,
            "RESULT_NOT_SANITIZED",
            "run result turn_ids must be unique",
        )
        _require(
            self.result_digest == self.compute_digest(),
            "RESULT_NOT_SANITIZED",
            "result_digest does not match the sanitized content",
        )

    def binding(self) -> dict[str, object]:
        """Return the canonical binding ``result_digest`` covers.

        Returns:
            A dict of run id, manifest digest, sanitization status, host evidence and the ordered turns.
        """
        return {
            "run_id": self.run_id,
            "manifest_digest": self.manifest_digest,
            "sanitization_status": self.sanitization_status,
            "host_evidence": dict(self.host_evidence),
            "turn_results": [
                t.as_dict() for t in sorted(self.turn_results, key=lambda t: t.turn_id)
            ],
        }

    def compute_digest(self) -> str:
        """Return the digest of :meth:`binding`.

        Returns:
            The canonical digest of the sanitized result content.
        """
        return digest(self.binding())


@dataclass(frozen=True)
class EvaluationResult:
    """The deterministic per-run metric matrix, validity and digest (BR7.5).

    Unavailable aggregates are ``None`` (never silently zero) and are explained by run-scope
    :class:`MetricValidity` records. ``evaluation_digest`` is a pure function of the computed matrix, so an
    identical replay yields an identical digest (NFR1.1).

    Attributes:
        run_id: The run id.
        manifest_digest: The manifest digest the evaluation is bound to.
        arm: The experiment arm, copied from the manifest for Gate A eligibility.
        turns_attempted: Always 60.
        turns_completed: Count of turns whose completion_status is ``completed``.
        correctness_numerator: Count of correct eligible-and-valid turns, or ``None``.
        correctness_denominator: Count of valid eligible turns, or ``None``.
        model_input_tokens_total: Sum of valid per-turn model-input tokens, or ``None``.
        model_input_tokens_median: Median of valid per-turn model-input tokens, or ``None``.
        retrieval_attempts: Sum of valid retrieval attempts, or ``None``.
        retrieval_successes: Sum of valid retrieval successes, or ``None``.
        retrieval_failures: ``attempts - successes`` over valid turns, or ``None``.
        latency_p50_ms: 50th-percentile latency over valid turns, or ``None``.
        latency_p95_ms: 95th-percentile latency over valid turns, or ``None``.
        error_count: Count of turns carrying an error code.
        fallback_count: Count of turns whose fallback_status is a declared fallback.
        metric_validity: One run-scope :class:`MetricValidity` per metric.
        evaluation_digest: The digest of the evaluation binding.
    """

    run_id: str
    manifest_digest: str
    arm: Arm
    turns_attempted: int
    turns_completed: int
    error_count: int
    fallback_count: int
    metric_validity: tuple[MetricValidity, ...]
    evaluation_digest: str
    correctness_numerator: int | None = None
    correctness_denominator: int | None = None
    model_input_tokens_total: int | None = None
    model_input_tokens_median: float | None = None
    retrieval_attempts: int | None = None
    retrieval_successes: int | None = None
    retrieval_failures: int | None = None
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None

    def __post_init__(self) -> None:
        """Reject structural inconsistencies and a mismatched digest fail-closed.

        Raises:
            ContractViolation: ``RUN_MANIFEST_INVALID`` for identity failures; ``METRIC_INVALID`` for a
                run-scope validity carrying a turn id; ``REPLAY_DIGEST_MISMATCH`` for a digest mismatch.
        """
        _require(
            _is_nonempty_str(self.run_id) and _is_nonempty_str(self.manifest_digest),
            "RUN_MANIFEST_INVALID",
            "evaluation run_id and manifest_digest must be non-empty strings",
        )
        _require(self.arm in ARMS, "RUN_MANIFEST_INVALID", f"unknown arm: {self.arm!r}")
        _require(
            self.turns_attempted == TURN_COUNT,
            "RUN_MANIFEST_INVALID",
            f"turns_attempted must be exactly {TURN_COUNT}",
        )
        _require(
            0 <= self.turns_completed <= TURN_COUNT,
            "RUN_MANIFEST_INVALID",
            "turns_completed must be within [0, 60]",
        )
        for mv in self.metric_validity:
            _require(
                mv.scope == "run" and mv.turn_id is None,
                "METRIC_INVALID",
                "evaluation metric validity must be run-scoped",
            )
        _require(
            self.evaluation_digest == self.compute_digest(),
            "REPLAY_DIGEST_MISMATCH",
            "evaluation_digest does not match the computed matrix",
        )

    def run_validity_for(self, metric_name: str) -> MetricValidity | None:
        """Return the run-scope validity for ``metric_name``, if present.

        Args:
            metric_name: The metric to look up.

        Returns:
            The matching :class:`MetricValidity`, or ``None``.
        """
        for mv in self.metric_validity:
            if mv.metric_name == metric_name:
                return mv
        return None

    def matrix_binding(self) -> dict[str, object]:
        """Return the canonical matrix the digest covers (excludes the digest itself).

        Returns:
            A dict of every computed aggregate and the ordered run-scope validity records.
        """
        return {
            "run_id": self.run_id,
            "manifest_digest": self.manifest_digest,
            "arm": self.arm,
            "turns_attempted": self.turns_attempted,
            "turns_completed": self.turns_completed,
            "correctness_numerator": self.correctness_numerator,
            "correctness_denominator": self.correctness_denominator,
            "model_input_tokens_total": self.model_input_tokens_total,
            "model_input_tokens_median": self.model_input_tokens_median,
            "retrieval_attempts": self.retrieval_attempts,
            "retrieval_successes": self.retrieval_successes,
            "retrieval_failures": self.retrieval_failures,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "error_count": self.error_count,
            "fallback_count": self.fallback_count,
            "metric_validity": [
                mv.as_dict()
                for mv in sorted(self.metric_validity, key=lambda m: m.metric_name)
            ],
        }

    def compute_digest(self) -> str:
        """Return the digest of :meth:`matrix_binding`.

        Returns:
            The canonical digest of the evaluation matrix.
        """
        return digest(self.matrix_binding())


@dataclass(frozen=True)
class GateDecision:
    """The isolated same-host Gate A decision for the OpenCode A arm (BR8).

    Attributes:
        gate_name: Always ``"Gate_A"``.
        outcome: ``"go"``, ``"no_go"`` or ``"not_evaluable"``.
        reasons: One or more stable reason codes justifying the outcome.
        evidence_digests: The evaluation digests the decision rests on.
    """

    outcome: GateOutcome
    reasons: tuple[str, ...]
    evidence_digests: tuple[str, ...]
    gate_name: Literal["Gate_A"] = "Gate_A"

    def __post_init__(self) -> None:
        """Reject an unknown outcome, empty reasons or unknown reason codes.

        Raises:
            ContractViolation: ``METRIC_INVALID`` on an unknown outcome or reason; a ``go`` must carry
                exactly the ``GATE_PASS`` reason.
        """
        _require(
            self.outcome in GATE_OUTCOMES,
            "METRIC_INVALID",
            f"unknown gate outcome: {self.outcome!r}",
        )
        _require(len(self.reasons) > 0, "METRIC_INVALID", "gate decision must carry a reason")
        for reason in self.reasons:
            _require(
                reason in REASON_CODES,
                "METRIC_INVALID",
                f"unknown gate reason: {reason!r}",
            )
        if self.outcome == "go":
            _require(
                self.reasons == ("GATE_PASS",),
                "METRIC_INVALID",
                "a go decision must carry exactly the GATE_PASS reason",
            )
        else:
            _require(
                "GATE_PASS" not in self.reasons,
                "METRIC_INVALID",
                "a non-go decision must not carry GATE_PASS",
            )
