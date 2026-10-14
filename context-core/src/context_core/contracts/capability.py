"""Capability publication and decision snapshots — ``capability_state`` kept apart from ``decision_lifecycle``.

Two state spaces live here and never collapse into one (BR2.4):

- ``capability_state`` (``full``/``native``/``experimental``/``partial``/``unsupported``) describes the
  *published* capability on a :class:`CapabilityDeclaration`.
- ``decision_lifecycle`` (``issued``/``valid``/``invalidated``/``superseded``/``retired``) describes the
  lifecycle of an immutable :class:`CapabilityDecision` snapshot.

A decision binds a closed normative tuple (host, host_version, practice, implementation_kind, coverage
id+digest, evidence digest, manifest id+revision+digest, contract identity). Revalidation compares that
tuple against the live manifest: all equal transitions ``issued``/``valid`` to ``valid``; any divergence
transitions to ``invalidated`` and the request is refused with ``CONTRACT_MISMATCH`` (BR1.4, BR2.2).

No framework import, no I/O. The one clock dependency (``issued_at``) is injected.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from .canonical import digest as _digest
from .errors import RefusalError
from .identity import ContractIdentity

__all__ = [
    "CAPABILITY_STATES",
    "DECISION_LIFECYCLES",
    "IMPLEMENTATION_KINDS",
    "PRACTICES",
    "CapabilityDecision",
    "CapabilityDeclaration",
    "CapabilityManifest",
    "CoverageSet",
    "EvidenceDigest",
    "ensure_usable",
    "issue_decision",
    "revalidate",
]

Practice = Literal["A", "B", "D"]
CapabilityState = Literal["full", "native", "experimental", "partial", "unsupported"]
ImplementationKind = Literal["adapter", "host"]
DecisionLifecycle = Literal["issued", "valid", "invalidated", "superseded", "retired"]

PRACTICES: frozenset[str] = frozenset({"A", "B", "D"})
CAPABILITY_STATES: frozenset[str] = frozenset({"full", "native", "experimental", "partial", "unsupported"})
IMPLEMENTATION_KINDS: frozenset[str] = frozenset({"adapter", "host"})
DECISION_LIFECYCLES: frozenset[str] = frozenset({"issued", "valid", "invalidated", "superseded", "retired"})

# States for which a non-empty coverage set is mandatory (entities.md CoverageSet constraint).
_COVERAGE_REQUIRED_STATES: frozenset[str] = frozenset({"full", "experimental", "partial"})


@dataclass(frozen=True)
class CoverageSet:
    """A closed set of tool classes, operations and lifecycle paths a capability covers.

    Attributes:
        coverage_set_id: Opaque identity of the set.
        tool_classes: Tool classes covered.
        lifecycle_paths: Lifecycle paths covered.
        operation_set: Operations covered.
        exclusions: Items explicitly excluded; an excluded item is refused, never approximated.
    """

    coverage_set_id: str
    tool_classes: tuple[str, ...] = ()
    lifecycle_paths: tuple[str, ...] = ()
    operation_set: tuple[str, ...] = ()
    exclusions: tuple[str, ...] = ()

    @property
    def coverage_set_digest(self) -> str:
        """SHA-256 over the canonical content of the set (id excluded)."""
        return _digest(
            {
                "tool_classes": sorted(self.tool_classes),
                "lifecycle_paths": sorted(self.lifecycle_paths),
                "operation_set": sorted(self.operation_set),
                "exclusions": sorted(self.exclusions),
            }
        )

    def covers(self, operation: str) -> bool:
        """Report whether ``operation`` is inside the set and not excluded.

        Args:
            operation: The operation to check.

        Returns:
            ``True`` when the operation is in ``operation_set`` and not in ``exclusions``.
        """
        return operation in self.operation_set and operation not in self.exclusions


@dataclass(frozen=True)
class EvidenceDigest:
    """Digest of the artifacts, tests and compatibility matrix backing a capability.

    Attributes:
        evidence_digest_id: Opaque identity of the evidence.
        artifact_digests: SHA-256 digests of the backing artifacts.
        compatibility_matrix_version: Version of the compatibility matrix.
    """

    evidence_digest_id: str
    artifact_digests: tuple[str, ...] = ()
    compatibility_matrix_version: str = "0.0"

    @property
    def evidence_digest(self) -> str:
        """SHA-256 over the canonical evidence content (id excluded)."""
        return _digest(
            {
                "artifact_digests": sorted(self.artifact_digests),
                "compatibility_matrix_version": self.compatibility_matrix_version,
            }
        )


@dataclass(frozen=True)
class CapabilityDeclaration:
    """A published capability, separate from the lifecycle of any decision snapshot.

    Attributes:
        declaration_id: Opaque identity of the declaration.
        practice: ``"A"``, ``"B"`` or ``"D"``.
        capability_state: One of :data:`CAPABILITY_STATES`.
        implementation_kind: ``"adapter"`` or ``"host"``.
        coverage: The coverage set the declaration binds.
        evidence: The evidence backing the declaration.
    """

    declaration_id: str
    practice: Practice
    capability_state: CapabilityState
    implementation_kind: ImplementationKind
    coverage: CoverageSet
    evidence: EvidenceDigest

    def __post_init__(self) -> None:
        """Reject an unknown enum value or an empty coverage set where one is required.

        Raises:
            ValueError: If ``practice``, ``capability_state`` or ``implementation_kind`` is outside its
                closed set, or ``capability_state`` requires coverage and the set is empty.
        """
        if self.practice not in PRACTICES:
            raise ValueError(f"unknown practice: {self.practice!r}")
        if self.capability_state not in CAPABILITY_STATES:
            raise ValueError(f"unknown capability_state: {self.capability_state!r}")
        if self.implementation_kind not in IMPLEMENTATION_KINDS:
            raise ValueError(f"unknown implementation_kind: {self.implementation_kind!r}")
        if self.capability_state in _COVERAGE_REQUIRED_STATES and not self.coverage.operation_set:
            raise ValueError(f"capability_state {self.capability_state!r} requires a non-empty coverage set")


@dataclass(frozen=True)
class CapabilityManifest:
    """A versioned manifest of declarations for one host/version.

    Attributes:
        manifest_id: Opaque identity of the manifest.
        host: Host identifier.
        host_version: Host version string.
        contract_identity: The contract identity the manifest is published under.
        revision: Positive integer; increments on every change.
        declarations: The published declarations.
    """

    manifest_id: str
    host: str
    host_version: str
    contract_identity: ContractIdentity
    revision: int
    declarations: tuple[CapabilityDeclaration, ...] = ()

    def __post_init__(self) -> None:
        """Reject a non-positive revision or a duplicated declaration coordinate.

        Raises:
            ValueError: If ``revision < 1`` or two declarations share the same
                practice/implementation_kind/coverage coordinate.
        """
        if self.revision < 1:
            raise ValueError("manifest revision must be >= 1")
        seen: set[tuple[str, str, str]] = set()
        for decl in self.declarations:
            coordinate = (decl.practice, decl.implementation_kind, decl.coverage.coverage_set_id)
            if coordinate in seen:
                raise ValueError(f"duplicate declaration coordinate: {coordinate!r}")
            seen.add(coordinate)

    @property
    def manifest_digest(self) -> str:
        """SHA-256 over the manifest's canonical content (id excluded, revision included)."""
        return _digest(
            {
                "host": self.host,
                "host_version": self.host_version,
                "contract_version": self.contract_identity.contract_version,
                "contract_digest": self.contract_identity.contract_digest,
                "revision": self.revision,
                "declarations": sorted(
                    (
                        {
                            "practice": decl.practice,
                            "capability_state": decl.capability_state,
                            "implementation_kind": decl.implementation_kind,
                            "coverage_set_id": decl.coverage.coverage_set_id,
                            "coverage_set_digest": decl.coverage.coverage_set_digest,
                            "evidence_digest": decl.evidence.evidence_digest,
                        }
                        for decl in self.declarations
                    ),
                    key=lambda row: (row["practice"], row["implementation_kind"], row["coverage_set_id"]),
                ),
            }
        )

    def declaration_for(self, practice: str) -> CapabilityDeclaration | None:
        """Return the declaration for ``practice``, or ``None`` when none is published.

        Args:
            practice: The practice to look up.

        Returns:
            The matching declaration, or ``None``.
        """
        for decl in self.declarations:
            if decl.practice == practice:
                return decl
        return None


@dataclass(frozen=True)
class CapabilityDecision:
    """An immutable snapshot of a capability decision used by a request.

    The normative tuple is every field below except ``decision_lifecycle``, ``issued_at`` and
    ``decision_revision``. ``capability_state`` is an attribute of the snapshot, independent of
    ``decision_lifecycle``.

    Attributes:
        decision_id: Opaque identity of the snapshot.
        decision_lifecycle: One of :data:`DECISION_LIFECYCLES`.
        host: Host identifier.
        host_version: Host version string.
        practice: ``"A"``, ``"B"`` or ``"D"``.
        implementation_kind: ``"adapter"`` or ``"host"``.
        capability_state: The published state at snapshot time.
        coverage_set_id: Coverage set identity.
        coverage_set_digest: Coverage set digest.
        manifest_id: Manifest identity.
        manifest_revision: Manifest revision.
        manifest_digest: Manifest digest.
        evidence_digest_id: Evidence identity.
        evidence_digest: Evidence digest.
        contract_identity: The bound contract identity.
        issued_at: Snapshot issue time (injected clock value).
        decision_revision: Positive integer snapshot revision.
        coverage_operations: Immutable snapshot of the operations the bound coverage set covers, used
            to decide whether a given operation may transform. Content changes are already reflected in
            ``coverage_set_digest`` (part of the normative tuple), so these are carried for the coverage
            check, not re-added to the tuple.
        coverage_exclusions: Immutable snapshot of operations explicitly excluded from coverage; an
            excluded operation is refused, never approximated.
    """

    decision_id: str
    decision_lifecycle: DecisionLifecycle
    host: str
    host_version: str
    practice: Practice
    implementation_kind: ImplementationKind
    capability_state: CapabilityState
    coverage_set_id: str
    coverage_set_digest: str
    manifest_id: str
    manifest_revision: int
    manifest_digest: str
    evidence_digest_id: str
    evidence_digest: str
    contract_identity: ContractIdentity
    issued_at: float
    decision_revision: int = 1
    coverage_operations: tuple[str, ...] = ()
    coverage_exclusions: tuple[str, ...] = ()

    def normative_tuple(self) -> tuple[object, ...]:
        """Return the closed normative tuple that identifies this snapshot.

        The tuple binds both the evidence identity (``evidence_digest_id``) and the evidence content
        (``evidence_digest``), so re-issuing the same evidence content under a new id still diverges and
        invalidates the snapshot, matching the U1 normative tuple.

        Returns:
            The tuple of normative fields, in a fixed order, used for revalidation equality.
        """
        return (
            self.host,
            self.host_version,
            self.practice,
            self.implementation_kind,
            self.coverage_set_id,
            self.coverage_set_digest,
            self.evidence_digest_id,
            self.evidence_digest,
            self.manifest_id,
            self.manifest_revision,
            self.manifest_digest,
            self.contract_identity,
        )


def issue_decision(
    manifest: CapabilityManifest,
    practice: str,
    *,
    decision_id: str,
    issued_at: float,
) -> CapabilityDecision:
    """Issue an immutable decision snapshot for ``practice`` from ``manifest``.

    The snapshot starts in ``issued`` and must pass :func:`revalidate` before a transform may use it.

    Args:
        manifest: The live manifest.
        practice: The practice to snapshot; a declaration for it must exist.
        decision_id: Opaque identity for the new snapshot.
        issued_at: The issue time (injected clock value).

    Returns:
        A fresh ``issued`` decision binding the full normative tuple.

    Raises:
        ValueError: If no declaration for ``practice`` exists in the manifest.
    """
    decl = manifest.declaration_for(practice)
    if decl is None:
        raise ValueError(f"no declaration for practice {practice!r} in manifest {manifest.manifest_id!r}")

    return CapabilityDecision(
        decision_id=decision_id,
        decision_lifecycle="issued",
        host=manifest.host,
        host_version=manifest.host_version,
        practice=decl.practice,
        implementation_kind=decl.implementation_kind,
        capability_state=decl.capability_state,
        coverage_set_id=decl.coverage.coverage_set_id,
        coverage_set_digest=decl.coverage.coverage_set_digest,
        manifest_id=manifest.manifest_id,
        manifest_revision=manifest.revision,
        manifest_digest=manifest.manifest_digest,
        evidence_digest_id=decl.evidence.evidence_digest_id,
        evidence_digest=decl.evidence.evidence_digest,
        contract_identity=manifest.contract_identity,
        issued_at=issued_at,
        coverage_operations=tuple(decl.coverage.operation_set),
        coverage_exclusions=tuple(decl.coverage.exclusions),
    )


def revalidate(decision: CapabilityDecision, manifest: CapabilityManifest) -> CapabilityDecision:
    """Revalidate ``decision`` against the live ``manifest``, returning the new snapshot.

    Every field of the normative tuple is compared with a decision freshly issued from the current
    manifest. All equal transitions the snapshot to ``valid``; any divergence transitions it to
    ``invalidated``. The input snapshot is never mutated — a new one is returned (snapshots are
    immutable).

    A snapshot already ``superseded`` or ``retired`` stays as it is: it can never return to use.

    Args:
        decision: The snapshot to revalidate.
        manifest: The live manifest.

    Returns:
        A snapshot in ``valid`` or ``invalidated`` (or unchanged if terminal).
    """
    if decision.decision_lifecycle in ("superseded", "retired"):
        return decision

    decl = manifest.declaration_for(decision.practice)
    if decl is None:
        return replace(decision, decision_lifecycle="invalidated")

    current = issue_decision(
        manifest,
        decision.practice,
        decision_id=decision.decision_id,
        issued_at=decision.issued_at,
    )
    if current.normative_tuple() == decision.normative_tuple():
        return replace(decision, decision_lifecycle="valid", capability_state=decl.capability_state)
    return replace(decision, decision_lifecycle="invalidated")


def ensure_usable(decision: CapabilityDecision, operation: str, correlation_id: str) -> None:
    """Raise unless ``decision`` may back a transform of ``operation`` right now.

    A snapshot not in ``valid`` cannot be used — ``CONTRACT_MISMATCH`` (BR2.4). An ``unsupported``
    capability, or an operation outside the snapshot's coverage, cannot transform —
    ``CAPABILITY_UNSUPPORTED`` (BR2.3). All checks run immediately before transforming, with no durable
    effect.

    Args:
        decision: The snapshot the request carries.
        operation: The operation about to run (``"transform"`` in the current slice).
        correlation_id: An opaque correlation id.

    Raises:
        RefusalError: ``CONTRACT_MISMATCH`` if the lifecycle is not ``valid``;
            ``CAPABILITY_UNSUPPORTED`` if the state is ``unsupported`` or the operation is uncovered.
    """
    if decision.decision_lifecycle != "valid":
        raise RefusalError.of(
            "CONTRACT_MISMATCH", operation, correlation_id, capability=decision.practice
        )
    if decision.capability_state == "unsupported":
        raise RefusalError.of(
            "CAPABILITY_UNSUPPORTED", operation, correlation_id, capability=decision.practice
        )
    if operation not in decision.coverage_operations or operation in decision.coverage_exclusions:
        raise RefusalError.of(
            "CAPABILITY_UNSUPPORTED", operation, correlation_id, capability=decision.practice
        )
