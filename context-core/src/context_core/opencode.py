"""``OpenCodeAdapter`` — a synchronous, offline boundary that drives OpenCode tool results through U1.

This module is the ``opencode-port-spec`` unit (U3): a thin, framework-agnostic adapter that reuses the
neutral :class:`~context_core.runtime.ContextRuntime` and the U1 contracts rather than reimplementing any
of their logic. There are **no real OpenCode hooks here** — the host integration is deliberately a plain,
synchronous Python boundary so the whole admission/refusal/fallback behaviour is testable offline with no
network, no AWS and no agent-framework import.

The boundary carries these immutable records:

- :class:`OpenCodeCoverageEvidence` — the six mandatory evidence digests (payload integrity,
  substitution, retrieval, metadata preservation, isolation, fallback) that back a Practice A coverage
  set; a set is unpublishable unless all six are present (rules BR3.3).
- :class:`OpenCodeCoverageSet` — the closed, versioned, non-empty set of tool classes eligible for
  Practice A, with its fixtures and the evidence above. Startup refuses to publish A unless the set is
  non-empty and every mandatory evidence digest is present (rules BR3.2, BR3.3).
- :class:`OpenCodeSessionBinding` — the per-request binding between an OpenCode call, the U1
  :class:`~context_core.contracts.CapabilityDecision` it speaks, and the authorized namespace. It
  carries the full normative provenance (contract/decision/manifest/evidence/coverage identity and
  digests, host, practice, state, namespace, authorization, operation and ``deadline_ms``) as
  explicitly serializable fields, revalidated for coherence against the decision before any effect
  (rules BR1.1, BR1.2).
- :class:`OpenCodeSourceAlreadyTruncated` — a received-already-truncated shape, preserved verbatim with
  its byte/metadata/shape/origin digests and preservation flags, and **no** reference to content that
  was never received (rules BR3.4).
- :class:`OpenCodeAdapterOutcome` — the single terminal result of one boundary operation, carrying
  ``evidence_ids`` on success/baseline and a :class:`~context_core.contracts.ClosedRefusalEnvelope`
  (exactly seven fields, no payload) on refusal (rules BR8.3, BR3.5).

Admission runs fail-closed in a fixed order (functional-spec Workflow W2, finding C5): host identity,
contract identity, host authorization, namespace — **before** any check that would reveal whether a
Practice B/D capability is supported. Only after authorization and namespace are satisfied does the
boundary negotiate the capability (B/D are ``CAPABILITY_UNSUPPORTED`` in this slice, rules BR5.1) and
then atomically revalidate the decision and the binding's provenance against the live manifest. A source
that already arrived truncated is preserved as a baseline *without* running the transform and *without*
creating any reference to content that was never received (rules BR3.4). ``shutdown`` discards a
namespace synchronously through the runtime (rules BR4.7 via :class:`ContextRuntime`).

Deferred, deliberately out of this slice (documented, not implemented):

- **Gate A / Gate D evaluation.** This boundary publishes Practice A over a proven coverage set and
  keeps B and D ``unsupported``; it does not collect or evaluate ``OpenCodeGateAEvidence`` or
  ``OpenCodeGateDEvidence``. D stays ``unsupported`` regardless of any A result (rules BR5.1).
- **Real OpenCode hook wiring.** No real hooks are registered. :meth:`OpenCodeAdapter.startup` validates
  a caller-declared ``public_hook_set`` against a local declarative allowlist
  (:data:`PUBLIC_HOOK_ALLOWLIST` / :data:`REQUIRED_PUBLIC_HOOKS`) that stands in for the host's hook
  surface; it resolves no credentials and carries no payload.
"""

from __future__ import annotations

import hashlib
import logging
import re
import secrets
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

from .contracts.canonical import digest as _canonical_digest
from .contracts.capability import (
    CapabilityDecision,
    CapabilityDeclaration,
    CapabilityManifest,
    CoverageSet,
    EvidenceDigest,
    issue_decision,
    revalidate,
)
from .contracts.errors import (
    CAPABILITIES,
    ClosedRefusalEnvelope,
    RefusalError,
    make_refusal,
)
from .contracts.idempotency import IdempotencyStore
from .contracts.identity import ContractIdentity
from .contracts.outcome import (
    NeutralResult,
    ProcessingOutcome,
    refused_outcome,
    success_outcome,
)
from .contracts.references import (
    Authorize,
    InMemoryContextStore,
    SessionNamespace,
    allow_all,
)
from .runtime import ContextRuntime, ProcessingRequest, TransformFn

__all__ = [
    "DEFAULT_DEADLINE_MS",
    "EVIDENCE_NAMES",
    "OPENCODE_HOST",
    "PUBLIC_HOOK_ALLOWLIST",
    "REQUIRED_PUBLIC_HOOKS",
    "AdapterOperation",
    "OpenCodeAdapter",
    "OpenCodeAdapterOutcome",
    "OpenCodeCoverageEvidence",
    "OpenCodeCoverageSet",
    "OpenCodeSessionBinding",
    "OpenCodeSourceAlreadyTruncated",
]

_LOGGER = logging.getLogger(__name__)

OPENCODE_HOST = "opencode"
"""The single host identifier this adapter serves; any other value is refused as a contract mismatch."""

DEFAULT_DEADLINE_MS = 30_000
"""The default per-request deadline, in milliseconds, when a binding does not name one."""

_SHA256_RE = re.compile(r"[0-9a-f]{64}")

# The six mandatory evidence facets a Practice A coverage set must prove before A can be published
# (entities.md OpenCodeCoverageSet, rules BR3.3). The order is stable so derived ids are stable.
EVIDENCE_NAMES: tuple[str, ...] = (
    "payload_integrity",
    "substitution",
    "retrieval",
    "metadata_preservation",
    "isolation",
    "fallback",
)

# Declarative stand-in for the host's public hook surface. Real OpenCode hooks are out of this slice
# (see module docstring): startup validates a caller-declared hook set against this allowlist instead of
# registering anything. ``REQUIRED_PUBLIC_HOOKS`` are the hooks this A-relevance boundary needs to be
# offered — the tool-result hook that feeds the transform and the session-shutdown hook that discards a
# namespace.
PUBLIC_HOOK_ALLOWLIST: frozenset[str] = frozenset({"tool.result.after", "session.shutdown"})
REQUIRED_PUBLIC_HOOKS: frozenset[str] = frozenset({"tool.result.after", "session.shutdown"})

# The operations the binding may name. ``transform`` is the Practice A relevance path handled by
# :meth:`OpenCodeAdapter.process`; ``store``/``retrieve`` back baseline preservation and authorized
# retrieval (:meth:`OpenCodeAdapter.store_baseline` / :meth:`OpenCodeAdapter.retrieve`); ``shutdown`` is
# the synchronous namespace discard handled by :meth:`OpenCodeAdapter.shutdown`. ``delete``/``delegate``
# are listed for binding fidelity but are not an entry point of this boundary (reached via the runtime).
BINDING_OPERATIONS: frozenset[str] = frozenset(
    {"transform", "store", "retrieve", "delete", "delegate", "shutdown"}
)
BINDING_PRACTICES: frozenset[str] = frozenset({"A", "B", "D", "none"})

AdapterOperation = Literal[
    "previewed", "projected", "retrieved", "stored", "baseline", "discarded", "refused"
]
"""The closed set of terminal operations an :class:`OpenCodeAdapterOutcome` may report."""


def _digest_of(content: Any) -> str | None:
    """Return the SHA-256 hex of ``content`` when it is ``bytes``/``str``, else ``None``.

    Args:
        content: The content to digest.

    Returns:
        The 64-character lowercase hex digest, or ``None`` when the content is neither bytes nor str.
    """
    if isinstance(content, bytes):
        return hashlib.sha256(content).hexdigest()
    if isinstance(content, str):
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
    return None


@dataclass(frozen=True)
class OpenCodeCoverageEvidence:
    """The six mandatory evidence digests backing a Practice A coverage set.

    Each facet is a SHA-256 hex digest of the artefact proving that facet for the covered tool classes.
    All six are required and must be well-formed; a coverage set with any facet missing or malformed
    cannot publish Practice A (rules BR3.3).

    Attributes:
        payload_integrity_digest: Evidence that an integral payload is preserved end to end.
        substitution_digest: Evidence that a substitution/preview is faithful.
        retrieval_digest: Evidence that retrieval of the preserved content works.
        metadata_preservation_digest: Evidence that functional metadata survives the transform.
        isolation_digest: Evidence that namespaces/subagents stay isolated.
        fallback_digest: Evidence that a recoverable failure preserves the baseline.
    """

    payload_integrity_digest: str
    substitution_digest: str
    retrieval_digest: str
    metadata_preservation_digest: str
    isolation_digest: str
    fallback_digest: str

    def __post_init__(self) -> None:
        """Reject a missing or malformed evidence digest.

        Raises:
            ValueError: If any of the six facets is empty or not a 64-character lowercase SHA-256 hex
                string.
        """
        for name, value in self.as_mapping().items():
            if not value or not _SHA256_RE.fullmatch(value):
                raise ValueError(f"evidence digest {name!r} must be a 64-char sha256 hex string")

    def as_mapping(self) -> dict[str, str]:
        """Return the six facets keyed by :data:`EVIDENCE_NAMES`, in stable order."""
        values = (
            self.payload_integrity_digest,
            self.substitution_digest,
            self.retrieval_digest,
            self.metadata_preservation_digest,
            self.isolation_digest,
            self.fallback_digest,
        )
        return dict(zip(EVIDENCE_NAMES, values, strict=True))

    @property
    def digests(self) -> tuple[str, ...]:
        """The six evidence digests, in :data:`EVIDENCE_NAMES` order."""
        return tuple(self.as_mapping().values())


@dataclass(frozen=True)
class OpenCodeCoverageSet:
    """A closed, versioned, non-empty set of tool classes eligible for Practice A.

    The set is immutable once constructed. Startup publishes Practice A only when at least one tool class
    is explicitly named, at least one fixture is present, and all six mandatory evidence digests are
    present; an empty allowlist, an empty fixture set, or missing evidence is rejected at construction so
    A can never be announced over an unproven, mutable or empty allowlist (rules BR3.2, BR3.3).

    Attributes:
        coverage_set_id: Opaque identity of the set.
        included_tool_classes: The explicitly named tool classes covered by A; must be non-empty.
        evidence: The six mandatory evidence digests backing the set.
        fixture_ids: The opaque fixture ids pinned for the covered classes; must be non-empty.
        excluded_tool_classes: Tool classes explicitly excluded; an excluded class is refused, never
            approximated.
        coverage_set_revision: Positive integer revision; a change requires a new revision and digest.
    """

    coverage_set_id: str
    included_tool_classes: tuple[str, ...]
    evidence: OpenCodeCoverageEvidence
    fixture_ids: tuple[str, ...] = ()
    excluded_tool_classes: tuple[str, ...] = ()
    coverage_set_revision: int = 1

    def __post_init__(self) -> None:
        """Reject an empty/overlapping allowlist, empty fixtures or a non-positive revision.

        Raises:
            ValueError: If ``included_tool_classes`` or ``fixture_ids`` is empty,
                ``coverage_set_revision`` is below 1, or a class appears in both the included and
                excluded lists. (Missing evidence is rejected by :class:`OpenCodeCoverageEvidence`.)
        """
        if not self.included_tool_classes:
            raise ValueError("OpenCodeCoverageSet.included_tool_classes must be non-empty")
        if not self.fixture_ids:
            raise ValueError("OpenCodeCoverageSet.fixture_ids must be non-empty")
        if self.coverage_set_revision < 1:
            raise ValueError("OpenCodeCoverageSet.coverage_set_revision must be >= 1")
        overlap = set(self.included_tool_classes) & set(self.excluded_tool_classes)
        if overlap:
            raise ValueError(f"tool classes cannot be both included and excluded: {sorted(overlap)!r}")

    @property
    def coverage_set_digest(self) -> str:
        """SHA-256 over the canonical content of the set (id excluded, revision/fixtures/evidence in)."""
        return _canonical_digest(
            {
                "revision": self.coverage_set_revision,
                "included_tool_classes": sorted(self.included_tool_classes),
                "excluded_tool_classes": sorted(self.excluded_tool_classes),
                "fixture_ids": sorted(self.fixture_ids),
                "evidence": self.evidence.as_mapping(),
            }
        )

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        """Stable, explicit evidence ids for the six facets, carried by A-relevance outcomes (BR8.3)."""
        return tuple(f"{self.coverage_set_id}:{name}" for name in EVIDENCE_NAMES)

    def to_contract_coverage(self) -> CoverageSet:
        """Project this allowlist onto the U1 :class:`~context_core.contracts.CoverageSet`.

        Practice A operates by transforming an oversized tool result, so the projected coverage covers
        exactly the ``transform`` operation over the included tool classes.

        Returns:
            The contract coverage set backing the Practice A declaration.
        """
        return CoverageSet(
            coverage_set_id=self.coverage_set_id,
            tool_classes=tuple(self.included_tool_classes),
            operation_set=("transform",),
            exclusions=tuple(self.excluded_tool_classes),
        )


@dataclass(frozen=True)
class OpenCodeSourceAlreadyTruncated:
    """A tool result that already arrived truncated at the boundary.

    This is an assertion about the *received* origin, never about the content the host did not deliver.
    The adapter preserves the received bytes, metadata, origin and their digests and creates no
    reference, size, hash or recovery promise for the unreceived content (rules BR3.4). There is
    deliberately no field that could hold such a reference.

    Attributes:
        source_shape_id: Opaque identity of this received shape.
        received_bytes: The bytes that were actually received; preserved verbatim as the baseline.
        received_metadata: The metadata bytes that were actually received; preserved verbatim. Empty
            when the host delivered none.
        origin: Where the truncation happened — ``"host_before_hook"`` or ``"upstream_adapter"``.
        bytes_preserved: ``True`` when the received bytes are preserved verbatim (always the case here).
        metadata_preserved: ``True`` when the received metadata is preserved verbatim.
        shape_preserved: ``True`` when the received shape (marker/origin/digests) is preserved verbatim.
    """

    source_shape_id: str
    received_bytes: bytes
    received_metadata: bytes = b""
    origin: Literal["host_before_hook", "upstream_adapter"] = "host_before_hook"
    bytes_preserved: bool = True
    metadata_preserved: bool = True
    shape_preserved: bool = True

    @property
    def marker(self) -> Literal["source_already_truncated"]:
        """The literal marker preserved in the outcome; always ``"source_already_truncated"``."""
        return "source_already_truncated"

    @property
    def received_bytes_digest(self) -> str:
        """SHA-256 over the received bytes."""
        return hashlib.sha256(self.received_bytes).hexdigest()

    @property
    def received_metadata_digest(self) -> str:
        """SHA-256 over the received metadata bytes (digest of empty bytes when none were received)."""
        return hashlib.sha256(self.received_metadata).hexdigest()

    @property
    def received_shape_digest(self) -> str:
        """SHA-256 over the canonical received shape — marker, origin and the received-content digests.

        The shape digest describes only what was received; it never encodes a size, hash or reference
        for content that was not received.
        """
        return _canonical_digest(
            {
                "marker": self.marker,
                "origin": self.origin,
                "received_bytes_digest": self.received_bytes_digest,
                "received_metadata_digest": self.received_metadata_digest,
            }
        )

    @property
    def origin_digest(self) -> str:
        """SHA-256 over the canonical origin assertion (origin + marker)."""
        return _canonical_digest({"marker": self.marker, "origin": self.origin})

    def provenance(self) -> dict[str, Any]:
        """Return the preserved provenance of the received shape, with no unreceived-content reference.

        Returns:
            A dict of the marker, origin, the four received-content digests and the preservation flags.
            There is deliberately no key referencing content that was never received (rules BR3.4).
        """
        return {
            "marker": self.marker,
            "origin": self.origin,
            "received_bytes_digest": self.received_bytes_digest,
            "received_metadata_digest": self.received_metadata_digest,
            "received_shape_digest": self.received_shape_digest,
            "origin_digest": self.origin_digest,
            "bytes_preserved": self.bytes_preserved,
            "metadata_preserved": self.metadata_preserved,
            "shape_preserved": self.shape_preserved,
        }


@dataclass(frozen=True)
class OpenCodeSessionBinding:
    """The per-request binding between an OpenCode call, its U1 decision and the authorized namespace.

    A binding carries the full identity the adapter revalidates fail-closed before any effect, as
    explicitly serializable fields (not only inside the decision object): the contract identity and its
    version/digest, the immutable capability decision together with its id/revision/manifest/evidence/
    coverage provenance, the published state and implementation kind, the host coordinates, the acting
    namespace and the host's authorization decision. Possessing a binding, digest or reference never
    widens the host's policy (rules BR1.1). The explicit provenance is revalidated for coherence against
    the carried decision and the live manifest before any effect (rules BR1.2); build a coherent binding
    with :meth:`OpenCodeAdapter.bind`.

    Attributes:
        binding_id: Opaque identity of the binding.
        request_id: Opaque correlation id for the request; also the runtime correlation value.
        contract_identity: The contract identity the request speaks.
        decision: The immutable U1 capability decision the request is bound to.
        practice: ``"A"``, ``"B"``, ``"D"`` or ``"none"`` (``"none"`` for operations not tied to a
            practice).
        operation: One of :data:`BINDING_OPERATIONS`.
        contract_version: The contract ``MAJOR.MINOR`` version; must mirror ``contract_identity``.
        contract_digest: The contract schema digest; must mirror ``contract_identity``.
        decision_id: The decision id; must mirror ``decision``.
        decision_revision: The decision revision; must mirror ``decision``.
        manifest_id: The manifest id the decision was issued from; must mirror ``decision``.
        manifest_revision: The manifest revision; must mirror ``decision``.
        manifest_digest: The manifest digest; must mirror ``decision``.
        evidence_digest_id: The evidence id; must mirror ``decision``.
        evidence_digest: The evidence digest; must mirror ``decision``.
        coverage_set_id: The coverage set id; must mirror ``decision``.
        coverage_set_revision: The coverage set revision; must mirror the adapter's coverage set.
        coverage_set_digest: The contract coverage digest; must mirror ``decision``.
        state: The published capability state; must mirror ``decision``.
        implementation_kind: ``"adapter"`` or ``"host"``; must mirror ``decision``.
        host: The host identifier; must be :data:`OPENCODE_HOST`.
        host_version: The host version string; must match the adapter's host version.
        workspace_id: Workspace identity (part of the effective namespace); must be non-empty.
        session_id: Session identity (part of the effective namespace); must be non-empty.
        subagent_id: Subagent identity, or ``None`` when the call is not from a subagent.
        authorization: The host policy decision; ``False`` is an authorization denial.
        namespace_id: The resolved, already-opened namespace id; required for a protected operation.
        deadline_ms: Relative per-request deadline in milliseconds; must be a positive integer.
        idempotency_scope: Idempotency scope for an effect operation.
        idempotency_key: Idempotency key for an effect operation.
        request_digest: Canonical digest of the request for an effect operation.
    """

    binding_id: str
    request_id: str
    contract_identity: ContractIdentity
    decision: CapabilityDecision
    practice: str = "A"
    operation: str = "transform"
    contract_version: str = ""
    contract_digest: str = ""
    decision_id: str = ""
    decision_revision: int = 0
    manifest_id: str = ""
    manifest_revision: int = 0
    manifest_digest: str = ""
    evidence_digest_id: str = ""
    evidence_digest: str = ""
    coverage_set_id: str = ""
    coverage_set_revision: int = 0
    coverage_set_digest: str = ""
    state: str = ""
    implementation_kind: str = ""
    host: str = OPENCODE_HOST
    host_version: str = ""
    workspace_id: str = ""
    session_id: str = ""
    subagent_id: str | None = None
    authorization: bool = True
    namespace_id: str | None = None
    deadline_ms: int = DEFAULT_DEADLINE_MS
    idempotency_scope: str = "opencode"
    idempotency_key: str = ""
    request_digest: str = ""

    def __post_init__(self) -> None:
        """Reject an unknown operation or practice outside the closed boundary vocabulary.

        Raises:
            ValueError: If ``operation`` is not in :data:`BINDING_OPERATIONS` or ``practice`` is not in
                :data:`BINDING_PRACTICES`.
        """
        if self.operation not in BINDING_OPERATIONS:
            raise ValueError(f"unknown binding operation: {self.operation!r}")
        if self.practice not in BINDING_PRACTICES:
            raise ValueError(f"unknown binding practice: {self.practice!r}")

    def provenance(self) -> dict[str, Any]:
        """Return the explicit, serializable normative provenance this binding carries.

        Returns:
            A dict of every provenance field (contract/decision/manifest/evidence/coverage identity and
            digests, host, practice, state, implementation kind, namespace and deadline), suitable for
            an audit record. Carries no payload.
        """
        return {
            "binding_id": self.binding_id,
            "request_id": self.request_id,
            "contract_version": self.contract_version,
            "contract_digest": self.contract_digest,
            "decision_id": self.decision_id,
            "decision_revision": self.decision_revision,
            "manifest_id": self.manifest_id,
            "manifest_revision": self.manifest_revision,
            "manifest_digest": self.manifest_digest,
            "evidence_digest_id": self.evidence_digest_id,
            "evidence_digest": self.evidence_digest,
            "coverage_set_id": self.coverage_set_id,
            "coverage_set_revision": self.coverage_set_revision,
            "coverage_set_digest": self.coverage_set_digest,
            "host": self.host,
            "host_version": self.host_version,
            "practice": self.practice,
            "state": self.state,
            "implementation_kind": self.implementation_kind,
            "workspace_id": self.workspace_id,
            "session_id": self.session_id,
            "subagent_id": self.subagent_id,
            "operation": self.operation,
            "deadline_ms": self.deadline_ms,
        }


@dataclass(frozen=True)
class OpenCodeAdapterOutcome:
    """The single terminal result of one boundary operation.

    A ``refused`` outcome carries only a :class:`~context_core.contracts.ClosedRefusalEnvelope` (its seven
    closed fields and nothing more) and never a payload, reference, evidence id or existence confirmation.
    Any other operation carries the authorized or preserved content plus the explicit ``evidence_ids``
    that back it, and no refusal (rules BR3.5, BR8.3).

    Attributes:
        outcome_id: Opaque identity of the outcome.
        binding_id: The binding this outcome belongs to.
        request_id: The request this outcome belongs to.
        operation: The terminal operation; one of :data:`AdapterOperation`.
        evidence_ids: The explicit evidence ids backing a successful/baseline A-relevance outcome; empty
            for a refusal and for operations that need no gate evidence.
        coverage_set_id: The coverage set in play, when applicable.
        source_shape_id: The received-shape id, when a truncated source was preserved.
        baseline_preserved: ``True`` when an integral baseline was preserved.
        transcript_durable_changed: ``True`` only when a durable transcript change happened (never in
            this slice: D is unsupported and the A path makes no durable transcript change).
        refusal: The closed refusal envelope, for ``refused``; ``None`` otherwise.
        content: The authorized or preserved content; ``None`` for a refusal.
        content_digest: SHA-256 of the content, when known.
        reference_id: The opaque reference id of a stored baseline (``stored``) or the reference just
            retrieved (``retrieved``); ``None`` otherwise and never present on a refusal.
    """

    outcome_id: str
    binding_id: str
    request_id: str
    operation: AdapterOperation
    evidence_ids: tuple[str, ...] = ()
    coverage_set_id: str | None = None
    source_shape_id: str | None = None
    baseline_preserved: bool = False
    transcript_durable_changed: bool = False
    refusal: ClosedRefusalEnvelope | None = None
    content: Any = None
    content_digest: str | None = None
    reference_id: str | None = None

    def __post_init__(self) -> None:
        """Enforce the refusal invariant: a refusal carries an envelope and no payload or evidence.

        Raises:
            ValueError: If the operation/refusal/content pairing contradicts the terminal invariant.
        """
        if self.operation == "refused":
            if self.refusal is None:
                raise ValueError("a refused outcome must carry a closed refusal envelope")
            if self.content is not None or self.content_digest is not None:
                raise ValueError("a refused outcome must not carry any payload")
            if self.reference_id is not None:
                raise ValueError("a refused outcome must not confirm a reference")
            if self.source_shape_id is not None:
                raise ValueError("a refused outcome must not confirm a received shape")
            if self.baseline_preserved:
                raise ValueError("a refused outcome preserves no baseline")
            if self.evidence_ids:
                raise ValueError("a refused outcome must not disclose evidence ids")
        elif self.refusal is not None:
            raise ValueError(f"operation {self.operation!r} must not carry a refusal")


class OpenCodeAdapter:
    """A synchronous, offline OpenCode boundary over :class:`ContextRuntime`.

    The adapter owns the live capability manifest and a neutral :class:`ContextRuntime`. It publishes
    Practice A over a fixed, non-empty :class:`OpenCodeCoverageSet` and declares B and D ``unsupported``.
    It holds no credentials, opens no sockets and writes no files; the clock and the host authorization
    callback are injected.

    Args:
        contract_identity: The contract identity the boundary speaks.
        coverage_set: The closed, non-empty, fully-evidenced Practice A coverage set.
        transform_fn: The neutral Practice A relevance callback; the only boundary to the relevance
            logic. Required to run a transform.
        host_version: The host version string the manifest and bindings must agree on.
        adapter_version: The adapter build version a startup caller must declare and that must match.
        a_state: The published Practice A state; ``"partial"`` or ``"full"``.
        manifest_revision: The manifest revision; increments when the published capabilities change.
        authorize: The host policy callback used by the reference store. Defaults to allow-all.
        clock: Zero-argument current-time source in seconds. Defaults to ``time.monotonic``.
        idempotency_retention_seconds: Retention window for the exactly-once idempotency store.
    """

    def __init__(
        self,
        contract_identity: ContractIdentity,
        coverage_set: OpenCodeCoverageSet,
        *,
        transform_fn: TransformFn | None = None,
        host_version: str,
        adapter_version: str = "0.1.0",
        a_state: Literal["partial", "full"] = "full",
        manifest_revision: int = 1,
        authorize: Authorize | None = None,
        clock: Callable[[], float] | None = None,
        idempotency_retention_seconds: float = 3600.0,
    ) -> None:
        """Build the manifest and wire the neutral runtime (see class docstring for arguments)."""
        import time

        if a_state not in ("partial", "full"):
            raise ValueError(f"Practice A state must be 'partial' or 'full', got {a_state!r}")

        self._contract_identity = contract_identity
        self._coverage_set = coverage_set
        self._host_version = host_version
        self._adapter_version = adapter_version
        self._clock = clock or time.monotonic

        self._manifest = self._build_manifest(
            contract_identity, coverage_set, host_version, a_state, manifest_revision
        )
        self._store = InMemoryContextStore(authorize=authorize or allow_all, clock=self._clock)
        self._idempotency = IdempotencyStore(retention_seconds=idempotency_retention_seconds)
        self._runtime = ContextRuntime(
            self._store,
            self._idempotency,
            contract_identity,
            manifest=self._manifest,
            transform_fn=transform_fn,
            clock=self._clock,
        )

    # ------------------------------------------------------------------ #
    # Public surface                                                     #
    # ------------------------------------------------------------------ #

    @property
    def manifest(self) -> CapabilityManifest:
        """The live capability manifest this adapter publishes."""
        return self._manifest

    @property
    def contract_identity(self) -> ContractIdentity:
        """The contract identity the boundary speaks."""
        return self._contract_identity

    @property
    def adapter_version(self) -> str:
        """The adapter build version a startup caller must declare."""
        return self._adapter_version

    def issue_decision(self, practice: str, *, decision_id: str | None = None) -> CapabilityDecision:
        """Issue an immutable capability decision for ``practice`` from the live manifest.

        A convenience over :func:`~context_core.contracts.issue_decision` so a caller can mint the
        decision a binding carries without reaching into the manifest directly.

        Args:
            practice: ``"A"``, ``"B"`` or ``"D"``; a declaration for it must exist.
            decision_id: Opaque id for the snapshot; a random id is generated when omitted.

        Returns:
            A fresh ``issued`` decision binding the manifest's normative tuple.
        """
        return issue_decision(
            self._manifest,
            practice,
            decision_id=decision_id or secrets.token_hex(16),
            issued_at=self._clock(),
        )

    def bind(
        self,
        decision: CapabilityDecision,
        *,
        operation: str = "transform",
        workspace_id: str,
        session_id: str,
        namespace_id: str | None = None,
        subagent_id: str | None = None,
        authorization: bool = True,
        deadline_ms: int = DEFAULT_DEADLINE_MS,
        practice: str | None = None,
        binding_id: str | None = None,
        request_id: str | None = None,
        idempotency_scope: str = "opencode",
        idempotency_key: str | None = None,
        request_digest: str | None = None,
    ) -> OpenCodeSessionBinding:
        """Build a fully-coherent :class:`OpenCodeSessionBinding` from a decision.

        Every explicit provenance field is derived from ``decision``, the adapter's contract identity
        and its coverage set, so the resulting binding passes the coherence revalidation by construction.
        Negative paths can tamper individual fields with :func:`dataclasses.replace`.

        Args:
            decision: The immutable capability decision the request is bound to.
            operation: One of :data:`BINDING_OPERATIONS`. Defaults to ``"transform"``.
            workspace_id: Workspace identity; must be non-empty for a protected operation.
            session_id: Session identity; must be non-empty for a protected operation.
            namespace_id: The already-opened namespace id; required for a protected operation.
            subagent_id: Subagent identity, when applicable.
            authorization: The host policy decision.
            deadline_ms: Relative per-request deadline in milliseconds; must be positive.
            practice: Override the binding practice; defaults to the decision's practice.
            binding_id: Opaque binding id; generated when omitted.
            request_id: Opaque request id; generated when omitted.
            idempotency_scope: Idempotency scope.
            idempotency_key: Idempotency key; generated when omitted.
            request_digest: Canonical request digest; a stable placeholder is used when omitted.

        Returns:
            The coherent binding.
        """
        identity = self._contract_identity
        coverage = self._coverage_set
        return OpenCodeSessionBinding(
            binding_id=binding_id or secrets.token_hex(8),
            request_id=request_id or secrets.token_hex(8),
            contract_identity=identity,
            decision=decision,
            practice=practice or decision.practice,
            operation=operation,
            contract_version=identity.contract_version,
            contract_digest=identity.contract_digest,
            decision_id=decision.decision_id,
            decision_revision=decision.decision_revision,
            manifest_id=decision.manifest_id,
            manifest_revision=decision.manifest_revision,
            manifest_digest=decision.manifest_digest,
            evidence_digest_id=decision.evidence_digest_id,
            evidence_digest=decision.evidence_digest,
            coverage_set_id=decision.coverage_set_id,
            coverage_set_revision=coverage.coverage_set_revision,
            coverage_set_digest=decision.coverage_set_digest,
            state=decision.capability_state,
            implementation_kind=decision.implementation_kind,
            host=OPENCODE_HOST,
            host_version=self._host_version,
            workspace_id=workspace_id,
            session_id=session_id,
            subagent_id=subagent_id,
            authorization=authorization,
            namespace_id=namespace_id,
            deadline_ms=deadline_ms,
            idempotency_scope=idempotency_scope,
            idempotency_key=idempotency_key or secrets.token_hex(8),
            request_digest=request_digest or ("d" * 64),
        )

    def open_namespace(
        self,
        workspace_id: str,
        session_id: str,
        *,
        lease_ttl: float,
        heartbeat_interval: float,
        abandonment_grace: float,
        memory_quota: int,
        file_quota: int = 0,
        subagent_id: str | None = None,
        parent_session_id: str | None = None,
    ) -> SessionNamespace:
        """Open a memory-only namespace in the boundary's store.

        A subagent namespace is isolated by default: it inherits no access without an explicit
        delegation. All retention values are required; none is invented here.

        Args:
            workspace_id: Workspace identity.
            session_id: Session identity.
            lease_ttl: Lease time-to-live, in seconds.
            heartbeat_interval: Heartbeat window, in seconds.
            abandonment_grace: Grace after abandonment, in seconds.
            memory_quota: Hard in-memory byte quota.
            file_quota: Hard file byte quota (recorded only; persistence disabled).
            subagent_id: Subagent identity, when applicable.
            parent_session_id: Parent session, when applicable.

        Returns:
            The created namespace.
        """
        return self._store.open_namespace(
            workspace_id,
            session_id,
            lease_ttl=lease_ttl,
            heartbeat_interval=heartbeat_interval,
            abandonment_grace=abandonment_grace,
            memory_quota=memory_quota,
            file_quota=file_quota,
            subagent_id=subagent_id,
            parent_session_id=parent_session_id,
        )

    def heartbeat(self, namespace_id: str) -> bool:
        """Renew the retention lease of a session namespace, proving the session is still alive.

        A namespace whose lease is not renewed within its heartbeat window is treated as abandoned and its
        references stop resolving. A host that keeps a session open must therefore renew it; a lease that is
        already expired, revoked or discarded cannot be revived.

        Args:
            namespace_id: The namespace to renew.

        Returns:
            ``True`` when the lease was renewed, ``False`` when it is unknown or no longer renewable.
        """
        try:
            # "store" is the operation a lease renewal belongs to in the closed refusal envelope.
            self._store.heartbeat(namespace_id, operation="store")
        except RefusalError:
            return False
        return True

    def startup(
        self,
        *,
        public_hook_set: Iterable[str],
        adapter_version: str,
        capability_manifest_digest: str,
        request_id: str | None = None,
    ) -> ProcessingOutcome:
        """Acknowledge startup, validate the host/manifest/hook identity, and publish A with B/D unsupported.

        Startup only ever publishes a non-empty, immutable Practice A coverage set (guaranteed by
        :class:`OpenCodeCoverageSet`), with B and D ``unsupported`` (functional-spec Workflow W1). It
        resolves no credential and opens no socket.

        Real OpenCode hooks are out of this slice; ``public_hook_set`` is validated against the local
        declarative allowlist (:data:`PUBLIC_HOOK_ALLOWLIST`) and must offer every hook in
        :data:`REQUIRED_PUBLIC_HOOKS`. The published hooks are the required set, carried as metadata with
        no payload.

        Args:
            public_hook_set: The hooks the host declares available; every one must be in the allowlist.
            adapter_version: The adapter build version the caller declares; must match this adapter.
            capability_manifest_digest: The manifest digest the caller expects; must match the live one.
            request_id: Opaque correlation id; a random id is generated when omitted.

        Returns:
            A ``started`` outcome publishing the capability states, the published hooks and the adapter
            version; or a ``refused`` outcome if the version/manifest/hook identity does not validate.
        """
        corr = request_id or secrets.token_hex(8)
        now = self._clock()

        # Validate the adapter/manifest identity before anything is published.
        if adapter_version != self._adapter_version:
            return refused_outcome(corr, make_refusal("CONTRACT_MISMATCH", "startup", corr), committed_at=now)
        if capability_manifest_digest != self._manifest.manifest_digest:
            return refused_outcome(corr, make_refusal("CONTRACT_MISMATCH", "startup", corr), committed_at=now)

        hooks = frozenset(public_hook_set)
        unknown = hooks - PUBLIC_HOOK_ALLOWLIST
        if unknown:
            # A hook outside the declarative allowlist is a malformed startup request.
            return refused_outcome(
                corr, make_refusal("REQUEST_SCHEMA_INVALID", "startup", corr), committed_at=now
            )
        if not REQUIRED_PUBLIC_HOOKS <= hooks:
            # A required hook was not offered: the boundary cannot be driven, so admission is denied.
            return refused_outcome(
                corr, make_refusal("ADMISSION_POLICY_DENIED", "startup", corr), committed_at=now
            )

        # Reuse the neutral runtime to publish the capability states, then augment the acknowledgement
        # metadata with the published hooks and adapter version (no payload, no credential).
        base = self._runtime.process(
            ProcessingRequest(
                request_id=corr,
                operation="startup",
                operation_family="admission",
                contract_identity=self._contract_identity,
            )
        )
        base_metadata = base.result.metadata if base.result is not None else None
        metadata: dict[str, Any] = dict(base_metadata or {})
        metadata["adapter_version"] = self._adapter_version
        metadata["public_hooks"] = sorted(REQUIRED_PUBLIC_HOOKS)
        metadata["capability_manifest_digest"] = self._manifest.manifest_digest
        return success_outcome(
            corr, "started", NeutralResult(kind="startup_ack", metadata=metadata), committed_at=now
        )

    def process(
        self,
        binding: OpenCodeSessionBinding,
        *,
        neutral_input: Any = None,
        baseline: Any = None,
        source: OpenCodeSourceAlreadyTruncated | None = None,
    ) -> OpenCodeAdapterOutcome:
        """Drive one OpenCode tool result through Practice A relevance filtering (Workflow W2).

        Admission runs fail-closed in a fixed order (finding C5): the ``transform`` entry guard and the
        payload-free schema checks (non-empty workspace/session, positive ``deadline_ms``), then host
        identity, contract identity, host authorization and namespace — all **before** the capability
        negotiation that would reveal whether Practice B/D is supported. Only after authorization and
        namespace does the boundary refuse an unsupported B/D (``CAPABILITY_UNSUPPORTED``, rules BR5.1)
        and then atomically revalidate the decision and the binding provenance against the live manifest.

        A source that already arrived truncated is preserved as a baseline without running the transform
        and without creating a reference to the unreceived content (rules BR3.4). A recoverable
        dependency failure before preview preserves the baseline; without a baseline it is a closed
        refusal (rules BR3.5).

        Args:
            binding: The per-request binding; its ``operation`` must be ``"transform"``.
            neutral_input: The neutral tool result handed to the relevance callback.
            baseline: The integral original to preserve on a fallback; ``None`` means no recoverable
                baseline exists.
            source: A received-already-truncated shape to preserve verbatim, when applicable.

        Returns:
            The single terminal :class:`OpenCodeAdapterOutcome`.
        """
        capability = binding.practice if binding.practice in CAPABILITIES else "none"

        # Entry-point guard and payload-free schema checks: these reveal nothing about capability.
        if binding.operation != "transform":
            return self._refuse(binding, "REQUEST_SCHEMA_INVALID", "transform", capability)
        if not binding.workspace_id or not binding.session_id:
            return self._refuse(binding, "REQUEST_SCHEMA_INVALID", "transform", capability)
        if binding.deadline_ms <= 0:
            # An expired/invalid deadline is rejected with no effect (Majors: deadline_ms).
            return self._refuse(binding, "REQUEST_SCHEMA_INVALID", "transform", capability)

        # Fail-closed admission in fixed order: host, contract, authorization, namespace (C5). This runs
        # before any capability-revealing check, so an unauthorized caller never learns B/D support.
        envelope = self._admit(binding, "transform", capability, require_namespace=True)
        if envelope is not None:
            return self._refused_outcome(binding, envelope)

        # Capability negotiation: only Practice A is supported in this slice (BR5.1). Reached only after
        # authorization and namespace have passed.
        if binding.practice != "A":
            return self._refuse(binding, "CAPABILITY_UNSUPPORTED", "transform", capability)

        # Atomic revalidation of the binding provenance and the decision against the live manifest.
        envelope = self._revalidate(binding, "transform", "A")
        if envelope is not None:
            return self._refused_outcome(binding, envelope)

        # A source that already arrived truncated: preserve the received bytes verbatim, run no
        # transform and create no reference to content that was never received (BR3.4).
        if source is not None:
            _LOGGER.debug(
                "opencode.source_already_truncated",
                extra={"request_id": binding.request_id, "source_shape_id": source.source_shape_id},
            )
            return OpenCodeAdapterOutcome(
                outcome_id=secrets.token_hex(16),
                binding_id=binding.binding_id,
                request_id=binding.request_id,
                operation="baseline",
                evidence_ids=self._coverage_set.evidence_ids,
                coverage_set_id=self._coverage_set.coverage_set_id,
                source_shape_id=source.source_shape_id,
                baseline_preserved=True,
                content=source.received_bytes,
                content_digest=source.received_bytes_digest,
            )

        # Map to the neutral runtime transform. The runtime revalidates the decision again and runs the
        # injected relevance callback — the only boundary to Practice A logic.
        now = self._clock()
        outcome = self._runtime.process(
            ProcessingRequest(
                request_id=binding.request_id,
                operation="transform",
                operation_family="runtime",
                contract_identity=binding.contract_identity,
                deadline_at=self._deadline_at(binding, now),
                idempotency_scope=binding.idempotency_scope,
                idempotency_key=binding.idempotency_key,
                request_digest=binding.request_digest,
                practice="A",
                capability_decision=binding.decision,
                neutral_input=neutral_input,
                baseline=baseline,
            )
        )
        return self._map_transform_outcome(binding, outcome, baseline)

    def shutdown(self, binding: OpenCodeSessionBinding) -> OpenCodeAdapterOutcome:
        """Discard the binding's namespace synchronously through the runtime.

        The namespace is zeroed, its grants revoked and its lease ended before this returns, so no
        content survives (rules BR4.7, enforced by :class:`ContextRuntime`). The request is admitted
        fail-closed (host, contract, authorization, namespace) before any discard happens.

        Args:
            binding: The binding naming the namespace to discard; ``operation`` must be ``"shutdown"``
                and ``namespace_id`` must be set.

        Returns:
            The single terminal :class:`OpenCodeAdapterOutcome`; ``discarded`` on success.
        """
        if binding.operation != "shutdown":
            return self._refuse(binding, "REQUEST_SCHEMA_INVALID", "shutdown", "none")
        if binding.deadline_ms <= 0:
            return self._refuse(binding, "REQUEST_SCHEMA_INVALID", "shutdown", "none")

        envelope = self._admit(binding, "shutdown", "none", require_namespace=True)
        if envelope is not None:
            return self._refused_outcome(binding, envelope)

        now = self._clock()
        outcome = self._runtime.process(
            ProcessingRequest(
                request_id=binding.request_id,
                operation="shutdown",
                operation_family="runtime",
                contract_identity=binding.contract_identity,
                deadline_at=self._deadline_at(binding, now),
                idempotency_scope=binding.idempotency_scope,
                idempotency_key=binding.idempotency_key,
                request_digest=binding.request_digest,
                namespace_id=binding.namespace_id,
            )
        )
        if outcome.status == "deleted":
            return OpenCodeAdapterOutcome(
                outcome_id=secrets.token_hex(16),
                binding_id=binding.binding_id,
                request_id=binding.request_id,
                operation="discarded",
            )
        if outcome.status == "refused":
            assert outcome.refusal is not None  # guaranteed by refused_outcome
            return self._refused_outcome(binding, outcome.refusal)
        # No other terminal is reachable for shutdown with a sane (non-expired) deadline.
        raise RuntimeError(f"unexpected shutdown terminal status: {outcome.status!r}")  # pragma: no cover

    def store_baseline(
        self, binding: OpenCodeSessionBinding, *, content: bytes, content_type: str = "text/plain"
    ) -> OpenCodeAdapterOutcome:
        """Store an integral baseline in the binding's namespace and return its opaque reference.

        This backs Practice A retrieval: when a covered, oversized result is previewed, the untouched
        original is preserved in the session namespace so it can later be retrieved verbatim through
        :meth:`retrieve`. The request is admitted fail-closed (host, contract, authorization, namespace)
        before any content is stored; the memory-only store enforces lease and quota.

        Args:
            binding: The per-request binding; its ``operation`` must be ``"store"``.
            content: The integral baseline bytes to preserve verbatim.
            content_type: MIME type recorded with the stored content.

        Returns:
            A ``stored`` outcome carrying the opaque ``reference_id``, or a closed refusal.
        """
        if binding.operation != "store":
            return self._refuse(binding, "REQUEST_SCHEMA_INVALID", "store", "none")
        if binding.deadline_ms <= 0:
            return self._refuse(binding, "REQUEST_SCHEMA_INVALID", "store", "none")

        envelope = self._admit(binding, "store", "none", require_namespace=True)
        if envelope is not None:
            return self._refused_outcome(binding, envelope)

        now = self._clock()
        outcome = self._runtime.process(
            ProcessingRequest(
                request_id=binding.request_id,
                operation="store",
                operation_family="runtime",
                contract_identity=binding.contract_identity,
                deadline_at=self._deadline_at(binding, now),
                idempotency_scope=binding.idempotency_scope,
                idempotency_key=binding.idempotency_key,
                request_digest=binding.request_digest,
                namespace_id=binding.namespace_id,
                content=content,
                content_type=content_type,
            )
        )
        if outcome.status == "stored":
            result = outcome.result
            assert result is not None  # guaranteed by success_outcome
            return OpenCodeAdapterOutcome(
                outcome_id=secrets.token_hex(16),
                binding_id=binding.binding_id,
                request_id=binding.request_id,
                operation="stored",
                coverage_set_id=self._coverage_set.coverage_set_id,
                reference_id=result.reference_id,
                content_digest=result.content_digest,
            )
        if outcome.status == "refused":
            assert outcome.refusal is not None  # guaranteed by refused_outcome
            return self._refused_outcome(binding, outcome.refusal)
        # No other terminal is reachable for an in-memory store with a sane (non-expired) deadline.
        raise RuntimeError(f"unexpected store terminal status: {outcome.status!r}")  # pragma: no cover

    def retrieve(
        self, binding: OpenCodeSessionBinding, *, reference_id: str
    ) -> OpenCodeAdapterOutcome:
        """Retrieve a previously stored baseline verbatim, under authorization, namespace and lease.

        A read-only operation: it resolves no credential, calls no reranker and makes no durable change.
        The request is admitted fail-closed (host, contract, authorization, namespace) before the store
        is touched, and the store itself denies a non-owner, an expired lease, or a missing/closed
        reference with a closed refusal that never confirms whether the reference exists.

        Args:
            binding: The per-request binding; its ``operation`` must be ``"retrieve"``.
            reference_id: The opaque reference to retrieve.

        Returns:
            A ``retrieved`` outcome carrying the authorized content, or a closed refusal.
        """
        if binding.operation != "retrieve":
            return self._refuse(binding, "REQUEST_SCHEMA_INVALID", "retrieve", "none")
        if not reference_id:
            return self._refuse(binding, "REQUEST_SCHEMA_INVALID", "retrieve", "none")
        if binding.deadline_ms <= 0:
            return self._refuse(binding, "REQUEST_SCHEMA_INVALID", "retrieve", "none")

        envelope = self._admit(binding, "retrieve", "none", require_namespace=True)
        if envelope is not None:
            return self._refused_outcome(binding, envelope)

        now = self._clock()
        outcome = self._runtime.process(
            ProcessingRequest(
                request_id=binding.request_id,
                operation="retrieve",
                operation_family="runtime",
                contract_identity=binding.contract_identity,
                deadline_at=self._deadline_at(binding, now),
                namespace_id=binding.namespace_id,
                reference_id=reference_id,
            )
        )
        if outcome.status == "retrieved":
            result = outcome.result
            assert result is not None  # guaranteed by success_outcome
            return OpenCodeAdapterOutcome(
                outcome_id=secrets.token_hex(16),
                binding_id=binding.binding_id,
                request_id=binding.request_id,
                operation="retrieved",
                coverage_set_id=self._coverage_set.coverage_set_id,
                reference_id=reference_id,
                content=result.content,
                content_digest=_digest_of(result.content),
            )
        if outcome.status == "refused":
            assert outcome.refusal is not None  # guaranteed by refused_outcome
            return self._refused_outcome(binding, outcome.refusal)
        # No other terminal is reachable for an in-memory retrieve with a sane (non-expired) deadline.
        raise RuntimeError(f"unexpected retrieve terminal status: {outcome.status!r}")  # pragma: no cover

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _deadline_at(binding: OpenCodeSessionBinding, now: float) -> float:
        """Convert the binding's ``deadline_ms`` to an absolute deadline in the runtime clock's units.

        This is the single place the millisecond deadline is turned into the clock's second-based
        absolute value. Callers reject a non-positive ``deadline_ms`` before reaching here.

        Args:
            binding: The binding carrying ``deadline_ms``.
            now: The current clock value, in seconds.

        Returns:
            The absolute deadline, in seconds.
        """
        return now + binding.deadline_ms / 1000.0

    @staticmethod
    def _build_manifest(
        contract_identity: ContractIdentity,
        coverage_set: OpenCodeCoverageSet,
        host_version: str,
        a_state: Literal["partial", "full"],
        revision: int,
    ) -> CapabilityManifest:
        """Build the manifest publishing A over the coverage set, with B and D ``unsupported``.

        The Practice A declaration binds all six mandatory evidence digests plus the coverage digest, so
        a change to any evidence facet changes the decision's evidence digest and invalidates a stale
        decision on revalidation.
        """
        a_declaration = CapabilityDeclaration(
            declaration_id="opencode-A",
            practice="A",
            capability_state=a_state,
            implementation_kind="adapter",
            coverage=coverage_set.to_contract_coverage(),
            evidence=EvidenceDigest(
                evidence_digest_id="opencode-A-evidence",
                artifact_digests=(*coverage_set.evidence.digests, coverage_set.coverage_set_digest),
                compatibility_matrix_version="1.0",
            ),
        )
        unsupported = tuple(
            CapabilityDeclaration(
                declaration_id=f"opencode-{practice}",
                practice=practice,  # type: ignore[arg-type]
                capability_state="unsupported",
                implementation_kind="adapter",
                coverage=CoverageSet(coverage_set_id=f"opencode-{practice}-none"),
                evidence=EvidenceDigest(evidence_digest_id=f"opencode-{practice}-evidence"),
            )
            for practice in ("B", "D")
        )
        return CapabilityManifest(
            manifest_id="opencode-manifest",
            host=OPENCODE_HOST,
            host_version=host_version,
            contract_identity=contract_identity,
            revision=revision,
            declarations=(a_declaration, *unsupported),
        )

    def _admit(
        self,
        binding: OpenCodeSessionBinding,
        operation: str,
        capability: str,
        *,
        require_namespace: bool,
    ) -> ClosedRefusalEnvelope | None:
        """Admit a binding fail-closed; return a closed refusal envelope on the first failure, else None.

        The order is fixed (functional-spec Workflow W2, finding C5): host identity, contract identity,
        host authorization, then namespace. Every check here is independent of the capability support
        state, so no step reveals whether Practice B/D is supported. The first failing check decides the
        refusal; no effect, reference or existence confirmation is produced.
        """
        # 1. Host identity: the binding and its decision must speak this host and version.
        if (
            binding.host != OPENCODE_HOST
            or binding.decision.host != OPENCODE_HOST
            or binding.host_version != self._host_version
            or binding.decision.host_version != self._host_version
        ):
            return self._envelope("CONTRACT_MISMATCH", operation, capability, binding.request_id)

        # 2. Contract identity: same major and identical digest as the boundary holds.
        if not binding.contract_identity.is_compatible_with(self._contract_identity):
            return self._envelope("CONTRACT_MISMATCH", operation, capability, binding.request_id)

        # 3. Host authorization is authoritative; a binding never widens it. Checked before any
        #    capability-revealing step so an unauthorized caller learns nothing about B/D (C5).
        if not binding.authorization:
            return self._envelope("AUTHORIZATION_HOST_DENIED", operation, capability, binding.request_id)

        # 4. Namespace: a protected operation must name an acting namespace with a non-empty
        #    workspace/session identity (BR4.1, Majors).
        if require_namespace and (
            not binding.namespace_id or not binding.workspace_id or not binding.session_id
        ):
            return self._envelope("NAMESPACE_INVALID", operation, capability, binding.request_id)

        return None

    def _revalidate(
        self, binding: OpenCodeSessionBinding, operation: str, capability: str
    ) -> ClosedRefusalEnvelope | None:
        """Atomically revalidate the binding provenance and decision; return an envelope on divergence.

        Two checks, both mapped to ``CONTRACT_MISMATCH`` (rules BR1.2): first the binding's explicit
        provenance must mirror the carried decision, the adapter's contract identity and the coverage
        set revision; then the decision itself must still be ``valid`` against the live manifest. A stale
        decision (manifest revision/digest changed) or tampered provenance refuses before any effect.
        """
        decision = binding.decision
        identity = self._contract_identity
        coverage = self._coverage_set
        coherent = (
            binding.contract_version == identity.contract_version
            and binding.contract_digest == identity.contract_digest
            and binding.contract_identity == decision.contract_identity
            and binding.decision_id == decision.decision_id
            and binding.decision_revision == decision.decision_revision
            and binding.manifest_id == decision.manifest_id
            and binding.manifest_revision == decision.manifest_revision
            and binding.manifest_digest == decision.manifest_digest
            and binding.evidence_digest_id == decision.evidence_digest_id
            and binding.evidence_digest == decision.evidence_digest
            and binding.coverage_set_id == decision.coverage_set_id
            and binding.coverage_set_digest == decision.coverage_set_digest
            and binding.coverage_set_revision == coverage.coverage_set_revision
            and binding.state == decision.capability_state
            and binding.implementation_kind == decision.implementation_kind
            and binding.practice == decision.practice
            and binding.host == decision.host
            and binding.host_version == decision.host_version
        )
        if not coherent:
            return self._envelope("CONTRACT_MISMATCH", operation, capability, binding.request_id)

        revalidated = revalidate(decision, self._manifest)
        if revalidated.decision_lifecycle != "valid":
            return self._envelope("CONTRACT_MISMATCH", operation, capability, binding.request_id)
        return None

    def _map_transform_outcome(
        self,
        binding: OpenCodeSessionBinding,
        outcome: ProcessingOutcome,
        baseline: Any,
    ) -> OpenCodeAdapterOutcome:
        """Map a runtime transform :class:`ProcessingOutcome` to an :class:`OpenCodeAdapterOutcome`."""
        evidence_ids = self._coverage_set.evidence_ids
        if outcome.status == "transformed":
            result = outcome.result
            assert result is not None  # guaranteed by success_outcome
            return OpenCodeAdapterOutcome(
                outcome_id=secrets.token_hex(16),
                binding_id=binding.binding_id,
                request_id=binding.request_id,
                operation="previewed",
                evidence_ids=evidence_ids,
                coverage_set_id=self._coverage_set.coverage_set_id,
                content=result.content,
                content_digest=result.content_digest,
            )
        if outcome.status == "fallback_original":
            result = outcome.result
            assert result is not None  # guaranteed by fallback_outcome
            return OpenCodeAdapterOutcome(
                outcome_id=secrets.token_hex(16),
                binding_id=binding.binding_id,
                request_id=binding.request_id,
                operation="baseline",
                evidence_ids=evidence_ids,
                coverage_set_id=self._coverage_set.coverage_set_id,
                baseline_preserved=True,
                content=result.content,
                content_digest=result.content_digest,
            )
        if outcome.status == "timed_out":
            # A deadline that passed before preview is a recoverable pre-preview failure: preserve the
            # baseline when one exists, otherwise refuse closed with no partial content (BR3.5).
            if baseline is not None:
                return OpenCodeAdapterOutcome(
                    outcome_id=secrets.token_hex(16),
                    binding_id=binding.binding_id,
                    request_id=binding.request_id,
                    operation="baseline",
                    evidence_ids=evidence_ids,
                    coverage_set_id=self._coverage_set.coverage_set_id,
                    baseline_preserved=True,
                    content=baseline,
                    content_digest=_digest_of(baseline),
                )
            return self._refuse(binding, "DEPENDENCY_UNAVAILABLE_NO_FALLBACK", "transform", "A")
        if outcome.status == "refused":
            assert outcome.refusal is not None  # guaranteed by refused_outcome
            return self._refused_outcome(binding, outcome.refusal)
        # No other terminal is reachable for a transform request.
        raise RuntimeError(f"unexpected transform terminal status: {outcome.status!r}")  # pragma: no cover

    def _refuse(
        self,
        binding: OpenCodeSessionBinding,
        code: str,
        operation: str,
        capability: str,
    ) -> OpenCodeAdapterOutcome:
        """Build a refused outcome from a stable code, operation and capability."""
        return self._refused_outcome(
            binding, self._envelope(code, operation, capability, binding.request_id)
        )

    def _refused_outcome(
        self, binding: OpenCodeSessionBinding, envelope: ClosedRefusalEnvelope
    ) -> OpenCodeAdapterOutcome:
        """Wrap a closed refusal envelope in a refused :class:`OpenCodeAdapterOutcome`."""
        _LOGGER.info(
            "opencode.refused",
            extra={
                "request_id": binding.request_id,
                "code": envelope.code,
                "category": envelope.category,
                "operation": envelope.operation,
            },
        )
        return OpenCodeAdapterOutcome(
            outcome_id=secrets.token_hex(16),
            binding_id=binding.binding_id,
            request_id=binding.request_id,
            operation="refused",
            refusal=envelope,
        )

    @staticmethod
    def _envelope(
        code: str, operation: str, capability: str, correlation_id: str
    ) -> ClosedRefusalEnvelope:
        """Build a closed refusal envelope, deriving category and retry from the stable catalog."""
        return make_refusal(code, operation, correlation_id, capability=capability)
