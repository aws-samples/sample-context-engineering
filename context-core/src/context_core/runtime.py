"""``ContextRuntime`` — the neutral orchestrator that runs W0–W4 over the U1 contracts.

This module is the second half of U1: a small, framework-agnostic runtime that drives one
:class:`ProcessingRequest` to exactly one :class:`~context_core.contracts.ProcessingOutcome`, reusing the
contracts already defined under :mod:`context_core.contracts` — it never reimplements their logic.

The runtime enforces the normative order (rules.md, "A ordem normativa é"):

1. **schema** — validate the closed operation vocabulary, the operation/family partition and the
   per-operation required fields; an unknown operation, a family mismatch or a missing field is
   ``REQUEST_SCHEMA_INVALID`` before any effect (BR7.1, BR1.3).
2. **idempotency** — for an operation with a durable effect (``transform``/``store``/``delete``/
   ``delegate``), reserve ``(scope, key)`` bound to ``request_digest``: a matching retry replays the
   committed outcome, a different digest is ``IDEMPOTENCY_CONFLICT`` (BR5.3).
3. **contract identity** — ``CONTRACT_MISMATCH`` unless the request speaks the identity the runtime
   holds (BR1.3).
4. **capability** — for ``transform`` only, revalidate the decision against the live manifest
   *immediately before* transforming and refuse an unsupported or uncovered capability (BR2.2–BR2.4).
5. **references** — ``store``/``retrieve``/``delete``/``catalog``/``delegate``/``shutdown`` go through
   the memory-only store, which checks namespace, host authorization, lease and quota (BR4.1–BR4.12).
   ``shutdown`` is a durable effect: it discards a namespace synchronously (zeroing its references,
   revoking its grants, ending its lease) before the terminal outcome is produced.
6. **transform** — a neutral, injected callback is the only boundary to Practice A/D relevance/graph
   logic; the runtime never reimplements it. A dependency failure with an integral baseline becomes a
   ``fallback_original`` outcome; without a baseline it is ``DEPENDENCY_UNAVAILABLE_NO_FALLBACK``
   (BR4.10, BR5.4).
7. **commit** — reserve, durable effect and the terminal outcome commit as one atomic boundary; a
   deadline that passes before commit yields ``timed_out`` with no effect, and a late response never
   creates a second effect (BR5.1, BR5.4, BR5.5).

Every deterministic failure is converted to a :func:`~context_core.contracts.refused_outcome` carrying a
:class:`~context_core.contracts.ClosedRefusalEnvelope`; no payload is ever propagated through a refusal.
There is no filesystem, no persistence, no network, no AWS and no agent-framework import here: the store
is the in-memory one, and the clock is injected.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from .contracts.capability import (
    CapabilityDecision,
    CapabilityManifest,
    ensure_usable,
    revalidate,
)
from .contracts.errors import (
    ADMISSION_OPERATIONS,
    OPERATIONS,
    RUNTIME_OPERATIONS,
    Operation,
    RefusalError,
)
from .contracts.idempotency import IdempotencyStore, ReservationState
from .contracts.identity import ContractIdentity, require_compatible
from .contracts.outcome import (
    FALLBACK_REASONS,
    NeutralResult,
    ProcessingOutcome,
    fallback_outcome,
    refused_outcome,
    success_outcome,
)
from .contracts.references import InMemoryContextStore

__all__ = [
    "EFFECT_OPERATIONS",
    "ContextRuntime",
    "DependencyUnavailable",
    "ProcessingRequest",
    "TransformFn",
]

OperationFamily = Literal["runtime", "admission"]

TransformFn = Callable[[Any], Any]
"""A neutral transform callback: it maps a neutral input to a neutral result, or raises
:class:`DependencyUnavailable` when a dependency (reranker, codec, local process) is unavailable."""

# Operations that create a durable effect and therefore reserve/commit an idempotency record.
# ``shutdown`` discards a namespace synchronously, so it is a durable effect and reserves/commits like
# the others. The remaining operations are read-only or pure liveness and are never reserved:
# ``retrieve``/``catalog`` read, and ``startup``/``health``/``preflight`` carry no durable effect (BR5.2).
EFFECT_OPERATIONS: frozenset[str] = frozenset({"transform", "store", "delete", "delegate", "shutdown"})


class DependencyUnavailable(Exception):
    """Raised by a :data:`TransformFn` when a dependency is unavailable before a recoverable preview.

    The runtime catches this and, per FR3.5/BR4.10, either preserves an integral baseline as a
    ``fallback_original`` outcome or — when no baseline is available — refuses deterministically with
    ``DEPENDENCY_UNAVAILABLE_NO_FALLBACK``. Any *other* exception from the callback is treated as a
    programming defect and is allowed to propagate rather than being masked as a fallback.

    Attributes:
        reason: Why the dependency was unavailable; one of
            :data:`~context_core.contracts.FALLBACK_REASONS`.
    """

    def __init__(self, reason: str = "local_process_unavailable") -> None:
        """Record the fallback reason.

        Args:
            reason: The fallback reason; must be in :data:`~context_core.contracts.FALLBACK_REASONS`.

        Raises:
            ValueError: If ``reason`` is not a known fallback reason.
        """
        if reason not in FALLBACK_REASONS:
            raise ValueError(f"unknown fallback reason: {reason!r}")
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class ProcessingRequest:
    """A single neutral runtime/admission request the runtime drives to one terminal outcome.

    Only ``request_id``, ``operation``, ``operation_family`` and ``contract_identity`` are always
    required; the remaining fields are per-operation and validated in the schema step. Effect operations
    additionally require ``idempotency_scope``, ``idempotency_key`` and ``request_digest``.

    Attributes:
        request_id: Opaque correlation id; also the default idempotency/correlation value.
        operation: The closed operation (see :data:`~context_core.contracts.OPERATIONS`).
        operation_family: ``"runtime"`` or ``"admission"``; must match the operation's family.
        contract_identity: The contract identity the request speaks.
        deadline_at: Absolute deadline in the runtime clock's units. Defaults to no deadline.
        idempotency_scope: Idempotency scope; required for an effect operation.
        idempotency_key: Idempotency key; required for an effect operation.
        request_digest: Canonical digest of the request; required for an effect operation.
        practice: ``"A"``/``"B"``/``"D"``; required for ``transform``.
        capability_decision: The capability snapshot; required for ``transform``.
        namespace_id: The acting namespace; required for every reference operation.
        neutral_input: The neutral input handed to the transform callback.
        baseline: The integral original result to preserve on a transform fallback; ``None`` means no
            recoverable baseline exists, so a dependency failure refuses instead of falling back.
        content: The bytes to store (``store`` only).
        content_type: MIME type for ``store``.
        reference_id: The target reference (``retrieve``/``delete``).
        recipient_namespace_id: The delegation recipient (``delegate``).
        reference_ids: The named references to delegate (``delegate``).
        grant_expires_at: Absolute grant expiry (``delegate``).
    """

    request_id: str
    operation: Operation
    operation_family: OperationFamily
    contract_identity: ContractIdentity
    deadline_at: float = float("inf")
    idempotency_scope: str = ""
    idempotency_key: str = ""
    request_digest: str = ""
    practice: str | None = None
    capability_decision: CapabilityDecision | None = None
    namespace_id: str | None = None
    neutral_input: Any = None
    baseline: Any = None
    content: bytes | None = None
    content_type: str = "text/plain"
    reference_id: str | None = None
    recipient_namespace_id: str | None = None
    reference_ids: tuple[str, ...] = ()
    grant_expires_at: float | None = None


# Reference operations that must name a namespace before any content is touched (BR4.1).
_REFERENCE_OPERATIONS: frozenset[str] = frozenset(
    {"store", "retrieve", "delete", "catalog", "delegate", "shutdown"}
)


class ContextRuntime:
    """Orchestrates W0–W4 for one request at a time over the in-memory U1 contracts.

    The runtime owns the shared stores and the live capability manifest, and it is handed a neutral
    transform callback. It holds no credentials, opens no sockets and writes no files.

    Args:
        store: The memory-only reference store.
        idempotency: The exactly-once idempotency store.
        contract_identity: The contract identity the runtime currently holds.
        manifest: The live capability manifest used to revalidate a transform decision. Required to run
            ``transform``; may be ``None`` when only non-transform operations are used.
        transform_fn: The neutral transform callback. Required to run ``transform``.
        clock: Zero-argument current-time source in seconds. Defaults to ``time.monotonic``.
    """

    def __init__(
        self,
        store: InMemoryContextStore,
        idempotency: IdempotencyStore,
        contract_identity: ContractIdentity,
        *,
        manifest: CapabilityManifest | None = None,
        transform_fn: TransformFn | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Wire the runtime to its stores, identity, manifest and clock."""
        import time

        self._store = store
        self._idempotency = idempotency
        self._contract_identity = contract_identity
        self._manifest = manifest
        self._transform_fn = transform_fn
        self._clock = clock or time.monotonic

    # ------------------------------------------------------------------ #
    # Public entry point                                                 #
    # ------------------------------------------------------------------ #

    def process(self, request: ProcessingRequest) -> ProcessingOutcome:
        """Drive ``request`` to exactly one terminal :class:`ProcessingOutcome`.

        Applies the normative order (schema, idempotency, contract, capability, references, transform,
        commit). Any deterministic precondition failure returns a refused outcome carrying a closed
        envelope; a deadline that passes before commit returns ``timed_out`` with no durable effect.

        Args:
            request: The request to process.

        Returns:
            The single terminal outcome. Never raises for a deterministic refusal.
        """
        now = self._clock()

        # 1. Schema and operation vocabulary.
        try:
            self._validate_schema(request)
        except RefusalError as error:
            return refused_outcome(request.request_id, error.envelope, committed_at=now)

        if request.operation in EFFECT_OPERATIONS:
            return self._process_with_effect(request, now)

        # Read-only / liveness operations: no reservation, no durable effect.
        try:
            return self._run_readonly(request, now)
        except RefusalError as error:
            return refused_outcome(request.request_id, error.envelope, committed_at=now)

    # ------------------------------------------------------------------ #
    # Effect path: reserve -> validate -> effect -> commit               #
    # ------------------------------------------------------------------ #

    def _process_with_effect(self, request: ProcessingRequest, now: float) -> ProcessingOutcome:
        """Run an effect operation under the atomic reserve/commit boundary."""
        scope, key, digest = request.idempotency_scope, request.idempotency_key, request.request_digest

        # 2. Idempotency: reserve, or replay/conflict/pending.
        try:
            reservation = self._idempotency.reserve(
                scope, key, digest, request.request_id, request.operation, now,
                correlation_id=request.request_id,
            )
        except RefusalError as error:  # IDEMPOTENCY_CONFLICT
            return refused_outcome(request.request_id, error.envelope, committed_at=now)

        if reservation.state is ReservationState.REPLAY:
            # The store guarantees a committed REPLAY always carries its terminal outcome; narrow the
            # type with an explicit guard rather than an ``assert`` (which -O would strip). A REPLAY
            # without an outcome is an internal invariant defect, not a request-level condition, so it
            # propagates as a defect rather than being masked as a refusal (consistent with W4).
            replayed = reservation.outcome
            if replayed is None:  # pragma: no cover - the idempotency store never returns this
                raise RuntimeError("REPLAY reservation is missing its committed outcome")
            return replayed
        if reservation.state is ReservationState.PENDING:
            # Another in-flight request holds this key. We never re-execute an effect; in this
            # synchronous slice there is nothing to wait on, so the terminal is a payload-free timeout.
            return self._timed_out(request, now)

        # 3..6. Validate and run the effect, converting a deterministic failure to a refusal outcome.
        try:
            outcome = self._run_effect(request, now)
        except RefusalError as error:
            outcome = refused_outcome(request.request_id, error.envelope, committed_at=now)
        except Exception:
            # An unexpected failure (for example a defect raised by the transform callback that is not a
            # DependencyUnavailable) must not strand the reservation in RESERVED and block every future
            # retry of this key. Release the reservation, then let the exception propagate unchanged
            # (W4). No partial effect is committed on this path.
            self._idempotency.cancel(scope, key)
            raise

        # 7. Commit atomically — except a timeout, which cancels the reservation and leaves no effect.
        if outcome.status == "timed_out":
            self._idempotency.cancel(scope, key)
            return outcome
        return self._idempotency.commit(scope, key, digest, outcome, now)

    def _run_effect(self, request: ProcessingRequest, now: float) -> ProcessingOutcome:
        """Validate contract/capability and perform one effect operation."""
        # 3. Contract identity.
        self._require_contract(request)

        if request.operation == "transform":
            return self._do_transform(request, now)

        # 7 (precedence). A deadline already past means the effect must not mutate state.
        if self._expired(request, now):
            return self._timed_out(request, now)

        if request.operation == "store":
            return self._do_store(request, now)
        if request.operation == "delete":
            return self._do_delete(request, now)
        if request.operation == "delegate":
            return self._do_delegate(request, now)
        # shutdown
        return self._do_shutdown(request, now)

    def _do_transform(self, request: ProcessingRequest, now: float) -> ProcessingOutcome:
        """Revalidate the decision against the live manifest, then run the neutral callback."""
        if self._manifest is None or self._transform_fn is None:
            # The runtime was not configured to transform; refuse rather than invent a capability.
            raise RefusalError.of("CAPABILITY_UNSUPPORTED", "transform", request.request_id)
        decision = request.capability_decision
        if decision is None:
            # Schema validation already guarantees this for ``transform``; the explicit refusal keeps
            # the invariant enforced even if assertions are stripped (-O) and narrows the type.
            raise RefusalError.of("REQUEST_SCHEMA_INVALID", "transform", request.request_id)

        # 4. Revalidate immediately before transforming, then enforce lifecycle/state/coverage.
        revalidated = revalidate(decision, self._manifest)
        ensure_usable(revalidated, "transform", request.request_id)

        # 7 (precedence). Deadline past before preview: no transform runs, no effect.
        if self._expired(request, now):
            return self._timed_out(request, now)

        # 6. The injected callback is the only boundary to relevance/graph logic.
        try:
            transformed = self._transform_fn(request.neutral_input)
        except DependencyUnavailable as dependency:
            return self._transform_fallback(request, dependency.reason, now)

        result = NeutralResult(
            kind="transformed_content",
            content=transformed,
            content_digest=_digest_of(transformed),
        )
        return success_outcome(request.request_id, "transformed", result, committed_at=now)

    def _transform_fallback(self, request: ProcessingRequest, reason: str, now: float) -> ProcessingOutcome:
        """Preserve an integral baseline as ``fallback_original`` or refuse when none exists."""
        if request.baseline is None:
            # No recoverable baseline: a single deterministic refusal, never a fallback that hides it.
            raise RefusalError.of("DEPENDENCY_UNAVAILABLE_NO_FALLBACK", "transform", request.request_id)
        return fallback_outcome(
            request.request_id,
            reason,
            request.baseline,
            committed_at=now,
            content_digest=_digest_of(request.baseline),
        )

    def _do_store(self, request: ProcessingRequest, now: float) -> ProcessingOutcome:
        """Store content and return a ``stored`` outcome carrying an opaque reference."""
        if request.namespace_id is None or request.content is None:
            raise RefusalError.of("REQUEST_SCHEMA_INVALID", "store", request.request_id)
        reference = self._store.store(
            request.namespace_id,
            request.content,
            content_type=request.content_type,
            correlation_id=request.request_id,
        )
        result = NeutralResult(
            kind="stored_reference",
            reference_id=reference.reference_id,
            content_digest=reference.content_digest,
        )
        return success_outcome(request.request_id, "stored", result, committed_at=now)

    def _do_delete(self, request: ProcessingRequest, now: float) -> ProcessingOutcome:
        """Delete the named reference and return a ``deleted`` acknowledgement."""
        if request.namespace_id is None or request.reference_id is None:
            raise RefusalError.of("REQUEST_SCHEMA_INVALID", "delete", request.request_id)
        self._store.delete(request.namespace_id, request.reference_id, correlation_id=request.request_id)
        result = NeutralResult(kind="deletion_ack", reference_id=request.reference_id)
        return success_outcome(request.request_id, "deleted", result, committed_at=now)

    def _do_delegate(self, request: ProcessingRequest, now: float) -> ProcessingOutcome:
        """Create a named delegation grant and return a ``delegated`` acknowledgement."""
        if (
            request.namespace_id is None
            or request.recipient_namespace_id is None
            or request.grant_expires_at is None
        ):
            raise RefusalError.of("REQUEST_SCHEMA_INVALID", "delegate", request.request_id)
        grant = self._store.delegate(
            request.namespace_id,
            request.recipient_namespace_id,
            request.reference_ids,
            expires_at=request.grant_expires_at,
            correlation_id=request.request_id,
        )
        result = NeutralResult(
            kind="delegation_ack",
            metadata={"grant_id": grant.grant_id, "recipient_namespace_id": grant.recipient_namespace_id},
        )
        return success_outcome(request.request_id, "delegated", result, committed_at=now)

    def _do_shutdown(self, request: ProcessingRequest, now: float) -> ProcessingOutcome:
        """Discard the namespace synchronously, then return a ``deleted`` acknowledgement.

        The store zeroes every reference the namespace owns, revokes its grants and marks its lease
        discarded before this returns, so the terminal is produced only after the content is gone.
        """
        if request.namespace_id is None:
            raise RefusalError.of("REQUEST_SCHEMA_INVALID", "shutdown", request.request_id)
        discarded = self._store.shutdown(request.namespace_id, correlation_id=request.request_id)
        result = NeutralResult(kind="deletion_ack", metadata={"references_discarded": discarded})
        return success_outcome(request.request_id, "deleted", result, committed_at=now)

    # ------------------------------------------------------------------ #
    # Read-only / liveness path                                          #
    # ------------------------------------------------------------------ #

    def _run_readonly(self, request: ProcessingRequest, now: float) -> ProcessingOutcome:
        """Run a read-only or liveness operation (no reservation, no durable effect)."""
        operation = request.operation
        if operation == "health":
            # Pure liveness: repeatable within the deadline with no contract gate (BR5.2).
            return success_outcome(
                request.request_id, "healthy", NeutralResult(kind="health_ack"), committed_at=now
            )
        # startup / preflight / retrieve / catalog all validate contract identity first.
        self._require_contract(request)

        if operation == "startup":
            return self._do_startup(request, now)
        if operation == "preflight":
            return success_outcome(
                request.request_id, "preflight_passed", NeutralResult(kind="preflight_ack"), committed_at=now
            )
        if operation == "retrieve":
            return self._do_retrieve(request, now)
        # catalog
        return self._do_catalog(request, now)

    def _do_startup(self, request: ProcessingRequest, now: float) -> ProcessingOutcome:
        """Acknowledge startup, publishing the manifest's capability states without resolving credentials."""
        capabilities: list[dict[str, str]] = []
        host = None
        host_version = None
        if self._manifest is not None:
            host = self._manifest.host
            host_version = self._manifest.host_version
            capabilities = [
                {"practice": declaration.practice, "capability_state": declaration.capability_state}
                for declaration in self._manifest.declarations
            ]
        result = NeutralResult(
            kind="startup_ack",
            metadata={"host": host, "host_version": host_version, "capabilities": capabilities},
        )
        return success_outcome(request.request_id, "started", result, committed_at=now)

    def _do_retrieve(self, request: ProcessingRequest, now: float) -> ProcessingOutcome:
        """Retrieve authorized content and return a ``retrieved`` outcome."""
        if request.namespace_id is None or request.reference_id is None:
            raise RefusalError.of("REQUEST_SCHEMA_INVALID", "retrieve", request.request_id)
        content, content_type = self._store.retrieve(
            request.namespace_id, request.reference_id, correlation_id=request.request_id
        )
        result = NeutralResult(
            kind="retrieved_content",
            content=content,
            reference_id=request.reference_id,
            metadata={"content_type": content_type},
        )
        return success_outcome(request.request_id, "retrieved", result, committed_at=now)

    def _do_catalog(self, request: ProcessingRequest, now: float) -> ProcessingOutcome:
        """Return a redacted snapshot of the namespace's live references."""
        if request.namespace_id is None:
            raise RefusalError.of("REQUEST_SCHEMA_INVALID", "catalog", request.request_id)
        references = self._store.catalog(request.namespace_id, correlation_id=request.request_id)
        snapshot = [
            {
                "reference_id": reference.reference_id,
                "lifecycle_state": reference.lifecycle_state,
                "content_digest": reference.content_digest,
            }
            for reference in references
        ]
        result = NeutralResult(kind="catalog_snapshot", metadata={"references": snapshot})
        return success_outcome(request.request_id, "cataloged", result, committed_at=now)

    # ------------------------------------------------------------------ #
    # Validation and helpers                                             #
    # ------------------------------------------------------------------ #

    def _validate_schema(self, request: ProcessingRequest) -> None:
        """Reject an unknown operation, a family mismatch or a missing required field.

        Raises:
            RefusalError: ``REQUEST_SCHEMA_INVALID`` for any schema violation (BR7.1, BR1.3).
        """
        operation = request.operation
        corr = request.request_id

        if operation not in OPERATIONS:
            raise RefusalError.of("REQUEST_SCHEMA_INVALID", "startup", corr)

        expected_family = "admission" if operation in ADMISSION_OPERATIONS else "runtime"
        if request.operation_family != expected_family:
            raise RefusalError.of("REQUEST_SCHEMA_INVALID", operation, corr)
        # Belt and braces: the two families together are exactly OPERATIONS.
        if operation not in (ADMISSION_OPERATIONS | RUNTIME_OPERATIONS):  # pragma: no cover - defensive
            raise RefusalError.of("REQUEST_SCHEMA_INVALID", operation, corr)

        if operation in EFFECT_OPERATIONS and not (
            request.idempotency_scope and request.idempotency_key and request.request_digest
        ):
            raise RefusalError.of("REQUEST_SCHEMA_INVALID", operation, corr)

        if operation in _REFERENCE_OPERATIONS and request.namespace_id is None:
            raise RefusalError.of("REQUEST_SCHEMA_INVALID", operation, corr)

        if operation == "transform" and (request.capability_decision is None or request.practice is None):
            raise RefusalError.of("REQUEST_SCHEMA_INVALID", operation, corr)

        if operation == "store" and request.content is None:
            raise RefusalError.of("REQUEST_SCHEMA_INVALID", operation, corr)

        if operation in ("retrieve", "delete") and request.reference_id is None:
            raise RefusalError.of("REQUEST_SCHEMA_INVALID", operation, corr)

        if operation == "delegate" and (
            request.recipient_namespace_id is None
            or not request.reference_ids
            or request.grant_expires_at is None
        ):
            raise RefusalError.of("REQUEST_SCHEMA_INVALID", operation, corr)

    def _require_contract(self, request: ProcessingRequest) -> None:
        """Raise ``CONTRACT_MISMATCH`` unless the request speaks the runtime's identity (BR1.3)."""
        capability = request.practice if request.practice is not None else "none"
        require_compatible(
            request.contract_identity,
            self._contract_identity,
            request.operation,
            request.request_id,
            capability=capability,
        )

    def _expired(self, request: ProcessingRequest, now: float) -> bool:
        """Report whether the request's deadline has passed at ``now``."""
        return now >= request.deadline_at

    @staticmethod
    def _timed_out(request: ProcessingRequest, now: float) -> ProcessingOutcome:
        """Build a payload-free ``timed_out`` terminal (BR5.5)."""
        return ProcessingOutcome(
            request_id=request.request_id,
            status="timed_out",
            committed_at=now,
            result=NeutralResult(kind="timeout_without_payload"),
        )


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
