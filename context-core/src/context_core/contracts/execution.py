"""``ExecutionRequest`` / ``ExecutionAdmission`` — the closed U1↔U2 admission boundary for offline/replay/live.

Admission is configuration, never payload transport: an :class:`ExecutionRequest` carries digests, mode,
budget, approval and preflight, and its ``payload`` field must be the literal ``"absent"``. No credential,
``boto3`` client or network call is ever resolved here — admission is pure local policy (BR6.1–BR6.5):

- ``offline`` accepts only the ``local-fake`` / ``local-scorer`` backends; approval and budget may be absent.
- ``replay`` accepts only a sanitized (non-remote) backend with matching fixture/catalog digests.
- ``live`` requires explicit approval by a permitted actor, a configured budget, an allow-listed
  remote backend and model, a requested preflight, and a single-use nonce.

:func:`admit` always returns an :class:`ExecutionAdmission` — a refusal becomes a ``refused`` admission
carrying one closed envelope, which U2 cannot override. No framework import, no I/O; the clock is injected.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Literal

from .errors import ClosedRefusalEnvelope, RefusalError
from .identity import ContractIdentity

__all__ = [
    "BACKEND_IDS",
    "EXECUTION_MODES",
    "AdmissionPolicy",
    "Approval",
    "ExecutionAdmission",
    "ExecutionBudget",
    "ExecutionPreflight",
    "ExecutionRequest",
    "PreflightResult",
    "admit",
]

ExecutionMode = Literal["offline", "live", "replay"]
BackendId = Literal["local-fake", "local-scorer", "remote-explicit"]

EXECUTION_MODES: frozenset[str] = frozenset({"offline", "live", "replay"})
BACKEND_IDS: frozenset[str] = frozenset({"local-fake", "local-scorer", "remote-explicit"})
_LOCAL_BACKENDS: frozenset[str] = frozenset({"local-fake", "local-scorer"})


@dataclass(frozen=True)
class Approval:
    """Approval and actor bound to an execution request.

    Attributes:
        status: ``"absent"`` or ``"explicit"``.
        approval_id: Opaque approval id; required for ``explicit``.
        actor_class: ``"maintainer"`` or ``"automation_with_recorded_approval"``.
    """

    status: Literal["absent", "explicit"] = "absent"
    approval_id: str | None = None
    actor_class: Literal["maintainer", "automation_with_recorded_approval"] | None = None


@dataclass(frozen=True)
class ExecutionBudget:
    """A declarative execution budget; no numeric value is fixed in this unit.

    Attributes:
        status: ``"absent"`` or ``"configured"``.
        ceiling_units: The ceiling, when configured. Deferred numeric value.
        currency_or_unit: The unit the ceiling is expressed in.
    """

    status: Literal["absent", "configured"] = "absent"
    ceiling_units: float | None = None
    currency_or_unit: str | None = None


@dataclass(frozen=True)
class ExecutionPreflight:
    """Preflight intent and single-use nonce before admission.

    Before an admission is ``admitted``, effective credential resolution and payload transmission are
    both false.

    Attributes:
        requested: Whether preflight was requested.
        credential_resolution_allowed: Caller intent; never effective before admission.
        payload_transmission_allowed: Caller intent; never effective before admission.
        nonce: Single-use nonce for this request.
    """

    nonce: str
    requested: bool = False
    credential_resolution_allowed: bool = False
    payload_transmission_allowed: bool = False


@dataclass(frozen=True)
class PreflightResult:
    """The closed result of admission preflight.

    Attributes:
        credential_resolution: ``not_attempted`` / ``permitted_after_admission`` / ``refused``.
        payload_transmission: ``not_attempted`` / ``permitted_after_admission`` / ``refused``.
    """

    credential_resolution: Literal["not_attempted", "permitted_after_admission", "refused"] = "not_attempted"
    payload_transmission: Literal["not_attempted", "permitted_after_admission", "refused"] = "not_attempted"


@dataclass(frozen=True)
class ExecutionRequest:
    """A request for execution admission; carries configuration and digests, never tool payload.

    Attributes:
        request_id: Opaque correlation id.
        idempotency_key: Replay-safe idempotency key.
        idempotency_scope: Idempotency scope.
        request_digest: Canonical digest of the request.
        contract_identity: The contract identity the request speaks.
        execution_mode: ``offline`` / ``live`` / ``replay``.
        host: Host identifier.
        host_version: Host version string.
        model_id: Model identifier.
        backend_id: ``local-fake`` / ``local-scorer`` / ``remote-explicit``.
        capability_manifest_digest: Digest of the capability manifest.
        fixture_catalog_digest: Digest of the fixture catalog.
        approval: Approval and actor.
        budget: Execution budget.
        preflight: Preflight intent and nonce.
        deadline_ms: Positive deadline in milliseconds.
        payload: Must be the literal ``"absent"`` — payload transport is forbidden here.
    """

    request_id: str
    idempotency_key: str
    idempotency_scope: str
    request_digest: str
    contract_identity: ContractIdentity
    execution_mode: ExecutionMode
    host: str
    host_version: str
    model_id: str
    backend_id: BackendId
    capability_manifest_digest: str
    fixture_catalog_digest: str
    approval: Approval
    budget: ExecutionBudget
    preflight: ExecutionPreflight
    deadline_ms: int
    payload: Literal["absent"] = "absent"


@dataclass(frozen=True)
class ExecutionAdmission:
    """The terminal U1 decision over an :class:`ExecutionRequest`.

    Attributes:
        admission_id: Opaque identity.
        request_id: The request this admission answers.
        admission_state: ``received`` / ``validating`` / ``admitted`` / ``refused`` / ``expired``.
        status: Terminal projection ``admitted`` / ``refused``.
        policy_digest: Digest of the policy applied.
        capability_manifest_digest: Bound manifest digest.
        fixture_catalog_digest: Bound fixture catalog digest.
        contract_identity: Bound contract identity.
        preflight_result: The closed preflight result.
        approval: The approval that was in play.
        budget: The budget that was in play.
        nonce: The single-use nonce bound to this admission.
        issued_at: Issue time (injected clock value).
        refusal: The closed envelope, present only when ``status`` is ``refused``.
    """

    admission_id: str
    request_id: str
    admission_state: Literal["received", "validating", "admitted", "refused", "expired"]
    status: Literal["admitted", "refused"]
    policy_digest: str
    capability_manifest_digest: str
    fixture_catalog_digest: str
    contract_identity: ContractIdentity
    preflight_result: PreflightResult
    approval: Approval
    budget: ExecutionBudget
    nonce: str
    issued_at: float
    refusal: ClosedRefusalEnvelope | None = None


@dataclass
class AdmissionPolicy:
    """The local policy :func:`admit` enforces. No credential, no network.

    Attributes:
        policy_digest: Digest of this policy, recorded on the admission.
        expected_contract_identity: The contract identity U1 currently holds.
        expected_capability_manifest_digest: The manifest digest U1 currently holds.
        expected_fixture_catalog_digest: The fixture catalog digest U1 currently holds.
        allowlisted_backends: Remote backends permitted for ``live``.
        allowlisted_models: Models permitted for ``live``.
        approved_actor_classes: Actor classes permitted to approve ``live``.
        used_nonces: Mutable set of nonces already consumed; a reused nonce is refused. Reads and
            writes go through :meth:`consume_nonce`, serialised by an internal lock so two concurrent
            ``live`` admissions can never consume the same nonce.
    """

    policy_digest: str
    expected_contract_identity: ContractIdentity
    expected_capability_manifest_digest: str
    expected_fixture_catalog_digest: str
    allowlisted_backends: frozenset[str] = frozenset({"remote-explicit"})
    allowlisted_models: frozenset[str] = frozenset()
    approved_actor_classes: frozenset[str] = frozenset(
        {"maintainer", "automation_with_recorded_approval"}
    )
    used_nonces: set[str] = field(default_factory=set)
    _nonce_lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def consume_nonce(self, nonce: str) -> bool:
        """Atomically consume ``nonce`` for a single ``live`` admission.

        Checks membership and records the nonce under one lock, so the check-then-act is indivisible:
        of two concurrent admissions presenting the same nonce, exactly one succeeds.

        Args:
            nonce: The single-use nonce to consume.

        Returns:
            ``True`` if the nonce was free and is now consumed; ``False`` if it was already used.
        """
        with self._nonce_lock:
            if nonce in self.used_nonces:
                return False
            self.used_nonces.add(nonce)
            return True


def admit(request: ExecutionRequest, policy: AdmissionPolicy, *, issued_at: float) -> ExecutionAdmission:
    """Admit or refuse ``request`` under ``policy``, returning a terminal admission.

    Always returns an :class:`ExecutionAdmission`: a failed precondition becomes a ``refused`` admission
    carrying exactly one closed envelope. No credential is resolved and no payload is transmitted on any
    path. A ``live`` admission consumes its nonce (recorded in ``policy.used_nonces``).

    Args:
        request: The execution request.
        policy: The local admission policy to enforce.
        issued_at: Issue time for the admission (injected clock value).

    Returns:
        The terminal admission.
    """
    import secrets

    try:
        _validate(request, policy)
        # Consume the single-use nonce atomically for live mode, so two concurrent live admissions
        # presenting the same nonce can never both be admitted.
        if request.execution_mode == "live" and not policy.consume_nonce(request.preflight.nonce):
            raise RefusalError.of("ADMISSION_POLICY_DENIED", "preflight", request.request_id)
    except RefusalError as error:
        return ExecutionAdmission(
            admission_id=secrets.token_hex(16),
            request_id=request.request_id,
            admission_state="refused",
            status="refused",
            policy_digest=policy.policy_digest,
            capability_manifest_digest=request.capability_manifest_digest,
            fixture_catalog_digest=request.fixture_catalog_digest,
            contract_identity=request.contract_identity,
            preflight_result=PreflightResult(),
            approval=request.approval,
            budget=request.budget,
            nonce=request.preflight.nonce,
            issued_at=issued_at,
            refusal=error.envelope,
        )

    if request.execution_mode == "live":
        # The nonce was consumed atomically above; nothing further to record here.
        preflight_result = PreflightResult(
            credential_resolution="permitted_after_admission",
            payload_transmission="permitted_after_admission",
        )
    else:
        preflight_result = PreflightResult()

    return ExecutionAdmission(
        admission_id=secrets.token_hex(16),
        request_id=request.request_id,
        admission_state="admitted",
        status="admitted",
        policy_digest=policy.policy_digest,
        capability_manifest_digest=request.capability_manifest_digest,
        fixture_catalog_digest=request.fixture_catalog_digest,
        contract_identity=request.contract_identity,
        preflight_result=preflight_result,
        approval=request.approval,
        budget=request.budget,
        nonce=request.preflight.nonce,
        issued_at=issued_at,
    )


def _validate(request: ExecutionRequest, policy: AdmissionPolicy) -> None:
    """Run the closed admission checks, raising :class:`RefusalError` on the first failure.

    Args:
        request: The execution request.
        policy: The admission policy.

    Raises:
        RefusalError: ``REQUEST_SCHEMA_INVALID``, ``CONTRACT_MISMATCH`` or ``ADMISSION_POLICY_DENIED``
            depending on which precondition fails.
    """
    operation = "preflight"
    corr = request.request_id

    # Schema: payload forbidden, enums closed, deadline positive.
    if request.payload != "absent":
        raise RefusalError.of("REQUEST_SCHEMA_INVALID", operation, corr)
    if request.execution_mode not in EXECUTION_MODES or request.backend_id not in BACKEND_IDS:
        raise RefusalError.of("REQUEST_SCHEMA_INVALID", operation, corr)
    if request.deadline_ms < 1:
        raise RefusalError.of("REQUEST_SCHEMA_INVALID", operation, corr)

    # Contract and digests must bind to what U1 holds.
    if not request.contract_identity.is_compatible_with(policy.expected_contract_identity):
        raise RefusalError.of("CONTRACT_MISMATCH", operation, corr)
    if request.capability_manifest_digest != policy.expected_capability_manifest_digest:
        raise RefusalError.of("CONTRACT_MISMATCH", operation, corr)
    if request.fixture_catalog_digest != policy.expected_fixture_catalog_digest:
        raise RefusalError.of("CONTRACT_MISMATCH", operation, corr)

    if request.execution_mode == "offline":
        if request.backend_id not in _LOCAL_BACKENDS:
            raise RefusalError.of("ADMISSION_POLICY_DENIED", operation, corr)
        return

    if request.execution_mode == "replay":
        # Sanitized replay: a local backend only; a remote backend would resolve credentials.
        if request.backend_id not in _LOCAL_BACKENDS:
            raise RefusalError.of("ADMISSION_POLICY_DENIED", operation, corr)
        return

    # live
    if request.approval.status != "explicit" or not request.approval.approval_id:
        raise RefusalError.of("ADMISSION_POLICY_DENIED", operation, corr)
    if request.approval.actor_class not in policy.approved_actor_classes:
        raise RefusalError.of("ADMISSION_POLICY_DENIED", operation, corr)
    if request.budget.status != "configured":
        raise RefusalError.of("ADMISSION_POLICY_DENIED", operation, corr)
    if request.backend_id != "remote-explicit" or request.backend_id not in policy.allowlisted_backends:
        raise RefusalError.of("ADMISSION_POLICY_DENIED", operation, corr)
    if request.model_id not in policy.allowlisted_models:
        raise RefusalError.of("ADMISSION_POLICY_DENIED", operation, corr)
    if not request.preflight.requested:
        raise RefusalError.of("ADMISSION_POLICY_DENIED", operation, corr)
    # The single-use nonce is checked and consumed atomically in ``admit`` via
    # ``policy.consume_nonce`` — not here — so the check-then-act cannot race.
