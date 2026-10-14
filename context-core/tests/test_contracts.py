"""Tests for the U1 ``context_core.contracts`` layer.

Covers happy paths plus the required failure cases: invalid schema, digest mismatch, closed refusal
shape, idempotency conflict, namespace denial, lease expiry, grant delegation and a denied live
admission. Everything runs offline with an injected clock; no network, AWS or framework import.
"""

from __future__ import annotations

import itertools

import pytest

from context_core.contracts import (
    AdmissionPolicy,
    Approval,
    AuthorizationContext,
    CapabilityDeclaration,
    CapabilityManifest,
    ClosedRefusalEnvelope,
    ContractIdentity,
    CoverageSet,
    EvidenceDigest,
    ExecutionBudget,
    ExecutionPreflight,
    ExecutionRequest,
    IdempotencyStore,
    InMemoryContextStore,
    NeutralResult,
    ProcessingOutcome,
    RefusalError,
    ReservationState,
    admit,
    canonical_json,
    digest,
    ensure_usable,
    issue_decision,
    make_refusal,
    refused_outcome,
    revalidate,
    success_outcome,
)
from context_core.contracts.errors import _ENVELOPE_FIELDS, CATALOG


class _Clock:
    """A deterministic monotonic clock the tests advance by hand."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


# --------------------------------------------------------------------------- #
# Canonical JSON / digest                                                     #
# --------------------------------------------------------------------------- #


def test_canonical_json_is_key_order_independent():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})


def test_canonical_json_keeps_unicode_literal():
    assert canonical_json({"k": "café"}) == '{"k":"café"}'


# --------------------------------------------------------------------------- #
# ContractIdentity                                                            #
# --------------------------------------------------------------------------- #


def test_contract_identity_compatible_same_major_and_digest():
    schema = {"fields": ["a", "b"]}
    one = ContractIdentity.of_schema("1.0", schema)
    two = ContractIdentity.of_schema("1.0", schema)
    assert one.is_compatible_with(two)


def test_contract_identity_incompatible_on_major_or_digest():
    base = ContractIdentity.of_schema("1.0", {"x": 1})
    other_major = ContractIdentity.of_schema("2.0", {"x": 1})
    other_digest = ContractIdentity.of_schema("1.0", {"x": 2})
    assert not base.is_compatible_with(other_major)
    assert not base.is_compatible_with(other_digest)


def test_contract_identity_rejects_bad_version_or_digest():
    with pytest.raises(ValueError):
        ContractIdentity("1", "a" * 64)
    with pytest.raises(ValueError):
        ContractIdentity("1.0", "nothex")


# --------------------------------------------------------------------------- #
# Closed refusal envelope                                                     #
# --------------------------------------------------------------------------- #


def test_refusal_envelope_has_exactly_the_closed_fields():
    envelope = make_refusal("CONTRACT_MISMATCH", "transform", "corr-1", capability="A")
    assert set(envelope.to_dict()) == _ENVELOPE_FIELDS
    assert envelope.status == "refused"
    assert envelope.category == "invalid_request"
    assert envelope.retryable is False


def test_refusal_envelope_category_comes_from_catalog():
    for code, entry in CATALOG.items():
        envelope = make_refusal(code, "store", "c")
        assert envelope.category == entry.category
        assert envelope.retryable == entry.retryable


def test_refusal_envelope_rejects_unknown_code_and_operation():
    with pytest.raises(ValueError):
        make_refusal("NOPE", "store", "c")
    with pytest.raises(ValueError):
        make_refusal("QUOTA_EXCEEDED", "fly", "c")


def test_refusal_envelope_carries_no_payload_fields():
    envelope = make_refusal("AUTHORIZATION_HOST_DENIED", "retrieve", "c")
    forbidden = {"payload", "content", "reference_id", "namespace_id", "size", "hash", "message"}
    assert forbidden.isdisjoint(envelope.to_dict())


# --------------------------------------------------------------------------- #
# Capability: state vs lifecycle, revalidation                               #
# --------------------------------------------------------------------------- #


def _manifest(capability_state: str = "full", *, revision: int = 1) -> CapabilityManifest:
    coverage = CoverageSet(
        coverage_set_id="cov-A",
        tool_classes=("search",),
        operation_set=("transform",),
    )
    evidence = EvidenceDigest(evidence_digest_id="ev-A", artifact_digests=("d1",), compatibility_matrix_version="1.0")
    declaration = CapabilityDeclaration(
        declaration_id="decl-A",
        practice="A",
        capability_state=capability_state,  # type: ignore[arg-type]
        implementation_kind="adapter",
        coverage=coverage,
        evidence=evidence,
    )
    return CapabilityManifest(
        manifest_id="man-1",
        host="opencode",
        host_version="1.2.3",
        contract_identity=ContractIdentity.of_schema("1.0", {"schema": "u1"}),
        revision=revision,
        declarations=(declaration,),
    )


def test_decision_starts_issued_and_revalidates_to_valid():
    manifest = _manifest()
    decision = issue_decision(manifest, "A", decision_id="dec-1", issued_at=0.0)
    assert decision.decision_lifecycle == "issued"
    revalidated = revalidate(decision, manifest)
    assert revalidated.decision_lifecycle == "valid"
    assert revalidated.capability_state == "full"


def test_decision_invalidated_when_manifest_revision_changes():
    manifest = _manifest(revision=1)
    decision = revalidate(issue_decision(manifest, "A", decision_id="d", issued_at=0.0), manifest)
    assert decision.decision_lifecycle == "valid"
    bumped = _manifest(revision=2)
    stale = revalidate(decision, bumped)
    assert stale.decision_lifecycle == "invalidated"


def test_state_and_lifecycle_are_independent():
    # An unsupported capability can still produce a valid snapshot: state != lifecycle.
    manifest = _manifest(capability_state="unsupported")
    decision = revalidate(issue_decision(manifest, "A", decision_id="d", issued_at=0.0), manifest)
    assert decision.decision_lifecycle == "valid"
    assert decision.capability_state == "unsupported"


def test_ensure_usable_refuses_unsupported_and_invalid():
    manifest = _manifest(capability_state="unsupported")
    decision = revalidate(issue_decision(manifest, "A", decision_id="d", issued_at=0.0), manifest)
    with pytest.raises(RefusalError) as caught:
        ensure_usable(decision, "transform", "corr")
    assert caught.value.envelope.code == "CAPABILITY_UNSUPPORTED"

    issued_only = issue_decision(_manifest(), "A", decision_id="d2", issued_at=0.0)
    with pytest.raises(RefusalError) as caught2:
        ensure_usable(issued_only, "transform", "corr")
    assert caught2.value.envelope.code == "CONTRACT_MISMATCH"


def test_declaration_rejects_empty_coverage_when_required():
    with pytest.raises(ValueError):
        CapabilityDeclaration(
            declaration_id="d",
            practice="A",
            capability_state="partial",
            implementation_kind="adapter",
            coverage=CoverageSet(coverage_set_id="empty"),
            evidence=EvidenceDigest(evidence_digest_id="e"),
        )


# --------------------------------------------------------------------------- #
# NeutralResult / ProcessingOutcome invariants                               #
# --------------------------------------------------------------------------- #


def test_success_outcome_carries_result_and_no_refusal():
    outcome = success_outcome(
        "req-1", "transformed", NeutralResult(kind="transformed_content", content="x"), committed_at=1.0
    )
    assert outcome.status == "transformed"
    assert outcome.refusal is None
    assert outcome.result is not None


def test_refused_outcome_rejects_a_result():
    envelope = make_refusal("REQUEST_SCHEMA_INVALID", "transform", "c")
    with pytest.raises(ValueError):
        ProcessingOutcome(
            request_id="r",
            status="refused",
            committed_at=1.0,
            refusal=envelope,
            result=NeutralResult(kind="transformed_content"),
        )


def test_success_status_rejects_a_refusal():
    envelope = make_refusal("REQUEST_SCHEMA_INVALID", "transform", "c")
    with pytest.raises(ValueError):
        ProcessingOutcome(request_id="r", status="transformed", committed_at=1.0, refusal=envelope)


def test_refused_outcome_helper_builds_valid_terminal():
    envelope = make_refusal("CONTRACT_MISMATCH", "transform", "c")
    outcome = refused_outcome("r", envelope, committed_at=2.0)
    assert outcome.status == "refused"
    assert outcome.result is None


# --------------------------------------------------------------------------- #
# Idempotency                                                                 #
# --------------------------------------------------------------------------- #


def test_idempotency_reserve_then_commit_then_replay():
    store = IdempotencyStore(retention_seconds=100)
    reservation = store.reserve("scope", "k", "digest-1", "req-1", "store", now=0.0)
    assert reservation.state is ReservationState.RESERVED

    outcome = success_outcome("req-1", "stored", NeutralResult(kind="stored_reference"), committed_at=1.0)
    store.commit("scope", "k", "digest-1", outcome, now=1.0)

    replay = store.reserve("scope", "k", "digest-1", "req-1-retry", "store", now=2.0)
    assert replay.state is ReservationState.REPLAY
    assert replay.outcome is outcome


def test_idempotency_conflict_on_different_digest():
    store = IdempotencyStore(retention_seconds=100)
    store.reserve("scope", "k", "digest-1", "req-1", "store", now=0.0)
    with pytest.raises(RefusalError) as caught:
        store.reserve("scope", "k", "digest-2", "req-2", "store", now=1.0)
    assert caught.value.envelope.code == "IDEMPOTENCY_CONFLICT"


def test_idempotency_pending_while_in_flight():
    store = IdempotencyStore(retention_seconds=100)
    store.reserve("scope", "k", "digest-1", "req-1", "store", now=0.0)
    again = store.reserve("scope", "k", "digest-1", "req-1", "store", now=0.0)
    assert again.state is ReservationState.PENDING


def test_idempotency_timeout_cancels_without_second_effect():
    store = IdempotencyStore(retention_seconds=100)
    store.reserve("scope", "k", "digest-1", "req-1", "store", now=0.0)
    store.cancel("scope", "k")
    # The key is free again, so a fresh request may reserve it.
    reservation = store.reserve("scope", "k", "digest-9", "req-9", "store", now=1.0)
    assert reservation.state is ReservationState.RESERVED


def test_commit_is_idempotent_and_rejects_wrong_digest():
    store = IdempotencyStore(retention_seconds=100)
    store.reserve("scope", "k", "digest-1", "req-1", "store", now=0.0)
    outcome = success_outcome("req-1", "stored", NeutralResult(kind="stored_reference"), committed_at=1.0)
    first = store.commit("scope", "k", "digest-1", outcome, now=1.0)
    second = store.commit("scope", "k", "digest-1", outcome, now=2.0)
    assert first is second
    with pytest.raises(ValueError):
        store.commit("scope", "k", "wrong", outcome, now=3.0)


# --------------------------------------------------------------------------- #
# References, namespaces, leases, delegation                                 #
# --------------------------------------------------------------------------- #


def _open(store: InMemoryContextStore, **overrides):
    params = {
        "lease_ttl": 300.0,
        "heartbeat_interval": 30.0,
        "abandonment_grace": 120.0,
        "memory_quota": 1024,
        "file_quota": 0,
    }
    params.update(overrides)
    return store.open_namespace("ws-1", "sess-1", **params)


def test_store_and_retrieve_round_trip():
    store = InMemoryContextStore(clock=_Clock())
    namespace = _open(store)
    reference = store.store(namespace.namespace_id, b"hello", content_type="text/plain")
    assert reference.lifecycle_state == "active"
    content, content_type = store.retrieve(namespace.namespace_id, reference.reference_id)
    assert content == b"hello"
    assert content_type == "text/plain"


def test_unknown_namespace_is_denied():
    store = InMemoryContextStore(clock=_Clock())
    with pytest.raises(RefusalError) as caught:
        store.store("ghost", b"x")
    assert caught.value.envelope.code == "NAMESPACE_INVALID"


def test_host_authorization_denial():
    def deny_retrieve(context: AuthorizationContext) -> bool:
        return context.operation != "retrieve"

    store = InMemoryContextStore(authorize=deny_retrieve, clock=_Clock())
    namespace = _open(store)
    reference = store.store(namespace.namespace_id, b"secret")
    with pytest.raises(RefusalError) as caught:
        store.retrieve(namespace.namespace_id, reference.reference_id)
    assert caught.value.envelope.code == "AUTHORIZATION_HOST_DENIED"


def test_quota_exceeded_before_copy():
    store = InMemoryContextStore(clock=_Clock())
    namespace = _open(store, memory_quota=4)
    with pytest.raises(RefusalError) as caught:
        store.store(namespace.namespace_id, b"toolong")
    assert caught.value.envelope.code == "QUOTA_EXCEEDED"


def test_lease_expiry_blocks_store():
    clock = _Clock()
    store = InMemoryContextStore(clock=clock)
    namespace = _open(store, lease_ttl=10.0)
    clock.advance(11.0)
    with pytest.raises(RefusalError) as caught:
        store.store(namespace.namespace_id, b"x")
    assert caught.value.envelope.code == "LEASE_EXPIRED"


def test_heartbeat_renews_lease():
    clock = _Clock()
    store = InMemoryContextStore(clock=clock)
    namespace = _open(store, lease_ttl=10.0, heartbeat_interval=100.0)
    clock.advance(8.0)
    store.heartbeat(namespace.namespace_id)
    clock.advance(5.0)  # 13 total, but renewed at 8 with ttl 10 -> expires at 18
    reference = store.store(namespace.namespace_id, b"ok")
    assert reference.lifecycle_state == "active"


def test_delete_discards_and_blocks_retrieve():
    store = InMemoryContextStore(clock=_Clock())
    namespace = _open(store)
    reference = store.store(namespace.namespace_id, b"gone-soon")
    store.delete(namespace.namespace_id, reference.reference_id)
    with pytest.raises(RefusalError) as caught:
        store.retrieve(namespace.namespace_id, reference.reference_id)
    assert caught.value.envelope.code == "REFERENCE_EXPIRED_OR_DELETED"


def test_subagent_isolated_without_grant():
    store = InMemoryContextStore(clock=_Clock())
    owner = _open(store)
    other = store.open_namespace(
        "ws-1",
        "sess-2",
        lease_ttl=300.0,
        heartbeat_interval=30.0,
        abandonment_grace=120.0,
        memory_quota=1024,
        file_quota=0,
    )
    reference = store.store(owner.namespace_id, b"private")
    with pytest.raises(RefusalError) as caught:
        store.retrieve(other.namespace_id, reference.reference_id)
    assert caught.value.envelope.code == "AUTHORIZATION_HOST_DENIED"


def test_delegation_grants_named_access_and_revocation_closes_it():
    clock = _Clock()
    store = InMemoryContextStore(clock=clock)
    owner = _open(store)
    recipient = store.open_namespace(
        "ws-1",
        "sess-2",
        lease_ttl=300.0,
        heartbeat_interval=30.0,
        abandonment_grace=120.0,
        memory_quota=1024,
        file_quota=0,
    )
    reference = store.store(owner.namespace_id, b"shared")
    grant = store.delegate(
        owner.namespace_id, recipient.namespace_id, (reference.reference_id,), expires_at=1000.0
    )

    content, _ = store.retrieve(recipient.namespace_id, reference.reference_id)
    assert content == b"shared"

    store.revoke_grant(grant.grant_id)
    with pytest.raises(RefusalError) as caught:
        store.retrieve(recipient.namespace_id, reference.reference_id)
    assert caught.value.envelope.code == "AUTHORIZATION_HOST_DENIED"


def test_delegation_rejects_empty_and_foreign_reference():
    store = InMemoryContextStore(clock=_Clock())
    owner = _open(store)
    recipient = store.open_namespace(
        "ws-1",
        "sess-2",
        lease_ttl=300.0,
        heartbeat_interval=30.0,
        abandonment_grace=120.0,
        memory_quota=1024,
        file_quota=0,
    )
    with pytest.raises(RefusalError) as empty:
        store.delegate(owner.namespace_id, recipient.namespace_id, (), expires_at=1000.0)
    assert empty.value.envelope.code == "REQUEST_SCHEMA_INVALID"

    with pytest.raises(RefusalError) as foreign:
        store.delegate(owner.namespace_id, recipient.namespace_id, ("not-a-ref",), expires_at=1000.0)
    assert foreign.value.envelope.code == "REQUEST_SCHEMA_INVALID"


# --------------------------------------------------------------------------- #
# C1 — synchronous namespace shutdown                                         #
# --------------------------------------------------------------------------- #


def test_shutdown_discards_own_references_and_zeros_content():
    store = InMemoryContextStore(clock=_Clock())
    namespace = _open(store)
    first = store.store(namespace.namespace_id, b"one")
    second = store.store(namespace.namespace_id, b"two")

    discarded = store.shutdown(namespace.namespace_id)
    assert discarded == 2
    # Content is zeroed and the lifecycle closed before the ack (BR4.7).
    assert store._references[first.reference_id].content == b""
    assert store._references[first.reference_id].lifecycle_state == "deleted"
    assert store._references[second.reference_id].lifecycle_state == "deleted"
    # Memory charge cleared and the lease marked discarded.
    assert store._memory_used[namespace.namespace_id] == 0
    assert store._leases[namespace.namespace_id].lease_state == "discarded"


def test_shutdown_blocks_retrieve_and_lease_renewal():
    store = InMemoryContextStore(clock=_Clock())
    namespace = _open(store)
    reference = store.store(namespace.namespace_id, b"gone")
    store.shutdown(namespace.namespace_id)

    with pytest.raises(RefusalError) as retrieve_err:
        store.retrieve(namespace.namespace_id, reference.reference_id)
    # The acting lease is discarded, so the lease gate refuses before the reference is located.
    assert retrieve_err.value.envelope.code == "LEASE_EXPIRED"

    with pytest.raises(RefusalError) as heartbeat_err:
        store.heartbeat(namespace.namespace_id)
    assert heartbeat_err.value.envelope.code == "LEASE_EXPIRED"


def test_shutdown_unknown_namespace_is_denied():
    store = InMemoryContextStore(clock=_Clock())
    with pytest.raises(RefusalError) as caught:
        store.shutdown("ghost")
    assert caught.value.envelope.code == "NAMESPACE_INVALID"


def test_shutdown_host_denied_touches_nothing():
    def deny_shutdown(context: AuthorizationContext) -> bool:
        return context.operation != "shutdown"

    store = InMemoryContextStore(authorize=deny_shutdown, clock=_Clock())
    namespace = _open(store)
    reference = store.store(namespace.namespace_id, b"intact")
    with pytest.raises(RefusalError) as caught:
        store.shutdown(namespace.namespace_id)
    assert caught.value.envelope.code == "AUTHORIZATION_HOST_DENIED"
    # No content was discarded on refusal.
    assert store._references[reference.reference_id].content == b"intact"
    assert store._references[reference.reference_id].lifecycle_state == "active"
    assert store._leases[namespace.namespace_id].lease_state == "active"


def test_shutdown_revokes_issued_grant_and_closes_recipient_access():
    store = InMemoryContextStore(clock=_Clock())
    owner = _open(store)
    recipient = store.open_namespace(
        "ws-1", "sess-2", lease_ttl=300.0, heartbeat_interval=30.0,
        abandonment_grace=120.0, memory_quota=1024, file_quota=0,
    )
    reference = store.store(owner.namespace_id, b"shared")
    grant = store.delegate(
        owner.namespace_id, recipient.namespace_id, (reference.reference_id,), expires_at=1_000_000.0
    )
    # The recipient can read it before the owner is discarded.
    content, _ = store.retrieve(recipient.namespace_id, reference.reference_id)
    assert content == b"shared"

    store.shutdown(owner.namespace_id)

    assert store._grants[grant.grant_id].status == "revoked"
    with pytest.raises(RefusalError) as caught:
        store.retrieve(recipient.namespace_id, reference.reference_id)
    # The reference is discarded, so the recipient is refused on the closed lifecycle.
    assert caught.value.envelope.code == "REFERENCE_EXPIRED_OR_DELETED"


def test_shutdown_is_safe_on_already_deleted_references():
    # A reference already deleted is not counted or touched again by shutdown.
    store = InMemoryContextStore(clock=_Clock())
    namespace = _open(store)
    kept = store.store(namespace.namespace_id, b"kept")
    doomed = store.store(namespace.namespace_id, b"drop")
    store.delete(namespace.namespace_id, doomed.reference_id)

    discarded = store.shutdown(namespace.namespace_id)
    assert discarded == 1  # only the still-live reference is discarded
    assert store._references[kept.reference_id].lifecycle_state == "deleted"
    assert store._references[doomed.reference_id].lifecycle_state == "deleted"


# --------------------------------------------------------------------------- #
# Execution admission                                                         #
# --------------------------------------------------------------------------- #


def _policy(**overrides) -> AdmissionPolicy:
    params = {
        "policy_digest": "p" * 0 + digest({"policy": 1}),
        "expected_contract_identity": ContractIdentity.of_schema("1.0", {"schema": "u1"}),
        "expected_capability_manifest_digest": "m" * 64,
        "expected_fixture_catalog_digest": "f" * 64,
        "allowlisted_models": frozenset({"model-ok"}),
    }
    params.update(overrides)
    return AdmissionPolicy(**params)  # type: ignore[arg-type]


def _request(policy: AdmissionPolicy, **overrides) -> ExecutionRequest:
    nonce = next(_NONCES)
    params = {
        "request_id": "req-1",
        "idempotency_key": "k",
        "idempotency_scope": "s",
        "request_digest": "d" * 64,
        "contract_identity": policy.expected_contract_identity,
        "execution_mode": "offline",
        "host": "opencode",
        "host_version": "1.0",
        "model_id": "model-ok",
        "backend_id": "local-fake",
        "capability_manifest_digest": policy.expected_capability_manifest_digest,
        "fixture_catalog_digest": policy.expected_fixture_catalog_digest,
        "approval": Approval(),
        "budget": ExecutionBudget(),
        "preflight": ExecutionPreflight(nonce=nonce),
        "deadline_ms": 1000,
    }
    params.update(overrides)
    return ExecutionRequest(**params)  # type: ignore[arg-type]


_NONCES = (f"nonce-{i}" for i in itertools.count())


def test_offline_admitted_with_local_backend():
    policy = _policy()
    admission = admit(_request(policy, execution_mode="offline", backend_id="local-fake"), policy, issued_at=0.0)
    assert admission.status == "admitted"
    assert admission.preflight_result.credential_resolution == "not_attempted"


def test_offline_remote_backend_denied():
    policy = _policy()
    admission = admit(
        _request(policy, execution_mode="offline", backend_id="remote-explicit"), policy, issued_at=0.0
    )
    assert admission.status == "refused"
    assert admission.refusal is not None
    assert admission.refusal.code == "ADMISSION_POLICY_DENIED"


def test_payload_present_is_schema_invalid():
    policy = _policy()
    admission = admit(_request(policy, payload="present"), policy, issued_at=0.0)  # type: ignore[arg-type]
    assert admission.status == "refused"
    assert admission.refusal is not None
    assert admission.refusal.code == "REQUEST_SCHEMA_INVALID"


def test_digest_mismatch_is_contract_mismatch():
    policy = _policy()
    admission = admit(_request(policy, capability_manifest_digest="z" * 64), policy, issued_at=0.0)
    assert admission.status == "refused"
    assert admission.refusal is not None
    assert admission.refusal.code == "CONTRACT_MISMATCH"


def test_replay_requires_local_backend():
    policy = _policy()
    ok = admit(_request(policy, execution_mode="replay", backend_id="local-scorer"), policy, issued_at=0.0)
    assert ok.status == "admitted"
    denied = admit(
        _request(policy, execution_mode="replay", backend_id="remote-explicit"), policy, issued_at=0.0
    )
    assert denied.status == "refused"
    assert denied.refusal is not None
    assert denied.refusal.code == "ADMISSION_POLICY_DENIED"


def test_live_admitted_with_all_conditions():
    policy = _policy()
    request = _request(
        policy,
        execution_mode="live",
        backend_id="remote-explicit",
        approval=Approval(status="explicit", approval_id="appr-1", actor_class="maintainer"),
        budget=ExecutionBudget(status="configured", ceiling_units=10.0, currency_or_unit="usd"),
        preflight=ExecutionPreflight(nonce=next(_NONCES), requested=True),
    )
    admission = admit(request, policy, issued_at=0.0)
    assert admission.status == "admitted"
    assert admission.preflight_result.credential_resolution == "permitted_after_admission"


def test_live_denied_without_approval():
    policy = _policy()
    request = _request(
        policy,
        execution_mode="live",
        backend_id="remote-explicit",
        approval=Approval(),  # missing explicit approval
        budget=ExecutionBudget(status="configured", ceiling_units=10.0, currency_or_unit="usd"),
        preflight=ExecutionPreflight(nonce=next(_NONCES), requested=True),
    )
    admission = admit(request, policy, issued_at=0.0)
    assert admission.status == "refused"
    assert admission.refusal is not None
    assert admission.refusal.code == "ADMISSION_POLICY_DENIED"
    assert admission.preflight_result.credential_resolution == "not_attempted"


def test_live_nonce_single_use():
    policy = _policy()
    nonce = next(_NONCES)
    build = lambda: _request(
        policy,
        execution_mode="live",
        backend_id="remote-explicit",
        approval=Approval(status="explicit", approval_id="appr-1", actor_class="maintainer"),
        budget=ExecutionBudget(status="configured", ceiling_units=10.0, currency_or_unit="usd"),
        preflight=ExecutionPreflight(nonce=nonce, requested=True),
    )
    first = admit(build(), policy, issued_at=0.0)
    assert first.status == "admitted"
    second = admit(build(), policy, issued_at=1.0)
    assert second.status == "refused"
    assert second.refusal is not None
    assert second.refusal.code == "ADMISSION_POLICY_DENIED"


def test_admission_refusal_envelope_is_closed_shape():
    policy = _policy()
    admission = admit(_request(policy, payload="present"), policy, issued_at=0.0)  # type: ignore[arg-type]
    assert isinstance(admission.refusal, ClosedRefusalEnvelope)
    assert set(admission.refusal.to_dict()) == _ENVELOPE_FIELDS


# --------------------------------------------------------------------------- #
# Regression: review findings on the U1 slice                                 #
# --------------------------------------------------------------------------- #


def test_delegation_rejects_deleted_reference():
    # C1: a reference that exists but is already deleted must be refused before any grant/link is
    # created and without resurrecting the reference lifecycle.
    store = InMemoryContextStore(clock=_Clock())
    owner = _open(store)
    recipient = store.open_namespace(
        "ws-1",
        "sess-2",
        lease_ttl=300.0,
        heartbeat_interval=30.0,
        abandonment_grace=120.0,
        memory_quota=1024,
        file_quota=0,
    )
    reference = store.store(owner.namespace_id, b"gone")
    store.delete(owner.namespace_id, reference.reference_id)

    with pytest.raises(RefusalError) as caught:
        store.delegate(
            owner.namespace_id, recipient.namespace_id, (reference.reference_id,), expires_at=1000.0
        )
    assert caught.value.envelope.code == "REFERENCE_EXPIRED_OR_DELETED"
    # No grant was created and the reference stays deleted (no lifecycle resurrection).
    assert store._grants == {}
    assert store._references[reference.reference_id].lifecycle_state == "deleted"


def test_delegation_rejects_mix_leaves_no_partial_grant():
    # C1: when a batch mixes a valid reference with a deleted one, the whole delegate refuses and the
    # valid reference is NOT flipped to ``delegated`` (all-or-nothing up-front validation).
    store = InMemoryContextStore(clock=_Clock())
    owner = _open(store)
    recipient = store.open_namespace(
        "ws-1",
        "sess-2",
        lease_ttl=300.0,
        heartbeat_interval=30.0,
        abandonment_grace=120.0,
        memory_quota=1024,
        file_quota=0,
    )
    good = store.store(owner.namespace_id, b"keep")
    doomed = store.store(owner.namespace_id, b"drop")
    store.delete(owner.namespace_id, doomed.reference_id)

    with pytest.raises(RefusalError) as caught:
        store.delegate(
            owner.namespace_id,
            recipient.namespace_id,
            (good.reference_id, doomed.reference_id),
            expires_at=1000.0,
        )
    assert caught.value.envelope.code == "REFERENCE_EXPIRED_OR_DELETED"
    assert store._grants == {}
    assert store._references[good.reference_id].lifecycle_state == "active"


def test_ensure_usable_refuses_operation_outside_coverage():
    # M1: a valid, supported decision still refuses an operation outside its coverage set.
    manifest = _manifest()  # coverage operation_set == ("transform",)
    decision = revalidate(issue_decision(manifest, "A", decision_id="d", issued_at=0.0), manifest)
    assert decision.decision_lifecycle == "valid"

    with pytest.raises(RefusalError) as caught:
        ensure_usable(decision, "store", "corr")
    assert caught.value.envelope.code == "CAPABILITY_UNSUPPORTED"

    # The covered operation is still permitted.
    ensure_usable(decision, "transform", "corr")


def test_ensure_usable_refuses_excluded_operation():
    # M1: an operation listed in exclusions is refused even if present in the operation set.
    coverage = CoverageSet(
        coverage_set_id="cov-x",
        tool_classes=("search",),
        operation_set=("transform",),
        exclusions=("transform",),
    )
    evidence = EvidenceDigest(evidence_digest_id="ev-x", artifact_digests=("d1",))
    declaration = CapabilityDeclaration(
        declaration_id="decl-x",
        practice="A",
        capability_state="native",  # native does not require a non-empty coverage set
        implementation_kind="adapter",
        coverage=coverage,
        evidence=evidence,
    )
    manifest = CapabilityManifest(
        manifest_id="man-x",
        host="opencode",
        host_version="1.0",
        contract_identity=ContractIdentity.of_schema("1.0", {"schema": "u1"}),
        revision=1,
        declarations=(declaration,),
    )
    decision = revalidate(issue_decision(manifest, "A", decision_id="d", issued_at=0.0), manifest)
    assert decision.decision_lifecycle == "valid"
    with pytest.raises(RefusalError) as caught:
        ensure_usable(decision, "transform", "corr")
    assert caught.value.envelope.code == "CAPABILITY_UNSUPPORTED"


def test_decision_invalidated_when_evidence_id_changes():
    # M1: evidence_digest_id is part of the normative tuple — re-issuing the same evidence content
    # under a new id diverges and invalidates the snapshot.
    manifest = _manifest()
    decision = revalidate(issue_decision(manifest, "A", decision_id="d", issued_at=0.0), manifest)
    assert decision.decision_lifecycle == "valid"

    coverage = CoverageSet(coverage_set_id="cov-A", tool_classes=("search",), operation_set=("transform",))
    # Same artifact digests + matrix version => identical evidence_digest, but a different identity.
    evidence = EvidenceDigest(
        evidence_digest_id="ev-B", artifact_digests=("d1",), compatibility_matrix_version="1.0"
    )
    declaration = CapabilityDeclaration(
        declaration_id="decl-A",
        practice="A",
        capability_state="full",
        implementation_kind="adapter",
        coverage=coverage,
        evidence=evidence,
    )
    rebranded = CapabilityManifest(
        manifest_id="man-1",
        host="opencode",
        host_version="1.2.3",
        contract_identity=ContractIdentity.of_schema("1.0", {"schema": "u1"}),
        revision=1,
        declarations=(declaration,),
    )
    # The content digests match, so only evidence_digest_id diverges in the normative tuple.
    assert manifest.manifest_digest == rebranded.manifest_digest
    stale = revalidate(decision, rebranded)
    assert stale.decision_lifecycle == "invalidated"


def test_lease_abandoned_renewed_within_grace():
    # M2: a missed heartbeat marks the lease abandoned; a renewal inside the grace window reactivates
    # it and clears the abandonment mark.
    clock = _Clock()
    store = InMemoryContextStore(clock=clock)
    namespace = _open(store, lease_ttl=1000.0, heartbeat_interval=10.0, abandonment_grace=50.0)
    clock.advance(20.0)  # heartbeat window lapsed at t=10; grace boundary at t=60

    lease = store.heartbeat(namespace.namespace_id)
    assert lease.lease_state == "active"
    assert lease.abandoned_at is None
    reference = store.store(namespace.namespace_id, b"ok")
    assert reference.lifecycle_state == "active"


def test_lease_expires_after_grace_blocks_renewal():
    # M2: past abandoned_at + abandonment_grace the lease is expired and can never be renewed again,
    # even though the TTL has not been reached.
    clock = _Clock()
    store = InMemoryContextStore(clock=clock)
    namespace = _open(store, lease_ttl=1000.0, heartbeat_interval=10.0, abandonment_grace=50.0)
    clock.advance(65.0)  # grace boundary t=60 passed; TTL (1000) not reached

    with pytest.raises(RefusalError) as caught:
        store.heartbeat(namespace.namespace_id)
    assert caught.value.envelope.code == "LEASE_EXPIRED"


def test_lease_evaluate_pins_abandonment_start_not_now():
    # M2: abandoned_at is pinned to when abandonment began (end of the heartbeat window), not to the
    # evaluation time, so a late first evaluation still measures grace from the right instant.
    clock = _Clock()
    store = InMemoryContextStore(clock=clock)
    namespace = _open(store, lease_ttl=1000.0, heartbeat_interval=10.0, abandonment_grace=50.0)
    lease = store._leases[namespace.namespace_id]
    assert lease.evaluate(40.0) == "abandoned"
    assert lease.abandoned_at == 10.0  # 0 (last heartbeat) + 10 (heartbeat_interval)
    assert lease.evaluate(61.0) == "expired"  # 10 + 50 grace boundary crossed


def test_live_nonce_consumed_atomically_under_concurrency():
    # Robustness: two concurrent live admissions presenting the same nonce — exactly one is admitted.
    import threading

    policy = _policy()
    nonce = next(_NONCES)

    def build() -> ExecutionRequest:
        return _request(
            policy,
            execution_mode="live",
            backend_id="remote-explicit",
            approval=Approval(status="explicit", approval_id="appr-1", actor_class="maintainer"),
            budget=ExecutionBudget(status="configured", ceiling_units=10.0, currency_or_unit="usd"),
            preflight=ExecutionPreflight(nonce=nonce, requested=True),
        )

    results: list[str] = []
    results_lock = threading.Lock()
    barrier = threading.Barrier(2)

    def run() -> None:
        request = build()
        barrier.wait()
        admission = admit(request, policy, issued_at=0.0)
        with results_lock:
            results.append(admission.status)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(results) == ["admitted", "refused"]
