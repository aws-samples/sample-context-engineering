"""Tests for the U1 ``context_core.runtime`` orchestrator (W0–W4 over the contracts).

Covers W4 startup/health/preflight, W1 schema/contract/capability gates, transform success and
fallback, the reference operations (store/retrieve/delete/catalog/delegate), idempotency replay and
conflict, the payload-free timeout-with-no-effect path, and the neutral-contract guarantee that the
runtime pulls in no network/AWS/agent-framework module. Everything runs offline with an injected clock.
"""

from __future__ import annotations

import pytest

from context_core.contracts import (
    CapabilityDeclaration,
    CapabilityManifest,
    ContractIdentity,
    CoverageSet,
    EvidenceDigest,
    IdempotencyStore,
    InMemoryContextStore,
    digest,
    issue_decision,
)
from context_core.contracts.errors import _ENVELOPE_FIELDS
from context_core.runtime import (
    EFFECT_OPERATIONS,
    ContextRuntime,
    DependencyUnavailable,
    ProcessingRequest,
)


class _Clock:
    """A deterministic clock the tests advance by hand."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


CONTRACT = ContractIdentity.of_schema("1.0", {"schema": "u1"})


def _manifest(capability_state: str = "full", *, revision: int = 1) -> CapabilityManifest:
    coverage = CoverageSet(
        coverage_set_id="cov-A",
        tool_classes=("search",),
        operation_set=("transform",),
    )
    evidence = EvidenceDigest(
        evidence_digest_id="ev-A", artifact_digests=("d1",), compatibility_matrix_version="1.0"
    )
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
        contract_identity=CONTRACT,
        revision=revision,
        declarations=(declaration,),
    )


def _runtime(
    *,
    manifest: CapabilityManifest | None = None,
    transform_fn=None,
    clock: _Clock | None = None,
    contract_identity: ContractIdentity = CONTRACT,
) -> tuple[ContextRuntime, InMemoryContextStore, _Clock]:
    clock = clock or _Clock()
    store = InMemoryContextStore(clock=clock)
    idempotency = IdempotencyStore(retention_seconds=1000.0)
    runtime = ContextRuntime(
        store,
        idempotency,
        contract_identity,
        manifest=manifest,
        transform_fn=transform_fn,
        clock=clock,
    )
    return runtime, store, clock


def _open(store: InMemoryContextStore, session_id: str = "sess-1"):
    return store.open_namespace(
        "ws-1",
        session_id,
        lease_ttl=300.0,
        heartbeat_interval=30.0,
        abandonment_grace=120.0,
        memory_quota=1024,
        file_quota=0,
    )


# --------------------------------------------------------------------------- #
# W4 — startup / health / preflight                                           #
# --------------------------------------------------------------------------- #


def test_startup_acknowledges_and_publishes_capabilities():
    runtime, _store, _clock = _runtime(manifest=_manifest())
    outcome = runtime.process(
        ProcessingRequest(
            request_id="r1",
            operation="startup",
            operation_family="admission",
            contract_identity=CONTRACT,
        )
    )
    assert outcome.status == "started"
    assert outcome.result is not None
    assert outcome.result.kind == "startup_ack"
    assert outcome.result.metadata is not None
    assert outcome.result.metadata["capabilities"] == [{"practice": "A", "capability_state": "full"}]


def test_health_and_preflight_are_repeatable_without_effect():
    runtime, _store, _clock = _runtime()
    for _ in range(3):
        health = runtime.process(
            ProcessingRequest(
                request_id="h", operation="health", operation_family="admission", contract_identity=CONTRACT
            )
        )
        assert health.status == "healthy"
        assert health.result is not None and health.result.kind == "health_ack"

    preflight = runtime.process(
        ProcessingRequest(
            request_id="p", operation="preflight", operation_family="admission", contract_identity=CONTRACT
        )
    )
    assert preflight.status == "preflight_passed"


def test_health_does_not_require_matching_contract():
    # Pure liveness must answer even when the caller's contract differs (BR5.2).
    runtime, _store, _clock = _runtime()
    other = ContractIdentity.of_schema("2.0", {"schema": "other"})
    outcome = runtime.process(
        ProcessingRequest(
            request_id="h", operation="health", operation_family="admission", contract_identity=other
        )
    )
    assert outcome.status == "healthy"


# --------------------------------------------------------------------------- #
# W1 — schema, family, contract                                               #
# --------------------------------------------------------------------------- #


def test_unknown_operation_is_schema_invalid():
    runtime, _store, _clock = _runtime()
    outcome = runtime.process(
        ProcessingRequest(
            request_id="r",
            operation="frobnicate",  # type: ignore[arg-type]
            operation_family="runtime",
            contract_identity=CONTRACT,
        )
    )
    assert outcome.status == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code == "REQUEST_SCHEMA_INVALID"
    assert set(outcome.refusal.to_dict()) == _ENVELOPE_FIELDS


def test_family_mismatch_is_schema_invalid():
    runtime, _store, _clock = _runtime()
    # transform is a runtime operation; declaring it admission is a schema error.
    outcome = runtime.process(
        ProcessingRequest(
            request_id="r",
            operation="transform",
            operation_family="admission",
            contract_identity=CONTRACT,
        )
    )
    assert outcome.status == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code == "REQUEST_SCHEMA_INVALID"


def test_effect_operation_without_idempotency_keys_is_schema_invalid():
    runtime, store, _clock = _runtime()
    namespace = _open(store)
    outcome = runtime.process(
        ProcessingRequest(
            request_id="r",
            operation="store",
            operation_family="runtime",
            contract_identity=CONTRACT,
            namespace_id=namespace.namespace_id,
            content=b"x",
            # idempotency_scope/key/request_digest intentionally omitted
        )
    )
    assert outcome.status == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code == "REQUEST_SCHEMA_INVALID"


def test_contract_mismatch_is_refused_before_effect():
    runtime, store, _clock = _runtime()
    namespace = _open(store)
    wrong_contract = ContractIdentity.of_schema("2.0", {"schema": "other"})
    outcome = runtime.process(
        ProcessingRequest(
            request_id="r",
            operation="store",
            operation_family="runtime",
            contract_identity=wrong_contract,
            namespace_id=namespace.namespace_id,
            content=b"payload",
            idempotency_scope="s",
            idempotency_key="k",
            request_digest=digest({"op": "store"}),
        )
    )
    assert outcome.status == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code == "CONTRACT_MISMATCH"
    # No reference was created despite the content being present on the request.
    assert runtime.process(
        ProcessingRequest(
            request_id="c", operation="catalog", operation_family="runtime",
            contract_identity=CONTRACT, namespace_id=namespace.namespace_id,
        )
    ).result.metadata["references"] == []


# --------------------------------------------------------------------------- #
# W0/W1 — transform capability gates and success/fallback                     #
# --------------------------------------------------------------------------- #


def _transform_request(decision, **overrides) -> ProcessingRequest:
    params = {
        "request_id": "t1",
        "operation": "transform",
        "operation_family": "runtime",
        "contract_identity": CONTRACT,
        "practice": "A",
        "capability_decision": decision,
        "neutral_input": "oversized tool result",
        "baseline": "oversized tool result",
        "idempotency_scope": "s",
        "idempotency_key": "t-key",
        "request_digest": digest({"op": "transform", "n": 1}),
    }
    params.update(overrides)
    return ProcessingRequest(**params)  # type: ignore[arg-type]


def test_transform_success_runs_neutral_callback():
    manifest = _manifest()
    decision = issue_decision(manifest, "A", decision_id="dec-1", issued_at=0.0)
    seen: list[object] = []

    def shrink(value):
        seen.append(value)
        return f"shrunk:{value}"

    runtime, _store, _clock = _runtime(manifest=manifest, transform_fn=shrink)
    outcome = runtime.process(_transform_request(decision))
    assert outcome.status == "transformed"
    assert outcome.result is not None
    assert outcome.result.kind == "transformed_content"
    assert outcome.result.content == "shrunk:oversized tool result"
    assert outcome.result.content_digest is not None
    assert seen == ["oversized tool result"]  # the callback is the only boundary to the logic


def test_transform_stale_decision_is_contract_mismatch():
    # Decision issued against revision 1, runtime now holds revision 2: revalidation invalidates it.
    old_manifest = _manifest(revision=1)
    decision = issue_decision(old_manifest, "A", decision_id="dec-1", issued_at=0.0)
    new_manifest = _manifest(revision=2)

    runtime, _store, _clock = _runtime(manifest=new_manifest, transform_fn=lambda v: v)
    outcome = runtime.process(_transform_request(decision))
    assert outcome.status == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code == "CONTRACT_MISMATCH"


def test_transform_unsupported_capability_is_refused():
    manifest = _manifest(capability_state="unsupported")
    decision = issue_decision(manifest, "A", decision_id="dec-1", issued_at=0.0)
    runtime, _store, _clock = _runtime(manifest=manifest, transform_fn=lambda v: v)
    outcome = runtime.process(_transform_request(decision))
    assert outcome.status == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code == "CAPABILITY_UNSUPPORTED"


def test_transform_fallback_preserves_baseline():
    manifest = _manifest()
    decision = issue_decision(manifest, "A", decision_id="dec-1", issued_at=0.0)

    def failing(_value):
        raise DependencyUnavailable("reranker_unavailable")

    runtime, _store, _clock = _runtime(manifest=manifest, transform_fn=failing)
    outcome = runtime.process(_transform_request(decision, baseline="the original"))
    assert outcome.status == "fallback_original"
    assert outcome.baseline_preserved is True
    assert outcome.result is not None
    assert outcome.result.kind == "baseline_original"
    assert outcome.result.content == "the original"
    assert outcome.result.fallback_reason == "reranker_unavailable"
    assert outcome.refusal is None


def test_transform_no_baseline_refuses_deterministically():
    manifest = _manifest()
    decision = issue_decision(manifest, "A", decision_id="dec-1", issued_at=0.0)

    def failing(_value):
        raise DependencyUnavailable("codec_unavailable")

    runtime, _store, _clock = _runtime(manifest=manifest, transform_fn=failing)
    outcome = runtime.process(_transform_request(decision, baseline=None))
    assert outcome.status == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code == "DEPENDENCY_UNAVAILABLE_NO_FALLBACK"


# --------------------------------------------------------------------------- #
# W1/W3 — reference operations                                                #
# --------------------------------------------------------------------------- #


def _store_request(namespace_id, content=b"hello", key="store-key", **overrides):
    params = {
        "request_id": "s1",
        "operation": "store",
        "operation_family": "runtime",
        "contract_identity": CONTRACT,
        "namespace_id": namespace_id,
        "content": content,
        "idempotency_scope": "scope",
        "idempotency_key": key,
        "request_digest": digest({"content": content.decode()}),
    }
    params.update(overrides)
    return ProcessingRequest(**params)  # type: ignore[arg-type]


def test_store_retrieve_delete_round_trip():
    runtime, store, _clock = _runtime()
    namespace = _open(store)

    stored = runtime.process(_store_request(namespace.namespace_id))
    assert stored.status == "stored"
    assert stored.result is not None
    reference_id = stored.result.reference_id
    assert reference_id is not None

    retrieved = runtime.process(
        ProcessingRequest(
            request_id="g1", operation="retrieve", operation_family="runtime",
            contract_identity=CONTRACT, namespace_id=namespace.namespace_id, reference_id=reference_id,
        )
    )
    assert retrieved.status == "retrieved"
    assert retrieved.result is not None
    assert retrieved.result.content == b"hello"
    assert retrieved.result.metadata == {"content_type": "text/plain"}

    deleted = runtime.process(
        ProcessingRequest(
            request_id="d1", operation="delete", operation_family="runtime",
            contract_identity=CONTRACT, namespace_id=namespace.namespace_id, reference_id=reference_id,
            idempotency_scope="scope", idempotency_key="del-key", request_digest=digest({"del": reference_id}),
        )
    )
    assert deleted.status == "deleted"

    gone = runtime.process(
        ProcessingRequest(
            request_id="g2", operation="retrieve", operation_family="runtime",
            contract_identity=CONTRACT, namespace_id=namespace.namespace_id, reference_id=reference_id,
        )
    )
    assert gone.status == "refused"
    assert gone.refusal is not None
    assert gone.refusal.code == "REFERENCE_EXPIRED_OR_DELETED"


def test_store_unknown_namespace_is_refused_without_payload():
    runtime, _store, _clock = _runtime()
    outcome = runtime.process(_store_request("ghost-namespace"))
    assert outcome.status == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code == "NAMESPACE_INVALID"
    # The closed envelope carries none of the request payload.
    forbidden = {"content", "payload", "reference_id", "namespace_id"}
    assert forbidden.isdisjoint(outcome.refusal.to_dict())


def test_catalog_lists_live_references():
    runtime, store, _clock = _runtime()
    namespace = _open(store)
    runtime.process(_store_request(namespace.namespace_id, content=b"one", key="k1"))
    runtime.process(_store_request(namespace.namespace_id, content=b"two", key="k2"))

    cataloged = runtime.process(
        ProcessingRequest(
            request_id="c1", operation="catalog", operation_family="runtime",
            contract_identity=CONTRACT, namespace_id=namespace.namespace_id,
        )
    )
    assert cataloged.status == "cataloged"
    assert cataloged.result is not None
    assert len(cataloged.result.metadata["references"]) == 2


def test_delegate_grants_named_access():
    runtime, store, _clock = _runtime()
    owner = _open(store, session_id="sess-owner")
    recipient = _open(store, session_id="sess-recipient")
    stored = runtime.process(_store_request(owner.namespace_id, content=b"shared", key="k-shared"))
    reference_id = stored.result.reference_id

    delegated = runtime.process(
        ProcessingRequest(
            request_id="dg1", operation="delegate", operation_family="runtime",
            contract_identity=CONTRACT, namespace_id=owner.namespace_id,
            recipient_namespace_id=recipient.namespace_id, reference_ids=(reference_id,),
            grant_expires_at=1000.0,
            idempotency_scope="scope", idempotency_key="dg-key", request_digest=digest({"dg": reference_id}),
        )
    )
    assert delegated.status == "delegated"
    assert delegated.result is not None
    assert "grant_id" in delegated.result.metadata

    # The recipient can now retrieve through the grant.
    retrieved = runtime.process(
        ProcessingRequest(
            request_id="rr", operation="retrieve", operation_family="runtime",
            contract_identity=CONTRACT, namespace_id=recipient.namespace_id, reference_id=reference_id,
        )
    )
    assert retrieved.status == "retrieved"
    assert retrieved.result.content == b"shared"


def test_delegate_empty_reference_set_is_schema_invalid():
    runtime, store, _clock = _runtime()
    owner = _open(store, session_id="sess-owner")
    recipient = _open(store, session_id="sess-recipient")
    outcome = runtime.process(
        ProcessingRequest(
            request_id="dg", operation="delegate", operation_family="runtime",
            contract_identity=CONTRACT, namespace_id=owner.namespace_id,
            recipient_namespace_id=recipient.namespace_id, reference_ids=(),
            grant_expires_at=1000.0,
            idempotency_scope="scope", idempotency_key="dg-key", request_digest=digest({"dg": 0}),
        )
    )
    assert outcome.status == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code == "REQUEST_SCHEMA_INVALID"


# --------------------------------------------------------------------------- #
# Idempotency: replay / conflict                                              #
# --------------------------------------------------------------------------- #


def test_idempotency_replays_same_outcome_without_second_effect():
    runtime, store, _clock = _runtime()
    namespace = _open(store)
    first = runtime.process(_store_request(namespace.namespace_id, key="dup"))
    assert first.status == "stored"

    # Same scope/key/digest retry: the committed outcome is replayed byte-for-byte, no new reference.
    second = runtime.process(
        _store_request(namespace.namespace_id, key="dup", request_id="s1-retry")
    )
    assert second is first  # the exact committed outcome object is returned

    cataloged = runtime.process(
        ProcessingRequest(
            request_id="c", operation="catalog", operation_family="runtime",
            contract_identity=CONTRACT, namespace_id=namespace.namespace_id,
        )
    )
    assert len(cataloged.result.metadata["references"]) == 1  # only one effect happened


def test_idempotency_conflict_on_different_digest():
    runtime, store, _clock = _runtime()
    namespace = _open(store)
    runtime.process(_store_request(namespace.namespace_id, content=b"first", key="same"))

    conflict = runtime.process(
        ProcessingRequest(
            request_id="s2", operation="store", operation_family="runtime",
            contract_identity=CONTRACT, namespace_id=namespace.namespace_id, content=b"second",
            idempotency_scope="scope", idempotency_key="same", request_digest=digest({"other": "digest"}),
        )
    )
    assert conflict.status == "refused"
    assert conflict.refusal is not None
    assert conflict.refusal.code == "IDEMPOTENCY_CONFLICT"


def test_refusal_is_committed_and_replayed():
    # A deterministic refusal is itself the committed terminal: the same key/digest replays it.
    runtime, _store, _clock = _runtime()
    request = _store_request("ghost", key="refuse-key")
    first = runtime.process(request)
    assert first.status == "refused"
    second = runtime.process(_store_request("ghost", key="refuse-key", request_id="retry"))
    assert second is first


# --------------------------------------------------------------------------- #
# Timeout: no durable effect                                                  #
# --------------------------------------------------------------------------- #


def test_timeout_before_commit_leaves_no_effect():
    clock = _Clock(start=100.0)
    runtime, store, _clock = _runtime(clock=clock)
    namespace = _open(store)
    # Deadline already in the past relative to the clock.
    outcome = runtime.process(
        _store_request(namespace.namespace_id, key="late", deadline_at=50.0)
    )
    assert outcome.status == "timed_out"
    assert outcome.result is not None
    assert outcome.result.kind == "timeout_without_payload"
    assert outcome.refusal is None

    # No reference was stored, and the key is free for a fresh (in-time) attempt.
    cataloged = runtime.process(
        ProcessingRequest(
            request_id="c", operation="catalog", operation_family="runtime",
            contract_identity=CONTRACT, namespace_id=namespace.namespace_id,
        )
    )
    assert cataloged.result.metadata["references"] == []

    retry = runtime.process(_store_request(namespace.namespace_id, key="late", request_id="s1-again"))
    assert retry.status == "stored"


def test_transform_timeout_does_not_run_callback():
    clock = _Clock(start=100.0)
    manifest = _manifest()
    decision = issue_decision(manifest, "A", decision_id="dec-1", issued_at=0.0)
    called: list[int] = []

    def tracked(value):
        called.append(1)
        return value

    runtime, _store, _clock = _runtime(manifest=manifest, transform_fn=tracked, clock=clock)
    outcome = runtime.process(_transform_request(decision, deadline_at=10.0))
    assert outcome.status == "timed_out"
    assert called == []  # the neutral callback never ran


# --------------------------------------------------------------------------- #
# C1 — shutdown as a durable effect through the runtime                       #
# --------------------------------------------------------------------------- #


def _shutdown_request(namespace_id, key="shutdown-key", **overrides) -> ProcessingRequest:
    params = {
        "request_id": "sd1",
        "operation": "shutdown",
        "operation_family": "runtime",
        "contract_identity": CONTRACT,
        "namespace_id": namespace_id,
        "idempotency_scope": "scope",
        "idempotency_key": key,
        "request_digest": digest({"op": "shutdown", "ns": namespace_id}),
    }
    params.update(overrides)
    return ProcessingRequest(**params)  # type: ignore[arg-type]


def test_shutdown_discards_namespace_and_blocks_retrieval():
    runtime, store, _clock = _runtime()
    namespace = _open(store)
    stored = runtime.process(_store_request(namespace.namespace_id, content=b"doomed", key="k-sd"))
    reference_id = stored.result.reference_id

    outcome = runtime.process(_shutdown_request(namespace.namespace_id))
    assert outcome.status == "deleted"
    assert outcome.result is not None
    assert outcome.result.kind == "deletion_ack"
    assert outcome.result.metadata["references_discarded"] == 1

    # The content is gone: retrieving the discarded reference is refused.
    gone = runtime.process(
        ProcessingRequest(
            request_id="g", operation="retrieve", operation_family="runtime",
            contract_identity=CONTRACT, namespace_id=namespace.namespace_id, reference_id=reference_id,
        )
    )
    assert gone.status == "refused"
    assert gone.refusal is not None
    assert gone.refusal.code in {"REFERENCE_EXPIRED_OR_DELETED", "LEASE_EXPIRED"}

    # The lease is discarded, so a fresh store on the same namespace is refused.
    after = runtime.process(_store_request(namespace.namespace_id, key="k-after"))
    assert after.status == "refused"
    assert after.refusal is not None
    assert after.refusal.code == "LEASE_EXPIRED"


def test_shutdown_contract_mismatch_refused_before_effect():
    runtime, store, _clock = _runtime()
    namespace = _open(store)
    stored = runtime.process(_store_request(namespace.namespace_id, content=b"keep", key="k-keep"))
    reference_id = stored.result.reference_id
    wrong_contract = ContractIdentity.of_schema("2.0", {"schema": "other"})

    outcome = runtime.process(_shutdown_request(namespace.namespace_id, contract_identity=wrong_contract))
    assert outcome.status == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code == "CONTRACT_MISMATCH"

    # No discard happened: the reference is still retrievable.
    retrieved = runtime.process(
        ProcessingRequest(
            request_id="g", operation="retrieve", operation_family="runtime",
            contract_identity=CONTRACT, namespace_id=namespace.namespace_id, reference_id=reference_id,
        )
    )
    assert retrieved.status == "retrieved"
    assert retrieved.result.content == b"keep"


def test_shutdown_without_namespace_is_schema_invalid():
    runtime, _store, _clock = _runtime()
    outcome = runtime.process(
        ProcessingRequest(
            request_id="sd", operation="shutdown", operation_family="runtime",
            contract_identity=CONTRACT,
            idempotency_scope="scope", idempotency_key="k", request_digest=digest({"op": "shutdown"}),
            # namespace_id intentionally omitted
        )
    )
    assert outcome.status == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code == "REQUEST_SCHEMA_INVALID"


def test_shutdown_is_idempotent_and_replays():
    runtime, store, _clock = _runtime()
    namespace = _open(store)
    runtime.process(_store_request(namespace.namespace_id, content=b"x", key="k-x"))
    first = runtime.process(_shutdown_request(namespace.namespace_id, key="sd-dup"))
    assert first.status == "deleted"
    # A matching retry replays the committed terminal rather than discarding a second time.
    second = runtime.process(_shutdown_request(namespace.namespace_id, key="sd-dup", request_id="sd-retry"))
    assert second is first


def test_shutdown_closes_delegated_reference_lifecycle():
    # Lifecycle regression: discarding the issuer namespace revokes its outstanding grants, so a
    # recipient that could read a delegated reference can no longer retrieve it.
    runtime, store, _clock = _runtime()
    owner = _open(store, session_id="sess-owner")
    recipient = _open(store, session_id="sess-recipient")
    stored = runtime.process(_store_request(owner.namespace_id, content=b"shared", key="k-share"))
    reference_id = stored.result.reference_id
    runtime.process(
        ProcessingRequest(
            request_id="dg", operation="delegate", operation_family="runtime",
            contract_identity=CONTRACT, namespace_id=owner.namespace_id,
            recipient_namespace_id=recipient.namespace_id, reference_ids=(reference_id,),
            grant_expires_at=1_000_000.0,
            idempotency_scope="scope", idempotency_key="dg", request_digest=digest({"dg": reference_id}),
        )
    )

    runtime.process(_shutdown_request(owner.namespace_id, key="sd-owner"))

    denied = runtime.process(
        ProcessingRequest(
            request_id="rr", operation="retrieve", operation_family="runtime",
            contract_identity=CONTRACT, namespace_id=recipient.namespace_id, reference_id=reference_id,
        )
    )
    assert denied.status == "refused"
    assert denied.refusal is not None
    assert denied.refusal.code in {"REFERENCE_EXPIRED_OR_DELETED", "AUTHORIZATION_HOST_DENIED"}


# --------------------------------------------------------------------------- #
# W1/W2/W3 — a dead lease blocks delegate / retrieve / catalog                #
# --------------------------------------------------------------------------- #


def test_retrieve_refused_when_acting_lease_expired():
    clock = _Clock()
    runtime, store, _clock = _runtime(clock=clock)
    namespace = store.open_namespace(
        "ws-1", "sess-short", lease_ttl=10.0, heartbeat_interval=5.0,
        abandonment_grace=0.0, memory_quota=1024, file_quota=0,
    )
    stored = runtime.process(_store_request(namespace.namespace_id, key="k-live"))
    reference_id = stored.result.reference_id
    clock.advance(11.0)  # lease TTL passed

    outcome = runtime.process(
        ProcessingRequest(
            request_id="g", operation="retrieve", operation_family="runtime",
            contract_identity=CONTRACT, namespace_id=namespace.namespace_id, reference_id=reference_id,
        )
    )
    assert outcome.status == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code == "LEASE_EXPIRED"


def test_catalog_refused_when_lease_expired():
    clock = _Clock()
    runtime, store, _clock = _runtime(clock=clock)
    namespace = store.open_namespace(
        "ws-1", "sess-short", lease_ttl=10.0, heartbeat_interval=5.0,
        abandonment_grace=0.0, memory_quota=1024, file_quota=0,
    )
    clock.advance(11.0)
    outcome = runtime.process(
        ProcessingRequest(
            request_id="c", operation="catalog", operation_family="runtime",
            contract_identity=CONTRACT, namespace_id=namespace.namespace_id,
        )
    )
    assert outcome.status == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code == "LEASE_EXPIRED"


def test_delegate_refused_when_recipient_lease_expired():
    clock = _Clock()
    runtime, store, _clock = _runtime(clock=clock)
    owner = _open(store, session_id="sess-owner")
    recipient = store.open_namespace(
        "ws-1", "sess-recipient", lease_ttl=10.0, heartbeat_interval=5.0,
        abandonment_grace=0.0, memory_quota=1024, file_quota=0,
    )
    stored = runtime.process(_store_request(owner.namespace_id, content=b"shared", key="k-share"))
    reference_id = stored.result.reference_id
    clock.advance(11.0)  # only the recipient lease lapses relative to the owner's longer TTL

    outcome = runtime.process(
        ProcessingRequest(
            request_id="dg", operation="delegate", operation_family="runtime",
            contract_identity=CONTRACT, namespace_id=owner.namespace_id,
            recipient_namespace_id=recipient.namespace_id, reference_ids=(reference_id,),
            grant_expires_at=1_000_000.0,
            idempotency_scope="scope", idempotency_key="dg", request_digest=digest({"dg": reference_id}),
        )
    )
    assert outcome.status == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code == "LEASE_EXPIRED"


# --------------------------------------------------------------------------- #
# W4 — an unexpected transform defect releases the reservation               #
# --------------------------------------------------------------------------- #


def test_unexpected_transform_defect_releases_reservation_and_propagates():
    manifest = _manifest()
    decision = issue_decision(manifest, "A", decision_id="dec-1", issued_at=0.0)
    calls: list[int] = []

    def boom(_value):
        calls.append(1)
        raise RuntimeError("unexpected defect in the transform callback")

    runtime, _store, _clock = _runtime(manifest=manifest, transform_fn=boom)
    # The defect is NOT a DependencyUnavailable, so it is not masked as a fallback: it propagates.
    with pytest.raises(RuntimeError, match="unexpected defect"):
        runtime.process(_transform_request(decision, idempotency_key="defect-key"))

    # The reservation was released (W4): a retry with the same key is not stranded as PENDING/timeout —
    # it re-enters the effect path and runs the callback again rather than replaying a committed outcome.
    with pytest.raises(RuntimeError, match="unexpected defect"):
        runtime.process(_transform_request(decision, idempotency_key="defect-key", request_id="t-retry"))
    assert calls == [1, 1]  # the callback ran on both attempts; the key was never stuck


# --------------------------------------------------------------------------- #
# Neutral-contract guard                                                      #
# --------------------------------------------------------------------------- #


def test_effect_operations_constant_matches_spec():
    assert EFFECT_OPERATIONS == {"transform", "store", "delete", "delegate", "shutdown"}


def test_dependency_unavailable_rejects_unknown_reason():
    with pytest.raises(ValueError):
        DependencyUnavailable("not_a_reason")


def test_runtime_module_imports_no_network_or_framework():
    """Importing the runtime pulls in no boto3/botocore/network client or agent framework."""
    import subprocess
    import sys

    probe = (
        "import sys, context_core.runtime\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in "
        "{'boto3','botocore','strands','langchain','langgraph','langchain_core'}]\n"
        "print('LEAK:' + ','.join(sorted(bad)) if bad else 'CLEAN')\n"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "CLEAN" in result.stdout, result.stdout
