"""Tests for the offline ``context_core.opencode`` boundary (opencode-port-spec, U3).

These exercise the synchronous, framework-free OpenCode adapter end to end over the real U1 contracts
and :class:`~context_core.runtime.ContextRuntime`: startup publishing A with B/D unsupported (validating
the declarative hook allowlist, adapter version and manifest digest), the Practice A happy path, the
authorization-before-capability precedence, a stale-decision/tampered-provenance refusal, B/D capability
refusals, the closed seven-field refusal envelope, the ``deadline_ms`` conversion and rejection, the
already-truncated source preserved (with its byte/metadata/shape/origin digests) without a transform or
reference, the explicit evidence ids on success/baseline, and the synchronous shutdown. Everything runs
offline with an injected clock; a subprocess probe asserts the module pulls in no agent framework.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace

from context_core.contracts import ContractIdentity
from context_core.contracts.errors import _ENVELOPE_FIELDS
from context_core.opencode import (
    EVIDENCE_NAMES,
    OPENCODE_HOST,
    OpenCodeAdapter,
    OpenCodeCoverageEvidence,
    OpenCodeCoverageSet,
    OpenCodeSessionBinding,
    OpenCodeSourceAlreadyTruncated,
)

CONTRACT = ContractIdentity.of_schema("1.0", {"schema": "u1"})
HOST_VERSION = "0.4.2"
ADAPTER_VERSION = "0.1.0"


class _Clock:
    """A deterministic clock the tests advance by hand."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def _evidence() -> OpenCodeCoverageEvidence:
    # Six distinct, well-formed sha256 digests standing in for the six mandatory evidence facets.
    return OpenCodeCoverageEvidence(
        payload_integrity_digest="11" * 32,
        substitution_digest="22" * 32,
        retrieval_digest="33" * 32,
        metadata_preservation_digest="44" * 32,
        isolation_digest="55" * 32,
        fallback_digest="66" * 32,
    )


def _coverage() -> OpenCodeCoverageSet:
    return OpenCodeCoverageSet(
        coverage_set_id="opencode-cov-A",
        included_tool_classes=("file_read", "web_fetch"),
        evidence=_evidence(),
        fixture_ids=("fx-file_read", "fx-web_fetch"),
        excluded_tool_classes=("shell_exec",),
    )


def _adapter(
    *,
    transform_fn=None,
    a_state="full",
    manifest_revision=1,
    clock: _Clock | None = None,
    coverage: OpenCodeCoverageSet | None = None,
) -> OpenCodeAdapter:
    return OpenCodeAdapter(
        CONTRACT,
        coverage or _coverage(),
        transform_fn=transform_fn,
        host_version=HOST_VERSION,
        adapter_version=ADAPTER_VERSION,
        a_state=a_state,
        manifest_revision=manifest_revision,
        clock=clock or _Clock(),
    )


def _binding(
    adapter: OpenCodeAdapter,
    *,
    practice: str = "A",
    operation: str = "transform",
    decision=None,
    authorization: bool = True,
    namespace_id: str = "ns-1",
    deadline_ms: int = 30_000,
    key: str | None = None,
) -> OpenCodeSessionBinding:
    """Build a coherent binding through the adapter, defaulting to a protected transform."""
    if decision is None:
        decision = adapter.issue_decision(practice if practice in ("A", "B", "D") else "A")
    return adapter.bind(
        decision,
        operation=operation,
        workspace_id="ws-1",
        session_id="sess-1",
        namespace_id=namespace_id,
        authorization=authorization,
        deadline_ms=deadline_ms,
        practice=practice,
        idempotency_key=key,
    )


# --------------------------------------------------------------------------- #
# Coverage set / evidence construction                                        #
# --------------------------------------------------------------------------- #


def test_empty_coverage_set_is_rejected_at_construction():
    try:
        OpenCodeCoverageSet(
            coverage_set_id="c", included_tool_classes=(), evidence=_evidence(), fixture_ids=("fx",)
        )
    except ValueError as error:
        assert "non-empty" in str(error)
    else:  # pragma: no cover - the constructor must reject an empty allowlist
        raise AssertionError("empty coverage set must be rejected")


def test_coverage_overlap_is_rejected():
    try:
        OpenCodeCoverageSet(
            coverage_set_id="c",
            included_tool_classes=("a", "b"),
            evidence=_evidence(),
            fixture_ids=("fx",),
            excluded_tool_classes=("b",),
        )
    except ValueError:
        pass
    else:  # pragma: no cover - include/exclude overlap must be rejected
        raise AssertionError("overlap must be rejected")


def test_empty_fixtures_are_rejected():
    try:
        OpenCodeCoverageSet(
            coverage_set_id="c",
            included_tool_classes=("a",),
            evidence=_evidence(),
            fixture_ids=(),
        )
    except ValueError as error:
        assert "fixture_ids" in str(error)
    else:  # pragma: no cover - an empty fixture set must be rejected
        raise AssertionError("empty fixture_ids must be rejected")


def test_malformed_evidence_digest_is_rejected():
    try:
        OpenCodeCoverageEvidence(
            payload_integrity_digest="not-a-digest",
            substitution_digest="22" * 32,
            retrieval_digest="33" * 32,
            metadata_preservation_digest="44" * 32,
            isolation_digest="55" * 32,
            fallback_digest="66" * 32,
        )
    except ValueError as error:
        assert "sha256" in str(error)
    else:  # pragma: no cover - a malformed evidence digest must be rejected
        raise AssertionError("malformed evidence digest must be rejected")


def test_coverage_set_exposes_six_stable_evidence_ids():
    coverage = _coverage()
    assert coverage.evidence_ids == tuple(f"opencode-cov-A:{name}" for name in EVIDENCE_NAMES)
    assert len(coverage.evidence_ids) == 6


# --------------------------------------------------------------------------- #
# Startup — A published, B/D unsupported, hook/version/manifest validation     #
# --------------------------------------------------------------------------- #


def _startup(adapter: OpenCodeAdapter):
    return adapter.startup(
        public_hook_set={"tool.result.after", "session.shutdown"},
        adapter_version=ADAPTER_VERSION,
        capability_manifest_digest=adapter.manifest.manifest_digest,
    )


def test_startup_publishes_a_and_marks_b_d_unsupported():
    adapter = _adapter(transform_fn=lambda v: v)
    outcome = _startup(adapter)
    assert outcome.status == "started"
    assert outcome.result is not None
    assert outcome.result.metadata is not None
    capabilities = {
        row["practice"]: row["capability_state"] for row in outcome.result.metadata["capabilities"]
    }
    assert capabilities == {"A": "full", "B": "unsupported", "D": "unsupported"}
    assert outcome.result.metadata["host"] == OPENCODE_HOST
    assert outcome.result.metadata["host_version"] == HOST_VERSION
    assert outcome.result.metadata["adapter_version"] == ADAPTER_VERSION
    assert outcome.result.metadata["public_hooks"] == ["session.shutdown", "tool.result.after"]


def test_startup_rejects_wrong_adapter_version():
    adapter = _adapter(transform_fn=lambda v: v)
    outcome = adapter.startup(
        public_hook_set={"tool.result.after", "session.shutdown"},
        adapter_version="9.9.9",
        capability_manifest_digest=adapter.manifest.manifest_digest,
    )
    assert outcome.status == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code == "CONTRACT_MISMATCH"


def test_startup_rejects_wrong_manifest_digest():
    adapter = _adapter(transform_fn=lambda v: v)
    outcome = adapter.startup(
        public_hook_set={"tool.result.after", "session.shutdown"},
        adapter_version=ADAPTER_VERSION,
        capability_manifest_digest="0" * 64,
    )
    assert outcome.status == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code == "CONTRACT_MISMATCH"


def test_startup_rejects_unknown_hook():
    adapter = _adapter(transform_fn=lambda v: v)
    outcome = adapter.startup(
        public_hook_set={"tool.result.after", "session.shutdown", "evil.hook"},
        adapter_version=ADAPTER_VERSION,
        capability_manifest_digest=adapter.manifest.manifest_digest,
    )
    assert outcome.status == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code == "REQUEST_SCHEMA_INVALID"


def test_startup_rejects_missing_required_hook():
    adapter = _adapter(transform_fn=lambda v: v)
    outcome = adapter.startup(
        public_hook_set={"tool.result.after"},
        adapter_version=ADAPTER_VERSION,
        capability_manifest_digest=adapter.manifest.manifest_digest,
    )
    assert outcome.status == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code == "ADMISSION_POLICY_DENIED"


# --------------------------------------------------------------------------- #
# Happy path — Practice A relevance filtering                                 #
# --------------------------------------------------------------------------- #


def test_process_a_happy_path_previews_through_neutral_callback():
    seen: list[object] = []

    def shrink(value):
        seen.append(value)
        return f"shrunk:{value}"

    adapter = _adapter(transform_fn=shrink)
    binding = _binding(adapter)
    outcome = adapter.process(binding, neutral_input="oversized tool result", baseline="oversized tool result")

    assert outcome.operation == "previewed"
    assert outcome.refusal is None
    assert outcome.content == "shrunk:oversized tool result"
    assert outcome.content_digest is not None
    assert outcome.coverage_set_id == "opencode-cov-A"
    assert outcome.baseline_preserved is False
    assert outcome.transcript_durable_changed is False
    # The outcome carries the six explicit evidence ids backing the A coverage (BR8.3).
    assert outcome.evidence_ids == adapter._coverage_set.evidence_ids
    assert len(outcome.evidence_ids) == 6
    # The injected callback is the only boundary to the relevance logic.
    assert seen == ["oversized tool result"]


def test_process_a_fallback_preserves_baseline():
    from context_core.runtime import DependencyUnavailable

    def failing(_value):
        raise DependencyUnavailable("reranker_unavailable")

    adapter = _adapter(transform_fn=failing)
    binding = _binding(adapter)
    outcome = adapter.process(binding, neutral_input="x", baseline="the original")

    assert outcome.operation == "baseline"
    assert outcome.baseline_preserved is True
    assert outcome.content == "the original"
    assert outcome.refusal is None
    assert outcome.evidence_ids == adapter._coverage_set.evidence_ids


# --------------------------------------------------------------------------- #
# Stale decision / tampered provenance                                        #
# --------------------------------------------------------------------------- #


def test_process_stale_decision_is_contract_mismatch():
    # A decision minted against manifest revision 1, replayed on an adapter at revision 2: the atomic
    # revalidation invalidates it before any transform runs.
    old_adapter = _adapter(transform_fn=lambda v: v, manifest_revision=1)
    stale_decision = old_adapter.issue_decision("A")

    new_adapter = _adapter(transform_fn=lambda v: v, manifest_revision=2)
    binding = _binding(new_adapter, decision=stale_decision)
    outcome = new_adapter.process(binding, neutral_input="x", baseline="x")

    assert outcome.operation == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code == "CONTRACT_MISMATCH"


def test_process_tampered_provenance_is_contract_mismatch():
    # The binding's explicit provenance must mirror its decision; tampering a single digest refuses
    # before any effect, even though the decision object itself is still valid.
    adapter = _adapter(transform_fn=lambda v: v)
    binding = _binding(adapter)
    tampered = replace(binding, manifest_digest="0" * 64)
    outcome = adapter.process(tampered, neutral_input="x", baseline="x")

    assert outcome.operation == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code == "CONTRACT_MISMATCH"


# --------------------------------------------------------------------------- #
# Authorization precedence over capability (C5)                               #
# --------------------------------------------------------------------------- #


def test_unauthorized_practice_b_is_authorization_denied_not_capability():
    # An unauthorized caller requesting an unsupported practice must get the authorization refusal, never
    # a capability confirmation: authorization is checked before any capability-revealing step (C5).
    adapter = _adapter(transform_fn=lambda v: v)
    binding = _binding(adapter, practice="B", decision=adapter.issue_decision("B"), authorization=False)
    outcome = adapter.process(binding, neutral_input="x", baseline="x")

    assert outcome.operation == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code == "AUTHORIZATION_HOST_DENIED"


# --------------------------------------------------------------------------- #
# B/D unsupported                                                             #
# --------------------------------------------------------------------------- #


def test_process_practice_b_is_capability_unsupported():
    adapter = _adapter(transform_fn=lambda v: v)
    binding = _binding(adapter, practice="B", decision=adapter.issue_decision("B"))
    outcome = adapter.process(binding, neutral_input="x", baseline="x")

    assert outcome.operation == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code == "CAPABILITY_UNSUPPORTED"
    assert outcome.refusal.capability == "B"


def test_process_practice_d_is_capability_unsupported():
    adapter = _adapter(transform_fn=lambda v: v)
    binding = _binding(adapter, practice="D", decision=adapter.issue_decision("D"))
    outcome = adapter.process(binding, neutral_input="x", baseline="x")

    assert outcome.operation == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code == "CAPABILITY_UNSUPPORTED"
    assert outcome.refusal.capability == "D"


# --------------------------------------------------------------------------- #
# deadline_ms                                                                 #
# --------------------------------------------------------------------------- #


def test_non_positive_deadline_is_rejected_without_effect():
    seen: list[object] = []

    def tracked(value):
        seen.append(value)
        return value

    adapter = _adapter(transform_fn=tracked)
    binding = _binding(adapter, deadline_ms=0)
    outcome = adapter.process(binding, neutral_input="x", baseline="x")

    assert outcome.operation == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code == "REQUEST_SCHEMA_INVALID"
    # No transform ran: the expired/invalid deadline is rejected before any effect.
    assert seen == []


def test_positive_deadline_ms_runs_the_transform():
    adapter = _adapter(transform_fn=lambda v: f"shrunk:{v}")
    binding = _binding(adapter, deadline_ms=1)
    outcome = adapter.process(binding, neutral_input="payload", baseline="payload")

    assert outcome.operation == "previewed"
    assert outcome.content == "shrunk:payload"


# --------------------------------------------------------------------------- #
# Closed refusal envelope — exactly seven fields, no payload                  #
# --------------------------------------------------------------------------- #


def test_authorization_denied_closed_refusal_has_exactly_seven_fields():
    adapter = _adapter(transform_fn=lambda v: v)
    binding = _binding(adapter, authorization=False)
    outcome = adapter.process(binding, neutral_input="secret payload", baseline="secret payload")

    assert outcome.operation == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code == "AUTHORIZATION_HOST_DENIED"
    # Exactly the seven closed fields, and no request payload leaked anywhere.
    assert set(outcome.refusal.to_dict()) == _ENVELOPE_FIELDS
    assert outcome.content is None
    assert outcome.content_digest is None
    assert outcome.baseline_preserved is False
    assert outcome.evidence_ids == ()


def test_wrong_host_is_contract_mismatch():
    adapter = _adapter(transform_fn=lambda v: v)
    binding = replace(_binding(adapter), host="claude-code")
    outcome = adapter.process(binding, neutral_input="x", baseline="x")

    assert outcome.operation == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code == "CONTRACT_MISMATCH"


def test_wrong_contract_identity_is_contract_mismatch():
    adapter = _adapter(transform_fn=lambda v: v)
    other = ContractIdentity.of_schema("2.0", {"schema": "other"})
    binding = replace(_binding(adapter), contract_identity=other)
    outcome = adapter.process(binding, neutral_input="x", baseline="x")

    assert outcome.operation == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code == "CONTRACT_MISMATCH"


def test_missing_namespace_is_namespace_invalid():
    adapter = _adapter(transform_fn=lambda v: v)
    binding = _binding(adapter, namespace_id="")
    outcome = adapter.process(binding, neutral_input="x", baseline="x")

    assert outcome.operation == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code == "NAMESPACE_INVALID"


# --------------------------------------------------------------------------- #
# Source already truncated — preserved without transform or reference         #
# --------------------------------------------------------------------------- #


def test_source_already_truncated_preserves_baseline_without_transform():
    calls: list[int] = []

    def tracked(value):
        calls.append(1)
        return value

    adapter = _adapter(transform_fn=tracked)
    binding = _binding(adapter)
    source = OpenCodeSourceAlreadyTruncated(
        source_shape_id="shape-1",
        received_bytes=b"partial bytes from host",
        received_metadata=b"content-type: text/plain",
        origin="host_before_hook",
    )
    outcome = adapter.process(binding, neutral_input=b"ignored", baseline=None, source=source)

    assert outcome.operation == "baseline"
    assert outcome.baseline_preserved is True
    assert outcome.source_shape_id == "shape-1"
    assert outcome.content == b"partial bytes from host"
    assert outcome.content_digest == source.received_bytes_digest
    assert outcome.refusal is None
    assert outcome.evidence_ids == adapter._coverage_set.evidence_ids
    # The transform callback never ran: no filtering of an already-truncated source.
    assert calls == []
    assert source.marker == "source_already_truncated"


def test_source_already_truncated_provenance_has_all_digests_and_no_unreceived_reference():
    source = OpenCodeSourceAlreadyTruncated(
        source_shape_id="shape-2",
        received_bytes=b"abc",
        received_metadata=b"meta",
        origin="upstream_adapter",
    )
    provenance = source.provenance()
    # The four received-content digests and the three preservation flags are present and well-formed.
    for key in (
        "received_bytes_digest",
        "received_metadata_digest",
        "received_shape_digest",
        "origin_digest",
    ):
        assert len(provenance[key]) == 64
    assert provenance["bytes_preserved"] is True
    assert provenance["metadata_preserved"] is True
    assert provenance["shape_preserved"] is True
    assert provenance["marker"] == "source_already_truncated"
    assert provenance["origin"] == "upstream_adapter"
    # No key references content that was never received (BR3.4): nothing hints at absent bytes.
    serialized = " ".join(provenance.keys()).lower()
    assert "unreceived" not in serialized
    assert "missing" not in serialized


# --------------------------------------------------------------------------- #
# Shutdown — synchronous namespace discard                                    #
# --------------------------------------------------------------------------- #


def _shutdown_binding(adapter: OpenCodeAdapter, *, namespace_id, authorization=True, key="sd"):
    return adapter.bind(
        adapter.issue_decision("A"),
        operation="shutdown",
        workspace_id="ws-1",
        session_id="sess-1",
        namespace_id=namespace_id,
        authorization=authorization,
        practice="none",
        idempotency_key=key,
    )


def test_shutdown_discards_namespace_synchronously():
    adapter = _adapter(transform_fn=lambda v: v)
    namespace = adapter.open_namespace(
        "ws-1",
        "sess-1",
        lease_ttl=300.0,
        heartbeat_interval=30.0,
        abandonment_grace=60.0,
        memory_quota=1024,
    )
    binding = _shutdown_binding(adapter, namespace_id=namespace.namespace_id, key="sd-1")
    outcome = adapter.shutdown(binding)

    assert outcome.operation == "discarded"
    assert outcome.refusal is None


def test_shutdown_without_namespace_is_namespace_invalid():
    adapter = _adapter(transform_fn=lambda v: v)
    binding = _shutdown_binding(adapter, namespace_id="", key="sd-2")
    outcome = adapter.shutdown(binding)

    assert outcome.operation == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code == "NAMESPACE_INVALID"
    assert set(outcome.refusal.to_dict()) == _ENVELOPE_FIELDS


def test_shutdown_authorization_denied_is_refused_before_effect():
    adapter = _adapter(transform_fn=lambda v: v)
    namespace = adapter.open_namespace(
        "ws-1",
        "sess-1",
        lease_ttl=300.0,
        heartbeat_interval=30.0,
        abandonment_grace=60.0,
        memory_quota=1024,
    )
    binding = _shutdown_binding(
        adapter, namespace_id=namespace.namespace_id, authorization=False, key="sd-3"
    )
    outcome = adapter.shutdown(binding)

    assert outcome.operation == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code == "AUTHORIZATION_HOST_DENIED"


# --------------------------------------------------------------------------- #
# Wrong entry point                                                           #
# --------------------------------------------------------------------------- #


def test_process_rejects_non_transform_operation():
    adapter = _adapter(transform_fn=lambda v: v)
    binding = _binding(adapter, operation="retrieve")
    outcome = adapter.process(binding, neutral_input="x")

    assert outcome.operation == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code == "REQUEST_SCHEMA_INVALID"


# --------------------------------------------------------------------------- #
# Store / retrieve — baseline preservation and authorized retrieval            #
# --------------------------------------------------------------------------- #


def _open_namespace(adapter: OpenCodeAdapter):
    """Open a memory-only namespace with the approved retention values."""
    return adapter.open_namespace(
        "ws-1",
        "sess-1",
        lease_ttl=300.0,
        heartbeat_interval=30.0,
        abandonment_grace=60.0,
        memory_quota=1 << 20,
    )


def _store_binding(adapter, namespace_id, *, authorization=True, key=None):
    return adapter.bind(
        adapter.issue_decision("A"),
        operation="store",
        workspace_id="ws-1",
        session_id="sess-1",
        namespace_id=namespace_id,
        authorization=authorization,
        practice="A",
        idempotency_key=key,
    )


def _retrieve_binding(adapter, namespace_id, *, authorization=True, deadline_ms=30_000, key=None):
    return adapter.bind(
        adapter.issue_decision("A"),
        operation="retrieve",
        workspace_id="ws-1",
        session_id="sess-1",
        namespace_id=namespace_id,
        authorization=authorization,
        practice="A",
        deadline_ms=deadline_ms,
        idempotency_key=key,
    )


def test_store_baseline_then_retrieve_roundtrips_verbatim():
    adapter = _adapter(transform_fn=lambda v: v)
    namespace = _open_namespace(adapter)

    stored = adapter.store_baseline(
        _store_binding(adapter, namespace.namespace_id, key="st-1"), content=b"the full original result"
    )
    assert stored.operation == "stored"
    assert stored.refusal is None
    assert stored.reference_id is not None

    retrieved = adapter.retrieve(
        _retrieve_binding(adapter, namespace.namespace_id, key="rt-1"), reference_id=stored.reference_id
    )
    assert retrieved.operation == "retrieved"
    assert retrieved.content == b"the full original result"
    assert retrieved.reference_id == stored.reference_id
    assert retrieved.refusal is None


def test_retrieve_unknown_reference_is_closed_refusal():
    adapter = _adapter(transform_fn=lambda v: v)
    namespace = _open_namespace(adapter)
    outcome = adapter.retrieve(
        _retrieve_binding(adapter, namespace.namespace_id, key="rt-2"), reference_id="not-a-real-ref"
    )
    assert outcome.operation == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code == "REFERENCE_EXPIRED_OR_DELETED"
    # A refusal confirms no reference, content or digest.
    assert outcome.reference_id is None
    assert outcome.content is None
    assert set(outcome.refusal.to_dict()) == _ENVELOPE_FIELDS


def test_retrieve_wrong_operation_is_schema_invalid():
    adapter = _adapter(transform_fn=lambda v: v)
    namespace = _open_namespace(adapter)
    # A binding built for transform must not be accepted by the retrieve entry point.
    binding = _binding(adapter, operation="transform", namespace_id=namespace.namespace_id)
    outcome = adapter.retrieve(binding, reference_id="ref")
    assert outcome.operation == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code == "REQUEST_SCHEMA_INVALID"


def test_retrieve_empty_reference_is_schema_invalid():
    adapter = _adapter(transform_fn=lambda v: v)
    namespace = _open_namespace(adapter)
    outcome = adapter.retrieve(_retrieve_binding(adapter, namespace.namespace_id, key="rt-3"), reference_id="")
    assert outcome.operation == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code == "REQUEST_SCHEMA_INVALID"


def test_retrieve_missing_namespace_is_namespace_invalid():
    adapter = _adapter(transform_fn=lambda v: v)
    outcome = adapter.retrieve(_retrieve_binding(adapter, "", key="rt-4"), reference_id="ref")
    assert outcome.operation == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code == "NAMESPACE_INVALID"


def test_retrieve_authorization_denied_before_any_read():
    adapter = _adapter(transform_fn=lambda v: v)
    namespace = _open_namespace(adapter)
    stored = adapter.store_baseline(
        _store_binding(adapter, namespace.namespace_id, key="st-5"), content=b"secret original"
    )
    assert stored.operation == "stored"
    outcome = adapter.retrieve(
        _retrieve_binding(adapter, namespace.namespace_id, authorization=False, key="rt-5"),
        reference_id=stored.reference_id,  # type: ignore[arg-type]
    )
    assert outcome.operation == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code == "AUTHORIZATION_HOST_DENIED"
    assert outcome.content is None


def test_store_authorization_denied_before_any_write():
    adapter = _adapter(transform_fn=lambda v: v)
    namespace = _open_namespace(adapter)
    outcome = adapter.store_baseline(
        _store_binding(adapter, namespace.namespace_id, authorization=False, key="st-6"),
        content=b"original",
    )
    assert outcome.operation == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code == "AUTHORIZATION_HOST_DENIED"
    assert outcome.reference_id is None


def test_retrieve_on_expired_lease_is_closed_refusal():
    clock = _Clock()
    adapter = _adapter(transform_fn=lambda v: v, clock=clock)
    namespace = adapter.open_namespace(
        "ws-1",
        "sess-1",
        lease_ttl=10.0,
        heartbeat_interval=5.0,
        abandonment_grace=1.0,
        memory_quota=1 << 20,
    )
    stored = adapter.store_baseline(
        _store_binding(adapter, namespace.namespace_id, key="st-7"), content=b"original"
    )
    assert stored.operation == "stored"
    # Advance past the lease TTL so the reference can no longer be retrieved.
    clock.advance(100.0)
    outcome = adapter.retrieve(
        _retrieve_binding(adapter, namespace.namespace_id, key="rt-7"),
        reference_id=stored.reference_id,  # type: ignore[arg-type]
    )
    assert outcome.operation == "refused"
    assert outcome.refusal is not None
    assert outcome.refusal.code == "LEASE_EXPIRED"
    assert outcome.content is None


def test_refused_outcome_rejects_a_reference_id():
    # The outcome invariant forbids a refusal that confirms a reference.
    from context_core.contracts.errors import make_refusal
    from context_core.opencode import OpenCodeAdapterOutcome

    try:
        OpenCodeAdapterOutcome(
            outcome_id="o",
            binding_id="b",
            request_id="r",
            operation="refused",
            refusal=make_refusal("REFERENCE_EXPIRED_OR_DELETED", "retrieve", "r", capability="A"),
            reference_id="leaked-ref",
        )
    except ValueError as error:
        assert "reference" in str(error)
    else:  # pragma: no cover - the invariant must reject a reference on a refusal
        raise AssertionError("a refused outcome must not carry a reference_id")


# --------------------------------------------------------------------------- #
# Neutral contract — no agent framework import                                #
# --------------------------------------------------------------------------- #

_PROBE = """
import sys
import context_core.opencode  # noqa: F401

forbidden = ("strands", "langchain", "langgraph", "langchain_core", "boto3", "botocore")
leaked = sorted(
    name
    for name in sys.modules
    for bad in forbidden
    if name == bad or name.startswith(bad + ".")
)
if leaked:
    print("LEAKED:" + ",".join(leaked))
    raise SystemExit(1)
print("CLEAN")
"""


def test_opencode_imports_no_framework_or_aws():
    """Importing the adapter must pull in no agent framework and no AWS SDK."""
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"probe failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    assert "CLEAN" in result.stdout, result.stdout
