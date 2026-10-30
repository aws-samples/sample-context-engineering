"""U1 ``shared-context-contracts`` — the neutral contract layer shared by every host adapter and runtime.

This subpackage holds the framework-agnostic contracts and behaviour common to adapters, runtime,
capabilities, admission and references: contract identity, the closed operation/error vocabulary,
capability publication with a decision snapshot whose lifecycle is separate from its state, the closed
refusal envelope and neutral outcome, offline/replay/live execution admission, in-memory exactly-once
idempotency, and memory-only namespaced references with leases and named delegation.

Everything here is stdlib-only: no ``strands``/``langchain``/``langgraph`` and no ``boto3``/network. All
clocks are injected and all retention values are required parameters (no persistence defaults).
"""

from .canonical import canonical_bytes, canonical_json, digest
from .capability import (
    CAPABILITY_STATES,
    DECISION_LIFECYCLES,
    IMPLEMENTATION_KINDS,
    PRACTICES,
    CapabilityDecision,
    CapabilityDeclaration,
    CapabilityManifest,
    CoverageSet,
    EvidenceDigest,
    ensure_usable,
    issue_decision,
    revalidate,
)
from .errors import (
    ADMISSION_OPERATIONS,
    CAPABILITIES,
    CATALOG,
    ERROR_CODES,
    OPERATIONS,
    REFUSAL_CATEGORIES,
    RUNTIME_OPERATIONS,
    ClosedRefusalEnvelope,
    RefusalError,
    make_refusal,
)
from .execution import (
    BACKEND_IDS,
    EXECUTION_MODES,
    AdmissionPolicy,
    Approval,
    ExecutionAdmission,
    ExecutionBudget,
    ExecutionPreflight,
    ExecutionRequest,
    PreflightResult,
    admit,
)
from .idempotency import IdempotencyStore, Reservation, ReservationState
from .identity import ContractIdentity, require_compatible
from .outcome import (
    FALLBACK_REASONS,
    OUTCOME_STATUSES,
    RESULT_KINDS,
    SUCCESS_STATUSES,
    NeutralResult,
    ProcessingOutcome,
    fallback_outcome,
    refused_outcome,
    success_outcome,
)
from .references import (
    AuthorizationContext,
    ContextReference,
    DelegationGrant,
    InMemoryContextStore,
    RetentionLease,
    SessionNamespace,
    allow_all,
)

__all__ = [
    "ADMISSION_OPERATIONS",
    "BACKEND_IDS",
    "CAPABILITIES",
    "CAPABILITY_STATES",
    "CATALOG",
    "DECISION_LIFECYCLES",
    "ERROR_CODES",
    "EXECUTION_MODES",
    "FALLBACK_REASONS",
    "IMPLEMENTATION_KINDS",
    "OPERATIONS",
    "OUTCOME_STATUSES",
    "PRACTICES",
    "REFUSAL_CATEGORIES",
    "RESULT_KINDS",
    "RUNTIME_OPERATIONS",
    "SUCCESS_STATUSES",
    "AdmissionPolicy",
    "Approval",
    "AuthorizationContext",
    "CapabilityDecision",
    "CapabilityDeclaration",
    "CapabilityManifest",
    "ClosedRefusalEnvelope",
    "ContextReference",
    "ContractIdentity",
    "CoverageSet",
    "DelegationGrant",
    "EvidenceDigest",
    "ExecutionAdmission",
    "ExecutionBudget",
    "ExecutionPreflight",
    "ExecutionRequest",
    "IdempotencyStore",
    "InMemoryContextStore",
    "NeutralResult",
    "PreflightResult",
    "ProcessingOutcome",
    "RefusalError",
    "Reservation",
    "ReservationState",
    "RetentionLease",
    "SessionNamespace",
    "admit",
    "allow_all",
    "canonical_bytes",
    "canonical_json",
    "digest",
    "ensure_usable",
    "fallback_outcome",
    "issue_decision",
    "make_refusal",
    "refused_outcome",
    "require_compatible",
    "revalidate",
    "success_outcome",
]
