"""The closed refusal envelope, the stable error catalog, and the closed operation/capability vocabulary.

Every deterministic failure in U1 becomes a :class:`ClosedRefusalEnvelope` carrying only the seven
allow-listed fields — nothing else may appear, in particular no payload, reference, size, hash,
namespace, free-text message, or authorization rule (NFR4, NFR9). The catalog in :data:`CATALOG` fixes,
for each stable code, its single category and its retry policy; a code belongs to exactly one category.

:class:`RefusalError` is the internal control-flow carrier: a deterministic precondition raises it, the
boundary catches it and reads :attr:`RefusalError.envelope`. No framework import, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "ADMISSION_OPERATIONS",
    "CAPABILITIES",
    "CATALOG",
    "ERROR_CODES",
    "OPERATIONS",
    "REFUSAL_CATEGORIES",
    "RUNTIME_OPERATIONS",
    "Capability",
    "ClosedRefusalEnvelope",
    "ErrorCode",
    "Operation",
    "RefusalCategory",
    "RefusalError",
    "make_refusal",
]

Operation = Literal[
    "startup",
    "health",
    "preflight",
    "transform",
    "store",
    "retrieve",
    "delete",
    "catalog",
    "delegate",
    "shutdown",
]
"""The closed operation vocabulary, shared by requests and refusals."""

ADMISSION_OPERATIONS: frozenset[str] = frozenset({"startup", "health", "preflight"})
"""Operations in the ``admission`` family: they neither transform nor carry payload."""

RUNTIME_OPERATIONS: frozenset[str] = frozenset(
    {"transform", "store", "retrieve", "delete", "catalog", "delegate", "shutdown"}
)
"""Operations in the ``runtime`` family: they use namespace, authorization and lifecycle."""

OPERATIONS: frozenset[str] = ADMISSION_OPERATIONS | RUNTIME_OPERATIONS
"""The whole closed operation set."""

Capability = Literal["A", "B", "D", "none"]
"""The three practices, plus ``none`` for a refusal not tied to a practice."""

CAPABILITIES: frozenset[str] = frozenset({"A", "B", "D", "none"})
"""The closed capability vocabulary used by the refusal envelope."""

ErrorCode = Literal[
    "AUTHORIZATION_HOST_DENIED",
    "NAMESPACE_INVALID",
    "CAPABILITY_UNSUPPORTED",
    "REFERENCE_EXPIRED_OR_DELETED",
    "DEPENDENCY_UNAVAILABLE_NO_FALLBACK",
    "REQUEST_SCHEMA_INVALID",
    "QUOTA_EXCEEDED",
    "CONTRACT_MISMATCH",
    "IDEMPOTENCY_CONFLICT",
    "LEASE_EXPIRED",
    "ADMISSION_POLICY_DENIED",
]
"""The closed set of stable error codes."""

RefusalCategory = Literal[
    "authorization_denied",
    "namespace_denied",
    "capability_unsupported",
    "expired_or_deleted",
    "dependency_unavailable",
    "invalid_request",
]
"""The closed set of refusal categories."""

REFUSAL_CATEGORIES: frozenset[str] = frozenset(
    {
        "authorization_denied",
        "namespace_denied",
        "capability_unsupported",
        "expired_or_deleted",
        "dependency_unavailable",
        "invalid_request",
    }
)
"""The closed set of refusal categories, as a set for membership checks."""


@dataclass(frozen=True)
class _CatalogEntry:
    """One row of the stable error catalog: a code bound to its category and retry policy.

    Attributes:
        category: The single category this code belongs to.
        retryable: Whether an automatic retry is ever permitted for this code. Every current code is
            non-retryable; the field exists so the catalog is the one source of that fact.
    """

    category: RefusalCategory
    retryable: bool


CATALOG: dict[str, _CatalogEntry] = {
    "AUTHORIZATION_HOST_DENIED": _CatalogEntry("authorization_denied", False),
    "NAMESPACE_INVALID": _CatalogEntry("namespace_denied", False),
    "CAPABILITY_UNSUPPORTED": _CatalogEntry("capability_unsupported", False),
    "REFERENCE_EXPIRED_OR_DELETED": _CatalogEntry("expired_or_deleted", False),
    "DEPENDENCY_UNAVAILABLE_NO_FALLBACK": _CatalogEntry("dependency_unavailable", False),
    "REQUEST_SCHEMA_INVALID": _CatalogEntry("invalid_request", False),
    "QUOTA_EXCEEDED": _CatalogEntry("invalid_request", False),
    "CONTRACT_MISMATCH": _CatalogEntry("invalid_request", False),
    "IDEMPOTENCY_CONFLICT": _CatalogEntry("invalid_request", False),
    "LEASE_EXPIRED": _CatalogEntry("expired_or_deleted", False),
    "ADMISSION_POLICY_DENIED": _CatalogEntry("invalid_request", False),
}
"""The stable catalog ``1.0``: code -> (category, retryable). The keys are the whole code set."""

ERROR_CODES: frozenset[str] = frozenset(CATALOG)
"""The closed set of stable error codes, as a set for membership checks."""

# The refusal envelope's field set is closed; this is the exact, complete list.
_ENVELOPE_FIELDS: frozenset[str] = frozenset(
    {"code", "category", "operation", "capability", "status", "correlation_id", "retryable"}
)


@dataclass(frozen=True)
class ClosedRefusalEnvelope:
    """A redacted, closed refusal — the only shape a deterministic failure takes.

    The field set is exactly the seven attributes below and nothing more (BR4.9, NFR4.1). No payload,
    reference, existence hint, size, hash, workspace/session/namespace id, free-text message or
    authorization rule may appear: that prohibition is what keeps a refusal from leaking.

    Attributes:
        code: The stable error code; must be a key of :data:`CATALOG`.
        category: The code's category, taken from the catalog — never chosen independently.
        operation: The operation that was refused; must be in :data:`OPERATIONS`.
        capability: The practice in play, or ``"none"``.
        status: Always the literal ``"refused"``.
        correlation_id: An opaque id for correlation; it must not encode a namespace.
        retryable: The code's retry policy, taken from the catalog.
    """

    code: ErrorCode
    category: RefusalCategory
    operation: Operation
    capability: Capability
    correlation_id: str
    retryable: bool
    status: Literal["refused"] = "refused"

    def to_dict(self) -> dict[str, object]:
        """Return the envelope as a plain dict carrying exactly the seven closed fields.

        Returns:
            A dict with keys ``code``, ``category``, ``operation``, ``capability``, ``status``,
            ``correlation_id`` and ``retryable`` — and no others.
        """
        return {
            "code": self.code,
            "category": self.category,
            "operation": self.operation,
            "capability": self.capability,
            "status": self.status,
            "correlation_id": self.correlation_id,
            "retryable": self.retryable,
        }


def make_refusal(
    code: str,
    operation: str,
    correlation_id: str,
    *,
    capability: str = "none",
) -> ClosedRefusalEnvelope:
    """Build a :class:`ClosedRefusalEnvelope`, deriving category and retry from the catalog.

    The caller names only the code, the operation, the correlation id and (optionally) the practice;
    the category and retry policy are read from :data:`CATALOG` so a code can never be paired with the
    wrong category.

    Args:
        code: A stable error code; must be in :data:`ERROR_CODES`.
        operation: The refused operation; must be in :data:`OPERATIONS`.
        correlation_id: An opaque correlation id.
        capability: The practice (``"A"``/``"B"``/``"D"``) or ``"none"``. Defaults to ``"none"``.

    Returns:
        The closed refusal envelope.

    Raises:
        ValueError: If ``code``, ``operation`` or ``capability`` is outside its closed set. A refusal
            built on an unknown code would itself be an invalid refusal, so this is a programming error
            rather than a runtime refusal.
    """
    if code not in ERROR_CODES:
        raise ValueError(f"unknown stable error code: {code!r}")
    if operation not in OPERATIONS:
        raise ValueError(f"unknown operation: {operation!r}")
    if capability not in CAPABILITIES:
        raise ValueError(f"unknown capability: {capability!r}")

    entry = CATALOG[code]
    return ClosedRefusalEnvelope(
        code=code,  # type: ignore[arg-type]
        category=entry.category,
        operation=operation,  # type: ignore[arg-type]
        capability=capability,  # type: ignore[arg-type]
        correlation_id=correlation_id,
        retryable=entry.retryable,
    )


class RefusalError(Exception):
    """A deterministic precondition failure, carrying the closed envelope it maps to.

    Raised inside the contract layer and caught at the boundary, which reads :attr:`envelope`. The
    exception message is deliberately only the code — the detail belongs in no observable surface.

    Attributes:
        envelope: The closed refusal envelope for this failure.
    """

    def __init__(self, envelope: ClosedRefusalEnvelope) -> None:
        """Wrap ``envelope`` as a raisable error.

        Args:
            envelope: The closed refusal to carry.
        """
        super().__init__(envelope.code)
        self.envelope = envelope

    @classmethod
    def of(
        cls,
        code: str,
        operation: str,
        correlation_id: str,
        *,
        capability: str = "none",
    ) -> RefusalError:
        """Build a :class:`RefusalError` from a code and operation, via :func:`make_refusal`.

        Args:
            code: A stable error code.
            operation: The refused operation.
            correlation_id: An opaque correlation id.
            capability: The practice or ``"none"``.

        Returns:
            A ready-to-raise refusal error.
        """
        return cls(make_refusal(code, operation, correlation_id, capability=capability))
