"""``NeutralResult`` and ``ProcessingOutcome`` — one terminal per request, success/fallback vs refusal.

A :class:`ProcessingOutcome` carries exactly one terminal ``status``. A success or fallback status carries
a :class:`NeutralResult` and no refusal; ``refused`` carries a :class:`ClosedRefusalEnvelope` and no
result (BR4.9, invariant 4 of the functional spec). ``fallback_original`` is a success-shaped result, not
a refusal: ``result.kind == "baseline_original"`` with ``baseline_preserved`` true.

The constructors below are the only sanctioned way to build an outcome, so the invariants hold by
construction. No framework import, no I/O; the ``committed_at`` clock value is injected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .errors import ClosedRefusalEnvelope

__all__ = [
    "FALLBACK_REASONS",
    "OUTCOME_STATUSES",
    "RESULT_KINDS",
    "SUCCESS_STATUSES",
    "NeutralResult",
    "ProcessingOutcome",
    "fallback_outcome",
    "refused_outcome",
    "success_outcome",
]

OutcomeStatus = Literal[
    "started",
    "healthy",
    "preflight_passed",
    "transformed",
    "stored",
    "retrieved",
    "cataloged",
    "deleted",
    "delegated",
    "fallback_original",
    "timed_out",
    "refused",
]

ResultKind = Literal[
    "startup_ack",
    "health_ack",
    "preflight_ack",
    "transformed_content",
    "stored_reference",
    "retrieved_content",
    "catalog_snapshot",
    "deletion_ack",
    "delegation_ack",
    "baseline_original",
    "timeout_without_payload",
]

FallbackReason = Literal[
    "store_unavailable",
    "reranker_unavailable",
    "codec_unavailable",
    "local_process_unavailable",
    "deadline_before_preview",
]

OUTCOME_STATUSES: frozenset[str] = frozenset(
    {
        "started",
        "healthy",
        "preflight_passed",
        "transformed",
        "stored",
        "retrieved",
        "cataloged",
        "deleted",
        "delegated",
        "fallback_original",
        "timed_out",
        "refused",
    }
)

RESULT_KINDS: frozenset[str] = frozenset(
    {
        "startup_ack",
        "health_ack",
        "preflight_ack",
        "transformed_content",
        "stored_reference",
        "retrieved_content",
        "catalog_snapshot",
        "deletion_ack",
        "delegation_ack",
        "baseline_original",
        "timeout_without_payload",
    }
)

FALLBACK_REASONS: frozenset[str] = frozenset(
    {
        "store_unavailable",
        "reranker_unavailable",
        "codec_unavailable",
        "local_process_unavailable",
        "deadline_before_preview",
    }
)

# Statuses that carry a NeutralResult. ``timed_out`` is excluded: it may or may not carry a baseline,
# handled by its own constructor path, and is not built by ``success_outcome``.
SUCCESS_STATUSES: frozenset[str] = frozenset(
    {
        "started",
        "healthy",
        "preflight_passed",
        "transformed",
        "stored",
        "retrieved",
        "cataloged",
        "deleted",
        "delegated",
    }
)


@dataclass(frozen=True)
class NeutralResult:
    """The success/fallback shape carrying only the authorized result of an operation.

    Attributes:
        kind: One of :data:`RESULT_KINDS`.
        content: The canonical authorized content, when the kind carries one.
        metadata: Functional metadata, when present.
        content_digest: SHA-256 of the content, when present.
        reference_id: A reference id — present only when a reference was created or is already valid.
        baseline_source: ``"original_tool_result"`` for a baseline, ``"none"`` otherwise.
        fallback_reason: Why a fallback happened, for ``baseline_original`` only.
    """

    kind: ResultKind
    content: Any = None
    metadata: dict[str, Any] | None = None
    content_digest: str | None = None
    reference_id: str | None = None
    baseline_source: Literal["original_tool_result", "none"] | None = None
    fallback_reason: FallbackReason | None = None

    def __post_init__(self) -> None:
        """Reject an unknown kind.

        Raises:
            ValueError: If ``kind`` is outside :data:`RESULT_KINDS`.
        """
        if self.kind not in RESULT_KINDS:
            raise ValueError(f"unknown result kind: {self.kind!r}")


@dataclass(frozen=True)
class ProcessingOutcome:
    """The single terminal result of a request.

    Build one via :func:`success_outcome`, :func:`fallback_outcome` or :func:`refused_outcome` so the
    success-vs-refusal invariant holds by construction.

    Attributes:
        request_id: The request this outcome belongs to.
        status: The single terminal status, one of :data:`OUTCOME_STATUSES`.
        result: The neutral result, for success/fallback statuses; ``None`` for ``refused``.
        refusal: The closed envelope, for ``refused``; ``None`` otherwise.
        baseline_preserved: ``True`` when a baseline original was preserved.
        terminal_revision: Positive integer terminal revision.
        committed_at: Commit time (injected clock value).
    """

    request_id: str
    status: OutcomeStatus
    committed_at: float
    result: NeutralResult | None = None
    refusal: ClosedRefusalEnvelope | None = None
    baseline_preserved: bool = False
    terminal_revision: int = 1

    def __post_init__(self) -> None:
        """Enforce the terminal invariants that are not already guaranteed by the constructors.

        Raises:
            ValueError: If ``status`` is unknown, or the result/refusal pairing contradicts the status.
        """
        if self.status not in OUTCOME_STATUSES:
            raise ValueError(f"unknown outcome status: {self.status!r}")
        if self.status == "refused":
            if self.refusal is None or self.result is not None:
                raise ValueError("a refused outcome must carry a refusal and no result")
        else:
            if self.refusal is not None:
                raise ValueError(f"status {self.status!r} must not carry a refusal")
        if self.status == "fallback_original":
            if self.result is None or self.result.kind != "baseline_original":
                raise ValueError("fallback_original requires a baseline_original result")
            if not self.baseline_preserved:
                raise ValueError("fallback_original requires baseline_preserved=True")


def success_outcome(
    request_id: str,
    status: str,
    result: NeutralResult,
    *,
    committed_at: float,
    terminal_revision: int = 1,
) -> ProcessingOutcome:
    """Build a success outcome carrying ``result`` and no refusal.

    Args:
        request_id: The request id.
        status: A success status in :data:`SUCCESS_STATUSES`.
        result: The neutral result to carry.
        committed_at: Commit time (injected clock value).
        terminal_revision: The terminal revision. Defaults to 1.

    Returns:
        The terminal success outcome.

    Raises:
        ValueError: If ``status`` is not a success status.
    """
    if status not in SUCCESS_STATUSES:
        raise ValueError(f"{status!r} is not a success status")
    return ProcessingOutcome(
        request_id=request_id,
        status=status,  # type: ignore[arg-type]
        committed_at=committed_at,
        result=result,
        terminal_revision=terminal_revision,
    )


def fallback_outcome(
    request_id: str,
    fallback_reason: str,
    content: Any,
    *,
    committed_at: float,
    metadata: dict[str, Any] | None = None,
    content_digest: str | None = None,
    terminal_revision: int = 1,
) -> ProcessingOutcome:
    """Build a ``fallback_original`` outcome preserving the original content.

    Args:
        request_id: The request id.
        fallback_reason: Why the fallback happened; must be in :data:`FALLBACK_REASONS`.
        content: The preserved original content.
        committed_at: Commit time (injected clock value).
        metadata: Preserved functional metadata.
        content_digest: SHA-256 of the content, when known.
        terminal_revision: The terminal revision. Defaults to 1.

    Returns:
        The terminal fallback outcome.

    Raises:
        ValueError: If ``fallback_reason`` is unknown.
    """
    if fallback_reason not in FALLBACK_REASONS:
        raise ValueError(f"unknown fallback reason: {fallback_reason!r}")
    result = NeutralResult(
        kind="baseline_original",
        content=content,
        metadata=metadata,
        content_digest=content_digest,
        baseline_source="original_tool_result",
        fallback_reason=fallback_reason,  # type: ignore[arg-type]
    )
    return ProcessingOutcome(
        request_id=request_id,
        status="fallback_original",
        committed_at=committed_at,
        result=result,
        baseline_preserved=True,
        terminal_revision=terminal_revision,
    )


def refused_outcome(
    request_id: str,
    refusal: ClosedRefusalEnvelope,
    *,
    committed_at: float,
    terminal_revision: int = 1,
) -> ProcessingOutcome:
    """Build a ``refused`` outcome carrying ``refusal`` and no result.

    Args:
        request_id: The request id.
        refusal: The closed refusal envelope.
        committed_at: Commit time (injected clock value).
        terminal_revision: The terminal revision. Defaults to 1.

    Returns:
        The terminal refused outcome.
    """
    return ProcessingOutcome(
        request_id=request_id,
        status="refused",
        committed_at=committed_at,
        refusal=refusal,
        terminal_revision=terminal_revision,
    )
