"""In-memory exactly-once idempotency — reserve, commit, replay and conflict under one lock.

A request with a durable effect is keyed by ``(idempotency_scope, idempotency_key)`` and bound to its
``request_digest``. The first arrival *reserves* the key; a matching retry *replays* the committed
outcome byte-for-byte; a retry with a different digest *conflicts* (``IDEMPOTENCY_CONFLICT``); a reserved
but not-yet-committed key is *pending*. Reserve, durable effect and commit form one atomic boundary: an
uncommitted reservation leaves no durable effect, so a timeout that fires before commit cancels cleanly
and never admits a second effect (BR5.1, BR5.4, BR5.5).

Thread-safe via a single lock. No framework import, no I/O; the retention clock is injected.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from .errors import RefusalError
from .outcome import ProcessingOutcome

__all__ = [
    "IdempotencyStore",
    "Reservation",
    "ReservationState",
]


class ReservationState(Enum):
    """The outcome of a :meth:`IdempotencyStore.reserve` call."""

    RESERVED = "reserved"
    """The key was free and is now reserved for this request; the caller may run the effect."""

    REPLAY = "replay"
    """The key is committed with the same digest; the committed outcome is returned, no re-run."""

    PENDING = "pending"
    """The key is reserved with the same digest by an in-flight request; the caller must wait, not re-run."""


@dataclass(frozen=True)
class Reservation:
    """The result of a reservation attempt.

    Attributes:
        state: What the caller should do next.
        outcome: The committed outcome, present only when ``state`` is :attr:`ReservationState.REPLAY`.
    """

    state: ReservationState
    outcome: ProcessingOutcome | None = None


@dataclass
class _Record:
    """Internal mutable record for one idempotency key."""

    request_digest: str
    request_id: str
    deduplication_state: Literal["reserved", "committed"]
    reserved_at: float
    expires_at: float
    outcome: ProcessingOutcome | None = None
    committed_at: float | None = None
    reservation_revision: int = 1


class IdempotencyStore:
    """An in-memory exactly-once store keyed by ``(scope, key)``.

    The store holds reservations and committed outcomes for the retention window. All transitions are
    serialised by one lock, so a reserve/commit race resolves to a single committed outcome.

    Args:
        retention_seconds: How long a committed/reserved record is kept before an arrival may re-reserve
            an expired key. Must be positive.
    """

    def __init__(self, retention_seconds: float) -> None:
        """Initialise an empty store.

        Args:
            retention_seconds: Retention window for a record, in seconds. Must be positive.

        Raises:
            ValueError: If ``retention_seconds`` is not positive.
        """
        if retention_seconds <= 0:
            raise ValueError("retention_seconds must be positive")
        self._retention_seconds = retention_seconds
        self._records: dict[tuple[str, str], _Record] = {}
        self._lock = threading.Lock()

    def reserve(
        self,
        scope: str,
        key: str,
        request_digest: str,
        request_id: str,
        operation: str,
        now: float,
        *,
        correlation_id: str | None = None,
    ) -> Reservation:
        """Reserve ``(scope, key)`` for ``request_digest``, or replay/conflict/pending.

        Args:
            scope: The idempotency scope.
            key: The idempotency key.
            request_digest: The canonical digest of the request.
            request_id: The request id reserving the key.
            operation: The operation, used only if a conflict refusal is raised.
            now: The current clock value, in seconds.
            correlation_id: Correlation id for a conflict refusal; defaults to ``request_id``.

        Returns:
            A :class:`Reservation` describing what the caller should do next.

        Raises:
            RefusalError: ``IDEMPOTENCY_CONFLICT`` when the key exists (reserved or committed) with a
                different digest. The existing record is preserved and no new effect is created.
        """
        with self._lock:
            record = self._records.get((scope, key))
            if record is not None and record.expires_at <= now:
                # The retention window closed; the key is free again.
                record = None
                del self._records[(scope, key)]

            if record is None:
                self._records[(scope, key)] = _Record(
                    request_digest=request_digest,
                    request_id=request_id,
                    deduplication_state="reserved",
                    reserved_at=now,
                    expires_at=now + self._retention_seconds,
                )
                return Reservation(ReservationState.RESERVED)

            if record.request_digest != request_digest:
                raise RefusalError.of(
                    "IDEMPOTENCY_CONFLICT", operation, correlation_id or request_id
                )

            if record.deduplication_state == "committed":
                if record.outcome is None:
                    raise ValueError("committed record is missing its outcome")
                return Reservation(ReservationState.REPLAY, outcome=record.outcome)

            return Reservation(ReservationState.PENDING)

    def commit(
        self,
        scope: str,
        key: str,
        request_digest: str,
        outcome: ProcessingOutcome,
        now: float,
    ) -> ProcessingOutcome:
        """Atomically convert the reservation for ``(scope, key)`` into a committed outcome.

        Idempotent against itself: committing twice with the same digest returns the already-committed
        outcome rather than overwriting it, so a late duplicate cannot create a second effect.

        Args:
            scope: The idempotency scope.
            key: The idempotency key.
            request_digest: The canonical digest of the request; must match the reservation.
            outcome: The terminal outcome to commit.
            now: The current clock value, in seconds.

        Returns:
            The committed outcome (the existing one if already committed).

        Raises:
            ValueError: If there is no reservation for the key, or the digest does not match it. These
                are programming errors in the boundary, not runtime refusals.
        """
        with self._lock:
            record = self._records.get((scope, key))
            if record is None:
                raise ValueError(f"no reservation to commit for ({scope!r}, {key!r})")
            if record.request_digest != request_digest:
                raise ValueError("commit digest does not match the reservation digest")
            if record.deduplication_state == "committed":
                if record.outcome is None:
                    raise ValueError("committed record is missing its outcome")
                return record.outcome
            record.deduplication_state = "committed"
            record.outcome = outcome
            record.committed_at = now
            return outcome

    def cancel(self, scope: str, key: str) -> None:
        """Drop an *uncommitted* reservation for ``(scope, key)``, leaving no durable effect.

        A committed record is left untouched: once committed, the outcome is terminal and replayable.
        This is the timeout path — a reservation whose deadline passes before commit is cancelled.

        Args:
            scope: The idempotency scope.
            key: The idempotency key.
        """
        with self._lock:
            record = self._records.get((scope, key))
            if record is not None and record.deduplication_state == "reserved":
                del self._records[(scope, key)]
