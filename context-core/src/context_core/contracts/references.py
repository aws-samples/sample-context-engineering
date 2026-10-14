"""In-memory namespaces, a configurable retention lease, opaque references and named delegation grants.

The store here is memory-only by design: file persistence is opt-in and stays disabled in this slice
(BR4.5). Every retention parameter — ``lease_ttl``, ``heartbeat_interval``, ``abandonment_grace``,
``memory_quota`` — is a *required* argument with no invented default, because the concrete numeric values
are deferred to NFR Requirements (BR4.8). A reference id is opaque and unguessable; every reference
operation revalidates namespace, host authorization and lease before touching content (BR4.1, BR4.4).

Authorization is a host-supplied callback: the store never decides policy, it only enforces the host's
decision. No framework import, no network, no AWS; the clock is injected.
"""

from __future__ import annotations

import hashlib
import secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from .errors import RefusalError

__all__ = [
    "AuthorizationContext",
    "Authorize",
    "ContextReference",
    "DelegationGrant",
    "InMemoryContextStore",
    "RetentionLease",
    "SessionNamespace",
    "allow_all",
]

LeaseState = Literal["active", "abandoned", "expired", "revoked", "discarded"]
ReferenceLifecycle = Literal["active", "delegated", "deleted", "expired"]
GrantStatus = Literal["active", "revoked", "expired"]


@dataclass(frozen=True)
class SessionNamespace:
    """A workspace/session/subagent namespace used by every reference.

    Creating a namespace grants no authorization by itself; a subagent is isolated by default.

    Attributes:
        namespace_id: Opaque identity of the namespace.
        workspace_id: Workspace identity.
        session_id: Session identity.
        subagent_id: Subagent identity, when the namespace belongs to a subagent.
        parent_session_id: Parent session, when the namespace is a subagent of another.
    """

    namespace_id: str
    workspace_id: str
    session_id: str
    subagent_id: str | None = None
    parent_session_id: str | None = None


@dataclass
class RetentionLease:
    """A retention lease: quota, heartbeat window and discard boundary of a namespace.

    All durations are in seconds and all are required. The lease evaluates its own state against the
    clock: a missed heartbeat window moves it to ``abandoned``, and reaching ``expires_at`` moves it to
    ``expired``. ``lease_revision`` and ``fencing_token`` fence concurrent writers.

    Attributes:
        lease_id: Opaque identity of the lease.
        namespace_id: The namespace this lease belongs to.
        lease_state: Current state.
        issued_at: Issue time.
        expires_at: Absolute expiry time.
        lease_ttl: Lease time-to-live, in seconds.
        heartbeat_interval: Maximum gap between heartbeats before abandonment, in seconds.
        abandonment_grace: Grace after abandonment before forced expiry, in seconds.
        memory_quota: Hard in-memory byte quota for the namespace.
        file_quota: Hard file byte quota; recorded but unused while persistence is disabled.
        last_heartbeat_at: Time of the last accepted heartbeat.
        abandoned_at: When abandonment began — the moment the heartbeat window first lapsed. ``None``
            while the lease is active. It is derived on the first ``evaluate`` that observes a missed
            heartbeat (and may be injected for tests); the abandonment grace runs from this instant.
        lease_revision: Monotonic revision, bumped on each accepted heartbeat.
        fencing_token: Opaque fencing token.
    """

    lease_id: str
    namespace_id: str
    issued_at: float
    expires_at: float
    lease_ttl: float
    heartbeat_interval: float
    abandonment_grace: float
    memory_quota: int
    file_quota: int
    lease_state: LeaseState = "active"
    last_heartbeat_at: float | None = None
    abandoned_at: float | None = None
    lease_revision: int = 1
    fencing_token: str = field(default_factory=lambda: secrets.token_hex(16))

    def evaluate(self, now: float) -> LeaseState:
        """Advance and return the lease state at ``now``.

        A terminal state (``expired``, ``revoked``, ``discarded``) is never walked back. Otherwise:

        - Reaching ``expires_at`` (the lease TTL boundary) moves the lease to ``expired``.
        - A missed heartbeat window moves the lease to ``abandoned`` and pins ``abandoned_at`` to the
          instant abandonment began (the end of the last heartbeat window), not to ``now``.
        - Once abandoned, reaching ``abandoned_at + abandonment_grace`` moves the lease to ``expired``.
          After that boundary the lease can never be renewed again (BR4 retention).

        Args:
            now: The current clock value, in seconds.

        Returns:
            The lease state after evaluation.
        """
        if self.lease_state in ("expired", "revoked", "discarded"):
            return self.lease_state
        if now >= self.expires_at:
            self.lease_state = "expired"
            return self.lease_state
        reference_point = self.last_heartbeat_at if self.last_heartbeat_at is not None else self.issued_at
        if now - reference_point > self.heartbeat_interval:
            if self.abandoned_at is None:
                # Abandonment began when the heartbeat window lapsed, not at the current evaluation.
                self.abandoned_at = reference_point + self.heartbeat_interval
            self.lease_state = "abandoned"
            if now >= self.abandoned_at + self.abandonment_grace:
                self.lease_state = "expired"
        return self.lease_state


@dataclass
class _Reference:
    """Internal mutable reference record. Content lives only in memory."""

    reference_id: str
    namespace_id: str
    content: bytes
    content_type: str
    content_digest: str
    lifecycle_state: ReferenceLifecycle = "active"
    reference_revision: int = 1


@dataclass
class _Link:
    """Internal link between a grant and one named reference."""

    reference_id: str
    link_status: GrantStatus = "active"


@dataclass
class DelegationGrant:
    """An explicit, specific, auditable grant of named references to another namespace.

    Attributes:
        grant_id: Opaque identity of the grant.
        issuer_namespace_id: The namespace issuing the grant.
        recipient_namespace_id: The namespace receiving access.
        reference_ids: The named references the grant covers.
        status: Grant status.
        issued_at: Issue time.
        expires_at: Expiry time.
        grant_revision: Monotonic revision.
    """

    grant_id: str
    issuer_namespace_id: str
    recipient_namespace_id: str
    reference_ids: tuple[str, ...]
    issued_at: float
    expires_at: float
    status: GrantStatus = "active"
    grant_revision: int = 1


@dataclass(frozen=True)
class ContextReference:
    """A public, read-only view of a stored reference returned to a caller.

    Attributes:
        reference_id: The opaque, unguessable id.
        namespace_id: The owning namespace.
        content_digest: SHA-256 of the stored content.
        lifecycle_state: Current lifecycle state.
        reference_revision: Current revision.
    """

    reference_id: str
    namespace_id: str
    content_digest: str
    lifecycle_state: ReferenceLifecycle
    reference_revision: int


@dataclass(frozen=True)
class AuthorizationContext:
    """What the store hands the host's authorization callback.

    Attributes:
        operation: The reference operation being authorized.
        namespace_id: The acting namespace.
        reference_id: The target reference, when the operation names one.
        recipient_namespace_id: The delegation recipient, for ``delegate``.
    """

    operation: str
    namespace_id: str
    reference_id: str | None = None
    recipient_namespace_id: str | None = None


Authorize = Callable[[AuthorizationContext], bool]
"""A host policy callback: returns ``True`` to allow the operation, ``False`` to deny it."""


def allow_all(_context: AuthorizationContext) -> bool:
    """A permissive authorization callback that allows every operation.

    Args:
        _context: The authorization context (ignored).

    Returns:
        Always ``True``.
    """
    return True


class InMemoryContextStore:
    """A memory-only reference store with host authorization, leases and named delegation.

    File persistence is intentionally not implemented in this slice: content is held in memory only, so
    nothing survives a restart. Every retention value is supplied per namespace via
    :meth:`open_namespace`; none is defaulted.

    Args:
        authorize: Host policy callback. Defaults to :func:`allow_all`.
        clock: Zero-argument callable returning the current time in seconds. Defaults to
            ``time.monotonic``.
    """

    def __init__(self, authorize: Authorize | None = None, clock: Callable[[], float] | None = None) -> None:
        """Initialise an empty store.

        Args:
            authorize: Host policy callback. Defaults to :func:`allow_all`.
            clock: Current-time source in seconds. Defaults to ``time.monotonic``.
        """
        import time

        self._authorize = authorize or allow_all
        self._clock = clock or time.monotonic
        self._namespaces: dict[str, SessionNamespace] = {}
        self._leases: dict[str, RetentionLease] = {}
        self._references: dict[str, _Reference] = {}
        self._grants: dict[str, DelegationGrant] = {}
        self._links: dict[str, list[_Link]] = {}
        self._memory_used: dict[str, int] = {}
        self._lock = threading.RLock()

    def open_namespace(
        self,
        workspace_id: str,
        session_id: str,
        *,
        lease_ttl: float,
        heartbeat_interval: float,
        abandonment_grace: float,
        memory_quota: int,
        file_quota: int,
        subagent_id: str | None = None,
        parent_session_id: str | None = None,
    ) -> SessionNamespace:
        """Create a namespace and its active lease.

        Every retention parameter is required; there is no persistence default to invent.

        Args:
            workspace_id: Workspace identity.
            session_id: Session identity.
            lease_ttl: Lease time-to-live, in seconds. Must be positive.
            heartbeat_interval: Heartbeat window, in seconds. Must be positive.
            abandonment_grace: Grace after abandonment, in seconds. Must be non-negative.
            memory_quota: Hard in-memory byte quota. Must be non-negative.
            file_quota: Hard file byte quota (recorded only; persistence disabled). Must be non-negative.
            subagent_id: Subagent identity, when applicable.
            parent_session_id: Parent session, when applicable.

        Returns:
            The created namespace.

        Raises:
            ValueError: If any numeric parameter is out of range.
        """
        if lease_ttl <= 0 or heartbeat_interval <= 0:
            raise ValueError("lease_ttl and heartbeat_interval must be positive")
        if abandonment_grace < 0 or memory_quota < 0 or file_quota < 0:
            raise ValueError("abandonment_grace, memory_quota and file_quota must be non-negative")

        now = self._clock()
        with self._lock:
            namespace = SessionNamespace(
                namespace_id=secrets.token_hex(16),
                workspace_id=workspace_id,
                session_id=session_id,
                subagent_id=subagent_id,
                parent_session_id=parent_session_id,
            )
            lease = RetentionLease(
                lease_id=secrets.token_hex(16),
                namespace_id=namespace.namespace_id,
                issued_at=now,
                expires_at=now + lease_ttl,
                lease_ttl=lease_ttl,
                heartbeat_interval=heartbeat_interval,
                abandonment_grace=abandonment_grace,
                memory_quota=memory_quota,
                file_quota=file_quota,
                last_heartbeat_at=now,
            )
            self._namespaces[namespace.namespace_id] = namespace
            self._leases[namespace.namespace_id] = lease
            self._memory_used[namespace.namespace_id] = 0
            return namespace

    def heartbeat(self, namespace_id: str, operation: str = "store", correlation_id: str = "") -> RetentionLease:
        """Renew the lease of ``namespace_id`` if it is still renewable.

        Args:
            namespace_id: The namespace whose lease to renew.
            operation: Operation for a refusal envelope, if one is raised.
            correlation_id: Correlation id for a refusal envelope.

        Returns:
            The renewed lease.

        Raises:
            RefusalError: ``NAMESPACE_INVALID`` if the namespace is unknown; ``LEASE_EXPIRED`` if the
                lease is already expired/revoked/discarded.
        """
        now = self._clock()
        with self._lock:
            lease = self._require_lease(namespace_id, operation, correlation_id)
            state = lease.evaluate(now)
            if state in ("expired", "revoked", "discarded"):
                raise RefusalError.of("LEASE_EXPIRED", operation, correlation_id or namespace_id)
            # ``active`` or ``abandoned`` (still within grace): a renewal reactivates the lease and
            # clears the abandonment mark. Past the grace boundary ``evaluate`` has already made the
            # lease ``expired`` above, so this path cannot resurrect an expired lease.
            lease.last_heartbeat_at = now
            lease.expires_at = now + lease.lease_ttl
            lease.lease_revision += 1
            lease.lease_state = "active"
            lease.abandoned_at = None
            return lease

    def store(
        self,
        namespace_id: str,
        content: bytes,
        *,
        content_type: str = "text/plain",
        correlation_id: str = "",
    ) -> ContextReference:
        """Store ``content`` in ``namespace_id`` and return an opaque reference.

        Order of checks (BR4.6): namespace, authorization, lease, then quota — the quota is enforced
        before any byte is copied.

        Args:
            namespace_id: The owning namespace.
            content: The raw bytes to store.
            content_type: MIME type of the content.
            correlation_id: Correlation id for a refusal envelope.

        Returns:
            The created reference.

        Raises:
            RefusalError: ``NAMESPACE_INVALID``, ``AUTHORIZATION_HOST_DENIED``, ``LEASE_EXPIRED`` or
                ``QUOTA_EXCEEDED`` as the matching precondition fails. No content is copied on refusal.
        """
        operation = "store"
        corr = correlation_id or namespace_id
        now = self._clock()
        with self._lock:
            self._require_namespace(namespace_id, operation, corr)
            self._require_authorized(AuthorizationContext(operation, namespace_id), operation, corr)
            lease = self._require_active_lease(namespace_id, operation, corr, now)

            used = self._memory_used.get(namespace_id, 0)
            if used + len(content) > lease.memory_quota:
                raise RefusalError.of("QUOTA_EXCEEDED", operation, corr)

            reference = _Reference(
                reference_id=secrets.token_urlsafe(24),
                namespace_id=namespace_id,
                content=bytes(content),
                content_type=content_type,
                content_digest=hashlib.sha256(content).hexdigest(),
            )
            self._references[reference.reference_id] = reference
            self._memory_used[namespace_id] = used + len(content)
            return self._view(reference)

    def retrieve(
        self,
        namespace_id: str,
        reference_id: str,
        *,
        correlation_id: str = "",
    ) -> tuple[bytes, str]:
        """Retrieve content for ``reference_id`` as seen by ``namespace_id``.

        The owner retrieves its own active (or delegated) reference. A non-owner retrieves only through
        an active delegation grant naming the reference; without one, access is denied with no hint that
        the reference exists.

        Args:
            namespace_id: The acting namespace.
            reference_id: The target reference.
            correlation_id: Correlation id for a refusal envelope.

        Returns:
            A tuple of (content bytes, content type).

        Raises:
            RefusalError: ``NAMESPACE_INVALID`` for an unknown namespace, ``AUTHORIZATION_HOST_DENIED``
                when the host denies or no grant exists, ``REFERENCE_EXPIRED_OR_DELETED`` when the
                reference is gone or its lifecycle is closed, ``LEASE_EXPIRED`` when the owner lease ended.
        """
        operation = "retrieve"
        corr = correlation_id or namespace_id
        now = self._clock()
        with self._lock:
            self._require_namespace(namespace_id, operation, corr)
            self._require_authorized(
                AuthorizationContext(operation, namespace_id, reference_id=reference_id), operation, corr
            )

            # The acting namespace must itself hold an active lease (W2): a recipient whose own lease
            # has ended cannot read even a reference delegated to it, and an owner with a dead lease
            # cannot read its own content. This runs before the reference is located so an expired
            # acting lease never confirms whether the reference exists.
            self._require_active_lease(namespace_id, operation, corr, now)

            reference = self._references.get(reference_id)
            if reference is None or reference.lifecycle_state in ("deleted", "expired"):
                raise RefusalError.of("REFERENCE_EXPIRED_OR_DELETED", operation, corr)

            # Enforce that the owner lease is still active too; the lease object itself is not needed
            # here. When the acting namespace is the owner this repeats the check above harmlessly.
            self._require_active_lease(reference.namespace_id, operation, corr, now)

            if reference.namespace_id != namespace_id and not self._has_active_grant(
                namespace_id, reference_id, now
            ):
                # No active grant: deny without confirming existence.
                raise RefusalError.of("AUTHORIZATION_HOST_DENIED", operation, corr)

            return reference.content, reference.content_type

    def delete(self, namespace_id: str, reference_id: str, *, correlation_id: str = "") -> None:
        """Discard the content of ``reference_id`` before acknowledging, and mark it deleted.

        Args:
            namespace_id: The owning namespace.
            reference_id: The reference to delete.
            correlation_id: Correlation id for a refusal envelope.

        Raises:
            RefusalError: ``NAMESPACE_INVALID``, ``AUTHORIZATION_HOST_DENIED`` or
                ``REFERENCE_EXPIRED_OR_DELETED`` as the matching precondition fails.
        """
        operation = "delete"
        corr = correlation_id or namespace_id
        with self._lock:
            self._require_namespace(namespace_id, operation, corr)
            self._require_authorized(
                AuthorizationContext(operation, namespace_id, reference_id=reference_id), operation, corr
            )
            reference = self._references.get(reference_id)
            if reference is None or reference.namespace_id != namespace_id:
                raise RefusalError.of("REFERENCE_EXPIRED_OR_DELETED", operation, corr)
            if reference.lifecycle_state in ("deleted", "expired"):
                raise RefusalError.of("REFERENCE_EXPIRED_OR_DELETED", operation, corr)

            # Discard content before the operation returns (BR4.7).
            used = self._memory_used.get(namespace_id, 0)
            self._memory_used[namespace_id] = max(0, used - len(reference.content))
            reference.content = b""
            reference.lifecycle_state = "deleted"
            reference.reference_revision += 1

    def shutdown(self, namespace_id: str, *, correlation_id: str = "") -> int:
        """Discard a namespace synchronously before acknowledging, leaving nothing retrievable.

        Order of checks: namespace, then host authorization. Once past them, the discard happens before
        this method returns (BR4.7): every reference the namespace still owns has its content zeroed and
        its lifecycle closed, every active grant the namespace issued *or* received is revoked together
        with its links, the lease is marked ``discarded`` and the memory charge is cleared. After a
        successful shutdown a later retrieve of any of those references is refused, and the discarded
        lease can never be renewed again.

        Args:
            namespace_id: The namespace to discard.
            correlation_id: Correlation id for a refusal envelope.

        Returns:
            The number of references whose content was discarded.

        Raises:
            RefusalError: ``NAMESPACE_INVALID`` if the namespace is unknown; ``AUTHORIZATION_HOST_DENIED``
                if the host denies the shutdown. No content is touched on refusal, and no content
                survives a successful shutdown.
        """
        operation = "shutdown"
        corr = correlation_id or namespace_id
        with self._lock:
            self._require_namespace(namespace_id, operation, corr)
            self._require_authorized(AuthorizationContext(operation, namespace_id), operation, corr)

            discarded = 0
            for reference in self._references.values():
                if reference.namespace_id == namespace_id and reference.lifecycle_state not in (
                    "deleted",
                    "expired",
                ):
                    # Zero the content before the operation returns, exactly as ``delete`` does.
                    reference.content = b""
                    reference.lifecycle_state = "deleted"
                    reference.reference_revision += 1
                    discarded += 1

            for grant_id, grant in self._grants.items():
                if grant.status != "active":
                    continue
                if namespace_id in (grant.issuer_namespace_id, grant.recipient_namespace_id):
                    grant.status = "revoked"
                    grant.grant_revision += 1
                    for link in self._links.get(grant_id, []):
                        link.link_status = "revoked"

            lease = self._leases.get(namespace_id)
            if lease is not None:
                lease.lease_state = "discarded"
            self._memory_used[namespace_id] = 0
            return discarded

    def delegate(
        self,
        issuer_namespace_id: str,
        recipient_namespace_id: str,
        reference_ids: tuple[str, ...],
        *,
        expires_at: float,
        correlation_id: str = "",
    ) -> DelegationGrant:
        """Create an auditable grant of named references from issuer to recipient.

        Args:
            issuer_namespace_id: The namespace issuing the grant (owner of the references).
            recipient_namespace_id: The namespace receiving access.
            reference_ids: The named references; must be non-empty and all owned by the issuer.
            expires_at: Absolute grant expiry, in the store clock's units.
            correlation_id: Correlation id for a refusal envelope.

        Returns:
            The created grant; the named references become ``delegated``.

        Raises:
            RefusalError: ``NAMESPACE_INVALID`` for an unknown namespace,
                ``REQUEST_SCHEMA_INVALID`` for an empty grant or a reference not owned by the issuer,
                ``AUTHORIZATION_HOST_DENIED`` when the host denies, ``LEASE_EXPIRED`` when either the
                issuer or the recipient lease is no longer active, ``REFERENCE_EXPIRED_OR_DELETED`` for
                a reference owned by the issuer but already deleted or expired.
        """
        operation = "delegate"
        corr = correlation_id or issuer_namespace_id
        now = self._clock()
        with self._lock:
            self._require_namespace(issuer_namespace_id, operation, corr)
            self._require_namespace(recipient_namespace_id, operation, corr)
            if not reference_ids:
                raise RefusalError.of("REQUEST_SCHEMA_INVALID", operation, corr)
            self._require_authorized(
                AuthorizationContext(
                    operation, issuer_namespace_id, recipient_namespace_id=recipient_namespace_id
                ),
                operation,
                corr,
            )
            # Both the issuer and the recipient must hold an active lease before any grant is created
            # (W1): a grant issued from, or granted to, a namespace whose retention has ended is refused
            # with ``LEASE_EXPIRED`` and leaves no grant, no link and no lifecycle change behind.
            self._require_active_lease(issuer_namespace_id, operation, corr, now)
            self._require_active_lease(recipient_namespace_id, operation, corr, now)
            # Validate every reference before any grant/link is created or any lifecycle is touched
            # (BR4.6, C1): an unknown or foreign reference is a schema error; a reference that exists
            # but is already deleted/expired cannot be delegated. All checks run up front so a refusal
            # leaves no partial grant and no lifecycle change behind.
            for reference_id in reference_ids:
                reference = self._references.get(reference_id)
                if reference is None or reference.namespace_id != issuer_namespace_id:
                    raise RefusalError.of("REQUEST_SCHEMA_INVALID", operation, corr)
                if reference.lifecycle_state in ("deleted", "expired"):
                    raise RefusalError.of("REFERENCE_EXPIRED_OR_DELETED", operation, corr)

            grant = DelegationGrant(
                grant_id=secrets.token_hex(16),
                issuer_namespace_id=issuer_namespace_id,
                recipient_namespace_id=recipient_namespace_id,
                reference_ids=tuple(reference_ids),
                issued_at=now,
                expires_at=expires_at,
            )
            self._grants[grant.grant_id] = grant
            self._links[grant.grant_id] = [_Link(reference_id=rid) for rid in reference_ids]
            for reference_id in reference_ids:
                self._references[reference_id].lifecycle_state = "delegated"
            return grant

    def catalog(self, namespace_id: str, *, correlation_id: str = "") -> tuple[ContextReference, ...]:
        """Return read-only views of the live references the namespace owns.

        A read-only listing: it copies nothing and returns only references the host authorizes and whose
        lifecycle is still open (``active`` or ``delegated``). ``deleted`` and ``expired`` references are
        omitted, so the catalog never confirms the existence of a reference that is gone (BR4.4).

        Args:
            namespace_id: The owning namespace.
            correlation_id: Correlation id for a refusal envelope.

        Returns:
            A tuple of public reference views, ordered by reference id.

        Raises:
            RefusalError: ``NAMESPACE_INVALID`` if the namespace is unknown; ``AUTHORIZATION_HOST_DENIED``
                if the host denies the catalog operation; ``LEASE_EXPIRED`` if the namespace lease is no
                longer active.
        """
        operation = "catalog"
        corr = correlation_id or namespace_id
        now = self._clock()
        with self._lock:
            self._require_namespace(namespace_id, operation, corr)
            self._require_authorized(AuthorizationContext(operation, namespace_id), operation, corr)
            # A namespace whose lease has ended cannot list its references (W3).
            self._require_active_lease(namespace_id, operation, corr, now)
            views = [
                self._view(reference)
                for reference in self._references.values()
                if reference.namespace_id == namespace_id
                and reference.lifecycle_state in ("active", "delegated")
            ]
            return tuple(sorted(views, key=lambda view: view.reference_id))

    def revoke_grant(self, grant_id: str) -> None:
        """Revoke ``grant_id``, closing its links and reverting unlinked references to ``active``.

        A reference stays ``delegated`` while any other active link covers it; it returns to ``active``
        only when no active link remains and its owner is still valid.

        Args:
            grant_id: The grant to revoke. Unknown ids are ignored.
        """
        now = self._clock()
        with self._lock:
            grant = self._grants.get(grant_id)
            if grant is None or grant.status != "active":
                return
            grant.status = "revoked"
            grant.grant_revision += 1
            for link in self._links.get(grant_id, []):
                link.link_status = "revoked"
            for reference_id in grant.reference_ids:
                reference = self._references.get(reference_id)
                if reference is None or reference.lifecycle_state in ("deleted", "expired"):
                    continue
                if not self._has_active_grant_for_reference(reference_id, now):
                    reference.lifecycle_state = "active"

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _view(self, reference: _Reference) -> ContextReference:
        """Return a public, immutable view of an internal reference record."""
        return ContextReference(
            reference_id=reference.reference_id,
            namespace_id=reference.namespace_id,
            content_digest=reference.content_digest,
            lifecycle_state=reference.lifecycle_state,
            reference_revision=reference.reference_revision,
        )

    def _require_namespace(self, namespace_id: str, operation: str, correlation_id: str) -> SessionNamespace:
        """Return the namespace or raise ``NAMESPACE_INVALID``."""
        namespace = self._namespaces.get(namespace_id)
        if namespace is None:
            raise RefusalError.of("NAMESPACE_INVALID", operation, correlation_id)
        return namespace

    def _require_lease(self, namespace_id: str, operation: str, correlation_id: str) -> RetentionLease:
        """Return the lease or raise ``NAMESPACE_INVALID`` when it is missing."""
        lease = self._leases.get(namespace_id)
        if lease is None:
            raise RefusalError.of("NAMESPACE_INVALID", operation, correlation_id)
        return lease

    def _require_active_lease(
        self, namespace_id: str, operation: str, correlation_id: str, now: float
    ) -> RetentionLease:
        """Return the lease if active, else raise ``LEASE_EXPIRED``."""
        lease = self._require_lease(namespace_id, operation, correlation_id)
        state = lease.evaluate(now)
        if state != "active":
            raise RefusalError.of("LEASE_EXPIRED", operation, correlation_id)
        return lease

    def _require_authorized(self, context: AuthorizationContext, operation: str, correlation_id: str) -> None:
        """Raise ``AUTHORIZATION_HOST_DENIED`` unless the host callback allows the operation."""
        if not self._authorize(context):
            raise RefusalError.of("AUTHORIZATION_HOST_DENIED", operation, correlation_id)

    def _has_active_grant(self, recipient_namespace_id: str, reference_id: str, now: float) -> bool:
        """Report whether an active grant lets ``recipient_namespace_id`` read ``reference_id``."""
        for grant in self._grants.values():
            if (
                grant.recipient_namespace_id == recipient_namespace_id
                and reference_id in grant.reference_ids
                and self._grant_active(grant, now)
            ):
                return True
        return False

    def _has_active_grant_for_reference(self, reference_id: str, now: float) -> bool:
        """Report whether any active grant still covers ``reference_id``."""
        for grant in self._grants.values():
            if reference_id in grant.reference_ids and self._grant_active(grant, now):
                return True
        return False

    @staticmethod
    def _grant_active(grant: DelegationGrant, now: float) -> bool:
        """Report whether ``grant`` is active and not past its expiry."""
        return grant.status == "active" and now < grant.expires_at
