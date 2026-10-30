"""``ContractIdentity`` — the canonical identity of a schema a request speaks.

A contract identity is a ``MAJOR.MINOR`` version plus the SHA-256 digest of the canonical schema. Two
identities are *compatible* only when their major versions match and their digests are equal; any
divergence is a ``CONTRACT_MISMATCH`` that must be refused before any durable effect (BR1.3, BR1.4).

No framework import, no I/O.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .canonical import digest as _digest
from .errors import RefusalError

__all__ = [
    "ContractIdentity",
    "require_compatible",
]

_VERSION_RE = re.compile(r"^(?P<major>\d+)\.(?P<minor>\d+)$")


@dataclass(frozen=True)
class ContractIdentity:
    """Canonical identity of a schema consumed by an operation.

    Attributes:
        contract_version: A ``MAJOR.MINOR`` string, e.g. ``"1.0"``. Major changes are breaking.
        contract_digest: The SHA-256 of the canonical schema; changes whenever the schema changes.
    """

    contract_version: str
    contract_digest: str

    def __post_init__(self) -> None:
        """Reject a malformed version string or digest.

        Raises:
            ValueError: If ``contract_version`` is not ``MAJOR.MINOR`` or ``contract_digest`` is not a
                64-character lowercase hex string.
        """
        if not _VERSION_RE.match(self.contract_version):
            raise ValueError(f"contract_version must be MAJOR.MINOR, got {self.contract_version!r}")
        if not re.fullmatch(r"[0-9a-f]{64}", self.contract_digest):
            raise ValueError("contract_digest must be a 64-character lowercase sha256 hex string")

    @property
    def major(self) -> int:
        """The major version component as an integer."""
        match = _VERSION_RE.match(self.contract_version)
        assert match is not None  # guaranteed by __post_init__
        return int(match.group("major"))

    @classmethod
    def of_schema(cls, contract_version: str, schema: object) -> ContractIdentity:
        """Build an identity whose digest is the canonical digest of ``schema``.

        Args:
            contract_version: The ``MAJOR.MINOR`` version string.
            schema: Any JSON-serialisable value standing for the canonical schema.

        Returns:
            The contract identity.
        """
        return cls(contract_version=contract_version, contract_digest=_digest(schema))

    def is_compatible_with(self, other: ContractIdentity) -> bool:
        """Report whether ``self`` may be used where ``other`` is expected.

        Compatible means same major version and identical digest. A differing minor alone is not
        enough: the digest must match too, since an additive minor still changes the schema bytes.

        Args:
            other: The expected identity.

        Returns:
            ``True`` when major versions match and digests are equal.
        """
        return self.major == other.major and self.contract_digest == other.contract_digest


def require_compatible(
    actual: ContractIdentity,
    expected: ContractIdentity,
    operation: str,
    correlation_id: str,
    *,
    capability: str = "none",
) -> None:
    """Raise ``CONTRACT_MISMATCH`` unless ``actual`` is compatible with ``expected``.

    Args:
        actual: The identity carried by the request.
        expected: The identity the runtime currently holds.
        operation: The operation being validated, for the refusal envelope.
        correlation_id: An opaque correlation id.
        capability: The practice or ``"none"``.

    Raises:
        RefusalError: ``CONTRACT_MISMATCH`` when the identities are incompatible. No effect occurs.
    """
    if not actual.is_compatible_with(expected):
        raise RefusalError.of("CONTRACT_MISMATCH", operation, correlation_id, capability=capability)
