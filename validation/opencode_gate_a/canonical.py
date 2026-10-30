"""Canonical JSON serialisation and SHA-256 digests for the offline Gate A layer (``canonical-json-v1``).

Every digest produced by this validation package — turn/fault assignment digests, the ``RunManifest``
binding, the sanitized result digest, the ``EvaluationResult`` digest and the Gate A evidence digests —
is the SHA-256 of the *canonical* JSON encoding defined here: object keys sorted by Unicode code point,
the tightest separators, UTF-8 without BOM, no ASCII escaping, and ``NaN``/``Infinity`` rejected.

Two structurally equal values therefore produce byte-identical encodings and the same digest on any
machine, which is what makes the offline replay in :mod:`evaluator` deterministic. This module is a pure
function of its argument: no framework import, no network, no clock, no filesystem (TECH-U2.8/U2.9).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Final

__all__ = [
    "CANONICAL_SPEC_VERSION",
    "DIGEST_PREFIX",
    "canonical_bytes",
    "canonical_json",
    "digest",
]

CANONICAL_SPEC_VERSION: Final[str] = "canonical-json-v1"
"""The versioned canonicalisation contract id that participates in the manifest binding."""

DIGEST_PREFIX: Final[str] = "sha256:"
"""The digest framing: every digest string is ``sha256:`` followed by 64 lowercase hex characters."""


def canonical_json(value: Any) -> str:
    """Encode ``value`` as canonical JSON text.

    Canonical means: object keys sorted, item/key separators without spaces, non-ASCII characters emitted
    literally rather than ``\\uXXXX``-escaped, and non-finite floats rejected. The result is stable across
    runs and machines, so it is safe to digest.

    Args:
        value: Any JSON-serialisable value (dict, list, str, int, finite float, bool, None).

    Returns:
        The canonical JSON string.

    Raises:
        ValueError: If ``value`` contains ``NaN``, ``Infinity`` or ``-Infinity``.
        TypeError: If ``value`` holds a type the JSON encoder cannot represent.
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_bytes(value: Any) -> bytes:
    """Encode ``value`` as canonical JSON and return its UTF-8 bytes.

    Args:
        value: Any JSON-serialisable value.

    Returns:
        The UTF-8 encoding of :func:`canonical_json` (no BOM, no trailing newline).
    """
    return canonical_json(value).encode("utf-8")


def digest(value: Any) -> str:
    """Return the framed SHA-256 digest of ``value``'s canonical encoding.

    Args:
        value: Any JSON-serialisable value.

    Returns:
        ``"sha256:"`` followed by the 64-character lowercase hexadecimal SHA-256 digest of the canonical
        UTF-8 bytes.
    """
    hexdigest = hashlib.sha256(canonical_bytes(value)).hexdigest()
    return f"{DIGEST_PREFIX}{hexdigest}"
