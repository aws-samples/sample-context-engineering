"""Canonical JSON serialisation and SHA-256 digests — the one way U1 turns a value into bytes.

Every digest in the contract layer (contract, manifest, coverage, evidence, request, policy) is the
SHA-256 of the *canonical* JSON encoding defined here: keys sorted, the tightest separators, UTF-8, and
no ASCII escaping. Two structurally equal values therefore produce byte-identical encodings and the same
digest on any machine, which is what makes replay and digest revalidation deterministic.

No framework import, no I/O, no clock: a pure function of its argument.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

__all__ = [
    "canonical_bytes",
    "canonical_json",
    "digest",
]


def canonical_json(value: Any) -> str:
    """Encode ``value`` as canonical JSON text.

    Canonical means: object keys sorted, item/key separators without spaces, and non-ASCII characters
    emitted literally rather than ``\\uXXXX``-escaped. The result is stable across runs and machines, so
    it is safe to digest.

    Args:
        value: Any JSON-serialisable value (dict, list, str, int, float, bool, None).

    Returns:
        The canonical JSON string.

    Raises:
        TypeError: If ``value`` holds a type the JSON encoder cannot represent.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_bytes(value: Any) -> bytes:
    """Encode ``value`` as canonical JSON and return its UTF-8 bytes.

    Args:
        value: Any JSON-serialisable value.

    Returns:
        The UTF-8 encoding of :func:`canonical_json`.
    """
    return canonical_json(value).encode("utf-8")


def digest(value: Any) -> str:
    """Return the SHA-256 hex digest of ``value``'s canonical encoding.

    Args:
        value: Any JSON-serialisable value.

    Returns:
        The 64-character lowercase hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(canonical_bytes(value)).hexdigest()
