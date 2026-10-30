"""Selectable CE-60 correctness oracle versions.

The oracle is a declared experiment variable, never a Gate A threshold: the 20 % model-input reduction and
the 3.0 pp correctness-drop limits live in :mod:`opencode_gate_a.contracts` and are not touched here.

* ``v3`` is the community oracle exactly as shipped (``src/accuracy.py``). It scored the v2 and v3
  campaigns and stays the default, so every earlier result keeps reproducing byte for byte.
* ``v4`` is the same oracle with ONE declared change. The C1-lambda-docs critical check
  ``async-retry-semantics`` asks whether the answer states that Lambda retries a failed asynchronous
  invocation twice. v3 accepts a fixed list of spellings ("twice", "two more times", ...). A model that
  answers "retries up to 2 more times" or "the default is 2 retries" states the same fact and is scored
  wrong, so one wording flip moves correctness by 3.33 pp, more than the 3.0 pp limit. v4 therefore also
  accepts the same meaning spelled with digits or words, with "more/additional" times, retries or attempts,
  with Markdown emphasis around the numeral. The answer language is NOT widened: the campaign pins the
  response language to English for both arms instead (see ``--system-prompt`` in the runner).

v4 is a strict superset of v3 for that check and identical to v3 everywhere else: every other turn, every
other check, the universal check, the eligibility set and the pass rule are the v3 code path. It is applied
to both arms by the same switch, so it cannot favour one arm.

The version is chosen with the ``CE60_ORACLE_VERSION`` environment variable and recorded in the run
evidence together with a digest of the v4 rule, so a result always says which oracle scored it.
"""

from __future__ import annotations

import hashlib
import os
import re
import types
from dataclasses import dataclass
from typing import Any, Final

from .contracts import ContractViolation

__all__ = [
    "DEFAULT_ORACLE_VERSION",
    "ORACLE_VERSIONS",
    "ORACLE_VERSION_ENV",
    "load_accuracy",
    "oracle_digest",
    "oracle_version",
    "retry_count_matches",
]

ORACLE_VERSION_ENV: Final[str] = "CE60_ORACLE_VERSION"
"""Environment variable that selects the oracle version for a run."""

DEFAULT_ORACLE_VERSION: Final[str] = "v3"
"""The oracle used when the variable is unset: the shipped community oracle."""

ORACLE_VERSIONS: Final[tuple[str, ...]] = ("v3", "v4")
"""The closed set of selectable versions."""

_C1_LABEL: Final[str] = "C1-lambda-docs"
_RETRY_CHECK_NAME: Final[str] = "async-retry-semantics"

# Markdown emphasis is removed before matching so "**2** more times" and "_2 retries_" read like plain text.
_EMPHASIS: Final[re.Pattern[str]] = re.compile(r"[*_`~]+")

# "Lambda retries the function twice", however a correct English answer words it. The text is case-folded
# and accent-stripped by the oracle's own normaliser first, so only lower-case ASCII appears here. The guard
# on the left keeps "12 retries" and "0.2 times" from matching; "20 times" fails because a digit follows the 2.
_RETRY_COUNT_PATTERN: Final[str] = (
    r"(?<![\w.])(?:"
    # twice | (2|two) [more|additional|extra|new|further] (times|retries|retry attempts|attempts|tries)
    r"twice"
    r"|(?:2|two)\s+(?:(?:more|additional|extra|new|further)\s+)?"
    r"(?:times|retries|retry\s+attempts|attempts|tries)"
    r")(?![\w])"
)
_RETRY_COUNT: Final[re.Pattern[str]] = re.compile(_RETRY_COUNT_PATTERN)

# One v4 view per imported community module; keeping the module itself guards against a reused id().
_V4_VIEWS: dict[int, tuple[Any, Any]] = {}


def retry_count_matches(normalized_answer: str) -> bool:
    """Return whether an already case-folded answer states the two-retry default in any accepted wording.

    Args:
        normalized_answer: The answer after the oracle's own normalisation (case-folded, accents removed).

    Returns:
        ``True`` when a v4-equivalent spelling of "retries twice" is present.
    """
    return _RETRY_COUNT.search(_EMPHASIS.sub("", normalized_answer)) is not None


def oracle_version() -> str:
    """Return the oracle version selected by the environment.

    Returns:
        ``v3`` when the variable is unset or empty, otherwise the validated value.

    Raises:
        ContractViolation: ``ORACLE_MISMATCH`` when the variable names an unknown version.
    """
    raw = os.environ.get(ORACLE_VERSION_ENV, "").strip()
    if raw == "":
        return DEFAULT_ORACLE_VERSION
    if raw not in ORACLE_VERSIONS:
        raise ContractViolation(
            "ORACLE_MISMATCH",
            f"{ORACLE_VERSION_ENV}={raw!r} is not one of {', '.join(ORACLE_VERSIONS)}",
        )
    return raw


def oracle_digest(version: str | None = None) -> str:
    """Return a stable digest identifying the rule set of an oracle version.

    ``v3`` is identified by its name alone because it is the unmodified shipped module. ``v4`` also
    hashes the equivalence pattern, so changing the accepted spellings changes the recorded digest.

    Args:
        version: The version to describe, or ``None`` for the one selected by the environment.

    Returns:
        A ``sha256:`` digest string.
    """
    chosen = oracle_version() if version is None else version
    material = f"ce60-oracle:{chosen}"
    if chosen == "v4":
        material += f":{_C1_LABEL}:{_RETRY_CHECK_NAME}:{_RETRY_COUNT_PATTERN}"
    return "sha256:" + hashlib.sha256(material.encode("utf-8")).hexdigest()


def load_accuracy(base: Any, version: str | None = None) -> Any:
    """Return the accuracy module to score with for the selected oracle version.

    Args:
        base: The imported community ``accuracy`` module (``from src import accuracy``).
        version: An explicit version, or ``None`` to read ``CE60_ORACLE_VERSION``.

    Returns:
        ``base`` itself for ``v3``; an object exposing ``_checks_for`` and ``score_turn`` with the declared
        v4 change for ``v4``.

    Raises:
        ContractViolation: ``ORACLE_MISMATCH`` for an unknown version or an oracle module that does not
            carry the C1 check this version rewrites.
    """
    chosen = oracle_version() if version is None else version
    if chosen not in ORACLE_VERSIONS:
        raise ContractViolation("ORACLE_MISMATCH", f"unknown CE-60 oracle version: {chosen!r}")
    if chosen == "v3":
        return base
    cached = _V4_VIEWS.get(id(base))
    if cached is None or cached[0] is not base:
        cached = (base, _build_v4(base))
        _V4_VIEWS[id(base)] = cached
    return cached[1]


def _build_v4(base: Any) -> Any:
    """Build the v4 view of ``base``: identical, except the C1 retry-count check is a superset."""
    check_type = base.Check
    original = base.CHECKS.get(_C1_LABEL)
    if original is None or not any(check.name == _RETRY_CHECK_NAME for check in original):
        raise ContractViolation(
            "ORACLE_MISMATCH",
            f"the community oracle has no {_C1_LABEL}/{_RETRY_CHECK_NAME} check to extend",
        )

    @dataclass(frozen=True)
    class EquivalentRetryCheck(check_type):  # type: ignore[misc, valid-type]
        """The v3 check, or any other accepted spelling of the same retry count."""

        def evaluate(self, answer: str) -> tuple[bool, str]:
            ok, reason = super().evaluate(answer)
            if ok:
                return True, ""
            if retry_count_matches(base._normalize(answer or "")):
                return True, ""
            return False, reason

    patched = tuple(
        EquivalentRetryCheck(
            name=check.name,
            all_of=check.all_of,
            any_of=check.any_of,
            none_of=check.none_of,
            weight=check.weight,
            critical=check.critical,
        )
        if check.name == _RETRY_CHECK_NAME
        else check
        for check in original
    )

    def checks_for(label: str) -> Any:
        return patched if label == _C1_LABEL else base._checks_for(label)

    def score_turn(label: str, answer: str) -> Any:
        # Reuse the shipped scoring routine unchanged: only the C1 entry is swapped for the duration of
        # the call, so the universal check, the empty-answer rule and the weights are the v3 code path.
        if label != _C1_LABEL:
            return base.score_turn(label, answer)
        base.CHECKS[_C1_LABEL] = patched
        try:
            return base.score_turn(label, answer)
        finally:
            base.CHECKS[_C1_LABEL] = original

    return types.SimpleNamespace(version="v4", _checks_for=checks_for, score_turn=score_turn)
