"""Offline U2 Gate A layer: deterministic contracts, sanitisation and evaluation for the CE-60 benchmark.

This package is the offline, replay-only slice of U2. It never touches the network, AWS, credentials or a
provider, and it never imports ``context-core``, the plugin or the existing runner. It defines:

* :mod:`canonical` — the ``canonical-json-v1`` serialisation and SHA-256 digests used by every binding.
* :mod:`contracts` — the frozen, fail-closed entities (``TurnAssignment``, ``StatPlan``, ``RunManifest``,
  ``MetricValidity``, ``SanitizedTurnResult``, ``SanitizedRunResult``, ``EvaluationResult``,
  ``GateDecision``), plus the terminal catalog they rely on.
* :mod:`sanitize` — the trust boundary turning untrusted raw turns into payload-free sanitized results.
* :mod:`evaluator` — the deterministic offline evaluator, the replay digest check and the isolated
  same-host Gate A decision.
"""

from __future__ import annotations

from .canonical import CANONICAL_SPEC_VERSION, canonical_bytes, canonical_json, digest
from .contracts import (
    CORRECTNESS_MAX_DROP_PP,
    MODEL_INPUT_MIN_REDUCTION,
    REASON_CODES,
    TURN_COUNT,
    ContractViolation,
    EvaluationResult,
    GateDecision,
    MetricValidity,
    RunManifest,
    SanitizedRunResult,
    SanitizedTurnResult,
    StatPlan,
    TerminalOutcomeCatalog,
    TerminalTurnDefinition,
    TurnAssignment,
)
from .evaluator import decide_gate_a, evaluate, replay
from .sanitize import sanitize_run, sanitize_turn

__all__ = [
    "CANONICAL_SPEC_VERSION",
    "CORRECTNESS_MAX_DROP_PP",
    "MODEL_INPUT_MIN_REDUCTION",
    "REASON_CODES",
    "TURN_COUNT",
    "ContractViolation",
    "EvaluationResult",
    "GateDecision",
    "MetricValidity",
    "RunManifest",
    "SanitizedRunResult",
    "SanitizedTurnResult",
    "StatPlan",
    "TerminalOutcomeCatalog",
    "TerminalTurnDefinition",
    "TurnAssignment",
    "canonical_bytes",
    "canonical_json",
    "decide_gate_a",
    "digest",
    "evaluate",
    "replay",
    "sanitize_run",
    "sanitize_turn",
]
