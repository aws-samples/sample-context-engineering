"""Sanitisation: raw, untrusted turn mappings become payload-free :class:`SanitizedTurnResult` objects.

The sanitizer is the trust boundary of the offline layer. A raw turn mapping is **untrusted data**: it
may carry a prompt, a raw response, a tool payload, credentials or logs. The sanitizer never copies any
of those — it reads only an allow-listed set of fields, classifies the terminal state **exclusively from
the terminal catalog** (never from the host's native status, BR7.3), and records one explicit
:class:`MetricValidity` per metric so an unavailable metric is visible rather than silently zeroed
(NFR1.3, NFR4.1).

No network, no AWS, no provider: a pure function of the raw mapping and the catalog.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Final

from .canonical import digest
from .contracts import (
    TURN_COUNT,
    ContractViolation,
    MetricValidity,
    SanitizedRunResult,
    SanitizedTurnResult,
    TerminalOutcomeCatalog,
    TerminalTurnDefinition,
)

__all__ = [
    "FORBIDDEN_RAW_KEYS",
    "HOST_SIGNALS",
    "RAW_METRIC_STATUSES",
    "sanitize_run",
    "sanitize_turn",
]

# Keys that may appear in an untrusted raw turn mapping and must never reach a sanitized result. The
# sanitizer ignores them; this set exists so a canary test can assert none of them leaked (NFR4.1).
FORBIDDEN_RAW_KEYS: Final[frozenset[str]] = frozenset(
    {
        "prompt",
        "prompts",
        "messages",
        "response",
        "raw_response",
        "completion",
        "transcript",
        "tool_payload",
        "payload",
        "tool_result",
        "arguments",
        "credentials",
        "credential",
        "secret",
        "secrets",
        "api_key",
        "api_keys",
        "token",
        "authorization",
        "logs",
        "log",
        "trace",
    }
)

# The host signal the runner observed, before catalog classification. ``success`` still has to pass the
# catalog matcher to count; the matcher result is carried separately in ``matcher_passed``.
HOST_SIGNALS: Final[frozenset[str]] = frozenset({"success", "refused", "error", "timeout"})

# The validity status a raw metric entry may declare for itself. ``valid`` additionally requires a value.
RAW_METRIC_STATUSES: Final[frozenset[str]] = frozenset(
    {"valid", "host_not_exposed", "not_applicable", "collection_failed", "redacted_by_policy"}
)

# The reason code attached to each non-valid metric status, so a downstream auditor sees why.
_STATUS_REASON: Final[dict[str, str]] = {
    "host_not_exposed": "METRIC_INVALID",
    "collection_failed": "METRIC_INVALID",
    "redacted_by_policy": "METRIC_INVALID",
    "not_applicable": "METRIC_INVALID",
}

_OBSERVABILITY_SCALAR_KEYS: Final[frozenset[str]] = frozenset(
    {
        "observability_schema_version",
        "telemetry_status",
        "message_count_max",
        "message_count_median",
        "tool_call_count_total",
        "tool_call_count_max",
        "mcp_tool_call_count_total",
        "session_compacted_observed_count",
        "session_message_count_final",
        "session_tool_call_count_final",
        "session_mcp_tool_call_count_final",
        "delta_message_count_sum",
        "delta_tool_call_count_sum",
        "delta_mcp_tool_call_count_sum",
        "delta_message_count_max",
        "delta_tool_call_count_max",
        "delta_mcp_tool_call_count_max",
        "timeout_count",
        "compaction_status",
        "compaction_reason",
        "compaction_observed_count",
    }
)
_OBSERVABILITY_LIST_KEYS: Final[frozenset[str]] = frozenset(
    {
        "timeout_turn_ids",
        "tool_names",
        "snapshot_gap_turn_ids",
        "snapshot_regression_turn_ids",
        "context_drop_turn_ids",
        "timeout_classifications",
        "timeout_last_tool_names",
        "timeout_last_tool_statuses",
        "timeout_assistant_states",
        "timeout_abort_statuses",
        "timeout_idle_statuses",
        "session_not_idle_turn_ids",
        "error_turn_ids",
        "error_codes",
        "error_status_codes",
        "error_retryable_flags",
    }
)
_MAX_OBSERVABILITY_TOOL_NAMES: Final[int] = 512
_MAX_OBSERVABILITY_TIMEOUT_IDS: Final[int] = TURN_COUNT
_MAX_OBSERVABILITY_NAME_LENGTH: Final[int] = 128


def _classify_terminal(
    *,
    host_signal: str,
    matcher_passed: bool,
    fallback_declared: bool,
    definition: TerminalTurnDefinition,
) -> str:
    """Classify a turn's terminal state from the catalog alone (BR7.3).

    The host's ``host_signal`` is only an observation; whether it is accepted is decided by the catalog
    ``definition``. A ``success`` signal whose matcher did not pass becomes ``invalid_terminal`` even
    though the host reported success.

    Args:
        host_signal: The observed host signal, one of :data:`HOST_SIGNALS`.
        matcher_passed: Whether the deterministic catalog matcher accepted the content.
        fallback_declared: Whether the turn used a declared fallback.
        definition: The catalog definition for this turn.

    Returns:
        One of the terminal-state values.
    """
    if host_signal == "timeout":
        return "timed_out"
    if host_signal == "error":
        return "unrecovered_error"
    if host_signal == "refused":
        if "refused_expected" in definition.accepted_states:
            return "refused_expected"
        return "invalid_terminal"
    # host_signal == "success"
    if not matcher_passed:
        return "invalid_terminal"
    if fallback_declared:
        if (
            "completed_with_fallback" in definition.accepted_states
            and definition.fallback_policy == "declared_fallback_only"
        ):
            return "completed_with_fallback"
        return "invalid_terminal"
    if "completed_expected" in definition.accepted_states:
        return "completed_expected"
    return "invalid_terminal"


def _metric_entry(raw_metrics: Mapping[str, Any], name: str) -> dict[str, Any]:
    """Return the raw metric sub-mapping for ``name`` as a plain dict.

    Args:
        raw_metrics: The ``metrics`` sub-mapping of a raw turn.
        name: The metric name.

    Returns:
        The metric mapping, or an empty dict when the metric is absent (treated as collection failure).
    """
    value = raw_metrics.get(name)
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _resolve_status(entry: Mapping[str, Any], *, has_value: bool) -> str:
    """Resolve the validity status of a single raw metric entry fail-closed.

    An entry with no explicit status and no value is a ``collection_failed`` — the sanitizer never
    invents a zero for a missing metric (NFR1.3).

    Args:
        entry: The raw metric mapping.
        has_value: Whether a usable value was extracted.

    Returns:
        A validity status string.

    Raises:
        ContractViolation: ``METRIC_INVALID`` if an explicit status is outside the closed set.
    """
    declared = entry.get("status")
    if declared is not None:
        if declared not in RAW_METRIC_STATUSES:
            raise ContractViolation("METRIC_INVALID", f"unknown raw metric status: {declared!r}")
        return str(declared)
    if entry.get("exposed") is False:
        return "host_not_exposed"
    if not has_value:
        return "collection_failed"
    return "valid"


def _build_validity(
    *, run_id: str, turn_id: str, metric_name: str, status: str
) -> MetricValidity:
    """Build a turn-scope :class:`MetricValidity` with a reason code for non-valid statuses.

    Args:
        run_id: The run id.
        turn_id: The turn id.
        metric_name: The metric name.
        status: The resolved validity status.

    Returns:
        The validity record.
    """
    reason = None if status == "valid" else _STATUS_REASON.get(status)
    return MetricValidity(
        scope="turn",
        run_id=run_id,
        metric_name=metric_name,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        turn_id=turn_id,
        reason_code=reason,
    )


def _int_or_none(value: Any) -> int | None:
    """Coerce ``value`` to a non-negative int, or ``None`` when absent/invalid.

    Args:
        value: The raw value.

    Returns:
        The int value, or ``None``.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _float_or_none(value: Any) -> float | None:
    """Coerce ``value`` to a float, or ``None`` when absent/invalid.

    Args:
        value: The raw value.

    Returns:
        The float value, or ``None``.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def sanitize_turn(
    raw_turn: Mapping[str, Any],
    *,
    run_id: str,
    catalog: TerminalOutcomeCatalog,
) -> SanitizedTurnResult:
    """Sanitize one raw turn mapping into a payload-free :class:`SanitizedTurnResult`.

    The function reads only allow-listed fields, so any forbidden key in ``raw_turn`` (prompt, response,
    tool payload, credential, log) is dropped by construction. The terminal state is classified from the
    catalog; each metric gets an explicit validity record.

    Args:
        raw_turn: The untrusted raw turn mapping.
        run_id: The run id this turn belongs to.
        catalog: The terminal outcome catalog that classifies the turn.

    Returns:
        The sanitized turn result.

    Raises:
        ContractViolation: ``RESULT_NOT_SANITIZED`` for a missing/invalid turn id or host signal;
            ``METRIC_INVALID`` for an out-of-set metric status; ``TERMINAL_CATALOG_INVALID`` when the
            turn is not in the catalog.
    """
    turn_id = raw_turn.get("turn_id")
    if not isinstance(turn_id, str) or turn_id.strip() == "":
        raise ContractViolation("RESULT_NOT_SANITIZED", "raw turn is missing a turn_id")

    host_signal = raw_turn.get("host_signal")
    if host_signal not in HOST_SIGNALS:
        raise ContractViolation(
            "RESULT_NOT_SANITIZED", f"raw turn {turn_id!r} has unknown host_signal: {host_signal!r}"
        )

    definition = catalog.definition(turn_id)
    matcher_passed = bool(raw_turn.get("matcher_passed", False))
    fallback_declared = raw_turn.get("fallback_status") == "declared_fallback"

    terminal_state = _classify_terminal(
        host_signal=str(host_signal),
        matcher_passed=matcher_passed,
        fallback_declared=fallback_declared,
        definition=definition,
    )
    completion_status = (
        "completed" if terminal_state in ("completed_expected", "completed_with_fallback") else "not_completed"
    )

    raw_metrics = raw_turn.get("metrics")
    metrics: Mapping[str, Any] = raw_metrics if isinstance(raw_metrics, Mapping) else {}

    validities: list[MetricValidity] = []

    # completion is catalog-derived and therefore always valid here.
    validities.append(
        _build_validity(run_id=run_id, turn_id=turn_id, metric_name="completion", status="valid")
    )

    # correctness — only applicable to eligible turns; non-eligible turns are not_applicable.
    correctness_score: float | None = None
    if definition.correctness_eligible:
        entry = _metric_entry(metrics, "correctness")
        score = _float_or_none(entry.get("value"))
        status = _resolve_status(entry, has_value=score is not None)
        if status == "valid":
            correctness_score = score
        validities.append(
            _build_validity(run_id=run_id, turn_id=turn_id, metric_name="correctness", status=status)
        )
    else:
        validities.append(
            _build_validity(
                run_id=run_id, turn_id=turn_id, metric_name="correctness", status="not_applicable"
            )
        )

    # model_input tokens.
    mi_entry = _metric_entry(metrics, "model_input")
    mi_value = _int_or_none(mi_entry.get("value"))
    mi_status = _resolve_status(mi_entry, has_value=mi_value is not None)
    model_input_tokens = mi_value if mi_status == "valid" else None
    validities.append(
        _build_validity(run_id=run_id, turn_id=turn_id, metric_name="model_input", status=mi_status)
    )

    # retrieval attempts/successes.
    rt_entry = _metric_entry(metrics, "retrieval")
    rt_attempts = _int_or_none(rt_entry.get("attempts"))
    rt_successes = _int_or_none(rt_entry.get("successes"))
    rt_has_value = rt_attempts is not None and rt_successes is not None
    rt_status = _resolve_status(rt_entry, has_value=rt_has_value)
    retrieval_attempts = rt_attempts if rt_status == "valid" else None
    retrieval_successes = rt_successes if rt_status == "valid" else None
    validities.append(
        _build_validity(run_id=run_id, turn_id=turn_id, metric_name="retrieval", status=rt_status)
    )

    # latency.
    lat_entry = _metric_entry(metrics, "latency")
    lat_value = _float_or_none(lat_entry.get("value"))
    lat_status = _resolve_status(lat_entry, has_value=lat_value is not None)
    latency_ms = lat_value if lat_status == "valid" else None
    validities.append(
        _build_validity(run_id=run_id, turn_id=turn_id, metric_name="latency", status=lat_status)
    )

    error_code = raw_turn.get("error_code")
    fallback_status = raw_turn.get("fallback_status")

    return SanitizedTurnResult(
        turn_id=turn_id,
        terminal_state=terminal_state,  # type: ignore[arg-type]
        completion_status=completion_status,  # type: ignore[arg-type]
        correctness_eligible=definition.correctness_eligible,
        metric_validity=tuple(validities),
        correctness_score=correctness_score,
        model_input_tokens=model_input_tokens,
        retrieval_attempts=retrieval_attempts,
        retrieval_successes=retrieval_successes,
        latency_ms=latency_ms,
        error_code=error_code if isinstance(error_code, str) else None,
        fallback_status=fallback_status if isinstance(fallback_status, str) else None,
    )


def sanitize_run(
    raw_run: Mapping[str, Any],
    *,
    manifest_digest: str,
    catalog: TerminalOutcomeCatalog,
) -> SanitizedRunResult:
    """Sanitize a raw run mapping into an immutable :class:`SanitizedRunResult` of exactly 60 turns.

    The host evidence is reduced to a payload-free dict (only non-forbidden, JSON-scalar fields survive),
    and the result digest is computed over the sanitized content so it can be revalidated at replay.

    Args:
        raw_run: The untrusted raw run mapping with ``run_id`` and a ``turns`` sequence.
        manifest_digest: The manifest digest this result is bound to.
        catalog: The terminal outcome catalog.

    Returns:
        The sanitized run result.

    Raises:
        ContractViolation: ``RESULT_NOT_SANITIZED`` for a missing run id, a wrong turn count, duplicate
            turns, or any per-turn sanitisation failure.
    """
    run_id = raw_run.get("run_id")
    if not isinstance(run_id, str) or run_id.strip() == "":
        raise ContractViolation("RESULT_NOT_SANITIZED", "raw run is missing a run_id")

    raw_turns = raw_run.get("turns")
    if not isinstance(raw_turns, (list, tuple)):
        raise ContractViolation("RESULT_NOT_SANITIZED", "raw run turns must be a sequence")
    if len(raw_turns) != TURN_COUNT:
        raise ContractViolation(
            "RESULT_NOT_SANITIZED", f"raw run must contain exactly {TURN_COUNT} turns"
        )

    sanitized = tuple(
        sanitize_turn(raw_turn, run_id=run_id, catalog=catalog) for raw_turn in raw_turns
    )

    host_evidence = _sanitize_host_evidence(raw_run.get("host_evidence"))
    sanitization_status = "passed"

    # Build the digest on the sanitized bytes, then construct the frozen result (which re-verifies it).
    provisional = {
        "run_id": run_id,
        "manifest_digest": manifest_digest,
        "sanitization_status": sanitization_status,
        "host_evidence": host_evidence,
        "turn_results": [t.as_dict() for t in sorted(sanitized, key=lambda t: t.turn_id)],
    }
    result_digest = digest(provisional)

    return SanitizedRunResult(
        run_id=run_id,
        manifest_digest=manifest_digest,
        turn_results=sanitized,
        host_evidence=host_evidence,
        sanitization_status="passed",
        result_digest=result_digest,
    )


def _sanitize_observability_evidence(raw: Any) -> dict[str, object]:
    """Keep only the bounded scalar/list observability schema inside host evidence."""
    if not isinstance(raw, Mapping):
        return {}
    clean: dict[str, object] = {}
    integer_keys = {
        "observability_schema_version",
        "message_count_max",
        "tool_call_count_total",
        "tool_call_count_max",
        "mcp_tool_call_count_total",
        "session_compacted_observed_count",
        "session_message_count_final",
        "session_tool_call_count_final",
        "session_mcp_tool_call_count_final",
        "delta_message_count_sum",
        "delta_tool_call_count_sum",
        "delta_mcp_tool_call_count_sum",
        "delta_message_count_max",
        "delta_tool_call_count_max",
        "delta_mcp_tool_call_count_max",
        "timeout_count",
        "compaction_observed_count",
    }
    enum_values = {
        "telemetry_status": {"ok", "partial", "unavailable", "not_attempted"},
        "compaction_status": {"observed", "none_observed", "unavailable"},
        "compaction_reason": {"unexplained_context_drop", "event_channel_not_observed"},
    }
    for key, value in raw.items():
        if key in _OBSERVABILITY_SCALAR_KEYS:
            if value is None:
                clean[key] = None
            elif key in enum_values:
                if isinstance(value, str) and value in enum_values[key]:
                    clean[key] = value
            elif key in integer_keys:
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    clean[key] = value
            elif (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and value >= 0
            ):
                clean[key] = value
            continue
        if key not in _OBSERVABILITY_LIST_KEYS or not isinstance(value, (list, tuple)):
            continue
        limit = _MAX_OBSERVABILITY_TIMEOUT_IDS if key != "tool_names" else _MAX_OBSERVABILITY_TOOL_NAMES
        if len(value) > limit:
            continue
        names = [
            item
            for item in value
            if isinstance(item, str)
            and item
            and len(item) <= _MAX_OBSERVABILITY_NAME_LENGTH
            and item.isprintable()
            and not any(character.isspace() for character in item)
            and all(character.isalnum() or character in "_.:-" for character in item)
        ]
        if len(names) == len(value):
            clean[key] = names
    return clean


def _sanitize_host_evidence(raw_evidence: Any) -> dict[str, object]:
    """Reduce host evidence to safe scalars and the bounded observability schema.

    Generic nested values are dropped. The only nested block allowed is ``observability``, whose fields
    are independently allow-listed and whose names/lists are bounded. This keeps diagnostic counts while
    preventing prompts, responses, arguments, tool payloads or logs from entering persisted evidence.
    """
    if not isinstance(raw_evidence, Mapping):
        return {}
    clean: dict[str, object] = {}
    for key, value in raw_evidence.items():
        if not isinstance(key, str) or key.lower() in FORBIDDEN_RAW_KEYS:
            continue
        if key == "observability":
            safe_observability = _sanitize_observability_evidence(value)
            if safe_observability:
                clean[key] = safe_observability
            continue
        if (isinstance(value, bool) or value is None or isinstance(value, (int, float, str))) and (
            not isinstance(value, float) or math.isfinite(value)
        ):
            clean[key] = value
    return clean
