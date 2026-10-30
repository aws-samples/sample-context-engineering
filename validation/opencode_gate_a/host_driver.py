"""Payload-free host bridge between the TypeScript CE-60 driver and the offline Gate A layer.

The real CE-60 host driver lives in TypeScript (``.opencode/runner/ce60.ts``): it starts OpenCode,
creates a session and sends the 60 prompts. This module is the Python half it shells out to. It does
only the parts that must be exact rather than driven by a live host:

* :func:`emit_turns` resolves the canonical 60-turn CE-60 script (``turn_id`` + ``prompt`` only) from
  the community scenario, or fails closed so the caller supplies ``--turns-json``. It never persists a
  raw prompt — it prints the script to stdout for the driver to consume in memory.
* :func:`finalize` takes the in-memory per-turn observations the driver captured (host signal, raw
  answer text, tool-call names, token/latency readings), derives the deterministic terminal /
  completion / correctness classification through the **existing** oracle (:mod:`accuracy`) and the
  pure sanitiser, and writes only the payload-free :class:`SanitizedRunResult`, evaluation and manifest.
  The raw answer text is used to score correctness and then discarded; it is never written to a file.

The trust boundary is the same one :mod:`sanitize` enforces: a raw prompt, response, tool payload,
credential or log never reaches an output file. Everything here is stdlib-only plus the offline Gate A
layer and the community oracle; no network, no AWS, no provider is touched.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from typing import Any, Final

from .canonical import digest
from .contracts import (
    CORRECTNESS_ELIGIBLE_COUNT,
    TURN_COUNT,
    ContractViolation,
    TerminalOutcomeCatalog,
    TerminalTurnDefinition,
)
from .evaluator import evaluate
from .oracle_versions import load_accuracy, oracle_digest, oracle_version
from .runner import (
    _evaluation_to_dict,
    _result_to_dict,
    _stat_plan_from_dict,
    _write_json,
    build_manifest,
    catalog_to_dict,
    manifest_to_dict,
)
from .sanitize import sanitize_run

__all__ = [
    "HOST_SIGNALS",
    "SCENARIO_TURNS_UNAVAILABLE",
    "TurnsUnavailable",
    "build_ce60_catalog",
    "correctness_eligibility",
    "emit_turns",
    "finalize",
    "load_scenario_turns",
    "main",
    "score_correctness",
]

SCENARIO_TURNS_UNAVAILABLE: Final[str] = "CE60_SCENARIO_TURNS_UNAVAILABLE"
"""Stable code emitted when the CE-60 scenario cannot provide 60 turns and no ``--turns-json`` was given."""


class TurnsUnavailable(RuntimeError):
    """Raised fail-closed when the 60 CE-60 turns cannot be resolved from any source.

    This is deliberately **not** a :class:`ContractViolation`: it is an input-resolution guard, not a
    provenance violation, and it carries the stable code :data:`SCENARIO_TURNS_UNAVAILABLE` so the caller
    can branch on the code and ask for an explicit ``--turns-json``.

    Attributes:
        code: Always :data:`SCENARIO_TURNS_UNAVAILABLE`.
    """

    def __init__(self, message: str) -> None:
        """Wrap the stable turns-unavailable code and a human-readable message.

        Args:
            message: A short description for logs and the CLI's stderr.
        """
        super().__init__(f"{SCENARIO_TURNS_UNAVAILABLE}: {message}")
        self.code = SCENARIO_TURNS_UNAVAILABLE

HOST_SIGNALS: Final[frozenset[str]] = frozenset({"success", "refused", "error", "timeout"})
"""The host signals the driver may report for a turn, mirrored from :mod:`sanitize`."""

_DEFAULT_RETRIEVAL_STATUS: Final[str] = "host_not_exposed"
"""OpenCode does not expose a CE-60 retrieval attempt/success count, so retrieval is host-not-exposed.

Marking it explicitly (rather than fabricating a zero) is what keeps Gate A honest: the evaluator then
reports retrieval as invalid and never returns ``go`` on a run whose retrieval evidence is absent.
"""

_METRICS_EVIDENCE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "metrics_schema_version",
        "metrics_status",
        "transform_count",
        "skipped_count",
        "previewed_count",
        "baseline_count",
        "refused_count",
        "filtered_count",
        "input_bytes",
        "output_bytes",
        "source_already_truncated_count",
        "fallback_count",
        "rerank_attempts",
        "rerank_successes",
        "rerank_failures",
        "retrieval_attempts",
        "retrieval_successes",
        "metrics_write_failed",
        "tool_classes",
    }
)

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
_POLICY_VALUE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9_.:,-]{0,1024}")
"""Declared tool-policy / scrubbed-env provenance: comma-separated identifiers only, never values."""
_MAX_OBSERVABILITY_TOOL_NAMES: Final[int] = 512
_MAX_OBSERVABILITY_TIMEOUT_IDS: Final[int] = TURN_COUNT
_MAX_OBSERVABILITY_NAME_LENGTH: Final[int] = 128


def _safe_evidence_scalars(value: Any, *, allowed: frozenset[str]) -> dict[str, object]:
    """Keep only explicitly allow-listed scalar evidence fields; drop all nested payload-shaped values."""
    if not isinstance(value, Mapping):
        return {}
    clean: dict[str, object] = {}
    for key, item in value.items():
        if key not in allowed:
            continue
        if isinstance(item, bool) or item is None or isinstance(item, (int, float, str)):
            clean[key] = item
    return clean


def _safe_observability_summary(value: Any) -> dict[str, object] | None:
    """Validate the versioned, bounded and payload-free session diagnostics."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ContractViolation("RESULT_NOT_SANITIZED", "observability_summary must be an object")

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
    for key, item in value.items():
        if not isinstance(key, str):
            continue
        if key in _OBSERVABILITY_SCALAR_KEYS:
            if item is None:
                clean[key] = None
                continue
            if key in enum_values:
                if not isinstance(item, str) or item not in enum_values[key]:
                    raise ContractViolation("RESULT_NOT_SANITIZED", f"observability_summary.{key} has an invalid value")
                clean[key] = item
                continue
            if key in integer_keys:
                if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                    raise ContractViolation(
                        "RESULT_NOT_SANITIZED", f"observability_summary.{key} must be a non-negative integer"
                    )
                clean[key] = item
                continue
            if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)):
                raise ContractViolation(
                    "RESULT_NOT_SANITIZED", f"observability_summary.{key} must be a finite number or null"
                )
            if item < 0:
                raise ContractViolation("RESULT_NOT_SANITIZED", f"observability_summary.{key} must be non-negative")
            clean[key] = item
            continue
        if key not in _OBSERVABILITY_LIST_KEYS:
            continue
        if not isinstance(item, (list, tuple)):
            raise ContractViolation("RESULT_NOT_SANITIZED", f"observability_summary.{key} must be a list")
        limit = _MAX_OBSERVABILITY_TIMEOUT_IDS if key != "tool_names" else _MAX_OBSERVABILITY_TOOL_NAMES
        if len(item) > limit:
            raise ContractViolation("RESULT_NOT_SANITIZED", f"observability_summary.{key} exceeds the limit of {limit} items")
        names: list[str] = []
        for entry in item:
            if (
                not isinstance(entry, str)
                or not entry
                or len(entry) > _MAX_OBSERVABILITY_NAME_LENGTH
                or not entry.isprintable()
                or any(character.isspace() for character in entry)
                or not all(character.isalnum() or character in "_.:-" for character in entry)
            ):
                raise ContractViolation("RESULT_NOT_SANITIZED", f"observability_summary.{key} contains an invalid name")
            names.append(entry)
        clean[key] = names
    return clean


def _model_provenance_evidence(value: Any) -> dict[str, object]:
    """Keep only the two model identifiers from host provenance; never persist nested payload data."""
    if not isinstance(value, Mapping):
        return {}
    provenance = value.get("model_provenance")
    if not isinstance(provenance, Mapping):
        return {}
    execution_model_id = provenance.get("execution_model_id")
    rerank_model_id = provenance.get("rerank_model_id")
    if execution_model_id is not None and not isinstance(execution_model_id, str):
        return {}
    if not isinstance(rerank_model_id, str) or not rerank_model_id.strip():
        return {}
    return {
        "model_provenance": {
            "execution_model_id": execution_model_id,
            "rerank_model_id": rerank_model_id,
        }
    }


def _tool_spec_evidence(observations: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    """Aggregate public schema/tool overhead separately; never estimate it from conversation tokens."""
    token_values = [obs.get("tool_spec_tokens") for obs in observations]
    count_values = [obs.get("tool_spec_count") for obs in observations]

    def valid(values: list[Any]) -> bool:
        return bool(values) and all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0
            for value in values
        )

    tokens_exposed = valid(token_values)
    counts_exposed = valid(count_values)
    return {
        "tool_spec_tokens": sum(token_values) if tokens_exposed else None,
        "tool_spec_tokens_status": "exposed" if tokens_exposed else _DEFAULT_RETRIEVAL_STATUS,
        "tool_spec_count": sum(count_values) if counts_exposed else None,
        "tool_spec_count_status": "exposed" if counts_exposed else _DEFAULT_RETRIEVAL_STATUS,
    }


def _scenario_root() -> str:
    """Return the absolute path to the community CE-60 scenario package root.

    Returns:
        The ``validation/community-plugin-A-B-D`` directory next to this package.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "community-plugin-A-B-D"))


def _with_scenario_path(scenario_root: str | None) -> tuple[str, bool]:
    """Insert the scenario root onto ``sys.path`` for a defensive import.

    Args:
        scenario_root: An explicit scenario root, or ``None`` to use the sibling default.

    Returns:
        A ``(root, added)`` pair; ``added`` is whether this call inserted the path (so the caller can
        remove it again).
    """
    root = scenario_root if scenario_root is not None else _scenario_root()
    added = os.path.isdir(root) and root not in sys.path
    if added:
        sys.path.insert(0, root)
    return root, added


def load_scenario_turns(scenario_root: str | None = None) -> list[dict[str, str]] | None:
    """Best-effort load of the 60 canonical CE-60 ``(turn_id, prompt)`` pairs from the scenario.

    The scenario module pulls heavy optional dependencies through its tool suite, so the import is
    defensive: any failure — module missing, dependency missing, or a turn count other than 60 — yields
    ``None`` so the caller falls back to an explicit ``--turns-json``. Only ``turn_id`` and ``prompt`` are
    read; no other scenario field is copied.

    Args:
        scenario_root: An explicit scenario package root, or ``None`` for the sibling default.

    Returns:
        A list of 60 ``{"turn_id", "prompt"}`` dicts in script order, or ``None`` when exactly 60 unique
        turns cannot be resolved.
    """
    root, added = _with_scenario_path(scenario_root)
    if not os.path.isdir(root):
        return None
    try:
        from src import scenario  # type: ignore[import-not-found]

        turns = [
            {"turn_id": turn.label, "prompt": turn.prompt}
            for turn in scenario.turns(total=TURN_COUNT)
        ]
    except Exception:  # noqa: BLE001 - any import/attr/dependency failure means fall back to --turns-json
        return None
    finally:
        if added and root in sys.path:
            sys.path.remove(root)
    ids = [t["turn_id"] for t in turns]
    if len(turns) != TURN_COUNT or len(set(ids)) != TURN_COUNT:
        return None
    if any(not isinstance(t["prompt"], str) or t["prompt"].strip() == "" for t in turns):
        return None
    return turns


def correctness_eligibility(
    turn_ids: Sequence[str], *, scenario_root: str | None = None
) -> frozenset[str]:
    """Return the subset of ``turn_ids`` that are correctness-eligible, from the community oracle.

    A turn is correctness-eligible exactly when the oracle (:mod:`accuracy`) carries scoreable
    expectations for its label. For the canonical CE-60 script this is exactly 30 of the 60 turns, which
    is what the terminal catalog requires.

    Args:
        turn_ids: The 60 turn ids to classify.
        scenario_root: An explicit scenario package root, or ``None`` for the sibling default.

    Returns:
        The frozenset of eligible turn ids.

    Raises:
        ContractViolation: ``ORACLE_MISMATCH`` when the oracle cannot be imported or does not yield
            exactly :data:`~opencode_gate_a.contracts.CORRECTNESS_ELIGIBLE_COUNT` eligible turns.
    """
    root, added = _with_scenario_path(scenario_root)
    try:
        from src import accuracy as community_accuracy  # type: ignore[import-not-found]

        accuracy = load_accuracy(community_accuracy)
        eligible = frozenset(
            turn_id for turn_id in turn_ids if accuracy._checks_for(turn_id) is not None
        )
    except ContractViolation:
        raise
    except Exception as exc:
        raise ContractViolation(
            "ORACLE_MISMATCH", f"CE-60 correctness oracle is unavailable: {exc}"
        ) from exc
    finally:
        if added and root in sys.path:
            sys.path.remove(root)
    if len(eligible) != CORRECTNESS_ELIGIBLE_COUNT:
        raise ContractViolation(
            "ORACLE_MISMATCH",
            f"oracle marked {len(eligible)} eligible turns, expected {CORRECTNESS_ELIGIBLE_COUNT}",
        )
    return eligible


def score_correctness(
    turn_id: str, answer_text: str, *, scenario_root: str | None = None
) -> bool:
    """Score one eligible turn's answer with the deterministic community oracle.

    Args:
        turn_id: The turn's canonical label.
        answer_text: The raw answer text (used only to score; never persisted).
        scenario_root: An explicit scenario package root, or ``None`` for the sibling default.

    Returns:
        ``True`` when the turn is materially correct (no critical oracle check failed), else ``False``.

    Raises:
        ContractViolation: ``ORACLE_MISMATCH`` when the oracle cannot be imported.
    """
    root, added = _with_scenario_path(scenario_root)
    try:
        from src import accuracy as community_accuracy  # type: ignore[import-not-found]

        accuracy = load_accuracy(community_accuracy)
        return bool(accuracy.score_turn(turn_id, answer_text).materially_correct)
    except ContractViolation:
        raise
    except Exception as exc:
        raise ContractViolation(
            "ORACLE_MISMATCH", f"CE-60 correctness oracle is unavailable: {exc}"
        ) from exc
    finally:
        if added and root in sys.path:
            sys.path.remove(root)


def build_ce60_catalog(
    turn_ids: Sequence[str],
    eligible: frozenset[str],
    *,
    version: str = "ce60-1.0",
) -> TerminalOutcomeCatalog:
    """Build the CE-60 terminal outcome catalog for ``turn_ids``.

    Every turn uses the deterministic-predicate matcher, accepts only ``completed_expected`` and forbids
    a fallback — CE-60 has no declared-fallback path. Exactly the ``eligible`` turns are
    correctness-eligible, which the catalog contract requires to be 30.

    Args:
        turn_ids: The 60 unique turn ids.
        eligible: The correctness-eligible subset.
        version: The catalog version label.

    Returns:
        A valid :class:`TerminalOutcomeCatalog`.
    """
    definitions = [
        TerminalTurnDefinition(
            turn_id=turn_id,
            terminal_matcher="deterministic_predicate",
            accepted_states=("completed_expected",),
            fallback_policy="not_allowed",
            correctness_eligible=turn_id in eligible,
        )
        for turn_id in turn_ids
    ]
    return TerminalOutcomeCatalog.build(
        terminal_catalog_id="ce60-opencode",
        version=version,
        turn_definitions=definitions,
    )


def _int_or_none(value: Any) -> int | None:
    """Coerce a raw observation value to a non-boolean int, or ``None`` when absent/invalid.

    Args:
        value: The raw value from a driver observation.

    Returns:
        The int value, or ``None``. A ``bool`` is rejected so ``True``/``False`` never reads as ``1``/``0``.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _metric_block(
    *,
    model_input: int | None,
    latency_ms: float | None,
    retrieval_attempts: int | None,
    retrieval_successes: int | None,
    correctness: float | None,
    retrieval_status: str | None = None,
) -> dict[str, Any]:
    """Assemble the payload-free per-turn metric block the sanitiser consumes.

    A ``None`` numeric reading is recorded as a collection failure (or, for retrieval, as the explicit
    ``host_not_exposed`` status) rather than a fabricated zero, so an unavailable metric is visible
    downstream. Retrieval is only reported as a real count when the host exposed one: when the driver
    supplies both an attempts and a successes reading, they are passed through as the ``retrieval``
    attempts/successes; otherwise retrieval is recorded as host-not-exposed.

    Args:
        model_input: Model-input tokens the host reported, or ``None`` when unavailable.
        latency_ms: Latency in milliseconds, or ``None`` when unavailable.
        retrieval_attempts: Observed retrieval attempts, or ``None`` when the host exposes no count.
        retrieval_successes: Observed retrieval successes, or ``None`` when the host exposes no count.
        correctness: The oracle correctness score for eligible turns, or ``None`` for non-eligible.

    Returns:
        The metric sub-mapping of a raw turn.
    """
    metrics: dict[str, Any] = {}
    if retrieval_status == "collection_failed":
        metrics["retrieval"] = {"status": "collection_failed"}
    elif retrieval_attempts is not None and retrieval_successes is not None:
        metrics["retrieval"] = {
            "attempts": int(retrieval_attempts),
            "successes": int(retrieval_successes),
        }
    else:
        metrics["retrieval"] = {"status": _DEFAULT_RETRIEVAL_STATUS}
    if model_input is not None:
        metrics["model_input"] = {"value": int(model_input)}
    else:
        metrics["model_input"] = {"status": "collection_failed"}
    if latency_ms is not None:
        metrics["latency"] = {"value": float(latency_ms)}
    else:
        metrics["latency"] = {"status": "collection_failed"}
    if correctness is not None:
        metrics["correctness"] = {"value": float(correctness)}
    return metrics


def _build_raw_run(
    *,
    run_id: str,
    observations: Sequence[dict[str, Any]],
    eligible: frozenset[str],
    host_evidence: dict[str, Any],
    scenario_root: str | None,
) -> dict[str, Any]:
    """Build the in-memory raw-run mapping from the driver's per-turn observations.

    Correctness for eligible turns is scored here from the raw answer text and injected as a metric
    value; the answer text itself is not copied into the mapping. The sanitiser downstream drops any
    stray forbidden field, so even a mis-supplied payload cannot escape to a file.

    Args:
        run_id: The run id.
        observations: The per-turn observations captured by the driver.
        eligible: The correctness-eligible turn ids.
        host_evidence: Payload-free host evidence (versions, arm, flags).
        scenario_root: The scenario root for the oracle.

    Returns:
        A raw-run mapping suitable for :func:`~opencode_gate_a.sanitize.sanitize_run`.

    Raises:
        ContractViolation: ``RESULT_NOT_SANITIZED`` for a malformed observation.
    """
    raw_turns: list[dict[str, Any]] = []
    for obs in observations:
        turn_id = obs.get("turn_id")
        if not isinstance(turn_id, str) or turn_id.strip() == "":
            raise ContractViolation("RESULT_NOT_SANITIZED", "observation is missing a turn_id")
        host_signal = obs.get("host_signal")
        if host_signal not in HOST_SIGNALS:
            raise ContractViolation(
                "RESULT_NOT_SANITIZED", f"turn {turn_id!r} has unknown host_signal: {host_signal!r}"
            )

        correctness: float | None = None
        if turn_id in eligible:
            answer_text = obs.get("answer_text") or ""
            correctness = (
                1.0
                if score_correctness(turn_id, str(answer_text), scenario_root=scenario_root)
                else 0.0
            )

        raw_turn: dict[str, Any] = {
            "turn_id": turn_id,
            "host_signal": host_signal,
            "matcher_passed": bool(obs.get("matcher_passed", False)),
            "metrics": _metric_block(
                model_input=obs.get("model_input"),
                latency_ms=obs.get("latency_ms"),
                retrieval_attempts=_int_or_none(obs.get("retrieval_attempts")),
                retrieval_successes=_int_or_none(obs.get("retrieval_successes")),
                retrieval_status=obs.get("retrieval_status") if isinstance(obs.get("retrieval_status"), str) else None,
                correctness=correctness,
            ),
        }
        error_code = obs.get("error_code")
        if isinstance(error_code, str) and error_code.strip():
            raw_turn["error_code"] = error_code
        raw_turns.append(raw_turn)

    return {"run_id": run_id, "turns": raw_turns, "host_evidence": host_evidence}


def _live_manifest_kwargs(live: dict[str, Any]) -> dict[str, Any]:
    """Translate the driver's ``live`` block into :func:`build_manifest` live keyword arguments.

    Args:
        live: The live admission/budget/price/stat-plan block.

    Returns:
        The keyword arguments binding the live preconditions into the manifest.

    Raises:
        ContractViolation: ``RUN_MANIFEST_INVALID`` when a required live field is missing.
    """
    required = (
        "execution_admission_id",
        "admission_policy_digest",
        "budget_ceiling_units",
        "budget_currency_or_unit",
        "price_provenance",
        "stat_plan",
    )
    missing = [name for name in required if live.get(name) in (None, "")]
    if missing:
        raise ContractViolation(
            "RUN_MANIFEST_INVALID", f"live run manifest is missing: {', '.join(missing)}"
        )
    return {
        "backend_id": "remote-explicit",
        "budget_status": "configured",
        "sample_plan_status": "approved",
        "execution_admission_id": live["execution_admission_id"],
        "admission_policy_digest": live["admission_policy_digest"],
        "budget_ceiling_units": float(live["budget_ceiling_units"]),
        "budget_currency_or_unit": live["budget_currency_or_unit"],
        "price_provenance": live["price_provenance"],
        "stat_plan": _stat_plan_from_dict(live["stat_plan"]),
    }


def finalize(payload: dict[str, Any]) -> dict[str, Any]:
    """Sanitize, evaluate and persist one arm's run — writing only payload-free artefacts.

    The driver hands over the per-turn observations it kept in memory. This function builds the CE-60
    catalog, scores correctness through the oracle, sanitises the run (dropping any payload), builds the
    fail-closed manifest, evaluates deterministically, and writes the catalog, manifest, sanitized
    result, evaluation and an evidence summary into ``output_dir``. No prompt, answer, tool payload,
    credential or log is written.

    Args:
        payload: The driver payload — ``mode`` (``offline``/``live``), ``arm``, ``run_id``, ``host``,
            ``host_version``, ``model_id``, ``host_provenance`` (payload-free model identities and
            tuning), ``observability_summary`` (bounded session counts/flags/names), ``output_dir``, ``turns``
            (60 observations), an optional ``scenario_root`` and, for live, a ``live`` block.

    Returns:
        A payload-free evidence summary dict (ids, digests, metric validity, output paths).

    Raises:
        ContractViolation: Any fail-closed condition surfaced while classifying, sanitising, building
            the manifest or evaluating.
    """
    mode = payload.get("mode", "offline")
    arm = payload["arm"]
    run_id = payload["run_id"]
    host = payload["host"]
    host_version = payload["host_version"]
    model_id = payload["model_id"]
    output_dir = payload["output_dir"]
    scenario_root = payload.get("scenario_root")
    observations = payload["turns"]

    if not isinstance(observations, list) or len(observations) != TURN_COUNT:
        raise ContractViolation(
            "RESULT_NOT_SANITIZED",
            f"a run must carry exactly {TURN_COUNT} turn observations (never truncate)",
        )
    turn_ids = [obs.get("turn_id") for obs in observations]
    if any(not isinstance(tid, str) or not tid.strip() for tid in turn_ids):
        raise ContractViolation("RESULT_NOT_SANITIZED", "every observation needs a turn_id")
    if len(set(turn_ids)) != TURN_COUNT:
        raise ContractViolation("RESULT_NOT_SANITIZED", "turn ids must be unique across the run")

    eligible = correctness_eligibility(turn_ids, scenario_root=scenario_root)  # type: ignore[arg-type]
    catalog = build_ce60_catalog(turn_ids, eligible)  # type: ignore[arg-type]

    retrieval_exposed = any(
        _int_or_none(obs.get("retrieval_attempts")) is not None for obs in observations
    )
    raw_observability = payload.get("observability_summary", payload.get("observability"))
    if raw_observability is None:
        host_provenance = payload.get("host_provenance")
        if isinstance(host_provenance, Mapping):
            raw_observability = host_provenance.get(
                "observability_summary", host_provenance.get("observability")
            )
    observability_summary = _safe_observability_summary(raw_observability)
    host_evidence: dict[str, Any] = {
        "host": host,
        "host_version": host_version,
        "arm": arm,
        "model_id": model_id,
        "execution_mode": mode,
        "retrieval_metric": "exposed" if retrieval_exposed else _DEFAULT_RETRIEVAL_STATUS,
        "metrics_status": _DEFAULT_RETRIEVAL_STATUS,
        **_tool_spec_evidence(observations),
        **_model_provenance_evidence(payload.get("host_provenance")),
        **_safe_evidence_scalars(payload.get("metrics_summary"), allowed=_METRICS_EVIDENCE_KEYS),
        # Which oracle scored correctness, so a result always says what judged it (see oracle_versions).
        "oracle_version": oracle_version(),
        "oracle_digest": oracle_digest(),
    }
    if observability_summary is not None:
        host_evidence["observability"] = observability_summary
    # Declared, arm-identical tool policy and the names (never values) of scrubbed env overrides.
    for policy_key in ("disabled_tools", "removed_env_keys", "system_prompt_digest", "plugin_options"):
        policy_value = payload.get(policy_key)
        if isinstance(policy_value, str) and _POLICY_VALUE.fullmatch(policy_value):
            host_evidence[policy_key] = policy_value
    raw_run = _build_raw_run(
        run_id=run_id,
        observations=observations,
        eligible=eligible,
        host_evidence=host_evidence,
        scenario_root=scenario_root,
    )

    common = {
        "run_id": run_id,
        "arm": arm,
        "host": host,
        "host_version": host_version,
        "model_id": model_id,
        "capability_manifest_digest": digest(
            ["ce60-capability", arm, host, host_version, model_id]
        ),
        "fixture_catalog_digest": digest(["ce60-fixture", sorted(turn_ids)]),
        "terminal_catalog_digest": catalog.digest,
        "order_seed": f"ce60-sequential:{run_id}",
        "turn_ids": list(turn_ids),
    }

    # The sanitized result is evaluated against an offline/replay manifest, because the offline
    # evaluator never evaluates a live manifest directly: a live run is evaluated as a replay of itself.
    # A live run additionally produces a separate live provenance manifest that binds the admission,
    # budget, price and approved stat plan.
    eval_manifest = build_manifest(
        execution_mode="offline" if mode == "offline" else "replay",
        backend_id="local-fake",
        budget_status="not_applicable",
        sample_plan_status="not_applicable",
        **common,  # type: ignore[arg-type]
    )
    live_manifest = None
    if mode == "live":
        live_manifest = build_manifest(
            execution_mode="live",
            **_live_manifest_kwargs(payload["live"]),
            **common,  # type: ignore[arg-type]
        )

    result = sanitize_run(raw_run, manifest_digest=eval_manifest.manifest_digest, catalog=catalog)
    evaluation = evaluate(result, manifest=eval_manifest, catalog=catalog)
    manifest = eval_manifest

    os.makedirs(output_dir, exist_ok=True)
    catalog_path = os.path.join(output_dir, f"{arm}-catalog.json")
    manifest_path = os.path.join(output_dir, f"{arm}-manifest.json")
    sanitized_path = os.path.join(output_dir, f"{arm}-sanitized.json")
    evaluation_path = os.path.join(output_dir, f"{arm}-evaluation.json")
    evidence_path = os.path.join(output_dir, f"{arm}-evidence.json")

    _write_json(catalog_path, catalog_to_dict(catalog))
    _write_json(manifest_path, manifest_to_dict(manifest, scope="full", host_evidence=host_evidence))
    _write_json(sanitized_path, _result_to_dict(result))
    _write_json(evaluation_path, _evaluation_to_dict(evaluation))
    live_manifest_path = None
    if live_manifest is not None:
        live_manifest_path = os.path.join(output_dir, f"{arm}-live-manifest.json")
        _write_json(live_manifest_path, manifest_to_dict(live_manifest, scope="full", host_evidence=host_evidence))

    evidence: dict[str, Any] = {
        "arm": arm,
        "run_id": run_id,
        "requested_mode": mode,
        "evaluation_mode": manifest.execution_mode,
        "host": host,
        "host_version": host_version,
        "model_id": model_id,
        "manifest_digest": manifest.manifest_digest,
        "live_manifest_digest": None if live_manifest is None else live_manifest.manifest_digest,
        "result_digest": result.result_digest,
        "evaluation_digest": evaluation.evaluation_digest,
        "turns_attempted": evaluation.turns_attempted,
        "turns_completed": evaluation.turns_completed,
        "error_count": evaluation.error_count,
        "correctness_numerator": evaluation.correctness_numerator,
        "correctness_denominator": evaluation.correctness_denominator,
        "model_input_tokens_median": evaluation.model_input_tokens_median,
        "metric_validity": [
            {"metric_name": mv.metric_name, "status": mv.status}
            for mv in sorted(evaluation.metric_validity, key=lambda m: m.metric_name)
        ],
        "host_evidence": host_evidence,
        "paths": {
            "catalog": catalog_path,
            "manifest": manifest_path,
            "live_manifest": live_manifest_path,
            "sanitized": sanitized_path,
            "evaluation": evaluation_path,
            "evidence": evidence_path,
        },
    }
    _write_json(evidence_path, evidence)
    return evidence


def emit_turns(turns_json: str | None = None, *, scenario_root: str | None = None) -> dict[str, Any]:
    """Resolve the 60 CE-60 ``(turn_id, prompt)`` pairs, from ``--turns-json`` or the scenario.

    Resolution order: an explicit ``turns_json`` file first, then the importable scenario. The result is
    validated to be exactly 60 unique, non-empty turns; otherwise the function fails closed so the
    caller supplies an explicit list rather than running a truncated benchmark.

    Args:
        turns_json: Path to a JSON file carrying the turns (an array or ``{"turns": [...]}``), or ``None``.
        scenario_root: An explicit scenario package root, or ``None`` for the sibling default.

    Returns:
        A ``{"turns": [...], "source": ...}`` dict with exactly 60 ``{"turn_id", "prompt"}`` entries.

    Raises:
        ContractViolation: ``CE60_SCENARIO_TURNS_UNAVAILABLE`` when neither source yields 60 valid turns.
    """
    turns: list[dict[str, str]] | None = None
    source = ""
    if turns_json:
        with open(turns_json, encoding="utf-8") as handle:
            data = json.load(handle)
        raw = data["turns"] if isinstance(data, dict) and "turns" in data else data
        if isinstance(raw, list):
            turns = [
                {"turn_id": str(item["turn_id"]), "prompt": str(item["prompt"])}
                for item in raw
                if isinstance(item, dict) and "turn_id" in item and "prompt" in item
            ]
            source = f"turns-json:{turns_json}"
    if turns is None:
        turns = load_scenario_turns(scenario_root)
        source = "scenario"

    if turns is None:
        raise TurnsUnavailable(
            "CE-60 turns are unavailable; pass --turns-json with exactly 60 turns (never truncate)"
        )
    ids = [t["turn_id"] for t in turns]
    if len(turns) != TURN_COUNT or len(set(ids)) != TURN_COUNT:
        raise TurnsUnavailable(
            f"expected exactly {TURN_COUNT} unique turns, got {len(turns)} ({len(set(ids))} unique)"
        )
    if any(not t["prompt"].strip() for t in turns):
        raise TurnsUnavailable("every turn must carry a non-empty prompt")
    return {"turns": turns, "source": source}


# --- CLI -------------------------------------------------------------------------------------------


def _cmd_emit_turns(args: argparse.Namespace) -> int:
    """Handle ``emit-turns``: print the 60 turn ids and prompts as JSON to stdout."""
    result = emit_turns(args.turns_json, scenario_root=args.scenario_root)
    json.dump(result, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


def _cmd_finalize(args: argparse.Namespace) -> int:
    """Handle ``finalize``: read the driver payload from stdin, write artefacts, print evidence."""
    payload = json.load(sys.stdin)
    evidence = finalize(payload)
    json.dump(evidence, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the host-bridge subcommands.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="opencode_gate_a.host_driver",
        description="Payload-free CE-60 host bridge: emit-turns / finalize (no live run here).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    emit = sub.add_parser("emit-turns", help="Resolve and print the 60 CE-60 turn ids and prompts.")
    emit.add_argument("--turns-json", default=None, help="Explicit turns file (array or {turns:[...]}).")
    emit.add_argument("--scenario-root", default=None, help="Override the CE-60 scenario package root.")
    emit.set_defaults(func=_cmd_emit_turns)

    fin = sub.add_parser("finalize", help="Sanitize/evaluate/persist one arm from a stdin payload.")
    fin.set_defaults(func=_cmd_finalize)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the host-bridge CLI.

    Args:
        argv: The argument vector, or ``None`` to read ``sys.argv``.

    Returns:
        ``0`` on success, ``3`` for the fail-closed turn-resolution guard, ``4`` for a contract
        violation.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except TurnsUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except ContractViolation as exc:
        print(str(exc), file=sys.stderr)
        return 4


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
