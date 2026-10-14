"""Payload-free per-run metrics for the OpenCode bridge.

The recorder deliberately accepts only normalized names, terminal operation labels, byte counts and
boolean/categorical flags. It never receives tool text, prompts, answers, arguments, credentials or
raw logs. Records are JSONL so a partially completed arm still leaves a readable prefix for the host
driver to summarize.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Final

__all__ = ["METRICS_SCHEMA_VERSION", "MetricsRecorder"]

METRICS_SCHEMA_VERSION: Final[int] = 1
_OPERATIONS: Final[frozenset[str]] = frozenset({"skipped", "previewed", "baseline", "refused"})
_RERANK_RESULTS: Final[frozenset[str]] = frozenset({"not_attempted", "success", "failure"})


class MetricsRecorder:
    """Append sanitized transform/retrieval observations and expose an aggregate summary."""

    def __init__(self, path: str | None = None) -> None:
        """Initialize a recorder and atomically create a fresh JSONL file when configured."""
        self._path = Path(path).expanduser() if path and path.strip() else None
        self._operations: Counter[str] = Counter()
        self._tool_classes: set[str] = set()
        self._transform_count = 0
        self._filtered_count = 0
        self._input_bytes = 0
        self._output_bytes = 0
        self._source_already_truncated_count = 0
        self._fallback_count = 0
        self._rerank_attempts = 0
        self._rerank_successes = 0
        self._rerank_failures = 0
        self._retrieval_attempts = 0
        self._retrieval_successes = 0
        self._write_failed = False
        if self._path is not None:
            self._initialize_file()

    def _initialize_file(self) -> None:
        """Atomically replace the per-run file with a payload-free readiness record."""
        assert self._path is not None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{self._path.name}.", suffix=".tmp", dir=self._path.parent
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {"event": "metrics_ready", "schema_version": METRICS_SCHEMA_VERSION},
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_name, self._path)
            finally:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)
        except OSError:
            self._write_failed = True

    def _append(self, record: Mapping[str, object]) -> None:
        """Append one complete JSONL record without allowing I/O to affect the transform."""
        if self._path is None:
            return
        try:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(record), separators=(",", ":"), ensure_ascii=False) + "\n")
                handle.flush()
        except OSError:
            self._write_failed = True

    def record_transform(
        self,
        *,
        tool_class: str,
        operation: str,
        input_bytes: int,
        output_bytes: int,
        filtered: bool,
        source_already_truncated: bool,
        fallback: bool,
        rerank_attempt: bool,
        rerank_result: str,
        retrieval_attempts: int,
        retrieval_successes: int,
    ) -> None:
        """Record one transform using only bounded counters, bytes and flags."""
        if operation not in _OPERATIONS:
            raise ValueError(f"unknown transform operation: {operation!r}")
        if rerank_result not in _RERANK_RESULTS:
            raise ValueError(f"unknown rerank result: {rerank_result!r}")
        if not isinstance(tool_class, str) or not tool_class.strip():
            raise ValueError("tool_class must be a non-empty normalized name")
        values = {
            "input_bytes": input_bytes,
            "output_bytes": output_bytes,
            "retrieval_attempts": retrieval_attempts,
            "retrieval_successes": retrieval_successes,
        }
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values.values()):
            raise ValueError("metric counters must be non-negative integers")

        self._operations[operation] += 1
        self._tool_classes.add(tool_class)
        self._transform_count += 1
        self._filtered_count += int(filtered)
        self._input_bytes += input_bytes
        self._output_bytes += output_bytes
        self._source_already_truncated_count += int(source_already_truncated)
        self._fallback_count += int(fallback)
        self._rerank_attempts += int(rerank_attempt)
        self._rerank_successes += int(rerank_result == "success")
        self._rerank_failures += int(rerank_result == "failure")
        self._retrieval_attempts = max(self._retrieval_attempts, retrieval_attempts)
        self._retrieval_successes = max(self._retrieval_successes, retrieval_successes)

        self._append(
            {
                "event": "transform",
                "schema_version": METRICS_SCHEMA_VERSION,
                "tool_class": tool_class,
                "operation": operation,
                "input_bytes": input_bytes,
                "output_bytes": output_bytes,
                "filtered": filtered,
                "source_already_truncated": source_already_truncated,
                "fallback": fallback,
                "rerank_attempt": rerank_attempt,
                "rerank_result": rerank_result,
                "retrieval_attempts": retrieval_attempts,
                "retrieval_successes": retrieval_successes,
            }
        )

    def record_retrieval(self, *, success: bool) -> None:
        """Record one retrieval attempt without recording a reference, argument or returned content."""
        self._retrieval_attempts += 1
        self._retrieval_successes += int(success)
        self._append(
            {
                "event": "retrieval",
                "schema_version": METRICS_SCHEMA_VERSION,
                "success": success,
                "retrieval_attempts": self._retrieval_attempts,
                "retrieval_successes": self._retrieval_successes,
            }
        )

    def summary(self) -> dict[str, object]:
        """Return a flat, payload-free aggregate suitable for host evidence."""
        return {
            "metrics_schema_version": METRICS_SCHEMA_VERSION,
            "metrics_status": "exposed" if self._path is not None else "not_configured",
            "transform_count": self._transform_count,
            "skipped_count": self._operations["skipped"],
            "previewed_count": self._operations["previewed"],
            "baseline_count": self._operations["baseline"],
            "refused_count": self._operations["refused"],
            "filtered_count": self._filtered_count,
            "input_bytes": self._input_bytes,
            "output_bytes": self._output_bytes,
            "source_already_truncated_count": self._source_already_truncated_count,
            "fallback_count": self._fallback_count,
            "rerank_attempts": self._rerank_attempts,
            "rerank_successes": self._rerank_successes,
            "rerank_failures": self._rerank_failures,
            "retrieval_attempts": self._retrieval_attempts,
            "retrieval_successes": self._retrieval_successes,
            "metrics_write_failed": self._write_failed,
            "tool_classes": ",".join(sorted(self._tool_classes)),
        }
