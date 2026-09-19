"""Per-call and per-turn measurement for the validation harness.

The measurement problem this module solves is the one flagged in the specs: the SDK's
``accumulated_usage`` reports the running total for the whole session, so a strategy
that reduces the cost of every individual call still shows a rising number. Comparing
sessions on that figure makes an improvement look like a regression.

So the harness measures the thing the strategies actually change: **input tokens per
model call**, captured at the boundary where the call is assembled. A wrap-phase
middleware on ``InvokeModelStage`` sees the final ``messages`` and ``tool_specs`` — after
every plugin has had its say — and the provider's own ``usage`` for that same call.
That gives both a provider-reported figure and an independently computed one, and the
two are reported side by side so a discrepancy is visible rather than hidden.

Timing is captured at three granularities, because they answer different questions:
``model_seconds`` isolates provider latency, ``turn_seconds`` includes tool execution and
the plugins' critical path, and ``wall_seconds`` covers the whole run.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import re
import time
from collections.abc import AsyncGenerator, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import mean, median
from typing import Any

from .config import estimate_tokens

# --- Bedrock invocation-log correlation ----------------------------------------------
#
# The provider's usage is read off the stream, which means a call that raises reports no
# usage at all while Bedrock still meters what it read. Bedrock's own invocation log does
# not have that blind spot, so the harness stamps every request with enough identity to be
# joined against it: ``requestMetadata`` travels to the log, and the response's request id
# travels back into the call record. The join then works in both directions, which is what
# makes a missing record distinguishable from a mismeasured one.

_TAG_ALLOWED = re.compile(r"[^a-zA-Z0-9\s:_@$#=/+,.-]")
"""Characters Bedrock rejects in a ``requestMetadata`` key or value."""


@dataclass
class _CallTag:
    """Identity of the call in flight, shared with the botocore hooks.

    Held in a :class:`contextvars.ContextVar` rather than passed down: the request is issued
    from a worker thread (``asyncio.to_thread`` in the Bedrock provider), and ``to_thread``
    copies the calling context into it. The mapping is copied but the objects are not, so
    ``request_ids`` mutated on the worker thread is visible to the middleware that set it.
    """

    metadata: dict[str, str]
    request_ids: list[str] = field(default_factory=list)


_CALL_TAG: contextvars.ContextVar[_CallTag | None] = contextvars.ContextVar("validation_call_tag", default=None)


def _sanitize_tag(value: Any) -> str:
    """Return ``value`` as a Bedrock-acceptable metadata string, truncated to the 256 limit."""
    return _TAG_ALLOWED.sub("-", str(value))[:256]


def install_log_tagging(client: Any, *, role: str) -> None:
    """Stamp every Converse request on ``client`` with its harness identity.

    Registers two botocore hooks on the client's own event emitter — the same mechanism the
    SDK uses for bearer auth, so it needs no change to the provider:

    * ``provide-client-params`` injects ``requestMetadata``, which Bedrock copies verbatim
      into the invocation log entry.
    * ``after-call`` reads ``ResponseMetadata.RequestId`` back, so the harness record points
      at the log entry as well as the other way round.

    Args:
        client: The ``bedrock-runtime`` boto3 client to instrument.
        role: Which model this client serves — e.g. ``"agent"`` or a diagnostic role.
            Recorded in the log so a strategy's own model calls stay separable from the
            agent's, which is the distinction the per-role cost columns depend on.
    """

    def inject_metadata(params: dict[str, Any], **_: Any) -> None:
        if "requestMetadata" in params:
            return
        tag = _CALL_TAG.get()
        metadata = {"harness": _sanitize_tag(RUN_TAG), "role": _sanitize_tag(role)}
        if tag is not None:
            metadata.update(tag.metadata)
        params["requestMetadata"] = metadata

    def capture_request_id(parsed: dict[str, Any] | None = None, **_: Any) -> None:
        tag = _CALL_TAG.get()
        if tag is None or role != "agent" or not isinstance(parsed, dict):
            return
        request_id = (parsed.get("ResponseMetadata") or {}).get("RequestId")
        if request_id:
            tag.request_ids.append(str(request_id))

    for operation in ("Converse", "ConverseStream"):
        client.meta.events.register(f"provide-client-params.bedrock-runtime.{operation}", inject_metadata)
        client.meta.events.register(f"after-call.bedrock-runtime.{operation}", capture_request_id)


RUN_TAG = "untagged"
"""Value of the ``harness`` metadata key, so one run's log entries are filterable.

Set once from ``--tag`` by :func:`set_run_tag`. A module-level value rather than plumbing:
the botocore hooks are registered per client, far from the argument parser, and every
configuration in a run shares the same tag by definition.
"""


def set_run_tag(tag: str | None) -> None:
    """Name the run in every invocation log entry it produces."""
    global RUN_TAG
    RUN_TAG = _sanitize_tag(tag or f"run-{int(time.time())}")


@contextlib.contextmanager
def tag_calls(**metadata: Any) -> Iterator[None]:
    """Stamp Bedrock calls made inside this block with ``metadata``.

    For traffic that is not one of the agent's measured calls — the harness's own diagnostic
    passes, above all. Those are billed like any other call but are deliberately left out of
    the strategy's cost, so without a distinct stamp they surface in the invocation log as an
    unexplained excess over what the run reported.

    A ``role`` given here overrides the client's registered role.
    """
    token = _CALL_TAG.set(_CallTag(metadata={key: _sanitize_tag(value) for key, value in metadata.items()}))
    try:
        yield
    finally:
        _CALL_TAG.reset(token)


@dataclass
class ModelCallRecord:
    """One model invocation, as it was actually sent to the provider."""

    turn_index: int
    call_index: int
    message_count: int
    message_chars: int
    tool_spec_count: int
    tool_spec_chars: int
    system_prompt_chars: int
    model_seconds: float = 0.0
    usage_input_tokens: int | None = None
    usage_output_tokens: int | None = None
    usage_total_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    stop_reason: str | None = None
    tool_names_sent: list[str] = field(default_factory=list)
    error: str | None = None
    request_ids: list[str] = field(default_factory=list)
    """Bedrock request ids this call produced, in order.

    Usually one. More than one means the provider retried inside a single logical call, and
    each attempt is metered separately — the reason CloudWatch can report more invocations
    than the harness reports calls.
    """
    started_at: str | None = None
    ended_at: str | None = None
    """Absolute UTC bounds of the call, so a record is locatable in the log without an id.

    Needed for exactly the calls that have no id: one that failed before a response arrived
    is billed and logged, but never yielded a request id back to the harness.
    """

    @property
    def estimated_message_tokens(self) -> int:
        return estimate_tokens("x" * self.message_chars)

    @property
    def estimated_tool_spec_tokens(self) -> int:
        return estimate_tokens("x" * self.tool_spec_chars)

    @property
    def estimated_input_tokens(self) -> int:
        """Our own estimate of the call's input size, independent of the provider."""
        return estimate_tokens("x" * (self.message_chars + self.tool_spec_chars + self.system_prompt_chars))

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "call_index": self.call_index,
            "message_count": self.message_count,
            "message_chars": self.message_chars,
            "tool_spec_count": self.tool_spec_count,
            "tool_spec_chars": self.tool_spec_chars,
            "system_prompt_chars": self.system_prompt_chars,
            "estimated_input_tokens": self.estimated_input_tokens,
            "estimated_message_tokens": self.estimated_message_tokens,
            "estimated_tool_spec_tokens": self.estimated_tool_spec_tokens,
            "model_seconds": round(self.model_seconds, 3),
            "usage_input_tokens": self.usage_input_tokens,
            "usage_output_tokens": self.usage_output_tokens,
            "usage_total_tokens": self.usage_total_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "stop_reason": self.stop_reason,
            "tool_names_sent": self.tool_names_sent,
            "error": self.error,
            "request_ids": self.request_ids,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }


@dataclass
class TurnRecord:
    """One user turn: the prompt, its wall time, and the calls it produced."""

    index: int
    label: str
    prompt: str
    turn_seconds: float = 0.0
    response_chars: int = 0
    response_text: str = ""
    """Kept so answer quality can be compared, not just cost.

    Relevance filtering does not claim to save tokens — it claims the same budget carries
    the passage that answers the question. That claim is only checkable against the text.
    """
    tool_calls: list[str] = field(default_factory=list)
    live_message_count: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "label": self.label,
            "prompt": self.prompt,
            "turn_seconds": round(self.turn_seconds, 3),
            "response_chars": self.response_chars,
            "response_text": self.response_text,
            "tool_calls": self.tool_calls,
            "live_message_count": self.live_message_count,
            "error": self.error,
        }


class RunCollector:
    """Accumulates every measurement for a single configuration's run."""

    def __init__(self, config_name: str) -> None:
        self.config_name = config_name
        self.calls: list[ModelCallRecord] = []
        self.log_tag: str = RUN_TAG
        """The ``harness`` metadata value its calls were stamped with, kept for the join."""
        self.turns: list[TurnRecord] = []
        self.wall_seconds: float = 0.0
        self.plugin_counters: dict[str, Any] = {}
        self.accuracy: dict[str, Any] = {}
        self.errors: list[str] = []
        # Set only by a resumed run: the turn a fresh agent took over on, and how many messages the
        # session restored into it. Without the message count, "the graph recovered" and "there was
        # nothing to recover" read identically.
        self.resumed_at_turn: int | None = None
        self.resumed_message_count: int | None = None
        self._turn_index = -1
        self._call_index = 0

    # --- turn lifecycle ---

    def begin_turn(self, index: int, label: str, prompt: str) -> TurnRecord:
        self._turn_index = index
        record = TurnRecord(index=index, label=label, prompt=prompt)
        self.turns.append(record)
        return record

    # --- middleware ---

    def middleware(self):
        """Return a wrap-phase handler for ``InvokeModelStage``.

        Wrap is the only phase that sees both the assembled context and every streamed
        event, which is what lets one handler report inputs and usage for the same call.
        Register it last so it observes the projection the plugins produced rather than
        the pre-plugin baseline.
        """

        async def handler(context: Any, next_fn: Any) -> AsyncGenerator[Any, None]:
            record = ModelCallRecord(
                turn_index=self._turn_index,
                call_index=self._call_index,
                message_count=len(context.messages),
                message_chars=_json_chars(context.messages),
                tool_spec_count=len(context.tool_specs or []),
                tool_spec_chars=_json_chars(context.tool_specs or []),
                system_prompt_chars=_json_chars(context.system_prompt),
                tool_names_sent=[
                    spec.get("name", "?") for spec in (context.tool_specs or []) if isinstance(spec, dict)
                ],
            )
            self._call_index += 1
            self.calls.append(record)

            # Set before the call so the botocore hooks, which run on the provider's worker
            # thread, see this call's identity rather than the previous one's.
            tag = _CallTag(
                metadata={
                    "config": _sanitize_tag(self.config_name),
                    "turn": _sanitize_tag(record.turn_index),
                    "call": _sanitize_tag(record.call_index),
                }
            )
            _CALL_TAG.set(tag)

            record.started_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
            started = time.perf_counter()
            try:
                async for event in next_fn(context):
                    _absorb_usage(record, event)
                    yield event
            except Exception as error:  # noqa: BLE001 - record and re-raise
                record.error = f"{type(error).__name__}: {error}"
                self.errors.append(record.error)
                raise
            finally:
                record.model_seconds = time.perf_counter() - started
                record.ended_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
                record.request_ids = list(tag.request_ids)

        return handler

    # --- aggregation ---

    def summary(self) -> dict[str, Any]:
        """Return the comparable figures for this run."""
        est_inputs = [call.estimated_input_tokens for call in self.calls]
        usage_inputs = [c.usage_input_tokens for c in self.calls if c.usage_input_tokens is not None]
        usage_outputs = [c.usage_output_tokens for c in self.calls if c.usage_output_tokens is not None]
        spec_tokens = [call.estimated_tool_spec_tokens for call in self.calls]
        msg_tokens = [call.estimated_message_tokens for call in self.calls]
        model_times = [call.model_seconds for call in self.calls]
        turn_times = [turn.turn_seconds for turn in self.turns]

        tool_uses: dict[str, int] = {}
        for turn in self.turns:
            for name in turn.tool_calls:
                tool_uses[name] = tool_uses.get(name, 0) + 1

        return {
            "config": self.config_name,
            "wall_seconds": round(self.wall_seconds, 3),
            "turns": len(self.turns),
            "resumed_at_turn": self.resumed_at_turn,
            "resumed_message_count": self.resumed_message_count,
            "model_calls": len(self.calls),
            "tool_uses_total": sum(tool_uses.values()),
            "tool_uses": dict(sorted(tool_uses.items(), key=lambda item: -item[1])),
            "retrievals": tool_uses.get("retrieve_offloaded_content", 0),
            # Retrieval count is the variable that explains the token totals: every
            # retrieval appends a large result to the history, which then rides along on
            # every subsequent call. A strategy that invites retrieval pays for it twice.
            "tokens": {
                "estimated_input_total": sum(est_inputs),
                "estimated_input_mean_per_call": _safe(mean, est_inputs),
                "estimated_input_median_per_call": _safe(median, est_inputs),
                "estimated_input_max_per_call": max(est_inputs, default=0),
                "usage_input_total": sum(usage_inputs),
                "usage_input_mean_per_call": _safe(mean, usage_inputs),
                "usage_input_max_per_call": max(usage_inputs, default=0),
                "usage_output_total": sum(usage_outputs),
                "tool_spec_tokens_total": sum(spec_tokens),
                "tool_spec_tokens_mean_per_call": _safe(mean, spec_tokens),
                "message_tokens_total": sum(msg_tokens),
                "message_tokens_mean_per_call": _safe(mean, msg_tokens),
                "cache_read_total": sum(c.cache_read_tokens or 0 for c in self.calls),
                "cache_write_total": sum(c.cache_write_tokens or 0 for c in self.calls),
            },
            "timing": {
                "model_seconds_total": round(sum(model_times), 3),
                "model_seconds_mean_per_call": _safe(mean, model_times, digits=3),
                "model_seconds_max_per_call": round(max(model_times, default=0.0), 3),
                "turn_seconds_total": round(sum(turn_times), 3),
                "turn_seconds_mean": _safe(mean, turn_times, digits=3),
                "turn_seconds_median": _safe(median, turn_times, digits=3),
                "turn_seconds_max": round(max(turn_times, default=0.0), 3),
                "overhead_seconds": round(sum(turn_times) - sum(model_times), 3),
            },
            "plugin_counters": self.plugin_counters,
            "accuracy": self.accuracy,
            "errors": self.errors,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "turns": [turn.to_dict() for turn in self.turns],
            "calls": [call.to_dict() for call in self.calls],
        }


# --- helpers ------------------------------------------------------------------------


def _json_chars(payload: Any) -> int:
    """Character count of ``payload`` serialized, as a stable proxy for wire size."""
    if payload is None:
        return 0
    if isinstance(payload, str):
        return len(payload)
    try:
        return len(json.dumps(payload, ensure_ascii=False, default=str))
    except Exception:  # noqa: BLE001 - measurement must never break a run
        return len(str(payload))


def _absorb_usage(record: ModelCallRecord, event: Any) -> None:
    """Pull usage off a ``ModelStopReason`` event if this is one.

    The event carries a 4-tuple under the ``"stop"`` key: (stop_reason, message, usage,
    metrics). Everything is defensive because an event shape change should degrade the
    measurement, not the run.
    """
    try:
        payload = event if isinstance(event, dict) else getattr(event, "_data", None)
        if not isinstance(payload, dict) or "stop" not in payload:
            return
        stop = payload["stop"]
        if not isinstance(stop, tuple) or len(stop) < 3:
            return
        record.stop_reason = str(stop[0])
        usage = stop[2]
        if not isinstance(usage, dict):
            return
        record.usage_input_tokens = usage.get("inputTokens")
        record.usage_output_tokens = usage.get("outputTokens")
        record.usage_total_tokens = usage.get("totalTokens")
        record.cache_read_tokens = usage.get("cacheReadInputTokens")
        record.cache_write_tokens = usage.get("cacheWriteInputTokens")
    except Exception:  # noqa: BLE001
        return


def _safe(fn: Any, values: list[Any], digits: int = 1) -> float:
    if not values:
        return 0.0
    return round(float(fn(values)), digits)


def aggregate(collectors: list[RunCollector]) -> dict[str, Any]:
    """Average several replays of the same configuration into one comparable entry.

    A single replay is not a measurement. The agent chooses its own tool path, and with
    ``temperature`` unavailable on Opus 4.8 there is no way to pin that choice — so two
    replays of the same configuration can differ by more than the strategies differ from
    each other. Observed directly: the graph swung from ~23% cheaper than baseline on one
    full run to ~24% cheaper on the next, because the agent happened to take an extra
    retrieval path.

    So numeric fields are reported as a mean with the spread alongside. When the spread
    across replays of one configuration overlaps the gap between configurations, the
    comparison has not resolved anything, and ``spread_pct`` is what makes that visible
    instead of leaving it to be inferred.
    """
    if not collectors:
        return {}
    if len(collectors) == 1:
        entry = collectors[0].to_dict()
        entry["repeats"] = 1
        return entry

    summaries = [collector.summary() for collector in collectors]
    merged: dict[str, Any] = {
        "config": summaries[0]["config"],
        "repeats": len(summaries),
        "errors": [error for summary in summaries for error in summary["errors"]],
    }

    for key in ("wall_seconds", "turns", "model_calls", "tool_uses_total", "retrievals"):
        merged[key] = _mean_of([summary.get(key, 0) for summary in summaries])
        merged[f"{key}_spread"] = _spread_of([summary.get(key, 0) for summary in summaries])

    for group in ("tokens", "timing"):
        merged[group] = {}
        for key in summaries[0][group]:
            values = [summary[group].get(key, 0) or 0 for summary in summaries]
            merged[group][key] = _mean_of(values)
            merged[group][f"{key}_spread_pct"] = _spread_pct(values)

    # Counters are strategy evidence, not a quantity to average: the maximum across
    # replays answers "did this ever engage", which is the question they are read for.
    merged["plugin_counters"] = _merge_counters([summary.get("plugin_counters", {}) for summary in summaries])

    # Accuracy averages, and its spread matters as much as its mean: a strategy whose
    # correctness swings between replays is a different risk from one that is steadily
    # slightly worse, and only the spread separates them.
    accuracies = [summary.get("accuracy", {}) for summary in summaries]
    scored = [item for item in accuracies if item]
    if scored:
        merged["accuracy"] = {
            "weighted_accuracy": _mean_of([item["weighted_accuracy"] for item in scored]),
            "weighted_accuracy_spread_pct": _spread_pct([item["weighted_accuracy"] for item in scored]),
            "material_correctness": _mean_of([item["material_correctness"] for item in scored]),
            "turns_materially_correct": _mean_of([item["turns_materially_correct"] for item in scored]),
            "turns_scored": scored[0]["turns_scored"],
            "critical_failures_total": _mean_of([item["critical_failures_total"] for item in scored]),
            "per_turn": _mean_per_turn_accuracy(scored),
        }

    return {
        "summary": merged,
        "turns": _mean_turns([collector.turns for collector in collectors]),
        "calls": [call.to_dict() for call in collectors[0].calls],
        "repeats": len(collectors),
        "per_repeat": [collector.to_dict() for collector in collectors],
    }


def _mean_of(values: list[Any]) -> float | int:
    numeric = [value for value in values if isinstance(value, (int, float))]
    if not numeric:
        return 0
    average = mean(numeric)
    return round(average, 2) if isinstance(average, float) else average


def _spread_of(values: list[Any]) -> float:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    if len(numeric) < 2:
        return 0.0
    return round(max(numeric) - min(numeric), 2)


def _spread_pct(values: list[Any]) -> float:
    """Peak-to-peak spread as a percentage of the mean."""
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    if len(numeric) < 2:
        return 0.0
    average = mean(numeric)
    if not average:
        return 0.0
    return round((max(numeric) - min(numeric)) / average * 100.0, 1)


def _merge_counters(counter_sets: list[dict[str, Any]]) -> dict[str, Any]:
    """Take the maximum of each numeric counter across replays."""
    merged: dict[str, Any] = {}
    for counters in counter_sets:
        for section, payload in counters.items():
            if isinstance(payload, dict):
                target = merged.setdefault(section, {})
                for key, value in payload.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        target[key] = max(target.get(key, 0), value)
                    else:
                        target.setdefault(key, value)
            elif isinstance(payload, (int, float)) and not isinstance(payload, bool):
                merged[section] = max(merged.get(section, 0), payload)
            else:
                merged.setdefault(section, payload)
    return merged


def _mean_turns(turn_sets: list[list[TurnRecord]]) -> list[dict[str, Any]]:
    """Average each turn's duration across replays, keeping the first replay's text."""
    if not turn_sets:
        return []
    length = min(len(turns) for turns in turn_sets)
    output: list[dict[str, Any]] = []
    for index in range(length):
        primary = turn_sets[0][index].to_dict()
        durations = [turns[index].turn_seconds for turns in turn_sets]
        primary["turn_seconds"] = round(mean(durations), 3)
        primary["turn_seconds_spread"] = round(max(durations) - min(durations), 3)
        output.append(primary)
    return output


def _mean_per_turn_accuracy(accuracies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average each turn's score across replays, and union the failures.

    Failures are unioned rather than intersected: a check that failed in one replay out of
    three is a real fragility, and intersecting would hide it behind the replays that
    happened to pass.
    """
    if not accuracies:
        return []

    by_label: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for item in accuracies:
        for turn in item.get("per_turn", []):
            label = turn["label"]
            if label not in by_label:
                by_label[label] = []
                order.append(label)
            by_label[label].append(turn)

    output: list[dict[str, Any]] = []
    for label in order:
        entries = by_label[label]
        scores = [entry["score"] for entry in entries]
        correct = [entry for entry in entries if entry.get("materially_correct")]
        failures: list[str] = []
        for entry in entries:
            for failure in entry.get("critical_failures", []):
                if failure not in failures:
                    failures.append(failure)
        output.append(
            {
                "label": label,
                "score": _mean_of(scores),
                "score_spread_pct": _spread_pct(scores),
                "materially_correct_rate": _mean_of([1.0 if entry in correct else 0.0 for entry in entries]),
                "critical_failures_seen": failures,
                "scored": entries[0].get("scored", True),
                "replays": len(entries),
            }
        )
    return output
