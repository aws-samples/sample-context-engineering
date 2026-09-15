"""Verify a run's token numbers against Bedrock's own invocation log.

The harness reads usage off the model stream. That is the provider's own figure, but it is
read at a place that has two blind spots: a call that raises never yields usage at all, and
a provider-side retry is one stream to the harness and two metered invocations to Bedrock.
Neither is visible from inside the run, and both understate consumption.

Bedrock's invocation log has neither blind spot. Every call is stamped with
``requestMetadata`` naming its configuration, turn and call index, so the join is exact
rather than inferred from timestamps — which matters because CloudWatch *metrics* bucket by
the minute and cannot separate configurations that ran back to back.

Usage:
    python -m validation.verify_logs results/run-verify60.json
    python -m validation.verify_logs results/run-verify60.json --per-call
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from . import runner
from .config import INVOCATION_LOG_GROUP, PRICING, RESULTS_DIR

_QUERY = """
fields requestMetadata.config as config,
       requestMetadata.role as role,
       requestMetadata.turn as turn,
       requestMetadata.call as call,
       requestId,
       modelId,
       input.inputTokenCount as input_tokens,
       output.outputTokenCount as output_tokens,
       input.cacheReadInputTokenCount as cache_read,
       input.cacheWriteInputTokenCount as cache_write
| filter requestMetadata.harness = "{tag}"
| sort @timestamp asc
| limit 10000
"""


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _run_bounds(payload: dict[str, Any]) -> tuple[datetime, datetime]:
    """Return the UTC span of every recorded call, padded for delivery lag.

    Falls back to a wide window when a run predates timestamped records, so an old file
    reports "no entries" rather than crashing.
    """
    stamps = [
        _parse_iso(call[key])
        for entry in payload.values()
        for call in entry.get("calls", [])
        for key in ("started_at", "ended_at")
        if isinstance(call.get(key), str)
    ]
    if not stamps:
        now = datetime.now(timezone.utc)
        return now - timedelta(days=1), now
    return min(stamps) - timedelta(minutes=5), max(stamps) + timedelta(minutes=15)


def fetch_log_records(tag: str, start: datetime, end: datetime, log_group: str) -> list[dict[str, Any]]:
    """Return every invocation-log entry stamped with ``tag``, via CloudWatch Logs Insights."""
    logs = runner.boto_session().client("logs")
    query = logs.start_query(
        logGroupName=log_group,
        startTime=int(start.timestamp()),
        endTime=int(end.timestamp()),
        queryString=_QUERY.format(tag=tag),
        limit=10_000,
    )
    query_id = query["queryId"]
    while True:
        time.sleep(2)
        result = logs.get_query_results(queryId=query_id)
        if result["status"] in ("Complete", "Failed", "Cancelled", "Timeout"):
            break
    if result["status"] != "Complete":
        raise RuntimeError(f"Logs Insights query ended as {result['status']}")

    records = []
    for row in result["results"]:
        record = {field["field"]: field["value"] for field in row if not field["field"].startswith("@")}
        for numeric in ("input_tokens", "output_tokens", "cache_read", "cache_write"):
            record[numeric] = int(float(record.get(numeric) or 0))
        records.append(record)
    return records


def _unlogged_calls(payload: dict[str, Any], records: list[dict[str, Any]]) -> int:
    """Return how many recorded agent calls have no matching log entry yet.

    Delivery is asynchronous and runs a couple of minutes behind, so this is the signal for
    "wait longer" — reading it as a discrepancy is how a verified run looks broken.
    """
    seen = {
        (record.get("config"), record.get("turn"), record.get("call"))
        for record in records
        if record.get("role") == "agent"
    }
    return sum(
        1
        for name, entry in payload.items()
        for call in entry.get("calls", [])
        if (name, str(call["turn_index"]), str(call["call_index"])) not in seen
    )


def _harness_totals(entry: dict[str, Any]) -> dict[str, Any]:
    """Return one configuration's harness-side figures, split by role like the log is."""
    summary = entry.get("summary", {})
    tokens = summary.get("tokens", {})
    return {
        "agent_input": int(tokens.get("usage_input_total", 0) or 0),
        "agent_output": int(tokens.get("usage_output_total", 0) or 0),
        "agent_calls": len(entry.get("calls", [])),
    }


def _delta(harness: int, real: int) -> str:
    if not harness:
        return "n/a" if not real else "  +inf"
    return f"{(real - harness) / harness * 100:+6.2f}%"


def compare(payload: dict[str, Any], records: list[dict[str, Any]], *, per_call: bool) -> int:
    """Print the comparison and return the number of material discrepancies found."""
    by_config: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for record in records:
        by_config.setdefault(record.get("config", "?"), {}).setdefault(record.get("role", "?"), []).append(record)

    names = [name for name in payload if name != "meta"]
    problems = 0

    print(f"{'config':<12}{'role':<9}{'calls h/log':>13}{'input harness':>15}{'input log':>13}{'Δ':>9}"
          f"{'out harness':>13}{'out log':>10}{'Δ':>9}")
    totals = {"h_in": 0, "l_in": 0, "h_out": 0, "l_out": 0, "h_calls": 0, "l_calls": 0}

    for name in names:
        harness = _harness_totals(payload[name])
        for role, h_in, h_out, h_calls in (
            ("agent", harness["agent_input"], harness["agent_output"], harness["agent_calls"]),
        ):
            logged = by_config.get(name, {}).get(role, [])
            if not logged and not h_calls:
                continue
            l_in = sum(record["input_tokens"] for record in logged)
            l_out = sum(record["output_tokens"] for record in logged)
            call_counts = "{}/{}".format(h_calls, len(logged))
            print(f"{name:<12}{role:<9}{call_counts:>13}{h_in:>15,}{l_in:>13,}"
                  f"{_delta(h_in, l_in):>9}{h_out:>13,}{l_out:>10,}{_delta(h_out, l_out):>9}")
            for key, value in (("h_in", h_in), ("l_in", l_in), ("h_out", h_out),
                               ("l_out", l_out), ("h_calls", h_calls), ("l_calls", len(logged))):
                totals[key] += value

    # Roles the harness does not meter at all — the diagnostic passes. Billed, so they belong
    # in the log total, but counting them against a strategy would be reporting the harness's
    # own instrumentation as the strategy's cost.
    unmetered = 0
    for name in names:
        for role, logged in sorted(by_config.get(name, {}).items()):
            if role == "agent":
                continue
            l_in = sum(record["input_tokens"] for record in logged)
            l_out = sum(record["output_tokens"] for record in logged)
            unmetered += l_in + l_out
            counts = "0/{}".format(len(logged))
            print(f"{name:<12}{role:<9}{counts:>13}{'not metered':>15}{l_in:>13,}{'':>9}"
                  f"{'not metered':>13}{l_out:>10,}")

    total_calls = "{}/{}".format(totals["h_calls"], totals["l_calls"])
    print(f"{'TOTAL':<12}{'':<9}{total_calls:>13}"
          f"{totals['h_in']:>15,}{totals['l_in']:>13,}{_delta(totals['h_in'], totals['l_in']):>9}"
          f"{totals['h_out']:>13,}{totals['l_out']:>10,}{_delta(totals['h_out'], totals['l_out']):>9}")

    cost_harness = (
        totals["h_in"] / 1e6 * PRICING.agent_input_per_mtok + totals["h_out"] / 1e6 * PRICING.agent_output_per_mtok
    )
    cost_log = totals["l_in"] / 1e6 * PRICING.agent_input_per_mtok + totals["l_out"] / 1e6 * PRICING.agent_output_per_mtok
    print(f"\ncost at the harness's own rates: reported ${cost_harness:,.2f}, metered ${cost_log:,.2f} "
          f"({_delta(int(cost_harness * 100), int(cost_log * 100)).strip()})")
    if unmetered:
        print(f"plus {unmetered:,} tokens of harness diagnostics, billed but outside every "
              "strategy's cost by design")

    # --- per-call join: this is what the metadata stamp buys over minute-bucketed metrics ---
    print("\n--- per-call reconciliation (agent calls) ---")
    for name in names:
        indexed: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for record in by_config.get(name, {}).get("agent", []):
            indexed.setdefault((record.get("turn", "?"), record.get("call", "?")), []).append(record)

        missing, retried, mismatched, recovered = [], [], [], []
        for call in payload[name].get("calls", []):
            key = (str(call["turn_index"]), str(call["call_index"]))
            logged = indexed.get(key, [])
            if not logged:
                missing.append(key)
                continue
            if len(logged) > 1:
                retried.append((key, len(logged)))
            logged_input = sum(record["input_tokens"] for record in logged)
            harness_input = call.get("usage_input_tokens")
            if harness_input is None:
                recovered.append((key, logged_input))
            elif logged_input != harness_input:
                mismatched.append((key, harness_input, logged_input))

        parts = [f"{len(payload[name].get('calls', []))} calls"]
        if missing:
            parts.append(f"{len(missing)} with no log entry")
        if retried:
            parts.append(f"{len(retried)} retried ({sum(count for _, count in retried)} invocations)")
        if recovered:
            parts.append(f"{len(recovered)} recovered from the log "
                         f"({sum(tokens for _, tokens in recovered):,} tokens the stream never reported)")
        if mismatched:
            parts.append(f"{len(mismatched)} token mismatches")
        print(f"  {name:<12} {', '.join(parts)}")

        problems += len(missing) + len(mismatched)
        if per_call:
            for key, harness_input, logged_input in mismatched:
                print(f"      turn {key[0]} call {key[1]}: harness {harness_input:,} vs log {logged_input:,}")
            for key, tokens in recovered:
                print(f"      turn {key[0]} call {key[1]}: no usage on the stream, log says {tokens:,}")
            for key, count in retried:
                print(f"      turn {key[0]} call {key[1]}: {count} invocations")

    # A request id captured on the harness side that the log does not know about, or the
    # reverse, means the stamp is not landing — worth failing loudly rather than reading as
    # a clean run with fewer records.
    harness_ids = {
        request_id
        for name in names
        for call in payload[name].get("calls", [])
        for request_id in call.get("request_ids", [])
    }
    # Agent role only: request ids are captured for the measured calls, so comparing against
    # every entry in the log would count the diagnostics' as unmatched.
    logged_ids = {
        record["requestId"] for record in records if record.get("requestId") and record.get("role") == "agent"
    }
    if harness_ids:
        orphans = harness_ids - logged_ids
        print(f"\nrequest ids: {len(harness_ids)} captured by the harness, {len(logged_ids)} in the log, "
              f"{len(orphans)} captured but unlogged")
        problems += len(orphans)
    else:
        print("\nrequest ids: none captured — the after-call hook did not run, so only the "
              "metadata direction of the join is available")

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="validation.verify_logs", description=__doc__)
    parser.add_argument("run_json", help="a run JSON produced by validation.run")
    parser.add_argument("--tag", help="override the harness tag to look for (default: meta.log_tag)")
    parser.add_argument("--per-call", action="store_true", help="list every discrepant call, not just counts")
    parser.add_argument("--wait", type=int, default=45, metavar="SECONDS",
                        help="pause between attempts while log delivery catches up (default: 45)")
    parser.add_argument("--attempts", type=int, default=8, metavar="N",
                        help="how many times to re-query before reporting what is missing (default: 8)")
    args = parser.parse_args(argv)

    path = Path(args.run_json)
    if not path.exists():
        # Accepts a bare name or a path relative to the repo root, since both are what a
        # reader copies out of the line run.py prints.
        path = RESULTS_DIR / path.name
    payload = json.loads(path.read_text(encoding="utf-8"))

    meta = next((entry.get("meta", {}) for entry in payload.values() if isinstance(entry, dict)), {})
    tag = args.tag or meta.get("log_tag")
    if not tag:
        print("no log tag in this run's meta and none given with --tag; the run predates "
              "invocation-log correlation.", file=sys.stderr)
        return 2
    log_group = meta.get("log_group") or INVOCATION_LOG_GROUP

    start, end = _run_bounds(payload)
    print(f"run:   {path}")
    print(f"tag:   {tag}")
    print(f"group: {log_group}")
    print(f"span:  {start:%Y-%m-%d %H:%M:%S} -> {end:%Y-%m-%d %H:%M:%S} UTC\n")

    records: list[dict[str, Any]] = []
    for attempt in range(args.attempts):
        records = fetch_log_records(tag, start, end, log_group)
        pending = _unlogged_calls(payload, records)
        if records and not pending:
            break
        if attempt < args.attempts - 1:
            print(f"  {pending} call(s) not yet delivered, retrying in {args.wait}s "
                  f"({attempt + 1}/{args.attempts})")
            time.sleep(args.wait)

    if not records:
        print("no invocation-log entries carry this tag. Either logging was off during the run, "
              "or delivery never caught up.", file=sys.stderr)
        return 2

    problems = compare(payload, records, per_call=args.per_call)
    print(f"\n{len(records)} invocation-log entries examined, {problems} discrepancies")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
