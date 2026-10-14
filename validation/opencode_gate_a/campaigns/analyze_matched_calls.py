"""Isolate the plugin's own effect from agent variance by matching identical tool calls.

Baseline and A are separate sessions of a stochastic agent. The model chooses the arguments of its tool
calls, so the SAME turn can read very different amounts of data in the two arms (for example a log tail
of 7.9 KB in one session and 15.8 KB in the other). That difference is unrelated to the plugin, yet it
moves the median model input that Gate A thresholds.

This script matches tool calls across the two persisted sessions by (turn, tool, ordinal within the turn)
AND requires the ORIGINAL result size to be identical in both arms. For each matched pair it reports the
change in the number of input tokens the provider billed for the next step. It reads sizes and counts only
from the local OpenCode store; no prompt, answer, argument or result text is read or printed.

Usage: python3 analyze_matched_calls.py <run_id>   (e.g. ce60-v3-sonnet55-r1)
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

MARKER = "[context-engineering]"


def open_store() -> tuple[sqlite3.Connection, Path]:
    src = Path.home() / ".local/share/opencode"
    tmp = Path(tempfile.mkdtemp())
    for name in ("opencode.db", "opencode.db-wal", "opencode.db-shm"):
        if (src / name).exists():
            shutil.copy(src / name, tmp / name)
    return sqlite3.connect(f"file:{tmp / 'opencode.db'}?mode=ro", uri=True), tmp


def session_steps(con: sqlite3.Connection, title: str) -> list[dict]:
    """Return assistant steps in order with turn, input-side tokens and tool-call sizes."""
    row = con.execute("select id from session where title=? order by time_created desc limit 1", (title,)).fetchone()
    if row is None:
        raise SystemExit(f"session not found: {title}")
    session_id = row[0]
    parts: dict[str, list[dict]] = {}
    for message_id, data in con.execute(
        "select message_id, data from part where session_id=? order by time_created", (session_id,)
    ):
        parts.setdefault(message_id, []).append(json.loads(data))
    steps: list[dict] = []
    turn = 0
    for message_id, data in con.execute(
        "select id, data from message where session_id=? order by time_created", (session_id,)
    ):
        message = json.loads(data)
        if message.get("role") == "user":
            turn += 1
        if message.get("role") != "assistant":
            continue
        tokens = message.get("tokens") or {}
        cache = tokens.get("cache") or {}
        calls = []
        for part in parts.get(message_id, []):
            if part.get("type") != "tool":
                continue
            output = (part.get("state") or {}).get("output")
            size = len(output.encode("utf-8")) if isinstance(output, str) else 0
            calls.append(
                {
                    "tool": part.get("tool"),
                    "bytes": size,
                    "previewed": isinstance(output, str) and MARKER in output,
                }
            )
        steps.append(
            {
                "turn": turn,
                "input": tokens.get("input", 0) + cache.get("read", 0) + cache.get("write", 0),
                "calls": calls,
            }
        )
    return steps


def analyze(run_id: str) -> dict:
    con, tmp = open_store()
    try:
        baseline = session_steps(con, f"ce60-baseline-{run_id}-baseline")
        treated = session_steps(con, f"ce60-A-{run_id}-A")
    finally:
        con.close()
        shutil.rmtree(tmp, ignore_errors=True)

    def first_step_after_call(steps: list[dict]) -> dict[tuple[int, str, int], tuple[int, int, bool]]:
        seen: dict[tuple[int, str], int] = {}
        out: dict[tuple[int, str, int], tuple[int, int, bool]] = {}
        for index, step in enumerate(steps):
            for call in step["calls"]:
                key2 = (step["turn"], str(call["tool"]))
                ordinal = seen.get(key2, 0)
                seen[key2] = ordinal + 1
                following = steps[index + 1]["input"] if index + 1 < len(steps) else None
                if following is not None:
                    out[(step["turn"], str(call["tool"]), ordinal)] = (call["bytes"], following, call["previewed"])
        return out

    b = first_step_after_call(baseline)
    a = first_step_after_call(treated)
    matched = []
    for key, (a_bytes, a_next, previewed) in a.items():
        if key not in b:
            continue
        b_bytes, b_next, _ = b[key]
        # A preview changes the persisted size, so identity is judged on the ORIGINAL size: the baseline
        # size when A previewed it, otherwise the sizes must simply be equal.
        same_original = previewed or a_bytes == b_bytes
        if not same_original:
            continue
        matched.append(
            {
                "turn": key[0],
                "tool": key[1],
                "baseline_bytes": b_bytes,
                "a_bytes": a_bytes,
                "previewed": previewed,
                "next_step_input_delta": a_next - b_next,
            }
        )
    previewed_pairs = [m for m in matched if m["previewed"]]
    untouched_pairs = [m for m in matched if not m["previewed"]]
    return {
        "run_id": run_id,
        "assistant_steps": {"baseline": len(baseline), "A": len(treated)},
        "matched_calls": len(matched),
        "previewed_matched": previewed_pairs,
        "untouched_matched_next_step_delta": {
            "n": len(untouched_pairs),
            "min": min((m["next_step_input_delta"] for m in untouched_pairs), default=None),
            "max": max((m["next_step_input_delta"] for m in untouched_pairs), default=None),
        },
        "step_input_total": {
            "baseline": sum(s["input"] for s in baseline),
            "A": sum(s["input"] for s in treated),
        },
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps(analyze(sys.argv[1]), indent=2))
