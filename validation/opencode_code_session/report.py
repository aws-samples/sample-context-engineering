#!/usr/bin/env python3.13
"""Read one coding-session run (baseline and plugin) and report what each arm cost and what it left behind.

Inputs, all in the run directory: ``<arm>-code-session.json`` (from ``.opencode/runner/code-session.ts``),
``<arm>-workcheck.json`` (from ``workcheck.py``) and, for the plugin arm, ``A-metrics.jsonl`` (the bridge's per-result
facts) and ``A-graph-durability.json``. Everything here is counts and sizes; the only text is the short answers to the
two recall probes, which are scored against ``prompts.json``.

This reports. It applies no threshold and decides nothing: one session per arm is an anecdote, not a sample, and the
agent's trajectory differs from run to run. The pre-written reading is in ``run.sh``.

Usage: python3.13 validation/opencode_code_session/report.py <run_dir> [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

PROMPTS = Path(__file__).resolve().parent / "prompts.json"
GRAPH_TOOLS = ("expand_card", "find_context")


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def score_probe(rules: dict, text: str) -> bool:
    lowered = text.lower()
    if not all(s.lower() in lowered for s in rules.get("all_of", [])):
        return False
    any_of = rules.get("any_of")
    if any_of and not any(s.lower() in lowered for s in any_of):
        return False
    patterns = rules.get("any_regex")
    return not patterns or any(re.search(p, text, re.IGNORECASE) for p in patterns)


def arm_summary(session: dict, probes: dict) -> dict:
    turns = session["turns"]
    calls = [c for t in turns for c in t.get("calls", [])]
    inputs = [c["model_input"] for c in calls]
    tools: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    for t in turns:
        tools.update(t.get("tool_counts", {}))
        errors.update(t.get("tool_errors", {}))
    test_runs = [(t["turn_id"], r) for t in turns for r in t.get("test_runs", [])]
    probe_rows = {}
    for turn in turns:
        rules = probes.get(turn["turn_id"])
        if rules is not None:
            text = turn.get("answer_text", "")
            probe_rows[turn["turn_id"]] = {
                "correct": score_probe(rules, text) if text else False,
                "answered": bool(text),
                "tool_calls": dict(turn.get("tool_counts", {})),
            }
    totals = session["totals"]
    return {
        "turns_sent": len(turns),
        "turns_ok": sum(1 for t in turns if t["host_signal"] == "success"),
        "timeouts": sum(1 for t in turns if t["host_signal"] == "timeout"),
        "errors": sum(1 for t in turns if t["host_signal"] == "error"),
        "model_calls": totals["model_calls"],
        "model_input": totals["model_input"],
        "input_uncached": totals["input"],
        "cache_read": totals["cache_read"],
        "cache_write": totals["cache_write"],
        "output": totals["output"],
        "reasoning": totals["reasoning"],
        "dollars": round(totals["cost"], 4),
        "call_input_median": int(statistics.median(inputs)) if inputs else 0,
        "call_input_peak": max(inputs) if inputs else 0,
        "wall_s": round(sum(t["wall_ms"] for t in turns) / 1000, 1),
        "tool_counts": dict(tools.most_common()),
        "tool_errors": dict(errors),
        "denied_calls": sum(t.get("denied", 0) for t in turns),
        "files_written": sum(t.get("files_written", 0) for t in turns),
        "test_runs": [{"turn": turn, **run} for turn, run in test_runs],
        "probes": probe_rows,
    }


def bridge_summary(path: Path) -> dict | None:
    if not path.is_file():
        return None
    operations: Counter[str] = Counter()
    rerank: Counter[str] = Counter()
    classes: Counter[str] = Counter()
    previewed_by_class: Counter[str] = Counter()
    saved = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") != "transform":
            continue
        operations[str(row.get("operation"))] += 1
        classes[str(row.get("tool_class"))] += 1
        rerank[str(row.get("rerank_result"))] += 1
        if row.get("operation") == "previewed":
            previewed_by_class[str(row.get("tool_class"))] += 1
            saved += max(0, int(row.get("input_bytes") or 0) - int(row.get("output_bytes") or 0))
    return {
        "transforms": sum(operations.values()),
        "operations": dict(operations),
        "rerank_results": dict(rerank),
        "tool_classes": dict(classes),
        "previewed_by_class": dict(previewed_by_class),
        "preview_bytes_saved": saved,
    }


def pct(treatment: float, baseline: float) -> float | None:
    return round((treatment - baseline) / baseline * 100, 1) if baseline else None


def build(run_dir: Path) -> dict:
    spec = json.loads(PROMPTS.read_text(encoding="utf-8"))
    probes = spec["probes"]
    # The directory's name, never its absolute path: the report is versioned and must not carry a machine's layout.
    result: dict = {"run": run_dir.name, "arms": {}}
    sessions = {}
    for arm in ("baseline", "plugin"):
        session = _load(run_dir / f"{arm}-code-session.json")
        if session is None:
            continue
        sessions[arm] = session
        summary = arm_summary(session, probes)
        work = _load(run_dir / f"{arm}-workcheck.json")
        if work is not None:
            summary["work"] = {k: v for k, v in work.items() if k not in {"files"}}
        result["arms"][arm] = summary
    plugin = sessions.get("plugin")
    if plugin is not None:
        result["arms"]["plugin"]["plugin_mode"] = plugin.get("plugin_mode")
        result["arms"]["plugin"]["plugin_options"] = plugin.get("plugin_options")
        result["arms"]["plugin"]["bridge"] = bridge_summary(run_dir / "A-metrics.jsonl")
        graph = _load(run_dir / "A-graph-durability.json")
        if graph is not None:
            result["arms"]["plugin"]["graph"] = {
                k: graph.get(k)
                for k in (
                    "requests",
                    "checked",
                    "stored_prefix_equal",
                    "mismatched_requests",
                    "operations",
                    "removed_messages_total",
                    "rejected_requests",
                    "graph_tool_calls",
                    "stored_graph_synthetic_messages",
                )
            }
    base, plug = result["arms"].get("baseline"), result["arms"].get("plugin")
    if base and plug:
        result["delta_pct"] = {
            key: pct(plug[key], base[key]) for key in ("model_input", "dollars", "model_calls", "output", "wall_s", "call_input_median", "call_input_peak")
        }
    return result


def render(result: dict) -> str:
    arms = result["arms"]
    names = [a for a in ("baseline", "plugin") if a in arms]
    rows = [
        ("turns ok / sent", lambda a: f"{a['turns_ok']}/{a['turns_sent']}"),
        ("model calls", lambda a: a["model_calls"]),
        ("input tokens (read+write+uncached)", lambda a: f"{a['model_input']:,}"),
        ("  cache read", lambda a: f"{a['cache_read']:,}"),
        ("  cache write", lambda a: f"{a['cache_write']:,}"),
        ("output tokens", lambda a: f"{a['output']:,}"),
        ("dollars (OpenCode cost counter)", lambda a: f"{a['dollars']:.3f}"),
        ("input per call, median / peak", lambda a: f"{a['call_input_median']:,} / {a['call_input_peak']:,}"),
        ("wall time (s)", lambda a: a["wall_s"]),
        ("files written (write/edit)", lambda a: a["files_written"]),
        ("denied calls", lambda a: a["denied_calls"]),
    ]
    out = [f"{'':40}" + "".join(f"{n:>22}" for n in names)]
    for label, fn in rows:
        out.append(f"{label:40}" + "".join(f"{fn(arms[n])!s:>22}" for n in names))
    if "delta_pct" in result:
        d = result["delta_pct"]
        out.append(
            "plugin vs baseline: "
            + ", ".join(f"{k} {v:+.1f}%" for k, v in d.items() if v is not None)
        )
    for n in names:
        a = arms[n]
        out.append(f"\n[{n}] tools: {a['tool_counts']}")
        if a["tool_errors"]:
            out.append(f"[{n}] tool errors: {a['tool_errors']}")
        out.append(f"[{n}] test runs: {[(r['turn'][:3], r['passed'], r['failed'], r['errors']) for r in a['test_runs']]}")
        out.append(f"[{n}] probes: {{" + ", ".join(f"{k[:3]}: {'ok' if v['correct'] else 'WRONG'} (tools {v['tool_calls']})" for k, v in a['probes'].items()) + "}")
        w = a.get("work")
        if w:
            out.append(
                f"[{n}] work: created {w['created_files']} files / {w['created_lines']} lines; expected present {w['expected_present']}/{w['expected_files']}; "
                f"json ok {w['json_ok']}/{w['json_files']}; python compiles {w['python_compile_ok']}/{w['python_files']}; "
                f"marketplace lists {w['marketplace_lists']}/{w['marketplace_expected']}; README section {w['readme_section_present']}; "
                f"touched existing packages {w['touched_existing_packages']}"
            )
    plug = arms.get("plugin")
    if plug:
        out.append(f"\n[plugin] mode={plug.get('plugin_mode')} options={plug.get('plugin_options')}")
        out.append(f"[plugin] practice A (bridge): {plug.get('bridge')}")
        out.append(f"[plugin] practice D (graph): {plug.get('graph')}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    result = build(args.run_dir.resolve())
    if not result["arms"]:
        print("no <arm>-code-session.json in the run directory", file=sys.stderr)
        return 2
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(render(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
