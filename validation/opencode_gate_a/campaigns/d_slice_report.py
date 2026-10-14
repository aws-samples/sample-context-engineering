"""The D slice report: practice D on top of A, read against the baseline of the same repetition and against A alone.

Applies the reading pre-registered in the header of ``ce60-v7.sh`` -- mechanically, so the label cannot be chosen after the
numbers are known. Reads two kinds of artifact and nothing else:

* the OpenCode store (a COPY of it) for tokens and dollars, from the same per-message counters the host keeps: input, cache read,
  cache write, output, and ``cost`` (the provider price list applied to them);
* the payload-free evidence files of each run (``baseline-evidence.json`` / ``A-evidence.json``) for correctness, and the
  graph's own files (``A-graph-durability.json``) for what D did.

The v6 reference (A alone) is recomputed here from the v6 sessions with the same function, so the comparison is not made against
a number remembered from an earlier report.

Usage (from the repository root)::

    python3.13 validation/opencode_gate_a/campaigns/d_slice_report.py --d ce60-v7-sonnet55-r1 --d ce60-v7-sonnet55-r2 \\
        --d ce60-v7-sonnet55-r3 --a ce60-v6-sonnet55-r1 --a ce60-v6-sonnet55-r2 --a ce60-v6-sonnet55-r3 [--out report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

_VALIDATION = Path(__file__).resolve().parents[3] / "validation"
if str(_VALIDATION) not in sys.path:
    sys.path.insert(0, str(_VALIDATION))

from opencode_gate_d.recorded import copied_store, find_session  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"

# Fixed in the header of ce60-v7.sh before any v7 repetition ran.
V6_A_TOTAL_INPUT_PCT = -13.9
V6_A_DOLLARS_PCT = -11.9
MARGIN_POINTS = 5.0
MIN_K_FLOOR = -1


@dataclass(frozen=True)
class Totals:
    """What one session consumed, from the host's own per-message counters."""

    model_calls: int
    input_total: int  # input + cache read + cache write, the Gate A model-input measure summed over every call
    cache_read: int
    cache_write: int
    output: int
    dollars: float


def session_totals(connection, session_id: str) -> Totals:  # type: ignore[no-untyped-def]
    calls = inp = read = write = out = 0
    dollars = 0.0
    for (data,) in connection.execute("select data from message where session_id = ?", (session_id,)):
        message = json.loads(data)
        if message.get("role") != "assistant":
            continue
        tokens = message.get("tokens") or {}
        cache = tokens.get("cache") or {}
        calls += 1
        inp += int(tokens.get("input") or 0)
        read += int(cache.get("read") or 0)
        write += int(cache.get("write") or 0)
        out += int(tokens.get("output") or 0)
        dollars += float(message.get("cost") or 0.0)
    return Totals(calls, inp + read + write, read, write, out, dollars)


def _pct(new: float, old: float) -> float:
    return 100.0 * (new - old) / old if old else float("nan")


def _evidence(run_id: str, arm: str) -> dict:
    return json.loads((RESULTS / run_id / f"{arm}-evidence.json").read_text(encoding="utf-8"))


def _graph(run_id: str) -> dict:
    path = RESULTS / run_id / "A-graph-durability.json"
    if not path.exists():
        return {}
    body = json.loads(path.read_text(encoding="utf-8"))
    body.pop("stored_digests", None)
    return body


def repetition(connection, run_id: str) -> dict:  # type: ignore[no-untyped-def]
    """One paired repetition: baseline against arm A of the same run id."""
    base_session = find_session(connection, f"ce60-baseline-{run_id}-baseline")
    arm_session = find_session(connection, f"ce60-A-{run_id}-A")
    if base_session is None or arm_session is None:
        raise SystemExit(f"{run_id}: session not found in the store (baseline={base_session}, A={arm_session})")
    base, arm = session_totals(connection, base_session), session_totals(connection, arm_session)
    base_ev, arm_ev = _evidence(run_id, "baseline"), _evidence(run_id, "A")
    return {
        "run_id": run_id,
        "baseline": asdict(base),
        "arm": asdict(arm),
        "tokens_pct": round(_pct(arm.input_total, base.input_total), 2),
        "dollars_pct": round(_pct(arm.dollars, base.dollars), 2),
        "cache_write_ratio": round(arm.cache_write / base.cache_write, 2) if base.cache_write else None,
        "output_pct": round(_pct(arm.output, base.output), 2),
        "calls_delta": arm.model_calls - base.model_calls,
        "correct": {
            "baseline": [base_ev["correctness_numerator"], base_ev["correctness_denominator"]],
            "arm": [arm_ev["correctness_numerator"], arm_ev["correctness_denominator"]],
            "k": arm_ev["correctness_numerator"] - base_ev["correctness_numerator"],
        },
        "median_model_input": {"baseline": base_ev["model_input_tokens_median"], "arm": arm_ev["model_input_tokens_median"]},
        "graph": _graph(run_id),
    }


def label(d_reps: list[dict]) -> dict:
    """The pre-registered labels, applied to the three means. Pure function of the repetitions."""
    mean_c = mean(rep["dollars_pct"] for rep in d_reps)
    mean_t = mean(rep["tokens_pct"] for rep in d_reps)
    min_k = min(rep["correct"]["k"] for rep in d_reps)
    useful = mean_c <= V6_A_DOLLARS_PCT - MARGIN_POINTS and min_k >= MIN_K_FLOOR
    tokens_only = not useful and mean_t <= V6_A_TOTAL_INPUT_PCT - MARGIN_POINTS
    return {
        "label": "D_USEFUL" if useful else "D_TOKENS_ONLY" if tokens_only else "D_NOT_WORTH_IT",
        "d_costs_correctness": min_k < MIN_K_FLOOR,
        "mean_tokens_pct": round(mean_t, 2),
        "mean_dollars_pct": round(mean_c, 2),
        "min_k": min_k,
        "thresholds": {
            "dollars_pct_at_most": V6_A_DOLLARS_PCT - MARGIN_POINTS,
            "tokens_pct_at_most": V6_A_TOTAL_INPUT_PCT - MARGIN_POINTS,
            "k_at_least": MIN_K_FLOOR,
        },
    }


def render(report: dict) -> str:
    lines = ["| repetition | tokens vs baseline | dollars vs baseline | cache writes (x baseline) | K (correct turns) | graph requests (projected) |", "|---|---:|---:|---:|---:|---:|"]
    for group, title in (("d", "A+D"), ("a", "A alone (v6)")):
        for rep in report[group]:
            graph = rep.get("graph") or {}
            projected = (graph.get("operations") or {}).get("projected", "-")
            lines.append(
                f"| {title} {rep['run_id']} | {rep['tokens_pct']:+.1f} % | {rep['dollars_pct']:+.1f} % | "
                f"{rep['cache_write_ratio']} | {rep['correct']['k']:+d} ({rep['correct']['arm'][0]}/{rep['correct']['arm'][1]} vs "
                f"{rep['correct']['baseline'][0]}/{rep['correct']['baseline'][1]}) | {graph.get('requests', '-')} ({projected}) |"
            )
    summary = report["label"]
    lines.append("")
    lines.append(
        f"**{summary['label']}** (mean tokens {summary['mean_tokens_pct']:+.1f} %, mean dollars {summary['mean_dollars_pct']:+.1f} %, "
        f"min K {summary['min_k']:+d}); A alone (recomputed): tokens {report['a_alone']['mean_tokens_pct']:+.1f} %, "
        f"dollars {report['a_alone']['mean_dollars_pct']:+.1f} %."
    )
    if summary["d_costs_correctness"]:
        lines.append("D_COSTS_CORRECTNESS: at least one repetition lost more than one correct turn against its baseline.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--d", action="append", required=True, help="Run id of an A+D repetition (repeatable).")
    parser.add_argument("--a", action="append", default=[], help="Run id of an A-alone repetition to read against (repeatable).")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    with copied_store() as connection:
        d_reps = [repetition(connection, run_id) for run_id in args.d]
        a_reps = [repetition(connection, run_id) for run_id in args.a]
    report = {
        "d": d_reps,
        "a": a_reps,
        "label": label(d_reps),
        "a_alone": {
            "mean_tokens_pct": round(mean(rep["tokens_pct"] for rep in a_reps), 2) if a_reps else None,
            "mean_dollars_pct": round(mean(rep["dollars_pct"] for rep in a_reps), 2) if a_reps else None,
        },
    }
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(render(report) if a_reps else json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
