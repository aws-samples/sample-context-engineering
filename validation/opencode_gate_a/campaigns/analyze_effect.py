"""Separate the plugin's effect from agent variance in a paired CE-60 repetition (payload-free).

Baseline and A are separate sessions of a stochastic agent, so the raw median difference mixes the
plugin's effect with run-to-run differences in how many tools the agent calls. This script reports, per
repetition, using only the sanitized artifacts:

  * ``pre``  : mean per-turn (A - baseline)/baseline model input BEFORE the first previewed result.
               The plugin has had no effect yet, so this estimates pure session-to-session noise.
  * ``post`` : the same quantity AFTER the first preview entered the context.
  * ``effect`` = post - pre, the part of the difference that is not explained by the noise level at the
    start of the same pair. It is a descriptive estimate, not a significance test.

The first previewed turn is the first turn where A's input diverges from baseline's because a result
crossed the gate; it is passed in explicitly (from the bridge metrics) rather than inferred.

Usage: python3 analyze_effect.py <results_dir> <first_preview_turn_id>
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def per_turn_model_input(run_dir: Path, arm: str) -> dict[str, int]:
    sanitized = json.loads((run_dir / f"{arm}-sanitized.json").read_text(encoding="utf-8"))
    return {t["turn_id"]: t["model_input_tokens"] for t in sanitized["turn_results"] if t["model_input_tokens"]}


def turn_order(run_dir: Path) -> list[str]:
    manifest = json.loads((run_dir / "baseline-live-manifest.json").read_text(encoding="utf-8"))
    return [a["turn_id"] for a in manifest["turn_assignments"]]


def analyze(run_dir: Path, first_preview_turn: str) -> dict[str, object]:
    order = turn_order(run_dir)
    baseline = per_turn_model_input(run_dir, "baseline")
    treated = per_turn_model_input(run_dir, "A")
    if first_preview_turn not in order:
        raise ValueError(f"unknown turn id: {first_preview_turn}")
    cut = order.index(first_preview_turn)
    deltas = {t: (treated[t] - baseline[t]) / baseline[t] * 100.0 for t in order if t in baseline and t in treated}
    pre = [deltas[t] for t in order[:cut] if t in deltas]
    post = [deltas[t] for t in order[cut:] if t in deltas]
    pre_mean = statistics.mean(pre) if pre else None
    post_mean = statistics.mean(post) if post else None
    return {
        "first_preview_turn": first_preview_turn,
        "turns_pre": len(pre),
        "turns_post": len(post),
        "pre_mean_pct": None if pre_mean is None else round(pre_mean, 2),
        "post_mean_pct": None if post_mean is None else round(post_mean, 2),
        "effect_pct": None if pre_mean is None or post_mean is None else round(post_mean - pre_mean, 2),
        "median_delta_pct": round(
            (statistics.median(treated.values()) / statistics.median(baseline.values()) - 1) * 100.0, 2
        ),
    }


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    print(json.dumps(analyze(Path(argv[1]), argv[2]), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
