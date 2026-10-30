"""Side-by-side table of every complete CE-60 repetition, from the payload-free artifacts only.

One row per repetition directory ``ce60-v<N>-<model>-r<K>`` that holds a Gate A decision: the correctness of
both arms, the median model input of both arms and the change, the Gate A outcome with its reasons, and what the
plugin did (previews, retrievals, and, when the repetition has a ``preview-outcomes.json``, how many previews
were read back in full, in part or not at all). Nothing is recomputed from the model's text.

Usage: python3 summarize_campaigns.py [results_dir]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

PATTERN = re.compile(r"^ce60-(v\d+)-(sonnet55|opus55)-r(\d+)$")
SHORT = {
    "GATE_MODEL_INPUT_INSUFFICIENT": "input",
    "GATE_CORRECTNESS_REGRESSION": "correctness",
    "GATE_RETRIEVAL_INVALID": "retrieval",
    "GATE_COMPLETION_REGRESSION": "completion",
    "GATE_METRIC_NOT_EVALUABLE": "not-evaluable",
}


def _load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def row(directory: Path) -> dict[str, Any] | None:
    """Collect one repetition's figures, or ``None`` when the directory is not a complete repetition."""
    decision = _load(directory / "gate-a-decision.json")
    base = _load(directory / "baseline-evidence.json")
    treated = _load(directory / "A-evidence.json")
    if not (decision and base and treated):
        return None
    base_mi, a_mi = base.get("model_input_tokens_median"), treated.get("model_input_tokens_median")
    change = None if not base_mi or a_mi is None else (a_mi / base_mi - 1) * 100
    host = treated.get("host_evidence", {})
    outcomes = _load(directory / "preview-outcomes.json")
    total = (outcomes or {}).get("total", {})
    return {
        "baseline_correct": f"{base['correctness_numerator']}/{base['correctness_denominator']}",
        "a_correct": f"{treated['correctness_numerator']}/{treated['correctness_denominator']}",
        "baseline_mi": base_mi,
        "a_mi": a_mi,
        "change_pct": change,
        "outcome": decision.get("outcome"),
        "reasons": [SHORT.get(r, r) for r in decision.get("reasons", [])],
        "previews": host.get("previewed_count"),
        "retrievals": host.get("retrieval_attempts"),
        "full": total.get("full_reads", total.get("re_expanded")),
        "partial": total.get("partial_reads"),
        "unread": total.get("unread"),
        "plugin_options": (
            "inert (write-back bug, void)"
            if (directory / "INVALID-ARM-A-PREFIX-WRITEBACK.txt").exists()
            else host.get("plugin_options") or "legacy"
        ),
    }


def collect(results: Path) -> list[tuple[str, str, int, dict[str, Any]]]:
    rows = []
    for directory in sorted(results.iterdir()):
        match = PATTERN.match(directory.name)
        if not match or not directory.is_dir():
            continue
        data = row(directory)
        if data is not None:
            rows.append((match.group(1), match.group(2), int(match.group(3)), data))
    return sorted(rows, key=lambda item: (item[0], item[1], item[2]))


def _fmt(value: Any) -> str:
    return "-" if value is None else f"{value:,.0f}" if isinstance(value, float) else str(value)


def render(rows: list[tuple[str, str, int, dict[str, Any]]]) -> str:
    lines = [
        "| campaign | model | rep | correct (base / A) | median input (base -> A) | change | gate | fails on | previews | reads full / part / none | plugin |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for campaign, model, rep, d in rows:
        change = "-" if d["change_pct"] is None else f"{d['change_pct']:+.1f}%"
        reads = "-" if d["full"] is None else f"{d['full']} / {_fmt(d['partial'])} / {_fmt(d['unread'])}"
        lines.append(
            f"| {campaign} | {model} | R{rep} | {d['baseline_correct']} / {d['a_correct']} | "
            f"{_fmt(d['baseline_mi'])} -> {_fmt(d['a_mi'])} | {change} | {d['outcome']} | "
            f"{', '.join(d['reasons']) or '-'} | {_fmt(d['previews'])} | {reads} | {d['plugin_options']} |"
        )
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    results = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parents[1] / "results"
    rows = collect(results)
    if not rows:
        print(f"no complete repetitions under {results}", file=sys.stderr)
        return 1
    print(render(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
