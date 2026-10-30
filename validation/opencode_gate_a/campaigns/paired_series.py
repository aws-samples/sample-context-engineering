"""Per-turn pairing of the two arms of one CE-60 run, to read a median before trusting it.

Gate A thresholds the MEDIAN of 60 per-turn model inputs. That one number hides where the two arms differ: a
filtered result makes arm A lighter from the turn it happens, and a tool that one arm re-runs and the other does
not makes the gap close again. This script prints, for every turn, the model input of each arm (the last step of
the turn: uncached input plus cache read plus cache write, as the runner records it), the difference, how many
model steps each arm took and which previews and retrievals arm A made. ``--tools 31,36`` lists the tool names and
result sizes of the given turns in both arms, which is how a re-run that only one arm made shows up.

It reads sizes, counts and tool names from the local OpenCode store; no prompt, answer or result text is printed.

Usage: python3 paired_series.py <run_id> [--tools <turn>[,<turn>...]]   (e.g. ce60-v6-sonnet55-r1)
"""

from __future__ import annotations

import shutil
import statistics
import sys
from pathlib import Path

from analyze_matched_calls import open_store, session_steps

RETRIEVAL_TOOL = "retrieve_context"
REPO_ROOT = Path(__file__).resolve().parents[3]


def turn_labels() -> list[str]:
    """Scenario labels in EXECUTION order (the catalog's order is grouped and differs)."""
    sys.path.insert(0, str(REPO_ROOT / "validation" / "community-plugin-A-B-D"))
    try:
        from src import scenario  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
    return [turn.label for turn in scenario.turns(total=60)]


def per_turn(steps: list[dict]) -> dict[int, dict]:
    """Collapse model steps into turns: the model input is the LAST step's, plus steps and tool calls."""
    turns: dict[int, dict] = {}
    for step in steps:
        entry = turns.setdefault(step["turn"], {"input": None, "steps": 0, "calls": []})
        entry["input"] = step["input"] or entry["input"]
        entry["steps"] += 1
        entry["calls"].extend(step["calls"])
    return turns


def short_tool(name: object) -> str:
    return str(name).replace("ce60-fixture_", "")


def render(run_id: str, tools_for: set[int]) -> str:
    con, tmp = open_store()
    try:
        baseline = per_turn(session_steps(con, f"ce60-baseline-{run_id}-baseline"))
        treated = per_turn(session_steps(con, f"ce60-A-{run_id}-A"))
    finally:
        con.close()
        shutil.rmtree(tmp, ignore_errors=True)

    labels = turn_labels()
    base_inputs = [t["input"] for _, t in sorted(baseline.items()) if t["input"]]
    arm_inputs = [t["input"] for _, t in sorted(treated.items()) if t["input"]]
    base_median = statistics.median(base_inputs)
    arm_median = statistics.median(arm_inputs)
    change = (arm_median / base_median - 1) * 100
    header = f"{'#':>2} {'turn':16s} {'baseline':>9} {'A':>9} {'A-base':>8} {'b.st':>4} {'a.st':>4}"
    lines = [
        f"{run_id}: median model input baseline={base_median:,.1f} A={arm_median:,.1f} change={change:+.2f}%",
        f"{header}  arm A previews (P) and retrievals (R)",
    ]
    for number in range(1, 61):
        b, a = baseline.get(number), treated.get(number)
        if not b or not a or not b["input"] or not a["input"]:
            continue
        marks = []
        for call in a["calls"]:
            if call["previewed"]:
                marks.append(f"P:{short_tool(call['tool'])}({call['bytes']})")
            elif short_tool(call["tool"]) == RETRIEVAL_TOOL:
                marks.append(f"R:({call['bytes']})")
        label = labels[number - 1] if number - 1 < len(labels) else "?"
        lines.append(
            f"{number:>2} {label:16s} {b['input']:>9,} {a['input']:>9,} {a['input'] - b['input']:>+8,} "
            f"{b['steps']:>4} {a['steps']:>4}  {' '.join(marks)[:90]}"
        )
    for number in sorted(tools_for):
        lines.append(
            f"--- turn {number} {labels[number - 1] if number - 1 < len(labels) else '?'}"
        )
        for arm, turns in (("baseline", baseline), ("A", treated)):
            entry = turns.get(number)
            names = (
                ", ".join(
                    f"{short_tool(c['tool'])}({c['bytes']}B)" for c in entry["calls"]
                )
                if entry
                else "(no turn)"
            )
            lines.append(f"  {arm}: {names or '(no tool calls)'}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if not argv or argv[0].startswith("-"):
        print(__doc__, file=sys.stderr)
        return 2
    tools_for: set[int] = set()
    if len(argv) >= 3 and argv[1] == "--tools":
        tools_for = {int(token) for token in argv[2].split(",") if token.strip()}
    print(render(argv[0], tools_for))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
