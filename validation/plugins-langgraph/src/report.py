"""Render a recorded run as the benchmark table, from its JSON alone.

**No credentials, no network, no model.** Everything the table prints was measured during the run
and written into the JSON; this module is arithmetic over that file. That is deliberate and it is a
property worth keeping: a published figure must be re-derivable by a reader who has the artefact and
no access to the account that produced it, and a correction to a rate is then an edit to
``config.MODEL_PRICING`` plus a re-render rather than a re-run.

    .venv/bin/python -m src.report results/run-<tag>.json

The columns are the ones ``BENCHMARK.md`` publishes for the Strands harness, in the same order, so a
LangGraph table and a Strands table can be read side by side:

    | Configuration | Answered | Scored lost | Refused | Total tokens | Δ vs heaviest | Accuracy |
    | Correct | Peak/call | Turn | Cost | $/correct |

Three of those columns exist because of a mistake this harness's sibling made and was corrected on.

**``Answered`` / ``Scored lost`` / ``Refused``, and why the delta is against the heaviest arm.** An
arm whose history outgrows the model's window stops completing turns, and a turn that never went out
spends nothing -- so a truncated arm reports a LOW token total that reads like efficiency. Measuring
the saving against such a baseline inverts the sign of the result. The reference is therefore the arm
that consumed the MOST tokens, whichever that is, and the completion columns sit to its left so no
reader can see a total without seeing whether the turns behind it finished.

**``Peak/call``** is the truncation-immune quantity: the largest single call the arm actually got
away with. An arm that overflowed has a peak at the ceiling and a truncated total, and those two
facts together are the finding.

Accuracy is only comparable where ``Answered`` equals the total. The renderer marks any arm that
fell short and says so under the table rather than leaving the reader to divide the columns.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import compare, config
from .config import RESULTS_DIR, RUN_CONFIGS

_HEADER = (
    "| Configuration | Answered | Scored lost | Refused | Total tokens | Δ vs heaviest | "
    "Accuracy | Correct | Peak/call | Turn | Cost | $/correct |"
)
_ALIGN = "|---|:--:|---:|---:|---:|---:|---:|:--:|---:|---:|---:|---:|"


_NOT_AN_ARM = frozenset({"meta"})
"""Top-level keys of a run JSON that are not configurations.

``compare.ordered_names`` appends anything it does not recognise so a sweep variant still renders,
which is right for an arm and wrong for the run's metadata: without this filter ``meta`` came out as
a sixth row reading 0 tokens, 0% accuracy and −100% against the reference. Measured in the smoke
test, which is what caught it.
"""


def arms_of(payload: dict[str, Any]) -> dict[str, Any]:
    """Return only the configuration blocks of a run payload, in reading order."""
    return {
        name: payload[name]
        for name in compare.ordered_names(payload)
        if name not in _NOT_AN_ARM and isinstance(payload.get(name), dict) and "summary" in payload[name]
    }


def _completion(entry: dict[str, Any]) -> tuple[int, int, int, int]:
    """Return ``(answered, total, scored_lost, refused)`` for one arm.

    ``answered`` counts turns that produced any assistant text, including a turn cut short at the
    output cap -- it answered and was interrupted, which is a different verdict from never answering
    and was being conflated with it. ``refused`` counts turns that errored with nothing recovered,
    which is what an overflow looks like. ``scored_lost`` is the scored turns among those, the ones
    that cost the arm accuracy rather than only cost it a filler line.

    Args:
        entry: One arm's block of the run JSON.

    Returns:
        The four counts, in that order.
    """
    turns = entry.get("turns") or []
    answered = sum(1 for turn in turns if (turn.get("response_text") or "").strip())
    refused = sum(
        1
        for turn in turns
        if turn.get("error") and not (turn.get("response_text") or "").strip()
    )
    per_turn = {
        item.get("label"): item
        for item in ((entry.get("summary") or {}).get("accuracy") or {}).get("per_turn") or []
    }
    scored_lost = sum(
        1
        for turn in turns
        if turn.get("label") in per_turn
        and turn.get("error")
        and not (turn.get("response_text") or "").strip()
    )
    return answered, len(turns), scored_lost, refused


def _peak(entry: dict[str, Any]) -> int:
    """Return the arm's largest single call's input, preferring the provider's own figure.

    ``usage_input_max_per_call`` is what the provider billed; the estimate is the fallback for a run
    whose provider reported no usage. Reported rather than averaged because the peak is what meets
    the context window, and a mean hides the call that did not fit.
    """
    tokens = (entry.get("summary") or {}).get("tokens") or {}
    return int(tokens.get("usage_input_max_per_call") or tokens.get("estimated_input_max_per_call") or 0)


def build_table(payload: dict[str, Any]) -> str:
    """Render the benchmark table for one run payload.

    Args:
        payload: A run JSON, keyed by configuration name, as :func:`write_outputs` wrote it.

    Returns:
        The markdown table plus the reading notes that belong directly below it.
    """
    arms = arms_of(payload)
    rows = compare.summary_rows(arms)
    if not rows:
        return "_no configurations in this run_"

    # The reference is the heaviest consumer, not the baseline: see the module docstring.
    heaviest = max(rows, key=lambda row: row.total_tokens)

    lines = [_HEADER, _ALIGN]
    truncated: list[str] = []
    for row in rows:
        entry = arms[row.name]
        answered, total, scored_lost, refused = _completion(entry)
        if total and answered < total:
            truncated.append(f"{row.label} ({answered}/{total})")

        delta = (
            "reference"
            if row is heaviest
            else f"{compare.delta_pct(row.total_tokens, heaviest.total_tokens):+.1f}%"
        )
        per_correct = row.cost / row.materially_correct if row.materially_correct else 0.0
        bound = "*" if row.cache_cost_is_bound else ""
        lines.append(
            f"| {row.label} | {answered}/{total} | {scored_lost} | {refused} "
            f"| {row.total_tokens:,.0f} | {delta} | {row.weighted_accuracy * 100:.1f}% "
            f"| {row.materially_correct:g}/{row.turns_scored:g} | {_peak(entry):,} "
            f"| {row.turn_seconds_mean:.1f}s | ${row.cost:,.2f}{bound} "
            f"| ${per_correct:,.2f} |"
        )

    lines.append("")
    lines.extend(_notes(payload, rows, heaviest, truncated))
    return "\n".join(lines)


def _notes(
    payload: dict[str, Any],
    rows: list[compare.Row],
    heaviest: compare.Row,
    truncated: list[str],
) -> list[str]:
    """Return the reading notes that go immediately below the table.

    Below the table on purpose. A reader who has to scroll to a separate section to learn that an arm
    abandoned turns will read the token column as a result, and that has happened.
    """
    meta = _meta_of(payload)
    model = meta.get("agent_model") or config.AGENT_MODEL_ID
    pricing = config.pricing_for(model) if model in config.MODEL_PRICING else config.PRICING

    notes = [
        f"Reference for Δ is **{heaviest.label}**, the arm that consumed the most tokens"
        + (
            " — which happens to be the baseline in this run, but is not chosen for that reason."
            if heaviest.name == "baseline"
            else " — not the baseline."
        )
        + " An arm that overflows the window abandons turns and therefore stops spending, so a delta "
        "taken against a truncated baseline reads backwards.",
        "",
        f"`{model}` at ${pricing.agent_input_per_mtok:.2f}/${pricing.agent_output_per_mtok:.2f} per "
        f"Mtok in/out, embedding ${pricing.embedding_per_mtok:.2f} per Mtok, rerank "
        f"${pricing.rerank_per_ksearchunit:.2f} per 1k search units. Region "
        f"`{meta.get('region', config.REGION)}`, "
        f"{meta.get('repeats', rows[0].repeats if rows else 1)} replica(s), framework "
        f"`{meta.get('framework', 'langgraph')}`.",
    ]

    if truncated:
        notes += [
            "",
            "**Did not finish every turn — window overflow: " + "; ".join(truncated) + ".** "
            "Accuracy for these arms is truncation, not a property of the strategy, and must not be "
            "compared with an arm that finished. `Peak/call` is the comparable quantity: it is "
            "measured on calls that went out.",
        ]
    else:
        notes += ["", "Every arm finished every turn, so the accuracy column is comparable across rows."]

    if any(row.cache_cost_is_bound for row in rows):
        notes += [
            "",
            "`*` — the model declares no prompt-cache rates, so cache traffic is billed at the "
            "uncached input rate. That is an upper bound, not a figure.",
        ]

    if len(rows) and rows[0].repeats == 1:
        notes += [
            "",
            "**n=1.** A single replica separates nothing smaller than its own run-to-run variance. "
            "Read a difference of one or two turns, or of some tens of percent in tokens, as "
            "unresolved rather than as a result.",
        ]

    return notes


def _meta_of(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the run's meta block, from wherever this payload version carries it."""
    if isinstance(payload.get("meta"), dict):
        return payload["meta"]
    for entry in payload.values():
        if isinstance(entry, dict) and isinstance(entry.get("meta"), dict):
            return entry["meta"]
    return {}


def build_report(payload: dict[str, Any]) -> str:
    """Render the full markdown report: the table, its notes, and the per-arm evidence.

    Args:
        payload: A run JSON, keyed by configuration name.

    Returns:
        The markdown document.
    """
    meta = _meta_of(payload)
    lines = [
        "# LangGraph A/B/D validation",
        "",
        f"Rendered {datetime.now(timezone.utc).isoformat(timespec='seconds')} from the run JSON. "
        "No credentials were used and no model was called: every figure below was measured during "
        "the run and is arithmetic over the recorded file.",
        "",
        "## Results",
        "",
        build_table(payload),
        "",
        "## What each strategy did",
        "",
    ]

    for name, entry in arms_of(payload).items():
        label = RUN_CONFIGS[name].label if name in RUN_CONFIGS else name
        counters = (entry.get("summary") or {}).get("plugin_counters") or {}
        lines += [f"### {label}", ""]
        if not counters:
            lines += ["_no counters recorded_", ""]
            continue
        for key in ("disclosure", "offloader", "graph", "embedding_cost", "rerank_observed"):
            section = counters.get(key)
            if isinstance(section, dict) and section:
                lines.append(f"- **{key}**: " + ", ".join(f"{k}={v}" for k, v in section.items()))
        lines += [
            f"- **registered_tools**: {counters.get('registered_tools', '?')}",
            f"- **live_messages_at_end**: {counters.get('live_messages_at_end', '?')}",
            "",
        ]

    if meta:
        lines += [
            "## Run metadata",
            "",
            "```json",
            json.dumps(meta, indent=2, sort_keys=True),
            "```",
            "",
        ]

    return "\n".join(lines)


def write_outputs(
    results: dict[str, dict[str, Any]],
    *,
    tag: str | None = None,
) -> tuple[Path, Path]:
    """Write the run JSON and its markdown report.

    Args:
        results: The payload to store, keyed by configuration name.
        tag: Run tag used in both filenames. Defaults to ``latest``.

    Returns:
        The JSON path and the markdown path.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"run-{tag or 'latest'}"
    json_path = RESULTS_DIR / f"{stem}.json"
    md_path = RESULTS_DIR / f"{stem}.md"

    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    md_path.write_text(build_report(results), encoding="utf-8")
    return json_path, md_path


def main() -> int:
    """Re-render a stored run. Usage: ``python -m src.report results/run-<tag>.json``."""
    if len(sys.argv) < 2:
        print(__doc__ or "", file=sys.stderr)
        print("usage: python -m src.report <results/run-TAG.json>", file=sys.stderr)
        return 2

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"no such file: {path}", file=sys.stderr)
        return 1

    payload = json.loads(path.read_text(encoding="utf-8"))
    out = path.with_suffix(".md")
    out.write_text(build_report(payload), encoding="utf-8")
    print(build_table(payload))
    print()
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
