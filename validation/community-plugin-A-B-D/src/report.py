"""Comparative report generation.

The report leads with the three things a decision needs: wall-clock time, per-turn time,
and token consumption — each as a total and as a per-call figure, because the totals and
the per-call numbers can disagree. A strategy that cuts input tokens per call while
adding cycles can end up costing more in total; that trade is the actual finding, and
hiding either half of it would misrepresent the result.

Token figures are reported twice: ``usage_*`` is what Bedrock billed, ``estimated_*`` is
what the harness computed from the assembled request. They should track each other. When
they do not, the gap is itself information — prompt caching, for instance, shows up as
billed input tokens far below the assembled size.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import RESULTS_DIR, RUN_CONFIGS


def _delta(current: float, baseline: float) -> str:
    """Format ``current`` as a signed percentage change against ``baseline``."""
    if not baseline:
        return "n/a"
    change = (current - baseline) / baseline * 100.0
    return f"{change:+.1f}%"


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def build_report(results: dict[str, dict[str, Any]], *, baseline: str = "baseline") -> str:
    """Render the comparative report as Markdown."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    order = [name for name in RUN_CONFIGS if name in results]
    base = results.get(baseline, {}).get("summary", {})

    lines: list[str] = [
        "# Context Strategy Validation — Comparative Report",
        "",
        f"Generated: {stamp}",
        "",
        "Three context-optimization strategies measured against a shared baseline on live",
        "Bedrock calls. Every configuration replays the same conversation script with the",
        "same tools and the same payloads, so the differences are the strategies.",
        "",
        "## Setup",
        "",
    ]

    meta = results.get(order[0], {}).get("meta", {}) if order else {}
    lines += [
        f"- Agent model: `{meta.get('agent_model', 'n/a')}`",
        f"- Rerank model: `{meta.get('rerank_model', 'n/a')}`",
        f"- Account / region: `{meta.get('account', 'n/a')}` / `{meta.get('region', 'n/a')}`",
        f"- Turns per run: {base.get('turns', 'n/a')}",
        f"- Replays per configuration: {meta.get('repeats', 1)}"
        + (
            ""
            if meta.get("repeats", 1) > 1
            else "  ⚠️ single replay — see *What this run does not settle*"
        ),
        f"- Tools registered: {results.get(baseline, {}).get('summary', {}).get('plugin_counters', {}).get('registered_tools', 'n/a')}",
        "",
        "All configurations include the ContextOffloader. Without it the oversized tool",
        "payloads overflow the window and the baseline fails rather than merely costing more,",
        "which would make this a report about crashing instead of about tokens. The baseline is",
        "therefore the offloader with its default positional prefix preview, and the `relevance`",
        "configuration changes only the preview strategy.",
        "",
        "## 1. Accuracy",
        "",
        "Read this table before the cost tables. A strategy that reduces tokens by discarding",
        "the passage the question needed would lead every cost table in this report, and nothing",
        "in them would notice. Cost is only a result once correctness holds.",
        "",
        "Checks are deterministic string assertions against values computed from the mocked tool",
        "implementations — no judge model, so scoring adds neither latency nor variance. Numeric",
        "expectations match the tools' own formatting, which also tests that no preview",
        "paraphrased a figure.",
        "",
        "- **Weighted**: fraction of expectation weight met across the run.",
        "- **Materially correct**: turns with no *critical* check failed — the stricter figure. A",
        "  turn can score 0.7 and still have missed the one number the user asked for.",
        "",
        "| Configuration | Weighted | Materially correct | Critical failures |",
        "|---|---:|---:|---:|",
    ]

    for name in order:
        summary = results[name]["summary"]
        acc = summary.get("accuracy") or {}
        if not acc:
            lines.append(f"| {RUN_CONFIGS[name].label} | n/a | n/a | n/a |")
            continue
        spread = acc.get("weighted_accuracy_spread_pct")
        spread_text = f" (±{spread:.0f}%)" if spread else ""
        lines.append(
            f"| {RUN_CONFIGS[name].label} "
            f"| {acc['weighted_accuracy'] * 100:.1f}%{spread_text} "
            f"| {acc.get('turns_materially_correct', 0):.1f}/{acc.get('turns_scored', 0)} "
            f"({acc.get('material_correctness', 0) * 100:.0f}%) "
            f"| {acc.get('critical_failures_total', 0):.1f} |"
        )

    lines += _accuracy_detail(results, order)

    lines += [
        "## 2. Wall-clock and turn timing",
        "",
        "| Configuration | Wall (s) | Turn total (s) | Turn mean (s) | Turn median (s) | Turn max (s) | Model total (s) | Overhead (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for name in order:
        summary = results[name]["summary"]
        timing = summary["timing"]
        lines.append(
            f"| {RUN_CONFIGS[name].label} "
            f"| {summary['wall_seconds']:,.1f} "
            f"| {timing['turn_seconds_total']:,.1f} "
            f"| {timing['turn_seconds_mean']:,.2f} "
            f"| {timing['turn_seconds_median']:,.2f} "
            f"| {timing['turn_seconds_max']:,.1f} "
            f"| {timing['model_seconds_total']:,.1f} "
            f"| {timing['overhead_seconds']:,.1f} |"
        )

    lines += [
        "",
        "`Overhead` is turn time minus model time: tool execution plus whatever the plugins do",
        "on the critical path. It is the column to watch for a strategy that buys tokens with",
        "latency.",
        "",
        "### Change against baseline",
        "",
        "| Configuration | Wall | Turn mean | Model total |",
        "|---|---:|---:|---:|",
    ]

    for name in order:
        if name == baseline:
            continue
        summary = results[name]["summary"]
        lines.append(
            f"| {RUN_CONFIGS[name].label} "
            f"| {_delta(summary['wall_seconds'], base.get('wall_seconds', 0))} "
            f"| {_delta(summary['timing']['turn_seconds_mean'], base.get('timing', {}).get('turn_seconds_mean', 0))} "
            f"| {_delta(summary['timing']['model_seconds_total'], base.get('timing', {}).get('model_seconds_total', 0))} |"
        )

    lines += [
        "",
        "## 3. Token consumption",
        "",
        "### Billed by the provider",
        "",
        "| Configuration | Calls | Input total | Input mean/call | Input max/call | Output total | Cache read |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for name in order:
        summary = results[name]["summary"]
        tokens = summary["tokens"]
        lines.append(
            f"| {RUN_CONFIGS[name].label} "
            f"| {summary['model_calls']} "
            f"| {tokens['usage_input_total']:,} "
            f"| {tokens['usage_input_mean_per_call']:,.0f} "
            f"| {tokens['usage_input_max_per_call']:,} "
            f"| {tokens['usage_output_total']:,} "
            f"| {tokens['cache_read_total']:,} |"
        )

    lines += [
        "",
        "### Assembled request, measured by the harness",
        "",
        "Split into the two components the strategies target separately: conversation history",
        "(the graph's domain, and the preview strategy's) and tool schemas (disclosure's).",
        "",
        "| Configuration | Input total | Input mean/call | History total | History mean/call | Schemas total | Schemas mean/call |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for name in order:
        tokens = results[name]["summary"]["tokens"]
        lines.append(
            f"| {RUN_CONFIGS[name].label} "
            f"| {tokens['estimated_input_total']:,} "
            f"| {tokens['estimated_input_mean_per_call']:,.0f} "
            f"| {tokens['message_tokens_total']:,} "
            f"| {tokens['message_tokens_mean_per_call']:,.0f} "
            f"| {tokens['tool_spec_tokens_total']:,} "
            f"| {tokens['tool_spec_tokens_mean_per_call']:,.0f} |"
        )

    lines += [
        "",
        "### Change against baseline",
        "",
        "`Noise` is the peak-to-peak spread of billed input tokens across replays of the same",
        "configuration, as a percentage of its own mean. A change smaller than the noise has not",
        "been resolved by this run, whatever its sign.",
        "",
        "| Configuration | Billed input | Assembled input | History | Schemas | Calls | Noise |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    base_tokens = base.get("tokens", {})
    for name in order:
        if name == baseline:
            continue
        summary = results[name]["summary"]
        tokens = summary["tokens"]
        noise = tokens.get("usage_input_total_spread_pct")
        lines.append(
            f"| {RUN_CONFIGS[name].label} "
            f"| {_delta(tokens['usage_input_total'], base_tokens.get('usage_input_total', 0))} "
            f"| {_delta(tokens['estimated_input_total'], base_tokens.get('estimated_input_total', 0))} "
            f"| {_delta(tokens['message_tokens_total'], base_tokens.get('message_tokens_total', 0))} "
            f"| {_delta(tokens['tool_spec_tokens_total'], base_tokens.get('tool_spec_tokens_total', 0))} "
            f"| {_delta(summary['model_calls'], base.get('model_calls', 0))} "
            f"| {'±' + format(noise, '.0f') + '%' if noise else 'n/a'} |"
        )

    lines += [
        "",
        "### Tool activity",
        "",
        "Retrieval count is the variable that explains most of the token totals: every",
        "`retrieve_offloaded_content` call appends a large result to the history, and that",
        "result then rides along on every subsequent call. A strategy that makes retrieval more",
        "attractive pays for it twice.",
        "",
        "| Configuration | Model calls | Tool uses | Retrievals |",
        "|---|---:|---:|---:|",
    ]

    for name in order:
        summary = results[name]["summary"]
        lines.append(
            f"| {RUN_CONFIGS[name].label} "
            f"| {summary['model_calls']} "
            f"| {summary.get('tool_uses_total', 0)} "
            f"| {summary.get('retrievals', 0)} |"
        )

    lines += [
        "",
        "## 4. Per-turn breakdown",
        "",
        "Turn time and the input tokens the turn's calls consumed, per configuration.",
        "",
    ]

    turn_labels = [turn["label"] for turn in results.get(order[0], {}).get("turns", [])]
    if turn_labels:
        header = "| Turn | " + " | ".join(RUN_CONFIGS[name].name for name in order) + " |"
        lines += [
            "### Turn duration (seconds)",
            "",
            header,
            "|---|" + "---:|" * len(order),
        ]
        for index, label in enumerate(turn_labels):
            row = [label]
            for name in order:
                turns = results[name]["turns"]
                row.append(f"{turns[index]['turn_seconds']:,.1f}" if index < len(turns) else "—")
            lines.append("| " + " | ".join(row) + " |")

        lines += [
            "",
            "### Input tokens assembled per turn",
            "",
            header,
            "|---|" + "---:|" * len(order),
        ]
        for index, label in enumerate(turn_labels):
            row = [label]
            for name in order:
                total = sum(
                    call["estimated_input_tokens"]
                    for call in results[name]["calls"]
                    if call["turn_index"] == index
                )
                row.append(f"{total:,}")
            lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## 5. Did each strategy actually engage?",
        "",
        "Token totals alone cannot distinguish a strategy that worked from one that was inert.",
        "These are the strategy-specific counters read off the plugins after each run.",
        "",
    ]

    for name in order:
        counters = results[name]["summary"].get("plugin_counters", {})
        lines += [f"**{RUN_CONFIGS[name].label}**", ""]
        if not counters:
            lines += ["- no counters captured", ""]
            continue
        graph = counters.get("graph")
        if graph:
            lines += _graph_evidence(graph)
        disclosure = counters.get("disclosure")
        if disclosure:
            lines.append(
                f"- Disclosure: {disclosure.get('searches', '?')} find_tools searches, "
                f"{disclosure.get('premature_cancellations', '?')} premature calls recovered, "
                f"{disclosure.get('exposed_count_at_end', '?')} tools exposed at end"
            )
        offloader = counters.get("offloader")
        if offloader:
            lines.append(
                f"- Offloader: strategy={offloader.get('preview_strategy')!r}, "
                f"rerank search units consumed={offloader.get('search_units')}"
            )
        resumed_at = results[name]["summary"].get("resumed_at_turn")
        if resumed_at is not None:
            lines.append(
                f"- **Resumed at turn {resumed_at}**: a fresh agent took over with "
                f"{results[name]['summary'].get('resumed_message_count')} messages restored from the "
                "session. A long-lived process restores nothing, so this is the only shape in which "
                "the graph's load path runs at all."
            )
        lines.append(
            f"- Live history at end: {counters.get('live_messages_at_end')} messages; "
            f"tools registered: {counters.get('registered_tools')}"
        )
        lines.append("")

    errors = {name: results[name]["summary"].get("errors", []) for name in order}
    if any(errors.values()):
        lines += ["## 6. Errors", ""]
        for name, items in errors.items():
            if items:
                lines.append(f"- **{name}**: {len(items)} error(s)")
                for item in items[:5]:
                    lines.append(f"  - `{item}`")
        lines.append("")

    lines += _significance(results, order, baseline)
    lines += _efficiency(results, order, baseline)
    lines += _interpretation(results, order, baseline)
    return "\n".join(lines)


def _graph_evidence(graph: dict[str, Any]) -> list[str]:
    """The graph's own evidence that it decided rather than merely ran.

    Four readings, and each one distinguishes a working graph from an inert one that happens to
    save tokens. The resolution distribution says whether the dialogue ladder has three rungs in
    practice or degenerated into two. The retrieval-cycle curve is the one that decides whether
    the graph works at all: descending means the fed-back note learned from the model's requests,
    flat or ascending means it traded tokens for latency. The full-pass count says how many calls
    the graph declined to touch. And the choice overhead has to stay near the cost of one
    embedding, or the matcher is in the wrong place whatever the token column says.
    """
    if "error" in graph:
        lines = [f"- Graph: state unavailable (`{graph['error']}`)"]
        return lines

    lines = [
        f"- Graph: {graph.get('cards', '?')} Cards at end "
        f"({graph.get('subject_cards', '?')} subject, {graph.get('artifact_cards', '?')} artifact), "
        f"{graph.get('links', '?')} Links, {graph.get('vectors_cached', '?')} description vectors cached, "
        f"{graph.get('reuse_at_end', '?')} fed-back notes live"
    ]

    if "addressed" in graph:
        lines.append(
            f"- Graph selection at end: {graph['addressed']} Cards addressed, "
            f"{graph['unaddressed']} left out and announced as searchable — a zero in the second "
            "number means the selection excluded nothing, so nothing else in this run is its effect"
        )

    per_turn = graph.get("per_turn") or {}
    if not per_turn:
        return lines
    if per_turn.get("unaddressed_max") is not None:
        lines.append(
            f"- Graph Cards left out of the call, per turn: peak {per_turn['unaddressed_max']}, "
            f"per turn {' '.join(str(count) for count in per_turn.get('unaddressed_per_turn', []))}"
        )

    dialogue = per_turn.get("dialogue_per_turn") or []
    if dialogue:
        lines.append(
            "- Graph dialogue axis per turn (full/description/title): "
            + " ".join(f"{full}/{descr}/{title}" for full, descr, title in dialogue)
        )
        lines.append(
            f"- Graph three-rung turns: {per_turn.get('turns_with_three_dialogue_rungs', 0)} of "
            f"{per_turn.get('turns_recorded', 0)} — a zero here means the ladder made a two-way "
            "decision, not a three-way one"
        )
    evidence = per_turn.get("evidence_per_turn") or []
    if evidence:
        lines.append(
            "- Graph evidence axis per turn (full/description; there is no title rung): "
            + " ".join(f"{full}/{descr}" for full, descr in evidence)
        )
    cycles = per_turn.get("retrieval_cycles_per_turn")
    if cycles is not None:
        lines.append(
            f"- Graph retrieval cycles per turn: {' '.join(str(count) for count in cycles)} "
            f"(total {per_turn.get('retrieval_cycles_total', 0)}, "
            f"first half {per_turn.get('retrieval_cycles_first_half', 0)} vs second half "
            f"{per_turn.get('retrieval_cycles_second_half', 0)}: "
            f"**{per_turn.get('retrieval_cycles_reading', 'n/a')}**)"
        )
    if "note_min_of_run" in per_turn:
        lines.append(
            f"- Graph note spread over the run: min {per_turn['note_min_of_run']:.3f}, "
            f"max {per_turn['note_max_of_run']:.3f}, mean of the per-turn medians "
            f"{per_turn.get('note_median_mean')}; "
            f"{per_turn.get('notes_below_floor_total')} of {per_turn.get('notes_scored_total')} "
            f"scored Cards fell below `collapse_floor` and "
            f"{per_turn.get('notes_above_threshold_total')} reached `expand_threshold` — a floor no "
            "note ever falls below is a threshold outside the matcher's range, not an unused rung"
        )
    if "compaction_ratio_median" in per_turn:
        lines.append(
            f"- Graph compaction ratio on Cards at description: median "
            f"{per_turn['compaction_ratio_median']}x, mean {per_turn.get('compaction_ratio_mean')}x "
            f"over {per_turn.get('compaction_ratio_samples')} samples"
        )
    lines.append(
        f"- Graph deliveries: {per_turn.get('deliveries', 0)}, of which "
        f"{per_turn.get('full_pass_deliveries', 0)} passed the context through untouched; "
        f"{per_turn.get('final_blocks_total', 0)} final blocks folded; "
        f"choice overhead mean {per_turn.get('choice_micros_mean', 0) / 1000:.3f}ms "
        "(one embedding round is ~150ms, and is not in this number)"
    )
    return lines


def _significance(results: dict[str, dict[str, Any]], order: list[str], baseline: str) -> list[str]:
    """State, per strategy, which differences this run actually resolved.

    Every other table in this report invites a conclusion. This one withholds the ones the
    data cannot support. The rule is deliberately crude — a difference counts as resolved
    only when it exceeds the combined peak-to-peak spread of the two configurations being
    compared — because a crude rule applied consistently is harder to argue past than a
    judgement call made per row.

    It earns its place: a single-replay run of this harness showed relevance filtering
    improving accuracy on the one turn whose answer sits mid-document, which is exactly what
    the design predicts. Three replays did not reproduce it — that turn came out a coin-flip
    for every configuration. Without this section the report would keep presenting the
    flattering read as a finding.
    """
    base = results.get(baseline, {}).get("summary", {})
    base_acc = base.get("accuracy") or {}
    base_tokens = base.get("tokens", {})
    repeats = results.get(order[0], {}).get("meta", {}).get("repeats", 1) if order else 1

    lines = [
        "## Which differences this run resolved",
        "",
        f"Replays per configuration: **{repeats}**. A difference is called resolved only when it",
        "exceeds the combined spread of the two configurations compared. Anything else is reported",
        "as unresolved regardless of its sign — the measurement does not support a claim either way.",
        "",
        "Total billed tokens depend on the tool path the agent chose, which varies per replay, so",
        "that column is noisy by nature. The *schemas* column is not: it is a projection computed",
        "from the registry, so its size is mechanically determined rather than behavioural. When a",
        "strategy's effect is real but the total is unresolved, the schemas column is usually where",
        "to look.",
        "",
        "| Comparison | Accuracy Δ | Accuracy noise | Tokens Δ | Token noise | Schemas Δ | Schema noise | Resolved |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    for name in order:
        if name == baseline:
            continue
        summary = results[name]["summary"]
        acc = summary.get("accuracy") or {}

        acc_delta = (acc.get("weighted_accuracy", 0) - base_acc.get("weighted_accuracy", 0)) * 100
        acc_noise = (
            acc.get("weighted_accuracy_spread_pct", 0) + base_acc.get("weighted_accuracy_spread_pct", 0)
        ) / 2
        tok_delta_pct = _delta(
            summary["tokens"]["usage_input_total"], base_tokens.get("usage_input_total", 0)
        )
        tok_noise = (
            summary["tokens"].get("usage_input_total_spread_pct", 0)
            + base_tokens.get("usage_input_total_spread_pct", 0)
        ) / 2

        try:
            tok_delta_abs = abs(float(tok_delta_pct.rstrip("%")))
        except ValueError:
            tok_delta_abs = 0.0

        spec_delta_pct = _delta(
            summary["tokens"]["tool_spec_tokens_total"], base_tokens.get("tool_spec_tokens_total", 0)
        )
        spec_noise = (
            summary["tokens"].get("tool_spec_tokens_total_spread_pct", 0)
            + base_tokens.get("tool_spec_tokens_total_spread_pct", 0)
        ) / 2
        try:
            spec_delta_abs = abs(float(spec_delta_pct.rstrip("%")))
        except ValueError:
            spec_delta_abs = 0.0

        acc_resolved = abs(acc_delta) > max(acc_noise, 1.0)
        tok_resolved = tok_delta_abs > max(tok_noise, 1.0)
        spec_resolved = spec_delta_abs > max(spec_noise, 1.0)

        verdict = [
            f"tokens {'✅' if tok_resolved else '✗'}",
            f"schemas {'✅' if spec_resolved else '✗'}",
            f"accuracy {'✅' if acc_resolved else '✗'}",
        ]

        lines.append(
            f"| {RUN_CONFIGS[name].name} vs {baseline} "
            f"| {acc_delta:+.1f} pp "
            f"| ±{acc_noise:.1f} pp "
            f"| {tok_delta_pct} "
            f"| ±{tok_noise:.0f}% "
            f"| {spec_delta_pct} "
            f"| ±{spec_noise:.0f}% "
            f"| {', '.join(verdict)} |"
        )

    lines += [
        "",
        "✅ resolved by this run · ✗ inside the noise, treat as no measured difference",
        "",
    ]
    if repeats < 3:
        lines += [
            "> ⚠️ Fewer than three replays. The spread columns are unreliable at this sample size,",
            "> so the verdicts above are indicative only. Re-run with `--repeats 3` or more.",
            "",
        ]
    return lines


def _efficiency(results: dict[str, dict[str, Any]], order: list[str], baseline: str) -> list[str]:
    """Tokens per materially-correct turn: the figure that settles the trade-off.

    Neither table alone decides anything. Cheapest-and-wrong and most-accurate-at-any-price
    are both easy to achieve and neither is useful. Dividing billed input tokens by the count
    of turns that answered correctly puts both on one axis, and lower is better.

    It is a ratio over a small denominator, so it is coarse — one turn moving in or out of
    'correct' shifts it visibly. It ranks; it does not measure.
    """
    lines = [
        "## Total cost of ownership",
        "",
        "Input tokens the agent assembled per configuration, against the baseline. The graph's",
        "embedding and rerank calls are billed separately — see the cost table in `compare.py`,",
        "which prices every auxiliary call a strategy makes.",
        "",
        "| Configuration | Agent input | vs baseline |",
        "|---|---:|---:|",
    ]

    def total_of(name: str) -> int:
        return int(results[name]["summary"]["tokens"]["usage_input_total"])

    base_total = total_of(baseline) if baseline in results else 0

    for name in order:
        total = total_of(name)
        lines.append(
            f"| {RUN_CONFIGS[name].label} "
            f"| {total:,} "
            f"| {_delta(total, base_total) if name != baseline else '—'} |"
        )

    lines += [
        "",
        "### Cost per correct answer",
        "",
        "Total input tokens divided by turns answered materially correctly. Lower is better. Neither",
        "the cost tables nor the accuracy table decides anything on its own — cheap and wrong is easy,",
        "and so is accurate at any price. This is the ratio that puts them on one axis. The denominator",
        "is small, so treat it as a ranking rather than a measurement.",
        "",
        "| Configuration | Total input | Correct turns | Tokens per correct turn | vs baseline |",
        "|---|---:|---:|---:|---:|",
    ]

    base_summary = results.get(baseline, {}).get("summary", {})
    base_acc = base_summary.get("accuracy") or {}
    base_correct = base_acc.get("turns_materially_correct", 0) or 0
    base_ratio = base_total / base_correct if base_correct else 0

    for name in order:
        acc = results[name]["summary"].get("accuracy") or {}
        correct = acc.get("turns_materially_correct", 0) or 0
        total = total_of(name)
        ratio = total / correct if correct else 0
        lines.append(
            f"| {RUN_CONFIGS[name].label} "
            f"| {total:,} "
            f"| {correct:.1f} "
            f"| {ratio:,.0f} "
            f"| {_delta(ratio, base_ratio) if name != baseline else '—'} |"
        )

    lines.append("")
    return lines


def _critical_of(turn: dict[str, Any]) -> list[str]:
    """Critical failures for a turn, from either the single-replay or aggregated shape.

    A single replay records ``critical_failures``; the aggregate unions them into
    ``critical_failures_seen``. Reading only one key silently dropped the warning markers
    from single-replay reports.
    """
    return turn.get("critical_failures_seen") or turn.get("critical_failures") or []


def _accuracy_detail(results: dict[str, dict[str, Any]], order: list[str]) -> list[str]:
    """Per-turn accuracy grid, plus the specific checks that failed.

    The grid is what localizes a regression to a turn, and the turns are not
    interchangeable: T9 is the one that depends on T1, so a drop there is evidence the
    graph collapsed the main subject and did not restore it. A drop on T5 or T8 points at
    the preview instead. Naming the failed checks is what makes the difference legible
    rather than leaving a number to be interpreted.
    """
    labels: list[str] = []
    for name in order:
        for turn in (results[name]["summary"].get("accuracy") or {}).get("per_turn", []):
            if turn["label"] not in labels:
                labels.append(turn["label"])
    if not labels:
        return [""]

    lines = [
        "",
        "### Per-turn accuracy",
        "",
        "| Turn | " + " | ".join(RUN_CONFIGS[name].name for name in order) + " |",
        "|---|" + "---:|" * len(order),
    ]

    for label in labels:
        row = [label]
        for name in order:
            entry = next(
                (
                    turn
                    for turn in (results[name]["summary"].get("accuracy") or {}).get("per_turn", [])
                    if turn["label"] == label
                ),
                None,
            )
            if not entry or not entry.get("scored", True):
                row.append("—")
            else:
                flag = " ⚠" if _critical_of(entry) else ""
                row.append(f"{entry['score'] * 100:.0f}%{flag}")
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "⚠ marks a turn where a critical check failed in at least one replay.",
        "",
    ]

    detail: list[str] = []
    for name in order:
        failures = [
            f"`{turn['label']}` → {', '.join(_critical_of(turn))}"
            for turn in (results[name]["summary"].get("accuracy") or {}).get("per_turn", [])
            if _critical_of(turn)
        ]
        if failures:
            detail.append(f"- **{RUN_CONFIGS[name].name}**: " + "; ".join(failures))

    if detail:
        lines += ["#### Critical checks that failed", "", *detail, ""]

    return lines


def _interpretation(results: dict[str, dict[str, Any]], order: list[str], baseline: str) -> list[str]:
    """Explain each strategy against the column it was supposed to move.

    Written as a check rather than a conclusion: each bullet states what the strategy
    targets, then reports whether this run's numbers show it. A strategy can engage
    correctly and still cost more, and that outcome should read as a finding rather than
    as a failure to measure.
    """
    base = results.get(baseline, {}).get("summary", {})
    base_tokens = base.get("tokens", {})

    def summary_of(name: str) -> dict[str, Any]:
        return results.get(name, {}).get("summary", {})

    lines = ["## Reading these numbers", ""]

    disclosure = summary_of("disclosure")
    if disclosure:
        counters = disclosure.get("plugin_counters", {}).get("disclosure", {})
        lines += [
            "**Disclosure** targets the *Schemas* column and is expected to add calls.",
            "",
            f"- Schemas: {_delta(disclosure['tokens']['tool_spec_tokens_total'], base_tokens.get('tool_spec_tokens_total', 0))} "
            f"— the largest single-column move in this run.",
            f"- Calls: {_delta(disclosure['model_calls'], base.get('model_calls', 0))}. "
            f"{counters.get('searches', '?')} were `find_tools` searches; "
            f"{counters.get('premature_cancellations', '?')} were premature calls that had to be retried "
            f"once the schema loaded.",
            "- Premature cancellations are the number to watch. Each one costs a cycle, and a cycle",
            "  carries the whole history — so they can hand back more than the schema saving. When this",
            "  count is high relative to searches, the catalog is telling the model a tool exists",
            "  without telling it enough to call correctly, and raising `catalog_chars` or naming the",
            "  frequent tools in `always_available` is the cheaper fix.",
            "",
        ]

    relevance = summary_of("relevance")
    if relevance:
        units = relevance.get("plugin_counters", {}).get("offloader", {}).get("search_units", 0)
        lines += [
            "**Relevance** does not target token totals at all. The preview budget is identical",
            "either way; what changes is which text fills it.",
            "",
            f"- Consumed {units} rerank search units, so the scorer ran and paged rather than",
            "  silently falling back to the positional prefix.",
            f"- Retrievals: {relevance.get('retrievals', 0)} against the baseline's {base.get('retrievals', 0)}. "
            f"Input tokens {_delta(relevance['tokens']['usage_input_total'], base_tokens.get('usage_input_total', 0))}.",
            "- If retrievals rose, that is the mechanism behind the token total, and it is worth",
            "  reading as a behavioural change rather than as overhead: the gap markers state how many",
            "  lines were omitted and that `line_range` will fetch them, which is an explicit invitation",
            "  to retrieve. The model taking it up is the feature working as designed. Whether it is",
            "  worth the tokens depends on whether the retrieved detail was needed — which the",
            "  `response_text` captured per turn in the JSON is what settles, not this table.",
            "",
        ]

    combined = summary_of("all")
    if combined and disclosure and relevance:
        lines += [
            "**Combined** is where interference would show.",
            "",
            f"- Schemas {_delta(combined['tokens']['tool_spec_tokens_total'], base_tokens.get('tool_spec_tokens_total', 0))} "
            f"and history {_delta(combined['tokens']['message_tokens_total'], base_tokens.get('message_tokens_total', 0))}.",
            "- The two touch different parts of the request — schemas and the content of a single tool",
            "  result — so their effects should be close to additive. A large shortfall against the",
            "  individual runs means they are fighting: the most likely cause is disclosure retiring a",
            "  schema the preview had just made the model want to call, which costs a rediscovery cycle.",
            "",
        ]

    lines += [
        "### What this run does not settle",
        "",
        "- One replay per configuration. The agent is non-deterministic and `temperature` cannot be",
        "  pinned on the agent model, so single-digit percentage differences are inside the noise. Treat",
        "  the direction and the large moves as signal, and re-run with a different `--tag` before",
        "  believing anything smaller.",
        "- Token totals are a proxy for cost, not cost. Input and output tokens are priced",
        "  differently, and the graph's embedding and rerank calls bill against separate models.",
        "- Answer quality is only auditable from the `response_text` fields in the JSON. No metric",
        "  here asserts that a cheaper run answered as well.",
        "",
    ]
    return lines


def write_outputs(results: dict[str, dict[str, Any]], *, tag: str | None = None) -> tuple[Path, Path]:
    """Persist the raw JSON and the rendered Markdown report."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = tag or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    json_path = RESULTS_DIR / f"run-{stamp}.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    report_path = RESULTS_DIR / f"report-{stamp}.md"
    report_path.write_text(build_report(results), encoding="utf-8")

    latest = RESULTS_DIR / "report-latest.md"
    latest.write_text(build_report(results), encoding="utf-8")

    return json_path, report_path
