"""The comparison table: baseline against each strategy, on tokens, accuracy, latency and cost.

Separate from :mod:`src.report`, which is a diagnostic document — every counter a
strategy exposes, so a number that looks wrong can be traced to the mechanism that produced it.
This module answers the other question, the one asked before any of that matters: *what did each
strategy cost, and what did it buy?* Six columns, one row per configuration.

**Cost here is measured units times a declared rate.** The units come from the run: provider
``usage`` for the agent, the graph's embedding calls counted after its cache, and rerank search
units counted as Bedrock bills them. The rates come from
:data:`src.config.PRICING`. So a wrong price is fixed by editing that class and
re-rendering, never by re-running.

**Auxiliary calls are the point of the cost column.** Two strategies here spend on a second
model: the graph on embeddings and relevance filtering on rerank. A table that showed only the
agent's tokens would rank exactly those two too favourably, because their saving is bought with a
bill that does not appear in the agent's usage.

Usage:
    python -m src.compare results/run-comparison.json
    python -m src.compare results/run-comparison.json --out results/comparison.md
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from . import config
from .config import AGENT_MODEL_ID, PRICING, RESULTS_DIR, RUN_CONFIGS

_ORDER = ("baseline", "disclosure", "relevance", "graph", "all", "graph-all")
"""Reading order: the reference first, then one strategy at a time, then the two stacks.

Anything else present in the payload is appended after these, so a variant run still renders.
"""

_BASELINE = "baseline"

_PLATEAU_TURNS = 20
"""Tail length used as the plateau when the script carries no filler turns.

Long enough to outlast a single expensive turn, short enough not to reach back into whatever the
script was doing before it settled.
"""


def _plateau(points: list[TurnPoint]) -> list[TurnPoint]:
    """Return the stretch of the series where the history is the only thing changing.

    The filler turns, when the script has them: they ask nothing that needs a tool and exist purely
    to lengthen the conversation, which makes them the only stretch where a rise in cost can be
    attributed to the history. Falls back to the tail for a script without filler.

    Args:
        points: One configuration's per-turn series.

    Returns:
        The plateau's points, in turn order.
    """
    filler = [point for point in points if "filler" in point.label]
    return filler if len(filler) >= 2 else points[-_PLATEAU_TURNS:]


def _slope(values: list[int]) -> float:
    """Least-squares slope of ``values`` against turn ordinal, in units per turn.

    First-to-last would be the cheaper measure and it is the wrong one here: the series is not
    monotonic — a turn that made eight model calls towers over its neighbours — so two endpoints can
    describe a rising curve as flat. A fit reads every point.
    """
    count = len(values)
    if count < 2:
        return 0.0
    mean_x = (count - 1) / 2
    mean_y = sum(values) / count
    variance = sum((index - mean_x) ** 2 for index in range(count))
    if not variance:
        return 0.0
    covariance = sum((index - mean_x) * (value - mean_y) for index, value in enumerate(values))
    return covariance / variance


@dataclass
class Row:
    """One configuration's figures, already reduced to what the table prints."""

    name: str
    label: str
    repeats: int
    agent_input: float
    agent_output: float
    agent_tokens: float
    embed_tokens: float
    aux_tokens: float
    total_tokens: float
    token_spread_pct: float
    weighted_accuracy: float
    materially_correct: float
    turns_scored: int
    accuracy_spread_pct: float
    wall_seconds: float
    turn_seconds_mean: float
    turn_seconds_spread_pct: float
    agent_cost: float
    embedding_cost: float
    rerank_cost: float
    embed_calls: int
    search_units: int
    cache_read: float = 0.0
    """Input tokens served from a prompt cache. Non-zero even on an uncached run against a model
    that caches implicitly, which the OpenAI models do on the Converse path."""
    cache_write: float = 0.0
    """Input tokens written to a prompt cache."""
    cache_cost_is_bound: bool = False
    """True when the model declares no cache rates and cache traffic was billed at the uncached
    input rate, making this row's cost an upper bound rather than a figure."""

    @property
    def cost(self) -> float:
        """All-in cost of the run: the agent plus every auxiliary call the strategy made."""
        return self.agent_cost + self.embedding_cost + self.rerank_cost


def _cost(tokens: float, per_mtok: float) -> float:
    return tokens / 1_000_000.0 * per_mtok


def _cache_cost(tokens: dict[str, Any]) -> tuple[float, bool]:
    """Return what this run's prompt-cache traffic cost, and whether that figure is a bound.

    Bedrock reports cached input separately from ``inputTokens``, so cache traffic contributes
    nothing to the uncached input figure and has to be billed on its own. An uncached run reports
    zeros here and this returns ``(0.0, False)``, which is why every previously published figure is
    unaffected by this function existing.

    Cache traffic appears whether or not a run asked for it: the OpenAI models cache implicitly on
    the Converse path, so a ``--cache off`` run against one still bills cache reads and writes.

    When the model declares no cache rates, the traffic is billed at the model's *uncached input*
    rate instead of being refused. Cache reads are always cheaper than uncached input and writes are
    a small multiple of it, so that is an upper bound rather than a guess -- and a bound that renders
    beats a run that cannot be costed at all. The second return value says so, and the report prints
    it.

    Args:
        tokens: The run's ``summary.tokens`` block.

    Returns:
        The cost of cache reads plus cache writes in USD, and True when rates were unavailable and
        the figure is therefore an upper bound.
    """
    read = float(tokens.get("cache_read_total", 0) or 0)
    write = float(tokens.get("cache_write_total", 0) or 0)
    if not read and not write:
        return 0.0, False

    write_rate = PRICING.cache_write_1h_per_mtok if config.CACHE_TTL == "1h" else PRICING.cache_write_5m_per_mtok
    read_rate = PRICING.cache_read_per_mtok
    if read_rate is None or write_rate is None:
        fallback = PRICING.agent_input_per_mtok
        return _cost(read + write, fallback), True

    return _cost(read, read_rate) + _cost(write, write_rate), False


def _row(name: str, entry: dict[str, Any]) -> Row:
    summary = entry.get("summary", {})
    tokens = summary.get("tokens", {})
    timing = summary.get("timing", {})
    accuracy = summary.get("accuracy", {})
    counters = summary.get("plugin_counters", {})

    agent_input = float(tokens.get("usage_input_total", 0) or 0)
    agent_output = float(tokens.get("usage_output_total", 0) or 0)

    embedding = counters.get("embedding_cost", {})
    embed_tokens = float(embedding.get("input_tokens", 0) or 0)

    # Both rerank users are summed here and broken out in the notes below the table: the
    # offloader's relevance preview and, when enabled, the graph's selection refinement.
    offloader_units = int(counters.get("offloader", {}).get("search_units", 0) or 0)
    graph_units = int(counters.get("rerank_cost", {}).get("search_units", 0) or 0)
    search_units = offloader_units + graph_units

    agent_tokens = agent_input + agent_output
    # The disclosure catalog is summarized by the agent's own model, outside the agent's `usage`, so
    # it is billed here at the agent's rates rather than silently left out of the arm that spent it.
    catalog_usage = counters.get("disclosure", {}).get("summary_usage", {}) or {}
    summary_input = float(catalog_usage.get("inputTokens", 0) or 0)
    summary_output = float(catalog_usage.get("outputTokens", 0) or 0)
    aux_tokens = embed_tokens + summary_input + summary_output

    cache_cost, cache_bounded = _cache_cost(tokens)

    return Row(
        name=name,
        label=RUN_CONFIGS[name].label if name in RUN_CONFIGS else name,
        repeats=int(entry.get("repeats", 1)),
        agent_input=agent_input,
        agent_output=agent_output,
        agent_tokens=agent_tokens,
        embed_tokens=embed_tokens,
        aux_tokens=aux_tokens,
        total_tokens=agent_tokens + aux_tokens,
        token_spread_pct=float(tokens.get("usage_input_total_spread_pct", 0.0) or 0.0),
        weighted_accuracy=float(accuracy.get("weighted_accuracy", 0.0) or 0.0),
        materially_correct=float(accuracy.get("turns_materially_correct", 0) or 0),
        turns_scored=int(accuracy.get("turns_scored", 0) or 0),
        accuracy_spread_pct=float(accuracy.get("weighted_accuracy_spread_pct", 0.0) or 0.0),
        wall_seconds=float(summary.get("wall_seconds", 0.0) or 0.0),
        turn_seconds_mean=float(timing.get("turn_seconds_mean", 0.0) or 0.0),
        turn_seconds_spread_pct=float(timing.get("turn_seconds_mean_spread_pct", 0.0) or 0.0),
        agent_cost=_cost(agent_input + summary_input, PRICING.agent_input_per_mtok)
        + _cost(agent_output + summary_output, PRICING.agent_output_per_mtok)
        + cache_cost,
        cache_read=float(tokens.get("cache_read_total", 0) or 0),
        cache_write=float(tokens.get("cache_write_total", 0) or 0),
        cache_cost_is_bound=cache_bounded,
        embedding_cost=_cost(embed_tokens, PRICING.embedding_per_mtok),
        rerank_cost=search_units / 1_000.0 * PRICING.rerank_per_ksearchunit,
        embed_calls=int(embedding.get("calls", 0) or 0),
        search_units=search_units,
    )


def ordered_names(payload: dict[str, Any]) -> list[str]:
    """Return the payload's configuration names in reading order.

    Anything not in :data:`_ORDER` is appended after it, so a run of sweep variants still renders
    rather than being silently dropped.
    """
    names = [name for name in _ORDER if name in payload]
    return names + [name for name in payload if name not in _ORDER]


def summary_rows(payload: dict[str, Any]) -> list[Row]:
    """Return one :class:`Row` per configuration, in reading order.

    Public because the HTML renderer in :mod:`src.chart` needs the same figures: accuracy,
    latency and cost live in the run JSON and not in the curve CSV, so both readers derive them here
    rather than each computing a cost of its own.

    Args:
        payload: A run JSON, keyed by configuration name.

    Returns:
        The rows, baseline first when present.
    """
    return [_row(name, payload[name]) for name in ordered_names(payload)]


def delta_pct(value: float, reference: float) -> float:
    """Return ``value`` against ``reference`` as a signed percentage, zero when there is no base."""
    if not reference:
        return 0.0
    return (value - reference) / reference * 100.0


def _signed(value: float, digits: int = 1) -> str:
    """Format a delta with an explicit sign, so a saving is never mistaken for a total."""
    return f"{value:+,.{digits}f}"


def _resolved(delta: float, spread_pct: float) -> str:
    """Say whether a difference is larger than the noise it was measured against.

    A replayed run reports the peak-to-peak spread of each configuration. When the gap between
    two configurations is inside that spread, the run has not separated them, and printing the
    gap without saying so invites a decision the data does not support.
    """
    if not spread_pct:
        return "—"
    return "yes" if abs(delta) > spread_pct else "no"


def build_comparison(payload: dict[str, Any]) -> str:
    """Render the comparison for a run payload.

    Args:
        payload: A run JSON, keyed by configuration name.

    Returns:
        The comparison as Markdown.

    Raises:
        KeyError: If the payload carries no ``baseline`` entry. Every column but accuracy is a
            difference against it, so there is nothing to render without one.
    """
    if _BASELINE not in payload:
        raise KeyError(f"payload has no {_BASELINE!r} entry; the deltas have nothing to compare against")

    names = ordered_names(payload)
    rows = summary_rows(payload)
    base = next(row for row in rows if row.name == _BASELINE)

    meta = payload[_BASELINE].get("meta", {})
    repeats = base.repeats
    lines: list[str] = []

    lines.append("# Strategy comparison")
    lines.append("")
    lines.append(
        f"Agent `{meta.get('agent_model', '?')}`, region `{meta.get('region', '?')}`, "
        f"{repeats} replay(s) per configuration."
    )
    lines.append("")
    lines.append("## Tokens, accuracy, latency, cost")
    lines.append("")
    lines.append(
        "| Configuration | Total tokens | Δ vs baseline | Δ % | Beats noise | Accuracy | "
        "Materially correct | Turn (s) | Wall (s) | Cost (USD) | Δ cost % |"
    )
    lines.append("|---|---:|---:|---:|:--:|---:|:--:|---:|---:|---:|---:|")

    for row in rows:
        token_delta = row.total_tokens - base.total_tokens
        token_delta_pct = delta_pct(row.total_tokens, base.total_tokens)
        cost_delta_pct = delta_pct(row.cost, base.cost)
        is_base = row.name == _BASELINE
        lines.append(
            f"| {row.label} "
            f"| {row.total_tokens:,.0f} "
            f"| {'—' if is_base else _signed(token_delta, 0)} "
            f"| {'—' if is_base else _signed(token_delta_pct)} "
            f"| {'—' if is_base else _resolved(token_delta_pct, row.token_spread_pct)} "
            f"| {row.weighted_accuracy * 100:.1f}% "
            f"| {row.materially_correct:.1f}/{row.turns_scored} "
            f"| {row.turn_seconds_mean:.1f} "
            f"| {row.wall_seconds:.0f} "
            f"| {row.cost:.2f} "
            f"| {'—' if is_base else _signed(cost_delta_pct)} |"
        )

    lines.append("")
    lines.append(
        "**Total tokens** is the agent's own `usage` (input plus output) plus every auxiliary token "
        "the strategy spent: the graph's embedding calls. **Δ vs baseline** "
        "is that total against the baseline's. **Beats noise** compares the token delta with this "
        "configuration's own peak-to-peak spread across replays; `no` means the run did not separate "
        "it from the baseline, whatever the percentage says. `—` means a single replay, which has no "
        "spread to compare against and is not a measurement."
    )
    lines.append("")

    lines.append("## Where the cost came from")
    lines.append("")
    lines.append(
        "| Configuration | Agent in | Cache read | Cache write | Agent out | Agent $ | "
        "Embed calls | Embed tokens | Embed $ | Search units | Rerank $ | Total $ |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row.label} "
            f"| {row.agent_input:,.0f} | {row.cache_read:,.0f} | {row.cache_write:,.0f} "
            f"| {row.agent_output:,.0f} | {row.agent_cost:.2f} "
            f"| {row.embed_calls} | {row.embed_tokens:,.0f} | {row.embedding_cost:.4f} "
            f"| {row.search_units} | {row.rerank_cost:.4f} "
            f"| {row.cost:.2f}{' ≤' if row.cache_cost_is_bound else ''} |"
        )
    if any(row.cache_cost_is_bound for row in rows):
        lines.append("")
        lines.append(
            "`≤` marks an **upper bound**, not a measurement: this model publishes no cache rates, so "
            "its cache reads and writes are billed here at the uncached input rate. A cache read is "
            "always cheaper than uncached input, so the true cost is below the figure shown."
        )
    if any(row.cache_read or row.cache_write for row in rows):
        lines.append("")
        lines.append(
            "**Agent in counts only UNCACHED input.** Bedrock reports cached input in its own fields, "
            "so on a model that caches, `Agent in` is the delta and the prefix sits in `Cache read` / "
            "`Cache write`. Total billed input is the three summed — comparing one model's `Agent in` "
            "against another's is comparing a delta to a whole prompt."
        )
    lines.append("")
    lines.append(
        "Rates are declared per model id in `src.config.MODEL_PRICING`, not read from the Price List "
        "API — that API carries no usage type for this account's agent model, embedding model or "
        "rerank model, so a lookup would silently match something else. Edit the rates and re-render; "
        "the units do not change."
    )
    lines.append("")
    lines.append(
        f"`{AGENT_MODEL_ID}` at ${PRICING.agent_input_per_mtok:.2f}/${PRICING.agent_output_per_mtok:.2f} "
        f"per Mtok in/out, embedding ${PRICING.embedding_per_mtok:.2f} per Mtok, "
        f"rerank ${PRICING.rerank_per_ksearchunit:.2f} per 1k search units. "
        + (
            f"Prompt caching ON at TTL {config.CACHE_TTL}, billed at ${PRICING.cache_read_per_mtok:.2f} read / "
            f"${(PRICING.cache_write_1h_per_mtok if config.CACHE_TTL == '1h' else PRICING.cache_write_5m_per_mtok) or 0:.2f} "
            "write per Mtok."
            if config.CACHE_TTL
            else "Prompt caching OFF."
        )
    )
    lines.append("")
    lines.append(
        "One search unit is one query against up to 100 documents. Both rerank users are summed in the "
        "column: the offloader's relevance preview, and the graph's selection refinement when enabled."
    )
    lines.append("")

    # A provider 500 loses a turn, which lowers that configuration's token total and its
    # accuracy at the same time. Both columns then read as a strategy result. Naming the
    # configuration is the least that keeps the table honest.
    faulted = [
        (name, payload[name]["summary"]["errors"]) for name in names if payload[name]["summary"].get("errors")
    ]
    if faulted:
        lines.append("## Turns lost to provider errors")
        lines.append("")
        for name, errors in faulted:
            label = RUN_CONFIGS[name].label if name in RUN_CONFIGS else name
            lines.append(f"- **{label}**: {len(errors)} error(s). `{errors[0][:160]}`")
        lines.append("")
        lines.append(
            "A lost turn lowers both the token total and the accuracy of that configuration, so both of "
            "its columns are understated by an amount that has nothing to do with its strategy."
        )
        lines.append("")

    lines.append("## What the auxiliary bill does to the ranking")
    lines.append("")
    for row in rows:
        if row.name == _BASELINE:
            continue
        aux = row.embedding_cost + row.rerank_cost
        if not aux:
            continue
        agent_only_pct = delta_pct(row.agent_cost, base.agent_cost)
        all_in_pct = delta_pct(row.cost, base.cost)
        share = aux / row.cost * 100.0 if row.cost else 0.0
        lines.append(
            f"- **{row.label}**: on the agent's tokens alone it is {_signed(agent_only_pct)}% against "
            f"baseline; with its own calls billed it is {_signed(all_in_pct)}%. The auxiliary calls are "
            f"{share:.1f}% of its cost."
        )
    lines.append("")

    return "\n".join(lines)


# --- Per-turn curve ----------------------------------------------------------------


@dataclass
class TurnPoint:
    """One turn of one configuration, as the curve plots it."""

    index: int
    label: str
    calls: int
    input_tokens: int
    output_tokens: int
    schema_tokens: int
    message_tokens: int


def turn_series(entry: dict[str, Any]) -> list[TurnPoint]:
    """Reduce a configuration's call records to one point per turn.

    The unit is the turn and not the call because a turn is what a user pays for: a single question
    costs however many model calls the agent decided to make, and a strategy that trades one
    expensive call for two cheap ones has not saved anything. Summing per turn makes that visible;
    a per-call series hides it.

    ``schema_tokens`` and ``message_tokens`` come from the harness's own character estimate rather
    than from the provider, because ``usage`` reports one number for the whole request and cannot
    say how much of it was tool schema. They are the only way to tell "the history stopped growing"
    apart from "the schema stopped being sent", which are different strategies with the same total.

    Args:
        entry: One configuration's entry from a run payload.

    Returns:
        One :class:`TurnPoint` per turn that produced at least one model call, in turn order.
    """
    labels = {turn["index"]: turn["label"] for turn in entry.get("turns", [])}
    grouped: dict[int, list[dict[str, Any]]] = {}
    for call in entry.get("calls", []):
        grouped.setdefault(int(call["turn_index"]), []).append(call)

    points: list[TurnPoint] = []
    for index in sorted(grouped):
        calls = grouped[index]
        points.append(
            TurnPoint(
                index=index,
                label=labels.get(index, f"turn-{index}"),
                calls=len(calls),
                input_tokens=sum(call.get("usage_input_tokens") or 0 for call in calls),
                output_tokens=sum(call.get("usage_output_tokens") or 0 for call in calls),
                schema_tokens=sum(call.get("estimated_tool_spec_tokens") or 0 for call in calls),
                message_tokens=sum(call.get("estimated_message_tokens") or 0 for call in calls),
            )
        )
    return points


def _chart(series: dict[str, list[float]], glyphs: dict[str, str], *, height: int = 18) -> list[str]:
    """Plot every configuration's per-turn input tokens on one set of axes.

    ASCII on purpose: the chart has to survive being pasted into a terminal, a pull request and a
    chat window, and a PNG survives none of those. One glyph per configuration rather than one
    chart each, because the question the curve answers is where two configurations diverge — and
    that is only readable when they share a y-axis.

    Args:
        series: One value per turn, per configuration name.
        glyphs: One plotting character per configuration name.
        height: Rows of plotting area.

    Returns:
        The chart as lines, y-axis labelled in thousands of tokens.
    """
    width = max((len(values) for values in series.values()), default=0)
    peak = max((value for values in series.values() for value in values), default=0.0)
    if not width or not peak:
        return ["(no calls recorded)"]

    grid = [[" "] * width for _ in range(height)]
    for name, values in series.items():
        glyph = glyphs[name]
        for column, value in enumerate(values):
            # Rounded down so a row is only inked once the value has actually reached it, which
            # keeps the baseline row meaning "zero" rather than "almost nothing".
            row = height - 1 - int(value / peak * (height - 1))
            grid[row][column] = glyph

    axis_width = len(f"{peak / 1000:,.0f}")
    lines = []
    for row, cells in enumerate(grid):
        value = peak * (height - 1 - row) / (height - 1) / 1000
        label = f"{value:>{axis_width},.0f}k" if row % 3 == 0 or row == height - 1 else " " * (axis_width + 1)
        lines.append(f"{label} |{''.join(cells)}")
    lines.append(" " * (axis_width + 1) + "+" + "-" * width)
    ticks = [" "] * width
    for column in range(0, width, 10):
        for offset, character in enumerate(str(column)):
            if column + offset < width:
                ticks[column + offset] = character
    lines.append(" " * (axis_width + 1) + " " + "".join(ticks) + "   turn")
    return lines


def build_curve(payload: dict[str, Any]) -> str:
    """Render the per-turn token curve for a run payload.

    Args:
        payload: A run JSON, keyed by configuration name.

    Returns:
        The curve section as Markdown.
    """
    names = [name for name in _ORDER if name in payload]
    names += [name for name in payload if name not in _ORDER]
    series = {name: turn_series(payload[name]) for name in names}
    series = {name: points for name, points in series.items() if points}
    if not series:
        return "# Per-turn token curve\n\n(no calls recorded)\n"

    glyphs = dict(zip(series, "#*o+x=~^", strict=False))

    lines = ["# Per-turn token curve", ""]
    for name, glyph in glyphs.items():
        label = RUN_CONFIGS[name].label if name in RUN_CONFIGS else name
        lines.append(f"- `{glyph}` {label}")
    lines.append("")
    lines.append("## Input tokens per turn")
    lines.append("")
    lines.append("Every model call of the turn summed — what one question actually cost.")
    lines.append("")
    lines.append("```")
    lines.extend(_chart({name: [point.input_tokens for point in points] for name, points in series.items()}, glyphs))
    lines.append("```")
    lines.append("")
    lines.append("## Input tokens per model call")
    lines.append("")
    lines.append(
        "The same series divided by the calls of each turn. Worth plotting separately because the axis "
        "above is set by one turn that made eight calls, which flattens everything else against the "
        "floor; and because this is the quantity the strategies actually change."
    )
    lines.append("")
    lines.append("```")
    lines.extend(
        _chart(
            {
                name: [point.input_tokens / max(1, point.calls) for point in points]
                for name, points in series.items()
            },
            glyphs,
        )
    )
    lines.append("```")
    lines.append("")

    lines.append("## Shape of each curve")
    lines.append("")
    lines.append(
        "| Configuration | First turn | Peak turn | Plateau/turn | Plateau/call | Slope/call | "
        "Schema share | Calls/turn |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, points in series.items():
        label = RUN_CONFIGS[name].label if name in RUN_CONFIGS else name
        values = [point.input_tokens for point in points]
        plateau = _plateau(points)
        per_turn = [point.input_tokens for point in plateau]
        # Per call, not per turn: the agent chooses how many calls a turn makes, and a turn that
        # dropped from two calls to one reads as a large saving on a per-turn slope while nothing
        # about the cost of a call changed. Normalising removes that artefact.
        per_call = [point.input_tokens / max(1, point.calls) for point in plateau]
        schema = sum(point.schema_tokens for point in points)
        estimated = schema + sum(point.message_tokens for point in points)
        share = schema / estimated * 100.0 if estimated else 0.0
        calls = sum(point.calls for point in points) / len(points)
        lines.append(
            f"| {label} | {values[0]:,} | {max(values):,} | {median(per_turn):,.0f} "
            f"| {median(per_call):,.0f} | {_signed(_slope(per_call), 0)} | {share:.1f}% | {calls:.1f} |"
        )
    lines.append("")
    lines.append(
        "**First turn** is the floor: one question against an empty history, so it is almost entirely "
        "tool schema and system prompt. **Peak** is the most expensive single turn, and it is driven by "
        "how many model calls that turn made, not by the history alone."
    )
    lines.append("")
    lines.append(
        "The **plateau** columns are measured over the filler turns only, because the script is "
        "deliberately not homogeneous: the scored turns come first and do real tool work, the filler turns "
        "exist only to lengthen the history, and the return turns close the script. Over the filler run, "
        "history is the only thing changing, so a rise there means what it says."
    )
    lines.append("")
    lines.append(
        "**Slope/call** is a least-squares fit of input tokens *per model call*, not per turn, and the "
        "normalisation is not cosmetic. The agent decides how many calls a turn makes; in this script the "
        "first filler turns take two and the rest take one, so a per-turn slope reports that step as a "
        "large *saving* while nothing about the cost of a call changed. Per call, the slope is the growth "
        "rate of the history and nothing else."
    )
    lines.append("")
    lines.append(
        "Read the floor and the slope as separate results. A strategy can lower the floor without "
        "flattening the growth, or flatten the growth without touching the floor, and the token total "
        "alone cannot tell those apart — which is the whole reason this curve exists."
    )
    lines.append("")
    return "\n".join(lines)


def write_curve_csv(payload: dict[str, Any], out_path: Path) -> Path:
    """Write the per-turn series as CSV, for plotting outside this harness."""
    names = [name for name in _ORDER if name in payload]
    names += [name for name in payload if name not in _ORDER]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["config", "turn", "label", "calls", "input_tokens", "output_tokens", "schema_tokens", "message_tokens"]
        )
        for name in names:
            for point in turn_series(payload[name]):
                writer.writerow(
                    [
                        name,
                        point.index,
                        point.label,
                        point.calls,
                        point.input_tokens,
                        point.output_tokens,
                        point.schema_tokens,
                        point.message_tokens,
                    ]
                )
    return out_path


def write_comparison(payload: dict[str, Any], out_path: Path) -> Path:
    """Write the comparison to ``out_path`` and return it."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_comparison(payload), encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="src.compare",
        description="Render the strategy comparison table from a run JSON.",
    )
    parser.add_argument("run_json", help="path to results/run-<tag>.json")
    parser.add_argument("--out", metavar="MD", help="write here instead of results/comparison-<tag>.md")
    parser.add_argument(
        "--curve",
        action="store_true",
        help="also render the per-turn token curve and write the series as CSV",
    )
    args = parser.parse_args()

    source = Path(args.run_json)
    payload = json.loads(source.read_text(encoding="utf-8"))
    tag = source.stem.removeprefix("run-")
    out_path = Path(args.out) if args.out else RESULTS_DIR / f"comparison-{tag}.md"

    document = build_comparison(payload)
    if args.curve:
        document = f"{document}\n{build_curve(payload)}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(document, encoding="utf-8")

    print(document)
    print(f"\nwritten: {out_path}")
    if args.curve:
        print(f"series:  {write_curve_csv(payload, RESULTS_DIR / f'curve-{tag}.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
