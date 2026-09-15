"""Render the per-turn token curve as a self-contained HTML page.

Reads the CSV that ``validation.compare --curve`` writes and emits one file with four charts. The
split is deliberate: ``compare.py`` answers *what did each strategy cost* in Markdown that pastes
into a pull request, and this answers *what shape did the cost have* in something you can hand to
someone who will not read a table.

**Inline SVG, no library, no CDN.** Three reasons, in order of weight. The harness has four
dependencies and a charting library would be the fifth, for output that is not part of the
measurement. A page that fetches a script from a CDN is a page that renders blank on a plane or
behind a proxy, which is exactly where a report gets read. And an SVG generated here is
diffable — a chart that changed is a chart whose text changed.

Interactivity is what SVG gives for free: ``<title>`` on each point is a native browser tooltip, and
the legend toggles series with one class swap. No hover engine, no layout pass, no build step.

Usage:
    python -m validation.chart validation/results/curve-comparison60.csv
    python -m validation.chart <csv> --out validation/results/curve.html
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path

from .compare import Row, delta_pct, summary_rows
from .config import RESULTS_DIR, RUN_CONFIGS

_ORDER = ("baseline", "disclosure", "relevance", "graph", "all", "graph-all")
"""Reading order: the reference first, then one strategy at a time, then the two stacks."""

_COLORS = {
    "baseline": "#57606a",
    "disclosure": "#0969da",
    "relevance": "#8250df",
    "graph": "#cf222e",
    "all": "#bf3989",
    "graph-all": "#116329",
}
"""One colour per configuration.

The baseline is deliberately the only grey: it is the reference, not a competitor, and colouring it
like the others invites reading the chart as seven rivals instead of six against one.

The two stacks got hues far from each other rather than hues near their members. Pairing them by
family read well as a legend and badly as a chart — two greens a few steps apart are the two lines a
reader most needs to tell apart, because they are the two candidates the comparison is between.
"""

_FALLBACK_COLOR = "#6e7781"

_WIDTH = 1000
_HEIGHT = 400
_MARGIN = {"top": 16, "right": 24, "bottom": 44, "left": 78}


@dataclass(frozen=True)
class Point:
    """One turn of one configuration, as the CSV carries it."""

    turn: int
    label: str
    calls: int
    input_tokens: int
    output_tokens: int
    schema_tokens: int
    message_tokens: int

    @property
    def input_per_call(self) -> float:
        return self.input_tokens / max(1, self.calls)

    @property
    def message_per_call(self) -> float:
        return self.message_tokens / max(1, self.calls)


def load(csv_path: Path) -> dict[str, list[Point]]:
    """Read the curve CSV, grouped by configuration and ordered by turn.

    Args:
        csv_path: Path to a ``curve-<tag>.csv`` written by ``validation.compare --curve``.

    Returns:
        Points per configuration name, in reading order, each list sorted by turn.

    Raises:
        ValueError: If the file carries no rows, so a blank page is never written silently.
    """
    grouped: dict[str, list[Point]] = {}
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(row["config"], []).append(
                Point(
                    turn=int(row["turn"]),
                    label=row["label"],
                    calls=int(row["calls"]),
                    input_tokens=int(row["input_tokens"]),
                    output_tokens=int(row["output_tokens"]),
                    schema_tokens=int(row["schema_tokens"]),
                    message_tokens=int(row["message_tokens"]),
                )
            )

    if not grouped:
        raise ValueError(f"{csv_path} carries no rows")

    names = [name for name in _ORDER if name in grouped]
    names += sorted(name for name in grouped if name not in _ORDER)
    return {name: sorted(grouped[name], key=lambda point: point.turn) for name in names}


def _label(name: str) -> str:
    return RUN_CONFIGS[name].label if name in RUN_CONFIGS else name


def _nice_ceiling(value: float, *, ticks: int = 4) -> tuple[float, float]:
    """Return an axis ceiling and tick step that land on readable round numbers.

    A ceiling of ``1,036,154`` labels the axis with the peak instead of with a number a reader can do
    arithmetic against. The ladder of step sizes is finer than the usual 1/2/5 because a coarse one
    rounds that peak up to ``2M`` and spends half the plot on empty space.

    Args:
        value: The largest value the axis must contain.
        ticks: Gridlines to aim for, excluding zero.

    Returns:
        The ceiling and the step between gridlines.
    """
    if value <= 0:
        return 1.0, 1.0
    raw_step = value / ticks
    magnitude = 10 ** math.floor(math.log10(raw_step))
    step = magnitude * 10
    for multiple in (1, 1.2, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 10):
        if magnitude * multiple >= raw_step:
            step = magnitude * multiple
            break
    return step * math.ceil(value / step), step


def _tokens(value: float) -> str:
    """Format a token count for an axis label."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:,.0f}k"
    return f"{value:,.0f}"


def _chart(
    chart_id: str,
    series: dict[str, list[tuple[int, float]]],
    *,
    y_title: str,
) -> str:
    """Render one line chart as inline SVG.

    Args:
        chart_id: Unique id, used to scope the legend's toggles to this chart.
        series: ``(turn, value)`` pairs per configuration name.
        y_title: Axis caption.

    Returns:
        The chart as an HTML fragment: the SVG plus its own legend.
    """
    plot_width = _WIDTH - _MARGIN["left"] - _MARGIN["right"]
    plot_height = _HEIGHT - _MARGIN["top"] - _MARGIN["bottom"]

    turns = [turn for points in series.values() for turn, _ in points]
    values = [value for points in series.values() for _, value in points]
    if not turns or not values:
        return "<p>(no data)</p>"

    max_turn = max(turns)
    ceiling, step = _nice_ceiling(max(values))

    def x_of(turn: int) -> float:
        return _MARGIN["left"] + (turn / max_turn * plot_width if max_turn else 0.0)

    def y_of(value: float) -> float:
        return _MARGIN["top"] + plot_height - (value / ceiling * plot_height)

    parts: list[str] = [
        f'<svg class="chart" id="{chart_id}" viewBox="0 0 {_WIDTH} {_HEIGHT}" '
        f'role="img" aria-label="{html.escape(y_title)} per turn">'
    ]

    # Gridlines and y labels first, so every series draws over them.
    gridline = 0.0
    while gridline <= ceiling + 1e-9:
        y_position = y_of(gridline)
        parts.append(
            f'<line class="grid" x1="{_MARGIN["left"]}" y1="{y_position:.1f}" '
            f'x2="{_MARGIN["left"] + plot_width}" y2="{y_position:.1f}"/>'
        )
        parts.append(
            f'<text class="ytick" x="{_MARGIN["left"] - 10}" y="{y_position + 4:.1f}">{_tokens(gridline)}</text>'
        )
        gridline += step

    tick_step = 10 if max_turn > 25 else 5
    for turn in range(0, max_turn + 1, tick_step):
        parts.append(
            f'<text class="xtick" x="{x_of(turn):.1f}" y="{_MARGIN["top"] + plot_height + 22}">{turn}</text>'
        )
    parts.append(
        f'<text class="axis-title" x="{_MARGIN["left"] + plot_width / 2:.1f}" '
        f'y="{_HEIGHT - 6}">turn</text>'
    )
    parts.append(
        f'<text class="axis-title" transform="translate(16,{_MARGIN["top"] + plot_height / 2:.1f}) '
        f'rotate(-90)">{html.escape(y_title)}</text>'
    )

    for name, points in series.items():
        color = _COLORS.get(name, _FALLBACK_COLOR)
        path = " ".join(
            f"{'M' if index == 0 else 'L'}{x_of(turn):.1f},{y_of(value):.1f}"
            for index, (turn, value) in enumerate(points)
        )
        parts.append(f'<g class="series" data-config="{name}" stroke="{color}" fill="{color}">')
        parts.append(f'<path class="line" d="{path}"/>')
        for turn, value in points:
            # A native <title> is the whole tooltip implementation. The dot is small enough not to
            # clutter a 60-point series and large enough to be a hover target.
            parts.append(
                f'<circle class="dot" cx="{x_of(turn):.1f}" cy="{y_of(value):.1f}" r="2.5">'
                f"<title>{html.escape(_label(name))} — turn {turn}: {value:,.0f}</title></circle>"
            )
        parts.append("</g>")

    parts.append("</svg>")

    legend = ['<div class="legend">']
    for name in series:
        color = _COLORS.get(name, _FALLBACK_COLOR)
        legend.append(
            f'<button class="key" data-chart="{chart_id}" data-config="{name}" '
            f'style="--key:{color}" aria-pressed="true">{html.escape(_label(name))}</button>'
        )
    legend.append("</div>")

    return "\n".join(parts) + "\n" + "\n".join(legend)


_CSS = """
:root { --ink:#1f2328; --muted:#656d76; --rule:#d8dee4; --bg:#ffffff; --panel:#f6f8fa; }
* { box-sizing:border-box; }
body { margin:0; padding:40px 24px 72px; background:var(--bg); color:var(--ink);
  font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; }
main { max-width:1060px; margin:0 auto; }
h1 { font-size:26px; margin:0 0 6px; letter-spacing:-.01em; }
h2 { font-size:19px; margin:0 0 4px; letter-spacing:-.01em; }
p.lede { color:var(--muted); margin:0 0 36px; }
p.note { color:var(--muted); margin:0 0 14px; font-size:13.5px; }
section { margin:0 0 44px; padding:20px 20px 12px; border:1px solid var(--rule); border-radius:8px; }
.chart { width:100%; height:auto; display:block; overflow:visible; }
.grid { stroke:var(--rule); stroke-width:1; }
.ytick { fill:var(--muted); font-size:11px; text-anchor:end; }
.xtick { fill:var(--muted); font-size:11px; text-anchor:middle; }
.axis-title { fill:var(--muted); font-size:11.5px; text-anchor:middle; }
.line { fill:none; stroke-width:1.9; stroke-linejoin:round; }
.dot { stroke:none; opacity:0; }
.dot:hover { opacity:1; r:4; }
.series.off { display:none; }
.series.faded { opacity:.16; }
.legend { display:flex; flex-wrap:wrap; gap:6px; margin:14px 0 4px; }
.key { appearance:none; cursor:pointer; font:inherit; font-size:12.5px; line-height:1;
  padding:6px 10px 6px 24px; border:1px solid var(--rule); border-radius:99px;
  background:var(--panel); color:var(--ink); position:relative; }
.key::before { content:""; position:absolute; left:9px; top:50%; transform:translateY(-50%);
  width:9px; height:9px; border-radius:50%; background:var(--key); }
.key[aria-pressed="false"] { color:var(--muted); opacity:.5; }
.key[aria-pressed="false"]::before { background:var(--rule); }
table { border-collapse:collapse; width:100%; font-size:13.5px; margin:4px 0 8px; }
th,td { border-bottom:1px solid var(--rule); padding:7px 10px; text-align:right; }
th:first-child,td:first-child { text-align:left; }
thead th { color:var(--muted); font-weight:600; white-space:nowrap; }
tbody tr:hover { background:var(--panel); }
tbody tr.ref td { color:var(--muted); }
tbody td { font-variant-numeric:tabular-nums; }
footer { color:var(--muted); font-size:12.5px; border-top:1px solid var(--rule); padding-top:16px; }
@media (prefers-color-scheme: dark) {
  :root { --ink:#e6edf3; --muted:#8b949e; --rule:#30363d; --bg:#0d1117; --panel:#161b22; }
}
"""

_JS = """
document.querySelectorAll('.key').forEach(function (key) {
  var chart = document.getElementById(key.dataset.chart);
  var group = chart.querySelector('[data-config="' + key.dataset.config + '"]');
  key.addEventListener('click', function () {
    var on = key.getAttribute('aria-pressed') === 'true';
    key.setAttribute('aria-pressed', String(!on));
    group.classList.toggle('off', on);
  });
  // Hovering a key highlights its series by fading the others, which is how you read one line
  // out of seven without turning the other six off one by one.
  key.addEventListener('mouseenter', function () {
    chart.querySelectorAll('.series').forEach(function (series) {
      if (series !== group) { series.classList.add('faded'); }
    });
  });
  key.addEventListener('mouseleave', function () {
    chart.querySelectorAll('.series').forEach(function (series) { series.classList.remove('faded'); });
  });
});
"""


def _shape_table(data: dict[str, list[Point]]) -> str:
    """Render the summary table: floor, peak, plateau and growth rate per configuration."""
    rows = []
    for name, points in data.items():
        plateau = [point for point in points if "filler" in point.label] or points[-20:]
        per_call = [point.input_per_call for point in plateau]
        peak = max(points, key=lambda point: point.input_tokens)
        total = sum(point.input_tokens + point.output_tokens for point in points)
        schema = sum(point.schema_tokens for point in points)
        estimated = schema + sum(point.message_tokens for point in points)
        median_per_call = sorted(per_call)[len(per_call) // 2]
        rows.append(
            "<tr>"
            f"<td>{html.escape(_label(name))}</td>"
            f"<td>{points[0].input_tokens:,}</td>"
            f"<td>{peak.input_tokens:,} <span style='color:var(--muted)'>({peak.calls} calls)</span></td>"
            f"<td>{median_per_call:,.0f}</td>"
            f"<td>{_slope(per_call):+,.0f}</td>"
            f"<td>{schema / estimated * 100 if estimated else 0:.1f}%</td>"
            f"<td>{total:,}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Configuration</th><th>First turn</th><th>Peak turn</th>"
        "<th>Plateau per call</th><th>Slope per call</th><th>Schema</th><th>Total</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _headline_table(rows: list[Row]) -> str:
    """Render the result table: tokens, accuracy, latency and cost against the baseline.

    First on the page because it is the answer, and the charts are the explanation. Its figures come
    from the run JSON rather than the curve CSV — accuracy, turn latency and the auxiliary bill are
    not in the series — and from :func:`validation.compare.summary_rows`, so the HTML and the
    Markdown report can never disagree about a cost.

    Args:
        rows: Ordered rows for the run, baseline included.

    Returns:
        The table as an HTML fragment.
    """
    base = next((row for row in rows if row.name == "baseline"), None)

    body = []
    for row in rows:
        is_base = base is not None and row.name == base.name
        token_delta = "—" if base is None or is_base else f"{delta_pct(row.total_tokens, base.total_tokens):+,.1f}%"
        cost_delta = "—" if base is None or is_base else f"{delta_pct(row.cost, base.cost):+,.1f}%"
        # The baseline is the reference, so it is the one row that must not read as a competitor.
        classes = ' class="ref"' if is_base else ""
        body.append(
            f"<tr{classes}>"
            f"<td>{html.escape(row.label)}</td>"
            f"<td>{row.total_tokens:,.0f}</td>"
            f"<td>{token_delta}</td>"
            f"<td>{row.weighted_accuracy * 100:.1f}%</td>"
            f"<td>{row.materially_correct:.0f}/{row.turns_scored}</td>"
            f"<td>{row.turn_seconds_mean:.1f}s</td>"
            f"<td>${row.cost:,.2f}</td>"
            f"<td>{cost_delta}</td>"
            "</tr>"
        )

    return (
        "<table class='headline'><thead><tr>"
        "<th>Configuration</th><th>Total tokens</th><th>Δ tokens</th><th>Accuracy</th>"
        "<th>Correct</th><th>Turn</th><th>Cost</th><th>Δ cost</th>"
        "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"
    )


def _slope(values: list[float]) -> float:
    """Least-squares slope against ordinal position, in units per turn."""
    count = len(values)
    if count < 2:
        return 0.0
    mean_x = (count - 1) / 2
    mean_y = sum(values) / count
    variance = sum((index - mean_x) ** 2 for index in range(count))
    if not variance:
        return 0.0
    return sum((index - mean_x) * (value - mean_y) for index, value in enumerate(values)) / variance


def build_html(
    data: dict[str, list[Point]],
    *,
    source: str,
    rows: list[Row] | None = None,
) -> str:
    """Render the whole page.

    Args:
        data: Points per configuration.
        source: Name of the CSV the page was built from, shown in the footer.
        rows: Summary rows from the run JSON. When absent the page renders the charts alone, because
            accuracy, latency and cost are not derivable from the curve.

    Returns:
        A self-contained HTML document.
    """
    per_turn = {name: [(point.turn, float(point.input_tokens)) for point in points] for name, points in data.items()}
    per_call = {name: [(point.turn, point.input_per_call) for point in points] for name, points in data.items()}
    history = {name: [(point.turn, point.message_per_call) for point in points] for name, points in data.items()}

    cumulative: dict[str, list[tuple[int, float]]] = {}
    for name, points in data.items():
        running = 0.0
        series = []
        for point in points:
            running += point.input_tokens + point.output_tokens
            series.append((point.turn, running))
        cumulative[name] = series

    peak_turn = max(
        (point for points in data.values() for point in points),
        key=lambda point: point.input_tokens,
    )
    turns = max(len(points) for points in data.values())

    sections = [
        (
            "Input tokens per turn",
            "Every model call of the turn summed — what one question actually cost. The axis is set by a "
            f"single turn: turn {peak_turn.turn} of the baseline (<code>{html.escape(peak_turn.label)}</code>) "
            f"made {peak_turn.calls} model calls and cost {peak_turn.input_tokens:,} tokens. It flattens "
            "everything else against the floor, which is why the next chart exists.",
            _chart("c-turn", per_turn, y_title="input tokens"),
        ),
        (
            "Input tokens per model call",
            "The same series divided by the calls of each turn. This is the quantity the strategies actually "
            "change: the agent decides how many calls a turn makes, so a turn that dropped from two calls to "
            "one looks like a large per-turn saving while nothing about the cost of a call changed.",
            _chart("c-call", per_call, y_title="tokens per call"),
        ),
        (
            "History alone, per call",
            "The harness's own estimate, because the provider's <code>usage</code> returns a single number and "
            "cannot say how much of it was tool schema. It is the only way to tell <em>the history stopped "
            "growing</em> apart from <em>the schema stopped being sent</em> — two different strategies with "
            "the same total.",
            _chart("c-history", history, y_title="history tokens per call"),
        ),
        (
            "Cumulative session cost",
            "Input plus output, summed turn by turn. This is the curve that matches the bill: the slope is the "
            "marginal cost of continuing the conversation, and the gap between two lines is what the choice of "
            "strategy is worth by the end of a long session.",
            _chart("c-total", cumulative, y_title="cumulative tokens"),
        ),
    ]

    body = "\n".join(
        f"<section><h2>{title}</h2><p class='note'>{note}</p>{chart}</section>" for title, note, chart in sections
    )

    headline = ""
    if rows:
        headline = (
            "<section><h2>The result</h2>"
            "<p class='note'><strong>Total tokens</strong> is the agent's own <code>usage</code>, input plus "
            "output, plus every auxiliary token the strategy spent on its own account: the graph's embedding "
            "calls. <strong>Cost</strong> applies the rates declared in "
            "<code>validation.config.PRICING</code> to measured units, embedding and rerank included — without "
            "that, the three strategies that buy their saving with a second model call would rank better than "
            "they are.</p>"
            f"{_headline_table(rows)}"
            "<p class='note'>One replay per configuration. The three large differences — −54%, −81.9% and the "
            "accuracy tie at 95.8% — support a decision; nothing under ~20% does, because that is inside the "
            "noise this harness has already measured on replayed runs.</p></section>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Per-turn token curve</title>
<style>{_CSS}</style>
</head>
<body>
<main>
<h1>Context strategies: cost, accuracy and the per-turn curve</h1>
<p class="lede">{len(data)} configurations, {turns} turns each, sequential, live Bedrock,
<strong>1 replay</strong>. Click a legend key to hide a series; hover it to highlight.</p>
{headline}
{body}
<section>
<h2>Shape of each curve</h2>
<p class="note">The <strong>plateau</strong> and the <strong>slope</strong> are measured over the filler
turns only, where the history is the only thing changing. Fitting the whole series measures the step
between the script's two regimes and reports a <em>falling</em> cost, which is an artefact of the script
rather than a property of any strategy.</p>
{_shape_table(data)}
</section>
<footer>
Generated from <code>{html.escape(source)}</code> by <code>validation.chart</code>. Inline SVG, no
library and no CDN — the file opens offline. One replay per configuration: differences under ~20% are
inside the harness's noise.
</footer>
</main>
<script>{_JS}</script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="validation.chart",
        description="Render the per-turn token curve CSV as a self-contained HTML page.",
    )
    parser.add_argument("csv", help="path to results/curve-<tag>.csv")
    parser.add_argument("--out", metavar="HTML", help="write here instead of results/curve-<tag>.html")
    parser.add_argument(
        "--run",
        metavar="JSON",
        help=(
            "run JSON for the result table at the top of the page. Defaults to the run-<tag>.json "
            "beside the CSV, since --curve names the two from the same tag."
        ),
    )
    args = parser.parse_args()

    source = Path(args.csv)
    data = load(source)

    run_path = Path(args.run) if args.run else source.with_name(f"run-{source.stem.removeprefix('curve-')}.json")
    rows: list[Row] | None = None
    if run_path.is_file():
        rows = summary_rows(json.loads(run_path.read_text(encoding="utf-8")))
    else:
        # Stated rather than silent: a page missing its result table looks like a rendering bug, and
        # the cause is a missing file the reader can point at.
        print(f"note:    {run_path.name} not found, rendering charts without the result table")

    out_path = Path(args.out) if args.out else RESULTS_DIR / f"{source.stem}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_html(data, source=source.name, rows=rows), encoding="utf-8")

    print(f"configs: {', '.join(data)}")
    print(f"turns:   {max(len(points) for points in data.values())}")
    if rows:
        print(f"table:   {run_path.name}")
    print(f"written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
