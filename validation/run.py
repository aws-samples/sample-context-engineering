"""Entry point: run the comparison and emit the metrics report.

Usage:
    python -m validation.run                          # all five configurations, full script
    python -m validation.run --configs baseline all    # just two
    python -m validation.run --turns 3                 # smoke run, first three turns
    python -m validation.run --smoke                   # cheapest run that still exercises all three
    python -m validation.run --report-only results/run-20260829-120000.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from . import accuracy, chart, compare, corpus, metrics, report, runner, scenario, tools
from .config import (
    ACCOUNT_ID,
    AGENT_MODEL_ID,
    CONFIGURATIONS,
    DEFAULT_CONFIGURATIONS,
    INVOCATION_LOG_GROUP,
    REGION,
    RERANK_MODEL_ID,
    RESULTS_DIR,
    TARGET_SCHEMA_TOKENS,
    WEB_TARGETS,
)


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    # The harness's own progress lines are the signal; the SDK's INFO chatter is not.
    for noisy in ("botocore", "boto3", "urllib3", "httpx", "strands.event_loop", "strands.tools"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if verbose:
        # The plugins report their own work at INFO/DEBUG. Turning these on is how you see
        # a projection or a retrieval happen, rather than inferring it from totals.
        for name in (
            "strands.vended_plugins.context_graph.plugin",
            "strands.vended_plugins.progressive_tool_disclosure.plugin",
            "strands.vended_plugins.context_offloader.plugin",
        ):
            logging.getLogger(name).setLevel(logging.DEBUG)


def _rescore(payload: dict) -> None:
    """Recompute accuracy over a stored run using the current expectations.

    Handles both single-replay entries and aggregated ones, re-deriving the aggregate from
    the per-replay scores so the mean and its spread stay consistent with the detail.
    """
    for name, entry in payload.items():
        per_repeat = entry.get("per_repeat")
        if per_repeat:
            rescored = []
            for replay in per_repeat:
                replay["summary"]["accuracy"] = accuracy.score_run(replay["turns"])
                rescored.append(replay["summary"]["accuracy"])
            entry["summary"]["accuracy"] = _merge_accuracy(rescored)
        else:
            entry["summary"]["accuracy"] = accuracy.score_run(entry["turns"])

        acc = entry["summary"]["accuracy"]
        print(
            f"rescored {name:11s} {acc['weighted_accuracy'] * 100:5.1f}% weighted, "
            f"{acc['turns_materially_correct']}/{acc['turns_scored']} materially correct"
        )


def _merge_accuracy(items: list[dict]) -> dict:
    """Average re-scored accuracy across replays, reusing the metrics aggregation rules."""
    if len(items) == 1:
        return items[0]
    return {
        "weighted_accuracy": round(sum(i["weighted_accuracy"] for i in items) / len(items), 4),
        "weighted_accuracy_spread_pct": round(
            (max(i["weighted_accuracy"] for i in items) - min(i["weighted_accuracy"] for i in items))
            / (sum(i["weighted_accuracy"] for i in items) / len(items) or 1)
            * 100.0,
            1,
        ),
        "material_correctness": round(sum(i["material_correctness"] for i in items) / len(items), 4),
        "turns_materially_correct": round(sum(i["turns_materially_correct"] for i in items) / len(items), 2),
        "turns_scored": items[0]["turns_scored"],
        "critical_failures_total": round(sum(i["critical_failures_total"] for i in items) / len(items), 2),
        "per_turn": metrics._mean_per_turn_accuracy(items),
    }


async def _prewarm() -> None:
    """Populate every cache before the first timed call.

    Without this, whichever configuration runs first pays the download cost and looks
    slower for a reason that has nothing to do with its strategy.
    """
    from .web import prewarm

    sizes = corpus.warm()
    print(f"corpus: {len(sizes)} AWS doc pages, {sum(sizes.values()):,} chars cached")
    web = await prewarm(list(WEB_TARGETS.values()))
    print(f"web:    {len(web)} pages rendered, {sum(web.values()):,} chars cached")


def _preflight() -> bool:
    """Verify credentials and model access before spending time on a run."""
    try:
        session = runner.boto_session()
        identity = session.client("sts").get_caller_identity()
    except Exception as error:  # noqa: BLE001
        print(f"credentials unavailable: {error}", file=sys.stderr)
        print(
            "configure credentials the usual way — an SSO session, exported keys, or a named "
            "profile in VALIDATION_AWS_PROFILE.",
            file=sys.stderr,
        )
        return False

    # Only checked when the operator asked for the guarantee. An unset ACCOUNT_ID means "whichever
    # account these credentials resolve to", which is what makes the harness runnable by anyone.
    if ACCOUNT_ID and identity["Account"] != ACCOUNT_ID:
        print(f"wrong account: expected {ACCOUNT_ID}, got {identity['Account']}", file=sys.stderr)
        return False

    print(f"account: {identity['Account']} ({REGION})")
    print(f"agent:   {AGENT_MODEL_ID}")
    print(f"rerank:  {RERANK_MODEL_ID}")
    print(f"tag:     {metrics.RUN_TAG}")

    # Invocation logging is the only independent check on the token numbers this harness
    # reports. It is account-wide state, not something the run configures, so a silent
    # failure here means the run cannot be verified afterwards and nothing else would say so.
    try:
        logging_config = (
            session.client("bedrock").get_model_invocation_logging_configuration().get("loggingConfig") or {}
        )
    except Exception as error:  # noqa: BLE001 - a missing permission must not block a run
        logging_config = {}
        print(f"warning: could not read invocation logging configuration: {error}", file=sys.stderr)

    log_group = (logging_config.get("cloudWatchConfig") or {}).get("logGroupName")
    if log_group == INVOCATION_LOG_GROUP:
        print(f"logs:    {log_group} (verify with validation/verify_logs.py)")
    else:
        print(
            f"warning: bedrock invocation logging does not target {INVOCATION_LOG_GROUP} "
            f"(currently {log_group or 'disabled'}). The run will work, but its token numbers "
            "will not be independently verifiable.",
            file=sys.stderr,
        )

    # The two calibrations that decide whether this harness measures anything. Both were
    # wrong in earlier versions and both silently understated a strategy: too little schema
    # made Progressive Tool Disclosure look marginal, and topic lines too short left the
    # graph with nothing worth compacting.
    specs = sum(len(json.dumps(t.tool_spec, ensure_ascii=False)) for t in tools.all_tools()) // 4
    mass = scenario.line_mass()
    print(f"schema:  ~{specs:,} tokens/call across {len(tools.all_tools())} tools (target {TARGET_SCHEMA_TOKENS:,})")
    print(f"lines:   {mass}")

    short = {phase: count for phase, count in mass.items() if phase != "return" and count < 5}
    if short:
        print(
            f"warning: topic lines shorter than 5 turns: {short}. "
            "The graph has little stale mass to compact, so its saving will read as ~0.",
            file=sys.stderr,
        )
    if specs < TARGET_SCHEMA_TOKENS * 0.7:
        print(
            f"warning: schema budget is {specs:,} tokens, well under the {TARGET_SCHEMA_TOKENS:,} "
            "measured on the real session. Disclosure's saving will be understated.",
            file=sys.stderr,
        )
    return True


def _write_html(payload: dict, *, tag: str | None) -> Path | None:
    """Render the per-turn curve CSV and the self-contained HTML page for a finished run.

    Mirrors report.sh (compare --curve, then chart) but in-process, so every run produces the
    HTML deliverable automatically alongside the JSON and the Markdown report. Best-effort: a
    rendering failure must never discard a run that already wrote its JSON and report, so it is
    caught and reported rather than raised.
    """
    stem = tag or "latest"
    try:
        curve_csv = RESULTS_DIR / f"curve-{stem}.csv"
        compare.write_curve_csv(payload, curve_csv)
        data = chart.load(curve_csv)
        rows = chart.summary_rows(payload)
        html_path = RESULTS_DIR / f"curve-{stem}.html"
        html_path.write_text(
            chart.build_html(data, source=curve_csv.name, rows=rows), encoding="utf-8"
        )
        return html_path
    except Exception as error:  # noqa: BLE001 — HTML is a deliverable, not a gate on the run
        print(f"note: HTML report not generated ({error})", file=sys.stderr)
        return None


async def _main_async(args: argparse.Namespace) -> int:
    if args.report_only:
        payload = json.loads(Path(args.report_only).read_text(encoding="utf-8"))
        if args.rescore:
            # Response text is persisted precisely so a correction to an expectation costs a
            # re-score rather than a re-run. Expectations are the part of this harness most
            # likely to be wrong, and re-running to fix one would make fixing it expensive
            # enough to discourage it.
            _rescore(payload)
        json_path, report_path = report.write_outputs(payload, tag=args.tag)
        html_path = _write_html(payload, tag=args.tag)
        print(f"raw:    {json_path}")
        print(f"report: {report_path}")
        if html_path:
            print(f"html:   {html_path}")
        return 0

    # Named before any client is built: the tag is injected by botocore hooks registered at
    # model construction, which happens inside the run.
    metrics.set_run_tag(args.tag)

    if not _preflight():
        return 2

    await _prewarm()

    configs = args.configs or list(DEFAULT_CONFIGURATIONS)
    unknown = [name for name in configs if name not in CONFIGURATIONS]
    if unknown:
        print(f"unknown configurations: {unknown}", file=sys.stderr)
        return 2

    turn_limit = args.turns
    phases = tuple(args.phases) if args.phases else None
    if args.smoke:
        # Line A plus the return: the cheapest script that still establishes a subject,
        # leaves it, and comes back. It cannot exercise the strategies properly — there is no
        # stale mass for the graph to compact — so it verifies wiring, not effect.
        turn_limit = None
        phases = ("line-a", "return")
        configs = ["baseline", "graph-all"] if not args.configs else configs
        print("smoke mode: line-a + return only. Verifies wiring, not the strategies' effect.")

    mode = "sequential" if args.sequential else "parallel"
    print(f"\nrunning {len(configs)} configuration(s) {mode}: {', '.join(configs)}")
    if not args.sequential:
        print("note: latency columns reflect contention between concurrent configurations;")
        print("      token and accuracy columns are unaffected. Use --sequential for clean timing.\n")

    results_by_name = await runner.run_all(
        configs,
        turn_limit=turn_limit,
        phases=phases,
        repeats=args.repeats,
        parallel=not args.sequential,
        total_turns=args.total_turns,
        resume_at=args.resume_at,
    )

    meta = {
        "agent_model": AGENT_MODEL_ID,
        "rerank_model": RERANK_MODEL_ID,
        "account": ACCOUNT_ID,
        "region": REGION,
        "repeats": args.repeats,
        # The join key for validation/verify_logs.py. Without it, finding a run's entries in
        # an account-wide log group means guessing at timestamps again.
        "log_tag": metrics.RUN_TAG,
        "log_group": INVOCATION_LOG_GROUP,
    }
    payload = {}
    for name, collectors in results_by_name.items():
        entry = metrics.aggregate(collectors)
        entry["meta"] = meta
        payload[name] = entry

    json_path, report_path = report.write_outputs(payload, tag=args.tag)
    html_path = _write_html(payload, tag=args.tag)

    print("\n" + "=" * 78)
    print(report.build_report(payload))
    print("=" * 78)
    print(f"\nraw:    {json_path}")
    print(f"report: {report_path}")
    if html_path:
        print(f"html:   {html_path}")
    print(f"latest: {RESULTS_DIR / 'report-latest.md'}")

    failed = [name for name, entry in payload.items() if entry["summary"]["errors"]]
    if failed:
        print(f"\nconfigurations with errors: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="validation.run",
        description="Compare context-graph, progressive-tool-disclosure and relevance-filtering.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        metavar="NAME",
        help=f"configurations to run (default: all). choices: {', '.join(CONFIGURATIONS)}",
    )
    parser.add_argument("--turns", type=int, metavar="N", help="run only the first N turns")
    parser.add_argument(
        "--total-turns",
        type=int,
        metavar="N",
        help=(
            "pad the script with unscored filler up to N turns, keeping the return turns last. "
            "The eighteen scored turns and their expectations are untouched, so accuracy stays "
            "comparable. For the two mechanisms that answer to conversation length rather than to "
            "subject structure: the per-Card cost of the graph's final block, and the cost of "
            "deriving the graph, which measured 30ms at 18 turns and 2.9s at 200."
        ),
    )
    parser.add_argument(
        "--resume-at",
        type=int,
        metavar="N",
        help=(
            "drop the agent at turn N and build a new one over the same session for the rest. "
            "This is the shape an ephemeral runtime has, and the only way this harness exercises "
            "what graph persistence exists for: a long-lived process restores nothing, so the load "
            "path never runs. Installs a FileSessionManager under validation/.sessions."
        ),
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        metavar="PHASE",
        help="restrict to these scenario phases: main branch payload discovery return",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        metavar="N",
        help=(
            "replay each configuration N times and report the mean with its spread. "
            "A single replay is noise-dominated: the agent picks its own tool path and "
            "temperature cannot be pinned on Opus 4.8, so use 3 or more before trusting "
            "a difference under ~20 percent."
        ),
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help=(
            "run configurations one at a time. Slower by roughly the number of configurations, "
            "but the only way to get latency figures free of contention between them. Token and "
            "accuracy numbers are identical either way."
        ),
    )
    parser.add_argument("--smoke", action="store_true", help="cheap run that still exercises all three strategies")
    parser.add_argument("--tag", metavar="TAG", help="name the output files instead of using a timestamp")
    parser.add_argument("--report-only", metavar="JSON", help="re-render a report from a previous run's JSON")
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="with --report-only, recompute accuracy from the stored answers using the current expectations",
    )
    parser.add_argument("--verbose", action="store_true", help="turn on the plugins' own debug logging")

    args = parser.parse_args()
    _configure_logging(args.verbose)
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
