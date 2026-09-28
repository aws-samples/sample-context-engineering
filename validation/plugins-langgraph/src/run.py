"""CLI entry point. Everything that costs money is behind an explicit gate.

    # re-render a recorded run -- no credentials, no network, free
    .venv/bin/python -m src.run --report-only results/run-lg01.json

    # what a live run WOULD do, without doing it: builds every arm, prints the wiring, spends nothing
    .venv/bin/python -m src.run --dry-run

    # the live run. Refuses unless --i-understand-this-spends-money is also passed.
    .venv/bin/python -m src.run --live --i-understand-this-spends-money --total-turns 60 --tag lg01

**The gate is the point of this module.** The Strands harness's ``run.py`` spends money by default
and is guarded only by the operator's attention; a 60-turn five-arm run against Opus 4.8 is a few
hundred dollars, and this harness is new enough that a stray ``python -m src.run`` in a test sweep or
a tab-completion accident is a plausible way to lose it. So two flags are required rather than one,
neither has a short form, and the second one has to be spelled out. Absent them, ``main`` runs the
dry build and exits 0 -- which also means this module is safe to import and safe to invoke from a
test.

Task 7.4 -- the live run -- is deliberately NOT executed as part of building this harness.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from . import accuracy, config, metrics, report, runner, scenario, tools
from .config import (
    ACCOUNT_ID,
    AGENT_MODEL_ID,
    CONFIGURATIONS,
    DEFAULT_CONFIGURATIONS,
    EMBED_MODEL_ID,
    REGION,
    RERANK_MODEL_ID,
    RESULTS_DIR,
    RUN_CONFIGS,
    TARGET_SCHEMA_TOKENS,
)

_MIDDLEWARE_LOGGERS = (
    "langgraph_context_graph",
    "langgraph_progressive_tool_disclosure",
    "langgraph_relevance_filter",
    "context_core",
)
"""The middleware packages' own loggers, raised by ``--verbose``.

Package roots, not module paths: the graph in particular reports from several modules, and a pattern
scoped to one of them silently loses the records the others emit.
"""

_SPEND_ACK = "--i-understand-this-spends-money"


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    for noisy in ("botocore", "boto3", "urllib3", "httpx", "langchain_aws", "langgraph"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if verbose:
        for name in _MIDDLEWARE_LOGGERS:
            logging.getLogger(name).setLevel(logging.DEBUG)


def _meta(names: list[str], repeats: int, turns: int) -> dict[str, Any]:
    """Return the run metadata the report re-renders from.

    Recorded so a stored run can be priced and read without the shell that produced it: the model id
    in particular is looked up in :data:`~.config.MODEL_PRICING` at render time, and reading it from
    the environment instead would bill a recorded run at whatever the current shell happens to say.

    The two tool-suite figures are here because they are the one calibration that differs from the
    Strands harness -- see ``tools._FILLER_TOOL_COUNT``. A table built from this JSON therefore
    always carries the evidence of which suite size produced it.
    """
    return {
        "framework": "langgraph",
        "agent_model": AGENT_MODEL_ID,
        "rerank_model": RERANK_MODEL_ID,
        "embed_model": EMBED_MODEL_ID,
        "region": REGION,
        "configurations": names,
        "repeats": repeats,
        "turns_requested": turns,
        "target_schema_tokens": TARGET_SCHEMA_TOKENS,
        "core_tool_count": len(tools.CORE_TOOLS),
        "filler_tool_count": len(tools.FILLER_TOOLS),
        "filler_schema_tokens": tools.filler_schema_tokens(),
        "filler_count_pinned": tools._FILLER_TOOL_COUNT is not None,
        "cache_ttl": config.CACHE_TTL,
        "max_output_tokens": config.MAX_OUTPUT_TOKENS,
        "window_regime": config.WINDOW_REGIME,
        "run_tag": metrics.RUN_TAG,
    }


def dry_run(names: list[str]) -> int:
    """Build every named arm and report the wiring, without invoking a model.

    This is the check that belongs in CI: it exercises arm construction, middleware ordering, the
    tool suite and the metrics wiring -- everything except the paid call. It creates boto3 clients,
    which is a local operation and needs no valid credential.

    Args:
        names: Configuration names to build.

    Returns:
        Process exit status: 0 when every arm built.
    """
    print("DRY RUN -- building every arm, invoking nothing.")
    print(f"agent model (not called): {AGENT_MODEL_ID}")
    print(f"tool suite: {len(tools.CORE_TOOLS)} core + 1 web + {len(tools.FILLER_TOOLS)} filler")
    print(f"filler schema: ~{tools.filler_schema_tokens():,} tokens against a {TARGET_SCHEMA_TOKENS:,} budget")
    print()

    session = runner.boto_session()
    failures = 0
    for name in names:
        run_config = RUN_CONFIGS[name]
        # A fresh RunConfig per build: ``extra`` is mutated by the builder, so reusing the module
        # level instance across arms would leak one arm's metered clients into the next one's record.
        fresh = config.RunConfig(
            name=run_config.name,
            disclosure=run_config.disclosure,
            relevance=run_config.relevance,
            graph=run_config.graph,
            label=run_config.label,
            notes=run_config.notes,
        )
        try:
            middleware = runner.build_middleware(fresh, session)
            order = " -> ".join(type(each).__name__ for each in middleware) or "(none)"
            extra_tools = sorted(
                each.name for mw in middleware for each in (getattr(mw, "tools", ()) or ())
            )
            print(f"{name:12s} middleware (outermost first): {order}")
            print(f"{'':12s} tools added by middleware: {extra_tools or '(none)'}")
        except Exception as error:  # noqa: BLE001
            failures += 1
            print(f"{name:12s} FAILED TO BUILD: {type(error).__name__}: {error}", file=sys.stderr)

    print()
    print("no Bedrock call was made." if not failures else f"{failures} arm(s) failed to build.")
    return 1 if failures else 0


def _preflight() -> bool:
    """Verify credentials, account and rates before spending on a run."""
    try:
        session = runner.boto_session()
        identity = session.client("sts").get_caller_identity()
    except Exception as error:  # noqa: BLE001
        print(f"credentials unavailable: {error}", file=sys.stderr)
        print(
            "configure credentials the usual way -- isengardcli assume, an SSO session, exported "
            "keys, or a named profile in VALIDATION_AWS_PROFILE.",
            file=sys.stderr,
        )
        return False

    if ACCOUNT_ID and identity["Account"] != ACCOUNT_ID:
        print(f"wrong account: expected {ACCOUNT_ID}, got {identity['Account']}", file=sys.stderr)
        return False

    print(f"account: {identity['Account']} ({REGION})")
    print(f"agent:   {AGENT_MODEL_ID}")
    print(f"rerank:  {RERANK_MODEL_ID}")
    print(f"embed:   {EMBED_MODEL_ID}")
    print(f"tag:     {metrics.RUN_TAG}")
    # Printed because the cost column is a deliverable: a run billed at the wrong model's rates
    # produces a number that looks plausible and is wrong, and this is the last moment to catch it.
    print(
        f"rates:   ${config.PRICING.agent_input_per_mtok:.2f}/"
        f"${config.PRICING.agent_output_per_mtok:.2f} per Mtok in/out"
    )
    print(f"cache:   {config.CACHE_TTL or 'off'}")
    return True


async def _live(args: argparse.Namespace, names: list[str]) -> int:
    """Run the live benchmark. The only path in this package that calls Bedrock."""
    if not _preflight():
        return 1

    collectors = await runner.run_all(
        names,
        turn_limit=args.turn_limit,
        phases=tuple(args.phases) if args.phases else None,
        repeats=args.repeats,
        parallel=not args.sequential,
        total_turns=args.total_turns,
    )

    payload: dict[str, Any] = {"meta": _meta(names, args.repeats, args.total_turns or 0)}
    for name, replays in collectors.items():
        if len(replays) == 1:
            payload[name] = {**replays[0].to_dict(), "repeats": 1}
        else:
            payload[name] = {
                "summary": metrics.aggregate(replays)["summary"]
                if "summary" in metrics.aggregate(replays)
                else metrics.aggregate(replays),
                "turns": replays[0].to_dict()["turns"],
                "calls": replays[0].to_dict()["calls"],
                "repeats": len(replays),
                "per_repeat": [collector.to_dict() for collector in replays],
            }

    json_path, md_path = report.write_outputs(payload, tag=args.tag)
    print()
    print(report.build_table(payload))
    print()
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


def _report_only(path_text: str) -> int:
    """Re-render a stored run. No credentials involved."""
    path = Path(path_text)
    if not path.exists():
        print(f"no such file: {path}", file=sys.stderr)
        return 1
    payload = json.loads(path.read_text(encoding="utf-8"))
    out = path.with_suffix(".md")
    out.write_text(report.build_report(payload), encoding="utf-8")
    print(report.build_table(payload))
    print()
    print(f"wrote {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser. Exposed so a test can assert the spend gate exists."""
    parser = argparse.ArgumentParser(
        prog="python -m src.run",
        description="LangGraph A/B/D validation harness. Defaults to a dry build that spends nothing.",
    )
    parser.add_argument("--live", action="store_true", help="actually invoke Bedrock (requires the ack flag)")
    parser.add_argument(
        _SPEND_ACK,
        dest="spend_ack",
        action="store_true",
        help="acknowledge that --live spends real money on Bedrock",
    )
    parser.add_argument("--dry-run", action="store_true", help="build every arm and invoke nothing (default)")
    parser.add_argument("--report-only", metavar="JSON", help="re-render a stored run and exit")
    parser.add_argument("--configs", nargs="+", choices=CONFIGURATIONS, help="arms to run")
    parser.add_argument("--turn-limit", type=int, help="run only the first N scripted turns")
    parser.add_argument("--total-turns", type=int, help="pad the script with filler up to N turns")
    parser.add_argument("--phases", nargs="+", choices=scenario.PHASES, help="restrict to these phases")
    parser.add_argument("--repeats", type=int, default=1, help="replay each arm N times")
    parser.add_argument("--sequential", action="store_true", help="run arms one at a time")
    parser.add_argument("--tag", default="latest", help="run tag, used in the output filenames")
    parser.add_argument("--verbose", action="store_true", help="raise the middleware loggers to DEBUG")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch. Spends money only when both gate flags are present."""
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    metrics.set_run_tag(args.tag)

    names = list(args.configs) if args.configs else list(DEFAULT_CONFIGURATIONS)

    if args.report_only:
        return _report_only(args.report_only)

    if not args.live:
        return dry_run(names)

    if not args.spend_ack:
        print(
            "--live spends real money: a 60-turn five-arm run against "
            f"{AGENT_MODEL_ID} is on the order of hundreds of dollars at list price.\n"
            f"Re-run with {_SPEND_ACK} if that is what you want.\n"
            "Nothing was invoked.",
            file=sys.stderr,
        )
        return 2

    return asyncio.run(_live(args, names))


if __name__ == "__main__":
    raise SystemExit(main())
