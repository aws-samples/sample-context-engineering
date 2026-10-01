"""Hermes harness entry point.

Offline (default) — build every arm and run them through a mocked model, no AWS credentials:

    .venv/bin/python -m src.run --offline --tag h01

Re-render a stored run:

    .venv/bin/python -m src.run --report-only results/run-h01.json

Live (spends real money on Bedrock — a separate, user-authorized step, Task 20.2):

    .venv/bin/python -m src.run --live --i-understand-this-spends-money --total-turns 60 --tag h01

The live path is intentionally a stub here: wiring a real Hermes agent to Bedrock is the paid run that
must not be launched without explicit authorization. The offline path is the verification the spec
requires (all five arms build and run through a mocked model; a recorded run re-renders with no creds).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import metrics, report, runner, scenario
from .config import CONFIGURATIONS, DEFAULT_CONFIGURATIONS, RUN_CONFIGS, RunConfig


class _FakeReranker:
    """Deterministic offline reranker: a chunk scores by shared words with the query. No network."""

    max_sources_per_query = 100

    async def score(self, query: str, chunks: list[str]) -> list[float]:
        terms = set(query.lower().split())
        return [min(1.0, len(terms & set(c.lower().split())) / max(1, len(terms))) for c in chunks]


class _FakeMatcher:
    """Deterministic offline similarity matcher for the context graph. No network."""

    def score(self, need: str, descriptions: Any) -> list[float]:
        terms = set(need.lower().split())
        return [len(terms & set(d.lower().split())) / max(1, len(terms)) for d in descriptions]


def _base_tool_schemas() -> list[dict]:
    from . import tools
    from ._toolshim import convert_to_openai_tool

    return [convert_to_openai_tool(t) for t in tools.all_tools()]


def build_engine(config: RunConfig) -> Any:
    """Map a RunConfig to the Hermes engine it selects, with offline fakes. None for the baseline."""
    if not (config.relevance or config.disclosure or config.graph):
        return None  # baseline: no engine

    relevance_cfg = {"reranker": _FakeReranker(), "chunk_tokens": 500, "preview_tokens": 2000, "relevance_threshold": 0.02}

    if config.relevance and config.disclosure and config.graph:
        from hermes_all_three import AllThreeEngine

        return AllThreeEngine(
            tool_specs=_base_tool_schemas(),
            relevance_threshold=0.02,
            max_result_tokens=4000,
            relevance_config=relevance_cfg,
            graph_kwargs={"matcher": _FakeMatcher()},
        )
    if config.relevance and not (config.disclosure or config.graph):
        from hermes_relevance_filter import RelevanceFilterEngine

        return RelevanceFilterEngine(max_result_tokens=4000, config=relevance_cfg)
    if config.disclosure and not (config.relevance or config.graph):
        from hermes_progressive_tool_disclosure import ProgressiveToolDisclosureEngine

        return ProgressiveToolDisclosureEngine(tool_specs=_base_tool_schemas())
    if config.graph and not (config.relevance or config.disclosure):
        from hermes_context_graph import ContextGraphEngine

        return ContextGraphEngine(matcher=_FakeMatcher())
    # leave-one-out arms: compose the present pair via AllThreeEngine with the absent practice disabled
    from hermes_all_three import AllThreeEngine

    engine = AllThreeEngine(
        tool_specs=_base_tool_schemas() if config.disclosure else [],
        relevance_threshold=0.02,
        max_result_tokens=4000,
        relevance_config=relevance_cfg,
        graph_kwargs={"matcher": _FakeMatcher()},
    )
    return engine


def _meta(names: list[str], repeats: int, turns: int, mode: str) -> dict[str, Any]:
    return {
        "framework": "hermes",
        "mode": mode,
        "configurations": names,
        "repeats": repeats,
        "turns": turns,
        "note": "Offline mock-model run — token/accuracy figures measure engine message-shaping, not a live LLM. "
        "Live Bedrock numbers are filled by the user-authorized --live run (Task 20.2).",
    }


def _offline(names: list[str], turns: tuple, repeats: int, tag: str) -> int:
    metrics.set_run_tag(tag)
    collectors = runner.run_all(names, turns, build_engine, RUN_CONFIGS, repeats=repeats)
    results: dict[str, Any] = {name: metrics.aggregate(cs) for name, cs in collectors.items()}
    results["meta"] = _meta(names, repeats, len(turns), "offline")
    json_path, md_path = report.write_outputs(results, tag=tag)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(report.build_table(results))
    return 0


def _report_only(path_text: str) -> int:
    payload = json.loads(Path(path_text).read_text())
    print(report.build_report(payload))
    return 0


def dry_run(names: list[str]) -> int:
    """Build every arm's engine and invoke nothing (default)."""
    for name in names:
        engine = build_engine(RUN_CONFIGS[name])
        label = getattr(engine, "name", "baseline (no engine)")
        tools_n = len(engine.get_tool_schemas()) if engine is not None else 0
        print(f"{name:16s} -> engine={label:28s} engine_tools={tools_n}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.run", description="Hermes context-engineering benchmark harness")
    p.add_argument("--offline", action="store_true", help="run every arm through a mocked model (no AWS)")
    p.add_argument("--live", action="store_true", help="invoke Bedrock (requires the ack flag) — Task 20.2, authorized only")
    p.add_argument("--i-understand-this-spends-money", action="store_true", dest="ack", help="acknowledge --live spends money")
    p.add_argument("--dry-run", action="store_true", help="build every arm and invoke nothing (default)")
    p.add_argument("--report-only", metavar="JSON", help="re-render a stored run and exit")
    p.add_argument("--configs", nargs="+", choices=CONFIGURATIONS, help="arms to run (default: the five standard arms)")
    p.add_argument("--turn-limit", type=int, help="run only the first N scripted turns")
    p.add_argument("--total-turns", type=int, help="pad the script with filler up to N turns")
    p.add_argument("--phases", nargs="+", choices=scenario.PHASES, help="restrict to these phases")
    p.add_argument("--repeats", type=int, default=1, help="replay each arm N times")
    p.add_argument("--tag", default="latest", help="run tag, used in the output filenames")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.report_only:
        return _report_only(args.report_only)

    names = list(args.configs) if args.configs else list(DEFAULT_CONFIGURATIONS)
    turns = scenario.turns(limit=args.turn_limit, phases=tuple(args.phases) if args.phases else None, total=args.total_turns)

    if args.live:
        if not args.ack:
            print("--live spends real money on Bedrock. Re-run with --i-understand-this-spends-money.", file=sys.stderr)
            return 2
        print(
            "The live Bedrock path is a user-authorized step (Task 20.2) and is intentionally not wired "
            "to launch from here. Run the offline harness with --offline, or wire the live agent "
            "deliberately once authorized.",
            file=sys.stderr,
        )
        return 2

    if args.offline:
        return _offline(names, turns, args.repeats, args.tag)

    return dry_run(names)  # default


if __name__ == "__main__":
    raise SystemExit(main())
