"""Gate D command line: run the contract suite, turn a live run into evidence, decide.

From ``validation/``::

    python3.13 -m opencode_gate_d suite --title ce60-A-ce60-v6-sonnet55-r1-A --out results/gate-d/suite.json
    python3.13 -m opencode_gate_d live  --run-dir ../validation/opencode_gate_a/results/ce60-v7-sonnet55-r1 --out results/gate-d/live-r1.json
    python3.13 -m opencode_gate_d decide --evidence results/gate-d/suite.json --evidence results/gate-d/live-r1.json \\
        --gate-a-outcome no_go --waiver-by owner --waiver-requirement FR5.1/BR5.1 --waiver-reason "..." --out results/gate-d/decision.json

Nothing here makes a network call or reads a credential: the suite uses the deterministic offline matcher, and the live
subcommand only reads the payload-free files the runner wrote.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from context_core.opencode_graph import GraphConfig

from .contracts import GateDRequestEvidence, absence_digest
from .evaluator import evaluate
from .recorded import copied_store, find_session, wire_transcript
from .suite import SuiteResult, run_suite, synthetic_wire


def _write(path: str | None, body: dict[str, Any]) -> None:
    text = json.dumps(body, indent=2, sort_keys=False) + "\n"
    if path is None:
        sys.stdout.write(text)
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _suite_body(
    result: SuiteResult, sources: list[str], *, unique_titles: bool, mode: str, keep_recent: int, keep_dialogue: int
) -> dict[str, Any]:
    return {
        "kind": "gate-d-suite-evidence",
        "sources": sources,
        "matcher": "deterministic-jaccard-offline",
        "unique_titles": unique_titles,
        "mode": mode,
        "keep_recent": keep_recent if mode == "inplace" else None,
        "keep_dialogue": keep_dialogue if mode == "inplace" else None,
        "counts": result.counts,
        "failures": result.failures,
        "requests": [request.to_dict() for request in result.requests],
    }


def cmd_suite(args: argparse.Namespace) -> int:
    wires: dict[str, list[dict[str, Any]]] = {}
    if not args.no_synthetic:
        wires["synthetic"] = synthetic_wire(args.synthetic_turns)
    if args.title:
        with copied_store(Path(args.store) if args.store else None) as connection:
            for title in args.title:
                session_id = find_session(connection, title)
                if session_id is None:
                    print(f"no session titled {title!r}", file=sys.stderr)
                    return 2
                wires[title] = wire_transcript(connection, session_id)
    config = GraphConfig(
        enabled=True,
        unique_titles=args.unique_titles,
        mode=args.mode,
        keep_recent=args.keep_recent,
        keep_dialogue=args.keep_dialogue,
    )
    result = run_suite(wires, gate_a_evidence_id=args.gate_a_evidence_id, config=config)
    _write(
        args.out,
        _suite_body(
            result,
            list(wires),
            unique_titles=args.unique_titles,
            mode=args.mode,
            keep_recent=args.keep_recent,
            keep_dialogue=args.keep_dialogue,
        ),
    )
    print(
        f"suite: {len(result.requests)} requests, counts={result.counts}, failures={result.failures}", file=sys.stderr
    )
    return 0 if not result.failures else 1


def live_evidence(run_dir: Path, *, arm: str, gate_a_evidence_id: str) -> dict[str, Any]:
    """Turn one live A+D run into request evidence, from the two payload-free files the runner wrote."""
    durability = json.loads((run_dir / f"{arm}-graph-durability.json").read_text(encoding="utf-8"))
    lines = [
        json.loads(line)
        for line in (run_dir / f"{arm}-graph-evidence.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stored = durability.get("stored_digests") or {}
    synthetic_in_store = int(durability.get("stored_graph_synthetic_messages", 0))
    latest_cards: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {}
    requests: list[GateDRequestEvidence] = []
    for line in lines:
        session = line.get("session_id") or "unknown"
        stats = line.get("stats") or {}
        if "card_sources" in stats:
            latest_cards[session] = tuple((c["card_id"], tuple(c["sources"])) for c in stats["card_sources"])
        cards = latest_cards.get(session, ())
        before = line["received"]["digest"]
        after = stored.get(str(line["request_index"]), "")
        scenarios = ["live"]
        if stats.get("mode") == "delivered":
            scenarios.append("tool_loop")
        if not cards:
            scenarios.append("pre_card")
        requests.append(
            GateDRequestEvidence(
                evidence_id=f"gd-live-{line['request_index']:05d}",
                gate_a_evidence_id=gate_a_evidence_id,
                request_id=f"{session}:{line['request_index']}",
                stream_id=session,
                transcript_before_digest=before,
                transcript_after_digest=after,
                card_ids_and_sources=cards,
                scenarios=tuple(scenarios),
                full_pass_digest=None,
                baseline_model_input_digest=before,
                canonical_equality=bool(after) and after == before,
                absence_digest=absence_digest(
                    {
                        "graph_messages_in_store": synthetic_in_store,
                        "durable_writes": 0,
                        "stored_transcript_changed": after != before,
                    }
                ),
                projected=line.get("operation") == "projected",
                epoch=int(stats.get("resets", 0)),
            )
        )
    return {
        "kind": "gate-d-live-evidence",
        "run_dir": str(run_dir),
        "arm": arm,
        "requests": [request.to_dict() for request in requests],
        "failures": [],
    }


def cmd_live(args: argparse.Namespace) -> int:
    body = live_evidence(Path(args.run_dir), arm=args.arm, gate_a_evidence_id=args.gate_a_evidence_id)
    _write(args.out, body)
    print(f"live: {len(body['requests'])} requests from {args.run_dir}", file=sys.stderr)
    return 0


def cmd_decide(args: argparse.Namespace) -> int:
    requests: list[GateDRequestEvidence] = []
    failures: list[str] = []
    for path in args.evidence:
        body = json.loads(Path(path).read_text(encoding="utf-8"))
        requests.extend(GateDRequestEvidence.from_dict(item) for item in body["requests"])
        failures.extend(body.get("failures", []))
    waiver = (
        {"by": args.waiver_by, "requirement": args.waiver_requirement, "reason": args.waiver_reason}
        if args.waiver_by
        else None
    )
    decision = evaluate(requests, gate_a_outcome=args.gate_a_outcome, waiver=waiver, scenario_failures=failures)
    _write(args.out, decision.to_dict())
    print(
        f"gate D: {decision.outcome} ({decision.precondition}) capability={decision.capability_state} "
        f"reasons={list(decision.reasons)}",
        file=sys.stderr,
    )
    return 0 if decision.outcome == "go" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="opencode_gate_d", description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    suite = sub.add_parser("suite", help="Run the five scenarios over recorded and synthetic transcripts.")
    suite.add_argument("--title", action="append", default=[], help="Exact title of a recorded session (repeatable).")
    suite.add_argument("--store", default=None, help="OpenCode data directory (default: ~/.local/share/opencode).")
    suite.add_argument("--no-synthetic", action="store_true")
    suite.add_argument(
        "--unique-titles",
        action="store_true",
        help="Run the binding with CONTEXT_GRAPH_UNIQUE_TITLES on (each turn gets a Card of its own).",
    )
    suite.add_argument(
        "--mode",
        choices=("reference", "inplace"),
        default="reference",
        help="The graph policy to run the scenarios under (CONTEXT_GRAPH_MODE).",
    )
    suite.add_argument(
        "--keep-recent",
        type=int,
        default=0,
        help="With --mode inplace: closed turns kept as they were (CONTEXT_GRAPH_KEEP_RECENT).",
    )
    suite.add_argument(
        "--keep-dialogue",
        type=int,
        default=0,
        help="With --mode inplace: closed turns that keep their question and text answers (CONTEXT_GRAPH_KEEP_DIALOGUE).",
    )
    suite.add_argument("--synthetic-turns", type=int, default=10)
    suite.add_argument("--gate-a-evidence-id", default="gate-a-v6-no_go")
    suite.add_argument("--out", default=None)
    suite.set_defaults(func=cmd_suite)

    live = sub.add_parser("live", help="Turn a live A+D run's graph files into request evidence.")
    live.add_argument("--run-dir", required=True)
    live.add_argument("--arm", default="A")
    live.add_argument("--gate-a-evidence-id", default="gate-a-v6-no_go")
    live.add_argument("--out", default=None)
    live.set_defaults(func=cmd_live)

    decide = sub.add_parser("decide", help="Decide Gate D from evidence files.")
    decide.add_argument("--evidence", action="append", required=True)
    decide.add_argument("--gate-a-outcome", required=True)
    decide.add_argument("--waiver-by", default=None)
    decide.add_argument("--waiver-requirement", default="FR5.1/BR5.1")
    decide.add_argument("--waiver-reason", default="")
    decide.add_argument("--out", default=None)
    decide.set_defaults(func=cmd_decide)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
