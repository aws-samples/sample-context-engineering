"""Replay a recorded session through the graph binding, request by request, and measure what D would have left out.

This is a counterfactual on the HISTORY, not on the model: the model's later turns would differ once its context differed,
so the numbers say how much of each recorded request practice D would have removed, not what a live run will save. It is
useful for two things -- checking the binding on real transcripts (do Cards form, do titles collide, does the removal ever
split a tool pair) and sizing the live experiment before it is paid for.

Sizes are ``chars / 4`` of the text a model would read (text, tool input, tool output). That is the repository's usual coarse
estimate, applied to both sides of the comparison, so the ratio is meaningful and the absolute count is not.

Usage (from ``validation/``)::

    python3.13 -m opencode_gate_d.replay --title ce60-A-ce60-v6-sonnet55-r1-A [--matcher keyword|embedding]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from context_core.graph import projection
from context_core.graph.matcher import SimilarityMatcher
from context_core.graph.state import CardChoice, Resolution, TurnChoice
from context_core.opencode_graph import GraphConfig, OpenCodeGraph

from .recorded import copied_store, find_session, request_prefixes, wire_transcript
from .suite import model_input

__all__ = [
    "RequestReplay",
    "ReplaySummary",
    "estimate_tokens",
    "replay_inplace",
    "replay_wire",
    "sticky_collapse",
    "summarize",
]

_RANK: dict[str, int] = {"full": 2, "description": 1, "title": 0}


def estimate_tokens(messages: Sequence[dict[str, Any]]) -> int:
    """``chars / 4`` over everything a model would read in these wire messages."""
    chars = 0
    for message in messages:
        for part in message.get("parts", ()):
            if part.get("type") == "text":
                chars += len(part.get("text", ""))
            elif part.get("type") == "tool":
                chars += len(json.dumps(part.get("input", {}), separators=(",", ":")))
                chars += len(part.get("output", "") or "") + len(part.get("error", "") or "")
    return -(-chars // 4)


@dataclass(frozen=True)
class RequestReplay:
    """One replayed request. Counts only."""

    index: int
    messages: int
    operation: str
    mode: str | None
    removed: int
    received_tokens: int
    sent_tokens: int
    block_chars: int
    cards: int
    pair_split: bool
    view_tokens: tuple[int, ...] = ()
    view_keys: tuple[str, ...] = ()
    removed_ids: frozenset[str] = frozenset()


@dataclass
class ReplaySummary:
    """The whole replay, reduced to what the report needs."""

    requests: list[RequestReplay] = field(default_factory=list)
    closed_turns: int = 0

    def totals(self) -> dict[str, Any]:
        received = sum(request.received_tokens for request in self.requests)
        sent = sum(request.sent_tokens for request in self.requests)
        engaged = [request for request in self.requests if request.operation == "projected"]
        return {
            "requests": len(self.requests),
            "engaged_requests": len(engaged),
            "received_tokens_est": received,
            "sent_tokens_est": sent,
            "reduction_pct": round(100.0 * (sent - received) / received, 2) if received else None,
            "max_received_tokens_est": max((request.received_tokens for request in self.requests), default=0),
            "max_sent_tokens_est": max((request.sent_tokens for request in self.requests), default=0),
            "removed_messages": sum(request.removed for request in self.requests),
            "block_chars": sum(request.block_chars for request in self.requests),
            "pair_splits": sum(1 for request in self.requests if request.pair_split),
            "cards_final": self.requests[-1].cards if self.requests else 0,
            "closed_turns_final": self.closed_turns,
            "cards_lost_to_title_collision": max(
                0, self.closed_turns - (self.requests[-1].cards if self.requests else 0)
            ),
        }


def _tool_ids(messages: Sequence[dict[str, Any]]) -> tuple[set[str], set[str]]:
    """The completed tool call ids present in these wire messages (a wire tool part carries call and result together)."""
    calls: set[str] = set()
    for message in messages:
        for part in message.get("parts", ()):
            if part.get("type") == "tool" and part.get("status") in {"completed", "error"}:
                calls.add(part["callID"])
    return calls, calls


def _key_of(message: dict[str, Any]) -> str:
    """The name of a message's content for the cache model: its id, plus its text when the graph made it."""
    if not str(message["id"]).startswith("graph-"):
        return str(message["id"])
    text = "".join(part.get("text", "") for part in message.get("parts", ()))
    return f"{message['id']}:{hashlib.sha1(text.encode()).hexdigest()[:12]}"


def replay_wire(
    wire: Sequence[dict[str, Any]],
    *,
    graph: OpenCodeGraph,
    session_id: str = "replay",
    on_request: Callable[[int, int], None] | None = None,
) -> ReplaySummary:
    """Run every request prefix of ``wire`` through ``graph`` and apply each answer the way the plugin would."""
    summary = ReplaySummary()
    baseline_views: list[tuple[tuple[str, ...], tuple[int, ...]]] = []
    prefixes = request_prefixes(wire)
    for index, prefix in enumerate(prefixes, start=1):
        response = graph.project(session_id, prefix, scoring_allowed=True)
        removed_ids = set(response.get("removed_ids", ())) if response.get("operation") == "projected" else set()
        block = response.get("final_block") if response.get("operation") == "projected" else None
        # What the plugin would send: removed messages out, a summary where an in-place collapse says, the block last.
        sent_messages = model_input(prefix, response)
        sent_tokens = estimate_tokens(sent_messages)
        # What the provider reads, in order, one entry per message. A stored message is immutable, so its id names its
        # content; a message the graph made (a summary, the block) is named by its text as well.
        view_keys = tuple(_key_of(message) for message in sent_messages)
        view_tokens = tuple(estimate_tokens([message]) for message in sent_messages)
        stats = response.get("stats") or {}
        summary.requests.append(
            RequestReplay(
                index=index,
                messages=len(prefix),
                operation=str(response.get("operation")),
                mode=stats.get("mode"),
                removed=len(removed_ids),
                received_tokens=estimate_tokens(prefix),
                sent_tokens=sent_tokens,
                block_chars=len(block or ""),
                cards=int(stats.get("cards", 0)),
                view_tokens=view_tokens,
                view_keys=view_keys,
                removed_ids=frozenset(removed_ids),
                # A wire tool part carries its call and result together, so a removal can never split a pair here; the
                # neutral view is where pairs are checked (see tests). Kept as a field so a regression shows in the data.
                pair_split=False,
            )
        )
        baseline_views.append((tuple(message["id"] for message in prefix), tuple(estimate_tokens([m]) for m in prefix)))
        if on_request is not None:
            on_request(index, len(prefixes))
    users = [
        message
        for message in wire
        if message["role"] == "user" and any(p.get("type") == "text" for p in message["parts"])
    ]
    summary.closed_turns = max(0, len(users) - 1)
    summary.baseline_views = baseline_views  # type: ignore[attr-defined]
    return summary


#: Relative prices of the three kinds of input token on a provider with prompt caching, in units of the plain input price.
CACHE_READ = 0.10
CACHE_WRITE = 1.25

#: Effective dollars per million tokens for Sonnet 5.5 on Bedrock, fitted from OpenCode's own per-message cost counters over
#: twelve recorded sessions (six baseline, six with a plugin): the fit is exact to the cent. A cache write costs 25 times a read.
PRICE_CACHE_READ = 0.10
PRICE_CACHE_WRITE = 2.50
PRICE_OUTPUT = 10.00

FIXED_PREFIX_TOKENS = 28_700
"""System prompt plus tool schemas of the CE-60 harness: what the first call of every session writes and every later call reads
(27.6k-28.8k in the recorded baselines). It is real, not estimated, so the calibration never scales it."""


def cache_model(
    summary: ReplaySummary, *, fixed_overhead_tokens: int = FIXED_PREFIX_TOKENS, checkpoints: bool = True
) -> dict[str, Any]:
    """Input-side cost of the replayed requests with and without the projection, under prompt caching.

    ``checkpoints=True`` (the default) is the model that matches the real bills; ``False`` is the older, optimistic one. See
    :func:`_run_cache`. ``fixed_overhead_tokens`` stands for the system prompt and tool schemas, read from cache on every
    request after the first. Output tokens are left out on purpose: they depend on the model's behaviour, which a replay of
    the history cannot say.
    """
    return_value: dict[str, Any] = {"model": "checkpoint-only" if checkpoints else "ideal-prefix"}
    projected_views = [(request.view_keys, request.view_tokens) for request in summary.requests]
    return_value["projected"] = _with_units(_run_cache(projected_views, fixed_overhead_tokens, checkpoints=checkpoints))
    # The un-projected history is not in the summary; callers pass it through ``summary.baseline_views`` when they have it.
    baseline_views = getattr(summary, "baseline_views", [])
    if baseline_views:
        return_value["baseline"] = _with_units(
            _run_cache(baseline_views, fixed_overhead_tokens, checkpoints=checkpoints)
        )
        base = return_value["baseline"]["cost_units"]
        return_value["cost_change_pct"] = round(100.0 * (return_value["projected"]["cost_units"] - base) / base, 2)
    return return_value


def _with_units(run: dict[str, float]) -> dict[str, float]:
    return {**run, "cost_units": run["read_tokens"] * CACHE_READ + run["write_tokens"] * CACHE_WRITE}


def replay_inplace(
    wire: Sequence[dict[str, Any]], *, graph: OpenCodeGraph, keep_recent: int, session_id: str = "replay"
) -> ReplaySummary:
    """SUPERSEDED by running the real binding with ``--mode inplace``: kept because earlier result files were made with it.

    SIZING EXPERIMENT: collapse each turn IN PLACE the moment it closes, for good, with no block at the end of the request.

    The reference puts every collapsed turn into one block appended after the last message and re-decides, turn by turn, which
    Cards are shown whole. Both move early parts of the request or rewrite its tail on every call, and on a provider that
    prices a cache write at 25 times a read that is what the bill is made of. This variant is the opposite on both counts:

    * a closed turn becomes ONE short message (its Card: title, description, numeric lines) at the position the turn had, as
      soon as the next user message arrives, and it is never expanded again by the projection (a recall would return the
      content at the tail, as a tool result, so it is not in this model);
    * the ``keep_recent`` most recent closed turns stay whole, so the model still sees the latest exchange as it was.

    Collapsing the turn that has just closed rewrites only what follows it (the new user message), so the stable prefix never
    moves. The Cards come from the real core (descriptions, titles, numeric lines); the policy that decides when to collapse
    is the only thing replaced, and it needs no embedding model. What the model would do with such a context is unknown.
    """
    summary = ReplaySummary()
    baseline_views: list[tuple[tuple[str, ...], tuple[int, ...]]] = []
    prefixes = request_prefixes(wire)
    for index, prefix in enumerate(prefixes, start=1):
        graph.project(session_id, prefix, scoring_allowed=False)  # derives the Cards; with no scoring it never removes
        state = graph._sessions[session_id].state  # noqa: SLF001 - an analysis tool reading the binding's own state
        card_by_message = {}
        for card in state.cards.values() if state else ():
            for message_id in (*card.dialogue_ids, *card.evidence_ids):
                card_by_message[message_id] = card

        starts = [
            i
            for i, m in enumerate(prefix)
            if m["role"] == "user" and any(p.get("type") == "text" and not p.get("synthetic") for p in m["parts"])
        ]
        bounds = [(start, starts[n + 1] if n + 1 < len(starts) else len(prefix)) for n, start in enumerate(starts)]
        closed = bounds[:-1]
        collapsed_ranges = closed[: max(0, len(closed) - keep_recent)]
        collapsed_at = {start: stop for start, stop in collapsed_ranges}
        covered = {i for start, stop in collapsed_ranges for i in range(start, stop)}

        keys: list[str] = []
        tokens: list[int] = []
        removed = 0
        position = 0
        while position < len(prefix):
            if position in collapsed_at:
                stop = collapsed_at[position]
                turn_ids = [prefix[i]["id"] for i in range(position, stop)]
                card = next((card_by_message[i] for i in turn_ids if i in card_by_message), None)
                text = (
                    f"{card.title}\n{card.description}" if card else prefix[position]["parts"][0].get("text", "")[:200]
                )
                keys.append(f"card:{prefix[position]['id']}")
                tokens.append(-(-len(text) // 4))
                removed += stop - position
                position = stop
                continue
            if position not in covered:
                keys.append(prefix[position]["id"])
                tokens.append(estimate_tokens([prefix[position]]))
            position += 1
        summary.requests.append(
            RequestReplay(
                index=index,
                messages=len(prefix),
                operation="projected" if removed else "unchanged",
                mode=None,
                removed=removed,
                received_tokens=estimate_tokens(prefix),
                sent_tokens=sum(tokens),
                block_chars=0,
                cards=len(card_by_message),
                pair_split=False,
                view_tokens=tuple(tokens),
                view_keys=tuple(keys),
            )
        )
        baseline_views.append((tuple(m["id"] for m in prefix), tuple(estimate_tokens([m]) for m in prefix)))
    users = [m for m in wire if m["role"] == "user" and any(p.get("type") == "text" for p in m["parts"])]
    summary.closed_turns = max(0, len(users) - 1)
    summary.baseline_views = baseline_views  # type: ignore[attr-defined]
    return summary


def calibrated_dollars(
    summary: ReplaySummary,
    *,
    real_cache_tokens: int,
    real_output_tokens: int,
    fixed_overhead_tokens: int = FIXED_PREFIX_TOKENS,
    checkpoints: bool = True,
    baseline_views: list[tuple[tuple[str, ...], tuple[int, ...]]] | None = None,
) -> dict[str, Any]:
    """The cache model in dollars, scaled so the un-projected history matches a REAL baseline session's token count.

    The replay counts the conversation as chars/4, which misestimates structured text. ``real_cache_tokens`` is the real
    baseline's cache read plus cache write; the part of it that is not the fixed prefix is what the baseline's conversation
    views are scaled to, and the same scale is applied to the projected views, so the comparison between the two does not
    depend on the misestimate. ``baseline_views`` are the un-projected requests of the REAL baseline session (when absent, the
    replayed session's own, which compares D with A alone and not with the baseline). Output tokens are the real baseline's,
    because what the model would write under another context is exactly what a replay of the history cannot say.
    """
    if baseline_views is None:
        baseline_views = getattr(summary, "baseline_views", [])
    base = _run_cache(list(baseline_views), 0, checkpoints=checkpoints)
    projected = _run_cache([(r.view_keys, r.view_tokens) for r in summary.requests], 0, checkpoints=checkpoints)
    prefix_reads = fixed_overhead_tokens * max(0, len(baseline_views) - 1)
    prefix_writes = fixed_overhead_tokens
    scale = (real_cache_tokens - prefix_reads - prefix_writes) / (base["read_tokens"] + base["write_tokens"])

    def dollars(run: dict[str, float], calls: int) -> float:
        fixed_reads = fixed_overhead_tokens * max(0, calls - 1)
        return (
            scale * (run["read_tokens"] * PRICE_CACHE_READ + run["write_tokens"] * PRICE_CACHE_WRITE)
            + fixed_reads * PRICE_CACHE_READ
            + prefix_writes * PRICE_CACHE_WRITE
        ) / 1e6 + real_output_tokens * PRICE_OUTPUT / 1e6

    base_dollars = dollars(base, len(baseline_views))
    projected_dollars = dollars(projected, len(summary.requests))
    base_writes = scale * base["write_tokens"] + prefix_writes
    projected_writes = scale * projected["write_tokens"] + prefix_writes
    base_total = scale * (base["read_tokens"] + base["write_tokens"]) + prefix_reads + prefix_writes
    projected_total = (
        scale * (projected["read_tokens"] + projected["write_tokens"])
        + fixed_overhead_tokens * max(0, len(summary.requests) - 1)
        + prefix_writes
    )
    return {
        "scale_to_real_tokens": round(scale, 3),
        "baseline_dollars": round(base_dollars, 3),
        "projected_dollars": round(projected_dollars, 3),
        "dollars_change_pct": round(100.0 * (projected_dollars - base_dollars) / base_dollars, 1),
        "cache_write_ratio": round(projected_writes / base_writes, 2),
        "tokens_change_pct": round(100.0 * (projected_total - base_total) / base_total, 1),
    }


def _run_cache(
    views: list[tuple[tuple[str, ...], tuple[int, ...]]], fixed_overhead_tokens: int, *, checkpoints: bool = True
) -> dict[str, float]:
    """Read and write tokens of a series of requests, one ``(message keys, message tokens)`` pair each.

    ``checkpoints=True`` models what OpenCode and the provider really do. OpenCode marks the system prompt and the last two
    messages of every request with a cache checkpoint, and the provider reads a request only up to the longest prefix that ENDS
    at a checkpoint an earlier request wrote and is identical to this request's prefix; everything after it is written (a
    write costs 25 times a read). Appending is therefore free, and editing an earlier message makes everything after the
    edit a write unless the edit sits right after a message that ended an earlier request. Validated against the three v8
    repetitions: writes within 10 %, dollars within 4-8 points, and the first call of every turn reading only the system prefix.

    ``checkpoints=False`` is the older model, which assumes a hit at the longest common prefix with the PREVIOUS request, as
    if a checkpoint stood at every message. It is exact for a history that only appends and optimistic for any edit; it
    predicted +134 % where the reference policy of D really cost +206 %, and -30 % where the dialogue tier of v8 really cost
    +17 %.
    """
    reads = writes = 0.0
    previous_keys: tuple[str, ...] = ()
    cached: set[tuple[str, ...]] = {()}
    for index, (keys, tokens) in enumerate(views):
        if checkpoints:
            common = next(end for end in range(len(keys), -1, -1) if keys[:end] in cached)
        else:
            common = 0
            for position, (key, previous) in enumerate(zip(keys, previous_keys)):
                if key != previous:
                    break
                common = position + 1
        reads += sum(tokens[:common]) + (fixed_overhead_tokens if index else 0)
        writes += sum(tokens[common:]) + (fixed_overhead_tokens if not index else 0)
        previous_keys = keys
        for end in (len(keys), len(keys) - 1):
            if end >= 1:
                cached.add(keys[:end])
    return {"read_tokens": reads, "write_tokens": writes}


def compare_with_live(summary: ReplaySummary, evidence_lines: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """How closely the replayed removals match what the live run removed, request by request (same position in the session)."""
    pairs = list(zip(summary.requests, evidence_lines))
    identical = sum(1 for replayed, live in pairs if set(live.get("removed_ids", ())) == replayed.removed_ids)
    overlap = [
        len(set(live.get("removed_ids", ())) & replayed.removed_ids)
        / max(1, len(set(live.get("removed_ids", ())) | replayed.removed_ids))
        for replayed, live in pairs
    ]
    return {
        "requests_compared": len(pairs),
        "identical_removal": identical,
        "mean_jaccard": round(sum(overlap) / len(overlap), 3) if overlap else None,
    }


def summarize(summary: ReplaySummary) -> dict[str, Any]:
    """The report body: totals plus the per-request series (counts only)."""
    return {
        "totals": summary.totals(),
        "series": [
            {
                "i": request.index,
                "messages": request.messages,
                "op": request.operation,
                "mode": request.mode,
                "removed": request.removed,
                "received": request.received_tokens,
                "sent": request.sent_tokens,
                "block_chars": request.block_chars,
                "cards": request.cards,
            }
            for request in summary.requests
        ],
    }


class sticky_collapse:  # noqa: N801 - used as a context manager, reads like a function
    """A SIZING EXPERIMENT, not a feature: let a Card step down but never back up, unless a retrieval raised it.

    The reference choice is recomputed from the question every turn, so a Card that was relevant last turn and is not now flips
    from full back to a title, and one that becomes relevant flips the other way. Each flip changes an early message, which moves
    the first difference between two consecutive requests forward and turns everything after it into cache writes. This wrapper
    keeps, per Card, the most collapsed resolution it has had on each axis, except while the Card holds a fed-back Note (a
    retrieval raised it). It patches the core's private choice step for the duration of a replay and restores it afterwards; it
    changes no file. What it can say is what the CONTEXT would be; what the model would do with it, only a live run says.
    """

    def __enter__(self) -> sticky_collapse:
        self._original = projection._compute_choice
        memo: dict[str, CardChoice] = {}
        original = self._original

        def lower(a: Resolution, b: Resolution) -> Resolution:
            return a if _RANK[a] <= _RANK[b] else b

        def wrapper(state, messages, matcher, body_budget, thresholds):  # type: ignore[no-untyped-def]
            choice = original(state, messages, matcher, body_budget, thresholds)
            if choice.full_pass:
                return choice
            decided: dict[str, CardChoice] = {}
            for title, current in choice.by_title.items():
                before = memo.get(title)
                if before is not None and title not in state.reuse:
                    current = CardChoice(
                        dialogue=lower(before.dialogue, current.dialogue),
                        evidence=lower(before.evidence, current.evidence),
                    )
                decided[title] = current
            memo.update(decided)
            from types import MappingProxyType

            return TurnChoice(by_title=MappingProxyType(decided), full_pass=False, selected=choice.selected)

        projection._compute_choice = wrapper  # type: ignore[assignment]
        return self

    def __exit__(self, *exc: object) -> None:
        projection._compute_choice = self._original  # type: ignore[assignment]


def _keyword_matcher() -> SimilarityMatcher:
    """A deterministic bag-of-words matcher for offline use: Jaccard over lower-cased word sets."""

    class Jaccard:
        def score(self, question: str, descriptions: Sequence[str]) -> list[float]:
            asked = {word for word in question.lower().split() if len(word) > 3}
            scores = []
            for description in descriptions:
                words = {word for word in description.lower().split() if len(word) > 3}
                scores.append(len(asked & words) / len(asked | words) if asked | words else 0.0)
            return scores

    return Jaccard()


def _embedding_matcher() -> SimilarityMatcher:
    import boto3
    from botocore.config import Config

    from context_core.graph.matcher import EmbeddingSimilarityMatcher

    session = boto3.Session(profile_name="claude-code", region_name="us-west-2")
    client = session.client(
        "bedrock-runtime",
        config=Config(connect_timeout=5, read_timeout=10, retries={"max_attempts": 2, "mode": "standard"}),
    )
    return EmbeddingSimilarityMatcher(model_id="cohere.embed-multilingual-v3", client=client)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opencode_gate_d.replay", description=__doc__)
    parser.add_argument("--title", required=True, help="Exact session title in the OpenCode store.")
    parser.add_argument("--matcher", choices=("keyword", "embedding"), default="embedding")
    parser.add_argument("--series", action="store_true", help="Also print the per-request series.")
    parser.add_argument(
        "--sticky",
        action="store_true",
        help="Sizing experiment: Cards step down but never back up (see sticky_collapse).",
    )
    parser.add_argument(
        "--inplace",
        type=int,
        default=None,
        metavar="K",
        help="Sizing experiment: collapse each closed turn in place for good, keeping the K most recent whole (see replay_inplace).",
    )
    parser.add_argument(
        "--unique-titles",
        action="store_true",
        help="Give every user turn a Card of its own (CONTEXT_GRAPH_UNIQUE_TITLES).",
    )
    parser.add_argument(
        "--mode",
        choices=("reference", "inplace"),
        default="reference",
        help="Run the real binding under this policy (CONTEXT_GRAPH_MODE).",
    )
    parser.add_argument("--keep-recent", type=int, default=0, help="With --mode inplace: CONTEXT_GRAPH_KEEP_RECENT.")
    parser.add_argument(
        "--keep-dialogue", type=int, default=0, help="With --mode inplace: CONTEXT_GRAPH_KEEP_DIALOGUE."
    )
    parser.add_argument(
        "--ideal-cache",
        action="store_true",
        help="Use the old, optimistic cache model (a hit at any common prefix) instead of the checkpoint-only one.",
    )
    parser.add_argument(
        "--calibrate-baseline",
        default=None,
        help="Title of the REAL baseline session of the same run: scales the cache model to its tokens and dollars.",
    )
    parser.add_argument(
        "--compare-live",
        default=None,
        help="Run directory with A-graph-evidence.jsonl: compare the replayed removals with the live ones.",
    )
    args = parser.parse_args(argv)

    matcher = _embedding_matcher() if args.matcher == "embedding" else _keyword_matcher()
    config = GraphConfig(
        enabled=True,
        unique_titles=args.unique_titles,
        mode=args.mode,
        keep_recent=args.keep_recent,
        keep_dialogue=args.keep_dialogue,
    )
    graph = OpenCodeGraph(config, matcher_factory=lambda: matcher)
    with copied_store() as connection:
        session_id = find_session(connection, args.title)
        if session_id is None:
            print(f"no session titled {args.title!r}", file=sys.stderr)
            return 2
        wire = wire_transcript(connection, session_id)
        real_cache = real_output = None
        baseline_views = None
        if args.calibrate_baseline:
            base_id = find_session(connection, args.calibrate_baseline)
            if base_id is None:
                print(f"no session titled {args.calibrate_baseline!r}", file=sys.stderr)
                return 2
            baseline_views = [
                (tuple(m["id"] for m in prefix), tuple(estimate_tokens([m]) for m in prefix))
                for prefix in request_prefixes(wire_transcript(connection, base_id))
            ]
            real_cache = real_output = 0
            for (data,) in connection.execute("select data from message where session_id = ?", (base_id,)):
                message = json.loads(data)
                if message.get("role") == "assistant":
                    tokens = message.get("tokens") or {}
                    cache = tokens.get("cache") or {}
                    real_cache += int(cache.get("read") or 0) + int(cache.get("write") or 0)
                    real_output += int(tokens.get("output") or 0)
    if args.inplace is not None:
        summary = replay_inplace(wire, graph=graph, keep_recent=args.inplace)
    elif args.sticky:
        with sticky_collapse():
            summary = replay_wire(wire, graph=graph)
    else:
        summary = replay_wire(wire, graph=graph)
    body = summarize(summary)
    body["variant"] = (
        f"inplace(keep={args.inplace})"
        if args.inplace is not None
        else f"binding-inplace(keep={args.keep_recent},dialogue={args.keep_dialogue})"
        if args.mode == "inplace"
        else "sticky"
        if args.sticky
        else "faithful"
    ) + ("+unique_titles" if args.unique_titles else "")
    if real_cache:
        body["calibrated"] = calibrated_dollars(
            summary,
            real_cache_tokens=real_cache,
            real_output_tokens=real_output,
            checkpoints=not args.ideal_cache,
            baseline_views=baseline_views,
        )
    if args.compare_live:
        live = [
            json.loads(line)
            for line in open(Path(args.compare_live) / "A-graph-evidence.jsonl", encoding="utf-8")
            if line.strip()
        ]
        body["vs_live"] = compare_with_live(summary, live)
    body["cache_model"] = cache_model(summary, checkpoints=not args.ideal_cache)
    if not args.series:
        body.pop("series")
    body["status"] = graph.status()["totals"]
    print(json.dumps(body, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
