"""The Gate D contract suite: the five scenarios, a full pass and the absence of durable effects, request by request.

Each scenario runs the real binding (:class:`context_core.opencode_graph.OpenCodeGraph`, over the real graph core) on
transcripts in the wire shape the plugin sends. Nothing here is a model of the binding: the only seam taken from outside is
the similarity matcher, and the suite uses a deterministic bag-of-words one so that it needs no network and so that two runs
of the same transcript must agree to the byte (the resume scenario depends on it). What the embedding model decides is
measured live, in the campaign; what the *binding* does with that decision is what is tested here.

A scenario reports two things: evidence records (one per request, :class:`~.contracts.GateDRequestEvidence`) and failures, the
scenario-level facts a single record cannot carry (a Card left without a source after a compaction, a restored turn that did
not come back). A failure is a ``no_go`` reason in the evaluator.

The five scenarios of FR5.2-FR5.4:

``tool_loop``  the model calls tools inside one user turn; the choice is frozen for the turn and the closed turns project the
               same way on every call of it.
``subagent``   two sessions share one bridge; each keeps a graph of its own and neither moves the other's.
``compact``    OpenCode compacts the session: earlier messages vanish. No crash, no Card without a source, no removal of a
               message that is not there.
``resume``     a fresh bridge given the same transcript derives the same Card ids, sources, removal and block.
``recovery``   what the graph collapsed can be asked back with ``expand_card``. Under the reference policy the turn is then in
               the next request; under ``inplace`` its content is in the answer and the request does not move at all.

Both policies of the binding run through the same scenarios (``GraphConfig.mode``). What differs is the shape of an answer: the
reference policy removes messages and appends one block, ``inplace`` replaces the assistant side of each old turn by one
summary where it was (``collapses``), so the helpers below read both.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from context_core.graph.matcher import SimilarityMatcher
from context_core.opencode_graph import GraphConfig, OpenCodeGraph, card_id
from opencode_gate_a.canonical import digest

from .contracts import GateDRequestEvidence, absence_digest
from .recorded import request_prefixes

__all__ = ["JaccardMatcher", "SuiteResult", "model_input", "run_suite", "synthetic_wire"]


class JaccardMatcher:
    """Deterministic, offline similarity: Jaccard over lower-cased words longer than three letters."""

    def score(self, question: str, descriptions: Sequence[str]) -> list[float]:
        asked = {word for word in question.lower().split() if len(word) > 3}
        scores = []
        for description in descriptions:
            words = {word for word in description.lower().split() if len(word) > 3}
            scores.append(len(asked & words) / len(asked | words) if asked | words else 0.0)
        return scores


def synthetic_wire(turns: int = 8, *, tool_steps: int = 2, prefix: str = "") -> list[dict[str, Any]]:
    """A conversation of ``turns`` user turns, each with ``tool_steps`` tool-calling steps and a final answer."""
    topics = (
        "alpha",
        "bravo",
        "charlie",
        "delta",
        "echo",
        "foxtrot",
        "golf",
        "hotel",
        "india",
        "juliet",
        "kilo",
        "lima",
    )
    wire: list[dict[str, Any]] = []
    for turn in range(turns):
        topic = topics[turn % len(topics)] + ("" if turn < len(topics) else f"{turn}")
        wire.append(
            {
                "id": f"{prefix}u{turn:03d}",
                "role": "user",
                "parts": [{"type": "text", "text": f"{topic} ledger: what is the {topic} total balance?"}],
            }
        )
        for step in range(tool_steps):
            wire.append(
                {
                    "id": f"{prefix}a{turn:03d}s{step}",
                    "role": "assistant",
                    "parts": [
                        {"type": "text", "text": f"reading the {topic} ledger part {step}"},
                        {
                            "type": "tool",
                            "tool": "read_ledger",
                            "callID": f"{prefix}call{turn:03d}s{step}",
                            "status": "completed",
                            "input": {"topic": topic, "part": step},
                            "output": f"{topic} ledger part {step}\ntotal: {1000 + turn}.{step}0\nitems: {turn + 10 * step}",
                        },
                    ],
                }
            )
        wire.append(
            {
                "id": f"{prefix}f{turn:03d}",
                "role": "assistant",
                "parts": [
                    {"type": "text", "text": f"The {topic} total balance is {1000 + turn}.00 across {turn + 10} items."}
                ],
            }
        )
    return wire


def model_input(prefix: Sequence[dict[str, Any]], response: dict[str, Any]) -> list[dict[str, Any]]:
    """The messages a model would be sent for ``prefix`` once the plugin has applied ``response``."""
    if response.get("operation") != "projected":
        return list(prefix)
    removed = set(response.get("removed_ids", ()))
    summaries = {item["at_id"]: item["text"] for item in response.get("collapses") or ()}
    sent: list[dict[str, Any]] = []
    for message in prefix:
        if message["id"] not in removed:
            sent.append(message)
        elif message["id"] in summaries:
            parts = [{"type": "text", "text": summaries[message["id"]], "synthetic": True}]
            sent.append({"id": f"graph-collapse-{message['id']}", "role": "assistant", "parts": parts})
    block = response.get("final_block")
    if block:
        sent.append(
            {"id": "graph-block", "role": "user", "parts": [{"type": "text", "text": block, "synthetic": True}]}
        )
    return sent


def _titles_in(response: dict[str, Any]) -> list[str]:
    """The Titles an answer shows the model: the lines of the reference block, or the Card of each in-place summary."""
    titles = [line[2:] for line in (response.get("final_block") or "").splitlines() if line.startswith("- ")]
    return titles + [str(item["title"]) for item in response.get("collapses") or () if item.get("title")]


def _shape(response: dict[str, Any]) -> tuple[Any, Any, Any]:
    """Everything in an answer that changes what the model is sent. Two answers with the same shape send the same request."""
    return response.get("removed_ids"), response.get("final_block"), response.get("collapses")


def _probe(message: dict[str, Any] | None) -> str | None:
    """A short piece of a message's content that a recall of it must contain: its first line of text or tool output."""
    for part in (message or {}).get("parts", ()):
        text = part.get("text") if part.get("type") == "text" else part.get("output")
        if isinstance(text, str) and text.strip():
            return text.strip().splitlines()[0][:60]
    return None


def compact(prefix: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """What OpenCode keeps after a compaction: one summary message, then the turn in progress."""
    last_user = max(index for index, message in enumerate(prefix) if message["role"] == "user")
    summary = {
        "id": "msg_summary",
        "role": "assistant",
        "parts": [{"type": "text", "text": "Summary of the earlier conversation: the user reviewed several ledgers."}],
    }
    return [summary, *prefix[last_user:]]


def _rekeyed(wire: Sequence[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    """The same conversation under different message and call ids, as a child session would hold it."""
    clone = copy.deepcopy(list(wire))
    for message in clone:
        message["id"] = prefix + message["id"]
        for part in message["parts"]:
            if part.get("type") == "tool":
                part["callID"] = prefix + part["callID"]
    return clone


@dataclass
class SuiteResult:
    """Everything the suite produced for the evaluator."""

    requests: list[GateDRequestEvidence] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)


class _Observer:
    """Runs requests through a graph and turns each one into an evidence record."""

    def __init__(
        self, result: SuiteResult, *, gate_a_evidence_id: str, config: GraphConfig, matcher: SimilarityMatcher
    ) -> None:
        self.result = result
        self.config = config
        self.matcher = matcher
        self.gate_a_evidence_id = gate_a_evidence_id
        self._sequence = 0
        # The same binding with the projection switched off: no score reaches the threshold in the reference policy, and the
        # in-place policy keeps every closed turn whole.
        self._full_pass = OpenCodeGraph(
            replace(config, expand_threshold=0.0, collapse_floor=0.0, keep_recent=10**6),
            matcher_factory=lambda: matcher,
        )

    def graph(self) -> OpenCodeGraph:
        """A fresh bridge-side graph, as a freshly started bridge process holds."""
        return OpenCodeGraph(self.config, matcher_factory=lambda: self.matcher)

    def fail(self, name: str) -> None:
        if name not in self.result.failures:
            self.result.failures.append(name)

    def _check_collapses(self, prefix: Sequence[dict[str, Any]], response: dict[str, Any]) -> None:
        """What an in-place answer must always be, whatever the transcript: summaries that replace assistant messages only."""
        collapses = response.get("collapses") or []
        if not collapses:
            return
        roles = {message["id"]: message["role"] for message in prefix}
        removed = list(response.get("removed_ids", ()))
        if any(roles.get(identity) != "assistant" for identity in removed):
            self.fail("inplace_removed_a_message_that_is_not_the_assistants")
        at_ids = [item["at_id"] for item in collapses]
        if len(set(at_ids)) != len(at_ids) or any(at_id not in removed for at_id in at_ids):
            self.fail("inplace_summary_does_not_replace_a_removed_message")
        if response.get("final_block"):
            self.fail("inplace_appended_a_block")

    def observe(
        self,
        graph: OpenCodeGraph,
        session_id: str,
        prefix: list[dict[str, Any]],
        scenarios: tuple[str, ...],
        *,
        delivered_tag: str | None = None,
        epoch: int = 0,
    ) -> dict[str, Any]:
        """Run one request, check it does not touch its input, and record the evidence."""
        snapshot = copy.deepcopy(prefix)
        before = digest(prefix)
        response = graph.project(session_id, prefix, scoring_allowed=True)
        after = digest(prefix)
        if prefix != snapshot:
            self.fail("binding_mutated_the_transcript")
        self._check_collapses(prefix, response)

        # The twin of the same request with the projection switched off: its model input must be the baseline's.
        twin = self._full_pass.project(f"{session_id}#full", prefix, scoring_allowed=True)
        full_pass_digest = digest(model_input(prefix, twin))
        baseline_digest = digest(prefix)

        stats = response.get("stats") or {}
        tags = list(scenarios) + ["full_pass"]
        if delivered_tag and stats.get("mode") == "delivered":
            tags.append(delivered_tag)
        cards = tuple((card["card_id"], tuple(card["sources"])) for card in graph.card_sources(session_id))
        if not cards:
            tags.append("pre_card")

        store_has_graph_messages = sum(1 for message in prefix if "zzzgraph" in message["id"])
        observation = {
            "graph_messages_in_store": store_has_graph_messages,
            "durable_writes": graph.status()["durable_writes"],
            "stored_transcript_changed": before != after,
        }
        self._sequence += 1
        self.result.requests.append(
            GateDRequestEvidence(
                evidence_id=f"gd-{self._sequence:05d}",
                gate_a_evidence_id=self.gate_a_evidence_id,
                request_id=f"{session_id}:{self._sequence}",
                stream_id=session_id,
                transcript_before_digest=before,
                transcript_after_digest=after,
                card_ids_and_sources=cards,
                scenarios=tuple(dict.fromkeys(tags)),
                full_pass_digest=full_pass_digest,
                baseline_model_input_digest=baseline_digest,
                canonical_equality=before == after,
                absence_digest=absence_digest(observation),
                projected=response.get("operation") == "projected",
                epoch=epoch,
            )
        )
        for tag in dict.fromkeys(tags):
            self.result.counts[tag] = self.result.counts.get(tag, 0) + 1
        return response


# -- the scenarios --------------------------------------------------------------------------------------


def _tool_loop(observer: _Observer, wire: list[dict[str, Any]], name: str) -> None:
    graph = observer.graph()
    first_in_turn: dict[str, dict[str, Any]] = {}
    for prefix in request_prefixes(wire):
        last_user = max(
            i for i, m in enumerate(prefix) if m["role"] == "user" and any(p["type"] == "text" for p in m["parts"])
        )
        turn = prefix[last_user]["id"]
        response = observer.observe(graph, f"{name}/tool_loop", prefix, ("tool_loop",), delivered_tag="tool_loop")
        if turn in first_in_turn:
            # The choice was frozen on the turn's first call: nothing already collapsed may move while the turn runs.
            if _shape(response) != _shape(first_in_turn[turn]):
                observer.fail("tool_loop_choice_moved_inside_a_turn")
        else:
            first_in_turn[turn] = response


def _subagent(observer: _Observer, wire: list[dict[str, Any]], name: str) -> None:
    shared = observer.graph()
    alone = observer.graph()
    parent_stream, child_stream = f"{name}/parent", f"{name}/child"
    child_wire = _rekeyed(wire[: max(8, len(wire) // 2)], "child_")
    parent_prefixes = request_prefixes(wire)
    child_prefixes = request_prefixes(child_wire)
    child_asked_back = False
    for index, prefix in enumerate(parent_prefixes):
        together = observer.observe(shared, parent_stream, prefix, ("subagent",))
        control = alone.project(f"{name}/control", prefix, scoring_allowed=True)
        if _shape(together) != _shape(control):
            observer.fail("subagent_state_leaked_between_sessions")
        if index < len(child_prefixes):
            answer = observer.observe(shared, child_stream, child_prefixes[index], ("subagent",))
            titles = _titles_in(answer)
            if titles and not child_asked_back:
                # The child asks for one of ITS turns back. The parent has a turn with the very same words, and it must
                # not come back: a retrieval belongs to the session that made it.
                shared.expand_card(child_stream, [titles[0]])
                child_asked_back = True
    sessions = shared.status()["sessions"]
    if not (sessions.get(parent_stream, {}).get("projections") and sessions.get(child_stream, {}).get("projections")):
        observer.fail("subagent_sessions_do_not_have_state_of_their_own")
    parent_ids = {card["card_id"] for card in shared.card_sources(parent_stream)}
    child_ids = {card["card_id"] for card in shared.card_sources(child_stream)}
    if parent_ids & child_ids:
        observer.fail("subagent_card_ids_collide")
    if {s for card in shared.card_sources(child_stream) for s in card["sources"]} & {m["id"] for m in wire}:
        observer.fail("subagent_card_points_into_the_parent")


def _compact(observer: _Observer, wire: list[dict[str, Any]], name: str) -> None:
    graph = observer.graph()
    stream = f"{name}/compact"
    prefixes = request_prefixes(wire)
    if len(prefixes) < 4:
        return
    long_prefix = prefixes[(3 * len(prefixes)) // 4]
    observer.observe(graph, stream, long_prefix, ("compact",))
    compacted = compact(long_prefix)
    present = {message["id"] for message in compacted}
    # The compaction is a defined break in Card identity: the Cards are rebuilt from what is left.
    response = observer.observe(graph, stream, compacted, ("compact",), epoch=1)
    if response.get("operation") == "error":
        observer.fail("compact_raised")
    if any(removed not in present for removed in response.get("removed_ids", ())):
        observer.fail("compact_removed_a_message_that_is_not_there")
    for card in graph.card_sources(stream):
        if not card["sources"] or any(source not in present for source in card["sources"]):
            observer.fail("compact_left_a_card_without_a_source")
    # The conversation goes on after the compaction.
    following = [message for message in wire if message["id"] not in {m["id"] for m in long_prefix}][:3]
    if following:
        observer.observe(graph, stream, [*compacted, *following], ("compact",), epoch=1)


def _resume(observer: _Observer, wire: list[dict[str, Any]], name: str) -> None:
    original = observer.graph()
    stream = f"{name}/resume"
    prefixes = request_prefixes(wire)
    if not prefixes:
        return
    middle = len(prefixes) // 2
    answers = [observer.observe(original, stream, prefix, ("resume",)) for prefix in prefixes[: middle + 1]]
    # A new process: nothing carried over except the host's transcript.
    restarted = observer.graph()
    again = observer.observe(restarted, stream, prefixes[middle], ("resume",))
    if _shape(again) != _shape(answers[middle]):
        observer.fail("resume_projects_differently")
    if restarted.card_sources(stream) != original.card_sources(stream):
        observer.fail("resume_changed_card_ids_or_sources")
    for prefix in prefixes[middle + 1 :]:
        before = original.project(stream, prefix, scoring_allowed=True)
        after = restarted.project(stream, prefix, scoring_allowed=True)
        if _shape(before) != _shape(after):
            observer.fail("resume_diverges_later")
            break


def _recovery(observer: _Observer, wire: list[dict[str, Any]], name: str) -> None:
    graph = observer.graph()
    stream = f"{name}/recovery"
    inplace = observer.config.mode == "inplace"
    by_id = {message["id"]: message for message in wire}
    for prefix in request_prefixes(wire):
        response = observer.observe(graph, stream, prefix, ("recovery",))
        if response.get("operation") != "projected":
            continue
        titles = _titles_in(response)
        if not titles:
            continue
        title = titles[0]
        sources = next((c["sources"] for c in graph.card_sources(stream) if c["card_id"] == card_id(stream, title)), [])
        if not sources:
            observer.fail("recovery_title_has_no_card")
            return
        gone = [source for source in sources if source in set(response["removed_ids"])]
        if not gone:
            continue
        answer = graph.expand_card(stream, [title])
        if inplace:
            # The content comes back as the answer, and the request does not move: putting the turn back where it was
            # would rewrite the cache from there on.
            probe = _probe(by_id.get(gone[0]))
            if "below, in full" not in answer or (probe is not None and probe not in answer):
                observer.fail("recovery_expand_card_did_not_return_the_turn")
                return
            again = observer.observe(graph, stream, prefix, ("recovery",))
            if _shape(again) != _shape(response):
                observer.fail("recovery_moved_the_request")
        else:
            if "arrives in full" not in answer:
                observer.fail("recovery_expand_card_refused")
                return
            restored = observer.observe(graph, stream, prefix, ("recovery",))
            if set(sources) & set(restored.get("removed_ids", ())):
                observer.fail("recovery_did_not_restore_the_turn")
        found = graph.find_context(stream, title, scoring_allowed=True)
        if title not in found:
            observer.fail("recovery_find_context_does_not_name_the_card")
        return
    observer.fail("recovery_never_had_anything_to_restore")


_SCENARIOS: tuple[Callable[[_Observer, list[dict[str, Any]], str], None], ...] = (
    _tool_loop,
    _subagent,
    _compact,
    _resume,
    _recovery,
)


def run_suite(
    wires: dict[str, list[dict[str, Any]]],
    *,
    gate_a_evidence_id: str,
    matcher: SimilarityMatcher | None = None,
    config: GraphConfig | None = None,
) -> SuiteResult:
    """Run every scenario over every transcript in ``wires`` (name -> wire transcript).

    Args:
        wires: The transcripts to run, recorded or synthetic. Each one is run through all five scenarios.
        gate_a_evidence_id: The Gate A evidence the records are filed under.
        matcher: The similarity matcher. Defaults to the deterministic offline one.
        config: The graph configuration. Defaults to the benchmark tuning, switched on.
    """
    result = SuiteResult()
    observer = _Observer(
        result,
        gate_a_evidence_id=gate_a_evidence_id,
        config=config or GraphConfig(enabled=True),
        matcher=matcher or JaccardMatcher(),
    )
    for name, wire in wires.items():
        for scenario in _SCENARIOS:
            scenario(observer, wire, name)
    return result
