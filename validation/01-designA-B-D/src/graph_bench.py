"""Offline retrieval benchmark for the Context Graph's note.

The live harness answers "what did the run cost and did it answer right". It cannot answer
**why** a Card was reached, because a live replay changes the tool path at the same time as it
changes the ranking, and one replay of eighteen turns has no power to separate the two. So the
question the graph is built on — is the note picking the right Cards? — has never been measured.

This measures it directly, and offline. It replays a recorded session's messages, rebuilds the
graph exactly as the writing half does, and asks one question per scored turn: **are the Cards
that hold the answer's own numbers ranked where the call would address them?**

The ground truth costs nothing to state, which is why it is trustworthy: it is the literals
``accuracy.CHECKS`` already asserts. A Card is required for turn T when it belongs to an earlier
turn and its messages carry a literal that T's critical checks demand. Nothing here is hand
labelled, so nothing here can be labelled to favour a configuration.

What each arm isolates:

- ``embedding`` against ``lexical`` and ``random`` says whether the embedding call earns its place.
  A lexical arm is the honest control: it is free, deterministic and offline, so if it ranks as
  well, the round trip is buying latency and nothing else.
- ``recency`` is the floor. Its note is flat, so the ranking degenerates to turn order — the
  answer a sliding window gives. A matcher that cannot beat it is not a matcher.
- ``no-links`` and ``no-propagation`` say whether the graph is a graph. If removing every edge
  does not move recall, the links are decoration and ``state.vectors`` exists to feed decoration.
- ``rerank`` says whether a second stage earns 2.6s.

Run::

    .venv/bin/python -m src.graph_bench            # every offline arm
    .venv/bin/python -m src.graph_bench --rerank   # adds the rerank arm

Only the embedding and rerank arms reach the network, and both are cached by the shared embedder,
so the whole sweep costs a handful of calls rather than a run.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from strands.vended_plugins.context_graph.cards import rebuild_into
from strands.vended_plugins.context_graph.scoring import compute_notes, select, titles_in_turn_order
from strands.vended_plugins.context_graph.state import Card, _GraphState

from . import accuracy, scenario
from .config import RESULTS_DIR, SESSIONS_DIR

# --- Corpus ------------------------------------------------------------------------

CARD_CONFIG = {"description_tokens": 100, "tags_per_card": 5, "rarity_weight": 0.70}
"""The shipped derivation defaults. Stated here so a benchmark arm cannot silently retune them."""

LINK_THRESHOLD = 0.50
"""The shipped default. The ``no-links`` arm raises it out of range instead of editing the code."""


def newest_session(root: Path = SESSIONS_DIR) -> Path:
    """Return the agent directory of the most recently written recorded session.

    Args:
        root: Directory the harness keeps its sessions under.

    Returns:
        The ``agents/<id>`` directory holding ``messages/`` and ``agent.json``.

    Raises:
        FileNotFoundError: If no recorded session carries messages.
    """
    candidates = sorted(root.glob("session_*/agents/*/messages"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"no recorded session under {root}. Produce one with: ./run.sh "
            "--configs gr-long-persist --total-turns 24 --resume-at 12 --tag corpus"
        )
    return candidates[-1].parent


def load_messages(agent_dir: Path) -> list[dict[str, Any]]:
    """Load a session's messages in wire order, keeping ``tracking_id``.

    Args:
        agent_dir: The ``agents/<id>`` directory of a recorded session.

    Returns:
        The messages as the SDK holds them, ordered by their stored index.
    """
    paths = sorted(
        (agent_dir / "messages").glob("message_*.json"),
        key=lambda path: int(re.search(r"(\d+)", path.name).group(1)),  # type: ignore[union-attr]
    )
    return [json.loads(path.read_text(encoding="utf-8"))["message"] for path in paths]


# --- Ground truth ------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Fold accents, case and whitespace, the same way ``accuracy`` does."""
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return re.sub(r"\s+", " ", stripped).casefold()


def _card_text(card: Card, texts_by_id: dict[str, str]) -> str:
    """Concatenate the normalized text of every message a Card addresses."""
    return " ".join(texts_by_id.get(identity, "") for identity in (*card.dialogue_ids, *card.evidence_ids))


def _texts_by_id(messages: Sequence[dict[str, Any]]) -> dict[str, str]:
    """Bucket the normalized text of each message, ``toolResult`` content included."""
    buckets: dict[str, str] = {}
    for message in messages:
        identity = message.get("tracking_id")
        if not identity:
            continue
        buckets[identity] = _normalize(json.dumps(message.get("content", []), ensure_ascii=False))
    return buckets


def critical_literals(label: str) -> tuple[str, ...]:
    """Literals a turn's *critical* checks demand, as retrieval targets.

    Only ``all_of`` and ``any_of`` of critical checks: a ``none_of`` names a wrong answer, which is
    not something a Card can hold, and a non-critical check is not what the question was asking for.

    Args:
        label: A scored turn label, as ``accuracy.CHECKS`` keys them.

    Returns:
        The literals, deduplicated. Empty for an unscored turn.
    """
    literals: list[str] = []
    for check in accuracy.CHECKS.get(label, ()):
        if not check.critical:
            continue
        literals.extend(check.all_of)
        literals.extend(check.any_of)
    # A retrieval target has to be specific enough to locate one Card rather than half the graph.
    # A figure or a long identifier is; an institution name that every account listing repeats is
    # not, and admitting it would credit an arm for reaching a Card that carries nothing unique.
    return tuple(
        dict.fromkeys(
            value for value in literals if any(char.isdigit() for char in value) or len(value) >= 12
        )
    )


_PROBE_LITERAL = re.compile(r"[A-Z][A-Z0-9_]{7,}|\d[\d.,]{3,}\d")
"""A figure or a long screaming-case identifier: the two shapes a fact is addressable by here.

Anything shorter matches half the corpus, and a target that matches half the corpus credits an arm
for reaching a Card that carries nothing unique.
"""

_PROBES_PER_SIDE = 4
"""Probes drawn per Card on each side of the Description split.

Several per Card rather than one, because a single literal makes the sample the size of the graph,
and a graph of two dozen Cards cannot separate two rankings. Each probe costs one cached embedding.
"""

_PROBE_PROMPT = "Sobre {literal}: em que ponto da conversa isso apareceu e o que foi levantado ali?"
"""One phrasing for every probe, so the query text is not a variable between arms.

Portuguese because the corpus is, and the default matcher is multilingual: a question in another
language would measure the model's cross-lingual behavior instead of the graph's ranking.
"""


@dataclass(frozen=True)
class Question:
    """One retrieval question: a turn's prompt and the earlier Cards holding its answer."""

    label: str
    turn: int
    prompt: str
    required: frozenset[str]
    """Titles of Cards from earlier turns whose messages carry a critical literal of this turn."""


def _labels_by_title(state: _GraphState, script: Sequence[Any]) -> dict[str, str]:
    """Map each Card Title to the scored turn whose prompt it is the prefix of.

    A Title is a *literal* prefix of its turn's user message, so the mapping is exact containment
    and never a similarity: a fuzzy match here silently attributes one turn's expectations to
    another Card, which is a ground truth that looks fine and is not. Only the scored turns are
    offered, because a filler turn has no expectation to attribute.

    Args:
        state: The rebuilt graph.
        script: The scored scenario turns.

    Returns:
        Title to label, for the Titles that matched exactly one turn.
    """
    matches: dict[str, list[str]] = {}
    for title in titles_in_turn_order(state):
        normalized = _normalize(title)
        if len(normalized) < 20:
            continue
        matches[title] = [turn.label for turn in script if _normalize(turn.prompt).startswith(normalized)]
    return {title: labels[0] for title, labels in matches.items() if len(labels) == 1}


def build_questions(state: _GraphState, messages: Sequence[dict[str, Any]], script: Sequence[Any]) -> list[Question]:
    """Derive the retrieval questions from the graph and the scenario's own expectations.

    A turn contributes a question only when an *earlier* Card holds one of its critical literals.
    A turn whose answer exists nowhere but in a tool call it is about to make is not a retrieval
    question, and scoring it would measure the tool path rather than the note.

    Args:
        state: The rebuilt graph.
        messages: The conversation the graph was derived from.
        script: The scenario turns, for the label of each prompt.

    Returns:
        One question per scored turn that depends on earlier context.
    """
    texts_by_id = _texts_by_id(messages)
    ordered = titles_in_turn_order(state)
    card_text = {title: _card_text(state.cards[title], texts_by_id) for title in ordered}

    label_of = _labels_by_title(state, script)

    questions: list[Question] = []
    for title in ordered:
        label = label_of.get(title)
        if not label:
            continue
        literals = critical_literals(label)
        if not literals:
            continue
        card = state.cards[title]
        required = {
            other
            for other in ordered
            if state.cards[other].turn < card.turn
            and any(_normalize(literal) in card_text[other] for literal in literals)
        }
        if required:
            questions.append(
                Question(label=label, turn=card.turn, prompt=title, required=frozenset(required))
            )
    return questions


def build_probes(state: _GraphState, messages: Sequence[dict[str, Any]]) -> list[Question]:
    """Derive one retrieval question per Card from a figure only that Card carries.

    The checks-derived set is the faithful one — it asks what the scenario actually asked — but it
    yields a handful of questions, which is not enough to separate two rankings. This set trades
    fidelity for sample size, and it is unambiguous by construction: the literal occurs in exactly
    one Card, so the required Card is known rather than inferred.

    What it measures is exact-fact retrieval, which is the case the graph's Tags exist for and the
    case an embedding is weakest at. Read alongside the checks-derived set rather than instead of it:
    an arm that wins here and loses there is good at identifiers and bad at questions.

    Args:
        state: The rebuilt graph.
        messages: The conversation the graph was derived from.

    Returns:
        One question per Card that carries a literal unique to it.
    """
    texts_by_id = _texts_by_id(messages)
    ordered = titles_in_turn_order(state)
    card_text = {title: _card_text(state.cards[title], texts_by_id) for title in ordered}

    owners: dict[str, set[str]] = {}
    for title, text in card_text.items():
        for literal in set(_PROBE_LITERAL.findall(text)):
            owners.setdefault(literal, set()).add(title)

    # Every probe is asked from the end of the conversation, so all Cards are visible and each
    # question is ranked against the same graph. Asking from just after the target Card instead
    # would hand the recency arm the answer, and the case worth measuring is the other one: a
    # question about something the conversation left behind.
    asked_at = max((card.turn for card in state.cards.values()), default=0) + 1

    probes: list[Question] = []
    for title in ordered:
        card = state.cards[title]
        unique = sorted(
            literal
            for literal, holders in owners.items()
            if holders == {title} and _normalize(literal) not in _normalize(title)
        )
        if not unique:
            continue

        # Split by whether the literal survived into the Description, because the Description is
        # what the matcher scores. A literal the Description dropped is a fact the note has no way
        # to see, however good the embedding is — so keeping the two apart is what separates "the
        # matcher cannot rank" from "the matcher was never shown the fact".
        described = _normalize(card.description)
        taken = {True: 0, False: 0}
        for literal in unique:
            visible = _normalize(literal) in described
            if taken[visible] >= _PROBES_PER_SIDE:
                continue
            taken[visible] += 1
            probes.append(
                Question(
                    label=f"probe{'+desc' if visible else '-desc'}:{literal}",
                    turn=asked_at,
                    prompt=_PROBE_PROMPT.format(literal=literal),
                    required=frozenset({title}),
                )
            )
    return probes


# --- Matchers ----------------------------------------------------------------------


class LexicalMatcher:
    """Free control arm: IDF-weighted token overlap between the question and each Description.

    Deterministic, offline and instant. It is the arm the embedding has to beat to justify a round
    trip on the critical path — not ``random``, which anything beats.
    """

    name = "lexical"

    def __init__(self, corpus: Sequence[str] = ()) -> None:
        """Fit inverse document frequency over the Descriptions it will be asked about."""
        self._document_frequency: dict[str, int] = {}
        self._total = max(1, len(corpus))
        for text in corpus:
            for token in set(_tokens(text)):
                self._document_frequency[token] = self._document_frequency.get(token, 0) + 1

    def score(self, question: str, descriptions: Sequence[str]) -> list[float]:
        """Return one overlap score per description, in ``[0, 1]``."""
        wanted = set(_tokens(question))
        if not wanted:
            return [0.0] * len(descriptions)
        weights = {token: self._idf(token) for token in wanted}
        total = sum(weights.values()) or 1.0
        return [
            min(1.0, sum(weights[token] for token in wanted if token in set(_tokens(text))) / total)
            for text in descriptions
        ]

    def _idf(self, token: str) -> float:
        frequency = self._document_frequency.get(token, 0)
        return math.log((self._total + 1) / (frequency + 1)) + 1.0


class RandomMatcher:
    """Seeded noise. The arm that says what the metric reads when the ranking carries no signal."""

    name = "random"

    def score(self, question: str, descriptions: Sequence[str]) -> list[float]:
        """Return a deterministic pseudo-random score per description."""
        return [
            ((hash((question, index, text[:32])) % 10_000) / 10_000.0) for index, text in enumerate(descriptions)
        ]


class RecencyMatcher:
    """The floor: score rises with position, so the ranking is newest first and ignores the question.

    This is the answer a sliding window gives, and it is the arm a matcher has to beat to exist. It
    must not be confused with a *flat* score: with every note equal, the note ordering falls through
    to its tie-break, which is the turn ordinal **ascending** — oldest first. That is not recency, it
    is the opposite of it, and on a script whose recall turns point back at the opening it scores
    deceptively well.
    """

    name = "recency"

    def score(self, question: str, descriptions: Sequence[str]) -> list[float]:
        """Return a score rising with position, so the last description scores highest."""
        count = len(descriptions)
        if count == 1:
            return [1.0]
        return [index / (count - 1) for index in range(count)]


def _tokens(text: str) -> list[str]:
    """Words and numbers of ``text``, normalized, with the very short ones dropped."""
    return [token for token in re.findall(r"[\w.,+-]{3,}", _normalize(text))]


# --- Scoring -----------------------------------------------------------------------


@dataclass
class ArmResult:
    """What one configuration scored over every question."""

    name: str
    ranks: list[float] = field(default_factory=list)
    """Rank of each required Card in the note ordering, 1 being best."""
    recall_at_selection: list[float] = field(default_factory=list)
    recall_note_only: list[float] = field(default_factory=list)
    """Recall with the recency window disabled: what the note alone reaches, plus one hop.

    The configured window is what ships, but on a short conversation it addresses most of the graph
    on its own, so it masks the ranking entirely. With the window off, the figure is the note's.
    """
    recall_at_5: list[float] = field(default_factory=list)
    separations: list[float] = field(default_factory=list)
    """Required note minus the median note: how far the signal sits above the crowd."""

    def summary(self) -> dict[str, Any]:
        """Reduce the per-question samples to the comparable figures."""
        return {
            "arm": self.name,
            "questions": len(self.recall_at_selection),
            "required_cards": len(self.ranks),
            "mrr": round(statistics.mean([1.0 / rank for rank in self.ranks]), 4) if self.ranks else 0.0,
            "median_rank": round(statistics.median(self.ranks), 2) if self.ranks else 0.0,
            "recall_at_5": round(statistics.mean(self.recall_at_5), 4) if self.recall_at_5 else 0.0,
            "recall_at_selection": round(statistics.mean(self.recall_at_selection), 4)
            if self.recall_at_selection
            else 0.0,
            "recall_note_only": round(statistics.mean(self.recall_note_only), 4)
            if self.recall_note_only
            else 0.0,
            "note_separation": round(statistics.mean(self.separations), 4) if self.separations else 0.0,
        }


def score_arm(
    name: str,
    state: _GraphState,
    questions: Sequence[Question],
    matcher: Any,
    *,
    recent_cards: int,
    select_top_k: int,
    reranker: Any | None = None,
    select_state: _GraphState | None = None,
) -> ArmResult:
    """Rank every question under one matcher and record where the required Cards landed.

    The graph is truncated to the turns that existed when the question was asked, so a Card cannot
    be ranked against Cards from the future. That truncation is the whole reason this is a
    benchmark and not a replay of the final state.

    Args:
        name: Arm name for the report.
        state: The full rebuilt graph.
        questions: The retrieval questions.
        matcher: Object exposing ``score(question, descriptions)``.
        recent_cards: Recency window the selection would use.
        select_top_k: How many Cards the note adds beyond the window.
        reranker: Optional second stage, applied to the note's candidates.
        select_state: Graph the selection hop walks, when it differs from the one the note was
            computed over. That difference is the arm which asks whether an edge can earn its place
            by widening what the call *reaches* without touching what the note *ranks*.

    Returns:
        The arm's samples.
    """
    result = ArmResult(name=name)

    for question in questions:
        visible = _truncated(state, question.turn)
        walked = _truncated(select_state, question.turn) if select_state is not None else visible
        if len(visible.cards) < 2:
            continue

        notes = compute_notes(visible, question.prompt, matcher)
        if not notes:
            continue
        if reranker is not None:
            notes = _reranked(visible, notes, question.prompt, reranker, select_top_k)

        ordered = sorted(notes, key=lambda title: (-notes[title], visible.cards[title].turn, title))
        position = {title: index + 1 for index, title in enumerate(ordered)}
        present = [title for title in question.required if title in visible.cards]
        if not present:
            continue

        selected = select(notes, walked, recent_cards=recent_cards, select_top_k=select_top_k)
        note_only = select(notes, walked, recent_cards=0, select_top_k=select_top_k)
        median_note = statistics.median(notes.values())

        for title in present:
            result.ranks.append(position[title])
            result.separations.append(notes[title] - median_note)
        result.recall_at_selection.append(sum(1 for t in present if t in selected) / len(present))
        result.recall_note_only.append(sum(1 for t in present if t in note_only) / len(present))
        result.recall_at_5.append(sum(1 for t in present if position[t] <= 5) / len(present))

    return result


def _truncated(state: _GraphState, turn: int) -> _GraphState:
    """Return the graph as it stood before ``turn``: later Cards and their edges removed."""
    kept = {title: card for title, card in state.cards.items() if card.turn < turn}
    links = {
        title: [link for link in edges if link.kind == "tool" or link.target in kept]
        for title, edges in state.links.items()
        if title in kept
    }
    return _GraphState(cards=kept, links=links, turn=turn, vectors=dict(state.vectors))


def _reranked(
    state: _GraphState,
    notes: dict[str, float],
    question: str,
    reranker: Any,
    select_top_k: int,
) -> dict[str, float]:
    """Renumber the note's top candidates in the reranker's order, as the plugin does."""
    from strands.vended_plugins.context_graph.ranking import rerank

    candidates = sorted(notes, key=lambda title: (-notes[title], state.cards[title].turn, title))
    candidates = candidates[: max(2, select_top_k * 2)]
    ordered = rerank(question, candidates, [state.cards[title].description for title in candidates], reranker)
    highest = max(notes.values(), default=0.0)
    renumbered = dict(notes)
    for position, title in enumerate(ordered):
        renumbered[title] = highest + len(ordered) - position
    return renumbered


# --- Arms --------------------------------------------------------------------------


def build_graph(messages: Sequence[dict[str, Any]], *, link_threshold: float, matcher: Any | None) -> _GraphState:
    """Rebuild the graph by scan, seeding the vector cache when a matcher can fill it.

    The similarity link is measured from ``state.vectors`` and never from a remote call, so a scan
    over an empty cache creates no ``similar`` edge at all. Seeding first is what lets an arm
    measure the links rather than measure their absence.

    Args:
        messages: The conversation.
        link_threshold: Similarity at or above which two Cards link. Out of range disables the edge.
        matcher: Source of description vectors, or ``None`` to leave the cache empty.

    Returns:
        The rebuilt graph.
    """
    state = _GraphState()
    rebuild_into(state, list(messages), link_threshold=link_threshold, **CARD_CONFIG)

    vectors = getattr(matcher, "vectors", None) if matcher is not None else None
    if callable(vectors):
        titles = titles_in_turn_order(state)
        descriptions = [state.cards[title].description for title in titles]
        computed = vectors(descriptions)
        if len(computed) == len(titles):
            for title, description, vector in zip(titles, descriptions, computed, strict=True):
                state.vectors[title] = (description, tuple(vector))
            # Re-scanned so the edges are measured against the cache we just filled.
            rebuild_into(state, list(messages), link_threshold=link_threshold, **CARD_CONFIG)
    return state


def run(
    agent_dir: Path,
    *,
    recent_cards: int,
    select_top_k: int,
    with_rerank: bool,
) -> dict[str, Any]:
    """Score every arm over one recorded session.

    Args:
        agent_dir: The recorded session's agent directory.
        recent_cards: Recency window the selection would use.
        select_top_k: How many Cards the note adds beyond the window.
        with_rerank: Whether to add the rerank arm, which reaches the network.

    Returns:
        The report payload.
    """
    messages = load_messages(agent_dir)
    embedder = _embedding_matcher()

    linked = build_graph(messages, link_threshold=LINK_THRESHOLD, matcher=embedder)
    # 1.01 is unreachable for a cosine similarity, so no edge is ever created: the arm measures the
    # graph without its similarity links rather than a graph with a different threshold.
    unlinked = build_graph(messages, link_threshold=1.01, matcher=None)
    # The measured p90 of the pairwise distribution. At the shipped 0.50 the edge fires on ~40% of
    # all pairs, which is a graph that says everything is related to everything.
    sparse = build_graph(messages, link_threshold=0.65, matcher=embedder)
    # No edge of any kind, so the note is the first pass alone. Without this arm, "the embedding
    # ranks badly" and "propagation is drowning a ranking that was fine" read identically.
    bare = _GraphState(cards=dict(linked.cards), links={}, turn=linked.turn, vectors=dict(linked.vectors))

    # Only the scored turns: a filler turn has no expectation, so it can contribute no question, and
    # padding the script to guess the recorded length is a guess this does not have to make.
    script = list(scenario.turns())
    questions = build_questions(linked, messages, script)
    probes = build_probes(linked, messages)

    descriptions = [linked.cards[title].description for title in titles_in_turn_order(linked)]
    lexical = LexicalMatcher(descriptions)

    def sweep(name: str, asked: Sequence[Question]) -> list[dict[str, Any]]:
        """Score every arm over one question family."""
        graded: list[tuple[str, _GraphState, Any, Any, _GraphState | None]] = [
            ("embedding", linked, embedder, None, None),
            ("lexical", linked, lexical, None, None),
            ("random", linked, RandomMatcher(), None, None),
            ("recency", linked, RecencyMatcher(), None, None),
            ("embedding, no similarity links", unlinked, embedder, None, None),
            ("embedding, sparse links (0.65)", sparse, embedder, None, None),
            ("embedding, no propagation at all", bare, embedder, None, None),
            # The candidate correction: the similarity edge stops feeding the note and is walked only
            # by the selection hop, which is the job the design gives it — "these two pull each
            # other" is a question about what the call reaches, not about what the note ranks.
            ("embedding, similar for the hop only", unlinked, embedder, None, linked),
            ("embedding, sparse similar for the hop only", unlinked, embedder, None, sparse),
        ]
        if with_rerank:
            graded.append(("embedding + rerank", linked, embedder, _reranker(), None))
            graded.append(("embedding + rerank, similar for the hop only", unlinked, embedder, _reranker(), linked))

        results = []
        for arm, graph, matcher, reranker, walked in graded:
            outcome = score_arm(
                arm,
                graph,
                asked,
                matcher,
                recent_cards=recent_cards,
                select_top_k=select_top_k,
                reranker=reranker,
                select_state=walked,
            )
            summary = outcome.summary()
            summary["family"] = name
            results.append(summary)
        return results

    in_description = [probe for probe in probes if probe.label.startswith("probe+desc")]
    out_of_description = [probe for probe in probes if probe.label.startswith("probe-desc")]

    arms = (
        sweep("checks", questions)
        + sweep("probes, fact in Description", in_description)
        + sweep("probes, fact only in messages", out_of_description)
    )

    return {
        "session": str(agent_dir),
        "messages": len(messages),
        "cards": len(linked.cards),
        "links_with_vectors": sum(len(edges) for edges in linked.links.values()),
        "similar_links": sum(
            1 for edges in linked.links.values() for link in edges if link.kind == "similar"
        ),
        "links_without_vectors": sum(len(edges) for edges in unlinked.links.values()),
        "similar_links_sparse": sum(
            1 for edges in sparse.links.values() for link in edges if link.kind == "similar"
        ),
        "vectors_cached": len(linked.vectors),
        "selection": {"recent_cards": recent_cards, "select_top_k": select_top_k},
        "questions": [
            {"label": question.label, "turn": question.turn, "required": sorted(question.required)}
            for question in questions
        ],
        "probes": [
            {"label": question.label, "turn": question.turn, "required": sorted(question.required)}
            for question in probes
        ],
        "arms": arms,
    }


def _embedding_matcher() -> Any:
    """Build the shipped default matcher on the harness's own session."""
    from strands.vended_plugins.context_graph import EmbeddingSimilarityMatcher

    from .runner import boto_session

    return EmbeddingSimilarityMatcher(boto_session=boto_session())


def _reranker() -> Any:
    """Build the reranker the harness uses for relevance filtering."""
    from strands._context_manager.methods.reranker import BedrockReranker

    from .config import RERANK_MODEL_ID
    from .runner import boto_session

    return BedrockReranker(model_id=RERANK_MODEL_ID, boto_session=boto_session())


# --- Report ------------------------------------------------------------------------


def render(payload: dict[str, Any]) -> str:
    """Render the benchmark as a readable table."""
    lines = [
        "# Context Graph — offline retrieval benchmark",
        "",
        f"Session: `{payload['session']}`",
        f"{payload['messages']} messages, {payload['cards']} Cards, "
        f"{len(payload['questions'])} checks-derived questions, {len(payload['probes'])} probes.",
        f"Links: {payload['links_with_vectors']} with the vector cache filled "
        f"({payload['similar_links']} of them `similar` at 0.50, "
        f"{payload['similar_links_sparse']} at 0.65), "
        f"{payload['links_without_vectors']} with the cache empty.",
        f"Selection under test: {payload['selection']['recent_cards']} recent + "
        f"top {payload['selection']['select_top_k']}.",
        "",
        "A required Card is one from an earlier turn whose messages carry a literal the turn's",
        "critical checks demand. Ranks are over the note ordering; 1 is best.",
        "",
        "| Family | Arm | N | MRR | Median rank | Recall@5 | Recall@selection | Recall, note only |"
        " Note separation |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in payload["arms"]:
        lines.append(
            f"| {arm['family']} | {arm['arm']} | {arm['required_cards']} | {arm['mrr']:.3f} | "
            f"{arm['median_rank']:.1f} | {arm['recall_at_5'] * 100:.1f}% | "
            f"{arm['recall_at_selection'] * 100:.1f}% | {arm['recall_note_only'] * 100:.1f}% | "
            f"{arm['note_separation']:+.3f} |"
        )
    lines += [
        "",
        "## Questions",
        "",
        "| Turn | Label | Required Cards |",
        "|---:|---|---:|",
    ]
    for question in payload["questions"]:
        lines.append(f"| {question['turn']} | {question['label']} | {len(question['required'])} |")
    return "\n".join(lines)


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(prog="src.graph_bench", description=__doc__)
    parser.add_argument("--session", metavar="DIR", help="recorded session agent directory")
    parser.add_argument("--recent-cards", type=int, default=10)
    parser.add_argument("--select-top-k", type=int, default=5)
    parser.add_argument("--rerank", action="store_true", help="add the rerank arm (reaches the network)")
    parser.add_argument("--tag", default="graph-bench")
    args = parser.parse_args()

    agent_dir = Path(args.session) if args.session else newest_session()
    payload = run(
        agent_dir,
        recent_cards=args.recent_cards,
        select_top_k=args.select_top_k,
        with_rerank=args.rerank,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{args.tag}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    report = render(payload)
    (RESULTS_DIR / f"{args.tag}.md").write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nraw:    {RESULTS_DIR / f'{args.tag}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
