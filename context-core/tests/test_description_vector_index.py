"""The description-vector index, and the ``similar`` Link that is measured from it and nothing else.

These tests exist because the suite passed with the index permanently empty. ``state.vectors`` had no writer anywhere in
the package and :meth:`EmbeddingSimilarityMatcher.vectors` had no caller outside its own tests, so
``_cached_similarity`` returned ``None`` for every pair and the ``similar`` Link never formed. Nothing failed: the three
structural Link kinds still form, the counters still report a Link total, and a graph with no similarity edge at all
reads as a working graph. A run confirmed it, reporting ``vectors_cached: 0`` on every arm that installed the plugin.

So the assertions below are deliberately about the WIRING rather than about the arithmetic: that scoring fills the
index, that the Link then forms end to end, and that the three paths which must NOT fill it still do not.

Ported from the Strands plugin's ``tests/test_description_vector_index.py``. The writer it covers moved from
``ContextGraph._cache_description_vectors`` to :func:`context_core.graph.projection._cache_description_vectors`, so the
turn is fired by calling :func:`~context_core.graph.projection.project` on a one-message conversation instead of by
constructing a ``BeforeInvocationEvent``. Every matcher is a stub: no embedding call, no network.
"""

from __future__ import annotations

from collections.abc import Sequence

from context_core.graph.projection import Thresholds, project
from context_core.graph.state import Card, _GraphState

DIMENSIONS = 4
"""Vector width. Small on purpose: these tests assert wiring, not embedding quality."""


class VectorMatcher:
    """A matcher that scores, publishes document vectors, and counts what it was asked for.

    Attributes:
        score_calls: Times :meth:`score` ran.
        vector_calls: Times :meth:`vectors` ran, which is what proves the index is filled on the scored path only.
    """

    def __init__(self, *, vectors_by_text: dict[str, list[float]] | None = None, publish: bool = True) -> None:
        """Answer a fixed similarity, and optionally publish per-text vectors.

        Args:
            vectors_by_text: Vector to publish per Description. A Description absent from the map gets a zero vector.
            publish: When false, :meth:`vectors` answers empty, standing in for an unavailable embedding.
        """
        self.vectors_by_text = vectors_by_text or {}
        self.publish = publish
        self.score_calls = 0
        self.vector_calls = 0

    def score(self, question: str, descriptions: Sequence[str]) -> Sequence[float]:
        """Return one similarity per Description.

        Args:
            question: Ignored.
            descriptions: The Descriptions being scored.

        Returns:
            A fixed similarity per Description, above any default threshold.
        """
        self.score_calls += 1
        return [0.9] * len(descriptions)

    def vectors(self, descriptions: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return the document vector of each Description.

        Args:
            descriptions: The Descriptions, in the order the vectors are wanted.

        Returns:
            One vector per Description, or empty when this matcher was built not to publish.
        """
        self.vector_calls += 1
        if not self.publish:
            return []
        return [self.vectors_by_text.get(text, [0.0] * DIMENSIONS) for text in descriptions]


class ScoreOnlyMatcher:
    """A matcher honouring the protocol and nothing more: ``score`` exists, ``vectors`` does not."""

    def score(self, question: str, descriptions: Sequence[str]) -> Sequence[float]:
        """Return one similarity per Description.

        Args:
            question: Ignored.
            descriptions: The Descriptions being scored.

        Returns:
            A fixed similarity per Description.
        """
        return [0.9] * len(descriptions)


def card(title: str, *, turn: int, description: str | None = None) -> Card:
    """Build a Subject Card carrying the fields this path reads.

    Args:
        title: Card Title, its identity in the graph.
        turn: Turn ordinal.
        description: Description text. Defaults to one derived from the title.

    Returns:
        The Card.
    """
    return Card(
        title=title,
        kind="subject",
        turn=turn,
        dialogue_ids=(f"m-{turn}",),
        evidence_ids=(),
        pairs=(),
        tool_names=frozenset(),
        references=(),
        numeric_lines=(),
        tags=(),
        description=description if description is not None else f"description of {title}",
    )


def seeded(*cards: Card) -> _GraphState:
    """Return a state holding ``cards``.

    Args:
        cards: Cards to place in the graph, in any order.

    Returns:
        The seeded state.
    """
    state = _GraphState()
    for entry in cards:
        state.cards[entry.title] = entry
    state.turn = len(cards)
    return state


def three() -> tuple[Card, ...]:
    """Three Cards, the default ``min_cards``, so the warm-up does not short-circuit.

    Returns:
        The Cards.
    """
    return (card("First", turn=0), card("Second", turn=1), card("Third", turn=2))


def turn(
    state: _GraphState,
    matcher: object,
    question: str = "what now?",
    *,
    min_cards: int = 3,
    link_threshold: float = 0.5,
) -> _GraphState:
    """Fire one projection against ``state`` and return the state it produced.

    The conversation is a single open user ask, so the write half closes no turn and the seeded Cards are the whole
    graph the read half sees -- the same setup the plugin's ``BeforeInvocationEvent`` gave it.

    Args:
        state: The graph state to project from. Read only, as :func:`project` copies it.
        matcher: The matcher under test.
        question: The turn's question.
        min_cards: Below this many Cards the choice is skipped entirely.
        link_threshold: Similarity at or above which two Cards link.

    Returns:
        The state the projection returned.
    """
    messages = [{"role": "user", "content": [{"text": question}]}]
    _, new_state = project(
        messages,
        state,
        matcher,  # type: ignore[arg-type]
        None,
        Thresholds(min_cards=min_cards, link_threshold=link_threshold),
    )
    return new_state


# --- the index is filled ---------------------------------------------------------------------------


def test_scoring_fills_the_index() -> None:
    """The regression this file is named for: after a scored turn the index holds every Card."""
    matcher = VectorMatcher()

    state = turn(seeded(*three()), matcher)

    assert set(state.vectors) == {"First", "Second", "Third"}
    assert matcher.vector_calls == 1


def test_the_index_stores_the_description_the_vector_was_computed_from() -> None:
    """``_cached_similarity`` compares that text before trusting the vector, so storing it is load-bearing."""
    matcher = VectorMatcher(vectors_by_text={"description of First": [1.0, 0.0, 0.0, 0.0]})

    state = turn(seeded(*three()), matcher)

    assert state.vectors["First"] == ("description of First", (1.0, 0.0, 0.0, 0.0))


def test_a_stale_title_does_not_survive_a_later_turn() -> None:
    """The index tracks the graph rather than accumulating every Title the conversation ever had."""
    before = seeded(*three())
    before.vectors["Gone"] = ("description of Gone", (0.0,) * DIMENSIONS)

    state = turn(before, VectorMatcher())

    assert "Gone" not in state.vectors


# --- and the Link it exists for now forms ----------------------------------------------------------


def test_the_similar_link_forms_once_the_index_is_filled() -> None:
    """End to end: two Cards pointing the same way link, and the one pointing elsewhere does not.

    This is the assertion that fails on the unwired package, with no error and no warning -- every pair measures as
    unmeasurable and the projection simply adds no edge.
    """
    same = [1.0, 0.0, 0.0, 0.0]
    other = [0.0, 1.0, 0.0, 0.0]
    matcher = VectorMatcher(
        vectors_by_text={
            "description of First": same,
            "description of Second": same,
            "description of Third": other,
        }
    )

    state = turn(seeded(*three()), matcher)

    linked = {link.target for link in state.links.get("First", []) if link.kind == "similar"}
    assert linked == {"Second"}


def test_a_card_registered_after_the_choice_is_linked_on_the_next_one() -> None:
    """The timing the package documents, asserted: the newest Card cannot be measured at its own registration.

    A Card is registered when the turn after it closes it, which is after that turn's choice ran -- so its Description
    has never been embedded and every pair touching it is unmeasurable. Filling the index alone does not fix this: the
    measurement has to be retried on the following choice, which is what ``link_newly_measurable`` is for.
    """
    from context_core.graph.cards import register_card

    same = [1.0, 0.0, 0.0, 0.0]
    matcher = VectorMatcher(
        vectors_by_text={
            "description of First": same,
            "description of Second": same,
            "description of Third": same,
            "description of Fourth": same,
        }
    )

    state = turn(seeded(*three()), matcher)
    register_card(state, card("Fourth", turn=3), [], link_threshold=0.5, tags_per_card=4, rarity_weight=1.0)

    assert not [link for link in state.links.get("Fourth", []) if link.kind == "similar"]

    state = turn(state, matcher, "and now?")

    linked = {link.target for link in state.links["Fourth"] if link.kind == "similar"}
    assert linked == {"First", "Second", "Third"}


def test_a_pair_below_the_threshold_gets_no_edge() -> None:
    """The second pass adds edges, it does not assert relatedness: orthogonal Descriptions stay unlinked."""
    matcher = VectorMatcher(
        vectors_by_text={
            "description of First": [1.0, 0.0, 0.0, 0.0],
            "description of Second": [0.0, 1.0, 0.0, 0.0],
            "description of Third": [0.0, 0.0, 1.0, 0.0],
        }
    )

    state = turn(seeded(*three()), matcher)

    assert not [link for links in state.links.values() for link in links if link.kind == "similar"]


# --- the three paths that must not fill it ---------------------------------------------------------


def test_the_warm_up_short_circuit_asks_for_no_vectors() -> None:
    """Below ``min_cards`` the decision is already made, and Requirement 11.3 forbids paying for it."""
    matcher = VectorMatcher()

    state = turn(seeded(card("Only", turn=0)), matcher, min_cards=3)

    assert matcher.score_calls == 0
    assert matcher.vector_calls == 0
    assert state.vectors == {}


def test_a_matcher_without_vectors_leaves_the_index_alone() -> None:
    """``vectors`` is optional on the protocol, so a score-only matcher costs Links and never the turn."""
    state = turn(seeded(*three()), ScoreOnlyMatcher())

    assert state.vectors == {}
    assert not state.choice.full_pass


def test_an_unavailable_embedding_leaves_the_index_alone() -> None:
    """An empty answer means unmeasurable, which must not become a partial index paired with wrong Titles."""
    matcher = VectorMatcher(publish=False)

    state = turn(seeded(*three()), matcher)

    assert matcher.vector_calls == 1
    assert state.vectors == {}
    assert not state.choice.full_pass


def test_a_raising_vectors_method_does_not_cost_the_turn() -> None:
    """Same posture as the scoring round: the choice still lands, degraded only in its Links."""

    class Raising(VectorMatcher):
        """Publishes by raising."""

        def vectors(self, descriptions: Sequence[str]) -> Sequence[Sequence[float]]:
            """Raise instead of answering.

            Args:
                descriptions: Ignored.

            Raises:
                RuntimeError: Always.
            """
            raise RuntimeError("embedding endpoint down")

    state = turn(seeded(*three()), Raising())

    assert state.vectors == {}
    assert not state.choice.full_pass
