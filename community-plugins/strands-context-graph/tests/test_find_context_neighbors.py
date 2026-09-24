"""``find_context`` listing the ``similar`` neighbours of each candidate it returns.

Until this existed the ``similar`` edge had no reader at all: it is measured on the write path and stored
with its similarity as the weight, ``_STRUCTURAL_WEIGHTS`` omits it so it propagates no Note, and no
retrieval path traversed it. These tests pin the traversal, the ordering, and the three cases where a
neighbour must NOT be offered.

The relation the edge holds is one the candidate ranking cannot see: ``find_context`` scores each
Description against the QUESTION, never against another Description, so two turns discussing the same
thing in different words are invisible to each other there.
"""

from __future__ import annotations

import pytest
from strands_context_graph.state import Link
from strands_context_graph.tools import _similar_neighbors, find_context

from tests.test_tools import CYCLE, TTL, card, graph


class _Matcher:
    """Scores every Description the same, so ordering under test is the neighbour ordering only."""

    def __init__(self, score: float = 0.9) -> None:
        self._score = score

    def score(self, question: str, descriptions: object) -> list[float]:
        return [self._score] * len(list(descriptions))  # type: ignore[arg-type]


class _MatcherByText:
    """Scores a Description by the title named in it, so a Card can be kept below the floor.

    A neighbour is only ever offered for a Card that is NOT itself a candidate, so any test about the
    listing needs at least one Card the ranking rejects -- which a uniform matcher cannot produce.
    """

    def __init__(self, low: str, low_score: float = 0.0, high_score: float = 0.9) -> None:
        self._low = low
        self._low_score = low_score
        self._high_score = high_score

    def score(self, question: str, descriptions: object) -> list[float]:
        return [
            self._low_score if self._low in text else self._high_score
            for text in descriptions  # type: ignore[union-attr]
        ]


def _link(state: object, source: str, target: str, weight: float, kind: str = "similar") -> None:
    """Add one directed edge, the way ``cards._link`` would."""
    state.links.setdefault(source, []).append(Link(kind=kind, target=target, weight=weight))  # type: ignore[attr-defined]


def _find(state: object, need: str = "the balance", neighbors: int = 3, matcher: object = None) -> str:
    return find_context(
        state,  # type: ignore[arg-type]
        need,
        None,
        matcher=matcher or _Matcher(),  # type: ignore[arg-type]
        collapse_floor=0.1,
        cycle=CYCLE,
        reuse_ttl_cycles=TTL,
        neighbors_per_candidate=neighbors,
    )


def test_a_candidates_similar_neighbours_are_listed_under_it() -> None:
    """The traversal the edge was always described as having, and never had.

    ``third`` is held below the floor so it is not a candidate itself -- which is the only situation in
    which a neighbour is worth offering: the ranking rejected it against the question, and the edge says
    it is related to a Card the ranking accepted.
    """
    state = graph(card("first", turn=0), card("second", turn=1), card("third", turn=2))
    _link(state, "first", "third", 0.71)

    answer = _find(state, matcher=_MatcherByText("third"))

    assert "related turns:" in answer
    assert "third (0.71)" in answer
    # And it is offered as a hint, not promoted to a candidate.
    assert "- title: third" not in answer


def test_neighbours_are_ordered_by_weight_and_capped() -> None:
    """Strongest first, and the cap is what keeps a hint from becoming a second ranking."""
    state = graph(*(card(name, turn=index) for index, name in enumerate("abcdef")))
    for target, weight in (("c", 0.50), ("d", 0.90), ("e", 0.70), ("f", 0.60)):
        _link(state, "a", target, weight)

    neighbors = _similar_neighbors(state, "a", {"a"}, 3)

    assert [title for title, _ in neighbors] == ["d", "e", "f"]


def test_a_candidate_is_never_offered_as_its_own_neighbour_or_another_candidates() -> None:
    """A Card already rendered in full costs tokens to name twice and adds nothing."""
    state = graph(card("first", turn=0), card("second", turn=1))
    _link(state, "first", "second", 0.80)
    _link(state, "second", "first", 0.80)

    # Both Cards clear the floor, so both are candidates and neither may appear as a neighbour.
    assert "related turns:" not in _find(state)


def test_only_the_similar_kind_is_traversed() -> None:
    """``follows``, ``artifact`` and ``tool`` carry Note instead; ``tool`` does not even target a Card."""
    state = graph(card("first", turn=0), card("second", turn=1), card("third", turn=2))
    _link(state, "first", "third", 1.0, kind="follows")
    _link(state, "first", "get_balance", 1.0, kind="tool")

    assert _similar_neighbors(state, "first", {"first"}, 3) == []


def test_a_dangling_edge_is_skipped_rather_than_raising() -> None:
    """Links are rebuilt by scan, so an edge onto a Title the graph dropped is stale, not corrupt."""
    state = graph(card("first", turn=0))
    _link(state, "first", "evicted", 0.90)

    assert _similar_neighbors(state, "first", {"first"}, 3) == []


def test_zero_reproduces_the_answer_shape_from_before_neighbours_existed() -> None:
    """The off switch has to be exact, because every published figure was measured without this."""
    state = graph(card("first", turn=0), card("second", turn=1), card("third", turn=2))
    _link(state, "first", "third", 0.71)

    assert "related turns:" not in _find(state, neighbors=0, matcher=_MatcherByText("third"))


def test_a_card_with_no_edges_yields_nothing() -> None:
    state = graph(card("lonely", turn=0))
    assert _similar_neighbors(state, "lonely", set(), 3) == []


def test_an_unknown_title_yields_nothing_rather_than_raising() -> None:
    state = graph(card("first", turn=0))
    assert _similar_neighbors(state, "never-registered", set(), 3) == []


@pytest.mark.parametrize("neighbors_per_candidate", [-1, 2.5, True, "3"])
def test_a_bad_neighbour_count_is_refused_at_construction(neighbors_per_candidate: object) -> None:
    """Validated before any handler is registered, so a typo fails loudly instead of silently listing none."""
    from strands_context_graph import ContextGraph

    with pytest.raises(ValueError, match="neighbors_per_candidate"):
        ContextGraph(neighbors_per_candidate=neighbors_per_candidate)  # type: ignore[arg-type]
