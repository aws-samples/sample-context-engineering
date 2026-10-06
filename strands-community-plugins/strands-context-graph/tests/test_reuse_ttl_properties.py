"""Property tests for the aging of the fed-back Note, over generated cycles, TTLs and graph shapes.

Feature: context-graph-plugin, Property 9: A fed-back Note expires after its TTL.
Validates: Requirements 13.3, 13.4.

``tests/test_scoring.py`` pins the arithmetic on hand-picked examples — one grant, one renewal, one removal. This module
asserts the claim as a rule over the whole space it is stated on: any graph shape, any grant cycle ``C``, any
``reuse_ttl_cycles = k``, any cycle counter the ageing pass is later run with.

The claim is read off the Note rather than off ``state.reuse``, since the increment reaching the Note is what the
property is about. It is read as a difference against the same graph scored with no fed-back Note at all, which is what
makes the assertion exact: the bonus lands on the requested Card's first pass, the propagation only ever adds, and a
Card never propagates Note to itself through its own tool, so the requested Card's Note moves by exactly
``_REUSE_BONUS`` while the grant stands and by nothing at all once it is gone.

The matcher is a table drawn per example and keyed by Description, never an embedding: the similarities vary across
examples but are fixed within one, so the difference between the two scorings isolates the ageing. No network, no model
call, no embedding is reached on any path.
"""

from collections.abc import Sequence

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from strands_context_graph.scoring import _REUSE_BONUS, compute_notes, expire_reuse, record_reuse
from strands_context_graph.state import Card, Link, _GraphState

PROPERTY_SETTINGS = settings(max_examples=100, deadline=None)

TOOL_NAMES = ["run_query", "search", "fetch_report"]


class TableMatcher:
    """A matcher answering from a table keyed by Description, so the scoring is fixed within an example."""

    def __init__(self, table: dict[str, float]) -> None:
        self.table = table

    def score(self, question: str, descriptions: Sequence[str]) -> Sequence[float]:
        """Return one similarity per Description, in the order received."""
        return [self.table.get(description, 0.0) for description in descriptions]


def card(title: str, turn: int) -> Card:
    """Build a subject Card carrying only the fields ``scoring`` reads."""
    return Card(
        title=title,
        kind="subject",
        turn=turn,
        dialogue_ids=("m-1",),
        evidence_ids=(),
        pairs=(),
        tool_names=frozenset(),
        references=(),
        numeric_lines=(),
        tags=(),
        description=f"description of {title}",
    )


@st.composite
def graphs(draw: st.DrawFn) -> _GraphState:
    """A graph of one to five Cards, wired with every Link kind the propagation may or may not walk.

    The shape is what varies: a Card may follow the one before it, cite another as an artifact, share a tool hub with
    its neighbours, or be reachable only by the ``similar`` edge the Note never travels. Every combination leaves the
    claim unchanged, which is the point of generating them.
    """
    count = draw(st.integers(min_value=1, max_value=5))
    titles = [f"t{index}" for index in range(count)]

    state = _GraphState()
    for turn, title in enumerate(titles):
        state.cards[title] = card(title, turn)
    state.turn = count

    for index, title in enumerate(titles):
        # Self-edges are excluded because the scan never derives one, not to keep the arithmetic tidy: a Card's Links
        # are drawn from the messages of other turns.
        others = [other for other in titles if other != title]
        links: list[Link] = []
        if index > 0 and draw(st.booleans()):
            links.append(Link(kind="follows", target=titles[index - 1], weight=1.0))
        if others and draw(st.booleans()):
            links.append(Link(kind="artifact", target=draw(st.sampled_from(others)), weight=1.0))
        if draw(st.booleans()):
            links.append(Link(kind="tool", target=draw(st.sampled_from(TOOL_NAMES)), weight=1.0))
        if others and draw(st.booleans()):
            similarity = draw(st.floats(min_value=0.5, max_value=0.8))
            links.append(Link(kind="similar", target=draw(st.sampled_from(others)), weight=similarity))
        if links:
            state.links[title] = links

    return state


@st.composite
def scenarios(draw: st.DrawFn) -> tuple[_GraphState, TableMatcher, str, int, int]:
    """A graph, its matcher table, the requested Title, the grant cycle ``C`` and the TTL ``k``."""
    state = draw(graphs())
    table = {
        state.cards[title].description: draw(st.floats(min_value=0.0, max_value=1.0)) for title in sorted(state.cards)
    }
    requested = draw(st.sampled_from(sorted(state.cards)))
    grant_cycle = draw(st.integers(min_value=0, max_value=20))
    ttl = draw(st.integers(min_value=1, max_value=5))
    return state, TableMatcher(table), requested, grant_cycle, ttl


def notes_without_reuse(state: _GraphState, matcher: TableMatcher) -> dict[str, float]:
    """Score the same graph with no fed-back Note in effect, the baseline the increment is measured against."""
    baseline = _GraphState()
    baseline.cards = dict(state.cards)
    baseline.links = {title: list(links) for title, links in state.links.items()}
    baseline.turn = state.turn
    return dict(compute_notes(baseline, "the question", matcher))


@given(scenario=scenarios(), offset=st.integers(min_value=-3, max_value=8))
@PROPERTY_SETTINGS
def test_the_fed_back_note_is_present_up_to_its_expiry_and_gone_past_it(scenario, offset):
    """Requirements 13.3, 13.4: granted at ``C`` with TTL ``k``, the increment stands while the counter is <= C + k."""
    state, matcher, requested, grant_cycle, ttl = scenario
    counter = max(0, grant_cycle + ttl + offset)

    baseline = notes_without_reuse(state, matcher)
    record_reuse(state, requested, grant_cycle, reuse_ttl_cycles=ttl)
    expire_reuse(state, counter)
    notes = dict(compute_notes(state, "the question", matcher))

    increment = _REUSE_BONUS if counter <= grant_cycle + ttl else 0.0
    assert notes[requested] == pytest.approx(baseline[requested] + increment)
    # Aging removes the Note outright rather than fading it, so past the expiry the whole graph is scored as if the
    # request had never happened — no Card keeps a residue of it, through a Link or otherwise.
    if increment == 0.0:
        assert notes == pytest.approx(baseline)
        assert requested not in state.reuse
    else:
        assert state.reuse[requested] == (_REUSE_BONUS, grant_cycle + ttl)


@given(scenario=scenarios())
@PROPERTY_SETTINGS
def test_walking_the_counter_forward_drops_the_note_exactly_once_past_its_expiry(scenario):
    """The ageing pass run once per cycle, as a real turn runs it: one transition, and it never comes back.

    The single-probe property above leaves open that repeated ageing over the same state could differ from one pass at
    the same counter — the pass reads a stored expiry rather than counting down, so it must not.
    """
    state, matcher, requested, grant_cycle, ttl = scenario
    baseline = notes_without_reuse(state, matcher)
    record_reuse(state, requested, grant_cycle, reuse_ttl_cycles=ttl)

    seen: list[bool] = []
    for counter in range(grant_cycle, grant_cycle + ttl + 3):
        expire_reuse(state, counter)
        notes = dict(compute_notes(state, "the question", matcher))
        present = notes[requested] == pytest.approx(baseline[requested] + _REUSE_BONUS)
        assert present or notes[requested] == pytest.approx(baseline[requested])
        seen.append(present)

    # Present for the first k + 1 cycles of the walk, absent for the rest: one transition, at the expiry, no revival.
    assert seen == [True] * (ttl + 1) + [False] * 2
