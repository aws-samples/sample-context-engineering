"""Property tests for the shape of a ``find_context`` answer.

Feature: context-graph-plugin, Property 8: find_context returns a bounded, floored, ordered result.

Validates: Requirements 12.9, 12.11.

The matcher is a deterministic score table rather than an embedding round, so the expected answer is computable in the
test from the same numbers the tool scored: the bound, the floor and both tie-breaks are then decidable rather than
approximated. The generators draw the shapes that make the ordering claim non-trivial — more candidates than the cap,
scores repeated across Cards so ties are reached, several Cards sharing one turn ordinal so the second tie-break
(the Title) is the only thing left to decide the order, and floors that fall exactly on a drawn score so the ``>=``
boundary is exercised in both directions.

Two answers are possible and both are held to a claim: a successful answer must carry exactly the candidates the table
puts at or above the floor, capped at five and ordered by ``(-similarity, turn, title)``; an empty answer must name the
``need`` and is only allowed when nothing cleared the floor.
"""

from collections.abc import Sequence

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from strands_context_graph.state import Card, ToolPair, _GraphState
from strands_context_graph.tools import _MAX_CANDIDATES, find_context

CYCLE = 7
"""One cycle counter, so a recorded fed-back Note has a checkable expiry."""

TTL = 5
"""One TTL, long enough that a recorded fed-back Note is observable."""

PROPERTY_SETTINGS = settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

TAGS = ("invoices", "payments", "balance")
"""Tags already in normalized form, so ``normalize`` is the identity over them and the narrowing is decidable here."""

SCORES = (0.0, 0.2, 0.4, 0.45, 0.6, 0.8, 1.0)
"""Quantized similarities. Repeated values are the point: they are what forces both tie-breaks to decide the order."""

FLOORS = (0.0, 0.4, 0.45, 0.6, 1.0)
"""Floors, each one of them also a drawn score, so the ``>=`` boundary is hit from both sides."""

title_strategy = st.text(alphabet=st.sampled_from(list("abcde")), min_size=1, max_size=3)
need_strategy = st.text(alphabet=st.sampled_from(list("abcdefg ")), min_size=1, max_size=12).filter(
    lambda text: bool(text.strip())
)


class TableMatcher:
    """A matcher answering from a table keyed by Description, so a score is a fact of the test rather than a guess."""

    def __init__(self, table: dict[str, float]) -> None:
        """Answer ``table[description]`` per Description received."""
        self.table = table

    def score(self, need: str, descriptions: Sequence[str]) -> Sequence[float]:
        """Return one similarity per Description, in the order received."""
        return [self.table[description] for description in descriptions]


def _card(title: str, turn: int, description: str, tags: tuple[str, ...]) -> Card:
    """Build a Card carrying only the fields ``find_context`` reads."""
    return Card(
        title=title,
        kind="subject",
        turn=turn,
        dialogue_ids=("m-1",),
        evidence_ids=(),
        pairs=(ToolPair(tool_use_id="tu-1", tool_name="calculator", tracking_ids=("a",), consumed=True),),
        tool_names=frozenset(),
        references=(),
        numeric_lines=(),
        tags=tags,
        description=description,
        reference=None,
    )


@st.composite
def search_cases(draw: st.DrawFn) -> tuple[_GraphState, TableMatcher, dict[str, float], str, str | None, float]:
    """A graph, the table its Descriptions score through, the per-Title similarities, the need, tag and floor."""
    titles = draw(st.lists(title_strategy, min_size=0, max_size=8, unique=True))

    state = _GraphState()
    table: dict[str, float] = {}
    similarities: dict[str, float] = {}
    for index, title in enumerate(titles):
        # One Description per Card, so the table holds one score per Card; the turn ordinal is drawn from a range
        # narrower than the Card count, which is what makes several Cards share a turn and forces the Title tie-break.
        description = f"description {index}"
        table[description] = draw(st.sampled_from(SCORES))
        similarities[title] = table[description]
        state.cards[title] = _card(title, draw(st.integers(min_value=0, max_value=3)), description, draw(tags()))

    state.turn = max((card.turn for card in state.cards.values()), default=-1) + 1

    return (
        state,
        TableMatcher(table),
        similarities,
        draw(need_strategy),
        draw(st.none() | st.sampled_from(TAGS)),
        draw(st.sampled_from(FLOORS)),
    )


@st.composite
def tags(draw: st.DrawFn) -> tuple[str, ...]:
    """A Card's tags, drawn from the same small pool the ``tag`` parameter is drawn from."""
    return tuple(draw(st.lists(st.sampled_from(TAGS), max_size=2, unique=True)))


def _expected(
    state: _GraphState,
    similarities: dict[str, float],
    tag: str | None,
    collapse_floor: float,
) -> list[str]:
    """The answer Property 8 claims, computed from the table the matcher answered from.

    Written out independently of the implementation: the candidate set narrowed by the tag, the floor applied with
    ``>=``, the order by descending similarity with the turn and then the Title breaking ties, and the cap last.
    """
    candidates = [
        title
        for title in state.cards
        if (tag is None or tag in state.cards[title].tags) and similarities[title] >= collapse_floor
    ]
    candidates.sort(key=lambda title: (-similarities[title], state.cards[title].turn, title))
    return candidates[:_MAX_CANDIDATES]


def _titles_of(answer: str) -> list[str]:
    """The Titles a successful answer rendered, in the order it rendered them."""
    return [line.removeprefix("- title: ") for line in answer.splitlines() if line.startswith("- title: ")]


@given(case=search_cases())
@PROPERTY_SETTINGS
def test_find_context_is_bounded_floored_and_ordered(
    case: tuple[_GraphState, TableMatcher, dict[str, float], str, str | None, float],
) -> None:
    """Feature: context-graph-plugin, Property 8: find_context returns a bounded, floored, ordered result.

    Validates: Requirements 12.9, 12.11.
    """
    state, matcher, similarities, need, tag, collapse_floor = case
    expected = _expected(state, similarities, tag, collapse_floor)

    answer = find_context(
        state, need, tag, matcher=matcher, collapse_floor=collapse_floor, cycle=CYCLE, reuse_ttl_cycles=TTL
    )

    if not expected:
        # Nothing cleared the floor, so the answer names the need rather than ranking noise (Requirement 12.12).
        assert f"nothing in this conversation matches '{need}'" in answer
        assert not _titles_of(answer)
        return

    returned = _titles_of(answer)
    # Bounded: never more than five, whatever the graph holds (Requirement 12.9).
    assert len(returned) <= _MAX_CANDIDATES
    # Floored: every candidate returned is one the graph did not dismiss (Requirement 12.9).
    assert all(similarities[title] >= collapse_floor for title in returned)
    # Ordered, and deterministically so: descending similarity, then turn, then Title (Requirement 12.11).
    assert returned == expected
    assert returned == sorted(returned, key=lambda title: (-similarities[title], state.cards[title].turn, title))
    # The count the answer states is the count it renders.
    assert f"{len(returned)} earlier turn(s) match '{need}'" in answer
