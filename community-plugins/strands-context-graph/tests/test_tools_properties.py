"""Property tests for the answering contract of the three retrieval tools.

Feature: context-graph-plugin, Property 7: Every retrieval tool answers, never raises.

Validates: Requirements 12.3, 12.6, 12.7, 12.12, 16.8.

The generators aim at the misses rather than the happy path, because the happy path is what the unit tests already pin
down. Each case draws a graph, a frozen Turn Choice over it, and then a request that may well not match any of it: a
Title copied wrong, a reference nothing ever offloaded, a block that yields no text, no storage at all, an empty
``need``, a matcher that raises or answers with the wrong number of similarities, a malformed ``line_range``, a pattern
that is not a usable regex.

Three claims are held over every one of them:

- **Every call answers with a string.** ``expand_card``, ``expand_artifact`` and ``find_context`` return prose on every
  path; nothing raises, because a raise would report the *tool* broken rather than the *request*
  (Requirements 12.3, 12.6, 12.7, 12.12, 16.8). Hypothesis surfaces a raise as the failure itself.
- **The answer names what was missing.** The Title asked for, the reference asked for, or the ``need`` searched for is
  quoted back, so the model can tell which of its requests came back empty.
- **A miss writes nothing.** On every error path the frozen Turn Choice is the same object — so no Card's Resolution
  moved — and ``state.reuse`` is unchanged, so no fed-back Note was recorded (Requirement 13.8). ``expand_artifact`` and
  ``find_context`` are held to the Resolution half on *every* path, success included: neither is allowed to move a
  Resolution at all.

The matcher is always a table or a deliberately broken object, and the agent stub trips on every attribute a model call
would travel through, so no property here can reach an embedding model.
"""

import asyncio
from collections.abc import Sequence
from types import MappingProxyType
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from strands_context_graph.state import Card, CardChoice, ToolPair, TurnChoice, _GraphState
from strands_context_graph.store import (
    InMemoryReferenceStore,
    absent_message,
    non_textual_message,
    unknown_message,
)
from strands_context_graph.tools import _MAX_CANDIDATES, expand_artifact, expand_card, find_context

PROPERTY_SETTINGS = settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

_MODEL_ATTRS = frozenset(
    {"model", "invoke_async", "stream_async", "structured_output", "structured_output_async", "converse"}
)
"""Every attribute a model call on an agent would travel through. Reaching for one during a retrieval is a bug."""


class Agent:
    """An agent stub carrying no plugin registry, so the optional Stash bridge is simply absent."""

    _plugin_registry = None

    def __getattr__(self, name: str) -> object:
        """Fail loudly on any attribute a model call would need."""
        if name in _MODEL_ATTRS:
            raise AssertionError(f"a retrieval tool reached for agent.{name}")
        raise AttributeError(name)


class TableMatcher:
    """A matcher answering from a table, so a ranking is decidable without an embedding model."""

    def __init__(self, scores: dict[str, float], *, default: float = 0.0) -> None:
        """Answer ``scores[description]`` per Description, falling back on ``default``."""
        self.scores = scores
        self.default = default

    def score(self, need: str, descriptions: Sequence[str]) -> Sequence[float]:
        """Return one similarity per Description, in the order received."""
        return [self.scores.get(description, self.default) for description in descriptions]


class BrokenMatcher:
    """A matcher that raises, standing in for a transport failure or a timeout."""

    def score(self, need: str, descriptions: Sequence[str]) -> Sequence[float]:
        """Raise, which the tool must read as "no candidate" and never propagate."""
        raise RuntimeError("embedding unavailable")


class ShortMatcher:
    """A matcher answering with the wrong number of similarities, breaking its own contract."""

    def score(self, need: str, descriptions: Sequence[str]) -> Sequence[float]:
        """Answer one similarity short, whatever was asked."""
        return [0.9] * max(0, len(descriptions) - 1)


class GarbageMatcher:
    """A matcher answering with values that are not numbers at all."""

    def score(self, need: str, descriptions: Sequence[str]) -> Sequence[float]:
        """Answer strings, which the tool must read as "no candidate"."""
        return ["not a number"] * len(descriptions)  # type: ignore[return-value]


class NoScoreMatcher:
    """An object supplied as a matcher that has no ``score`` member at all."""


# ---- generators -------------------------------------------------------------------------------

text_strategy = st.text(
    alphabet=st.sampled_from(list("abcdefghijklmnopqrstuvwxyz 0123456789.,:$-_")),
    min_size=1,
    max_size=24,
)
"""Text that stays inside what the Cards and the prose actually carry, never empty."""

empty_need_strategy = st.sampled_from(["", "   ", "\t\n"])
"""The three ways of being empty that must never reach the matcher."""

CARD_TAGS = ("invoice", "payment", "1200")
"""The Tags a generated Card may carry, so a filter that narrows and one that finds nothing are both reachable."""

line_range_strategy = st.one_of(
    st.none(),
    st.just({"start": 1, "end": 2}),
    st.just({"start": 900, "end": 901}),
    st.just({"start": 2, "end": 1}),
    st.just({}),
    st.just({"from": 1}),
    st.just({"start": "x", "end": "y"}),
)
"""Line ranges: usable, out of range, inverted, and three malformed shapes."""

pattern_strategy = st.one_of(st.none(), st.just("March"), st.just("["), st.just("(unclosed"), text_strategy)
"""Patterns, including two that are not usable regexes."""

block_strategy = st.one_of(
    st.just("line one\nR$ 1.200,00 in March\nline three"),
    st.just(""),
    st.just({"image": {"format": "png"}}),
    st.just(b"raw bytes"),
    st.just(12345),
)
"""Decoded blocks: text, empty text, and three that yield no text at all."""

unusable_matcher_strategy = st.one_of(
    st.builds(BrokenMatcher),
    st.builds(ShortMatcher),
    st.builds(GarbageMatcher),
    st.builds(NoScoreMatcher),
)
"""The four ways of being unusable: raising, answering short, answering nonsense, having no ``score`` at all."""


def _card(title: str, *, turn: int, kind: str, description: str, tags: tuple[str, ...]) -> Card:
    """Build a Card carrying only the fields the three tools read."""
    return Card(
        title=title,
        kind=kind,  # type: ignore[arg-type]
        turn=turn,
        dialogue_ids=(f"m-{turn}",),
        evidence_ids=(),
        pairs=(ToolPair(tool_use_id=f"tu-{turn}", tool_name="calculator", tracking_ids=(f"m-{turn}",), consumed=True),),
        tool_names=frozenset(),
        references=(),
        numeric_lines=(),
        tags=tags,
        description=description,
        reference=f"mem_1_tu-{turn}_0" if kind == "artifact" else None,
    )


@st.composite
def graphs(draw: st.DrawFn) -> _GraphState:
    """A graph of drawn Cards under a drawn Turn Choice, sometimes empty and sometimes a full pass."""
    titles = draw(st.lists(text_strategy, min_size=0, max_size=6, unique=True))

    state = _GraphState()
    for turn, title in enumerate(titles):
        state.cards[title] = _card(
            title,
            turn=turn,
            kind=draw(st.sampled_from(["subject", "subject", "artifact"])),
            description=draw(text_strategy),
            tags=tuple(draw(st.lists(st.sampled_from(CARD_TAGS), max_size=2, unique=True))),
        )
    state.turn = len(titles)

    if draw(st.booleans()):
        state.choice = TurnChoice(by_title=MappingProxyType({}), full_pass=True)
    else:
        state.choice = TurnChoice(
            by_title=MappingProxyType(
                {
                    title: CardChoice(
                        dialogue=draw(st.sampled_from(["full", "description", "title"])),
                        evidence=draw(st.sampled_from(["full", "description", "title"])),
                    )
                    for title in titles
                }
            ),
            full_pass=False,
        )

    # A fed-back Note already standing, so "a miss records nothing" is asserted against a non-empty map too.
    for title in titles:
        if draw(st.booleans()):
            state.reuse[title] = (1.0, 42)

    return state


@st.composite
def card_requests(draw: st.DrawFn) -> tuple[_GraphState, str]:
    """A graph and a Title asked for, drawn from the Titles it holds *and* from Titles it does not.

    Correlated on purpose: a Title drawn freely would practically never name a Card, and the property would then only
    ever assert the miss.
    """
    state = draw(graphs())
    candidates = [*state.cards, "nowhere", "", draw(text_strategy)]
    return state, draw(st.sampled_from(candidates))


@st.composite
def artifact_requests(draw: st.DrawFn) -> tuple[_GraphState, str, InMemoryReferenceStore]:
    """A graph, a reference asked for, and a store that may or may not hold it.

    The stored reference is drawn from the same candidate set as the requested one, so own-store hits and misses both
    occur often instead of the hit being an accident.
    """
    state = draw(graphs())
    references = [
        *(card.reference for card in state.cards.values() if card.reference),
        "mem_1_tu-3_0",
        "mem_9_absent_0",
        "",
    ]

    store = InMemoryReferenceStore()
    if draw(st.booleans()):
        store.put(draw(st.sampled_from(references)), draw(block_strategy))

    return state, draw(st.sampled_from(references)), store


@st.composite
def find_context_requests(draw: st.DrawFn) -> tuple[_GraphState, str, str | None, object, float]:
    """A graph, a ``need``, a Tag filter, a matcher and a floor, weighted so both outcomes occur.

    Drawn together rather than independently: with five independent draws the search practically always came back empty,
    and the property would then never see what a *successful* search writes.
    """
    state = draw(graphs())

    need = draw(text_strategy) if draw(st.integers(0, 3)) else draw(empty_need_strategy)

    tag_kind = draw(st.sampled_from(["none", "none", "none", "carried", "junk", "empty"]))
    if tag_kind == "none":
        tag: str | None = None
    elif tag_kind == "carried":
        tag = draw(st.sampled_from(CARD_TAGS))
    elif tag_kind == "junk":
        tag = draw(text_strategy)
    else:
        tag = ""

    floor = draw(st.sampled_from([0.0, 0.45, 0.5, 0.9]))
    if draw(st.booleans()):
        matcher: object = TableMatcher({}, default=draw(st.sampled_from([0.0, 0.3, 0.5, 0.9, 1.0])))
    else:
        matcher = draw(unusable_matcher_strategy)

    return state, need, tag, matcher, floor


# ---- shared assertions ------------------------------------------------------------------------


def _answered(answer: object) -> str:
    """Assert the tool answered with a string rather than raising, and hand the answer back."""
    assert isinstance(answer, str)
    return answer


def _assert_wrote_nothing(
    state: _GraphState,
    choice_before: TurnChoice,
    reuse_before: dict[str, tuple[float, int]],
) -> None:
    """Assert no Card's Resolution and no fed-back Note moved.

    The Turn Choice is checked by identity rather than by equality: it is frozen for the whole turn, so a rewrite that
    happened to land on the same Resolutions is still a rewrite that must not have happened on an error path.
    """
    assert state.choice is choice_before
    assert state.reuse == reuse_before


# ---- expand_card ------------------------------------------------------------------------------


@given(case=card_requests(), cycle=st.integers(min_value=0, max_value=50), ttl=st.integers(0, 5))
@PROPERTY_SETTINGS
def test_expand_card_answers_and_a_miss_writes_nothing(case: tuple[_GraphState, str], cycle: int, ttl: int) -> None:
    """Feature: context-graph-plugin, Property 7: Every retrieval tool answers, never raises.

    Validates: Requirements 12.3, 12.6, 12.7, 12.12, 16.8.
    """
    state, title = case
    choice_before = state.choice
    reuse_before = dict(state.reuse)
    titles_before = set(state.cards)

    answer = _answered(expand_card(state, title, cycle=cycle, reuse_ttl_cycles=ttl))

    # Whatever happened, the Title asked for is named back and the graph itself is untouched.
    assert f"'{title}'" in answer
    assert set(state.cards) == titles_before

    card = state.cards.get(title)
    if card is None or card.kind != "subject":
        _assert_wrote_nothing(state, choice_before, reuse_before)
        return

    # The one success path: both axes raised for the rest of the turn, and every other Card left as decided.
    if choice_before.full_pass:
        assert state.choice is choice_before
    else:
        assert state.choice.by_title[title] == CardChoice(dialogue="full", evidence="full")
        for other in titles_before - {title}:
            assert state.choice.by_title[other] == choice_before.by_title[other]
    if ttl == 0:
        assert state.reuse == reuse_before
    else:
        assert state.reuse[title] == (1.0, cycle + ttl)


# ---- expand_artifact --------------------------------------------------------------------------


def _artifact_missed(answer: str, reference: str) -> bool:
    """Whether the answer is one of the misses: no storage, unknown, non-textual, or an unusable read request."""
    return (
        answer in {absent_message(reference), unknown_message(reference), non_textual_message(reference)}
        or answer.startswith("expand_artifact | line_range=")
        or answer.startswith(f"expand_artifact | reference '{reference}' | ")
    )


@given(
    case=artifact_requests(),
    line_range=line_range_strategy,
    pattern=pattern_strategy,
    cycle=st.integers(min_value=0, max_value=50),
    ttl=st.integers(0, 5),
)
@PROPERTY_SETTINGS
def test_expand_artifact_answers_and_never_moves_a_resolution(
    case: tuple[_GraphState, str, InMemoryReferenceStore],
    line_range: dict[str, Any] | None,
    pattern: str | None,
    cycle: int,
    ttl: int,
) -> None:
    """Feature: context-graph-plugin, Property 7: Every retrieval tool answers, never raises.

    Validates: Requirements 12.3, 12.6, 12.7, 12.12, 16.8.
    """
    state, reference, store = case
    choice_before = state.choice
    reuse_before = dict(state.reuse)

    answer = _answered(
        asyncio.run(
            expand_artifact(
                state,
                store,
                Agent(),
                reference,
                line_range,
                pattern,
                cycle=cycle,
                reuse_ttl_cycles=ttl,
            )
        )
    )

    # No read, successful or not, ever moves a Resolution: the content asked for is in the answer itself.
    assert state.choice is choice_before

    if _artifact_missed(answer, reference):
        # Every miss names what was missing — the reference, or the unusable range that was asked for — and records no
        # fed-back Note. A malformed range is named rather than the reference, since the reference never got looked at.
        if answer.startswith("expand_artifact | line_range="):
            assert f"line_range=<{line_range!r}>" in answer
        else:
            assert f"'{reference}'" in answer
        assert state.reuse == reuse_before


# ---- find_context -----------------------------------------------------------------------------


@given(
    case=find_context_requests(),
    cycle=st.integers(min_value=0, max_value=50),
    ttl=st.integers(0, 5),
)
@PROPERTY_SETTINGS
def test_find_context_answers_and_a_miss_writes_nothing(
    case: tuple[_GraphState, str, str | None, object, float],
    cycle: int,
    ttl: int,
) -> None:
    """Feature: context-graph-plugin, Property 7: Every retrieval tool answers, never raises.

    Validates: Requirements 12.3, 12.6, 12.7, 12.12, 16.8.
    """
    state, need, tag, matcher, collapse_floor = case
    choice_before = state.choice
    reuse_before = dict(state.reuse)

    answer = _answered(
        find_context(
            state,
            need,
            tag,
            matcher=matcher,  # type: ignore[arg-type]
            collapse_floor=collapse_floor,
            cycle=cycle,
            reuse_ttl_cycles=ttl,
        )
    )

    # A search reports, it never re-decides: no Resolution moves on any path.
    assert state.choice is choice_before
    # The need searched for is named back, whether anything matched or not.
    assert f"'{need}'" in answer

    if "nothing in this conversation matches" in answer:
        if tag is not None:
            assert f"tagged '{tag}'" in answer
        _assert_wrote_nothing(state, choice_before, reuse_before)
        return

    # The one success path: a bounded result, each candidate a Card of this graph, each carrying its fed-back Note.
    titles = [line.removeprefix("- title: ") for line in answer.splitlines() if line.startswith("- title: ")]
    assert 1 <= len(titles) <= _MAX_CANDIDATES
    assert all(title in state.cards for title in titles)
    if ttl == 0:
        assert state.reuse == reuse_before
    else:
        assert all(state.reuse[title] == (1.0, cycle + ttl) for title in titles)
