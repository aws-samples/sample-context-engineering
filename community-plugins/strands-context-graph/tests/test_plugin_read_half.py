"""Unit tests for the read half: the ``BeforeInvocationEvent`` hook that freezes the turn's choice.

The hook itself decides nothing — ``scoring`` does — so what is worth asserting here is the composition: that the
choice is computed once and frozen by type, that the matcher is reached exactly once per turn and not at all when the
warm-up already answers, that the question comes off the invocation's input messages, and that any failure inside the
computation lands on a full pass with exactly one warning carrying ``exc_info``.

The agent is a plain class carrying only ``messages``, so it is weak-referenceable, which the per-agent state map
requires.
"""

import logging
from collections.abc import Sequence
from types import MappingProxyType

import pytest
from strands.hooks.events import BeforeInvocationEvent

from strands_context_graph import ContextGraph
from strands_context_graph.matcher import EmbeddingSimilarityMatcher
from strands_context_graph.state import Card, CardChoice, TurnChoice, _GraphState


class _Agent:
    """The one member the read half reads off an agent."""

    def __init__(self, messages=None):
        self.messages = messages if messages is not None else []


class _Metrics:
    """The one member the fed-back Note is aged by."""

    def __init__(self, cycle_count: int) -> None:
        """Stand in for ``agent.event_loop_metrics``."""
        self.cycle_count = cycle_count


class CountingMatcher:
    """A matcher answering a fixed similarity per Description and counting what it was asked."""

    def __init__(self, value: float = 0.9) -> None:
        """Answer ``value`` for every Description."""
        self.value = value
        self.calls = 0
        self.seen: list[tuple[str, tuple[str, ...]]] = []

    def score(self, question: str, descriptions: Sequence[str]) -> Sequence[float]:
        """Return one similarity per Description, recording the question it was scored against."""
        self.calls += 1
        self.seen.append((question, tuple(descriptions)))
        return [self.value] * len(descriptions)


def card(title: str, *, turn: int) -> Card:
    """Build a Card carrying only the fields the read half passes through to ``scoring``."""
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
        description=f"description of {title}",
    )


def user(text: str) -> dict:
    """A plain user ask."""
    return {"role": "user", "content": [{"text": text}]}


def assistant(text: str) -> dict:
    """An assistant answer."""
    return {"role": "assistant", "content": [{"text": text}]}


def wired(plugin: ContextGraph, agent: _Agent, *cards: Card, turn: int | None = None) -> _GraphState:
    """Seed the plugin's per-agent state with ``cards`` and return it."""
    state = _GraphState()
    for entry in cards:
        state.cards[entry.title] = entry
    state.turn = turn if turn is not None else len(cards)
    plugin._states[agent] = state
    return state


def three_cards() -> tuple[Card, ...]:
    """Three Cards, which is the default ``min_cards``, so the warm-up does not short-circuit."""
    return (card("First", turn=0), card("Second", turn=1), card("Third", turn=2))


def test_choice_is_frozen_and_immutable():
    """The stored choice is a ``MappingProxyType``, so no later call can edit the turn's decisions."""
    matcher = CountingMatcher()
    plugin = ContextGraph(matcher=matcher)
    agent = _Agent()
    state = wired(plugin, agent, *three_cards(), turn=3)

    plugin._on_before_invocation(BeforeInvocationEvent(agent=agent, messages=[user("What now?")]))

    assert isinstance(state.choice.by_title, MappingProxyType)
    assert not state.choice.full_pass
    assert set(state.choice.by_title) == {"First", "Second", "Third"}
    with pytest.raises(TypeError):
        state.choice.by_title["First"] = None  # type: ignore[index]


def test_one_embedding_call_per_turn_reused_by_every_model_call():
    """The matcher is reached once per turn, and every later read of the turn sees the same choice object."""
    matcher = CountingMatcher()
    plugin = ContextGraph(matcher=matcher)
    agent = _Agent()
    state = wired(plugin, agent, *three_cards(), turn=3)

    plugin._on_before_invocation(BeforeInvocationEvent(agent=agent, messages=[user("What now?")]))
    frozen = state.choice

    # The autonomous tool loop's later calls read the state, they do not re-enter the hook: the object is the same one.
    assert plugin._states[agent].choice is frozen
    assert matcher.calls == 1

    # The next turn recomputes, which is the second and only other call.
    plugin._on_before_invocation(BeforeInvocationEvent(agent=agent, messages=[user("And then?")]))
    assert plugin._states[agent].choice is not frozen
    assert matcher.calls == 2


def test_below_min_cards_never_reaches_the_matcher():
    """The warm-up decides by the size of the graph, so no embedding is paid and no client is resolved."""
    matcher = CountingMatcher()
    plugin = ContextGraph(matcher=matcher, min_cards=3)
    agent = _Agent()
    state = wired(plugin, agent, card("First", turn=0), card("Second", turn=1), turn=2)

    plugin._on_before_invocation(BeforeInvocationEvent(agent=agent, messages=[user("What now?")]))

    assert state.choice.full_pass
    assert matcher.calls == 0
    assert plugin._resolved_matcher is None


def test_first_turn_of_a_fresh_agent_is_a_full_pass():
    """No Card means nothing to choose between, so everything travels at Full Content."""
    matcher = CountingMatcher()
    plugin = ContextGraph(matcher=matcher)
    agent = _Agent()

    plugin._on_before_invocation(BeforeInvocationEvent(agent=agent, messages=[user("First ever question")]))

    state = plugin._states[agent]
    assert state.choice.full_pass
    assert matcher.calls == 0
    assert state.turn == 1


def test_question_comes_from_the_invocation_messages():
    """The turn's question is not in ``agent.messages`` yet, so it is read off the event's input messages."""
    matcher = CountingMatcher()
    plugin = ContextGraph(matcher=matcher)
    agent = _Agent([user("An older ask"), assistant("An older answer")])
    wired(plugin, agent, *three_cards(), turn=3)

    plugin._on_before_invocation(
        BeforeInvocationEvent(
            agent=agent,
            messages=[user("An older ask"), assistant("An older answer"), user("The live question")],
        )
    )

    question, descriptions = matcher.seen[0]
    assert question == "The live question"
    # Ascending turn order, and one Description per Card: the sequence the similarities come back aligned to.
    assert descriptions == ("description of First", "description of Second", "description of Third")


def test_question_falls_back_to_the_history_when_the_event_carries_none():
    """``messages`` is optional on the event; the history is the only other place the question can be."""
    matcher = CountingMatcher()
    plugin = ContextGraph(matcher=matcher)
    agent = _Agent([user("The only question there is")])
    wired(plugin, agent, *three_cards(), turn=3)

    plugin._on_before_invocation(BeforeInvocationEvent(agent=agent, messages=None))

    assert matcher.seen[0][0] == "The only question there is"


class BrokenMatcher:
    """A matcher that raises, standing in for a transport failure or a timeout."""

    def score(self, question: str, descriptions: Sequence[str]) -> Sequence[float]:
        """Raise, which ``scoring`` reads as "score nothing, send everything"."""
        raise RuntimeError("embedding unavailable")


def test_a_failed_matcher_degrades_to_a_full_pass_without_a_warning(caplog):
    """An unusable matcher is scoring's own fail-open path: full content everywhere, and no warning from here."""
    plugin = ContextGraph(matcher=BrokenMatcher())
    agent = _Agent()
    state = wired(plugin, agent, *three_cards(), turn=3)

    with caplog.at_level(logging.DEBUG, logger="strands_context_graph.plugin"):
        plugin._on_before_invocation(BeforeInvocationEvent(agent=agent, messages=[user("What now?")]))

    assert state.choice.full_pass
    assert [record for record in caplog.records if record.levelno >= logging.WARNING] == []


def test_any_exception_stores_a_full_pass_and_logs_exactly_one_warning(monkeypatch, caplog):
    """Requirement 16.1: a failure anywhere in the computation is one warning with ``exc_info`` and a full pass."""
    from strands_context_graph import plugin as plugin_module

    def explode(*args, **kwargs):
        raise RuntimeError("distribution failed")

    monkeypatch.setattr(plugin_module, "distribute", explode)

    plugin = ContextGraph(matcher=CountingMatcher())
    agent = _Agent()
    state = wired(plugin, agent, *three_cards(), turn=3)

    with caplog.at_level(logging.WARNING, logger="strands_context_graph.plugin"):
        plugin._on_before_invocation(BeforeInvocationEvent(agent=agent, messages=[user("What now?")]))

    assert state.choice.full_pass
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert warnings[0].exc_info is not None
    # No failure state is kept: the next turn computes a choice again.
    monkeypatch.undo()
    plugin._on_before_invocation(BeforeInvocationEvent(agent=agent, messages=[user("And now?")]))
    assert not plugin._states[agent].choice.full_pass


def test_turn_ordinal_advances_and_retrieval_cycles_reset():
    """The ordinal names the turn now opening, and the retrieval counter is per turn."""
    plugin = ContextGraph(matcher=CountingMatcher())
    agent = _Agent()
    state = wired(plugin, agent, *three_cards(), turn=3)
    state.retrieval_cycles = 4

    plugin._on_before_invocation(BeforeInvocationEvent(agent=agent, messages=[user("What now?")]))

    assert state.turn == 4
    assert state.retrieval_cycles == 0


def test_the_hook_ages_the_fed_back_notes_before_reading_them():
    """An expired fed-back Note is gone by the time the choice sums the map, and a live one is untouched."""
    plugin = ContextGraph(matcher=CountingMatcher(), reuse_ttl_cycles=2)
    agent = _Agent()
    agent.event_loop_metrics = _Metrics(cycle_count=6)
    state = wired(plugin, agent, *three_cards(), turn=3)
    state.reuse["First"] = (1.0, 4)
    state.reuse["Second"] = (1.0, 8)

    plugin._on_before_invocation(BeforeInvocationEvent(agent=agent, messages=[user("What now?")]))

    assert state.reuse == {"Second": (1.0, 8)}


def test_a_resolution_raise_does_not_survive_the_next_turn():
    """A tool's elevation lives in the frozen choice, so the next turn's recompute is what ends it (Req. 12.14)."""
    plugin = ContextGraph(matcher=CountingMatcher(0.0))
    agent = _Agent()
    state = wired(plugin, agent, *three_cards(), turn=3)
    # What ``expand_card`` leaves behind: the frozen choice of the turn in progress, with one Card raised.
    state.choice = TurnChoice(
        by_title=MappingProxyType({"First": CardChoice(dialogue="full", evidence="full")}),
        full_pass=False,
    )

    plugin._on_before_invocation(BeforeInvocationEvent(agent=agent, messages=[user("Something else entirely")]))

    assert state.choice.by_title["First"].dialogue == "title"


def test_state_is_created_on_first_use_and_is_per_agent():
    """Two agents wired to the same instance never share a Card or a choice."""
    plugin = ContextGraph(matcher=CountingMatcher())
    first, second = _Agent(), _Agent()

    plugin._on_before_invocation(BeforeInvocationEvent(agent=first, messages=[user("First agent")]))
    plugin._on_before_invocation(BeforeInvocationEvent(agent=second, messages=[user("Second agent")]))

    assert plugin._states[first] is not plugin._states[second]
    assert plugin._state_for(first) is plugin._states[first]


def test_the_default_matcher_is_resolved_lazily_and_kept_apart():
    """``matcher=None`` stays observable as the configuration it was, and the default is built once, on first need."""
    plugin = ContextGraph()

    assert plugin._matcher is None
    assert plugin._resolved_matcher is None

    resolved = plugin._matcher_for()

    assert isinstance(resolved, EmbeddingSimilarityMatcher)
    assert plugin._matcher is None
    assert plugin._matcher_for() is resolved


def test_a_supplied_matcher_is_never_replaced():
    """A supplied matcher is returned as it was given, and no default is ever built beside it."""
    matcher = CountingMatcher()
    plugin = ContextGraph(matcher=matcher)

    assert plugin._matcher_for() is matcher
    assert plugin._resolved_matcher is None
