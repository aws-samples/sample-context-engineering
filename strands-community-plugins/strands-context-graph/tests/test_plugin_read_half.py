"""Unit tests for the read half under D-1: the Turn Choice computed by the projection on every model call.

The binding decides nothing -- ``project`` does -- so what is worth asserting here is the composition as the plugin
wires it: that the stored choice is frozen by type, that the matcher is reached once per model call and not at all
when the warm-up already answers, that the question comes off the call's own messages, that the reuse clock and the
retrieval counter advance per call, and that any failure lands on a full pass with exactly one warning carrying
``exc_info``.

Each test drives ``plugin._projection.deliver`` -- the ``InvokeModelStage.Input`` handler -- with a hand-built
``InvokeModelContext``. The projection writes a NEW state per call, so assertions always re-read
``plugin._states[agent]``.

The agent is a plain class carrying only ``messages``, so it is weak-referenceable, which the per-agent state map
requires.
"""

import asyncio
import logging
from collections.abc import Sequence
from types import MappingProxyType

import pytest
from context_core.graph.matcher import EmbeddingSimilarityMatcher
from context_core.graph.state import Card, CardChoice, TurnChoice, _GraphState
from strands._middleware.stages import InvokeModelContext

from strands_context_graph import ContextGraph


class _Agent:
    """The one member the projection reads off an agent."""

    def __init__(self, messages=None):
        self.messages = messages if messages is not None else []


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


def call(plugin: ContextGraph, agent: _Agent, messages: list[dict]) -> _GraphState:
    """Run one model call's projection over ``messages`` and return the state it left."""
    context = InvokeModelContext(
        agent=agent,
        messages=messages,
        system_prompt=None,
        tool_specs=[],
        tool_choice=None,
        invocation_state={},
        model=object(),
        projected_input_tokens=None,
        dynamic_trailing_blocks=0,
    )
    asyncio.run(plugin._projection.deliver(context))
    return plugin._states[agent]


def test_choice_is_frozen_and_immutable():
    """The stored choice is a ``MappingProxyType``, so nothing after the projection can edit the call's decisions."""
    plugin = ContextGraph(matcher=CountingMatcher())
    agent = _Agent()
    wired(plugin, agent, *three_cards(), turn=3)

    state = call(plugin, agent, [user("What now?")])

    assert isinstance(state.choice.by_title, MappingProxyType)
    assert not state.choice.full_pass
    assert set(state.choice.by_title) == {"First", "Second", "Third"}
    with pytest.raises(TypeError):
        state.choice.by_title["First"] = None  # type: ignore[index]


def test_one_embedding_round_per_model_call():
    """D-1: every model call recomputes the choice, so the matcher is reached once per call.

    Was "once per turn, reused by every model call": the tool loop's second call now scores again. A matcher with a
    vector cache (the default one) pays nothing for the Descriptions it has already embedded.
    """
    matcher = CountingMatcher()
    plugin = ContextGraph(matcher=matcher)
    agent = _Agent()
    wired(plugin, agent, *three_cards(), turn=3)

    first = call(plugin, agent, [user("What now?")]).choice
    assert matcher.calls == 1

    second = call(plugin, agent, [user("What now?")]).choice
    assert matcher.calls == 2
    assert second is not first


def test_below_min_cards_never_reaches_the_matcher():
    """The warm-up decides by the size of the graph, so no embedding is paid and no client is resolved."""
    matcher = CountingMatcher()
    plugin = ContextGraph(matcher=matcher, min_cards=3)
    agent = _Agent()
    wired(plugin, agent, card("First", turn=0), card("Second", turn=1), turn=2)

    state = call(plugin, agent, [user("What now?")])

    assert state.choice.full_pass
    assert matcher.calls == 0
    assert plugin._resolved_matcher is None


def test_first_call_of_a_fresh_agent_is_a_full_pass():
    """No Card means nothing to choose between, so everything travels at Full Content."""
    matcher = CountingMatcher()
    plugin = ContextGraph(matcher=matcher)
    agent = _Agent()

    state = call(plugin, agent, [user("First ever question")])

    assert state.choice.full_pass
    assert matcher.calls == 0
    assert state.turn == 1


def test_question_comes_from_the_calls_own_messages():
    """The question is the last user text of the messages this model call carries.

    Was read off ``BeforeInvocationEvent.messages`` with a fallback to the history; under D-1 the call's messages are
    the only source, so the fallback test went with the hook.
    """
    matcher = CountingMatcher()
    plugin = ContextGraph(matcher=matcher)
    agent = _Agent()
    wired(plugin, agent, *three_cards(), turn=3)

    call(plugin, agent, [user("An older ask"), assistant("An older answer"), user("The live question")])

    question, descriptions = matcher.seen[0]
    assert question == "The live question"
    # Ascending turn order, and one Description per Card: the sequence the similarities come back aligned to.
    assert descriptions == ("description of First", "description of Second", "description of Third")


class BrokenMatcher:
    """A matcher that raises, standing in for a transport failure or a timeout."""

    def score(self, question: str, descriptions: Sequence[str]) -> Sequence[float]:
        """Raise, which ``scoring`` reads as "score nothing, send everything"."""
        raise RuntimeError("embedding unavailable")


def test_a_failed_matcher_degrades_to_a_full_pass_without_a_warning(caplog):
    """An unusable matcher is scoring's own fail-open path: full content everywhere, and no warning from here."""
    plugin = ContextGraph(matcher=BrokenMatcher())
    agent = _Agent()
    wired(plugin, agent, *three_cards(), turn=3)

    with caplog.at_level(logging.DEBUG, logger="context_core.graph.projection"):
        state = call(plugin, agent, [user("What now?")])

    assert state.choice.full_pass
    assert [record for record in caplog.records if record.levelno >= logging.WARNING] == []


def test_any_exception_stores_a_full_pass_and_logs_exactly_one_warning(monkeypatch, caplog):
    """Requirement 16.1: a failure anywhere in the computation is one warning with ``exc_info`` and a full pass."""
    from context_core.graph import projection as projection_module

    def explode(*args, **kwargs):
        raise RuntimeError("distribution failed")

    monkeypatch.setattr(projection_module, "distribute", explode)

    plugin = ContextGraph(matcher=CountingMatcher())
    agent = _Agent()
    wired(plugin, agent, *three_cards(), turn=3)

    with caplog.at_level(logging.WARNING, logger="context_core.graph.projection"):
        state = call(plugin, agent, [user("What now?")])

    assert state.choice.full_pass
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert warnings[0].exc_info is not None
    # No failure state is kept: the next call computes a choice again.
    monkeypatch.undo()
    assert not call(plugin, agent, [user("And now?")]).choice.full_pass


def test_turn_ordinal_advances_and_retrieval_cycles_reset_per_call():
    """D-1: the ordinal advances by one per model call and the retrieval counter restarts each call (was: per turn)."""
    plugin = ContextGraph(matcher=CountingMatcher())
    agent = _Agent()
    state = wired(plugin, agent, *three_cards(), turn=3)
    state.retrieval_cycles = 4

    after = call(plugin, agent, [user("What now?")])

    assert after.turn == 4
    assert after.retrieval_cycles == 0
    assert call(plugin, agent, [user("What now?")]).turn == 5


def test_the_projection_ages_the_fed_back_notes_before_reading_them():
    """An expired fed-back Note is gone by the time the choice sums the map, and a live one is untouched.

    D-1: the clock is the graph's own ``turn`` ordinal (one per model call), not ``agent.event_loop_metrics``.
    """
    plugin = ContextGraph(matcher=CountingMatcher(), reuse_ttl_cycles=2)
    agent = _Agent()
    state = wired(plugin, agent, *three_cards(), turn=6)
    state.reuse["First"] = (1.0, 4)
    state.reuse["Second"] = (1.0, 8)

    assert call(plugin, agent, [user("What now?")]).reuse == {"Second": (1.0, 8)}


def test_a_resolution_raise_does_not_survive_the_next_call():
    """A tool's elevation lives in the stored choice, so the next call's recompute ends it (Req. 12.14, per call)."""
    plugin = ContextGraph(matcher=CountingMatcher(0.0))
    agent = _Agent()
    state = wired(plugin, agent, *three_cards(), turn=3)
    # What ``expand_card`` leaves behind: the stored choice, with one Card raised.
    state.choice = TurnChoice(
        by_title=MappingProxyType({"First": CardChoice(dialogue="full", evidence="full")}),
        full_pass=False,
    )

    assert call(plugin, agent, [user("Something else entirely")]).choice.by_title["First"].dialogue == "title"


def test_state_is_created_on_first_use_and_is_per_agent():
    """Two agents wired to the same instance never share a Card or a choice."""
    plugin = ContextGraph(matcher=CountingMatcher())
    first, second = _Agent(), _Agent()

    call(plugin, first, [user("First agent")])
    call(plugin, second, [user("Second agent")])

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
