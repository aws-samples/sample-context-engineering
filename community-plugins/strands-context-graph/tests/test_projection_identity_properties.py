"""Property tests for the identity delivery.

Feature: context-graph-plugin, Property 2: A full pass is the identity delivery.
Validates: Requirements 1.11, 2.20.

``tests/test_projection.py`` pins the identity deliveries one hand-written history at a time; these properties assert
the universal claim over generated ones. The regression switch is the whole point of the requirement: with
``expand_threshold=0.0`` the call that leaves the plugin must be the call that would have left without it, so a bad
answer can be blamed on the graph or cleared of it by flipping one number.

Two routes reach the full pass, and both are covered:

- through the plugin's own read half, ``expand_threshold=0.0``, which decides by configuration before the matcher is
  ever reached — so the choice is a full pass whatever the history and whatever the question;
- through a choice that collapses nothing, every Card at ``full`` on both parts with ``full_pass`` false, which is the
  ordinary turn where the scoring happened to keep everything.

"Identical field by field" is asserted against a snapshot of the received context taken *before* the delivery, since the
claim is that the delivery returns that same object untouched: the ``messages`` list by object identity,
``dynamic_trailing_blocks`` unmoved, no block folded into the last user message, and every other field equal.
"""

import asyncio
import copy
import dataclasses
from collections.abc import Sequence
from types import MappingProxyType
from typing import Any

from hypothesis import assume, given, settings
from hypothesis import strategies as st
from strands._middleware.registry import MiddlewareRegistry
from strands._middleware.stages import InvokeModelContext
from strands.hooks.events import BeforeInvocationEvent

from strands_context_graph import ContextGraph
from strands_context_graph.cards import rebuild
from strands_context_graph.projection import Projection
from strands_context_graph.state import CardChoice, TurnChoice, _GraphState

PROPERTY_SETTINGS = settings(max_examples=100, deadline=None)

DESCRIPTION_TOKENS = 100
"""The plugin's own default, so the budgeting these properties see is the one production sees."""

RARITY_WEIGHT = 0.5
"""Weight of the rarity term when the rebuild scan ranks textual tag candidates. Irrelevant to the claim."""

TEXT = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126),
    min_size=1,
    max_size=40,
)
"""Printable single-line text: the histories vary in shape, not in encoding."""


class _Agent:
    """The members the delivery path and the read half read off an agent, and nothing else.

    A plain class, so it is weak-referenceable — which the per-agent state map requires.
    """

    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self.messages = messages
        self.state = None
        self._middleware_registry = MiddlewareRegistry()


class UnusedMatcher:
    """A matcher that fails the test if it is ever reached.

    ``expand_threshold=0.0`` is decided by configuration, so no embedding round is owed on any turn.
    """

    def __init__(self) -> None:
        self.calls = 0

    def score(self, question: str, descriptions: Sequence[str]) -> Sequence[float]:
        self.calls += 1
        raise AssertionError("the full-pass switch must decide before the matcher is reached")


def user(text: str, tracking_id: str | None) -> dict[str, Any]:
    """A plain user ask: a turn boundary, since it carries no tool result."""
    message: dict[str, Any] = {"role": "user", "content": [{"text": text}]}
    if tracking_id is not None:
        message["tracking_id"] = tracking_id
    return message


def assistant(text: str, tracking_id: str | None) -> dict[str, Any]:
    """An assistant answer, part of its turn's dialogue."""
    message: dict[str, Any] = {"role": "assistant", "content": [{"text": text}]}
    if tracking_id is not None:
        message["tracking_id"] = tracking_id
    return message


def tool_use(tool_use_id: str, name: str, tracking_id: str) -> dict[str, Any]:
    """The assistant half of a tool pair."""
    return {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": tool_use_id, "name": name, "input": {}}}],
        "tracking_id": tracking_id,
    }


def tool_result(tool_use_id: str, text: str, tracking_id: str) -> dict[str, Any]:
    """The user half of a tool pair. Not a turn boundary: it carries a tool result."""
    return {
        "role": "user",
        "content": [{"toolResult": {"toolUseId": tool_use_id, "content": [{"text": text}], "status": "success"}}],
        "tracking_id": tracking_id,
    }


@st.composite
def histories(draw: st.DrawFn) -> list[dict[str, Any]]:
    """A conversation of closed turns plus the open turn carrying the live question.

    Each closed turn opens on a plain user ask and may carry a tool pair before its assistant answer, so the generated
    histories cover the interleaved evidence the removal would otherwise have to reconcile. A message now and then
    arrives with no ``tracking_id``, the derivation-lag shape, which yields a turn the graph cannot address.
    """
    messages: list[dict[str, Any]] = []

    def identity(*, present: bool = True) -> str | None:
        return f"m{len(messages)}" if present else None

    for _ in range(draw(st.integers(min_value=1, max_value=4))):
        messages.append(user(draw(TEXT), identity(present=draw(st.booleans()))))
        if draw(st.booleans()):
            pair_id = f"tu-{len(messages)}"
            name = draw(st.sampled_from(["run_query", "search", "fetch_report"]))
            messages.append(tool_use(pair_id, name, f"m{len(messages)}"))
            messages.append(tool_result(pair_id, draw(TEXT), f"m{len(messages)}"))
        messages.append(assistant(draw(TEXT), identity()))

    # The open turn: the question the model is about to answer, which no Card covers yet.
    messages.append(user(draw(TEXT), f"m{len(messages)}"))
    return messages


def graph_over(messages: list[dict[str, Any]]) -> _GraphState:
    """The graph the rebuild scan derives from ``messages`` — the same Cards the incremental path would hold."""
    return rebuild(
        messages,
        description_tokens=DESCRIPTION_TOKENS,
        tags_per_card=5,
        rarity_weight=RARITY_WEIGHT,
        link_threshold=0.5,
    )


def no_collapse_choice(state: _GraphState) -> TurnChoice:
    """A choice that addresses every Card and collapses none of them: the ordinary no-collapse turn."""
    return TurnChoice(
        by_title=MappingProxyType(
            {title: CardChoice(dialogue="full", evidence="full") for title in state.cards},
        ),
        full_pass=False,
    )


def context_over(messages: list[dict[str, Any]], agent: _Agent) -> InvokeModelContext:
    """A real ``InvokeModelContext`` over ``messages``, with every field set to something recognizable."""
    return InvokeModelContext(
        agent=agent,
        messages=messages,
        system_prompt="a system prompt",
        tool_specs=[{"name": "run_query", "description": "runs", "inputSchema": {}}],
        tool_choice=None,
        invocation_state={"key": "value"},
        model=object(),
        projected_input_tokens=1234,
        dynamic_trailing_blocks=2,
    )


def snapshot(context: InvokeModelContext) -> dict[str, Any]:
    """Every field of ``context`` as it stands, so the delivery can be compared against what it received."""
    return {field.name: getattr(context, field.name) for field in dataclasses.fields(context)}


def assert_identity_delivery(
    delivered: InvokeModelContext,
    context: InvokeModelContext,
    before: dict[str, Any],
    deep_messages: list[dict[str, Any]],
) -> None:
    """Assert ``delivered`` is the received context, unchanged in every field and carrying no folded block."""
    # Not merely equal: the requirement is the context itself, so no downstream link can tell the plugin ran.
    assert delivered is context
    assert delivered.messages is before["messages"]
    assert delivered.dynamic_trailing_blocks == before["dynamic_trailing_blocks"]
    for name, value in before.items():
        assert getattr(delivered, name) is value, name
    # The messages themselves are untouched too: no block appended to the last user message, nothing dropped.
    assert delivered.messages == deep_messages


def deliver(handler: Projection, context: InvokeModelContext) -> InvokeModelContext:
    """Run one delivery to completion. Synchronous, so Hypothesis drives the examples and not an event loop policy."""
    return asyncio.run(handler.deliver(context))


def delivery_of(state: _GraphState | None, messages: list[dict[str, Any]]) -> tuple[Projection, InvokeModelContext]:
    """A ``Projection`` wired to one agent holding ``state``, plus that agent's per-call context."""
    agent = _Agent(messages)
    states: dict[_Agent, _GraphState] = {} if state is None else {agent: state}
    return Projection(states, description_tokens=DESCRIPTION_TOKENS), context_over(list(messages), agent)


@given(messages=histories())
@PROPERTY_SETTINGS
def test_expand_threshold_zero_delivers_the_received_context_by_identity(messages):
    """Requirement 2.20: the regression switch, read through the plugin's own ``BeforeInvocationEvent`` half.

    The choice comes from the plugin rather than being handed in, so the property covers the configuration seam too: at
    ``expand_threshold=0.0`` the read half answers a full pass off the size of the graph and the switch alone, paying no
    embedding, and the delivery that reads that choice returns the context it was given.
    """
    matcher = UnusedMatcher()
    plugin = ContextGraph(expand_threshold=0.0, collapse_floor=0.0, matcher=matcher)
    agent = _Agent(messages)
    plugin._states[agent] = graph_over(messages)

    plugin._on_before_invocation(BeforeInvocationEvent(agent=agent, messages=messages))

    assert plugin._states[agent].choice.full_pass
    assert matcher.calls == 0

    context = context_over(list(messages), agent)
    before = snapshot(context)
    deep_messages = copy.deepcopy(context.messages)

    assert_identity_delivery(deliver(plugin._projection, context), context, before, deep_messages)


@given(messages=histories())
@PROPERTY_SETTINGS
def test_a_choice_that_collapses_nothing_delivers_the_received_context_by_identity(messages):
    """Requirement 1.11: the no-collapse turn — every Card addressed, none below Full Content, nothing folded."""
    state = graph_over(messages)
    # A graph with no Card would satisfy the claim trivially: the interesting example is a choice that addresses Cards
    # and keeps all of them whole.
    assume(state.cards)
    state.choice = no_collapse_choice(state)
    handler, context = delivery_of(state, messages)
    before = snapshot(context)
    deep_messages = copy.deepcopy(context.messages)

    assert_identity_delivery(deliver(handler, context), context, before, deep_messages)


@given(messages=histories())
@PROPERTY_SETTINGS
def test_the_full_pass_short_circuit_delivers_the_received_context_by_identity(messages):
    """The same claim off the ``full_pass`` flag itself, which is where every degradation path lands."""
    state = graph_over(messages)
    state.choice = TurnChoice(by_title=MappingProxyType({}), full_pass=True)
    handler, context = delivery_of(state, messages)
    before = snapshot(context)
    deep_messages = copy.deepcopy(context.messages)

    assert_identity_delivery(deliver(handler, context), context, before, deep_messages)


@given(messages=histories())
@PROPERTY_SETTINGS
def test_an_agent_with_no_graph_delivers_the_received_context_by_identity(messages):
    """A state that does not exist yet is a fresh state, and a fresh state is a full pass."""
    handler, context = delivery_of(None, messages)
    before = snapshot(context)
    deep_messages = copy.deepcopy(context.messages)

    assert_identity_delivery(deliver(handler, context), context, before, deep_messages)


@given(messages=histories())
@PROPERTY_SETTINGS
def test_the_identity_delivery_never_touches_the_agent_history(messages):
    """The history stays the object it was, whichever route reached the full pass (Requirement 1.4)."""
    state = graph_over(messages)
    state.choice = no_collapse_choice(state)
    handler, context = delivery_of(state, messages)
    history = context.agent.messages
    before = copy.deepcopy(history)

    deliver(handler, context)

    assert context.agent.messages is history
    assert context.agent.messages == before
