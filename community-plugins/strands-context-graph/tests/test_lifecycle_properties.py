"""Property tests for the untouched live history, driven through the whole plugin.

Feature: context-graph-plugin, Property 1: The live history is never mutated.
Validates: Requirements 1.3, 1.4, 3.4.

``tests/test_state_isolation_properties.py`` asserts the derivation-only slice of this property — the Card holds
identities, not content, and two agents never share a graph. This module asserts the whole of it: every engagement point
the plugin owns runs over one generated conversation, in the order a real turn runs them, and ``agent.messages`` comes
out the same object, holding the same dicts, carrying the same keys and values it carried before.

One example drives, in order:

- ``_on_message_added`` once per message, the incremental construction of the graph;
- ``_on_after_tool_call`` for every tool result an offloader replaced, which is where artifact Cards and the plugin's
  own store are written;
- ``_on_before_invocation``, which scores the graph and freezes the turn's choice;
- the ``InvokeModelStage.Input`` delivery, handed the live list itself — the strongest form of the claim, since a
  removal that filtered in place would be visible in the history rather than only in the call;
- a retrieval tool, at least one per example and all three whenever the conversation offers something to ask for, since
  a tool that raised a Card's Resolution back up must still not write anything back;
- the delivery a second time, the autonomous tool loop's next model call reading the same frozen choice.

The matcher is a stub scoring by Description length, so roughly half the Cards collapse and the delivery does real work
on most examples; ``min_cards=1`` keeps the warm-up short circuit out of the way and a small ``body_budget`` forces the
mixed resolutions. No embedding, no network, no model call is reached on any path.
"""

import asyncio
import copy
from collections.abc import Sequence
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st
from strands._middleware.registry import MiddlewareRegistry
from strands._middleware.stages import InvokeModelContext
from strands.hooks.events import AfterToolCallEvent, BeforeInvocationEvent, MessageAddedEvent
from strands.types.tools import ToolContext

from strands_context_graph import ContextGraph

PROPERTY_SETTINGS = settings(max_examples=100, deadline=None)

BODY_BUDGET = 40
"""Small enough that the Full Content resolutions run out of budget partway down the graph, which is the turn where the
delivery both removes and folds."""

TEXT = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126),
    min_size=1,
    max_size=40,
)
"""Printable single-line text: the histories vary in shape, not in encoding."""

TOOL_NAMES = ["run_query", "search", "fetch_report"]


class _Metrics:
    """The one member the fed-back Note is aged by."""

    def __init__(self, cycle_count: int = 3) -> None:
        self.cycle_count = cycle_count


class _Agent:
    """The members the four handlers, the delivery and the three tools read off an agent, and nothing else.

    A plain class, so it is weak-referenceable — which the per-agent state and store maps both require.
    """

    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self.messages = messages
        self.state = None
        self.event_loop_metrics = _Metrics()
        self._middleware_registry = MiddlewareRegistry()


class AlternatingMatcher:
    """A deterministic matcher: every other Card of the graph is relevant, the rest are not.

    Arbitrary on purpose. The claim under test is about what the plugin writes, not about which Card it picks, and a
    split verdict is what makes the delivery remove and fold instead of passing the call through untouched. Alternating
    by position rather than by content guarantees the split on every graph of more than one Card, which a
    content-derived score does not.
    """

    def score(self, question: str, descriptions: Sequence[str]) -> Sequence[float]:
        """Return one similarity per Description, in the order received."""
        return [0.9 if index % 2 == 0 else 0.1 for index, _ in enumerate(descriptions)]


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
def conversations(draw: st.DrawFn) -> list[dict[str, Any]]:
    """A conversation of closed turns plus the open turn carrying the live question.

    Each closed turn opens on a plain user ask and may carry a tool pair before its assistant answer; that pair's result
    is sometimes an offloader's placeholder naming a reference, which is what gives the ``AfterToolCallEvent`` half and
    ``expand_artifact`` something to work with. A message now and then arrives with no ``tracking_id``, the
    derivation-lag shape, which yields a turn the graph cannot address.
    """
    messages: list[dict[str, Any]] = []

    for _ in range(draw(st.integers(min_value=1, max_value=4))):
        identity = f"m{len(messages)}"
        messages.append(user(draw(TEXT), identity if draw(st.booleans()) else None))

        if draw(st.booleans()):
            pair_id = f"tu-{len(messages)}"
            messages.append(tool_use(pair_id, draw(st.sampled_from(TOOL_NAMES)), f"m{len(messages)}"))
            body = draw(TEXT)
            if draw(st.booleans()):
                # What an offloader leaves behind: the preview, then the address of the content it took away.
                body = f"{body}\n[table: 4 rows, 900 bytes | ref: mem_1_{pair_id}_0]"
            messages.append(tool_result(pair_id, body, f"m{len(messages)}"))

        messages.append(assistant(draw(TEXT), f"m{len(messages)}"))

    # The open turn: the question the model is about to answer, which no Card covers yet.
    messages.append(user(draw(TEXT), f"m{len(messages)}"))
    return messages


def plugin() -> ContextGraph:
    """A plugin configured so the choice is a real one: every Card scored, the budget running out partway down."""
    return ContextGraph(matcher=AlternatingMatcher(), min_cards=1, body_budget=BODY_BUDGET)


def context_over(graph: ContextGraph, agent: _Agent) -> InvokeModelContext:
    """A real ``InvokeModelContext`` whose ``messages`` is the live history itself.

    Handing the delivery the live list is deliberate: it is the arrangement in which an in-place removal would be
    indistinguishable from a correct one inside the call, and visible only in ``agent.messages``.
    """
    return InvokeModelContext(
        agent=agent,
        messages=agent.messages,
        system_prompt="a system prompt",
        tool_specs=[{"name": name, "description": "runs", "inputSchema": {}} for name in TOOL_NAMES],
        tool_choice=None,
        invocation_state={"key": "value"},
        model=object(),
        projected_input_tokens=1234,
        dynamic_trailing_blocks=0,
    )


def tool_context(agent: _Agent) -> ToolContext:
    """The context a retrieval tool is called with, carrying the agent whose graph it reads."""
    return ToolContext(
        tool_use={"toolUseId": "tu-retrieval", "name": "expand_card", "input": {}},
        agent=agent,  # type: ignore[arg-type]
        invocation_state={},
    )


def references_of(messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Every offloaded tool result of ``messages``, as ``(tool_use_id, preview text)`` pairs."""
    found: list[tuple[str, str]] = []
    for message in messages:
        for block in message.get("content") or ():
            result = block.get("toolResult") if isinstance(block, dict) else None
            if not isinstance(result, dict):
                continue
            text = "".join(part.get("text", "") for part in result.get("content") or ())
            if "ref: " in text:
                found.append((str(result.get("toolUseId")), text))
    return found


def names_by_tool_use_id(messages: list[dict[str, Any]]) -> dict[str, str]:
    """The tool name behind each ``toolUseId``, so the artifact half is fired with the name the pair carries."""
    names: dict[str, str] = {}
    for message in messages:
        for block in message.get("content") or ():
            use = block.get("toolUse") if isinstance(block, dict) else None
            if isinstance(use, dict):
                names[str(use.get("toolUseId"))] = str(use.get("name") or "")
    return names


def fire_artifact_halves(graph: ContextGraph, agent: _Agent) -> None:
    """Run the ``AfterToolCallEvent`` half once per offloaded result of the history."""
    names = names_by_tool_use_id(agent.messages)
    for tool_use_id, text in references_of(agent.messages):
        graph._on_after_tool_call(
            AfterToolCallEvent(
                agent=agent,  # type: ignore[arg-type]
                selected_tool=None,
                tool_use={"toolUseId": tool_use_id, "name": names.get(tool_use_id, "run_query"), "input": {}},
                invocation_state={},
                result={"toolUseId": tool_use_id, "status": "success", "content": [{"text": text}]},
            )
        )


def call_retrieval_tools(graph: ContextGraph, agent: _Agent) -> None:
    """Call at least one retrieval tool, and all three whenever the conversation offers something to ask for.

    Every one of them is called for its effect on the graph, not for its answer: a Resolution raise and a fed-back Note
    both live in the Graph State, and neither may reach the history.
    """
    context = tool_context(agent)
    state = graph._states[agent]

    titles = [title for title, card in state.cards.items() if card.kind == "subject"]
    asyncio.run(graph.expand_card(titles=[titles[0] if titles else "no such turn"], tool_context=context))

    asyncio.run(graph.find_context(need=first_question(agent.messages), tool_context=context))

    artifacts = [card.reference for card in state.cards.values() if card.kind == "artifact" and card.reference]
    if artifacts:
        asyncio.run(graph.expand_artifact(reference=artifacts[0], tool_context=context))


def first_question(messages: list[dict[str, Any]]) -> str:
    """The text of the first user ask, which is as good a ``need`` as any: the ranking is not under test here."""
    for message in messages:
        if message.get("role") != "user":
            continue
        for block in message.get("content") or ():
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                return block["text"]
    return "anything"


def deliver(graph: ContextGraph, context: InvokeModelContext) -> InvokeModelContext:
    """Run one delivery to completion. Synchronous, so Hypothesis drives the examples and not an event loop policy."""
    return asyncio.run(graph._projection.deliver(context))


def run_lifecycle(graph: ContextGraph, agent: _Agent, incoming: list[dict[str, Any]]) -> None:
    """Walk ``incoming`` into ``agent`` message by message, then run the read half, the delivery and the tools."""
    for message in incoming:
        agent.messages.append(message)
        graph._on_message_added(MessageAddedEvent(agent=agent, message=message))  # type: ignore[arg-type]

    fire_artifact_halves(graph, agent)
    graph._on_before_invocation(BeforeInvocationEvent(agent=agent, messages=agent.messages))  # type: ignore[arg-type]

    deliver(graph, context_over(graph, agent))
    call_retrieval_tools(graph, agent)
    # The autonomous tool loop's next model call: the same frozen choice, a second delivery over the same history.
    deliver(graph, context_over(graph, agent))


def assert_history_untouched(
    agent: _Agent,
    history: list[dict[str, Any]],
    identities: list[int],
    before: list[dict[str, Any]],
) -> None:
    """Assert ``agent.messages`` is the list it was, holding the dicts it held, with the content it had."""
    # The same list object: nothing was rebound, and no copy was quietly put in its place.
    assert agent.messages is history
    # The same dicts, by identity: no message was replaced by an edited copy, and none was inserted or dropped.
    assert [id(message) for message in agent.messages] == identities
    # The same keys and values, to any depth: no block appended, no ``tracking_id`` rewritten, no field added.
    assert agent.messages == before
    # Requirement 1.10 read from the same angle: nothing of the graph is written to message metadata.
    assert [sorted(message) for message in agent.messages] == [sorted(message) for message in before]
    # The other two places the graph is forbidden to write: the agent's own state stays as it was found.
    assert agent.state is None


@given(incoming=conversations())
@PROPERTY_SETTINGS
def test_the_full_lifecycle_leaves_the_live_history_byte_identical(incoming):
    """Requirements 1.3, 1.4, 3.4: four handlers, a delivery and a retrieval tool later, the history is untouched."""
    graph = plugin()
    agent = _Agent([])
    history = agent.messages

    run_lifecycle(graph, agent, incoming)

    assert_history_untouched(agent, history, [id(message) for message in incoming], copy.deepcopy(incoming))


@given(incoming=conversations())
@PROPERTY_SETTINGS
def test_the_lifecycle_over_a_restored_history_leaves_it_byte_identical(incoming):
    """The rebuild-by-scan route: a process that inherited the conversation writes to it no more than the other one.

    The same claim off the other derivation path — the graph arrives by one scan over a history that fired no event —
    which is also the path that reads the offloaded previews out of the messages themselves.
    """
    graph = plugin()
    agent = _Agent(incoming)
    history = agent.messages
    identities = [id(message) for message in incoming]
    before = copy.deepcopy(incoming)

    # A restore populates ``agent.messages`` directly; the next message added is what the fresh process sees first.
    graph._on_message_added(MessageAddedEvent(agent=agent, message=incoming[-1]))  # type: ignore[arg-type]
    fire_artifact_halves(graph, agent)
    graph._on_before_invocation(BeforeInvocationEvent(agent=agent, messages=agent.messages))  # type: ignore[arg-type]
    deliver(graph, context_over(graph, agent))
    call_retrieval_tools(graph, agent)
    deliver(graph, context_over(graph, agent))

    assert_history_untouched(agent, history, identities, before)


@given(incoming=conversations())
@PROPERTY_SETTINGS
def test_a_collapsing_delivery_builds_its_own_list_rather_than_filtering_the_history(incoming):
    """The mechanism behind the property: a delivery that removes returns a new list over the same message objects.

    Asserted here because it is the one way the history could plausibly be mutated — a removal implemented as a filter
    in place would read as correct from inside the call and would be visible only in ``agent.messages``.
    """
    graph = plugin()
    agent = _Agent([])
    history = agent.messages
    identities = [id(message) for message in incoming]
    before = copy.deepcopy(incoming)

    for message in incoming:
        agent.messages.append(message)
        graph._on_message_added(MessageAddedEvent(agent=agent, message=message))  # type: ignore[arg-type]
    fire_artifact_halves(graph, agent)
    graph._on_before_invocation(BeforeInvocationEvent(agent=agent, messages=agent.messages))  # type: ignore[arg-type]

    delivered = deliver(graph, context_over(graph, agent))

    if delivered.messages is not history:
        # A removal took place: the delivered list is its own, holding our message objects — the last one aside, which
        # the fold replaces by a copy carrying the trailing block — and the history kept every one of them.
        assert set(map(id, delivered.messages[:-1])) <= set(identities)
        assert len(delivered.messages) <= len(history)
    assert_history_untouched(agent, history, identities, before)
