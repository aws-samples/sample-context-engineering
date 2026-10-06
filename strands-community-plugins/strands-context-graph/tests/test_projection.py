"""Unit tests of the delivery: the one handler where the messages sent to the provider change.

Every test here drives ``Projection.deliver`` with a real ``InvokeModelContext`` and a real ``MiddlewareRegistry``. The
handler composes the removal, the SDK's injection primitive and the compaction, and the assertions worth making are
exactly about that composition — which context object comes back, which fields moved, and what happens when one of the
two halves raises. A double over the primitive would hide the field it increments.

The agent is a stub carrying the three members the path reads: ``messages`` (asserted untouched), ``state`` (read by the
primitive when it builds its ``InjectionContext``) and ``_middleware_registry`` (the registration seam). It is a plain
class so it is weak-referenceable, which the per-agent state map requires.
"""

import copy
import logging
from types import MappingProxyType

import pytest
from strands._middleware.registry import MiddlewareRegistry
from strands._middleware.stages import InvokeModelContext, InvokeModelStage

from strands_context_graph.projection import Projection, current_turn_ids
from strands_context_graph.state import Card, CardChoice, TurnChoice, _GraphState

DESCRIPTION_TOKENS = 100
"""The plugin's own default, so the budgeting these tests see is the one production sees."""

COLLAPSED_TITLE = "Collapse me"
"""Title of the one Card the choice puts at description in every fixture below."""


class _Agent:
    """The three members the delivery path reads off an agent, and nothing else."""

    def __init__(self, messages):
        self.messages = messages
        self.state = None
        self._middleware_registry = MiddlewareRegistry()


def user(text, tracking_id):
    """A plain user ask: a turn boundary, since it carries no tool result."""
    return {"role": "user", "content": [{"text": text}], "tracking_id": tracking_id}


def assistant(text, tracking_id):
    """An assistant answer, part of its turn's dialogue."""
    return {"role": "assistant", "content": [{"text": text}], "tracking_id": tracking_id}


def tool_use(tool_use_id, tracking_id):
    """The assistant half of a tool pair."""
    return {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": tool_use_id, "name": "run_query", "input": {}}}],
        "tracking_id": tracking_id,
    }


def tool_result(tool_use_id, tracking_id):
    """The user half of a tool pair. Not a turn boundary: it carries a tool result."""
    return {
        "role": "user",
        "content": [{"toolResult": {"toolUseId": tool_use_id, "content": [{"text": "42"}], "status": "success"}}],
        "tracking_id": tracking_id,
    }


def history():
    """Three turns: one kept whole, one the choice collapses, and the open turn carrying the live question.

    The collapsed turn sits in the middle on purpose. The removal never drops the first user message, so a Card over
    turn one could not leave the call whole and the compaction would then fold nothing.
    """
    return [
        user("Warm up", "m0"),
        assistant("Warming up", "m1"),
        user(COLLAPSED_TITLE, "m2"),
        assistant("Collapsing", "m3"),
        user("Latest question", "m4"),
    ]


def graph():
    """A graph over :func:`history`: turn one full, turn two at description, the open turn uncovered."""
    state = _GraphState()
    state.cards["Warm up"] = card("Warm up", 1, ("m0", "m1"))
    state.cards[COLLAPSED_TITLE] = card(COLLAPSED_TITLE, 2, ("m2", "m3"), description="Asked to be collapsed.")
    state.turn = 3
    state.choice = TurnChoice(
        by_title=MappingProxyType({COLLAPSED_TITLE: CardChoice(dialogue="description", evidence="full")}),
        full_pass=False,
    )
    return state


def card(title, turn, dialogue_ids, *, evidence_ids=(), description=""):
    """A Card carrying only the fields the removal and the final block read."""
    return Card(
        title=title,
        kind="subject",
        turn=turn,
        dialogue_ids=tuple(dialogue_ids),
        evidence_ids=tuple(evidence_ids),
        pairs=(),
        tool_names=frozenset(),
        references=(),
        numeric_lines=(),
        tags=(),
        description=description,
    )


def context_over(messages, agent=None):
    """A real ``InvokeModelContext`` over ``messages``, with every field set to something recognizable."""
    return InvokeModelContext(
        agent=agent if agent is not None else _Agent(messages),
        messages=messages,
        system_prompt="a system prompt",
        tool_specs=[{"name": "run_query", "description": "runs", "inputSchema": {}}],
        tool_choice=None,
        invocation_state={"key": "value"},
        model=object(),
        projected_input_tokens=1234,
        dynamic_trailing_blocks=2,
    )


def projection(state=None, messages=None, *, description_tokens=DESCRIPTION_TOKENS):
    """A ``Projection`` wired to one agent, plus the agent's context. ``state=None`` leaves the agent unregistered."""
    messages = history() if messages is None else messages
    agent = _Agent(messages)
    states = {}
    if state is not None:
        states[agent] = state
    return Projection(states, description_tokens=description_tokens, retrieval_tools=lambda: ("expand_card", "expand_artifact", "find_context")), context_over(messages, agent)


def warnings_of(caplog):
    """The warning records the delivery itself emitted, the primitive's own fail-open logging excluded."""
    return [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING and record.name == "strands_context_graph.projection"
    ]


# ---- the open turn --------------------------------------------------------------------------------


def test_current_turn_ids_reads_the_trailing_range():
    assert current_turn_ids(history()) == frozenset({"m4"})


def test_current_turn_ids_is_empty_without_a_boundary():
    assert current_turn_ids([assistant("No turn opened", "m0")]) == frozenset()


# ---- the identity deliveries ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unregistered_agent_delivers_the_received_context_by_identity():
    handler, context = projection()

    assert await handler.deliver(context) is context


@pytest.mark.asyncio
async def test_full_pass_delivers_the_received_context_by_identity():
    state = graph()
    state.choice = TurnChoice(by_title=MappingProxyType({}), full_pass=True)
    handler, context = projection(state)

    delivered = await handler.deliver(context)

    assert delivered is context
    assert delivered.messages is context.messages
    assert delivered.dynamic_trailing_blocks == 2


@pytest.mark.asyncio
async def test_a_choice_that_collapses_nothing_delivers_the_received_context_by_identity():
    state = graph()
    state.choice = TurnChoice(
        by_title=MappingProxyType({COLLAPSED_TITLE: CardChoice(dialogue="full", evidence="full")}),
        full_pass=False,
    )
    handler, context = projection(state)

    delivered = await handler.deliver(context)

    assert delivered is context
    assert delivered.messages is context.messages


@pytest.mark.asyncio
async def test_a_card_covering_only_the_open_turn_delivers_by_identity():
    """The open turn is subtracted from the request, so the only Card at description asks for nothing."""
    state = _GraphState()
    state.cards["Latest question"] = card("Latest question", 3, ("m4",), description="The live question.")
    state.choice = TurnChoice(
        by_title=MappingProxyType({"Latest question": CardChoice(dialogue="description", evidence="full")}),
        full_pass=False,
    )
    handler, context = projection(state)

    assert await handler.deliver(context) is context


# ---- the delivery that changes something ---------------------------------------------------------


@pytest.mark.asyncio
async def test_delivery_replaces_only_messages_and_dynamic_trailing_blocks():
    handler, context = projection(graph())
    before = copy.deepcopy(context.messages)

    delivered = await handler.deliver(context)

    assert delivered is not context
    # ``m4`` is the message the block was folded into. The SDK's fold rebuilds that one message from ``role``,
    # ``content`` and ``metadata``, so the per-call copy of it carries no ``tracking_id`` — the identity lives on in
    # ``agent.messages``, which this path never touches.
    assert [message.get("tracking_id") for message in delivered.messages] == ["m0", "m1", None]
    assert delivered.dynamic_trailing_blocks == 3
    # The received context is untouched in place: same list object, same messages, same counter.
    assert context.messages == before
    assert context.dynamic_trailing_blocks == 2
    # Every other field carried over, by identity where the field is a reference.
    assert delivered.agent is context.agent
    assert delivered.system_prompt == context.system_prompt
    assert delivered.tool_specs is context.tool_specs
    assert delivered.tool_choice == context.tool_choice
    assert delivered.invocation_state is context.invocation_state
    assert delivered.model is context.model
    assert delivered.projected_input_tokens == context.projected_input_tokens


@pytest.mark.asyncio
async def test_the_block_is_folded_into_the_last_user_message_with_no_role():
    handler, context = projection(graph())

    delivered = await handler.deliver(context)

    assert len(delivered.messages) == 3  # No message inserted for the block.
    last = delivered.messages[-1]
    assert last["role"] == "user"
    assert last["content"][0] == {"text": "Latest question"}
    assert "role" not in last["content"][-1]
    folded = last["content"][-1]["text"]
    assert COLLAPSED_TITLE in folded
    assert "Asked to be collapsed." in folded
    # The Card kept whole contributes nothing, so its title is in the messages and not in the block.
    assert "Warm up" not in folded


@pytest.mark.asyncio
async def test_the_agent_history_is_never_read_or_written():
    handler, context = projection(graph())
    agent_messages = context.agent.messages
    before = copy.deepcopy(agent_messages)

    await handler.deliver(context)

    assert context.agent.messages is agent_messages
    assert context.agent.messages == before


@pytest.mark.asyncio
async def test_two_runs_over_the_same_inputs_deliver_the_same_messages():
    state = graph()
    first, _ = projection(state)
    first_delivered = await first.deliver(context_over(history()))
    second, _ = projection(state)
    second_delivered = await second.deliver(context_over(history()))

    assert first_delivered.messages == second_delivered.messages
    assert first_delivered.dynamic_trailing_blocks == second_delivered.dynamic_trailing_blocks


@pytest.mark.asyncio
async def test_the_fold_runs_on_a_tool_result_turn_too():
    """``trigger="everyTurn"``: the autonomous loop's calls carry the block, not just the turn's first call."""
    messages = [*history(), tool_use("t1", "m5"), tool_result("t1", "m6")]
    handler, context = projection(graph(), messages)

    delivered = await handler.deliver(context)

    assert [message.get("tracking_id") for message in delivered.messages] == ["m0", "m1", "m4", "m5", None]
    assert delivered.dynamic_trailing_blocks == 3
    assert COLLAPSED_TITLE in delivered.messages[-1]["content"][-1]["text"]


# ---- atomic degradation --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_failing_compaction_degrades_to_the_received_context(monkeypatch, caplog):
    def boom(*_args, **_kwargs):
        raise RuntimeError("compaction failed")

    monkeypatch.setattr("strands_context_graph.projection.render_final_block", boom)
    handler, context = projection(graph())

    with caplog.at_level(logging.WARNING):
        delivered = await handler.deliver(context)

    # Not merely equal: the removal must not reach the provider without the block that describes it.
    assert delivered is context
    assert delivered.messages is context.messages
    assert len(warnings_of(caplog)) == 1


@pytest.mark.asyncio
async def test_a_failing_removal_degrades_to_the_received_context(monkeypatch, caplog):
    def boom(*_args, **_kwargs):
        raise RuntimeError("removal failed")

    monkeypatch.setattr("strands_context_graph.projection.apply_removal", boom)
    handler, context = projection(graph())

    with caplog.at_level(logging.WARNING):
        delivered = await handler.deliver(context)

    assert delivered is context
    assert len(warnings_of(caplog)) == 1


@pytest.mark.asyncio
async def test_the_next_call_retries_after_a_failure(monkeypatch):
    calls = []

    def once(*args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            raise RuntimeError("transient")
        return "<collapsed_turns>\n- " + COLLAPSED_TITLE + "\n</collapsed_turns>"

    monkeypatch.setattr("strands_context_graph.projection.render_final_block", once)
    handler, context = projection(graph())

    assert await handler.deliver(context) is context

    retried = await handler.deliver(context_over(history(), context.agent))

    assert retried is not context
    assert len(calls) == 2
    assert retried.dynamic_trailing_blocks == 3


@pytest.mark.asyncio
async def test_no_failure_state_survives_the_call(monkeypatch):
    """The in-flight record is reset around the one await, so a later render sees no stale error."""
    from strands_context_graph import projection as module

    def boom(*_args, **_kwargs):
        raise RuntimeError("compaction failed")

    monkeypatch.setattr(module, "render_final_block", boom)
    handler, context = projection(graph())

    await handler.deliver(context)

    assert module._DELIVERY.get() is None


# ---- registration ---------------------------------------------------------------------------------


def test_register_puts_the_delivery_first_on_the_input_phase():
    agent = _Agent(history())

    async def other(context):
        return context

    agent._middleware_registry.add_middleware(InvokeModelStage.Input, other)
    handler = Projection({}, description_tokens=DESCRIPTION_TOKENS, retrieval_tools=lambda: ("expand_card", "expand_artifact", "find_context"))

    handler.register(agent)

    tagged = agent._middleware_registry._handlers[InvokeModelStage]
    assert len(tagged) == 2
    assert tagged[0].phase == "input"
    # The registry wraps an input handler in an adapter, so the delivery is identified through its closure.
    assert handler.deliver in [cell.cell_contents for cell in tagged[0].handler.__closure__]
    assert other in [cell.cell_contents for cell in tagged[1].handler.__closure__]
