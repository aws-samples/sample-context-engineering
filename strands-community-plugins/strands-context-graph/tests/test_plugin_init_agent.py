"""Unit tests of what ``init_agent`` wires, and of the ``NullConversationManager`` precondition.

Driven through a real ``Agent`` rather than a fake, because the claims under test are about the agent the SDK assembles:
the tool registry is filled by the plugin registry's own discovery, not by ``init_agent``, and the hook registry and the
middleware registry already carry the SDK's own entries before this plugin adds anything. Every count is therefore taken
against a baseline agent built identically *without* the plugin, so the assertions are about the plugin's delta and not
about whatever the SDK happens to wire on its own.

The model is a stub that raises if it is ever reached: nothing here runs a turn, and the no-model-call contract makes
that a fact rather than a hope.
"""

import warnings

import pytest
from strands import Agent
from strands._middleware.stages import InvokeModelStage
from strands.agent.conversation_manager import (
    NullConversationManager,
    SlidingWindowConversationManager,
    SummarizingConversationManager,
)
from strands.hooks.events import AfterToolCallEvent, BeforeInvocationEvent, MessageAddedEvent
from strands.models.model import Model

from strands_context_graph import ContextGraph

SYSTEM_PROMPT = "You are a helpful assistant."

HISTORY = [
    {"role": "user", "content": [{"text": "what did the migration cost"}], "tracking_id": "m0"},
    {"role": "assistant", "content": [{"text": "about 412 USD"}], "tracking_id": "m1"},
]


class _Model(Model):
    """A model that cannot be called, which is the point: ``init_agent`` reaches no model."""

    def update_config(self, **kwargs):
        """Accept anything; nothing reads it."""

    def get_config(self):
        """No configuration to report."""
        return {}

    async def stream(self, *args, **kwargs):
        """Fail loudly if a turn is ever run from these tests."""
        raise AssertionError("the model was called")
        yield

    async def structured_output(self, *args, **kwargs):
        """Fail loudly if a turn is ever run from these tests."""
        raise AssertionError("the model was called")
        yield


def build(*, plugin=None, manager=None, messages=None):
    """Build an agent, with the plugin when one is given and otherwise as the baseline to compare against."""
    return Agent(
        model=_Model(),
        system_prompt=SYSTEM_PROMPT,
        conversation_manager=manager if manager is not None else NullConversationManager(),
        messages=[dict(message) for message in messages] if messages is not None else None,
        plugins=[plugin] if plugin is not None else [],
    )


def wire(*, plugin=None, manager=None, messages=None):
    """Build an agent and return it with the warnings its wiring produced."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        agent = build(plugin=plugin, manager=manager, messages=messages)
    return agent, caught


def callbacks_for(agent, event_type):
    """The hook callbacks registered on ``agent`` for one event type, SDK's own included."""
    return [entry.callback for entry in agent.hooks._registered_callbacks.get(event_type, [])]


def input_handlers(agent):
    """The ``InvokeModelStage`` handler list, which is where the delivery is registered."""
    return agent._middleware_registry._handlers.get(InvokeModelStage, [])


def test_exactly_the_three_retrieval_tools_are_registered():
    """Requirement 1.2: the three ``@tool`` members and no fourth tool."""
    added = set(build(plugin=ContextGraph()).tool_registry.registry) - set(build().tool_registry.registry)

    assert added == {"expand_card", "expand_artifact", "find_context"}


def test_exactly_one_handler_of_each_hook_type():
    """Requirement 1.1: one ``BeforeInvocationEvent``, one ``MessageAddedEvent``, one ``AfterToolCallEvent``."""
    graph = ContextGraph()
    agent = build(plugin=graph)

    for event_type, handler in (
        (BeforeInvocationEvent, graph._on_before_invocation),
        (MessageAddedEvent, graph._on_message_added),
        (AfterToolCallEvent, graph._on_after_tool_call),
    ):
        assert callbacks_for(agent, event_type).count(handler) == 1


def test_exactly_one_input_middleware_handler_is_added():
    """Requirement 1.1: one ``InvokeModelStage.Input`` handler, and it is the stage's first."""
    graph = ContextGraph()
    agent = build(plugin=graph)

    handlers = input_handlers(agent)
    assert len(handlers) - len(input_handlers(build())) == 1
    # The registration seam is the delivery's: index zero is where ``Projection.register`` moves its own handler.
    assert handlers[0].phase == "input"


def test_no_other_hook_type_gains_a_handler():
    """Requirement 1.1 read from the other side: four engagement points, not five."""
    wired = build(plugin=ContextGraph()).hooks._registered_callbacks
    baseline = build().hooks._registered_callbacks

    grew = {
        event_type.__name__
        for event_type, callbacks in wired.items()
        if len(callbacks) != len(baseline.get(event_type, []))
    }
    assert grew == {"BeforeInvocationEvent", "MessageAddedEvent", "AfterToolCallEvent"}


def test_the_system_prompt_is_untouched():
    """Requirement 1.3: the prompt comes out as it went in."""
    assert build(plugin=ContextGraph()).system_prompt == SYSTEM_PROMPT


def test_the_history_is_untouched():
    """Requirements 1.3 and 1.10: the live messages, Durable Identities included, are left exactly as given."""
    assert build(plugin=ContextGraph(), messages=HISTORY).messages == HISTORY


def test_nothing_but_the_three_tools_changes_in_the_registry():
    """Requirement 1.3: no existing entry is replaced or reconfigured."""
    wired = build(plugin=ContextGraph()).tool_registry.registry
    baseline = build().tool_registry.registry

    assert {name: spec for name, spec in wired.items() if name in baseline} == baseline


def test_a_wired_agent_presents_a_graph_with_no_card():
    """The state is created eagerly, and a fresh state is the full pass that behaves as no plugin at all."""
    graph = ContextGraph()
    agent = build(plugin=graph)

    state = graph._states[agent]
    assert state.cards == {}
    assert state.turn == 0
    assert state.choice.full_pass is True


def test_null_conversation_manager_stays_silent():
    """Requirement 1.8: the precondition is met, so nothing is said."""
    _, caught = wire(plugin=ContextGraph(), manager=NullConversationManager())

    assert caught == []


@pytest.mark.parametrize(
    "manager",
    [
        SlidingWindowConversationManager(window_size=2),
        SummarizingConversationManager(),
    ],
    ids=["sliding_window", "summarizing"],
)
def test_a_trimming_manager_warns_exactly_once_and_names_it(manager):
    """Requirement 1.6: one developer-facing notice, naming the manager and what it can drop."""
    _, caught = wire(plugin=ContextGraph(), manager=manager)

    assert len(caught) == 1
    assert issubclass(caught[0].category, UserWarning)
    message = str(caught[0].message)
    assert type(manager).__name__ in message
    assert "NullConversationManager" in message
    assert "fold" in message


def test_the_notice_is_a_warning_and_not_a_log_record(caplog):
    """Developer-facing, so it is controllable with ``-W`` and is not routed to an operator's logs."""
    with caplog.at_level("WARNING"), pytest.warns(UserWarning):
        build(plugin=ContextGraph(), manager=SlidingWindowConversationManager(window_size=2))

    assert caplog.records == []


def test_the_warned_agent_is_wired_exactly_like_any_other():
    """Requirement 1.7: the notice degrades and never blocks."""
    graph = ContextGraph()

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        agent = build(plugin=graph, manager=SlidingWindowConversationManager(window_size=2))

    assert set(agent.tool_registry.registry) >= {"expand_card", "expand_artifact", "find_context"}
    assert callbacks_for(agent, MessageAddedEvent).count(graph._on_message_added) == 1
    assert callbacks_for(agent, AfterToolCallEvent).count(graph._on_after_tool_call) == 1
    assert callbacks_for(agent, BeforeInvocationEvent).count(graph._on_before_invocation) == 1
    assert input_handlers(agent)[0].phase == "input"


def test_the_conversation_manager_is_left_alone():
    """The notice names a risk; it does not remove, replace or reconfigure the manager."""
    manager = SlidingWindowConversationManager(window_size=2)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        agent = build(plugin=ContextGraph(), manager=manager)

    assert agent.conversation_manager is manager
    assert agent.conversation_manager.window_size == 2


def test_one_instance_on_two_agents_keeps_the_states_independent():
    """Requirement 1.5: two agents wired from one instance never share a Card."""
    graph = ContextGraph()
    first = build(plugin=graph)
    second = build(plugin=graph)

    graph._states[first].cards["a title"] = object()

    assert graph._states[second].cards == {}
    assert graph._states[first] is not graph._states[second]


def test_one_instance_on_two_agents_registers_one_handler_of_each_type_per_agent():
    """Requirement 1.5: the second wiring adds one of each there, and nothing more to the first."""
    graph = ContextGraph()
    agents = [build(plugin=graph), build(plugin=graph)]

    for agent in agents:
        assert callbacks_for(agent, MessageAddedEvent).count(graph._on_message_added) == 1
        assert callbacks_for(agent, AfterToolCallEvent).count(graph._on_after_tool_call) == 1
        assert callbacks_for(agent, BeforeInvocationEvent).count(graph._on_before_invocation) == 1
        assert len(input_handlers(agent)) - len(input_handlers(build())) == 1


def test_an_agent_exposing_no_conversation_manager_stays_silent():
    """The quiet direction: an agent with nothing that trims is told nothing."""

    class _Agent:
        messages = []

        def add_hook(self, callback, event_type=None):
            """Accept the registration; these assertions are about the notice."""

        _middleware_registry = None

    agent = _Agent()
    agent._middleware_registry = build()._middleware_registry

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ContextGraph()._warn_on_destructive_manager(agent)

    assert caught == []
