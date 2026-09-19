"""Unit tests of the ordering rule against a memory manager: what is detected, and what is said about it.

The rule is that the graph removes messages before a memory manager folds its block into the same
``InvokeModelStage.Input``. Half of it the plugin controls, by registering its delivery at index zero; the other half it
can only report on, which is the notice under test here.

Everything is driven through ``ContextGraph.init_agent`` with a real ``MiddlewareRegistry``, so the position the plugin
claims is the position the registry actually holds. The memory manager is a stub carrying the two members the detection
reads — memory folding is checked by member, not by type — and a plain class, so the per-agent state map can weakly
reference the agent.
"""

import warnings

import pytest
from strands._middleware.registry import MiddlewareRegistry
from strands._middleware.stages import InvokeModelStage
from strands.agent.conversation_manager import NullConversationManager

from strands_context_graph.plugin import ContextGraph


class _MemoryManager:
    """The two members the detection reads off a memory manager, and nothing else.

    Args:
        injects: What ``_injection_config`` holds. ``False`` is injection turned off, which folds nothing.
    """

    def __init__(self, injects=True):
        self._injection_config = {} if injects else False

    async def _provide_memory_context(self, messages, config):
        """The render callback that makes this object a memory manager, by member."""
        return "remembered"


class _PluginRegistry:
    """The private map the plugin reads to find what is already wired to the agent."""

    def __init__(self, plugins=None):
        self._plugins = dict(plugins or {})


class _Agent:
    """The members ``init_agent`` and the detection read, and nothing else."""

    def __init__(self, *, plugins=None, memory_manager=None):
        self.messages = []
        self.state = None
        self._middleware_registry = MiddlewareRegistry()
        self._plugin_registry = _PluginRegistry(plugins)
        self.memory_manager = memory_manager
        self.conversation_manager = NullConversationManager()
        self.hooks = []

    def add_hook(self, callback, event_type=None):
        """Record the hook registration, which is all these tests need of it."""
        self.hooks.append((event_type, callback))


async def other_fold(context):
    """A second input handler, standing in for a memory manager's fold in the same stage."""
    return context


def wire(agent):
    """Wire a fresh plugin to ``agent`` and return the warnings the wiring produced."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ContextGraph().init_agent(agent)
    return caught


def input_handlers(agent):
    """The stage's handler list, which is where the claimed ordering is either true or not."""
    return agent._middleware_registry._handlers[InvokeModelStage]


def test_no_memory_manager_stays_silent():
    """Nothing to order against, so nothing is said."""
    assert wire(_Agent()) == []


def test_memory_manager_behind_the_delivery_stays_silent():
    """A fold already in the stage ends up behind the delivery, which is the rule holding."""
    agent = _Agent(plugins={"strands:memory": _MemoryManager()})
    agent._middleware_registry.add_middleware(InvokeModelStage.Input, other_fold)

    caught = wire(agent)

    assert caught == []
    assert len(input_handlers(agent)) == 2


def test_memory_manager_with_no_observable_fold_warns_once():
    """A memory manager whose fold is not in this stage is the case the plugin cannot claim ordering for."""
    caught = wire(_Agent(plugins={"strands:memory": _MemoryManager()}))

    assert len(caught) == 1
    assert issubclass(caught[0].category, UserWarning)


def test_the_notice_names_the_ordering_requirement_and_the_risk():
    """Both halves of Requirement 9.6 are in the text: what the order must be, and what a violation costs."""
    message = str(wire(_Agent(plugins={"strands:memory": _MemoryManager()}))[0].message)

    assert "InvokeModelStage" in message
    assert "before" in message
    assert "projected out" in message


def test_two_memory_managers_still_warn_exactly_once():
    """The notice is about the agent's wiring, not about each manager on it."""
    plugins = {"strands:memory": _MemoryManager(), "other:memory": _MemoryManager()}

    assert len(wire(_Agent(plugins=plugins))) == 1


def test_memory_manager_reached_through_the_agent_attribute_is_detected():
    """The SDK wires a memory manager through its own parameter too, where the plugin registry shows nothing."""
    assert len(wire(_Agent(memory_manager=_MemoryManager()))) == 1


def test_injection_turned_off_stays_silent():
    """Nothing is folded, so no block can be projected out and there is no risk to name."""
    assert wire(_Agent(plugins={"strands:memory": _MemoryManager(injects=False)})) == []


def test_a_plugin_that_is_not_a_memory_manager_stays_silent():
    """Detection is by member: an ordinary plugin carries neither member and is not one."""
    assert wire(_Agent(plugins={"strands:other": object()})) == []


def test_an_agent_without_the_private_registry_stays_silent():
    """An agent exposing no plugin map reads as an agent with no memory manager, the quiet direction."""
    agent = _Agent()
    del agent._plugin_registry

    assert wire(agent) == []


def test_the_delivery_is_registered_first_whether_or_not_the_notice_fires():
    """The notice degrades and never blocks: the warned agent is wired like any other."""
    agent = _Agent(plugins={"strands:memory": _MemoryManager()})
    agent._middleware_registry.add_middleware(InvokeModelStage.Input, other_fold)
    graph = ContextGraph()

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        graph.init_agent(agent)

    handlers = input_handlers(agent)
    assert len(handlers) == 2
    # The registration seam is the delivery's: index zero is the handler the plugin just inserted there.
    assert handlers[0].phase == "input"


def test_wiring_two_agents_warns_for_each():
    """Nothing is recorded on the instance, so a second wiring site is told too."""
    graph = ContextGraph()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        graph.init_agent(_Agent(plugins={"strands:memory": _MemoryManager()}))
        graph.init_agent(_Agent(plugins={"strands:memory": _MemoryManager()}))

    assert len(caught) == 2


def test_the_notice_is_a_warning_and_not_a_log_record(caplog):
    """Developer-facing, so it is controllable with ``-W`` and is not routed to an operator's logs."""
    with caplog.at_level("WARNING"), pytest.warns(UserWarning):
        ContextGraph().init_agent(_Agent(plugins={"strands:memory": _MemoryManager()}))

    assert caplog.records == []
