"""Integration test of the whole disclosure cycle, offline, against a deterministic search double.

Validates: Requirements 12.5.

The other test files each hold one part of the mechanism still: the projection algebra, the catalog
entry, the index contract, the exposure lifecycle, the degradation paths. None of them walks the loop
the model actually walks. This one does, once, in order:

1. a first projection, where the tool the model needs arrives as a catalog entry — a name and a short
   description, with an empty closed ``inputSchema`` and no parameters to call it with;
2. a search, which is how the model says what it needs and is what exposes the tool;
3. the next projection, which carries that tool's registered full specification, parameters included;
4. the call, which succeeds against the real registered tool and renews the exposure, so the schema
   stays resident past the TTL it would otherwise have aged out of.

The point of running it as one sequence is that every step consumes what the previous one produced:
the search only exposes a name the first projection listed, the second projection only widens because
the search wrote an exposure, and the renewal is only observable because the call went through the
registered tool rather than a stand-in.

Requirement 12.5 is the other half of the file. Every search path here goes through a ``ToolIndex``
double, and the suite performs zero network calls — asserted rather than assumed: the test body runs
with socket construction, connection and name resolution all replaced by refusals, so a search
implementation that reached for the network would fail here instead of quietly needing it. The event
loop is created before that guard goes up, because ``asyncio`` builds its self-pipe out of a socket
pair at loop construction and a ban installed earlier would fail the runner rather than the code
under test. The model is a stub that raises when called, so no provider call goes out either.
"""

import asyncio
import copy
import dataclasses
import socket
from collections.abc import Coroutine, Iterator
from types import SimpleNamespace
from typing import Any, TypeVar, cast

import pytest
from strands import Agent
from strands.agent.agent import Agent as AgentType
from strands.hooks.events import BeforeToolCallEvent
from strands.models.model import Model
from strands.tools.decorator import tool
from strands.types._events import ToolResultEvent
from strands.types.tools import ToolContext, ToolSpec

from strands_progressive_tool_disclosure import ProgressiveToolDisclosure, ToolMatch
from strands_progressive_tool_disclosure._compat import InvokeModelContext
from strands_progressive_tool_disclosure.plugin import _CATALOG_SIGIL, FIND_TOOLS_NAME

EMPTY_CLOSED_SCHEMA = {"json": {"type": "object", "properties": {}, "additionalProperties": False}}
"""The schema a catalog entry carries. Its presence is how an unexposed tool is recognized."""

TTL_CYCLES = 3
"""Short on purpose: the renewal assertion has to outlive a full TTL to mean anything."""

NEED = "list the transactions of an investment account"
"""What the model says it is trying to do. A capability, not a guess at a tool name."""

_Resolved = TypeVar("_Resolved")


class _StubModel(Model):
    """A model that exists so an ``Agent`` can be constructed. A call to it is a test failure."""

    def update_config(self, **kwargs: Any) -> None:
        """Accept anything; nothing reads it."""

    def get_config(self) -> dict[str, Any]:
        """No configuration to report."""
        return {}

    async def stream(self, *args: Any, **kwargs: Any) -> Any:
        """Refuse: this test drives the plugin's own steps, never a provider call."""
        raise AssertionError("the language model was called")
        yield

    async def structured_output(self, *args: Any, **kwargs: Any) -> Any:
        """Refuse: this test drives the plugin's own steps, never a provider call."""
        raise AssertionError("the language model was called")
        yield


@tool
def list_investment_transactions(account: str, since: str) -> str:
    """List the transactions of an investment account.

    The second sentence exists to be cut: a catalog entry has a token budget, and what the model gets
    to read of this description is the boundary-cut prefix that fits it.

    Args:
        account: Investment account to report on.
        since: Earliest transaction date to include.
    """
    return f"{account}:{since}"


@tool
def send_wire(account: str, amount: str) -> str:
    """Send a wire transfer from an account.

    Args:
        account: Account to debit.
        amount: How much to send.
    """
    return f"{account}:{amount}"


class _ContainmentIndex:
    """Search double that ranks a name by how many of the need's words its full text contains.

    Deterministic and offline, which is what Requirement 12.5 asks of every search path: the ranking
    is a word count over the text ``build`` received, ties break by indexing order, and the same need
    ranks the same way on every run. It counts its calls because "the search is what exposed the tool"
    is a claim about a number of invocations, not an inference from the projection that followed.
    """

    def __init__(self) -> None:
        """Start with nothing indexed and nothing searched."""
        self.built: list[list[str]] = []
        self.searches = 0
        self._texts: dict[str, str] = {}

    def build(self, specs: list[ToolSpec]) -> None:
        """Index the offered specifications by name and full description."""
        self.built.append([spec["name"] for spec in specs])
        self._texts = {spec["name"]: f"{spec['name']} {spec.get('description', '')}".lower() for spec in specs}

    def search(self, need: str, top_k: int) -> list[ToolMatch]:
        """Rank the indexed names by how many of the need's distinct words they contain."""
        self.searches += 1
        words = {word for word in need.lower().replace("_", " ").split() if word}

        scored = [
            ToolMatch(name=name, score=float(sum(word in text for word in words))) for name, text in self._texts.items()
        ]
        ranked = [match for match in scored if match.score > 0]

        # ``sorted`` is stable, so equal scores keep the indexing order: repeated searches are equal.
        return sorted(ranked, key=lambda match: -match.score)[:top_k]


@pytest.fixture
def runner() -> Iterator[asyncio.AbstractEventLoop]:
    """Provide an event loop built *before* the network ban, and close it after the ban is lifted.

    ``asyncio`` creates its self-pipe from a socket pair when the loop is constructed. Constructing the
    loop here, as a dependency of the ban, is what keeps the ban a statement about the code under test
    rather than about the test runner.
    """
    loop = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        loop.close()


@pytest.fixture
def no_network(runner: asyncio.AbstractEventLoop, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make socket construction, connection and name resolution raise for the duration of the test.

    Requirement 12.5: the search paths are exercised through doubles and the suite performs zero
    network calls. Construction is banned rather than only connection, so an implementation that merely
    *prepares* to reach out fails here too, and the failure names the call it attempted.
    """

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the test attempted a network call")

    for attribute in ("socket", "socketpair", "create_connection", "create_server", "getaddrinfo"):
        monkeypatch.setattr(socket, attribute, refuse)


def _agent(plugin: ProgressiveToolDisclosure) -> Agent:
    """Build an offline agent carrying the two registered tools plus the plugin's search tool."""
    return Agent(
        model=_StubModel(),
        tools=[list_investment_transactions, send_wire],
        plugins=[plugin],
    )


def _model_call(agent: Agent) -> InvokeModelContext:
    """Build the invocation context of one model call, offering every registered specification.

    Offering the whole registry is what keeps the call off the passthrough path: a name the registry
    does not have, or a missing ``find_tools``, is a structural guard rather than a projection.
    """
    candidates: dict[str, Any] = {
        "agent": agent,
        "messages": [],
        "system_prompt": None,
        "tool_specs": [entry.tool_spec for entry in agent.tool_registry.registry.values()],
        "tool_choice": None,
        "invocation_state": {},
        "model": agent.model,
    }
    declared = {field.name for field in dataclasses.fields(InvokeModelContext)}

    return InvokeModelContext(**{name: value for name, value in candidates.items() if name in declared})


def _run(loop: asyncio.AbstractEventLoop, step: Coroutine[Any, Any, _Resolved]) -> _Resolved:
    """Drive one awaited step to completion on the pre-built loop."""
    return loop.run_until_complete(step)


def _projected(context: InvokeModelContext) -> dict[str, ToolSpec]:
    """Index a projection by tool name, for asking which form each tool arrived in."""
    return {spec["name"]: spec for spec in context.tool_specs}


def _call_tool(loop: asyncio.AbstractEventLoop, agent: Agent, name: str, tool_input: dict[str, Any]) -> ToolResultEvent:
    """Run the pre-call hook and then the registered tool, returning the tool's final result event.

    Both halves matter. The hook is what renews the exposure and what would cancel a call made off a
    catalog entry, and the registered tool is what turns "the schema arrived" into "the call works":
    the arguments are validated against the very specification the projection just carried.
    """
    event = BeforeToolCallEvent(
        agent=cast("AgentType", agent),
        selected_tool=None,
        tool_use={"toolUseId": "t1", "name": name, "input": tool_input},
        invocation_state={},
    )
    agent.hooks.invoke_callbacks(event)
    assert not event.cancel_tool, f"the call to '{name}' was cancelled: {event.cancel_tool}"

    async def execute() -> ToolResultEvent:
        last: Any = None
        async for item in agent.tool_registry.registry[name].stream(event.tool_use, {}):
            last = item
        return cast("ToolResultEvent", last)

    return _run(loop, execute())


def test_the_full_disclosure_cycle_runs_offline_against_a_deterministic_index_double(
    runner: asyncio.AbstractEventLoop,
    no_network: None,
) -> None:
    """Requirement 12.5: projection, search, next projection and call, with zero network calls.

    One sequence, each step consuming the previous one's output, against a deterministic ``ToolIndex``
    double and with the network unavailable.
    """
    index = _ContainmentIndex()
    # ``top_k=1`` so the step under test is unambiguous: one need, one tool exposed, and the tools the
    # need only brushes against stay catalog entries.
    plugin = ProgressiveToolDisclosure(index=index, ttl_cycles=TTL_CYCLES, catalog_tokens=20, top_k=1)
    agent = _agent(plugin)
    registered = agent.tool_registry.registry["list_investment_transactions"].tool_spec
    registry_before = copy.deepcopy({name: entry.tool_spec for name, entry in agent.tool_registry.registry.items()})

    # 1. First projection: the tool is a catalog entry, so the model can read that the capability
    # exists and cannot yet call it — there are no parameters in what it received.
    first = _projected(_run(runner, plugin._projection_handler(_model_call(agent))))
    assert set(first) == set(agent.tool_names)
    assert first[FIND_TOOLS_NAME] == agent.tool_registry.registry[FIND_TOOLS_NAME].tool_spec

    entry = first["list_investment_transactions"]
    assert entry["name"] == registered["name"]
    assert entry["inputSchema"] == EMPTY_CLOSED_SCHEMA
    # The sigil is what tells the model this is a listing: an entry carrying a real name, a readable
    # description and a valid empty schema is otherwise indistinguishable from a tool that genuinely
    # takes no arguments, and the only statement otherwise lived in find_tools' own description.
    assert entry["description"].startswith(_CATALOG_SIGIL)
    body = entry["description"][len(_CATALOG_SIGIL) :]
    assert registered["description"].startswith(body.removesuffix("..."))
    assert entry != registered
    # The index was built once, over the specifications this very call offered — nothing else.
    assert index.built == [[spec["name"] for spec in _model_call(agent).tool_specs]]
    assert index.searches == 0

    # 2. The search: the model describes the need in its own words, and that is what exposes the tool.
    result = _run(runner, plugin.find_tools(NEED, cast("ToolContext", SimpleNamespace(agent=agent))))
    assert index.searches == 1
    assert "list_investment_transactions" in result
    # The search result is a message, and a message is resident: it carries no schema, ever.
    assert "inputSchema" not in result
    assert "since" not in result
    assert plugin._states[agent].exposed == {"list_investment_transactions": 0}

    # 3. The next projection carries the registered full specification, parameters included, while
    # everything the search did not match stays a catalog entry.
    agent.event_loop_metrics.cycle_count = 1
    second = _projected(_run(runner, plugin._projection_handler(_model_call(agent))))
    assert second["list_investment_transactions"] == registered
    assert set(second["list_investment_transactions"]["inputSchema"]["json"]["required"]) == {"account", "since"}
    assert second["send_wire"]["inputSchema"] == EMPTY_CLOSED_SCHEMA

    # 4. The call: the arguments validate against the specification that just arrived, the tool runs,
    # and the call renews the exposure at the current cycle.
    agent.event_loop_metrics.cycle_count = 2
    event = _call_tool(runner, agent, "list_investment_transactions", {"account": "IA-1", "since": "2026-01-01"})
    assert event.tool_result["status"] == "success"
    assert event.tool_result["content"] == [{"text": "IA-1:2026-01-01"}]
    assert plugin._states[agent].exposed == {"list_investment_transactions": 2}

    # The renewal is what it claims to be: the schema is still resident a full TTL after the *call*,
    # which is past the point the original search-time exposure would have aged out.
    agent.event_loop_metrics.cycle_count = 2 + TTL_CYCLES
    renewed = _projected(_run(runner, plugin._projection_handler(_model_call(agent))))
    assert renewed["list_investment_transactions"] == registered

    # And one search paid for all of it: the projections in between never went back to the index.
    assert index.searches == 1
    assert plugin._states[agent].searches == 1

    # The whole cycle changed what calls were told about, not what the agent has.
    assert {name: entry.tool_spec for name, entry in agent.tool_registry.registry.items()} == registry_before
