"""Integration test of the whole disclosure cycle, offline, against a deterministic search double.

Validates: Requirements 12.5.

The other test files each hold one part of the mechanism still: the projection algebra, the catalog
block, the summary path, the index contract, the exposure lifecycle, the degradation paths. None of
them walks the loop the model actually walks. This one does, once, in order:

1. a first projection, where ``tool_specs`` carries the two plugin tools and NOTHING else, and the tool
   the model needs arrives as one ``- name: summary`` line of the catalog in the system prompt — a name
   and a summary, with no parameters anywhere in the call to invoke it with;
2. a search, which is how the model says what it needs when no catalog name fits, and which only
   lists: it exposes nothing, so the schema is still not loaded after it answers;
3. a load, ``get_tool_details([name])``, which is the one path to a schema and the step that exposes;
4. the next projection, which carries that tool's registered full specification, parameters included,
   and drops its catalog line — the projection and the catalog partition the registry;
5. the call, which succeeds against the real registered tool and renews the exposure, so the schema
   stays resident past the TTL it would otherwise have aged out of.

The point of running it as one sequence is that every step consumes what the previous one produced:
the load only resolves a name the catalog listed, the second projection only widens because the load
wrote an exposure, and the renewal is only observable because the call went through the registered
tool rather than a stand-in.

Requirement 12.5 is the other half of the file. Every search path here goes through a ``ToolIndex``
double, and the suite performs zero network calls — asserted rather than assumed: the test body runs
with socket construction, connection and name resolution all replaced by refusals, so a search
implementation that reached for the network would fail here instead of quietly needing it. The event
loop is created before that guard goes up, because ``asyncio`` builds its self-pipe out of a socket
pair at loop construction and a ban installed earlier would fail the runner rather than the code
under test. The model is a stub that raises when called, and the catalog lines come from a
deterministic summarizer stub, so no provider call goes out for a summary either.
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
from strands_progressive_tool_disclosure.plugin import (
    _DETAILS_LOADED_HEADER,
    _MATCHES_HEADER,
    _PLUGIN_TOOL_NAMES,
    FIND_TOOLS_NAME,
    GET_TOOL_DETAILS_NAME,
)

CATALOG_CHARS = 80
"""Character budget of one catalog line. The default, and wide enough that the short description of one
of the two registered tools fits it verbatim while the other has to be summarized."""

TTL_CYCLES = 3
"""Short on purpose: the renewal assertion has to outlive a full TTL to mean anything."""

TOP_K = 2
"""Two, so a plugin tool can rank inside the window and be seen being filtered out of the listing."""

NEED = "list the transactions of an investment account"
"""What the model says it is trying to do. A capability, not a guess at a tool name."""

CATALOG_LINE_PREFIX = "- "
"""What opens a catalog line, in the system-prompt block and in both tools' answers alike."""

WANTED = "list_investment_transactions"
"""The tool the whole sequence is about: cataloged, searched for, loaded, then called."""

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

    The second sentence exists to overrun the budget: a catalog line is at most ``catalog_chars``
    characters, so this description cannot be used verbatim and is what sends the summarizer a tool.

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


class _StubSummarizer:
    """Summarizer that writes a deterministic line and records which tools it was asked about.

    Offline and byte-stable, which is what keeps the catalog assertions exact. It records its calls
    because "a description that fits is used verbatim and costs no call" and "a summary is written
    once and cached" are claims about a number of invocations, not inferences from the lines produced.
    """

    def __init__(self) -> None:
        """Start having been asked about nothing."""
        self.asked: list[str] = []

    def __call__(self, spec: ToolSpec, max_chars: int) -> str:
        """Return a line derived from the tool's name, within the budget."""
        self.asked.append(spec["name"])
        return f"summary of {spec['name']}"[:max_chars]


class _ContainmentIndex:
    """Search double that ranks a name by how many of the need's words its full text contains.

    Deterministic and offline, which is what Requirement 12.5 asks of every search path: the ranking
    is a word count over the text ``build`` received, ties break by indexing order, and the same need
    ranks the same way on every run. It counts its calls because "the search did not load anything"
    is a claim about a number of invocations next to an unchanged exposure map, not an inference from
    the projection that followed.
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
    """Build an offline agent carrying the two registered tools plus the plugin's two vended tools."""
    return Agent(
        model=_StubModel(),
        tools=[list_investment_transactions, send_wire],
        plugins=[plugin],
    )


def _model_call(agent: Agent) -> InvokeModelContext:
    """Build the invocation context of one model call, offering every registered specification.

    Offering the whole registry is what keeps the call off the passthrough path: a name the registry
    does not have, or a missing plugin tool, is a structural guard rather than a projection.
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


def _tool_context(agent: Agent) -> ToolContext:
    """Build the minimal tool context the two vended tools read: the agent, and nothing else."""
    return cast("ToolContext", SimpleNamespace(agent=agent))


def _projected(context: InvokeModelContext) -> dict[str, ToolSpec]:
    """Index a projection by tool name, for asking which tools are carrying a full specification."""
    return {spec["name"]: spec for spec in context.tool_specs}


def _listed(text: str) -> dict[str, str]:
    """Read the ``- name: summary`` lines of ``text`` as name to summary, ignoring everything else.

    One parser for the three places a tool is named by line: the system-prompt catalog block, the
    search result and the load result. They share the shape on purpose — the model reads one listing
    format wherever it meets a tool it has not loaded — and asserting them through one reader is what
    keeps that shared shape a property of the test rather than a coincidence of three copies.

    Args:
        text: A rendered block: a system prompt, or a tool's answer to the model.

    Returns:
        The line of each listed tool, empty when the text lists none.
    """
    listed: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith(CATALOG_LINE_PREFIX):
            continue
        name, _, summary = line.removeprefix(CATALOG_LINE_PREFIX).partition(":")
        listed[name.strip()] = summary.strip()
    return listed


def _catalog(context: InvokeModelContext) -> dict[str, str]:
    """Read the projected context's system-prompt catalog block as name to summary.

    The block is the only place an unloaded tool is named, so this is how the test asks what the model
    was told about a tool it cannot yet call.

    Args:
        context: The projected context.

    Returns:
        The catalog line of each listed tool, empty when no block was appended.
    """
    prompt = context.system_prompt
    return _listed(prompt) if isinstance(prompt, str) else {}


def _call_tool(loop: asyncio.AbstractEventLoop, agent: Agent, name: str, tool_input: dict[str, Any]) -> ToolResultEvent:
    """Run the pre-call hook and then the registered tool, returning the tool's final result event.

    Both halves matter. The hook is what renews the exposure and what would cancel a call made off a
    catalog line, and the registered tool is what turns "the schema arrived" into "the call works":
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
    """Requirement 12.5: catalog, search, load, next projection and call, with zero network calls.

    One sequence, each step consuming the previous one's output, against a deterministic ``ToolIndex``
    double, a deterministic summarizer and with the network unavailable.
    """
    index = _ContainmentIndex()
    summarizer = _StubSummarizer()
    plugin = ProgressiveToolDisclosure(
        index=index,
        summarizer=summarizer,
        ttl_cycles=TTL_CYCLES,
        catalog_chars=CATALOG_CHARS,
        top_k=TOP_K,
    )
    agent = _agent(plugin)
    registry = agent.tool_registry.registry
    registered = registry[WANTED].tool_spec
    registry_before = copy.deepcopy({name: entry.tool_spec for name, entry in registry.items()})

    # 1. First projection: the two plugin tools carry full specifications and nothing else does, so the
    # model can read that the capability exists and cannot call it — it has no parameters for it.
    first = _run(runner, plugin._projection_handler(_model_call(agent)))
    projected = _projected(first)
    assert set(projected) == set(_PLUGIN_TOOL_NAMES)
    assert projected[FIND_TOOLS_NAME] == registry[FIND_TOOLS_NAME].tool_spec
    assert projected[GET_TOOL_DETAILS_NAME] == registry[GET_TOOL_DETAILS_NAME].tool_spec

    # The catalog is in the system prompt, one line per tool that is NOT in the projection, and the
    # rule that governs those names arrives in the same block: they are loaded by name, not called.
    prompt = first.system_prompt
    assert isinstance(prompt, str)
    assert GET_TOOL_DETAILS_NAME in prompt
    assert _catalog(first) == {
        # Over the budget, so this line was written by the summarizer.
        WANTED: f"summary of {WANTED}",
        # Already within it, so it is its own best summary and cost no call.
        "send_wire": "Send a wire transfer from an account.",
    }
    assert summarizer.asked == [WANTED]
    assert all(len(summary) <= CATALOG_CHARS for summary in _catalog(first).values())
    # A catalog line is a name and a summary. No schema travels in it, so no parameter name appears.
    assert "since" not in prompt
    # The index was built once, over the specifications this very call offered — nothing else.
    assert index.built == [[spec["name"] for spec in _model_call(agent).tool_specs]]
    assert index.searches == 0

    # 2. The search: the model describes the need in its own words, and gets names back. It finds; it
    # does not load, so the exposure map is exactly as empty after it as it was before.
    found = _run(runner, plugin.find_tools(NEED, _tool_context(agent)))
    assert index.searches == 1
    assert found.splitlines()[0] == _MATCHES_HEADER
    # A plugin tool can rank — this need is quoted in find_tools' own description — and is never listed.
    assert _listed(found) == {WANTED: f"summary of {WANTED}"}
    # The answer is a message, and a message is resident: it carries no schema, ever.
    assert "inputSchema" not in found
    assert "since" not in found
    assert plugin._states[agent].exposed == {}
    assert plugin._states[agent].searches == 1
    assert plugin._states[agent].loads == 0

    # 3. The load: the one step that exposes a schema, and the only way to a callable tool.
    loaded = _run(runner, plugin.get_tool_details([WANTED], _tool_context(agent)))
    assert loaded.splitlines()[0] == _DETAILS_LOADED_HEADER
    assert _listed(loaded) == {WANTED: f"summary of {WANTED}"}
    assert plugin._states[agent].loads == 1
    assert plugin._states[agent].exposed == {WANTED: 0}
    # Loading resolves names against the registry: it never goes back to the index.
    assert index.searches == 1

    # 4. The next projection carries the registered full specification, parameters included, and the
    # catalog drops that name: the projection and the catalog partition the registry between them.
    agent.event_loop_metrics.cycle_count = 1
    second = _run(runner, plugin._projection_handler(_model_call(agent)))
    second_specs = _projected(second)
    assert second_specs[WANTED] == registered
    assert set(second_specs[WANTED]["inputSchema"]["json"]["required"]) == {"account", "since"}
    assert "send_wire" not in second_specs
    assert _catalog(second) == {"send_wire": "Send a wire transfer from an account."}
    # The summary was written once and cached; a second projection does not pay for it again.
    assert summarizer.asked == [WANTED]

    # 5. The call: the arguments validate against the specification that just arrived, the tool runs,
    # and the call renews the exposure at the current cycle.
    agent.event_loop_metrics.cycle_count = 2
    event = _call_tool(runner, agent, WANTED, {"account": "IA-1", "since": "2026-01-01"})
    assert event.tool_result["status"] == "success"
    assert event.tool_result["content"] == [{"text": "IA-1:2026-01-01"}]
    assert plugin._states[agent].exposed == {WANTED: 2}

    # The renewal is what it claims to be: the schema is still resident a full TTL after the *call*,
    # which is past the point the original load-time exposure would have aged out.
    agent.event_loop_metrics.cycle_count = 2 + TTL_CYCLES
    renewed = _run(runner, plugin._projection_handler(_model_call(agent)))
    assert _projected(renewed)[WANTED] == registered
    assert WANTED not in _catalog(renewed)

    # And one search and one load paid for all of it: the projections in between went back to neither.
    assert index.searches == 1
    assert plugin._states[agent].searches == 1
    assert plugin._states[agent].loads == 1

    # The whole cycle changed what calls were told about, not what the agent has.
    assert {name: entry.tool_spec for name, entry in registry.items()} == registry_before
