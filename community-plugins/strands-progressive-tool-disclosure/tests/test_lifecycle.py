"""Property tests for the exposure lifecycle: TTL expiration, renewal by use and premature calls.

Feature: progressive-tool-disclosure-plugin, Property 15: TTL expiration respects the boundary.
Validates: Requirements 7.1, 7.2, 7.5.

Feature: progressive-tool-disclosure-plugin, Property 17: Expiration never touches the registry or
``tool_names``.
Validates: Requirements 7.7, 9.6.

Feature: progressive-tool-disclosure-plugin, Property 18: A premature call is cancelled exactly under
the four-condition conjunction.
Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7.

Feature: progressive-tool-disclosure-plugin, Property 16: Repeated use keeps a schema resident without
re-searching.
Validates: Requirements 7.3, 7.4.

The lifecycle is where the plugin decides what the model is allowed to see, so these properties are
asserted against the real primitives (``_expire``, ``_renew``, ``_on_before_tool_call``) and, where the
claim is about what reaches the provider, against a real ``Agent`` carrying a real ``ToolRegistry``.
The agent runs against a stub model that raises if it is ever called, and the search index is a
deterministic double that counts its calls, so no example reaches the network and "without
re-searching" is a countable assertion rather than an inference.
"""

import asyncio
import copy
import dataclasses
from collections.abc import Coroutine
from types import SimpleNamespace
from typing import Any, TypeVar, cast

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from strands import Agent
from strands.agent.agent import Agent as AgentType
from strands.hooks.events import BeforeToolCallEvent
from strands.models.model import Model
from strands.tools.decorator import tool
from strands.types.tools import ToolContext

from strands_progressive_tool_disclosure import ProgressiveToolDisclosure, ToolMatch
from strands_progressive_tool_disclosure._compat import InvokeModelContext
from strands_progressive_tool_disclosure.plugin import (
    FIND_TOOLS_NAME,
    _DisclosureState,
    _expire,
    _requires_parameters,
)

PROPERTY_SETTINGS = settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

EMPTY_CLOSED_SCHEMA = {"json": {"type": "object", "properties": {}, "additionalProperties": False}}
"""The schema a catalog entry carries. Its presence is how an unexposed tool is recognized."""

REQUIRING_TOOLS = ("list_accounts", "send_wire")
"""Registered tools that declare a required parameter: the premature-call candidates."""

PARAMETERLESS_TOOLS = ("ping", "status")
"""Registered tools callable with no arguments, so an empty call to them is legitimate."""

REGISTERED_TOOLS = (*REQUIRING_TOOLS, *PARAMETERLESS_TOOLS)

name_strategy = st.sampled_from([f"tool_{position}" for position in range(6)])

cycle_strategy = st.integers(min_value=0, max_value=40)

ttl_strategy = st.integers(min_value=1, max_value=6)

_Resolved = TypeVar("_Resolved")


class _StubModel(Model):
    """A model that exists only so an ``Agent`` can be constructed. A call to it is a test failure."""

    def update_config(self, **kwargs: Any) -> None:
        """Accept anything; nothing reads it."""

    def get_config(self) -> dict[str, Any]:
        """No configuration to report."""
        return {}

    async def stream(self, *args: Any, **kwargs: Any) -> Any:
        """Refuse: these properties are about projection and exposure, never about a model call."""
        raise AssertionError("the language model was called")
        yield

    async def structured_output(self, *args: Any, **kwargs: Any) -> Any:
        """Refuse: these properties are about projection and exposure, never about a model call."""
        raise AssertionError("the language model was called")
        yield


@tool
def list_accounts(owner: str) -> str:
    """List the accounts of an owner.

    Args:
        owner: Whose accounts to list.
    """
    return owner


@tool
def send_wire(account: str, amount: str) -> str:
    """Send a wire transfer from an account.

    Args:
        account: Account to debit.
        amount: How much to send.
    """
    return f"{account}:{amount}"


@tool
def ping() -> str:
    """Report whether the service answers."""
    return "pong"


@tool
def status() -> str:
    """Report the current service status."""
    return "ok"


class _CountingIndex:
    """Deterministic search double that ranks a fixed name list and counts what it was asked.

    It ignores the need entirely: the properties here are about the exposure lifecycle, and a scorer
    that reacts to wording would make "searched once" depend on the generated text. The call count is
    the point — Requirement 7.4 is a claim about how many searches a stretch of repeated use costs.
    """

    def __init__(self, ranked: tuple[str, ...] = REQUIRING_TOOLS) -> None:
        """Rank ``ranked``, restricted on each search to the names last built."""
        self._ranked = ranked
        self.built: list[list[str]] = []
        self.searches = 0

    def build(self, specs: list[dict[str, Any]]) -> None:
        """Record the names offered for indexing, in order."""
        self.built.append([spec["name"] for spec in specs])

    def search(self, need: str, top_k: int) -> list[ToolMatch]:
        """Return the fixed ranking, clipped to ``top_k`` and to what was last built."""
        self.searches += 1
        if top_k <= 0 or not self.built:
            return []

        available = set(self.built[-1])
        ranked = [name for name in self._ranked if name in available]

        return [ToolMatch(name=name, score=float(len(ranked) - position)) for position, name in enumerate(ranked)][
            :top_k
        ]


def _agent(plugin: ProgressiveToolDisclosure) -> Agent:
    """Build an offline agent carrying the four registered tools plus the plugin's search tool.

    Args:
        plugin: The plugin to register, which is what adds ``find_tools`` and the pre-call hook.

    Returns:
        An agent whose registry is real, so ``tool_names`` and the registered specifications are the
        ones the properties assert over.
    """
    return Agent(
        model=_StubModel(),
        tools=[list_accounts, send_wire, ping, status],
        plugins=[plugin],
    )


def _model_call(agent: Agent) -> InvokeModelContext:
    """Build the invocation context of one model call, offering every registered specification.

    Offering the whole registry is what keeps the call off the passthrough path: a name the registry
    does not have, or a missing ``find_tools``, is a structural guard rather than a projection.

    Args:
        agent: Agent of the call.

    Returns:
        A context whose ``tool_specs`` are the registered full specifications.
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


def _before_tool_call(agent: Agent, name: str, tool_input: dict[str, Any] | None) -> BeforeToolCallEvent:
    """Build a real pre-call event, so the event's own write guards are exercised.

    Args:
        agent: Agent of the call.
        name: Tool name the model asked for.
        tool_input: Arguments the model passed, empty for a call made off a catalog entry.

    Returns:
        The event the hook receives.
    """
    return BeforeToolCallEvent(
        agent=cast("AgentType", agent),
        selected_tool=None,
        tool_use={"toolUseId": "t1", "name": name, "input": tool_input if tool_input is not None else {}},
        invocation_state={},
    )


def _run(step: Coroutine[Any, Any, _Resolved]) -> _Resolved:
    """Drive one awaited plugin step to completion from a synchronous test.

    The tests stay synchronous on purpose: Hypothesis drives them, and it has no runner for a
    coroutine test function — an ``async def`` body under ``@given`` is never awaited, so every
    assertion in it would pass by not running at all.

    Args:
        step: The coroutine to run: a projection or a search.

    Returns:
        What the step returned.
    """
    return asyncio.run(step)


def _projected(context: InvokeModelContext) -> dict[str, dict[str, Any]]:
    """Index a projection by tool name, for asking which form each tool arrived in."""
    return {spec["name"]: spec for spec in context.tool_specs}


def _is_catalog_entry(spec: dict[str, Any]) -> bool:
    """Report whether ``spec`` is the reduced form rather than the registered full specification."""
    return spec["inputSchema"] == EMPTY_CLOSED_SCHEMA


def _registry_snapshot(agent: Agent) -> dict[str, dict[str, Any]]:
    """Copy every registered specification, for comparing the registry before and after."""
    return {name: copy.deepcopy(entry.tool_spec) for name, entry in agent.tool_registry.registry.items()}


@PROPERTY_SETTINGS
@given(
    exposures=st.dictionaries(name_strategy, cycle_strategy, max_size=6),
    cycle=cycle_strategy,
    ttl_cycles=ttl_strategy,
    searches=st.integers(min_value=0, max_value=9),
)
def test_expiration_keeps_exactly_the_exposures_inside_the_boundary(
    exposures: dict[str, int],
    cycle: int,
    ttl_cycles: int,
    searches: int,
) -> None:
    """Feature: progressive-tool-disclosure-plugin, Property 15.

    TTL expiration respects the boundary.

    Validates: Requirements 7.1, 7.5.
    """
    state = _DisclosureState(
        exposed=dict(exposures),
        fingerprint=frozenset({"find_tools"}),
        searches=searches,
        premature_cancellations=searches,
    )

    _expire(state, cycle, ttl_cycles)

    # The boundary belongs to the live side: ``<=`` keeps an exposure aged exactly ``ttl_cycles``.
    exp_exposed = {name: last_used for name, last_used in exposures.items() if cycle - last_used <= ttl_cycles}
    assert state.exposed == exp_exposed

    # Expiration decides nothing else: the fingerprint and the counters are another concern entirely.
    assert state.fingerprint == frozenset({"find_tools"})
    assert (state.searches, state.premature_cancellations) == (searches, searches)

    # Idempotent at the same cycle, so a second projection in one cycle cannot drop more.
    _expire(state, cycle, ttl_cycles)
    assert state.exposed == exp_exposed


@PROPERTY_SETTINGS
@given(ttl_cycles=ttl_strategy, last_used=cycle_strategy)
def test_the_exposure_survives_at_the_boundary_and_is_dropped_one_cycle_later(ttl_cycles: int, last_used: int) -> None:
    """Feature: progressive-tool-disclosure-plugin, Property 15.

    TTL expiration respects the boundary.

    Validates: Requirements 7.1, 7.5.
    """
    at_boundary = _DisclosureState(exposed={"list_accounts": last_used})
    _expire(at_boundary, last_used + ttl_cycles, ttl_cycles)
    assert at_boundary.exposed == {"list_accounts": last_used}

    past_boundary = _DisclosureState(exposed={"list_accounts": last_used})
    _expire(past_boundary, last_used + ttl_cycles + 1, ttl_cycles)
    assert past_boundary.exposed == {}


@PROPERTY_SETTINGS
@given(age=st.integers(min_value=0, max_value=8), ttl_cycles=ttl_strategy, cycle=st.integers(min_value=8, max_value=30))
def test_a_live_exposure_projects_its_full_specification_and_an_expired_one_a_catalog_entry(
    age: int,
    ttl_cycles: int,
    cycle: int,
) -> None:
    """Feature: progressive-tool-disclosure-plugin, Property 15.

    TTL expiration respects the boundary.

    Validates: Requirements 7.1, 7.2, 7.5.
    """
    plugin = ProgressiveToolDisclosure(index=_CountingIndex(), ttl_cycles=ttl_cycles)
    agent = _agent(plugin)
    agent.event_loop_metrics.cycle_count = cycle
    plugin._states[agent] = _DisclosureState(exposed={"list_accounts": cycle - age})

    projected = _projected(_run(plugin._projection_handler(_model_call(agent))))

    exp_live = age <= ttl_cycles
    registered = agent.tool_registry.registry["list_accounts"].tool_spec

    # Requirement 7.2: while the exposure is live, the call carries the parameters the model needs.
    assert (projected["list_accounts"] == registered) is exp_live
    assert _is_catalog_entry(projected["list_accounts"]) is not exp_live

    # Requirement 7.1: the exposure is dropped from the state, not merely omitted from this call.
    assert ("list_accounts" in plugin._states[agent].exposed) is exp_live


@PROPERTY_SETTINGS
@given(
    exposures=st.dictionaries(st.sampled_from(REGISTERED_TOOLS), cycle_strategy, max_size=4),
    cycle=cycle_strategy,
    ttl_cycles=ttl_strategy,
)
def test_expiration_never_touches_the_registry_or_tool_names(
    exposures: dict[str, int],
    cycle: int,
    ttl_cycles: int,
) -> None:
    """Feature: progressive-tool-disclosure-plugin, Property 17.

    Expiration never touches the registry or ``tool_names``.

    Validates: Requirements 7.7, 9.6.
    """
    plugin = ProgressiveToolDisclosure(index=_CountingIndex(), ttl_cycles=ttl_cycles)
    agent = _agent(plugin)
    agent.event_loop_metrics.cycle_count = cycle
    plugin._states[agent] = _DisclosureState(exposed=dict(exposures))

    exp_names = sorted(agent.tool_names)
    exp_registry = _registry_snapshot(agent)
    exp_entries = dict(agent.tool_registry.registry)

    # Expiration through the handler, then directly: both are the same write, and neither may reach
    # past the exposure map. A second projection ages nothing further at the same cycle.
    _run(plugin._projection_handler(_model_call(agent)))
    _run(plugin._projection_handler(_model_call(agent)))
    _expire(plugin._states[agent], cycle, ttl_cycles)

    # Requirement 9.6: what the agent reports as its tools is invariant to the exposure state — an
    # expired schema withdraws a projection, it does not unregister a tool.
    assert sorted(agent.tool_names) == exp_names
    assert set(agent.tool_names) == set(exp_registry)

    # Requirement 7.7: the registry keeps the same entries, and their specifications are unmodified —
    # the catalog entry is a copy sent to the provider, never a rewrite of what was registered.
    assert agent.tool_registry.registry == exp_entries
    assert _registry_snapshot(agent) == exp_registry
    assert FIND_TOOLS_NAME in agent.tool_registry.registry


@PROPERTY_SETTINGS
@given(
    name=st.sampled_from([*REGISTERED_TOOLS, FIND_TOOLS_NAME, "not_a_tool"]),
    was_exposed=st.booleans(),
    always_available=st.lists(st.sampled_from(REGISTERED_TOOLS), max_size=2, unique=True),
    tool_input=st.sampled_from([{}, None, {"owner": "A1"}, {"account": "A1", "amount": "10"}]),
    cycle=cycle_strategy,
)
def test_a_premature_call_is_cancelled_exactly_under_the_three_condition_conjunction(
    name: str,
    was_exposed: bool,
    always_available: list[str],
    tool_input: dict[str, Any] | None,
    cycle: int,
) -> None:
    """Feature: progressive-tool-disclosure-plugin, Property 18.

    A premature call is cancelled exactly under the three-condition conjunction.

    The conjunction used to carry a fourth conjunct, ``not event.tool_use.get("input")``, which let a
    call with INVENTED arguments run against a schema the model had never seen. Whether the model left
    the arguments out or made them up is not a distinction it could have made: it had a name and one
    line of description either way. The invented case is the more dangerous of the two, because
    permissive arguments can return a confidently wrong answer that nothing marks as suspect, where an
    empty call fails loudly and is retried with the real schema.

    So ``tool_input`` is still parameterized here, and the assertion is now that it makes NO difference.

    Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7.
    """
    plugin = ProgressiveToolDisclosure(index=_CountingIndex(), always_available=always_available)
    agent = _agent(plugin)
    agent.event_loop_metrics.cycle_count = cycle

    registry = agent.tool_registry.registry
    is_registered = name in registry
    if was_exposed and is_registered:
        plugin._states[agent] = _DisclosureState(exposed={name: cycle})

    event = _before_tool_call(agent, name, tool_input)
    exp_tool_use = copy.deepcopy(event.tool_use)
    exp_registry = _registry_snapshot(agent)

    plugin._on_before_tool_call(event)

    # The conjunction, spelled out: each conjunct false on its own is a reason to let the call run.
    # The arguments the call carried are absent from it on purpose.
    exp_cancelled = (
        is_registered
        and not was_exposed
        and name not in always_available
        and _requires_parameters(registry[name].tool_spec if is_registered else {})
    )
    assert bool(event.cancel_tool) is exp_cancelled
    if exp_cancelled:
        # Requirement 8.1: the message names the tool, so the model knows which call to repeat.
        assert name in cast("str", event.cancel_tool)
        # Requirement 8.2: the cancelled call paid for the schema — the next projection carries it.
        assert plugin._states[agent].exposed[name] == cycle
        assert plugin._states[agent].premature_cancellations == 1

    if is_registered:
        # Requirement 7.3: a call is the strongest evidence the schema is still worth sending.
        assert plugin._states[agent].exposed[name] == cycle
    else:
        # Requirement 8.6: an unknown name has no specification to expose, so no state is created.
        assert plugin._states.get(agent) is None

    # Requirement 8.7: the only writes are the disclosure state and, when premature, ``cancel_tool``.
    assert event.tool_use == exp_tool_use
    assert _registry_snapshot(agent) == exp_registry


@PROPERTY_SETTINGS
@given(
    ttl_cycles=st.integers(min_value=1, max_value=10),
    extra_cycles=st.integers(min_value=1, max_value=10),
)
def test_repeated_use_keeps_a_schema_resident_without_re_searching(
    ttl_cycles: int,
    extra_cycles: int,
) -> None:
    """Feature: progressive-tool-disclosure-plugin, Property 16.

    Repeated use keeps a schema resident without re-searching.

    Validates: Requirements 7.3, 7.4.
    """
    index = _CountingIndex(ranked=("list_accounts",))
    plugin = ProgressiveToolDisclosure(index=index, ttl_cycles=ttl_cycles)
    agent = _agent(plugin)
    registered = agent.tool_registry.registry["list_accounts"].tool_spec

    # Cold start: the tool is a catalog entry, and one search is what exposes it.
    first = _projected(_run(plugin._projection_handler(_model_call(agent))))
    assert _is_catalog_entry(first["list_accounts"])

    _run(plugin.find_tools("list the accounts of an owner", cast("ToolContext", SimpleNamespace(agent=agent))))
    assert index.searches == 1

    # Used in every cycle of a stretch longer than the TTL: the call renews, the projection carries
    # the full specification, and no cycle of the stretch pays for a second search.
    for cycle in range(1, ttl_cycles + extra_cycles + 1):
        agent.event_loop_metrics.cycle_count = cycle
        plugin._on_before_tool_call(_before_tool_call(agent, "list_accounts", {"owner": "A1"}))

        projected = _projected(_run(plugin._projection_handler(_model_call(agent))))
        assert projected["list_accounts"] == registered
        assert plugin._states[agent].exposed["list_accounts"] == cycle

    # Requirement 7.4: zero searches after the first exposure, across a stretch longer than the TTL.
    assert index.searches == 1
    assert plugin._states[agent].searches == 1

    # And the moment use stops, the exposure ages out: residency is bought by use, not granted.
    agent.event_loop_metrics.cycle_count += ttl_cycles + 1
    lapsed = _projected(_run(plugin._projection_handler(_model_call(agent))))
    assert _is_catalog_entry(lapsed["list_accounts"])
