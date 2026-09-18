"""Property tests for the search tool: what one ``find_tools`` invocation exposes, and what it says.

Feature: progressive-tool-disclosure-plugin, Property 9: Search exposes exactly the registry-present matches, and is
idempotent on repeat.

Validates: Requirements 5.2, 5.3, 5.9.

Feature: progressive-tool-disclosure-plugin, Property 10: A blank need short-circuits the search.

Validates: Requirements 5.6.

Feature: progressive-tool-disclosure-plugin, Property 11: Search text lists names and short descriptions only.

Validates: Requirements 5.4, 5.5, 5.11.

Search is the only way back from a catalog entry to a full schema, so three separate claims about it are asserted here.

The first is about the exposure set. A match is an opinion of the index, not a fact about the agent: an index may rank a
name the registry does not have — a stale entry, a tool that was unregistered, a double that simply invents one — and
such a name has no specification to project and no description to report. So the exposures are asserted as an exact
equality against the registry-present matches, which makes the absent ones' exclusion a claim in its own right rather
than something the assertion happens to tolerate. The doubles here deliberately rank absent names. The idempotence half
is asserted across two invocations with the same need at two different cycles: the second one must produce the same set
of keys and only move their last-use cycle forward, because a repeat search is the model asking again, not a new
disclosure.

The second is the blank need. It is asserted by counting: ``search`` is called zero times. Returning the right guidance
while still ranking a top_k of noise would satisfy a test that only read the return value.

The third is what the text carries. Names and short descriptions, a statement that the parameters arrive next call, and
nothing else — in particular no ``inputSchema``, asserted through parameter names chosen not to occur in any
description, so a schema leaking into the text is detectable by substring. The need itself must not be echoed either,
which is asserted with a sentinel token carried in every generated need.

Everything runs offline: a stub model that raises if called, deterministic ``ToolIndex`` doubles that record what they
were asked, and a ``ToolContext`` built by hand for a direct call. No network, no model call, no disk.
"""

import asyncio
from collections.abc import Coroutine, Sequence
from typing import Any, TypeVar

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from strands import Agent, tool
from strands.models.model import Model
from strands.types.tools import ToolContext, ToolSpec

import strands_progressive_tool_disclosure.plugin as plugin_module
from strands_progressive_tool_disclosure import ProgressiveToolDisclosure
from strands_progressive_tool_disclosure.index import ToolMatch
from strands_progressive_tool_disclosure.plugin import FIND_TOOLS_NAME

PROPERTY_SETTINGS = settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

_Resolved = TypeVar("_Resolved")


class _StubModel(Model):
    """A model that exists to be constructed. A call to it is a bug in the test, not a result."""

    def update_config(self, **kwargs: Any) -> None:
        """Accept anything; nothing reads it."""

    def get_config(self) -> dict[str, Any]:
        """No configuration to report."""
        return {}

    async def stream(self, *args: Any, **kwargs: Any) -> Any:
        """Refuse: these properties are about the search tool, never about a model call."""
        raise AssertionError("the language model was called")
        yield

    async def structured_output(self, *args: Any, **kwargs: Any) -> Any:
        """Refuse: these properties are about the search tool, never about a model call."""
        raise AssertionError("the language model was called")
        yield


# The parameter names below are deliberately unlike any word of any description: a schema leaking into the search result
# is then detectable by substring, which is what Requirement 5.4 is asserted with.
@tool
def list_accounts(owner_ref: str) -> str:
    """List the accounts of an owner.

    Args:
        owner_ref: Whose accounts to list.
    """
    return owner_ref


@tool
def send_wire(wire_ref: str, wire_amount: str) -> str:
    """Send a wire transfer from an account, once the sum has been approved.

    Args:
        wire_ref: Account to debit.
        wire_amount: How much to send.
    """
    return f"{wire_ref}:{wire_amount}"


@tool
def audit_trail(trail_ref: str) -> str:
    """Read the audit trail of an account, from the oldest entry to the newest one, including every
    change of ownership and every transfer that was refused along the way.

    Args:
        trail_ref: Account whose trail to read.
    """
    return trail_ref


@tool
def ping() -> str:
    """Report whether the service answers."""
    return "pong"


TOOLS = [list_accounts, send_wire, audit_trail, ping]

REGISTERED_NAMES = ("list_accounts", "send_wire", "audit_trail", "ping")
"""Names the registry has, so a match on one of them is a match with a specification behind it."""

ABSENT_NAMES = ("retired_report", "moved_transfer", "ghost_tool")
"""Names no registry has. An index is free to rank them, and the plugin has to leave them out of both the exposures and
the text — the case Requirements 5.3 and 5.8 single out."""

SCHEMA_PARAMETERS = ("owner_ref", "wire_ref", "wire_amount", "trail_ref")
"""Every parameter name of every registered tool: none occurs in any description, so finding one in the search result
means a schema reached the text."""

SCHEMA_MARKERS = ("inputSchema", "properties", "additionalProperties", '"type"', "'type'")
"""Structural markers of a serialized schema, for the same substring check."""

SENTINEL = "zqqx"
"""Carried in every generated need and in no description, so an echoed need is detectable by substring."""

ranked_strategy = st.lists(
    st.sampled_from([*REGISTERED_NAMES, *ABSENT_NAMES]),
    max_size=5,
    unique=True,
)

need_strategy = st.sampled_from(
    [
        f"{SENTINEL} list the accounts of an owner",
        f"move money between accounts {SENTINEL}",
        f"read the {SENTINEL} audit trail",
        f"{SENTINEL}",
    ]
)

blank_need_strategy = st.sampled_from(["", " ", "   ", "\t", "\n", "\t \n ", "\r\n"])

catalog_tokens_strategy = st.one_of(st.none(), st.integers(min_value=1, max_value=40))

top_k_strategy = st.integers(min_value=1, max_value=5)

cycle_strategy = st.integers(min_value=0, max_value=20)

seeded_strategy = st.lists(st.sampled_from(REGISTERED_NAMES), max_size=2, unique=True)


class _ScriptedIndex:
    """A ``ToolIndex`` double that ranks a fixed name list and records what it was asked.

    It ignores the need: these properties are about what an invocation does with the matches, and a scorer that reacted
    to wording would make the expected exposure set a function of generated text. What it records is the point —
    Requirement 5.2 is a claim about how many times ``search`` runs and with which ``top_k``.

    The ranking is not filtered against any registry, deliberately: ranking a name the registry does not have is the
    case the plugin has to drop, so the double has to be able to produce it.
    """

    def __init__(self, ranked: Sequence[str]) -> None:
        """Rank ``ranked``, in the given order, whatever the need."""
        self._ranked = tuple(ranked)
        self.builds: list[list[str]] = []
        self.calls: list[tuple[str, int]] = []

    def build(self, specs: Sequence[ToolSpec]) -> None:
        """Record the names offered for indexing, in order."""
        self.builds.append([spec["name"] for spec in specs])

    def search(self, need: str, top_k: int) -> list[ToolMatch]:
        """Record the invocation and return the fixed ranking, clipped to ``top_k``."""
        self.calls.append((need, top_k))
        ranked = list(self._ranked[:top_k]) if top_k > 0 else []
        return [ToolMatch(name=name, score=float(len(ranked) - position)) for position, name in enumerate(ranked)]


def _run(step: Coroutine[Any, Any, _Resolved]) -> _Resolved:
    """Drive one awaited plugin step to completion from a synchronous test.

    The tests stay synchronous on purpose: Hypothesis drives them, and an ``async def`` body under ``@given`` is never
    awaited, so every assertion in it would pass by not running at all.

    Args:
        step: The coroutine to run: here, always a search.

    Returns:
        What the step returned.
    """
    return asyncio.run(step)


def _agent(plugin: ProgressiveToolDisclosure) -> Agent:
    """Build an offline agent carrying the tool pool and ``plugin``.

    Registering through ``plugins`` is what puts the search tool in the registry, which is the state every claim here
    is made about.

    Args:
        plugin: The plugin to register.

    Returns:
        An agent whose registry is real, so the registry-present matches are the real ones.
    """
    return Agent(model=_StubModel(), tools=TOOLS, plugins=[plugin])


def _tool_context(agent: Agent) -> ToolContext:
    """The context the framework would hand the search tool, built by hand for a direct call."""
    return ToolContext(
        tool_use={"toolUseId": "use-search", "name": FIND_TOOLS_NAME, "input": {}},
        agent=agent,
        invocation_state={},
    )


def _seed(
    plugin: ProgressiveToolDisclosure,
    agent: Agent,
    exposed: Sequence[str],
    cycle: int,
) -> plugin_module._DisclosureState:
    """Give ``agent`` a disclosure state with exposures at ``cycle``, and set the cycle counter.

    Pre-existing exposures matter to the idempotence claim: what a repeat search may do to an exposure it did not
    create is nothing at all.

    Args:
        plugin: Plugin holding the state map.
        agent: Agent whose state to seed.
        exposed: Names to expose.
        cycle: Cycle to record as the last use, and as the agent's current cycle.

    Returns:
        The seeded state, which is the same object the plugin reads.
    """
    agent.event_loop_metrics.cycle_count = cycle
    state = plugin_module._state_for(plugin._states, agent)
    for name in exposed:
        state.exposed[name] = cycle
    return state


def _expected_matches(ranked: Sequence[str], top_k: int) -> list[str]:
    """The names the double will return for a non-blank need: the ranking clipped to ``top_k``."""
    return list(ranked[:top_k])


def _present(names: Sequence[str], agent: Agent) -> list[str]:
    """The subset of ``names`` the agent's registry actually has, in order."""
    registry = agent.tool_registry.registry
    return [name for name in names if name in registry]


@PROPERTY_SETTINGS
@given(
    ranked=ranked_strategy,
    need=need_strategy,
    top_k=top_k_strategy,
    catalog_tokens=catalog_tokens_strategy,
    cycle=cycle_strategy,
    later=st.integers(min_value=1, max_value=7),
    seeded=seeded_strategy,
)
def test_search_exposes_exactly_the_registry_present_matches_and_is_idempotent_on_repeat(
    ranked: list[str],
    need: str,
    top_k: int,
    catalog_tokens: int | None,
    cycle: int,
    later: int,
    seeded: list[str],
) -> None:
    """Feature: progressive-tool-disclosure-plugin, Property 9.

    Search exposes exactly the registry-present matches, and is idempotent on repeat.

    Validates: Requirements 5.2, 5.3, 5.9.
    """
    index = _ScriptedIndex(ranked)
    plugin = ProgressiveToolDisclosure(catalog_tokens=catalog_tokens, top_k=top_k, index=index)
    agent = _agent(plugin)
    state = _seed(plugin, agent, seeded, cycle)

    _run(plugin.find_tools(need=need, tool_context=_tool_context(agent)))

    matched = _expected_matches(ranked, top_k)
    present = _present(matched, agent)
    absent = [name for name in matched if name not in present]

    # Requirement 5.2: one search, with the top_k fixed at construction rather than the caller's wish.
    assert index.calls == [(need, top_k)], f"expected one search with top_k={top_k}, got {index.calls}"

    # Requirement 5.3: an exact equality, so a registry-absent match excluding itself is a claim and not a tolerance.
    exp_first = {name: cycle for name in [*seeded, *present]}
    assert state.exposed == exp_first
    for name in absent:
        assert name not in state.exposed, f"a registry-absent match was exposed: {name}"

    # Requirement 5.9: the same need again, one cycle further on. Same keys, and only their last use moves.
    agent.event_loop_metrics.cycle_count = cycle + later
    _run(plugin.find_tools(need=need, tool_context=_tool_context(agent)))

    assert index.calls == [(need, top_k), (need, top_k)], "a repeat search did not search exactly once more"
    assert set(state.exposed) == set(exp_first), "a repeat search changed the exposure set"
    for name in present:
        assert state.exposed[name] == cycle + later, f"a repeat search did not renew {name}"
    for name in seeded:
        if name not in present:
            # An exposure the search did not match is not the search's to touch.
            assert state.exposed[name] == cycle, f"a repeat search renewed an unmatched exposure: {name}"

    # The search decides exposures and nothing else: the registry it read from comes out as it went in.
    assert set(agent.tool_registry.registry) == {FIND_TOOLS_NAME, *REGISTERED_NAMES}


@PROPERTY_SETTINGS
@given(
    ranked=ranked_strategy,
    need=blank_need_strategy,
    top_k=top_k_strategy,
    catalog_tokens=catalog_tokens_strategy,
    cycle=cycle_strategy,
    seeded=seeded_strategy,
)
def test_a_blank_need_short_circuits_the_search(
    ranked: list[str],
    need: str,
    top_k: int,
    catalog_tokens: int | None,
    cycle: int,
    seeded: list[str],
) -> None:
    """Feature: progressive-tool-disclosure-plugin, Property 10.

    A blank need short-circuits the search.

    Validates: Requirements 5.6.
    """
    index = _ScriptedIndex(ranked)
    plugin = ProgressiveToolDisclosure(catalog_tokens=catalog_tokens, top_k=top_k, index=index)
    agent = _agent(plugin)
    state = _seed(plugin, agent, seeded, cycle)
    before = dict(state.exposed)

    result = _run(plugin.find_tools(need=need, tool_context=_tool_context(agent)))

    # Requirement 5.6: zero searches. Returning the right guidance while still ranking a top_k of noise would pass a
    # test that only read the return value, which is why this is asserted by counting.
    assert index.calls == [], f"a blank need reached the index: {index.calls}"
    assert result == plugin_module._EMPTY_NEED_GUIDANCE
    assert plugin_module._MATCHES_HEADER not in result, "a blank need announced matches"

    # Nothing was ranked, so nothing is exposed: a blank need costs the cycle and changes no disclosure.
    assert state.exposed == before
    # The cycle is still counted as spent searching: what the counter measures is cycles, not answers.
    assert state.searches == 1


@PROPERTY_SETTINGS
@given(
    ranked=ranked_strategy,
    need=need_strategy,
    top_k=top_k_strategy,
    catalog_tokens=catalog_tokens_strategy,
    cycle=cycle_strategy,
)
def test_search_text_lists_names_and_short_descriptions_only(
    ranked: list[str],
    need: str,
    top_k: int,
    catalog_tokens: int | None,
    cycle: int,
) -> None:
    """Feature: progressive-tool-disclosure-plugin, Property 11.

    Search text lists names and short descriptions only.

    Validates: Requirements 5.4, 5.5, 5.11.
    """
    index = _ScriptedIndex(ranked)
    plugin = ProgressiveToolDisclosure(catalog_tokens=catalog_tokens, top_k=top_k, index=index)
    agent = _agent(plugin)
    _seed(plugin, agent, (), cycle)

    result = _run(plugin.find_tools(need=need, tool_context=_tool_context(agent)))

    registry = agent.tool_registry.registry
    present = _present(_expected_matches(ranked, top_k), agent)

    if not present:
        # Nothing to list: the header over an empty list would read as a failure of the tool rather than of the wording.
        assert result == plugin_module._NO_MATCH_GUIDANCE
    else:
        # Requirement 5.5: the model is told the parameters are one turn away, which is what makes a retry worthwhile.
        exp_lines = [f"- {name}: {plugin._short_description(registry[name].tool_spec)}" for name in present]
        assert result == "\n".join([plugin_module._MATCHES_HEADER, *exp_lines])

        # Requirement 5.4: name and short description, from the registry, for each exposed match.
        for name in present:
            assert f"- {name}: " in result

    # Requirement 5.4: no schema, in any form. The parameter names occur in no description, so a schema that reached
    # the text is visible as a substring.
    for marker in (*SCHEMA_PARAMETERS, *SCHEMA_MARKERS):
        assert marker not in result, f"the search result carried schema content: {marker}"

    # Requirement 5.11: composed from the registry alone. The need carries a sentinel that no description has, so an
    # echoed need is visible the same way.
    assert SENTINEL not in result, "the search result echoed the need"

    # And a registry-absent match is in neither the text nor the exposures.
    for name in ABSENT_NAMES:
        assert name not in result, f"a registry-absent match was listed: {name}"
        assert name not in plugin._states[agent].exposed
