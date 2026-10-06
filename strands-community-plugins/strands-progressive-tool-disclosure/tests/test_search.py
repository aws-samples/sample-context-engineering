"""Property tests for the search tool: what one ``find_tools`` invocation lists, and what it does not do.

Feature: progressive-tool-disclosure-plugin, Property 9: Search lists exactly the registry-present matches and
exposes nothing.

Validates: Requirements 5.2, 5.3, 5.9.

Feature: progressive-tool-disclosure-plugin, Property 10: A blank need short-circuits the search.

Validates: Requirements 5.6.

Feature: progressive-tool-disclosure-plugin, Property 11: Search text lists names and short descriptions only.

Validates: Requirements 5.4, 5.5, 5.11.

Search no longer reaches a schema: it finds, and ``get_tool_details`` loads. So the central claim of this file is a
negative one, asserted on its own rather than as a side condition — ``find_tools`` exposes NOTHING. The disclosure
state is captured before the invocation and compared field for field after it: no name is added to ``exposed``, no
pre-existing exposure has its last-use cycle moved, and ``loads`` stays at zero, because a search is not a load. The
summary cache is asserted empty for the same reason: a search composes its lines by truncation off the registry, so no
summarizer may run and the stub model never has to answer.

The listing half is asserted as an exact equality against the registry-present, non-plugin matches. A match is an
opinion of the index, not a fact about the agent: an index may rank a name the registry does not have — a stale entry,
a tool that was unregistered, a double that simply invents one — and such a name has no specification and no
description to report. The doubles here deliberately rank absent names, and the two plugin tools as well, which are
excluded for a different reason: they travel in full on every call, so listing them would send the model to load what
it already holds. The idempotence half is asserted across two invocations with the same need at two different cycles:
the second changes nothing either, because a repeat search is the model asking again, not a new disclosure.

The blank need is asserted by counting: ``search`` is called zero times. Returning the right guidance while still
ranking a top_k of noise would satisfy a test that only read the return value.

What the text carries is names and short descriptions, a statement that the parameters arrive next call, and nothing
else — in particular no ``inputSchema``, asserted through parameter names chosen not to occur in any description, so a
schema leaking into the text is detectable by substring. The need itself must not be echoed either, which is asserted
with a sentinel token carried in every generated need.

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
from strands_progressive_tool_disclosure.plugin import FIND_TOOLS_NAME, GET_TOOL_DETAILS_NAME

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

PLUGIN_NAMES = (FIND_TOOLS_NAME, GET_TOOL_DETAILS_NAME)
"""The two plugin tools. The registry has them and an index may rank them, but a search must not list them: they carry
their full specification on every call already, so naming them would send the model to load what it holds."""

ABSENT_NAMES = ("retired_report", "moved_transfer", "ghost_tool")
"""Names no registry has. An index is free to rank them, and the plugin has to leave them out of the text — the case
Requirement 5.3 singles out."""

SCHEMA_PARAMETERS = ("owner_ref", "wire_ref", "wire_amount", "trail_ref")
"""Every parameter name of every registered tool: none occurs in any description, so finding one in the search result
means a schema reached the text."""

SCHEMA_MARKERS = ("inputSchema", "properties", "additionalProperties", '"type"', "'type'")
"""Structural markers of a serialized schema, for the same substring check."""

SENTINEL = "zqqx"
"""Carried in every generated need and in no description, so an echoed need is detectable by substring."""

ranked_strategy = st.lists(
    st.sampled_from([*REGISTERED_NAMES, *ABSENT_NAMES, *PLUGIN_NAMES]),
    max_size=6,
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

catalog_chars_strategy = st.one_of(st.none(), st.integers(min_value=1, max_value=40))

top_k_strategy = st.integers(min_value=1, max_value=6)

cycle_strategy = st.integers(min_value=0, max_value=20)

seeded_strategy = st.lists(st.sampled_from(REGISTERED_NAMES), max_size=2, unique=True)


class _ScriptedIndex:
    """A ``ToolIndex`` double that ranks a fixed name list and records what it was asked.

    It ignores the need: these properties are about what an invocation does with the matches, and a scorer that reacted
    to wording would make the expected listing a function of generated text. What it records is the point —
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

    Registering through ``plugins`` is what puts both plugin tools in the registry, which is the state every claim here
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

    Pre-existing exposures matter to the claim that a search exposes nothing: what a search may do to an exposure it
    did not create, and to one it "matched", is in both cases nothing at all.

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


def _listable(names: Sequence[str], agent: Agent) -> list[str]:
    """The subset of ``names`` a search may list: present in the registry, and not a plugin tool."""
    registry = agent.tool_registry.registry
    return [name for name in names if name in registry and name not in PLUGIN_NAMES]


@PROPERTY_SETTINGS
@given(
    ranked=ranked_strategy,
    need=need_strategy,
    top_k=top_k_strategy,
    catalog_chars=catalog_chars_strategy,
    cycle=cycle_strategy,
    later=st.integers(min_value=1, max_value=7),
    seeded=seeded_strategy,
)
def test_search_lists_the_registry_present_matches_and_exposes_nothing(
    ranked: list[str],
    need: str,
    top_k: int,
    catalog_chars: int | None,
    cycle: int,
    later: int,
    seeded: list[str],
) -> None:
    """Feature: progressive-tool-disclosure-plugin, Property 9.

    Search lists exactly the registry-present matches and exposes nothing.

    Validates: Requirements 5.2, 5.3, 5.9.
    """
    index = _ScriptedIndex(ranked)
    plugin = ProgressiveToolDisclosure(catalog_chars=catalog_chars, top_k=top_k, index=index)
    agent = _agent(plugin)
    state = _seed(plugin, agent, seeded, cycle)
    before = dict(state.exposed)

    result = _run(plugin.find_tools(need=need, tool_context=_tool_context(agent)))

    matched = _expected_matches(ranked, top_k)
    listable = _listable(matched, agent)
    dropped = [name for name in matched if name not in listable]

    # Requirement 5.2: one search, with the top_k fixed at construction rather than the caller's wish.
    assert index.calls == [(need, top_k)], f"expected one search with top_k={top_k}, got {index.calls}"

    # Requirement 5.3: the listing is an exact equality, so a dropped match excluding itself is a claim of its own and
    # not something the assertion happens to tolerate.
    if listable:
        registry = agent.tool_registry.registry
        expected_lines = [f"- {name}: {plugin._short_description(registry[name].tool_spec)}" for name in listable]
        assert result == "\n".join([plugin_module._MATCHES_HEADER, *expected_lines])
    else:
        assert result == plugin_module._NO_MATCH_GUIDANCE
    for name in dropped:
        # As a line, not as a substring: the header names ``get_tool_details`` itself, and that mention is the
        # instruction, not a listing of it.
        assert f"- {name}:" not in result, f"a match with nothing to load was listed: {name}"

    # THE claim of the redesign: a search finds, it does not disclose. Not one name is added to the exposures and not
    # one pre-existing exposure has its last use moved -- asserted as an equality against the state captured before.
    added = set(state.exposed) - set(before)
    assert state.exposed == before, f"find_tools exposed something: added=<{added}> | exposed=<{state.exposed}>"
    for name in listable:
        assert name not in state.exposed or name in seeded, f"a listed match was exposed by the search: {name}"

    # A search is not a load, and the two counters say which cycle was spent on which.
    assert state.searches == 1
    assert state.loads == 0, "a search counted as a load"

    # No summarizer ran: a search composes its lines by truncation off the registry, and the stub model would have
    # raised if the default summarizer had been reached.
    assert plugin._summaries == {}, "a search populated the summary cache"

    # Requirement 5.9: the same need again, one cycle further on. Same absence of disclosure, one more search.
    agent.event_loop_metrics.cycle_count = cycle + later
    repeat = _run(plugin.find_tools(need=need, tool_context=_tool_context(agent)))

    assert index.calls == [(need, top_k), (need, top_k)], "a repeat search did not search exactly once more"
    assert repeat == result, "a repeat search answered differently"
    assert state.exposed == before, "a repeat search changed the exposures"
    assert state.searches == 2
    assert state.loads == 0

    # The search decides nothing about the registry: what it read from comes out as it went in.
    assert set(agent.tool_registry.registry) == {*PLUGIN_NAMES, *REGISTERED_NAMES}


@PROPERTY_SETTINGS
@given(
    ranked=ranked_strategy,
    need=blank_need_strategy,
    top_k=top_k_strategy,
    catalog_chars=catalog_chars_strategy,
    cycle=cycle_strategy,
    seeded=seeded_strategy,
)
def test_a_blank_need_short_circuits_the_search(
    ranked: list[str],
    need: str,
    top_k: int,
    catalog_chars: int | None,
    cycle: int,
    seeded: list[str],
) -> None:
    """Feature: progressive-tool-disclosure-plugin, Property 10.

    A blank need short-circuits the search.

    Validates: Requirements 5.6.
    """
    index = _ScriptedIndex(ranked)
    plugin = ProgressiveToolDisclosure(catalog_chars=catalog_chars, top_k=top_k, index=index)
    agent = _agent(plugin)
    state = _seed(plugin, agent, seeded, cycle)
    before = dict(state.exposed)

    result = _run(plugin.find_tools(need=need, tool_context=_tool_context(agent)))

    # Requirement 5.6: zero searches. Returning the right guidance while still ranking a top_k of noise would pass a
    # test that only read the return value, which is why this is asserted by counting.
    assert index.calls == [], f"a blank need reached the index: {index.calls}"
    assert result == plugin_module._EMPTY_NEED_GUIDANCE
    assert plugin_module._MATCHES_HEADER not in result, "a blank need announced matches"

    # Nothing was ranked, and nothing would have been exposed anyway: the state is untouched either way.
    assert state.exposed == before
    assert state.loads == 0
    # The cycle is still counted as spent searching: what the counter measures is cycles, not answers.
    assert state.searches == 1


@PROPERTY_SETTINGS
@given(
    ranked=ranked_strategy,
    need=need_strategy,
    top_k=top_k_strategy,
    catalog_chars=catalog_chars_strategy,
    cycle=cycle_strategy,
)
def test_search_text_lists_names_and_short_descriptions_only(
    ranked: list[str],
    need: str,
    top_k: int,
    catalog_chars: int | None,
    cycle: int,
) -> None:
    """Feature: progressive-tool-disclosure-plugin, Property 11.

    Search text lists names and short descriptions only.

    Validates: Requirements 5.4, 5.5, 5.11.
    """
    index = _ScriptedIndex(ranked)
    plugin = ProgressiveToolDisclosure(catalog_chars=catalog_chars, top_k=top_k, index=index)
    agent = _agent(plugin)
    state = _seed(plugin, agent, (), cycle)

    result = _run(plugin.find_tools(need=need, tool_context=_tool_context(agent)))

    registry = agent.tool_registry.registry
    listable = _listable(_expected_matches(ranked, top_k), agent)

    if not listable:
        # Nothing to list: the header over an empty list would read as a failure of the tool rather than of the wording.
        assert result == plugin_module._NO_MATCH_GUIDANCE
    else:
        # Requirement 5.5: the model is told nothing is loaded yet and which tool loads it, which is what makes the
        # listing actionable now that a search does not disclose.
        expected_lines = [f"- {name}: {plugin._short_description(registry[name].tool_spec)}" for name in listable]
        assert result == "\n".join([plugin_module._MATCHES_HEADER, *expected_lines])
        assert GET_TOOL_DETAILS_NAME in result, "the listing did not name the tool that loads a specification"

        # Requirement 5.4: name and short description, from the registry, for each listed match.
        for name in listable:
            assert f"- {name}: " in result

    # Requirement 5.4: no schema, in any form. The parameter names occur in no description, so a schema that reached
    # the text is visible as a substring.
    for marker in (*SCHEMA_PARAMETERS, *SCHEMA_MARKERS):
        assert marker not in result, f"the search result carried schema content: {marker}"

    # Requirement 5.11: composed from the registry alone. The need carries a sentinel that no description has, so an
    # echoed need is visible the same way.
    assert SENTINEL not in result, "the search result echoed the need"

    # A registry-absent match is in neither the text nor the exposures -- and neither is anything else: the search
    # exposes nothing at all.
    for name in ABSENT_NAMES:
        assert name not in result, f"a registry-absent match was listed: {name}"
    assert state.exposed == {}, f"find_tools exposed something: {state.exposed}"
