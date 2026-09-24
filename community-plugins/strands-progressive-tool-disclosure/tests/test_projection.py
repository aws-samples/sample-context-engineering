"""Property tests for the projection: the five-block union, its guards, and the state behind it.

Feature: progressive-tool-disclosure-plugin, Property 1: Projection is the ordered, de-duplicated union of the five
blocks.
Validates: Requirements 3.1, 3.2, 3.3.

Feature: progressive-tool-disclosure-plugin, Property 2: Projection names are a subset of incoming names.
Validates: Requirements 3.4.

Feature: progressive-tool-disclosure-plugin, Property 3: Projection is never empty and always contains the search tool.
Validates: Requirements 3.5, 9.5.

Feature: progressive-tool-disclosure-plugin, Property 4: A suppressed catalog omits every reduced entry and keeps the
full-spec blocks.
Validates: Requirements 3.6.

Feature: progressive-tool-disclosure-plugin, Property 5: The projection is deterministic.
Validates: Requirements 3.8.

Feature: progressive-tool-disclosure-plugin, Property 6: The projection changes only ``tool_specs``.
Validates: Requirements 3.9.

Feature: progressive-tool-disclosure-plugin, Property 20: History-referenced tools are kept, unknown references are
dropped.
Validates: Requirements 9.3, 9.4.

Feature: progressive-tool-disclosure-plugin, Property 14: The index is built once per registry fingerprint.
Validates: Requirements 6.5, 6.6, 6.7, 6.11.

Feature: progressive-tool-disclosure-plugin, Property 19: Passthrough triggers on either structural guard.
Validates: Requirements 9.1, 9.2, 9.9.

Feature: progressive-tool-disclosure-plugin, Property 21: Per-agent state is isolated.
Validates: Requirements 1.4, 10.5.

The projection is the whole point of the plugin, so these properties run the real handler against a real ``Agent``
carrying a real ``ToolRegistry`` — what the model would be told about a call is what gets asserted, not an intermediate
the handler happens to compute. Two consequences shape the assertions below.

First, the full-specification blocks emit the *incoming* specification object rather than a copy, so "arrived at full
specification" is checked by object identity and "arrived as a catalog entry" by the empty, closed schema. An identity
check cannot be satisfied by a spec that was rebuilt, re-ordered or re-described on the way through.

Second, the search index is a deterministic double that records every ``build`` and every ``search``. "Built once per
fingerprint" is then a count rather than an inference, and no example reaches the network: the double scores by
substring, on names, over what it was last handed.

Incoming specifications are deep copies of the registered ones and the search tool is placed at a generated position
rather than first, so the claim that the projection *leads* with the search tool is tested against an incoming order
that does not already agree with it.
"""

import asyncio
import copy
import dataclasses
from collections.abc import Coroutine, Sequence
from typing import Any, TypeVar, cast

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from strands import Agent, tool
from strands.agent.agent import Agent as AgentType
from strands.hooks.events import BeforeToolCallEvent
from strands.models.model import Model
from strands.types.tools import ToolContext

from strands_progressive_tool_disclosure import ProgressiveToolDisclosure, ToolMatch
from strands_progressive_tool_disclosure._compat import InvokeModelContext
from strands_progressive_tool_disclosure.plugin import (
    FIND_TOOLS_NAME,
    _CATALOG_SIGIL,
    _DisclosureState,
    _truncate_description,
)

PROPERTY_SETTINGS = settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

EMPTY_CLOSED_SCHEMA = {"json": {"type": "object", "properties": {}, "additionalProperties": False}}
"""The schema a catalog entry carries. Its presence is how an unexposed tool is recognized in a projection."""

UNKNOWN_NAMES = ("ghost_tool", "StructuredOutput")
"""Names no registry has: the first stands for a stale history reference, the second for the synthetic specification
forced structured output swaps in."""

_Resolved = TypeVar("_Resolved")


class _StubModel(Model):
    """A model that exists only so an ``Agent`` can be constructed. A call to it is a bug in the test."""

    def update_config(self, **kwargs: Any) -> None:
        """Accept anything; nothing reads it."""

    def get_config(self) -> dict[str, Any]:
        """No configuration to report."""
        return {}

    async def stream(self, *args: Any, **kwargs: Any) -> Any:
        """Refuse: these properties are about the projection, never about a model call."""
        raise AssertionError("the language model was called")
        yield

    async def structured_output(self, *args: Any, **kwargs: Any) -> Any:
        """Refuse: these properties are about the projection, never about a model call."""
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
def wire_transfer(account: str, amount: str) -> str:
    """Send a wire transfer from an account. Settles same day when the window is still open.

    Args:
        account: Account to debit.
        amount: How much to send.
    """
    return f"{account}:{amount}"


@tool
def audit_log(account: str) -> str:
    """Read the audit log of an account, oldest entry first.

    Args:
        account: Account whose log to read.
    """
    return account


@tool
def send_email(recipient: str) -> str:
    """Send an email to a recipient.

    Args:
        recipient: Who receives the email.
    """
    return recipient


@tool
def read_document(path: str) -> str:
    """Read a document from the document store.

    Args:
        path: Where the document lives.
    """
    return path


TOOLS = [list_accounts, wire_transfer, audit_log, send_email, read_document]

TOOL_NAMES = [registered.tool_name for registered in TOOLS]

names_strategy = st.lists(st.sampled_from(TOOL_NAMES), min_size=1, max_size=len(TOOL_NAMES), unique=True)

subset_strategy = st.lists(st.sampled_from(TOOL_NAMES), max_size=3, unique=True)

catalog_tokens_strategy = st.one_of(st.none(), st.integers(min_value=1, max_value=40))

ttl_strategy = st.integers(min_value=1, max_value=6)

cycle_strategy = st.integers(min_value=0, max_value=12)

position_strategy = st.integers(min_value=0, max_value=len(TOOL_NAMES))

need_strategy = st.sampled_from(["list the accounts", "move money", "read the audit trail", "send a message"])


class _RecordingIndex:
    """Deterministic search double that records every ``build`` and ``search``, and scores by substring.

    The records are the point. "Built once per fingerprint" is a claim about how many times ``build`` ran, so it has to
    be counted; and a tool that arrives late has to be *findable*, so what the last build was handed is kept and the
    search is restricted to it. Scoring by substring over names keeps the ranking a function of the need alone — no
    corpus statistics, no tie-breaking surprises, and nothing to reach the network for.
    """

    def __init__(self) -> None:
        """Start with no builds and no searches recorded."""
        self.builds: list[list[str]] = []
        self.searches: list[str] = []

    def build(self, specs: Sequence[dict[str, Any]]) -> None:
        """Record the names offered for indexing, in the order they arrived."""
        self.builds.append([spec["name"] for spec in specs])

    def search(self, need: str, top_k: int) -> Sequence[ToolMatch]:
        """Rank the last-built names containing a term of ``need``, best first, clipped to ``top_k``."""
        self.searches.append(need)
        if top_k <= 0 or not self.builds:
            return []

        terms = [term for term in need.replace("_", " ").lower().split() if term]
        ranked = [name for name in self.builds[-1] if any(term in name.lower() for term in terms)]

        return [ToolMatch(name=name, score=float(len(ranked) - position)) for position, name in enumerate(ranked)][
            :top_k
        ]


def _agent(plugin: ProgressiveToolDisclosure) -> Agent:
    """Build an offline agent carrying the tool pool and ``plugin``.

    Registering through ``plugins`` is what puts the search tool in the registry and the projection handler in the
    middleware registry, which is the state every claim here is made about.

    Args:
        plugin: The plugin to register.

    Returns:
        An agent whose registry is real, so the registered specifications are the ones the properties assert over.
    """
    return Agent(model=_StubModel(), tools=TOOLS, plugins=[plugin])


def _incoming_specs(agent: Agent, names: Sequence[str], find_tools_position: int = 0) -> list[dict[str, Any]]:
    """The specifications a call arrives with: ``names`` with the search tool inserted at a given position.

    Deep copies rather than the registry's own objects, so an example cannot reach the registry the next example reads,
    and so "the projection emitted the incoming specification" is a claim identity can actually settle.

    Args:
        agent: Agent whose registry holds the full specifications.
        names: Tool names the call offers, excluding the search tool.
        find_tools_position: Where the search tool sits in the incoming order. Not pinned to the front, so the
            projection's own ordering is what has to put it first.

    Returns:
        The incoming specifications, in arrival order.
    """
    registry = agent.tool_registry.registry
    specs = [copy.deepcopy(registry[name].tool_spec) for name in names]
    specs.insert(min(find_tools_position, len(specs)), copy.deepcopy(registry[FIND_TOOLS_NAME].tool_spec))
    return specs


def _messages(referenced: Sequence[str]) -> list[dict[str, Any]]:
    """A retained history whose ``toolUse`` blocks reference ``referenced``, in order.

    Args:
        referenced: Tool names the history talks about. May include names no registry has.

    Returns:
        The retained history of the call.
    """
    messages: list[dict[str, Any]] = [{"role": "user", "content": [{"text": "reconcile the ledger"}]}]
    for position, name in enumerate(referenced):
        messages.append(
            {
                "role": "assistant",
                "content": [{"toolUse": {"toolUseId": f"use-{position}", "name": name, "input": {}}}],
            }
        )
    return messages


def _context(agent: Agent, tool_specs: list[dict[str, Any]], messages: list[dict[str, Any]]) -> InvokeModelContext:
    """Build an invocation context, restricted to the fields this SDK's context declares.

    The context gained fields across the supported SDK range; filtering by the declared names keeps a neighbouring
    field from failing these tests for a reason that has nothing to do with the projection.

    Args:
        agent: Agent of the call.
        tool_specs: Specifications the call offers.
        messages: Retained history of the call.

    Returns:
        The context the ``InvokeModelStage.Input`` handler receives.
    """
    candidates: dict[str, Any] = {
        "agent": agent,
        "messages": messages,
        "system_prompt": "You are a helpful assistant.",
        "tool_specs": tool_specs,
        "tool_choice": None,
        "invocation_state": {},
        "model": agent.model,
    }
    declared = {field.name for field in dataclasses.fields(InvokeModelContext)}
    return InvokeModelContext(**{name: value for name, value in candidates.items() if name in declared})


def _seed_state(
    plugin: ProgressiveToolDisclosure,
    agent: Agent,
    exposed: Sequence[str],
    cycle: int,
) -> _DisclosureState:
    """Give ``agent`` a disclosure state whose exposures are live at ``cycle``, and set the cycle counter.

    The exposures are recorded at the current cycle so expiration keeps every one of them: which exposures survive is
    another property's claim, and letting it interfere here would make a projection failure ambiguous.

    Args:
        plugin: Plugin holding the per-agent state map.
        agent: Agent whose state to seed.
        exposed: Names to expose.
        cycle: Cycle counter to run the call at.

    Returns:
        The seeded disclosure state.
    """
    agent.event_loop_metrics.cycle_count = cycle
    state = _DisclosureState(exposed=dict.fromkeys(exposed, cycle))
    plugin._states[agent] = state
    return state


def _tool_context(agent: Agent) -> ToolContext:
    """The context the framework would hand the search tool, built by hand for a direct call."""
    return ToolContext(
        tool_use={"toolUseId": "use-search", "name": FIND_TOOLS_NAME, "input": {}},
        agent=agent,
        invocation_state={},
    )


def _before_tool_call(agent: Agent, name: str) -> BeforeToolCallEvent:
    """A pre-call event for an argument-less call to ``name``: the shape of a call made off a catalog entry."""
    return BeforeToolCallEvent(
        agent=cast("AgentType", agent),
        selected_tool=None,
        tool_use={"toolUseId": "use-premature", "name": name, "input": {}},
        invocation_state={},
    )


def _run(step: Coroutine[Any, Any, _Resolved]) -> _Resolved:
    """Drive one awaited plugin step to completion from a synchronous test.

    The tests stay synchronous on purpose: Hypothesis drives them, and it has no runner for a coroutine test function —
    an ``async def`` body under ``@given`` is never awaited, so every assertion in it would pass by not running.

    Args:
        step: The coroutine to run: a projection or a search.

    Returns:
        What the step returned.
    """
    return asyncio.run(step)


def _names(specs: Sequence[dict[str, Any]]) -> list[str]:
    """The names of ``specs``, in order."""
    return [spec["name"] for spec in specs]


def _projected(context: InvokeModelContext) -> dict[str, dict[str, Any]]:
    """Index a projection by tool name, for asking which form each tool arrived in."""
    return {spec["name"]: spec for spec in context.tool_specs}


def _is_catalog_entry(spec: dict[str, Any]) -> bool:
    """Report whether ``spec`` is the reduced form rather than a full specification."""
    return spec["inputSchema"] == EMPTY_CLOSED_SCHEMA


def _expected_projection(
    incoming: Sequence[dict[str, Any]],
    exposed: Sequence[str],
    referenced: Sequence[str],
    always_available: Sequence[str],
    catalog_tokens: int | None,
) -> list[dict[str, Any]]:
    """Compose the projection the properties expect, independently of the plugin's own composition.

    Written as the requirement reads — five blocks in a fixed order, each name taken at most once, the incoming order
    inside every block — rather than by calling the plugin's composer, which would make the assertion circular.

    Args:
        incoming: Specifications received in the call, in arrival order.
        exposed: Names with a live exposure.
        referenced: Names the retained history references.
        always_available: Names configured to carry a full specification on every call.
        catalog_tokens: Catalog budget in tokens, or ``None`` to omit every catalog entry.

    Returns:
        The expected projection: the very incoming objects for the four full-specification blocks, and freshly built
        reduced entries for the catalog block.
    """
    expected: list[dict[str, Any]] = []
    seen: set[str] = set()

    for block in ({FIND_TOOLS_NAME}, set(always_available), set(exposed), set(referenced)):
        for spec in incoming:
            if spec["name"] in block and spec["name"] not in seen:
                seen.add(spec["name"])
                expected.append(spec)

    if catalog_tokens is not None:
        for spec in incoming:
            if spec["name"] not in seen:
                seen.add(spec["name"])
                expected.append(
                    {
                        "name": spec["name"],
                        "description": _CATALOG_SIGIL
                        + _truncate_description(spec["description"], catalog_tokens),
                        "inputSchema": copy.deepcopy(EMPTY_CLOSED_SCHEMA),
                    }
                )

    return expected


def _project_once(
    *,
    names: Sequence[str],
    find_tools_position: int = 0,
    exposed: Sequence[str] = (),
    referenced: Sequence[str] = (),
    always_available: Sequence[str] = (),
    catalog_tokens: int | None = 20,
    ttl_cycles: int = 5,
    cycle: int = 0,
    catalog_in_system_prompt: bool = False,
) -> tuple[ProgressiveToolDisclosure, Agent, InvokeModelContext, InvokeModelContext]:
    """Run one projection end to end, and hand back everything an assertion may need to look at.

    Args:
        names: Tool names the call offers, excluding the search tool.
        find_tools_position: Where the search tool sits in the incoming order.
        exposed: Names to expose, live at ``cycle``.
        referenced: Names the retained history references.
        always_available: Names configured to carry a full specification on every call.
        catalog_tokens: Catalog budget in tokens, or ``None``.
        ttl_cycles: Cycles an exposure survives after its last use.
        cycle: Cycle counter to run the call at.
        catalog_in_system_prompt: Place the catalog in the system prompt instead of in ``tool_specs``.

    Returns:
        The plugin, the agent, the context received by the handler, and the context it returned.
    """
    plugin = ProgressiveToolDisclosure(
        catalog_tokens=catalog_tokens,
        ttl_cycles=ttl_cycles,
        always_available=tuple(always_available),
        index=_RecordingIndex(),
        catalog_in_system_prompt=catalog_in_system_prompt,
    )
    agent = _agent(plugin)
    _seed_state(plugin, agent, exposed, cycle)

    context = _context(agent, _incoming_specs(agent, names, find_tools_position), _messages(referenced))
    return plugin, agent, context, _run(plugin._projection_handler(context))


@PROPERTY_SETTINGS
@given(
    names=names_strategy,
    find_tools_position=position_strategy,
    exposed=subset_strategy,
    referenced=subset_strategy,
    always_available=subset_strategy,
    catalog_tokens=catalog_tokens_strategy,
    ttl_cycles=ttl_strategy,
    cycle=cycle_strategy,
)
def test_the_projection_is_the_ordered_de_duplicated_union_of_the_five_blocks(
    names: list[str],
    find_tools_position: int,
    exposed: list[str],
    referenced: list[str],
    always_available: list[str],
    catalog_tokens: int | None,
    ttl_cycles: int,
    cycle: int,
) -> None:
    """Feature: progressive-tool-disclosure-plugin, Property 1.

    Projection is the ordered, de-duplicated union of the five blocks.

    Validates: Requirements 3.1, 3.2, 3.3.
    """
    _, _, context, result = _project_once(
        names=names,
        find_tools_position=find_tools_position,
        exposed=exposed,
        referenced=referenced,
        always_available=always_available,
        catalog_tokens=catalog_tokens,
        ttl_cycles=ttl_cycles,
        cycle=cycle,
    )

    incoming = context.tool_specs
    expected = _expected_projection(incoming, exposed, referenced, always_available, catalog_tokens)

    # Requirement 3.1: the union, in the fixed block order, with the incoming order inside each block.
    assert _names(result.tool_specs) == _names(expected)
    assert result.tool_specs == expected

    # Requirement 3.3: each name once, so no tool is described to the provider twice.
    assert len(_names(result.tool_specs)) == len(set(_names(result.tool_specs)))

    # Requirement 3.2: a name any of the first four blocks reached carries the specification that arrived — asserted by
    # identity, which a re-described or rebuilt spec could not satisfy. Everything else is a reduced entry.
    full_spec_names = {FIND_TOOLS_NAME, *always_available, *exposed, *referenced} & set(_names(incoming))
    incoming_by_name = {spec["name"]: spec for spec in incoming}
    for name, spec in _projected(result).items():
        if name in full_spec_names:
            assert spec is incoming_by_name[name], f"{name} did not arrive at full specification"
        else:
            assert _is_catalog_entry(spec), f"{name} did not arrive as a catalog entry"


@PROPERTY_SETTINGS
@given(
    names=names_strategy,
    find_tools_position=position_strategy,
    exposed=subset_strategy,
    referenced=st.lists(st.sampled_from([*TOOL_NAMES, *UNKNOWN_NAMES]), max_size=4, unique=True),
    always_available=st.lists(st.sampled_from([*TOOL_NAMES, *UNKNOWN_NAMES]), max_size=3, unique=True),
    catalog_tokens=catalog_tokens_strategy,
    cycle=cycle_strategy,
)
def test_the_projection_names_are_a_subset_of_the_incoming_names(
    names: list[str],
    find_tools_position: int,
    exposed: list[str],
    referenced: list[str],
    always_available: list[str],
    catalog_tokens: int | None,
    cycle: int,
) -> None:
    """Feature: progressive-tool-disclosure-plugin, Property 2.

    Projection names are a subset of incoming names.

    Validates: Requirements 3.4.
    """
    # Exposures, references and ``always_available`` all range over names the call may not be offering, including names
    # no registry has: none of those channels may put a name into the projection on its own.
    _, _, context, result = _project_once(
        names=names,
        find_tools_position=find_tools_position,
        exposed=exposed,
        referenced=referenced,
        always_available=always_available,
        catalog_tokens=catalog_tokens,
        cycle=cycle,
    )

    incoming_names = set(_names(context.tool_specs))

    assert set(_names(result.tool_specs)) <= incoming_names
    # Requirement 3.10: a configured name the call did not offer is omitted rather than raised over.
    assert not set(_names(result.tool_specs)) & set(UNKNOWN_NAMES)


@PROPERTY_SETTINGS
@given(
    names=names_strategy,
    find_tools_position=position_strategy,
    exposed=subset_strategy,
    referenced=subset_strategy,
    always_available=subset_strategy,
    catalog_tokens=catalog_tokens_strategy,
    ttl_cycles=ttl_strategy,
    cycle=cycle_strategy,
)
def test_the_projection_is_never_empty_and_always_leads_with_the_search_tool(
    names: list[str],
    find_tools_position: int,
    exposed: list[str],
    referenced: list[str],
    always_available: list[str],
    catalog_tokens: int | None,
    ttl_cycles: int,
    cycle: int,
) -> None:
    """Feature: progressive-tool-disclosure-plugin, Property 3.

    Projection is never empty and always contains the search tool.

    Validates: Requirements 3.5, 9.5.
    """
    _, _, context, result = _project_once(
        names=names,
        find_tools_position=find_tools_position,
        exposed=exposed,
        referenced=referenced,
        always_available=always_available,
        catalog_tokens=catalog_tokens,
        ttl_cycles=ttl_cycles,
        cycle=cycle,
    )

    assert result is not context, "the projection did not apply"

    # Requirement 9.5: a non-empty ``toolConfig`` is what keeps the provider's protocol error out of reach.
    assert len(result.tool_specs) >= 1
    assert FIND_TOOLS_NAME in _names(result.tool_specs)

    # First, whatever position it arrived in: that ordering is what makes the projection non-empty by construction
    # rather than by the accident of some other block having a member.
    assert _names(result.tool_specs)[0] == FIND_TOOLS_NAME


@PROPERTY_SETTINGS
@given(names=names_strategy, find_tools_position=position_strategy, cycle=cycle_strategy)
def test_the_leanest_configuration_still_projects_the_search_tool(
    names: list[str],
    find_tools_position: int,
    cycle: int,
) -> None:
    """Feature: progressive-tool-disclosure-plugin, Property 3.

    Projection is never empty and always contains the search tool.

    Validates: Requirements 3.5, 9.5.
    """
    # The one configuration in which every other block is empty: no catalog, no exposure, no reference, nothing always
    # available. What is left is the floor of the projection.
    _, _, _, result = _project_once(
        names=names,
        find_tools_position=find_tools_position,
        catalog_tokens=None,
        cycle=cycle,
    )

    assert _names(result.tool_specs) == [FIND_TOOLS_NAME]


@PROPERTY_SETTINGS
@given(
    names=names_strategy,
    find_tools_position=position_strategy,
    exposed=subset_strategy,
    referenced=subset_strategy,
    always_available=subset_strategy,
    ttl_cycles=ttl_strategy,
    cycle=cycle_strategy,
)
def test_a_suppressed_catalog_omits_every_reduced_entry_and_keeps_the_full_spec_blocks(
    names: list[str],
    find_tools_position: int,
    exposed: list[str],
    referenced: list[str],
    always_available: list[str],
    ttl_cycles: int,
    cycle: int,
) -> None:
    """Feature: progressive-tool-disclosure-plugin, Property 4.

    A suppressed catalog omits every reduced entry and keeps the full-spec blocks.

    Validates: Requirements 3.6.
    """
    _, _, context, result = _project_once(
        names=names,
        find_tools_position=find_tools_position,
        exposed=exposed,
        referenced=referenced,
        always_available=always_available,
        catalog_tokens=None,
        ttl_cycles=ttl_cycles,
        cycle=cycle,
    )

    incoming = context.tool_specs
    incoming_by_name = {spec["name"]: spec for spec in incoming}
    exp_full_spec = {FIND_TOOLS_NAME, *always_available, *exposed, *referenced} & set(incoming_by_name)

    # ``catalog_tokens=None`` removes exactly one block. The other four are unaffected, so the projection is precisely
    # the names the four of them reach — nothing reduced, and nothing missing either.
    assert set(_names(result.tool_specs)) == exp_full_spec
    for name, spec in _projected(result).items():
        assert spec is incoming_by_name[name], f"{name} was reduced despite the catalog being suppressed"
        assert not _is_catalog_entry(spec)


@PROPERTY_SETTINGS
@given(
    names=names_strategy,
    find_tools_position=position_strategy,
    exposed=subset_strategy,
    referenced=subset_strategy,
    always_available=subset_strategy,
    catalog_tokens=catalog_tokens_strategy,
    ttl_cycles=ttl_strategy,
    cycle=cycle_strategy,
)
def test_the_projection_is_deterministic(
    names: list[str],
    find_tools_position: int,
    exposed: list[str],
    referenced: list[str],
    always_available: list[str],
    catalog_tokens: int | None,
    ttl_cycles: int,
    cycle: int,
) -> None:
    """Feature: progressive-tool-disclosure-plugin, Property 5.

    The projection is deterministic.

    Validates: Requirements 3.8.
    """
    configuration: dict[str, Any] = {
        "names": names,
        "find_tools_position": find_tools_position,
        "exposed": exposed,
        "referenced": referenced,
        "always_available": always_available,
        "catalog_tokens": catalog_tokens,
        "ttl_cycles": ttl_cycles,
        "cycle": cycle,
    }

    plugin, agent, _, first = _project_once(**configuration)

    # Same plugin, same agent, same state: a second call in the same cycle has to produce the same list. Built as a
    # fresh context over fresh copies of the incoming specifications, so equality here is about content and order and
    # not about the two calls having been handed the same objects.
    repeat = _context(
        agent,
        _incoming_specs(agent, names, find_tools_position),
        _messages(referenced),
    )
    second = _run(plugin._projection_handler(repeat))

    assert _names(second.tool_specs) == _names(first.tool_specs)
    assert second.tool_specs == first.tool_specs

    # And across instances: an identically configured plugin on an identically built agent projects the same list, so
    # the order is a function of the state, the history and the configuration alone.
    _, _, _, elsewhere = _project_once(**configuration)
    assert _names(elsewhere.tool_specs) == _names(first.tool_specs)
    assert elsewhere.tool_specs == first.tool_specs


@PROPERTY_SETTINGS
@given(
    names=names_strategy,
    find_tools_position=position_strategy,
    exposed=subset_strategy,
    referenced=subset_strategy,
    always_available=subset_strategy,
    catalog_tokens=catalog_tokens_strategy,
    ttl_cycles=ttl_strategy,
    cycle=cycle_strategy,
)
def test_the_projection_changes_only_tool_specs(
    names: list[str],
    find_tools_position: int,
    exposed: list[str],
    referenced: list[str],
    always_available: list[str],
    catalog_tokens: int | None,
    ttl_cycles: int,
    cycle: int,
) -> None:
    """Feature: progressive-tool-disclosure-plugin, Property 6.

    The projection changes only ``tool_specs``.

    Validates: Requirements 3.9.
    """
    plugin = ProgressiveToolDisclosure(
        catalog_tokens=catalog_tokens,
        ttl_cycles=ttl_cycles,
        always_available=tuple(always_available),
        index=_RecordingIndex(),
    )
    agent = _agent(plugin)
    _seed_state(plugin, agent, exposed, cycle)

    incoming = _incoming_specs(agent, names, find_tools_position)
    messages = _messages(referenced)
    context = _context(agent, incoming, messages)

    exp_incoming = copy.deepcopy(incoming)
    exp_messages = copy.deepcopy(messages)
    exp_registry = {name: copy.deepcopy(entry.tool_spec) for name, entry in agent.tool_registry.registry.items()}
    exp_tool_names = sorted(agent.tool_names)

    result = _run(plugin._projection_handler(context))

    # Every declared field other than ``tool_specs`` is carried over by the same object, so the handler cannot have
    # rewritten one in a way a value comparison would forgive.
    for field in dataclasses.fields(InvokeModelContext):
        if field.name != "tool_specs":
            assert getattr(result, field.name) is getattr(context, field.name), f"the handler changed {field.name}"

    # The received list is a defensive copy the handler is free to replace but not to edit: the call it was handed has
    # to read the same after the projection as before it.
    assert context.tool_specs == exp_incoming
    assert result.messages is messages
    assert messages == exp_messages

    # And nothing reached past the context: the registry the agent keeps is the source of every full specification, so
    # a projection that edited it would be rewriting the tool rather than the call.
    assert {name: entry.tool_spec for name, entry in agent.tool_registry.registry.items()} == exp_registry
    assert sorted(agent.tool_names) == exp_tool_names


@PROPERTY_SETTINGS
@given(
    names=names_strategy,
    find_tools_position=position_strategy,
    exposed=subset_strategy,
    unknown_referenced=st.lists(st.sampled_from(UNKNOWN_NAMES), max_size=2, unique=True),
    catalog_tokens=catalog_tokens_strategy,
    ttl_cycles=ttl_strategy,
    cycle=cycle_strategy,
)
def test_history_referenced_tools_are_kept_and_unknown_references_are_dropped(
    names: list[str],
    find_tools_position: int,
    exposed: list[str],
    unknown_referenced: list[str],
    catalog_tokens: int | None,
    ttl_cycles: int,
    cycle: int,
) -> None:
    """Feature: progressive-tool-disclosure-plugin, Property 20.

    History-referenced tools are kept, unknown references are dropped.

    Validates: Requirements 9.3, 9.4.
    """
    # The history references everything the call offers, plus names nothing has. The offered ones must come back at
    # full specification whatever the exposure state and whatever the budget says; the others must simply not appear.
    referenced = [*names, *unknown_referenced]

    plugin = ProgressiveToolDisclosure(
        catalog_tokens=catalog_tokens,
        ttl_cycles=ttl_cycles,
        index=_RecordingIndex(),
    )
    agent = _agent(plugin)
    _seed_state(plugin, agent, exposed, cycle)

    incoming = _incoming_specs(agent, names, find_tools_position)
    context = _context(agent, incoming, _messages(referenced))

    result = _run(plugin._projection_handler(context))
    projected = _projected(result)
    incoming_by_name = {spec["name"]: spec for spec in incoming}

    # Requirement 9.3: a ``toolUse`` in the retained history without its definition in the call is a protocol error, so
    # a referenced tool keeps its full specification even under a suppressed catalog and with no exposure at all.
    for name in names:
        assert projected[name] is incoming_by_name[name], f"{name} was referenced but not projected in full"

    # Requirement 9.4: an unknown reference is omitted, and the projection finishes rather than raising over it.
    assert not set(projected) & set(unknown_referenced)
    assert set(projected) == set(incoming_by_name)


@PROPERTY_SETTINGS
@given(
    names=st.lists(st.sampled_from(TOOL_NAMES), min_size=2, max_size=len(TOOL_NAMES), unique=True),
    find_tools_position=position_strategy,
    repeats=st.integers(min_value=1, max_value=4),
    catalog_tokens=catalog_tokens_strategy,
    cycle=cycle_strategy,
)
def test_the_index_is_built_once_per_registry_fingerprint(
    names: list[str],
    find_tools_position: int,
    repeats: int,
    catalog_tokens: int | None,
    cycle: int,
) -> None:
    """Feature: progressive-tool-disclosure-plugin, Property 14.

    The index is built once per registry fingerprint.

    Validates: Requirements 6.5, 6.6, 6.7, 6.11.
    """
    # The last name stands for a tool that arrives late: registered all along, but only offered from the second round of
    # calls on. That is what a runtime registration looks like from the projection's side — the fingerprint changes.
    late = names[-1]
    initial = names[:-1]

    index = _RecordingIndex()
    plugin = ProgressiveToolDisclosure(catalog_tokens=catalog_tokens, index=index)
    agent = _agent(plugin)
    _seed_state(plugin, agent, (), cycle)
    state = plugin._states[agent]

    # Requirements 6.5 and 6.6: built once on the first projection, and not again while the fingerprint holds.
    for _ in range(repeats):
        _run(plugin._projection_handler(_context(agent, _incoming_specs(agent, initial, find_tools_position), [])))

    assert len(index.builds) == 1, f"the index was built {len(index.builds)} times for one fingerprint"
    assert state.fingerprint == frozenset([FIND_TOOLS_NAME, *initial])
    assert set(index.builds[0]) == frozenset([FIND_TOOLS_NAME, *initial])

    # Requirement 6.7: the late tool changes the fingerprint, which buys exactly one rebuild, however many calls the
    # new fingerprint then sees.
    for _ in range(repeats):
        _run(plugin._projection_handler(_context(agent, _incoming_specs(agent, names, find_tools_position), [])))

    assert len(index.builds) == 2, f"a changed fingerprint produced {len(index.builds)} builds"
    assert state.fingerprint == frozenset([FIND_TOOLS_NAME, *names])
    assert set(index.builds[1]) == frozenset([FIND_TOOLS_NAME, *names])

    # Requirement 6.11: and it is findable from that projection on — the rebuild is what makes the late tool reachable
    # through the search rather than merely present in the call.
    assert index.searches == []
    _run(plugin.find_tools(need=late.replace("_", " "), tool_context=_tool_context(agent)))

    assert index.searches == [late.replace("_", " ")]
    assert late in state.exposed, f"{late} was indexed but not findable"


@PROPERTY_SETTINGS
@given(
    names=names_strategy,
    find_tools_position=position_strategy,
    guard=st.sampled_from(["unknown_name", "no_search_tool", "both"]),
    unknown=st.sampled_from(UNKNOWN_NAMES),
    exposed=subset_strategy,
    referenced=subset_strategy,
    always_available=subset_strategy,
    catalog_tokens=catalog_tokens_strategy,
    cycle=cycle_strategy,
)
def test_passthrough_triggers_on_either_structural_guard(
    names: list[str],
    find_tools_position: int,
    guard: str,
    unknown: str,
    exposed: list[str],
    referenced: list[str],
    always_available: list[str],
    catalog_tokens: int | None,
    cycle: int,
) -> None:
    """Feature: progressive-tool-disclosure-plugin, Property 19.

    Passthrough triggers on either structural guard.

    Validates: Requirements 9.1, 9.2, 9.9.
    """
    index = _RecordingIndex()
    plugin = ProgressiveToolDisclosure(
        catalog_tokens=catalog_tokens,
        always_available=tuple(always_available),
        index=index,
    )
    agent = _agent(plugin)
    _seed_state(plugin, agent, exposed, cycle)

    incoming = _incoming_specs(agent, names, find_tools_position)

    if guard in ("unknown_name", "both"):
        # Requirement 9.9: the forced-structured-output case is recognized by the unregistered name itself, not by a
        # mode flag on the context — so a synthetic specification is all it takes to abstain.
        synthetic: dict[str, Any] = {"name": unknown, "description": "x", "inputSchema": {}}
        incoming.insert(min(find_tools_position, len(incoming)), synthetic)
    if guard in ("no_search_tool", "both"):
        # The window between ``init_agent`` and the plugin registry registering the vended tool: with no way back to a
        # hidden schema, there is nothing to hide.
        incoming = [spec for spec in incoming if spec["name"] != FIND_TOOLS_NAME]

    context = _context(agent, incoming, _messages(referenced))
    exp_incoming = copy.deepcopy(incoming)

    result = _run(plugin._projection_handler(context))

    # Requirements 9.1 and 9.2: unchanged by object identity, not merely equal — the stage gets back what it handed in.
    assert result is context
    assert result.tool_specs is incoming
    assert incoming == exp_incoming

    # Abstaining is total: a passthrough reads no state, builds no index and records no fingerprint.
    assert index.builds == []
    assert plugin._states[agent].fingerprint is None


@PROPERTY_SETTINGS
@given(
    names=names_strategy,
    find_tools_position=position_strategy,
    exposed=subset_strategy,
    referenced=subset_strategy,
    catalog_tokens=catalog_tokens_strategy,
    cycle=cycle_strategy,
)
def test_the_projection_applies_when_neither_structural_guard_holds(
    names: list[str],
    find_tools_position: int,
    exposed: list[str],
    referenced: list[str],
    catalog_tokens: int | None,
    cycle: int,
) -> None:
    """Feature: progressive-tool-disclosure-plugin, Property 19.

    Passthrough triggers on either structural guard.

    Validates: Requirements 9.1, 9.2.
    """
    # The other side of the biconditional. Without it, a handler that passed everything through would satisfy the
    # passthrough test above and reduce nothing at all.
    _, _, context, result = _project_once(
        names=names,
        find_tools_position=find_tools_position,
        exposed=exposed,
        referenced=referenced,
        catalog_tokens=catalog_tokens,
        cycle=cycle,
    )

    assert result is not context
    assert result.tool_specs is not context.tool_specs


@PROPERTY_SETTINGS
@given(
    names=st.lists(st.sampled_from(TOOL_NAMES), min_size=1, max_size=len(TOOL_NAMES), unique=True),
    first_need=need_strategy,
    second_need=need_strategy,
    premature=st.sampled_from(TOOL_NAMES),
    always_available=subset_strategy,
    catalog_tokens=catalog_tokens_strategy,
    ttl_cycles=ttl_strategy,
    cycle=cycle_strategy,
)
def test_per_agent_state_is_isolated(
    names: list[str],
    first_need: str,
    second_need: str,
    premature: str,
    always_available: list[str],
    catalog_tokens: int | None,
    ttl_cycles: int,
    cycle: int,
) -> None:
    """Feature: progressive-tool-disclosure-plugin, Property 21.

    Per-agent state is isolated.

    Validates: Requirements 1.4, 10.5.
    """
    plugin = ProgressiveToolDisclosure(
        catalog_tokens=catalog_tokens,
        ttl_cycles=ttl_cycles,
        always_available=tuple(always_available),
        index=_RecordingIndex(),
    )
    first = _agent(plugin)
    second = _agent(plugin)

    # Requirement 10.4: constructing an agent does not create a disclosure state. So "the other agent is unchanged"
    # starts out as "the other agent has no state at all", which is the strongest form of the claim.
    assert plugin._states.get(first) is None
    assert plugin._states.get(second) is None

    first.event_loop_metrics.cycle_count = cycle
    second.event_loop_metrics.cycle_count = cycle

    # Every channel that writes state, exercised on the first agent only: a projection, a search, and a premature call.
    _run(plugin._projection_handler(_context(first, _incoming_specs(first, names), [])))
    _run(plugin.find_tools(need=first_need, tool_context=_tool_context(first)))
    plugin._on_before_tool_call(_before_tool_call(first, premature))

    assert plugin._states.get(second) is None, "activity on one agent created state on another"
    after_first = copy.deepcopy(dataclasses.asdict(plugin._states[first]))

    # Now the second agent runs the same channels with a need of its own.
    _run(plugin._projection_handler(_context(second, _incoming_specs(second, names), [])))
    _run(plugin.find_tools(need=second_need, tool_context=_tool_context(second)))
    plugin._on_before_tool_call(_before_tool_call(second, premature))

    # Requirement 1.4 and 10.5: two agents, one plugin instance, two states — and updating one leaves the other exactly
    # as it was, down to the observability counters.
    assert plugin._states[first] is not plugin._states[second]
    assert dataclasses.asdict(plugin._states[first]) == after_first, "the second agent's activity reached the first"
    assert plugin._states[second].exposed is not plugin._states[first].exposed
