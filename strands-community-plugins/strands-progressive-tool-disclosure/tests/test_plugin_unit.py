"""Unit tests for construction, the two guidance branches of the search, and the without-plugin case.

Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.11 (task 16.1) and Requirements
2.8, 5.7, 9.7, 10.4, 10.6, 12.5 (task 16.2).

These are plain deterministic examples rather than properties. Every claim here is about a single
point instead of a space: a default is one value, a rejected parameter is one message, a blank need is
one branch, and "the plugin is not registered" is one arrangement. A generator would sample the same
example repeatedly and hide, behind a property name, the fact that nothing varies.

The redesign moved the catalog into the system prompt alone and split discovery in two: ``find_tools``
searches and lists, ``get_tool_details`` is the one thing that loads a schema. So the budget asserted
here is ``catalog_chars`` -- characters, not tokens -- the summarizer is a constructor parameter with its
own validation, and a search records no exposure at all. What a loading call does with an exposure is
asserted in ``tests/test_details_and_summary.py``; this file only uses one to have an exposure to lose.

Everything runs offline. The model is a stub that raises if it is ever called, the summarizer is a
deterministic stub so no description can reach that model, the search index is a deterministic double
that records what it was asked, and the constructor tests run with the socket layer replaced by one that
raises -- so "no network call at construction" is enforced rather than assumed, which is what
Requirement 12.5 asks of the suite as a whole.
"""

import asyncio
import copy
import dataclasses
import gc
import socket
import weakref
from collections.abc import Coroutine, Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, TypeVar, cast

import pytest
from strands import Agent
from strands.models.model import Model
from strands.tools.decorator import tool
from strands.types.tools import ToolContext, ToolSpec

from strands_progressive_tool_disclosure import LexicalToolIndex, ProgressiveToolDisclosure, ToolMatch
from strands_progressive_tool_disclosure._compat import InvokeModelContext, InvokeModelStage
from strands_progressive_tool_disclosure.plugin import (
    _DEFAULT_CATALOG_CHARS,
    _DEFAULT_TOP_K,
    _DEFAULT_TTL_CYCLES,
    _EMPTY_NEED_GUIDANCE,
    _MATCHES_HEADER,
    _NO_MATCH_GUIDANCE,
    _PLUGIN_TOOL_NAMES,
    FIND_TOOLS_NAME,
    GET_TOOL_DETAILS_NAME,
)

_Resolved = TypeVar("_Resolved")


class _StubModel(Model):
    """A model that exists only so an ``Agent`` can be constructed. A call to it is a test failure."""

    def update_config(self, **kwargs: Any) -> None:
        """Accept anything; nothing reads it."""

    def get_config(self) -> dict[str, Any]:
        """No configuration to report."""
        return {}

    async def stream(self, *args: Any, **kwargs: Any) -> Any:
        """Refuse: these tests are about construction and projection, never about a model call."""
        raise AssertionError("the language model was called")
        yield

    async def structured_output(self, *args: Any, **kwargs: Any) -> Any:
        """Refuse: these tests are about construction and projection, never about a model call."""
        raise AssertionError("the language model was called")
        yield


def _stub_summarizer(spec: ToolSpec, max_chars: int) -> str:
    """Write a catalog line without a model call, so every example stays offline and byte-stable.

    A description longer than the limit would otherwise reach the default summarizer, which calls the
    agent's own model -- the stub above, which fails the test. Passing this instead keeps the assertions
    about construction, search and projection rather than about a fallback path.

    Args:
        spec: Full specification as registered.
        max_chars: Character limit of the line.

    Returns:
        A deterministic line derived from the tool's name.
    """
    return f"summary of {spec['name']}"[:max_chars]


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


class _RecordingIndex:
    """Search double that returns a fixed ranking and records every build and every search."""

    def __init__(self, ranked: tuple[str, ...] = ()) -> None:
        """Rank ``ranked`` on every search, regardless of the need."""
        self._ranked = ranked
        self.built: list[list[str]] = []
        self.needs: list[str] = []

    def build(self, specs: list[dict[str, Any]]) -> None:
        """Record the names offered for indexing, in order."""
        self.built.append([spec["name"] for spec in specs])

    def search(self, need: str, top_k: int) -> list[ToolMatch]:
        """Record the need and return the fixed ranking, clipped to ``top_k``."""
        self.needs.append(need)
        return [ToolMatch(name=name, score=float(len(self._ranked) - i)) for i, name in enumerate(self._ranked)][:top_k]


def _plugin(**overrides: Any) -> ProgressiveToolDisclosure:
    """Build the plugin with the offline defaults these tests need.

    Args:
        **overrides: Constructor arguments to replace the offline defaults with.

    Returns:
        A plugin whose search is a recording double and whose summarizer never calls a model.
    """
    return ProgressiveToolDisclosure(**{"index": _RecordingIndex(), "summarizer": _stub_summarizer, **overrides})


def _agent(plugin: ProgressiveToolDisclosure | None = None) -> Agent:
    """Build an offline agent carrying the two registered tools, with or without the plugin.

    Args:
        plugin: The plugin to register, or ``None`` for the without-plugin arrangement — the baseline
            Requirement 9.7 compares against.

    Returns:
        An agent whose registry is real, so the projected and registered specifications are the ones
        the assertions read.
    """
    return Agent(
        model=_StubModel(),
        tools=[list_accounts, send_wire],
        plugins=[plugin] if plugin is not None else [],
    )


def _model_call(agent: Agent) -> InvokeModelContext:
    """Build the invocation context of one model call, offering every registered specification.

    Args:
        agent: Agent of the call.

    Returns:
        A context whose ``tool_specs`` are the registered full specifications, which is what keeps the
        call off the passthrough path: both plugin tools are in the registry, so both are offered.
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


def _run(step: Coroutine[Any, Any, _Resolved]) -> _Resolved:
    """Drive one awaited plugin step to completion from a synchronous test.

    Args:
        step: The coroutine to run: a projection, a search or a load.

    Returns:
        What the step returned.
    """
    return asyncio.run(step)


def _tool_context(agent: Agent) -> ToolContext:
    """Build the injected context a search or a load invocation receives."""
    return cast("ToolContext", SimpleNamespace(agent=agent))


def _input_handler_count(agent: Agent) -> int:
    """How many input-phase handlers the agent holds for the invoke-model stage.

    This is where a registered projection handler shows up, so it is also how the absence of one is
    asserted: the registry wraps the handler before storing it, leaving the count as the only thing
    that can be compared.
    """
    stage = InvokeModelStage.Input._stage
    return sum(1 for tagged in agent._middleware_registry._handlers.get(stage, []) if tagged.phase == "input")


@contextmanager
def _no_network() -> Iterator[None]:
    """Replace the socket layer with one that raises, for the duration of the block.

    Requirement 2.8 is a claim about what construction does *not* do, and an assertion over a return
    value cannot express it. Taking the network away is what makes the claim testable: a constructor
    that resolved a model endpoint, fetched an embedding or summarized a description would raise here
    instead of passing quietly.
    """
    real_socket, real_connection = socket.socket, socket.create_connection

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the network was used")

    socket.socket, socket.create_connection = refuse, refuse  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket, socket.create_connection = real_socket, real_connection


# ---------------------------------------------------------------------------------------------------
# Task 16.1 - constructor defaults and validation errors
# ---------------------------------------------------------------------------------------------------


def test_the_constructor_adopts_the_documented_defaults():
    """Requirements 2.6, 2.7, 2.11: the defaults are 80 chars / no summarizer / 5 / 3 / () / lexical."""
    plugin = ProgressiveToolDisclosure()

    # The budget is counted in CHARACTERS: the catalog line is a summary of a description, and both are
    # text the plugin measures itself rather than numbers a tokenizer reports.
    assert plugin._catalog_chars == _DEFAULT_CATALOG_CHARS == 80
    # No summarizer means the agent's own model writes the lines, decided per call rather than here.
    assert plugin._summarizer is None
    assert plugin._ttl_cycles == _DEFAULT_TTL_CYCLES == 3
    assert plugin._top_k == _DEFAULT_TOP_K == 3
    assert plugin._always_available == ()
    # Requirement 2.6: no index given means the standard-library lexical one, which needs no network.
    assert isinstance(plugin._index, LexicalToolIndex)
    # Requirement 12.6: the community plugin reports the same identifier as the vended one.
    assert plugin.name == "strands:progressive-tool-disclosure"


def test_a_configured_value_replaces_the_default_and_the_sequence_is_frozen():
    """Requirement 2.10: what is passed is what the instance keeps, immune to a later mutation."""
    mutable = ["list_accounts"]
    index = _RecordingIndex()

    plugin = ProgressiveToolDisclosure(
        catalog_chars=None,
        summarizer=_stub_summarizer,
        ttl_cycles=2,
        top_k=7,
        always_available=mutable,
        index=index,
    )
    mutable.append("send_wire")

    assert plugin._catalog_chars is None
    assert plugin._summarizer is _stub_summarizer
    assert plugin._ttl_cycles == 2
    assert plugin._top_k == 7
    assert plugin._index is index
    # The caller's list cannot reconfigure the instance after the fact: the tuple is a copy.
    assert plugin._always_available == ("list_accounts",)


@pytest.mark.parametrize(
    ("kwargs", "expected_fragments"),
    [
        # Requirement 2.1: None or an integer >= 1, and neither a bool nor a float counts. Zero is
        # refused rather than treated as a suppression: that is what None is for, and a limit of zero
        # characters would emit catalog lines nothing fits in.
        ({"catalog_chars": 0}, ("catalog_chars", "None", "1")),
        ({"catalog_chars": -1}, ("catalog_chars", "1")),
        ({"catalog_chars": True}, ("catalog_chars", "1")),
        ({"catalog_chars": 80.0}, ("catalog_chars", "1")),
        ({"catalog_chars": "80"}, ("catalog_chars", "1")),
        # The summarizer is called per tool on the projection path, so a wrong type has to fail here
        # rather than on the first call that meets a long description.
        ({"summarizer": 7}, ("summarizer", "callable")),
        ({"summarizer": "use the model"}, ("summarizer", "callable")),
        ({"summarizer": ["a summary"]}, ("summarizer", "callable")),
        # Requirement 2.2: ttl_cycles is counted in cycles, so only an integer >= 1 has a meaning.
        ({"ttl_cycles": 0}, ("ttl_cycles", "1")),
        ({"ttl_cycles": -3}, ("ttl_cycles", "1")),
        ({"ttl_cycles": True}, ("ttl_cycles", "1")),
        ({"ttl_cycles": 5.0}, ("ttl_cycles", "1")),
        ({"ttl_cycles": None}, ("ttl_cycles", "1")),
        # Requirement 2.3: same for top_k, which is a count of tools one search lists.
        ({"top_k": 0}, ("top_k", "1")),
        ({"top_k": True}, ("top_k", "1")),
        ({"top_k": 3.0}, ("top_k", "1")),
        ({"top_k": "3"}, ("top_k", "1")),
        # Requirement 2.4: a sequence of non-empty strings. A bare string would configure one name
        # per character, so it is caught rather than honored.
        ({"always_available": "list_accounts"}, ("always_available", "strings")),
        ({"always_available": b"list_accounts"}, ("always_available", "strings")),
        ({"always_available": [""]}, ("always_available", "strings")),
        ({"always_available": ["list_accounts", 7]}, ("always_available", "strings")),
        ({"always_available": 7}, ("always_available", "strings")),
        # Requirement 2.5: the message names the member that is missing or is not callable.
        ({"index": object()}, ("index", "build")),
        ({"index": SimpleNamespace(build=lambda specs: None)}, ("index", "search")),
        ({"index": SimpleNamespace(build=None, search=lambda need, top_k: [])}, ("index", "build")),
        # Requirement 2.11: None or a callable taking the agent.
    ],
)
def test_an_invalid_parameter_raises_a_value_error_naming_the_parameter(
    kwargs: dict[str, Any],
    expected_fragments: tuple[str, ...],
):
    """Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.11.

    The message is part of the contract: the developer reading it has to learn which parameter was
    refused and what it accepts, without opening the source.
    """
    with pytest.raises(ValueError) as failure:
        ProgressiveToolDisclosure(**kwargs)

    message = str(failure.value)
    for fragment in expected_fragments:
        assert fragment in message


def test_none_suppresses_the_catalog_and_is_not_confused_with_a_refused_zero():
    """Requirement 2.1: ``None`` is a supported configuration, ``0`` is a mistake.

    The two are adjacent in a way that invites a truthiness check, and a truthiness check would accept
    the mistake. Asserted together so the distinction cannot regress into one.
    """
    assert ProgressiveToolDisclosure(catalog_chars=None)._catalog_chars is None

    with pytest.raises(ValueError, match="catalog_chars"):
        ProgressiveToolDisclosure(catalog_chars=0)


def test_a_failed_construction_registers_nothing_and_builds_nothing():
    """Requirement 2.9: a construction that raises leaves no handler, hook or tool behind.

    Asserted against an agent that already exists, because that is the only thing a half-built plugin
    could have reached: the validators all run before ``Plugin.__init__``, so there is no instance to
    register and no index to build.
    """
    bystander = _agent()
    exp_handlers = _input_handler_count(bystander)
    exp_names = sorted(bystander.tool_names)
    index = _RecordingIndex()

    # The invalid parameter sits after ``index`` in the signature and before it in validation order,
    # so a validator that ran late would have instantiated and built something by now.
    with pytest.raises(ValueError):
        ProgressiveToolDisclosure(index=index, top_k=0)
    with pytest.raises(ValueError):
        ProgressiveToolDisclosure(index=index, always_available=[""])

    assert index.built == []
    assert _input_handler_count(bystander) == exp_handlers
    assert sorted(bystander.tool_names) == exp_names
    assert FIND_TOOLS_NAME not in bystander.tool_registry.registry
    assert GET_TOOL_DETAILS_NAME not in bystander.tool_registry.registry


def test_construction_performs_no_network_call_and_builds_neither_index_nor_summary():
    """Requirement 2.8: construction fixes the configuration and does nothing else.

    Three halves matter for a cold start. A build at construction time would index a registry the first
    projection has not offered yet, a summary would call a model before any tool is known, and a network
    call would make importing a plugin an I/O event.
    """
    index = _RecordingIndex()

    with _no_network():
        plugin = ProgressiveToolDisclosure(index=index)
        default_plugin = ProgressiveToolDisclosure()

    assert index.built == []
    assert plugin._states == {}
    # No description has been seen yet, so there is nothing to have summarized.
    assert plugin._summaries == {}
    assert isinstance(default_plugin._index, LexicalToolIndex)


def test_both_plugin_tools_are_registered_and_neither_exists_without_the_plugin():
    """Registering the plugin vends exactly two tools: the search and the load.

    They are the pair the projection carries in full on every call and the pair the passthrough guard
    looks for, so "both are registered" is the precondition every other claim rests on.
    """
    assert _PLUGIN_TOOL_NAMES == {FIND_TOOLS_NAME, GET_TOOL_DETAILS_NAME}

    with_plugin = _agent(_plugin())
    plain = _agent()

    assert _PLUGIN_TOOL_NAMES <= set(with_plugin.tool_registry.registry)
    assert not _PLUGIN_TOOL_NAMES & set(plain.tool_registry.registry)
    # The two domain tools are registered either way: the plugin adds, it never replaces.
    assert {"list_accounts", "send_wire"} <= set(plain.tool_registry.registry)


# ---------------------------------------------------------------------------------------------------
# Task 16.2 - search branches, the without-plugin case, and state creation
# ---------------------------------------------------------------------------------------------------


def test_a_search_that_ranks_nothing_returns_the_reformulation_guidance():
    """Requirement 5.7: zero matches is a wording problem, so the model is told how to try again."""
    index = _RecordingIndex(ranked=())
    plugin = _plugin(index=index)
    agent = _agent(plugin)

    result = _run(plugin.find_tools("something no tool does", _tool_context(agent)))

    assert result == _NO_MATCH_GUIDANCE
    # Requirement 5.2: the search ran exactly once, and this is the branch after it, not before.
    assert index.needs == ["something no tool does"]
    assert plugin._states[agent].exposed == {}
    assert plugin._states[agent].searches == 1


def test_a_search_whose_every_match_is_absent_from_the_registry_returns_the_same_guidance():
    """Requirement 5.7: a header with an empty list under it would read as a broken tool.

    From where the model stands this case and the zero-match one are the same one, so they answer the
    same way — and neither records an exposure, since a search no longer exposes anything at all.
    """
    index = _RecordingIndex(ranked=("unregistered_tool", "also_gone"))
    plugin = _plugin(index=index)
    agent = _agent(plugin)

    result = _run(plugin.find_tools("list the accounts of an owner", _tool_context(agent)))

    assert result == _NO_MATCH_GUIDANCE
    assert plugin._states[agent].exposed == {}
    # Requirement 5.10: a search never reaches the registry, so an absent match stays absent.
    assert "unregistered_tool" not in agent.tool_registry.registry


@pytest.mark.parametrize("need", ["", " ", "\t\n  "])
def test_a_blank_need_short_circuits_the_search_and_asks_for_a_description(need: str):
    """Requirement 5.6: a blank need cannot rank anything, so ``search`` is invoked zero times.

    The cycle is still counted: what the counter measures is cycles spent searching, and this one was
    spent whether or not the index was consulted.
    """
    index = _RecordingIndex(ranked=("list_accounts",))
    plugin = _plugin(index=index)
    agent = _agent(plugin)

    result = _run(plugin.find_tools(need, _tool_context(agent)))

    assert result == _EMPTY_NEED_GUIDANCE
    assert index.needs == []
    assert plugin._states[agent].exposed == {}
    assert plugin._states[agent].searches == 1


def test_a_search_that_matches_a_registered_tool_lists_it_and_loads_nothing():
    """Requirements 5.4, 5.5: the guidance branches are only meaningful against a listing one.

    Without this example, both branches above would pass for a search that can never list anything.

    A search is now search alone: it lists the name and one line about it, points at the loading tool,
    and records no exposure. Handing out the schema here would make the two tools two ways to do the
    same thing, and the model would have no reason to prefer either.
    """
    index = _RecordingIndex(ranked=("list_accounts",))
    plugin = _plugin(index=index)
    agent = _agent(plugin)

    result = _run(plugin.find_tools("list the accounts of an owner", _tool_context(agent)))

    assert result not in (_NO_MATCH_GUIDANCE, _EMPTY_NEED_GUIDANCE)
    assert result.startswith(_MATCHES_HEADER)
    assert "- list_accounts:" in result
    # Names and one line each: no schema travels in a tool result, on either branch.
    assert "inputSchema" not in result
    # The header is what tells the model where to go next, and that is the loading tool.
    assert GET_TOOL_DETAILS_NAME in result
    # Nothing was exposed: the model has to ask for the schema, and asking is the other tool's job.
    assert plugin._states[agent].exposed == {}
    assert plugin._states[agent].loads == 0


def test_without_the_plugin_the_tool_specs_are_the_registered_ones_field_for_field():
    """Requirement 9.7: not registering the plugin is not registering a rewrite.

    The comparison is against the registry rather than against a recorded baseline, because the
    registered specification *is* the without-feature behaviour. The projected specifications of an
    equivalent agent that does carry the plugin are asserted to differ, so the equality above is a
    result and not a tautology about an inert test.
    """
    plain = _agent()
    context = _model_call(plain)
    registered = [entry.tool_spec for entry in plain.tool_registry.registry.values()]

    # No handler on the input phase, so nothing stands between the registry and the provider.
    assert _input_handler_count(plain) == 0
    assert context.tool_specs == registered
    for projected, source in zip(context.tool_specs, registered, strict=True):
        assert projected is source
        assert projected.keys() == source.keys()
        assert all(projected[field] == source[field] for field in source)
    # And no catalog: the system prompt of a call without the plugin is the caller's own.
    assert context.system_prompt is None

    # The same arrangement with the plugin registers exactly one handler on that phase, which is the
    # one thing the plain agent above is missing.
    assert _input_handler_count(_agent(_plugin())) == 1


def test_the_projection_of_a_registered_plugin_differs_from_the_registered_specifications():
    """Requirement 9.7, other side: the without-plugin equality above is worth something.

    A projection that happened to equal the registry would make the previous test pass for the wrong
    reason, so the difference is asserted explicitly, on the same tools and the same call shape.
    """
    plugin = _plugin()
    agent = _agent(plugin)
    registered = copy.deepcopy([entry.tool_spec for entry in agent.tool_registry.registry.values()])

    result = _run(plugin._projection_handler(_model_call(agent)))
    projected = result.tool_specs

    assert projected != registered
    # With nothing exposed and nothing referenced, the projection is the two plugin tools and no more.
    assert [spec["name"] for spec in projected] == [FIND_TOOLS_NAME, GET_TOOL_DETAILS_NAME]
    # The tools that lost their place in ``tool_specs`` are named in the prompt instead, so the call
    # still knows they exist. What that block looks like is asserted in tests/test_catalog.py.
    assert isinstance(result.system_prompt, str)
    assert "- list_accounts:" in result.system_prompt
    assert "- send_wire:" in result.system_prompt
    # And the registry itself is untouched: what the projection emits is a selection, not an edit.
    assert [entry.tool_spec for entry in agent.tool_registry.registry.values()] == registered


def test_no_disclosure_state_exists_until_the_first_projection():
    """Requirement 10.4: constructing an agent with the plugin creates no state for it.

    State is what the plugin learned from a call, so before the first call there is nothing to hold —
    and after it, exactly one state, with no exposures and the fingerprint the build just wrote.
    """
    index = _RecordingIndex()
    plugin = _plugin(index=index)
    agent = _agent(plugin)

    assert plugin._states.get(agent) is None
    assert index.built == []

    _run(plugin._projection_handler(_model_call(agent)))

    state = plugin._states[agent]
    assert state.exposed == {}
    assert state.searches == 0
    assert state.loads == 0
    # Requirement 10.6: the fingerprint is what a restart loses, and it costs one build to rebuild. It
    # is keyed by ``(name, description)`` rather than by name, so re-registering a tool with a new
    # description buys the rebuild and the new catalog line that go with it.
    assert state.fingerprint == frozenset(
        (name, entry.tool_spec.get("description") or "") for name, entry in agent.tool_registry.registry.items()
    )
    assert len(index.built) == 1

    # A second projection over the same registry reuses the build and still holds one state.
    _run(plugin._projection_handler(_model_call(agent)))
    assert len(index.built) == 1
    assert len(plugin._states) == 1


def test_the_disclosure_state_is_dropped_along_with_the_agent():
    """Requirement 10.1: weak keys, so one plugin instance serving many agents keeps none alive."""
    plugin = _plugin(index=_RecordingIndex(ranked=("list_accounts",)))

    def one_session() -> weakref.ref[Agent]:
        """Run a full cycle for an agent that goes out of scope when this returns."""
        agent = _agent(plugin)
        _run(plugin._projection_handler(_model_call(agent)))
        _run(plugin.get_tool_details(names=["list_accounts"], tool_context=_tool_context(agent)))
        assert plugin._states[agent].exposed != {}
        return weakref.ref(agent)

    reference = one_session()
    gc.collect()

    assert reference() is None
    assert len(plugin._states) == 0


def test_a_fresh_state_reestablishes_the_exposure_with_one_load():
    """Requirement 10.6: losing the state costs one load per tool, never a broken projection.

    A process restart is simulated by dropping the state map entry, which is exactly what survives a
    restart: nothing. The following projection is valid, and one loading call puts the schema back.
    """
    index = _RecordingIndex(ranked=("list_accounts",))
    plugin = _plugin(index=index)
    agent = _agent(plugin)

    _run(plugin._projection_handler(_model_call(agent)))
    _run(plugin.get_tool_details(names=["list_accounts"], tool_context=_tool_context(agent)))
    exposed_before = dict(plugin._states[agent].exposed)
    assert exposed_before != {}

    del plugin._states[agent]

    restarted = _run(plugin._projection_handler(_model_call(agent))).tool_specs
    restarted_names = [spec["name"] for spec in restarted]
    # Requirement 9.5: a projection is never empty, and both plugin tools are in it, state or no state.
    assert _PLUGIN_TOOL_NAMES <= set(restarted_names)
    assert "list_accounts" not in restarted_names
    assert plugin._states[agent].exposed == {}

    _run(plugin.get_tool_details(names=["list_accounts"], tool_context=_tool_context(agent)))
    assert plugin._states[agent].exposed == exposed_before
    # Rebuilding the exposure took a load, not a search: the index was never consulted again.
    assert index.needs == []
