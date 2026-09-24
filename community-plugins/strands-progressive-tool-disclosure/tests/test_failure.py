"""Property tests for failure degradation: the agent is never left without tool specifications.

Feature: progressive-tool-disclosure-plugin, Property 22: Any projection-path failure degrades to the received context
with one warning.

Validates: Requirements 11.1, 11.2, 11.3, 11.4, 11.5, 11.6.

Feature: progressive-tool-disclosure-plugin, Property 24: A failing summarizer degrades to truncation and the
projection still succeeds.

Validates: Requirements 11.8.

The claim under test is narrow and precise: a failure anywhere on the projection path costs the projection and nothing
else. The received context comes back by object identity, the exception does not reach the stage, and the operator hears
about it exactly once — one ``warning`` carrying ``exc_info``, not one per step of a path that has five of them. The
single-log part is what these tests assert hardest, because a projection that logged five warnings would still return a
valid context and would still pass a test that only checked the return value.

Failures are injected at each of the five points the requirement enumerates, one per example:

- ``build`` raises, which additionally must leave the registry fingerprint unwritten so the next call rebuilds rather
  than searching over a half-built index;
- expiration raises, through a cycle counter that refuses to be read;
- the catalog's arrival in the system prompt raises, through a prompt of a shape nothing can append to;
- the block union raises, through an ``always_available`` container that refuses membership part-way through the
  projection.

Two paths degrade differently and are asserted separately. The search returns guidance to the model instead of a
context, records zero exposures for that invocation, and logs its own single warning. The summarizer degrades furthest
from a failure: a summary that cannot be produced falls back to boundary truncation, every tool still gets a catalog
line, and the projection applies as if nothing had happened.

Everything runs offline: a stub model that raises if called, a deterministic summarizer on every plugin that keeps a
catalog, ``ToolIndex`` doubles that raise on demand, and contexts built by hand. No network, no model call, no disk.
"""

import asyncio
import dataclasses
import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from copy import deepcopy
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from strands import Agent, tool
from strands.models.model import Model
from strands.types.tools import ToolContext

import strands_progressive_tool_disclosure.plugin as plugin_module
from strands_progressive_tool_disclosure import ProgressiveToolDisclosure
from strands_progressive_tool_disclosure._compat import InvokeModelContext
from strands_progressive_tool_disclosure.index import ToolMatch
from strands_progressive_tool_disclosure.plugin import FIND_TOOLS_NAME, GET_TOOL_DETAILS_NAME

PROPERTY_SETTINGS = settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


class _StubModel(Model):
    """A model that exists to be constructed. A call to it is a bug in the test, not a result."""

    def update_config(self, **kwargs: Any) -> None:
        """Accept anything; nothing reads it."""

    def get_config(self) -> dict[str, Any]:
        """No configuration to report."""
        return {}

    async def stream(self, *args: Any, **kwargs: Any) -> Any:
        """Refuse: these tests assert a degradation, never a model call."""
        raise AssertionError("the language model was called")
        yield

    async def structured_output(self, *args: Any, **kwargs: Any) -> Any:
        """Refuse: these tests assert a degradation, never a model call."""
        raise AssertionError("the language model was called")
        yield


@tool
def list_accounts(owner: str) -> str:
    """List the accounts of an owner."""
    return "accounts"


@tool
def wire_transfer(account: str, amount: str) -> str:
    """Send a wire transfer from an account."""
    return "sent"


@tool
def audit_log(account: str) -> str:
    """Read the audit log of an account."""
    return "log"


@tool
def send_email(recipient: str) -> str:
    """Send an email to a recipient."""
    return "sent"


TOOLS = [list_accounts, wire_transfer, audit_log, send_email]

TOOL_NAMES = [registered.tool_name for registered in TOOLS]


def _stub_summarizer(spec: dict[str, Any], max_chars: int) -> str:
    """Summarize deterministically and offline: the tool's name, as its own catalog line.

    Every plugin here that keeps a catalog gets this one. The default summarizer would reach for the agent's model,
    and the stub model raises — which would put a warning of its own in the records these tests count.
    """
    return f"{spec['name']} summary"


class _SilentIndex:
    """A ``ToolIndex`` double that indexes nothing and finds nothing, and never raises.

    The projection paths that are not about the index need one that stays out of the way: whatever the example failed
    at, the single warning it produced has to be the injected failure's and not this double's.
    """

    def build(self, specs: Sequence[dict[str, Any]]) -> None:
        """Accept the specifications and keep nothing."""

    def search(self, need: str, top_k: int) -> Sequence[ToolMatch]:
        """Find nothing, deterministically."""
        return []


class _RaisingIndex:
    """A ``ToolIndex`` double that raises on demand, counting the attempts.

    The count is the point for ``build``: requirement 11.2 is about the *next* call, so what has to be observed is a
    second attempt, not merely a first failure.
    """

    def __init__(self, *, fail_build: bool = False, fail_search: bool = False) -> None:
        """Arm the failures this double is asked for."""
        self._fail_build = fail_build
        self._fail_search = fail_search
        self.builds = 0
        self.searches = 0

    def build(self, specs: Sequence[dict[str, Any]]) -> None:
        """Count the attempt, then raise when armed to."""
        self.builds += 1
        if self._fail_build:
            raise RuntimeError("the index could not be built")

    def search(self, need: str, top_k: int) -> Sequence[ToolMatch]:
        """Count the attempt, then raise when armed to."""
        self.searches += 1
        if self._fail_search:
            raise RuntimeError("the index could not be searched")
        return []


class _UnreadableMetrics:
    """Event loop metrics whose cycle counter refuses to be read.

    This is how expiration is made to fail: the cycle counter is the first thing it reads, and it is read before any
    projection work has been done, so the failure lands squarely inside the expiration step.
    """

    @property
    def cycle_count(self) -> int:
        """Refuse to report a cycle."""
        raise RuntimeError("the cycle counter could not be read")


class _UnaskableNames(tuple):
    """An ``always_available`` container that refuses to answer a membership question.

    This is how the block union is made to fail. ``always_available`` is read in exactly one place on the projection
    path — the second of the four blocks ``_compose_projection`` visits — so the failure lands inside the union itself,
    after the plugin tools' block has already been composed, and nowhere earlier.
    """

    def __contains__(self, item: object) -> bool:
        """Refuse the question."""
        raise RuntimeError("the always-available names could not be read")


FAILURE_POINTS = ["build", "expire", "catalog", "union"]
"""The projection-path points that can fail, plus ``build`` from 11.1: one injected per example."""

names_strategy = st.lists(st.sampled_from(TOOL_NAMES), min_size=1, max_size=len(TOOL_NAMES), unique=True)

catalog_chars_strategy = st.one_of(st.none(), st.integers(min_value=1, max_value=40))

ttl_strategy = st.integers(min_value=1, max_value=6)

cycle_strategy = st.integers(min_value=0, max_value=12)

subset_strategy = st.lists(st.sampled_from(TOOL_NAMES), max_size=2, unique=True)

need_strategy = st.sampled_from(["list the accounts", "move money between accounts", "read the audit trail"])

failure_point_strategy = st.sampled_from(FAILURE_POINTS)

# Every way a summarizer can fail to produce a line: raising outright, raising from the awaited branch, and answering
# with something that is not a usable summary. All of them have to reach the same truncation.
summarizer_failure_strategy = st.sampled_from(
    ["raises", "raises_async", "returns_none", "returns_empty", "returns_whitespace", "returns_int", "returns_object"]
)


class _Records(logging.Handler):
    """A handler that keeps the records, so "exactly one warning" can be counted rather than assumed."""

    def __init__(self) -> None:
        """Start with no records."""
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        """Keep the record."""
        self.records.append(record)


@contextmanager
def _captured_logs() -> Iterator[_Records]:
    """Capture the plugin logger's records for the duration of one example.

    Attached per example and removed afterwards: the counts these tests make are per call, and a handler that outlived
    its example would make the second example's "exactly one" read as two.
    """
    handler = _Records()
    logger = plugin_module.logger
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


def _warnings_with_traceback(records: Sequence[logging.LogRecord]) -> list[logging.LogRecord]:
    """The warning-level records carrying ``exc_info`` — what the requirement asks the operator to receive."""
    return [record for record in records if record.levelno == logging.WARNING and record.exc_info is not None]


def _debugs_with_traceback(records: Sequence[logging.LogRecord]) -> list[logging.LogRecord]:
    """The debug-level records carrying ``exc_info``."""
    return [record for record in records if record.levelno == logging.DEBUG and record.exc_info is not None]


def _at_or_above(records: Sequence[logging.LogRecord], level: int) -> list[logging.LogRecord]:
    """Every record at ``level`` or above, whether it carries ``exc_info`` or not."""
    return [record for record in records if record.levelno >= level]


def _agent(plugin: ProgressiveToolDisclosure) -> Agent:
    """Build an offline agent carrying the tool pool and ``plugin``.

    Registering through ``plugins`` is what puts both plugin tools in the registry and the projection handler in the
    middleware registry, which is the state every claim here is made about.
    """
    return Agent(model=_StubModel(), tools=TOOLS, plugins=[plugin])


def _incoming_specs(agent: Agent, names: Sequence[str]) -> list[dict[str, Any]]:
    """The specifications a call arrives with: both plugin tools first, then ``names``, deep-copied.

    Both are present because the projection only applies when both are: without ``get_tool_details`` the model has no
    way to load a hidden schema, so a call missing either one is passed through and no degradation could be observed.

    Copies rather than the registry's own objects, so an example that mutates or wraps an incoming specification cannot
    reach the registry the next example reads.
    """
    registry = agent.tool_registry.registry
    return [deepcopy(registry[name].tool_spec) for name in [FIND_TOOLS_NAME, GET_TOOL_DETAILS_NAME, *names]]


def _messages(referenced: Sequence[str]) -> list[dict[str, Any]]:
    """A retained history that references ``referenced`` through ``toolUse`` blocks."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": [{"text": "what did the migration cost"}]}]
    for position, name in enumerate(referenced):
        messages.append(
            {
                "role": "assistant",
                "content": [{"toolUse": {"toolUseId": f"use-{position}", "name": name, "input": {}}}],
            }
        )
    return messages


def _context(
    agent: Agent,
    tool_specs: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    system_prompt: Any = "You are a helpful assistant.",
) -> InvokeModelContext:
    """Build an invocation context, restricted to the fields this SDK's context declares.

    The context gained fields across the supported SDK range; filtering by the declared names keeps a neighbouring
    field from failing these tests for a reason that has nothing to do with degradation.
    """
    candidates: dict[str, Any] = {
        "agent": agent,
        "messages": messages,
        "system_prompt": system_prompt,
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
) -> plugin_module._DisclosureState:
    """Give ``agent`` a disclosure state with live exposures at ``cycle``, and set the cycle counter."""
    agent.event_loop_metrics.cycle_count = cycle
    state = plugin_module._state_for(plugin._states, agent)
    for name in exposed:
        state.exposed[name] = cycle
    return state


def _tool_context(agent: Agent) -> ToolContext:
    """The context the framework would hand the search tool, built by hand for a direct call."""
    return ToolContext(
        tool_use={"toolUseId": "use-search", "name": FIND_TOOLS_NAME, "input": {}},
        agent=agent,
        invocation_state={},
    )


def _registered_description(agent: Agent, name: str) -> str:
    """The description ``name`` is registered with, whitespace collapsed as a catalog line collapses it."""
    return " ".join((agent.tool_registry.registry[name].tool_spec.get("description") or "").split())


def _catalog_lines(system_prompt: Any) -> list[str]:
    """The ``- name: summary`` lines of the catalog block appended to ``system_prompt``."""
    text = system_prompt if isinstance(system_prompt, str) else "\n".join(block["text"] for block in system_prompt)
    return [line for line in text.splitlines() if line.startswith("- ")]


@given(
    names=names_strategy,
    catalog_chars=catalog_chars_strategy,
    ttl_cycles=ttl_strategy,
    cycle=cycle_strategy,
    exposed=subset_strategy,
    always_available=subset_strategy,
    referenced=subset_strategy,
    failure_point=failure_point_strategy,
)
@PROPERTY_SETTINGS
def test_any_projection_path_failure_degrades_to_the_received_context_with_exactly_one_warning(
    names: list[str],
    catalog_chars: int | None,
    ttl_cycles: int,
    cycle: int,
    exposed: list[str],
    always_available: list[str],
    referenced: list[str],
    failure_point: str,
) -> None:
    """Feature: progressive-tool-disclosure-plugin, Property 22.

    Validates: Requirements 11.1, 11.5, 11.6.

    For a failure injected at any of the five points on the projection path, the handler returns the received context
    by object identity, emits exactly one warning carrying ``exc_info``, logs nothing more severe, and propagates
    nothing to the stage — reaching this assertion at all is what proves the last part.
    """
    # The catalog only reaches the system prompt when a limit is configured and at least one incoming tool is left for
    # it, so that injection reserves the last incoming name for the catalog and gives the limit a value.
    catalog_target = names[-1]
    if failure_point == "catalog":
        catalog_chars = 20 if catalog_chars is None else catalog_chars
        exposed = [name for name in exposed if name != catalog_target]
        always_available = [name for name in always_available if name != catalog_target]
        referenced = [name for name in referenced if name != catalog_target]

    fail_build = failure_point == "build"
    plugin = ProgressiveToolDisclosure(
        catalog_chars=catalog_chars,
        summarizer=_stub_summarizer,
        ttl_cycles=ttl_cycles,
        always_available=tuple(always_available),
        index=_RaisingIndex(fail_build=True) if fail_build else _SilentIndex(),
    )
    agent = _agent(plugin)
    _seed_state(plugin, agent, exposed, cycle)

    incoming = _incoming_specs(agent, names)
    messages = _messages(referenced)
    system_prompt: Any = "You are a helpful assistant."

    if failure_point == "expire":
        agent.event_loop_metrics = _UnreadableMetrics()
    elif failure_point == "catalog":
        # A prompt that is neither text nor a list of blocks: the catalog has nowhere to be appended.
        system_prompt = 7
    elif failure_point == "union":
        # Set after construction on purpose: the validation the constructor runs is not what is under test here.
        plugin._always_available = _UnaskableNames(always_available)

    context = _context(agent, incoming, messages, system_prompt)

    with _captured_logs() as captured:
        result = asyncio.run(plugin._projection_handler(context))

    assert result is context, f"{failure_point} failure did not degrade to the received context"

    warnings = _warnings_with_traceback(captured.records)
    assert len(warnings) == 1, f"expected one warning with exc_info, got {len(warnings)}"
    assert _at_or_above(captured.records, logging.WARNING) == warnings, "a projection failure logged more than once"


@given(
    names=names_strategy,
    catalog_chars=catalog_chars_strategy,
    cycle=cycle_strategy,
    referenced=subset_strategy,
)
@PROPERTY_SETTINGS
def test_a_failed_build_leaves_the_fingerprint_unwritten_so_the_next_call_rebuilds(
    names: list[str],
    catalog_chars: int | None,
    cycle: int,
    referenced: list[str],
) -> None:
    """Feature: progressive-tool-disclosure-plugin, Property 22.

    Validates: Requirements 11.1, 11.2, 11.6.

    A ``build`` that raises must not record the registry fingerprint: the next projection has to attempt the build
    again rather than search over a half-built index. Two identical calls therefore produce two build attempts, two
    passthroughs, and one warning each.
    """
    index = _RaisingIndex(fail_build=True)
    plugin = ProgressiveToolDisclosure(catalog_chars=catalog_chars, summarizer=_stub_summarizer, index=index)
    agent = _agent(plugin)
    state = _seed_state(plugin, agent, (), cycle)

    context = _context(agent, _incoming_specs(agent, names), _messages(referenced))

    for attempt in (1, 2):
        with _captured_logs() as captured:
            result = asyncio.run(plugin._projection_handler(context))

        assert result is context
        assert index.builds == attempt, "a failed build was not attempted again on the next call"
        assert state.fingerprint is None, "a failed build recorded the registry fingerprint"
        assert len(_warnings_with_traceback(captured.records)) == 1
        assert len(_at_or_above(captured.records, logging.WARNING)) == 1


@given(
    need=need_strategy,
    catalog_chars=catalog_chars_strategy,
    top_k=st.integers(min_value=1, max_value=4),
    cycle=cycle_strategy,
    exposed=subset_strategy,
)
@PROPERTY_SETTINGS
def test_a_failing_search_returns_guidance_with_exactly_one_warning_and_records_no_exposure(
    need: str,
    catalog_chars: int | None,
    top_k: int,
    cycle: int,
    exposed: list[str],
) -> None:
    """Feature: progressive-tool-disclosure-plugin, Property 22.

    Validates: Requirements 11.3, 11.4.

    A ``search`` that raises degrades differently from the projection: the model gets guidance rather than an
    exception, the invocation records zero exposures — the exposures it started with are exactly the ones it ends with,
    which a search has no business changing in any case — and the operator gets one warning carrying ``exc_info``.
    """
    index = _RaisingIndex(fail_search=True)
    plugin = ProgressiveToolDisclosure(
        catalog_chars=catalog_chars, summarizer=_stub_summarizer, top_k=top_k, index=index
    )
    agent = _agent(plugin)
    state = _seed_state(plugin, agent, exposed, cycle)
    before = dict(state.exposed)

    with _captured_logs() as captured:
        result = asyncio.run(plugin.find_tools(need=need, tool_context=_tool_context(agent)))

    assert result == plugin_module._SEARCH_FAILED_GUIDANCE
    assert index.searches == 1, "the search was not attempted exactly once"
    assert state.exposed == before, "a failing search recorded an exposure"

    warnings = _warnings_with_traceback(captured.records)
    assert len(warnings) == 1, f"expected one warning with exc_info, got {len(warnings)}"
    assert _at_or_above(captured.records, logging.WARNING) == warnings, "a failing search logged more than once"


@given(
    names=names_strategy,
    catalog_chars=st.integers(min_value=1, max_value=24),
    ttl_cycles=ttl_strategy,
    cycle=cycle_strategy,
    failure=summarizer_failure_strategy,
)
@PROPERTY_SETTINGS
def test_a_failing_summarizer_falls_back_to_truncation_and_the_projection_still_succeeds(
    names: list[str],
    catalog_chars: int,
    ttl_cycles: int,
    cycle: int,
    failure: str,
) -> None:
    """Feature: progressive-tool-disclosure-plugin, Property 24.

    Validates: Requirements 11.8.

    A summarizer that raises — from the synchronous call or from the awaited branch — or that answers with anything
    that is not a usable line costs nothing but the summary. Every catalog-eligible tool still gets a line, that line
    is the boundary truncation of its own description and fits the limit, the projection applies, and nothing reaches
    the caller as an error. The limit is capped low enough that the descriptions here do not fit it, so the summarizer
    is genuinely reached rather than skipped by a description that was already short enough.
    """
    plugin = ProgressiveToolDisclosure(
        catalog_chars=catalog_chars,
        summarizer=_failing_summarizer(failure),
        ttl_cycles=ttl_cycles,
        index=_SilentIndex(),
    )
    agent = _agent(plugin)
    state = _seed_state(plugin, agent, (), cycle)

    incoming = _incoming_specs(agent, names)
    context = _context(agent, incoming, _messages(()))

    with _captured_logs() as captured:
        result = asyncio.run(plugin._projection_handler(context))

    assert result is not context, "a failing summarizer cost the projection"
    assert [spec["name"] for spec in result.tool_specs] == [FIND_TOOLS_NAME, GET_TOOL_DETAILS_NAME], (
        "the projection carried something other than the two plugin tools"
    )

    lines = _catalog_lines(result.system_prompt)
    assert len(lines) == len(names), "a tool was left out of the catalog"
    for name in names:
        description = _registered_description(agent, name)
        expected = plugin_module._truncate_description(description, catalog_chars)
        assert f"- {name}: {expected}" in lines, f"{name} was not catalogued by truncation"
        assert len(expected) <= catalog_chars, "the truncation fallback overran the limit"

    # A custom summarizer is supplied, so the default one is never built and there is no auxiliary cost to report.
    assert state.summary_usage == {}, "a custom summarizer reported model usage"

    # The failure is the summarizer's, not the projection's: one warning per tool it failed on, and nothing worse.
    failed = len(names) if failure in ("raises", "raises_async") else 0
    assert len(_warnings_with_traceback(captured.records)) == failed, "the summary failures were not logged once each"
    assert _at_or_above(captured.records, logging.ERROR) == [], "a summary failure was logged as an error"


def _failing_summarizer(failure: str) -> Any:
    """Build a summarizer that fails in the named way.

    ``raises_async`` fails from inside the awaitable rather than from the call, which is the branch a synchronous
    summarizer cannot reach: the failure has to be caught around the ``await``, not merely around the call.
    """

    async def raises_async(spec: dict[str, Any], max_chars: int) -> str:
        raise RuntimeError("the summarizer is unavailable")

    def raises(spec: dict[str, Any], max_chars: int) -> str:
        raise RuntimeError("the summarizer is unavailable")

    junk: dict[str, Any] = {
        "returns_none": None,
        "returns_empty": "",
        "returns_whitespace": "   \n\t ",
        "returns_int": 7,
        "returns_object": object(),
    }
    if failure == "raises":
        return raises
    if failure == "raises_async":
        return raises_async
    return lambda spec, max_chars: junk[failure]
