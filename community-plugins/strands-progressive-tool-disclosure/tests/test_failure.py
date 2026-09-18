"""Property tests for failure degradation: the agent is never left without tool specifications.

Feature: progressive-tool-disclosure-plugin, Property 22: Any projection-path failure degrades to the received context
with one warning.

Validates: Requirements 11.1, 11.2, 11.3, 11.4, 11.5, 11.6.

Feature: progressive-tool-disclosure-plugin, Property 23: A failing supplemental source degrades to history only and
proceeds.

Validates: Requirements 11.7.

The claim under test is narrow and precise: a failure anywhere on the projection path costs the projection and nothing
else. The received context comes back by object identity, the exception does not reach the stage, and the operator hears
about it exactly once — one ``warning`` carrying ``exc_info``, not one per step of a path that has five of them. The
single-log part is what these tests assert hardest, because a projection that logged five warnings would still return a
valid context and would still pass a test that only checked the return value.

Failures are injected at each of the five points the requirement enumerates, one per example:

- ``build`` raises, which additionally must leave the registry fingerprint unwritten so the next call rebuilds rather
  than searching over a half-built index;
- expiration raises, through a cycle counter that refuses to be read;
- the history scan raises, through a malformed message;
- catalog assembly raises, through an incoming specification with no ``description``;
- the block union raises, through a specification that stops answering to ``name`` part-way through the projection.

The search path degrades differently and is asserted separately: it returns guidance to the model instead of a context,
records zero exposures for that invocation, and logs its own single warning. The supplemental referenced source degrades
differently again — one ``debug``, not a warning, and the projection *proceeds*: a source that fails must not cost the
token reduction. That one is asserted against a baseline plugin configured with no source at all, so "degrades to
history only" is checked as an equality against the projection that never had a source, field for field.

Everything runs offline: a stub model that raises if called, ``ToolIndex`` doubles that raise on demand, and contexts
built by hand. No network, no model call, no disk.
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
from strands_progressive_tool_disclosure.plugin import FIND_TOOLS_NAME

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


class _FadingSpec(dict):
    """A specification that answers to ``name`` a fixed number of times, then raises.

    Two reads happen before the union: the passthrough guard tests each incoming name against the registry, and the
    fingerprint collects them. Letting exactly those two through puts the failure in the block union itself — the last
    of the five points requirement 11.5 enumerates, and the one with no malformed-data route into it.
    """

    def __init__(self, spec: dict[str, Any], reads_before_failing: int) -> None:
        """Wrap ``spec``, fading after ``reads_before_failing`` reads of ``name``."""
        super().__init__(spec)
        self._reads_left = reads_before_failing

    def __getitem__(self, key: str) -> Any:
        """Return the field, unless this is one read of ``name`` too many."""
        if key == "name":
            if self._reads_left <= 0:
                raise RuntimeError("the specification could not be read")
            self._reads_left -= 1
        return super().__getitem__(key)


FAILURE_POINTS = ["build", "expire", "history", "catalog", "union"]
"""The five points requirement 11.5 enumerates, plus ``build`` from 11.1: one injected per example."""

names_strategy = st.lists(st.sampled_from(TOOL_NAMES), min_size=1, max_size=len(TOOL_NAMES), unique=True)

catalog_tokens_strategy = st.one_of(st.none(), st.integers(min_value=1, max_value=40))

ttl_strategy = st.integers(min_value=1, max_value=6)

cycle_strategy = st.integers(min_value=0, max_value=12)

subset_strategy = st.lists(st.sampled_from(TOOL_NAMES), max_size=2, unique=True)

need_strategy = st.sampled_from(["list the accounts", "move money between accounts", "read the audit trail"])

failure_point_strategy = st.sampled_from(FAILURE_POINTS)

# Every way the supplemental source can fail: raising, returning something that cannot be iterated at all, and
# returning an iterable whose elements are not names. The plugin has to reach the same degradation from all of them.
source_failure_strategy = st.sampled_from(
    ["raises", "returns_int", "returns_none", "returns_object", "yields_nonstring"]
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

    Registering through ``plugins`` is what puts the search tool in the registry and the projection handler in the
    middleware registry, which is the state every claim here is made about.
    """
    return Agent(model=_StubModel(), tools=TOOLS, plugins=[plugin])


def _incoming_specs(agent: Agent, names: Sequence[str]) -> list[dict[str, Any]]:
    """The specifications a call arrives with: the search tool first, then ``names``, deep-copied.

    Copies rather than the registry's own objects, so an example that mutates or wraps an incoming specification cannot
    reach the registry the next example reads.
    """
    registry = agent.tool_registry.registry
    return [deepcopy(registry[name].tool_spec) for name in [FIND_TOOLS_NAME, *names]]


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


def _context(agent: Agent, tool_specs: list[dict[str, Any]], messages: list[dict[str, Any]]) -> InvokeModelContext:
    """Build an invocation context, restricted to the fields this SDK's context declares.

    The context gained fields across the supported SDK range; filtering by the declared names keeps a neighbouring
    field from failing these tests for a reason that has nothing to do with degradation.
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


@given(
    names=names_strategy,
    catalog_tokens=catalog_tokens_strategy,
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
    catalog_tokens: int | None,
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
    # Catalog assembly only runs over a tool no full-spec block claimed, and only when a budget is configured, so that
    # injection reserves the last incoming name for the catalog and gives the budget a value.
    catalog_target = names[-1]
    if failure_point == "catalog":
        catalog_tokens = 20 if catalog_tokens is None else catalog_tokens
        exposed = [name for name in exposed if name != catalog_target]
        always_available = [name for name in always_available if name != catalog_target]
        referenced = [name for name in referenced if name != catalog_target]

    fail_build = failure_point == "build"
    plugin = ProgressiveToolDisclosure(
        catalog_tokens=catalog_tokens,
        ttl_cycles=ttl_cycles,
        always_available=tuple(always_available),
        index=_RaisingIndex(fail_build=True) if fail_build else _SilentIndex(),
    )
    agent = _agent(plugin)
    _seed_state(plugin, agent, exposed, cycle)

    incoming = _incoming_specs(agent, names)
    messages = _messages(referenced)

    if failure_point == "expire":
        agent.event_loop_metrics = _UnreadableMetrics()
    elif failure_point == "history":
        # A message that is not a mapping: the scan reads ``content`` off every message it is handed.
        messages = ["the history was corrupted"]  # type: ignore[list-item]
    elif failure_point == "catalog":
        # The reserved specification loses its description, which is the one field a catalog entry budgets.
        stripped = dict(incoming[-1])
        stripped.pop("description", None)
        incoming[-1] = stripped
        assert incoming[-1]["name"] == catalog_target
    elif failure_point == "union":
        incoming = [_FadingSpec(spec, reads_before_failing=2) for spec in incoming]

    context = _context(agent, incoming, messages)

    with _captured_logs() as captured:
        result = asyncio.run(plugin._projection_handler(context))

    assert result is context, f"{failure_point} failure did not degrade to the received context"

    warnings = _warnings_with_traceback(captured.records)
    assert len(warnings) == 1, f"expected one warning with exc_info, got {len(warnings)}"
    assert _at_or_above(captured.records, logging.WARNING) == warnings, "a projection failure logged more than once"


@given(
    names=names_strategy,
    catalog_tokens=catalog_tokens_strategy,
    cycle=cycle_strategy,
    referenced=subset_strategy,
)
@PROPERTY_SETTINGS
def test_a_failed_build_leaves_the_fingerprint_unwritten_so_the_next_call_rebuilds(
    names: list[str],
    catalog_tokens: int | None,
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
    plugin = ProgressiveToolDisclosure(catalog_tokens=catalog_tokens, index=index)
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
    catalog_tokens=catalog_tokens_strategy,
    top_k=st.integers(min_value=1, max_value=4),
    cycle=cycle_strategy,
    exposed=subset_strategy,
)
@PROPERTY_SETTINGS
def test_a_failing_search_returns_guidance_with_exactly_one_warning_and_records_no_exposure(
    need: str,
    catalog_tokens: int | None,
    top_k: int,
    cycle: int,
    exposed: list[str],
) -> None:
    """Feature: progressive-tool-disclosure-plugin, Property 22.

    Validates: Requirements 11.3, 11.4.

    A ``search`` that raises degrades differently from the projection: the model gets guidance rather than an
    exception, the invocation records zero exposures — the exposures it started with are exactly the ones it ends with
    — and the operator gets one warning carrying ``exc_info``.
    """
    index = _RaisingIndex(fail_search=True)
    plugin = ProgressiveToolDisclosure(catalog_tokens=catalog_tokens, top_k=top_k, index=index)
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
    catalog_tokens=catalog_tokens_strategy,
    ttl_cycles=ttl_strategy,
    cycle=cycle_strategy,
    exposed=subset_strategy,
    always_available=subset_strategy,
    referenced=subset_strategy,
    failure=source_failure_strategy,
)
@PROPERTY_SETTINGS
def test_a_failing_supplemental_source_degrades_to_history_only_and_proceeds(
    names: list[str],
    catalog_tokens: int | None,
    ttl_cycles: int,
    cycle: int,
    exposed: list[str],
    always_available: list[str],
    referenced: list[str],
    failure: str,
) -> None:
    """Feature: progressive-tool-disclosure-plugin, Property 23.

    Validates: Requirements 11.7.

    A supplemental source that raises, that returns something non-iterable, or that yields a non-name degrades to the
    retained history alone — asserted as an equality against the projection of an identically configured plugin that
    has no source at all. The projection *proceeds*: the returned context is a new one carrying the projection, not
    the received context passed through. Exactly one debug record carries ``exc_info``, and nothing is logged at
    warning level: a source is supplemental, and losing it is not the operator's problem.
    """
    sources: dict[str, Any] = {
        "raises": _raising_source,
        "returns_int": lambda agent: 7,
        "returns_none": lambda agent: None,
        "returns_object": lambda agent: object(),
        "yields_nonstring": lambda agent: ["audit_log", 3],
    }

    configuration: dict[str, Any] = {
        "catalog_tokens": catalog_tokens,
        "ttl_cycles": ttl_cycles,
        "always_available": tuple(always_available),
    }

    plugin = ProgressiveToolDisclosure(index=_SilentIndex(), referenced_source=sources[failure], **configuration)
    agent = _agent(plugin)
    _seed_state(plugin, agent, exposed, cycle)

    # One set of incoming specifications for both calls, so the comparison below is about the source and nothing else.
    incoming = _incoming_specs(agent, names)
    messages = _messages(referenced)
    context = _context(agent, incoming, messages)

    with _captured_logs() as captured:
        result = asyncio.run(plugin._projection_handler(context))

    baseline_plugin = ProgressiveToolDisclosure(index=_SilentIndex(), referenced_source=None, **configuration)
    baseline_agent = _agent(baseline_plugin)
    _seed_state(baseline_plugin, baseline_agent, exposed, cycle)
    baseline_context = _context(baseline_agent, deepcopy(incoming), messages)
    baseline = asyncio.run(baseline_plugin._projection_handler(baseline_context))

    assert result is not context, "a failing source degraded to passthrough instead of proceeding"
    assert baseline is not baseline_context, "the baseline projection did not apply"
    assert result.tool_specs == baseline.tool_specs, "a failing source changed the projection"
    assert result.messages is messages, "the projection changed the retained history"

    debugs = _debugs_with_traceback(captured.records)
    assert len(debugs) == 1, f"expected one debug with exc_info, got {len(debugs)}"
    assert _at_or_above(captured.records, logging.WARNING) == [], "a failing source logged at warning level or above"


def _raising_source(agent: Agent) -> Sequence[str]:
    """A supplemental referenced source that fails outright."""
    raise RuntimeError("the supplemental source is unavailable")
