"""Tests for the loading tool, the catalog summaries, and the default model summarizer.

Covers the surface the redesign introduced and nothing else: ``get_tool_details`` as the one way a
schema reaches the next projection, the summary pipeline that writes a catalog line once per tool, the
default summarizer built over the agent's own model, and the pre-call guard's exemption for the loading
tool itself.

Three claims run through the file. A loading call is idempotent in its effect and forgiving in its
input -- a bare string, a duplicate and a blank name are all tolerated, an unknown name is reported
rather than raised on. A summary is produced at most once per ``(name, description)`` pair and is
clamped to the limit whatever the summarizer answers, so the catalog is byte-stable and bounded. And
the default summarizer is one plain model call: no tools, no history, its usage accounted.

Everything runs offline. Two models appear: a stub that fails the test if it is called at all, and a
scripted one that yields the stream events a summary call reads, so "the model was called once, with no
tool specifications" is asserted against a recording rather than assumed.
"""

import asyncio
import dataclasses
from collections.abc import AsyncGenerator, Coroutine, Sequence
from types import SimpleNamespace
from typing import Any, TypeVar, cast

import pytest
from strands import Agent
from strands.agent.agent import Agent as AgentType
from strands.hooks.events import BeforeToolCallEvent
from strands.models.model import Model
from strands.tools.decorator import tool
from strands.types.tools import ToolContext, ToolSpec

from strands_progressive_tool_disclosure import ProgressiveToolDisclosure, ToolMatch
from strands_progressive_tool_disclosure._compat import InvokeModelContext
from strands_progressive_tool_disclosure.plugin import (
    _CATALOG_PROMPT_HEADER,
    _DETAILS_EMPTY_GUIDANCE,
    _DETAILS_LOADED_HEADER,
    _DETAILS_UNKNOWN,
    _SUMMARY_SYSTEM_PROMPT,
    FIND_TOOLS_NAME,
    GET_TOOL_DETAILS_NAME,
    _model_summarizer,
    _truncate_description,
)

_Resolved = TypeVar("_Resolved")

_CATALOG_CHARS = 40
"""Limit used throughout: small enough that one registered description overruns it and another does not."""


# ---------------------------------------------------------------------------------------------------
# Tools. One description fits the limit, one does not -- the two branches of the summary pipeline.
# ---------------------------------------------------------------------------------------------------


@tool
def check_balance(account: str) -> str:
    """Report the balance
    of one account.

    Args:
        account: Account to read.
    """
    return account


@tool
def send_wire(account: str, amount: str) -> str:
    """Send a wire transfer from an investment account to an external beneficiary, settling on the next
    business day and returning the settlement identifier together with the fee that was charged.

    Args:
        account: Account to debit.
        amount: How much to send.
    """
    return f"{account}:{amount}"


@tool
def list_positions(account: str) -> str:
    """List every open position of an investment account, with its quantity, its average entry price and
    the unrealized result as of the last close.

    Args:
        account: Account to read.
    """
    return account


_FITTING_DESCRIPTION = "Report the balance of one account."
"""``check_balance``'s description with its whitespace collapsed: what a verbatim line looks like."""


# ---------------------------------------------------------------------------------------------------
# Models and doubles
# ---------------------------------------------------------------------------------------------------


class _StubModel(Model):
    """A model that exists only so an ``Agent`` can be constructed. A call to it is a test failure."""

    def update_config(self, **kwargs: Any) -> None:
        """Accept anything; nothing reads it."""

    def get_config(self) -> dict[str, Any]:
        """No configuration to report."""
        return {}

    async def stream(self, *args: Any, **kwargs: Any) -> AsyncGenerator[dict[str, Any], None]:
        """Refuse: a test using this model is not a test about a model call."""
        raise AssertionError("the language model was called")
        yield {}

    async def structured_output(self, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        """Refuse: nothing here asks for structured output."""
        raise AssertionError("the language model was called")
        yield {}


class _ScriptedModel(Model):
    """A model that replays one scripted summary stream and records how it was called.

    The recording is what makes the shape of a summary call assertable: which messages it carried, what
    it passed for ``tool_specs``, and which system prompt it was given.
    """

    def __init__(self, chunks: Sequence[str] = ("a short ", "summary"), usage: dict[str, int] | None = None) -> None:
        """Reply with ``chunks`` joined, reporting ``usage`` in the metadata event.

        Args:
            chunks: Text deltas, emitted one event each, as a provider streams them.
            usage: Usage to report, or ``None`` for the default eleven in and five out.
        """
        self._chunks = tuple(chunks)
        self._usage = {"inputTokens": 11, "outputTokens": 5} if usage is None else dict(usage)
        self.calls: list[dict[str, Any]] = []

    def update_config(self, **kwargs: Any) -> None:
        """Accept anything; nothing reads it."""

    def get_config(self) -> dict[str, Any]:
        """No configuration to report."""
        return {}

    async def stream(
        self,
        messages: Any,
        tool_specs: Any = None,
        system_prompt: Any = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Record the call, then yield the text deltas and one metadata event."""
        self.calls.append({"messages": messages, "tool_specs": tool_specs, "system_prompt": system_prompt})
        for chunk in self._chunks:
            yield {"contentBlockDelta": {"delta": {"text": chunk}}}
        # Events a summary call ignores travel in the same stream, so one of each is emitted.
        yield {"contentBlockDelta": {"delta": {"toolUse": {"input": "{}"}}}}
        yield {"metadata": {"usage": {**self._usage, "totalTokens": sum(self._usage.values())}}}

    async def structured_output(self, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        """Refuse: the summarizer never asks for structured output."""
        raise AssertionError("structured output was requested")
        yield {}


class _RecordingSummarizer:
    """Summarizer double: answers a fixed line, or raises, and records every spec it was handed."""

    def __init__(self, answer: object = "a recorded summary", raises: bool = False) -> None:
        """Answer ``answer``, or raise when ``raises``.

        Args:
            answer: What to return, including the non-string and empty answers the clamp rejects.
            raises: Whether to fail instead of answering, which is the fallback-to-truncation path.
        """
        self._answer = answer
        self._raises = raises
        self.seen: list[tuple[str, int]] = []

    def __call__(self, spec: ToolSpec, max_chars: int) -> object:
        """Record ``(name, max_chars)`` and answer."""
        self.seen.append((spec["name"], max_chars))
        if self._raises:
            raise RuntimeError("no summary today")
        return self._answer

    @property
    def names(self) -> list[str]:
        """Names this summarizer was asked about, in call order."""
        return [name for name, _ in self.seen]


class _AsyncSummarizer(_RecordingSummarizer):
    """The same double, async: the summarizer contract accepts either shape."""

    async def __call__(self, spec: ToolSpec, max_chars: int) -> object:  # type: ignore[override]
        """Record the call after one suspension point, then answer."""
        await asyncio.sleep(0)
        return super().__call__(spec, max_chars)


class _SilentIndex:
    """Index double that indexes nothing and ranks nothing: these tests never search."""

    def __init__(self) -> None:
        """Start with no recorded build."""
        self.built: list[list[str]] = []

    def build(self, specs: list[dict[str, Any]]) -> None:
        """Record the names offered for indexing."""
        self.built.append([spec["name"] for spec in specs])

    def search(self, need: str, top_k: int) -> list[ToolMatch]:
        """Rank nothing."""
        return []


# ---------------------------------------------------------------------------------------------------
# Arrangement helpers
# ---------------------------------------------------------------------------------------------------


def _plugin(**kwargs: Any) -> ProgressiveToolDisclosure:
    """Build the plugin with the small catalog limit and the silent index unless overridden."""
    return ProgressiveToolDisclosure(**{"catalog_chars": _CATALOG_CHARS, "index": _SilentIndex(), **kwargs})


def _agent(plugin: ProgressiveToolDisclosure, *, model: Model | None = None, tools: list[Any] | None = None) -> Agent:
    """Build an offline agent carrying the plugin and, by default, the three registered tools.

    Args:
        plugin: Plugin to register.
        model: Model of the agent, defaulting to the stub that refuses to be called.
        tools: Tools to register, defaulting to all three.

    Returns:
        An agent whose registry is real, so the specifications the assertions read are the registered
        ones.
    """
    return Agent(
        model=_StubModel() if model is None else model,
        tools=[check_balance, send_wire, list_positions] if tools is None else tools,
        plugins=[plugin],
    )


def _model_call(agent: Agent, system_prompt: str | None = None) -> InvokeModelContext:
    """Build the invocation context of one model call, offering every registered specification."""
    candidates: dict[str, Any] = {
        "agent": agent,
        "messages": [],
        "system_prompt": system_prompt,
        "tool_specs": [entry.tool_spec for entry in agent.tool_registry.registry.values()],
        "tool_choice": None,
        "invocation_state": {},
        "model": agent.model,
    }
    declared = {field.name for field in dataclasses.fields(InvokeModelContext)}

    return InvokeModelContext(**{name: value for name, value in candidates.items() if name in declared})


def _before_tool_call(agent: Agent, name: str, tool_input: dict[str, Any] | None = None) -> BeforeToolCallEvent:
    """Build a real pre-call event, so the event's own write guards are exercised."""
    return BeforeToolCallEvent(
        agent=cast("AgentType", agent),
        selected_tool=None,
        tool_use={"toolUseId": "t1", "name": name, "input": tool_input if tool_input is not None else {}},
        invocation_state={},
    )


def _tool_context(agent: Agent) -> ToolContext:
    """Build the injected context a tool invocation receives."""
    return cast("ToolContext", SimpleNamespace(agent=agent))


def _run(step: Coroutine[Any, Any, _Resolved]) -> _Resolved:
    """Drive one awaited plugin step to completion from a synchronous test."""
    return asyncio.run(step)


def _load(plugin: ProgressiveToolDisclosure, agent: Agent, names: Any) -> str:
    """Invoke the loading tool once and return what the model would read."""
    return _run(plugin.get_tool_details(names, _tool_context(agent)))


def _summary_of(plugin: ProgressiveToolDisclosure, agent: Agent, name: str) -> str:
    """Return the cached catalog line of ``name``, as the projection wrote it."""
    spec = agent.tool_registry.registry[name].tool_spec
    return plugin._summaries[(name, spec.get("description") or "")]


def _description_of(agent: Agent, name: str) -> str:
    """Return the registered description of ``name``, verbatim."""
    return agent.tool_registry.registry[name].tool_spec.get("description") or ""


# ---------------------------------------------------------------------------------------------------
# A -- get_tool_details: what a loading call exposes, tolerates, reports and counts
# ---------------------------------------------------------------------------------------------------


def test_a_loading_call_exposes_every_requested_name_at_the_current_cycle():
    """The one way a schema reaches the next projection: named, it is exposed as of this cycle."""
    plugin = _plugin()
    agent = _agent(plugin)
    agent.event_loop_metrics.cycle_count = 4

    result = _load(plugin, agent, ["send_wire", "list_positions"])

    assert plugin._states[agent].exposed == {"send_wire": 4, "list_positions": 4}
    # The specification travels in tool_specs, not in this text: the result names what was loaded.
    assert result.startswith(_DETAILS_LOADED_HEADER)
    assert result.count("\n- ") == 2
    assert "inputSchema" not in result


def test_a_loading_call_renews_an_existing_exposure_at_the_later_cycle():
    """Renewal and first exposure are the same write, so a reload resets the idle count to zero."""
    plugin = _plugin()
    agent = _agent(plugin)
    agent.event_loop_metrics.cycle_count = 1
    _load(plugin, agent, ["send_wire"])

    agent.event_loop_metrics.cycle_count = 6
    _load(plugin, agent, ["send_wire"])

    assert plugin._states[agent].exposed == {"send_wire": 6}


def test_a_bare_string_is_tolerated_as_one_name():
    """A model loading a single tool may not wrap it in a list, and that call still has to work."""
    plugin = _plugin()
    agent = _agent(plugin)

    result = _load(plugin, agent, "send_wire")

    assert "send_wire" in plugin._states[agent].exposed
    assert result.startswith(_DETAILS_LOADED_HEADER)
    assert result.count("\n- ") == 1


def test_duplicates_and_blank_names_are_dropped_so_each_tool_is_listed_once():
    """Whitespace and repetition are noise in a generated argument list, not a request for two loads."""
    plugin = _plugin()
    agent = _agent(plugin)

    result = _load(plugin, agent, ["send_wire", " send_wire ", "", "   ", "send_wire"])

    assert plugin._states[agent].exposed == {"send_wire": agent.event_loop_metrics.cycle_count}
    assert result.count("\n- ") == 1


def test_an_unknown_name_is_reported_and_never_exposed():
    """A name that is not a tool has no schema to load, so it is named back rather than swallowed."""
    plugin = _plugin()
    agent = _agent(plugin)

    result = _load(plugin, agent, ["send_wire", "wire_money", "list_transactions"])

    assert plugin._states[agent].exposed == {"send_wire": agent.event_loop_metrics.cycle_count}
    assert result.startswith(_DETAILS_LOADED_HEADER)
    assert result.endswith(_DETAILS_UNKNOWN.format(names="wire_money, list_transactions"))


def test_a_call_naming_only_unknown_tools_reports_them_without_a_loaded_header():
    """A header with nothing under it would read as a broken tool, so nothing was loaded and it says so."""
    plugin = _plugin()
    agent = _agent(plugin)

    result = _load(plugin, agent, ["wire_money"])

    assert result == _DETAILS_UNKNOWN.format(names="wire_money")
    assert _DETAILS_LOADED_HEADER not in result
    assert plugin._states[agent].exposed == {}


@pytest.mark.parametrize("names", [[], None, ["", "  ", "\t"], [7, None], "  "])
def test_a_call_naming_nothing_usable_returns_the_guidance(names: Any):
    """Guidance rather than an error: the model can recover inside its own turn."""
    plugin = _plugin()
    agent = _agent(plugin)

    result = _load(plugin, agent, names)

    assert result == _DETAILS_EMPTY_GUIDANCE
    assert plugin._states[agent].exposed == {}


def test_every_loading_call_counts_one_cycle_spent_loading():
    """``loads`` measures cycles the session spent loading, so the empty and unknown calls count too."""
    plugin = _plugin()
    agent = _agent(plugin)

    _load(plugin, agent, ["send_wire"])
    _load(plugin, agent, [])
    _load(plugin, agent, ["wire_money"])

    assert plugin._states[agent].loads == 3
    # Loading is not searching: the two counters are separate readings of the same session.
    assert plugin._states[agent].searches == 0


def test_the_loaded_lines_carry_the_catalog_summary_of_each_tool():
    """The line the model reads on loading is the line it read in the catalog -- one summary per tool."""
    summarizer = _RecordingSummarizer(answer="Wires money out of an account.")
    plugin = _plugin(summarizer=summarizer)
    agent = _agent(plugin)
    _run(plugin._projection_handler(_model_call(agent)))

    result = _load(plugin, agent, ["send_wire", "check_balance"])

    assert f"- send_wire: {_summary_of(plugin, agent, 'send_wire')}" in result
    assert "- send_wire: Wires money out of an account." in result
    # check_balance fit the limit, so its line is its own description, collapsed.
    assert f"- check_balance: {_FITTING_DESCRIPTION}" in result


def test_a_loading_call_before_any_projection_falls_back_to_a_truncated_description():
    """No summary cached yet is not a reason to answer with nothing, nor with a full description."""
    plugin = _plugin()
    agent = _agent(plugin)

    result = _load(plugin, agent, ["send_wire"])

    expected = _truncate_description(" ".join(_description_of(agent, "send_wire").split()), _CATALOG_CHARS)
    assert f"- send_wire: {expected}" in result
    assert len(expected) <= _CATALOG_CHARS


# ---------------------------------------------------------------------------------------------------
# B -- summaries: written once per description, clamped, and skipped where they are not wanted
# ---------------------------------------------------------------------------------------------------


def test_a_description_that_fits_the_limit_is_used_verbatim_and_costs_no_call():
    """A description that already fits is the best possible summary of itself."""
    summarizer = _RecordingSummarizer()
    plugin = _plugin(summarizer=summarizer)
    agent = _agent(plugin, tools=[check_balance])

    _run(plugin._projection_handler(_model_call(agent)))

    assert summarizer.seen == []
    assert _summary_of(plugin, agent, "check_balance") == _FITTING_DESCRIPTION


def test_a_long_description_is_summarized_once_and_the_line_is_cached():
    """The catalog is byte-stable across calls because the summary behind it is written once."""
    summarizer = _RecordingSummarizer()
    plugin = _plugin(summarizer=summarizer)
    agent = _agent(plugin, tools=[send_wire])

    _run(plugin._projection_handler(_model_call(agent)))
    _run(plugin._projection_handler(_model_call(agent)))

    assert summarizer.names == ["send_wire"]
    assert summarizer.seen == [("send_wire", _CATALOG_CHARS)]
    assert _summary_of(plugin, agent, "send_wire") == "a recorded summary"


def test_a_second_agent_on_the_same_plugin_reuses_the_cached_line():
    """A summary depends on the description alone, so it is shared where the exposures are not."""
    summarizer = _RecordingSummarizer()
    plugin = _plugin(summarizer=summarizer)
    first = _agent(plugin, tools=[send_wire])
    _run(plugin._projection_handler(_model_call(first)))

    second = _agent(plugin, tools=[send_wire])
    _run(plugin._projection_handler(_model_call(second)))

    assert summarizer.names == ["send_wire"]
    assert plugin._states[first] is not plugin._states[second]


def test_an_async_summarizer_is_awaited():
    """Either shape is accepted: a network-backed summarizer cannot be synchronous."""
    summarizer = _AsyncSummarizer(answer="An async line.")
    plugin = _plugin(summarizer=summarizer)
    agent = _agent(plugin, tools=[send_wire])

    _run(plugin._projection_handler(_model_call(agent)))

    assert summarizer.names == ["send_wire"]
    assert _summary_of(plugin, agent, "send_wire") == "An async line."


@pytest.mark.parametrize(
    ("kwargs", "case"),
    [
        ({"raises": True}, "the summarizer raised"),
        ({"answer": ""}, "an empty answer"),
        ({"answer": "   "}, "whitespace only"),
        ({"answer": None}, "no answer at all"),
        ({"answer": 7}, "a non-string answer"),
        ({"answer": ["a line"]}, "a list instead of a line"),
    ],
)
def test_a_summarizer_that_answers_nothing_usable_falls_back_to_truncation(kwargs: dict[str, Any], case: str):
    """No tool is ever left without a line, and no summarizer failure escapes the projection."""
    summarizer = _RecordingSummarizer(**kwargs)
    plugin = _plugin(summarizer=summarizer)
    agent = _agent(plugin, tools=[send_wire])

    projected = _run(plugin._projection_handler(_model_call(agent)))

    expected = _truncate_description(" ".join(_description_of(agent, "send_wire").split()), _CATALOG_CHARS)
    assert _summary_of(plugin, agent, "send_wire") == expected, case
    assert summarizer.names == ["send_wire"]
    # The projection itself came out whole: a failed summary is not a failed call.
    assert f"- send_wire: {expected}" in str(projected.system_prompt)


@pytest.mark.parametrize(
    "answer",
    [
        "x" * 200,
        "This summary runs well past the limit it was given and keeps going for a while longer.",
        '"  Quoted   and   spaced   out   well   past   the   limit   given  "',
    ],
)
def test_an_overrunning_answer_is_clamped_to_the_limit(answer: str):
    """A summarizer that overruns cannot overrun the catalog: the clamp is applied after the fact."""
    plugin = _plugin(summarizer=_RecordingSummarizer(answer=answer))
    agent = _agent(plugin, tools=[send_wire])

    _run(plugin._projection_handler(_model_call(agent)))

    line = _summary_of(plugin, agent, "send_wire")
    assert 0 < len(line) <= _CATALOG_CHARS
    # One line, no wrapping quotes: a model asked for a bare line still answers with one sometimes.
    assert "\n" not in line
    assert not line.startswith('"')


def test_the_plugin_tools_are_never_summarized():
    """Both are emitted in full on every call, so a catalog line for them would list a callable tool."""
    summarizer = _RecordingSummarizer()
    plugin = _plugin(summarizer=summarizer)
    agent = _agent(plugin)

    projected = _run(plugin._projection_handler(_model_call(agent)))

    assert FIND_TOOLS_NAME not in summarizer.names
    assert GET_TOOL_DETAILS_NAME not in summarizer.names
    # Every other registered tool has a line -- summarized when long, verbatim when it already fits.
    assert sorted(name for name, _ in plugin._summaries) == ["check_balance", "list_positions", "send_wire"]
    for name in (FIND_TOOLS_NAME, GET_TOOL_DETAILS_NAME):
        assert f"- {name}:" not in str(projected.system_prompt)


def test_a_suppressed_catalog_summarizes_nothing_and_leaves_the_prompt_alone():
    """``catalog_chars=None`` buys the cheapest configuration: no catalog block, and no summary call."""
    summarizer = _RecordingSummarizer()
    plugin = _plugin(catalog_chars=None, summarizer=summarizer)
    agent = _agent(plugin)

    projected = _run(plugin._projection_handler(_model_call(agent, system_prompt="operator prompt")))

    assert summarizer.seen == []
    assert plugin._summaries == {}
    assert projected.system_prompt == "operator prompt"
    assert _CATALOG_PROMPT_HEADER.split("\n", 1)[0] not in str(projected.system_prompt)
    # The projection still happened: the catalog is what was suppressed, not the disclosure.
    assert [spec["name"] for spec in projected.tool_specs] == [FIND_TOOLS_NAME, GET_TOOL_DETAILS_NAME]


def test_a_redescribed_tool_gets_its_own_line_rather_than_the_stale_one():
    """The cache is keyed by ``(name, description)``, so a re-registration is a new line, not a stale one.

    The second call also offers one further tool, because the cache is only consulted when the set of
    incoming names changed -- a description that moves under an unchanged name set is not noticed. That
    is the fingerprint's granularity, not the cache's: the keys asserted below are the cache's contract.
    """
    summarizer = _RecordingSummarizer()
    plugin = _plugin(summarizer=summarizer)
    agent = _agent(plugin, tools=[send_wire, check_balance])
    context = _model_call(agent)
    wire = next(spec for spec in context.tool_specs if spec["name"] == "send_wire")
    original = wire.get("description") or ""

    without_balance = [spec for spec in context.tool_specs if spec["name"] != "check_balance"]
    _run(plugin._projection_handler(dataclasses.replace(context, tool_specs=without_balance)))

    redescribed = {**wire, "description": original + " Now also reports the counterparty bank."}
    widened = [redescribed if spec["name"] == "send_wire" else spec for spec in context.tool_specs]
    _run(plugin._projection_handler(dataclasses.replace(context, tool_specs=widened)))

    assert summarizer.names == ["send_wire", "send_wire"]
    assert ("send_wire", original) in plugin._summaries
    assert ("send_wire", redescribed["description"]) in plugin._summaries
    # check_balance joined on the second call and fit the limit, so it got a line without a call.
    assert plugin._summaries[("check_balance", _description_of(agent, "check_balance"))] == _FITTING_DESCRIPTION


# ---------------------------------------------------------------------------------------------------
# C -- the default summarizer: one plain model call, accounted
# ---------------------------------------------------------------------------------------------------


def test_the_default_summarizer_returns_the_streamed_text_and_accounts_its_usage():
    """A summary is an auxiliary cost of the strategy, so it is visible next to what it saves."""
    model = _ScriptedModel(chunks=("Wires money ", "out of an account."))
    usage: dict[str, int] = {}
    summarize = _model_summarizer(model, usage)
    spec = cast("ToolSpec", {"name": "send_wire", "description": "Send a wire transfer."})

    line = _run(cast("Coroutine[Any, Any, str]", summarize(spec, _CATALOG_CHARS)))

    assert line == "Wires money out of an account."
    assert usage == {"calls": 1, "inputTokens": 11, "outputTokens": 5}


def test_the_default_summarizer_accumulates_across_calls():
    """The counters are a session total, not a reading of the last call."""
    model = _ScriptedModel()
    usage: dict[str, int] = {}
    summarize = _model_summarizer(model, usage)
    spec = cast("ToolSpec", {"name": "send_wire", "description": "Send a wire transfer."})

    _run(cast("Coroutine[Any, Any, str]", summarize(spec, _CATALOG_CHARS)))
    _run(cast("Coroutine[Any, Any, str]", summarize(spec, _CATALOG_CHARS)))

    assert usage == {"calls": 2, "inputTokens": 22, "outputTokens": 10}
    assert len(model.calls) == 2


def test_the_default_summarizer_calls_the_model_with_no_tools_and_no_history():
    """It goes to ``model.stream`` directly, so it passes through no middleware and cannot recurse."""
    model = _ScriptedModel()
    summarize = _model_summarizer(model, {})
    spec = cast("ToolSpec", {"name": "send_wire", "description": "Send a wire transfer."})

    _run(cast("Coroutine[Any, Any, str]", summarize(spec, 25)))

    (call,) = model.calls
    assert call["tool_specs"] is None
    assert call["system_prompt"] == _SUMMARY_SYSTEM_PROMPT.format(max_chars=25)
    # One user message carrying the name and the description, and nothing else.
    assert len(call["messages"]) == 1
    assert call["messages"][0]["role"] == "user"
    text = call["messages"][0]["content"][0]["text"]
    assert "send_wire" in text
    assert "Send a wire transfer." in text


def test_with_no_summarizer_configured_the_agents_own_model_writes_the_lines():
    """The default is the agent's model, and its usage lands in that agent's disclosure state."""
    model = _ScriptedModel(chunks=("Moves money out ", "of an account."))
    plugin = _plugin()
    agent = _agent(plugin, model=model, tools=[send_wire, check_balance])

    _run(plugin._projection_handler(_model_call(agent)))

    # One call: check_balance fit the limit and never reached the model.
    assert len(model.calls) == 1
    assert _summary_of(plugin, agent, "send_wire") == "Moves money out of an account."
    assert plugin._states[agent].summary_usage == {"calls": 1, "inputTokens": 11, "outputTokens": 5}


def test_a_summary_call_that_reports_no_usage_still_counts_the_call():
    """A provider that reports no usage metadata leaves the call count as the only reading, not an error."""
    model = _ScriptedModel(usage={})
    plugin = _plugin()
    agent = _agent(plugin, model=model, tools=[send_wire])

    _run(plugin._projection_handler(_model_call(agent)))

    assert plugin._states[agent].summary_usage == {"calls": 1}


# ---------------------------------------------------------------------------------------------------
# D -- the premature-call guard exempts the loading tool
# ---------------------------------------------------------------------------------------------------


def test_the_premature_call_guard_never_cancels_the_loading_tool():
    """Cancelling ``get_tool_details`` would cancel the only way out of the catalog."""
    plugin = _plugin()
    agent = _agent(plugin)
    event = _before_tool_call(agent, GET_TOOL_DETAILS_NAME, {"names": ["send_wire"]})

    plugin._on_before_tool_call(event)

    assert not event.cancel_tool
    assert plugin._states[agent].premature_cancellations == 0


def test_the_guard_exempts_the_loading_tool_called_with_no_arguments_too():
    """``names`` is a required parameter, so without the exemption an empty call would be cancelled."""
    plugin = _plugin()
    agent = _agent(plugin)
    event = _before_tool_call(agent, GET_TOOL_DETAILS_NAME, {})

    plugin._on_before_tool_call(event)

    assert not event.cancel_tool
    assert plugin._states[agent].premature_cancellations == 0


def test_a_tool_loaded_through_the_loading_tool_is_no_longer_premature():
    """The exemption is only meaningful against a call the guard does cancel."""
    plugin = _plugin()
    agent = _agent(plugin)

    premature = _before_tool_call(agent, "send_wire", {"account": "1", "amount": "2"})
    plugin._on_before_tool_call(premature)
    assert premature.cancel_tool
    assert plugin._states[agent].premature_cancellations == 1

    _load(plugin, agent, ["send_wire"])
    retried = _before_tool_call(agent, "send_wire", {"account": "1", "amount": "2"})
    plugin._on_before_tool_call(retried)

    assert not retried.cancel_tool
    assert plugin._states[agent].premature_cancellations == 1
