"""Shared doubles for the progressive-tool-disclosure tests: tools, a request builder, a fake model.

Nothing here reaches a provider. The model side is either a recording handler standing in for the rest
of the middleware chain, or :class:`ScriptedModel`, which replays prepared ``AIMessage`` objects and
records the tool list and system prompt it was called with.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool, tool
from langchain_core.utils.function_calling import convert_to_openai_tool

from langgraph_progressive_tool_disclosure.middleware import ProgressiveToolDisclosureMiddleware

# --------------------------------------------------------------------------------------------------
# Domain tools. Two with a required parameter (the guard applies) and one without (it does not).
# --------------------------------------------------------------------------------------------------


@tool
def get_balance(account_id: str) -> str:
    """Return the current cash balance of one bank account, in the account's own currency."""
    return "1000"


@tool
def list_investment_transactions(account_id: str, since: str) -> str:
    """List the transactions of an investment account since a date, with amount, date and counterparty."""
    return "3 transactions"


@tool
def current_time() -> str:
    """Return the current time. Takes no arguments at all, so an empty call to it is legitimate."""
    return "12:00"


DOMAIN_TOOLS: list[BaseTool] = [get_balance, list_investment_transactions, current_time]


def bound_tools(middleware: ProgressiveToolDisclosureMiddleware) -> list[BaseTool]:
    """Return what ``create_agent`` binds: the domain tools plus the middleware's own two."""
    return [*DOMAIN_TOOLS, *middleware.tools]


def names_of(tools: Sequence[Any]) -> set[str]:
    """Return the names of a bound tool list, whatever shape its entries are in."""
    out: set[str] = set()
    for entry in tools:
        if isinstance(entry, BaseTool):
            out.add(entry.name)
        elif isinstance(entry, dict):
            body = entry.get("function", entry)
            if body.get("name"):
                out.add(body["name"])
    return out


# --------------------------------------------------------------------------------------------------
# The request, and the handler that records what the middleware handed on.
# --------------------------------------------------------------------------------------------------


def make_request(
    *,
    tools: Sequence[Any],
    messages: Sequence[BaseMessage] = (),
    system_message: SystemMessage | None = None,
    state: Mapping[str, Any] | None = None,
) -> ModelRequest:
    """Build a ``ModelRequest`` the way the agent's model node would, with no graph behind it."""
    return ModelRequest(
        model=None,
        messages=list(messages),
        system_message=system_message,
        tool_choice=None,
        tools=list(tools),
        response_format=None,
        state=dict(state or {"messages": list(messages)}),
        runtime=None,
        model_settings={},
    )


class RecordingHandler:
    """Stands in for the rest of the middleware chain: records the request, returns an empty response."""

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def __call__(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(result=[AIMessage(content="ok")], structured_response=None)

    @property
    def last(self) -> ModelRequest:
        """The request the model was actually called with."""
        assert self.requests, "the handler was never called"
        return self.requests[-1]


def run(
    middleware: ProgressiveToolDisclosureMiddleware, request: ModelRequest
) -> tuple[ModelRequest, RecordingHandler]:
    """Push ``request`` through ``middleware.wrap_model_call`` and return what the model saw."""
    handler = RecordingHandler()
    middleware.wrap_model_call(request, handler)
    return handler.last, handler


# --------------------------------------------------------------------------------------------------
# The tool-call side.
# --------------------------------------------------------------------------------------------------


def make_tool_runtime(
    *, tools: Sequence[Any], state: Mapping[str, Any] | None = None, tool_call_id: str = "tc"
) -> Any:
    """Build the object the tool runtime injection would pass, with only the members read here."""
    return SimpleNamespace(tools=list(tools), state=dict(state or {"messages": []}), tool_call_id=tool_call_id)


def make_tool_call_request(
    *,
    name: str,
    args: Mapping[str, Any] | None = None,
    tools: Sequence[Any],
    messages: Sequence[BaseMessage] = (),
    state: Mapping[str, Any] | None = None,
    call_id: str = "tc1",
) -> ToolCallRequest:
    """Build a ``ToolCallRequest`` the way the tool node would."""
    resolved_state = dict(state or {}) or {"messages": list(messages)}
    resolved_state.setdefault("messages", list(messages))
    return ToolCallRequest(
        tool_call={"id": call_id, "name": name, "args": dict(args or {}), "type": "tool_call"},
        tool=None,
        state=resolved_state,
        runtime=make_tool_runtime(tools=tools, state=resolved_state, tool_call_id=call_id),
    )


class CountingHandler:
    """Runs a tool call by recording it. Its call count is what proves the guard did or did not fire."""

    def __init__(self) -> None:
        self.calls: list[ToolCallRequest] = []

    def __call__(self, request: ToolCallRequest) -> Any:
        self.calls.append(request)
        return "ran"


# --------------------------------------------------------------------------------------------------
# A fake chat model, for the one end-to-end test through create_agent.
# --------------------------------------------------------------------------------------------------


class ScriptedModel(BaseChatModel):
    """Replays prepared ``AIMessage`` objects and records the tools and prompt of every call.

    The recording is what the end-to-end test asserts on: it is the only place the *actual* tool list a
    provider would have been charged for is observable.
    """

    responses: list[AIMessage] = []
    calls: list[dict[str, Any]] = []
    summary_calls: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
        """Record nothing here: the binding is per call, and ``_generate`` is where it is observed."""
        declared = [convert_to_openai_tool(t) if isinstance(t, BaseTool) else t for t in tools]
        return self.bind(tools=declared, **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        system = "\n".join(m.text for m in messages if isinstance(m, SystemMessage))
        # The default summarizer calls this same model directly. Those calls are answered with a canned
        # catalog line and recorded apart, so they neither consume the script nor count as agent turns.
        if system.startswith("You write catalog lines for tools."):
            self.summary_calls.append(list(messages))
            line = AIMessage(content="summarized line", usage_metadata={"input_tokens": 40, "output_tokens": 4, "total_tokens": 44})
            return ChatResult(generations=[ChatGeneration(message=line)])
        self.calls.append(
            {
                "tools": sorted(names_of(kwargs.get("tools") or [])),
                "system": system,
                "messages": list(messages),
            }
        )
        index = len(self.calls) - 1
        reply = self.responses[index] if index < len(self.responses) else AIMessage(content="done", id=f"final-{index}")
        return ChatResult(generations=[ChatGeneration(message=reply)])
