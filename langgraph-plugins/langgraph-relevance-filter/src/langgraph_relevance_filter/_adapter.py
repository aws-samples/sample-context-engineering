"""Adapter between LangChain ``BaseMessage`` objects and the ``context_core`` neutral shape.

This is the ONLY module in the LangGraph binding that touches both worlds. ``context_core`` never imports
LangChain; the binding never leaks a LangChain type into the core. Everything crossing the boundary is
converted here.

Neutral shape (see ``context_core.message``)::

    {"role": "user"|"assistant"|..., "content": [block, ...]}
    block ∈ {"text": str} | {"json": Any}
                | {"toolUse": {"toolUseId","name","input"}}
                | {"toolResult": {"toolUseId","status","content":[block,...]}}

The mapping to LangChain message types:

    HumanMessage   <-> {"role": "user",      "content": [{"text": ...}, ...]}
    SystemMessage  <-> {"role": "system",    "content": [{"text": ...}]}
    AIMessage      <-> {"role": "assistant", "content": [{"text": ...}?, {"toolUse": ...}*]}
    ToolMessage    <-> a {"toolResult": ...} block, carried on a user-role neutral message
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from context_core.message import NeutralMessage

__all__ = ["to_neutral", "to_neutral_list", "tool_message_to_result_block", "result_block_to_content"]

_ROLE_BY_TYPE = {"human": "user", "system": "system", "ai": "assistant", "tool": "tool"}


def _content_to_text_blocks(content: Any) -> list[dict[str, Any]]:
    """Normalise a LangChain message ``content`` (str or block list) into neutral text/json blocks."""
    if isinstance(content, str):
        return [{"text": content}] if content else []
    blocks: list[dict[str, Any]] = []
    for part in content or []:
        if isinstance(part, str):
            if part:
                blocks.append({"text": part})
        elif isinstance(part, dict):
            if part.get("type") == "text" and part.get("text"):
                blocks.append({"text": part["text"]})
            else:
                blocks.append({"json": part})
    return blocks


def tool_message_to_result_block(msg: ToolMessage) -> dict[str, Any]:
    """Convert a LangChain ``ToolMessage`` into a neutral ``{"toolResult": ...}`` block."""
    status = "error" if getattr(msg, "status", None) == "error" else "success"
    return {
        "toolResult": {
            "toolUseId": msg.tool_call_id,
            "status": status,
            "content": _content_to_text_blocks(msg.content),
        }
    }


def result_block_to_content(result_block: dict[str, Any]) -> Any:
    """Render a neutral ``toolResult`` block's content back to a LangChain ``ToolMessage`` content value.

    A single text block collapses to a plain string (what a tool normally returns); anything richer is
    returned as a list of content-part dicts.
    """
    content = result_block.get("toolResult", {}).get("content", [])
    if len(content) == 1 and "text" in content[0]:
        return content[0]["text"]
    parts: list[Any] = []
    for block in content:
        if "text" in block:
            parts.append({"type": "text", "text": block["text"]})
        elif "json" in block:
            parts.append(block["json"])
    return parts


_CALL_PART_TYPES = frozenset({"tool_use", "tool_call", "function_call"})
"""Content-part types that restate a tool call ``AIMessage.tool_calls`` already carries."""


def _is_call_part(block: dict[str, Any]) -> bool:
    """Whether a neutral block is a provider's content-part copy of a tool call."""
    part = block.get("json")
    return isinstance(part, dict) and part.get("type") in _CALL_PART_TYPES


ATTACHED_TEXT_KEY = "context_core_attached_to_tool_result"
"""``additional_kwargs`` flag on a ``HumanMessage`` that is really text attached to the tool-result
message before it (see :func:`to_langchain`). Shared by the three adapters, so a message split by one
middleware is joined back by the next."""


def to_neutral(msg: BaseMessage) -> NeutralMessage:
    """Convert one LangChain message to a neutral message dict."""
    role = _ROLE_BY_TYPE.get(msg.type, msg.type)

    if isinstance(msg, ToolMessage):
        return {"role": "user", "content": [tool_message_to_result_block(msg)]}

    content = _content_to_text_blocks(msg.content)

    if isinstance(msg, AIMessage):
        # A provider such as Bedrock also carries each call as a ``tool_use`` part of ``content``.
        # ``tool_calls`` is the canonical form and becomes the ``toolUse`` block below, so the content
        # copy is dropped: kept, it survives as an opaque ``json`` block when a core removes the
        # ``toolUse`` (the disclosure fold does), and the provider then receives a toolUse with no
        # toolResult.
        content = [block for block in content if not _is_call_part(block)]
        for call in msg.tool_calls or []:
            content.append(
                {
                    "toolUse": {
                        "toolUseId": call.get("id", ""),
                        "name": call.get("name", ""),
                        "input": call.get("args", {}),
                    }
                }
            )

    return {"role": role, "content": content}


def to_neutral_list(messages: list[BaseMessage]) -> list[NeutralMessage]:
    """Convert a list of LangChain messages to neutral messages, in order."""
    return to_neutral_list_with_sources(messages)[0]


def to_neutral_list_with_sources(
    messages: Sequence[BaseMessage],
) -> tuple[list[NeutralMessage], list[list[BaseMessage]]]:
    """Convert to neutral messages and say which LangChain messages each one came from.

    Usually one to one. The exception undoes :func:`to_langchain`'s split: a neutral ``user`` message
    carrying tool results AND text renders as ``ToolMessage`` objects followed by a ``HumanMessage``
    marked with :data:`ATTACHED_TEXT_KEY`. Read back naively, that marked message would be a fresh user
    turn, and an inner middleware would see the current turn start AT it -- the disclosure fold then
    treats the turn's own ``get_tool_details`` exchanges as closed and folds them away, so the model
    never sees its load and reloads forever. The marked text is folded back onto the preceding
    tool-result message instead, restoring the exact neutral message the outer middleware produced.

    Returns:
        The neutral messages, and for each one the LangChain messages it stands for.
    """
    neutral: list[NeutralMessage] = []
    sources: list[list[BaseMessage]] = []
    for message in messages:
        previous = neutral[-1] if neutral else None
        if (
            isinstance(message, HumanMessage)
            and message.additional_kwargs.get(ATTACHED_TEXT_KEY)
            and previous is not None
            and previous.get("role") == "user"
            and any("toolResult" in block for block in previous.get("content") or ())
        ):
            previous["content"] = [*previous["content"], *_content_to_text_blocks(message.content)]
            sources[-1].append(message)
            continue
        neutral.append(to_neutral(message))
        sources.append([message])
    return neutral, sources
