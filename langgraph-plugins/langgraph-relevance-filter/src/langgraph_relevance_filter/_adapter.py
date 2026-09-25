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


def to_neutral(msg: BaseMessage) -> NeutralMessage:
    """Convert one LangChain message to a neutral message dict."""
    role = _ROLE_BY_TYPE.get(msg.type, msg.type)

    if isinstance(msg, ToolMessage):
        return {"role": "user", "content": [tool_message_to_result_block(msg)]}

    content = _content_to_text_blocks(msg.content)

    if isinstance(msg, AIMessage):
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
    return [to_neutral(m) for m in messages]
