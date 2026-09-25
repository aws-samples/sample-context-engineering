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

Both directions are here. ``to_neutral`` also carries ``BaseMessage.id`` across as the neutral
``tracking_id``, which is the Durable Identity every graph Card is built out of: a message carrying none
takes part in no Card, so an id-less history projects whole. ``to_langchain`` puts it back, so a message
that survived a projection reaches the provider under the id it is persisted with.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from context_core.message import NeutralMessage

__all__ = [
    "result_block_to_content",
    "to_langchain",
    "to_langchain_list",
    "to_neutral",
    "to_neutral_list",
    "tool_message_to_result_block",
]

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


def to_neutral(msg: BaseMessage) -> NeutralMessage:
    """Convert one LangChain message to a neutral message dict.

    ``msg.id`` travels as ``tracking_id``, the Durable Identity the graph addresses a message by. A message
    without one contributes no identity, so its turn yields no Card and projects whole -- the same quiet
    direction the core takes everywhere else.
    """
    role = _ROLE_BY_TYPE.get(msg.type, msg.type)

    if isinstance(msg, ToolMessage):
        return _identified({"role": "user", "content": [tool_message_to_result_block(msg)]}, msg)

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

    return _identified({"role": role, "content": content}, msg)


def _identified(neutral: NeutralMessage, msg: BaseMessage) -> NeutralMessage:
    """Attach ``msg.id`` to ``neutral`` as its ``tracking_id``, when the message carries one."""
    if getattr(msg, "id", None):
        neutral["tracking_id"] = msg.id
    return neutral


def to_neutral_list(messages: list[BaseMessage]) -> list[NeutralMessage]:
    """Convert a list of LangChain messages to neutral messages, in order."""
    return [to_neutral(m) for m in messages]


# ---- neutral -> LangChain --------------------------------------------------------------------------


def _blocks_to_content(blocks: list[dict[str, Any]]) -> Any:
    """Render neutral text/json blocks back to a LangChain ``content`` value.

    A single text block collapses to a plain string, which is what the overwhelming majority of messages
    are and what ``_content_to_text_blocks`` received in the first place. Anything richer -- several
    blocks, or a ``json`` block -- is returned as a list of content parts, so nothing is lost by being
    flattened into one string.
    """
    if len(blocks) == 1 and "text" in blocks[0]:
        return blocks[0]["text"]
    parts: list[Any] = []
    for block in blocks:
        if "text" in block:
            parts.append({"type": "text", "text": block["text"]})
        elif "json" in block:
            parts.append(block["json"])
    return parts


def to_langchain(message: NeutralMessage) -> list[BaseMessage]:
    """Convert one neutral message back to the LangChain message(s) it stands for.

    A **list**, because the mapping is not one to one in this direction: a neutral ``user`` message is the
    carrier for tool results, and the projection's own compaction appends its final block as a text block
    to the last ``user`` message -- which in an autonomous tool loop is the message carrying a
    ``toolResult``. LangChain has no message type holding both, so that one neutral message renders as the
    ``ToolMessage`` followed by a ``HumanMessage`` with the appended text. The tool result therefore stays
    immediately behind the tool call, which is the ordering a provider requires, and the folded text lands
    behind it as trailing per-call content.

    ``tracking_id`` is restored as the message ``id``. When a neutral message splits, the identity goes to
    the first message out -- the one that existed before the fold -- and the synthetic carrier of the
    folded text gets none: it is per-call content, not a persisted message.

    Args:
        message: A neutral message. Read only.

    Returns:
        The LangChain messages, in order. Empty when the neutral message carries no content at all.
    """
    role = message.get("role")
    blocks = list(message.get("content") or ())
    identity = message.get("tracking_id")

    if role == "assistant":
        tool_calls = [
            {
                "name": block["toolUse"].get("name", ""),
                "args": block["toolUse"].get("input", {}) or {},
                "id": block["toolUse"].get("toolUseId", ""),
                "type": "tool_call",
            }
            for block in blocks
            if "toolUse" in block
        ]
        body = [block for block in blocks if "text" in block or "json" in block]
        if not body and not tool_calls:
            # Nothing at all: several providers reject an empty message outright.
            return []
        return [AIMessage(content=_blocks_to_content(body), tool_calls=tool_calls, id=identity)]

    if role == "system":
        return [SystemMessage(content=_blocks_to_content(blocks), id=identity)] if blocks else []

    results = [block for block in blocks if "toolResult" in block]
    body = [block for block in blocks if "text" in block or "json" in block]

    if role == "tool" or results:
        out: list[BaseMessage] = []
        for position, block in enumerate(results):
            result = block["toolResult"]
            out.append(
                ToolMessage(
                    content=result_block_to_content(block),
                    tool_call_id=result.get("toolUseId", ""),
                    status="error" if result.get("status") == "error" else "success",
                    # Only the first carries the identity: the rest are blocks of the same neutral message.
                    id=identity if position == 0 else None,
                )
            )
        if body:
            # The compaction's final block, folded onto a tool-result turn. It is per-call content, so it
            # travels as its own message and claims no Durable Identity.
            out.append(HumanMessage(content=_blocks_to_content(body)))
        return out

    return [HumanMessage(content=_blocks_to_content(blocks), id=identity)] if blocks else []


def to_langchain_list(messages: Sequence[NeutralMessage]) -> list[BaseMessage]:
    """Convert a neutral message list back to LangChain messages, in order.

    The inverse of :func:`to_neutral_list` for everything that round-trips: a message list converted out
    and back is the same list of LangChain messages, by value. It is deliberately **not** injective the
    other way -- see :func:`to_langchain` for the two shapes that do not exist as a single LangChain
    message.

    Args:
        messages: The neutral messages, oldest first. Read only.

    Returns:
        The LangChain messages, oldest first.
    """
    return [converted for message in messages for converted in to_langchain(message)]
