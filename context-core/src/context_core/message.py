"""The neutral message shape and accessors — the contract between every framework binding and the core.

**No framework import is permitted in this module or anywhere under ``context_core``.** A framework
binding (LangGraph today, possibly Strands later) converts its native message objects to and from the
shapes defined here at its own boundary; the core only ever sees these plain dicts.

Shape
-----
A *message* is a dict::

    {"role": "user" | "assistant" | ..., "content": [block, ...]}

A *block* is a dict with exactly one of the recognised keys:

    {"text": str}                         a text block
    {"json": <json-serialisable>}         a structured block
    {"toolUse": {"toolUseId", "name", "input"}}   a tool call
    {"toolResult": {"toolUseId", "status", "content": [block, ...]}}  a tool result

This mirrors the shape the Strands SDK uses internally (which is why recreating the core from the Strands
source read as reference is faithful), but nothing here imports or depends on Strands.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = [
    "NeutralBlock",
    "NeutralMessage",
    "latest_user_text",
    "text_blocks",
    "tool_result_blocks",
    "tool_use_blocks",
]

# A block/message are plain dicts; aliased for readability rather than validated as TypedDicts, so an
# adapter can hand over the framework's own dict without a copy.
NeutralBlock = dict[str, Any]
NeutralMessage = dict[str, Any]


def text_blocks(message: NeutralMessage) -> list[str]:
    """Return the text of every ``{"text": ...}`` block in ``message``, in order.

    Args:
        message: A neutral message.

    Returns:
        The non-empty text strings, oldest block first.
    """
    return [block["text"] for block in message.get("content", []) if block.get("text")]


def tool_use_blocks(message: NeutralMessage) -> list[NeutralBlock]:
    """Return the ``toolUse`` sub-dicts of every tool call in ``message``.

    Args:
        message: A neutral message.

    Returns:
        Each block's ``toolUse`` mapping (``toolUseId`` / ``name`` / ``input``), in order.
    """
    return [block["toolUse"] for block in message.get("content", []) if "toolUse" in block]


def tool_result_blocks(message: NeutralMessage) -> list[NeutralBlock]:
    """Return the ``toolResult`` sub-dicts of every tool result in ``message``.

    Args:
        message: A neutral message.

    Returns:
        Each block's ``toolResult`` mapping (``toolUseId`` / ``status`` / ``content``), in order.
    """
    return [block["toolResult"] for block in message.get("content", []) if "toolResult" in block]


def latest_user_text(messages: Sequence[NeutralMessage]) -> str:
    """Return the newest ``user`` message's text, or ``""`` when the history holds none.

    A turn carrying only a tool result is not a question, so only ``user`` messages with at least one
    ``text`` block count; multiple text blocks in that message are joined by a line break. This mirrors
    the Strands ``_latest_question`` helper, on the neutral shape.

    Args:
        messages: The conversation history, oldest first.

    Returns:
        The newest user question, or ``""``.
    """
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        texts = text_blocks(message)
        if texts:
            return "\n".join(texts)
    return ""
