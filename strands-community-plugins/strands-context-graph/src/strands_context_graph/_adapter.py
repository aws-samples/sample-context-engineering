"""Strands messages and tool results <-> the neutral shape of :mod:`context_core.message`.

The neutral shape was modeled on the Strands SDK's own message dicts, so this adapter converts nothing: a Strands
``Message`` already *is* a neutral message and a ``ToolResult`` a neutral ``toolResult`` body. What it guarantees:

- **``tracking_id`` survives.** Cards address messages by this key, so the adapter hands over the same dicts and never
  rebuilds a message without it.
- **Unknown blocks pass through untouched.** ``reasoningContent``, ``image``, ``document``, ``cachePoint`` blocks and
  message-level keys such as ``metadata`` stay in place; the core ignores keys it does not recognize.
- **Nothing is mutated.** The core never writes into what it receives; the lists returned here are new.

This is the only module of the package that touches both ``strands`` types and :mod:`context_core.message`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from context_core.message import NeutralMessage

if TYPE_CHECKING:
    from ._compat import Message, Messages, ToolResult

__all__ = ["message_to_neutral", "messages_from_neutral", "messages_to_neutral", "result_to_neutral"]


def message_to_neutral(message: Message) -> NeutralMessage:
    """Return one Strands message as a neutral message: the same dict."""
    return cast(NeutralMessage, message)


def messages_to_neutral(messages: Messages) -> list[NeutralMessage]:
    """Return the Strands messages as neutral messages: a new list over the same message dicts."""
    return [cast(NeutralMessage, message) for message in messages]


def messages_from_neutral(messages: Sequence[NeutralMessage]) -> Messages:
    """Return neutral messages as Strands messages: a new list over the same message dicts."""
    return cast("Messages", list(messages))


def result_to_neutral(result: ToolResult) -> dict[str, Any]:
    """Return a Strands ``ToolResult`` as the body of a neutral ``toolResult`` block, the same dict."""
    return cast("dict[str, Any]", result)
