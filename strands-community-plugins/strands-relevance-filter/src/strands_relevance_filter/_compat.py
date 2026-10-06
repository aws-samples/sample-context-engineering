"""Every Strands SDK symbol this package uses, in one place.

The rest of the package imports from here, never from ``strands`` directly, so an SDK rename or move is a one-file
patch. Private symbols are marked as such and degrade instead of breaking the import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Imported at runtime, not under TYPE_CHECKING: @hook resolves the handler's type hints at decoration time to infer
# which event it subscribes to.
from strands import tool
from strands.hooks.events import AfterInvocationEvent, AfterToolCallEvent
from strands.plugins import Plugin, hook
from strands.types.tools import ToolContext, ToolResult, ToolResultContent

if TYPE_CHECKING:
    from strands.agent.agent import Agent as Agent
    from strands.types.content import Message as Message
    from strands.types.content import Messages as Messages

try:  # Private: the SDK may rename or move this class.
    from strands.agent._agent_as_tool import _AgentAsTool
except ImportError:  # pragma: no cover
    # Without the class there is no way to recognize a delegation tool, so that guard degrades to the caller's own
    # ``should_filter`` (or is documented as their responsibility).
    _AgentAsTool = None  # type: ignore[assignment,misc]

__all__ = [
    "AfterInvocationEvent",
    "AfterToolCallEvent",
    "Plugin",
    "ToolContext",
    "ToolResult",
    "ToolResultContent",
    "_AgentAsTool",
    "hook",
    "tool",
]
