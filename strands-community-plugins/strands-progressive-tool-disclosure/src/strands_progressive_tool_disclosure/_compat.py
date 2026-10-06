"""Every Strands SDK symbol this package uses, in one place, including the private middleware API.

The rest of the package imports from here, never from ``strands`` directly, so an upstream rename touches one file
and the compatibility test has a single target. The middleware seam (``InvokeModelStage``) has no public API yet;
when one lands, this module is the one adoption point, and the rest of the package does not change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Private: there is no public middleware API yet.
from strands._middleware.stages import InvokeModelContext, InvokeModelStage

# Imported at runtime, not under TYPE_CHECKING: @hook resolves the handler's type hints at decoration time to infer
# which event it subscribes to, and @tool reads ToolContext the same way.
from strands.hooks.events import AfterToolCallEvent, BeforeToolCallEvent
from strands.plugins import Plugin, hook
from strands.tools.decorator import tool
from strands.types.content import Messages, SystemPrompt
from strands.types.tools import ToolContext, ToolSpec

if TYPE_CHECKING:
    from strands.agent.agent import Agent as Agent
    from strands.models.model import Model as Model

__all__ = [
    "AfterToolCallEvent",
    "BeforeToolCallEvent",
    "InvokeModelContext",
    "InvokeModelStage",
    "Messages",
    "Plugin",
    "SystemPrompt",
    "ToolContext",
    "ToolSpec",
    "hook",
    "tool",
]
