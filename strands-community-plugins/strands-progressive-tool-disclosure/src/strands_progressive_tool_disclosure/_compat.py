"""Every Strands SDK symbol this package uses, in one place, including the middleware API.

The rest of the package imports from here, never from ``strands`` directly, so an upstream rename touches one file
and the compatibility test has a single target. The middleware seam (``InvokeModelStage``) became public in Strands
1.62 (``strands.middleware``), which removed the private ``strands._middleware`` package without an alias; this module
imports from whichever of the two the installed SDK has, so the rest of the package runs unchanged on both sides.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

try:  # Strands 1.62 and later: the middleware system is public.
    from strands.middleware import InvokeModelContext, InvokeModelStage, MiddlewareInputPhase
    from strands.middleware._registry import MiddlewareRegistry
except ImportError:  # Strands 1.61 and earlier: the same classes, under the private package.
    from strands._middleware.registry import MiddlewareRegistry
    from strands._middleware.stages import InvokeModelContext, InvokeModelStage
    from strands._middleware.types import MiddlewareInputPhase

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
    "MiddlewareInputPhase",
    "MiddlewareRegistry",
    "Plugin",
    "SystemPrompt",
    "ToolContext",
    "ToolSpec",
    "hook",
    "tool",
]
