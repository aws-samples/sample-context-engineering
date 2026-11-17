"""Every Strands SDK symbol this package uses, in one place, including the private ones.

The rest of the package imports from here, never from ``strands`` directly, so an SDK rename or move is a one-file
patch. The ``InvokeModelStage`` middleware stage became public in Strands 1.62 (``strands.middleware``), which removed
the private ``strands._middleware`` package without an alias, so it is imported from whichever of the two the installed
SDK has. One private seam remains: the ``registry._handlers`` reorder done by the delivery, which reads the registry
object it is handed and imports nothing.

The three SDK primitives the plugin used to import -- ``_is_user_turn``, ``_get_tool_use_ids`` / ``is_pinned`` and
``_create_injection_middleware`` -- are no longer imported: ``context_core.graph`` carries framework-free copies, and
``tests/test_sdk_parity.py`` fails (or skips naming the symbol) when the SDK rule moves away from them.

Importing this module also registers the Strands locations of the three optional bridges the core's artifact store
reaches through :data:`context_core.graph.store.HOST_SYMBOLS`: text recovery, targeted reads, and ``ContextManager``
Stash interop. ``setdefault`` keeps any location a caller registered first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from context_core.graph.store import HOST_SYMBOLS

from strands.agent.conversation_manager import NullConversationManager

# Imported at runtime, not under TYPE_CHECKING: @tool reads ToolContext from the signature at decoration time.
from strands.hooks.events import AfterToolCallEvent
from strands.plugins import Plugin
from strands.tools.decorator import tool
from strands.types.tools import ToolContext

try:  # Strands 1.62 and later: the middleware system is public.
    from strands.middleware import InvokeModelContext, InvokeModelStage
    from strands.middleware._registry import MiddlewareRegistry
except ImportError:  # Strands 1.61 and earlier: the same classes, under the private package.
    from strands._middleware.registry import MiddlewareRegistry
    from strands._middleware.stages import InvokeModelContext, InvokeModelStage

if TYPE_CHECKING:
    from strands.agent.agent import Agent as Agent
    from strands.types.content import Message as Message
    from strands.types.content import Messages as Messages
    from strands.types.tools import ToolResult as ToolResult

__all__ = [
    "AfterToolCallEvent",
    "InvokeModelContext",
    "InvokeModelStage",
    "MiddlewareRegistry",
    "NullConversationManager",
    "Plugin",
    "ToolContext",
    "tool",
]

STRANDS_HOST_SYMBOLS: dict[str, tuple[str, str]] = {
    "extract_text": ("strands._context_manager.retrieval_tool", "_extract_text"),
    "search_content": ("strands.vended_plugins.context_offloader.search", "_search_content"),
    "context_manager": ("strands._context_manager.context_manager", "ContextManager"),
}
"""Where the Strands SDK keeps the three optional bridges. Each is resolved lazily and degrades when absent."""

for _key, _location in STRANDS_HOST_SYMBOLS.items():
    HOST_SYMBOLS.setdefault(_key, _location)
