"""Resolve Hermes's ``ContextEngine`` ABC — the real one when the host is installed, a faithful stub
matching the pinned contract (``CONTRACT.md``) otherwise.

Hermes Agent (``hermes-agent``) is version ``0.0.0`` and not published to PyPI, so a wheel of this
package cannot hard-depend on it and still install from an index. The import is therefore deferred to
here: ``agent.context_engine.ContextEngine`` is used when it imports, and a local stub carrying the same
abstract members and attribute defaults is used when it does not. The stub lets the engine be imported,
unit-tested and wheel-verified with no Hermes install; at runtime inside Hermes the real ABC is used, so
``isinstance(engine, ContextEngine)`` holds and ``register_context_engine`` accepts it.

The stub mirrors ``agent/context_engine.py`` exactly for the members this binding touches; it is NOT a
general reimplementation of Hermes.
"""

from __future__ import annotations

import copy
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

__all__ = ["ContextEngine", "HERMES_PRESENT"]

try:  # Real host present: use its ABC so isinstance/registration work.
    from agent.context_engine import ContextEngine  # type: ignore[assignment]

    HERMES_PRESENT = True
except Exception:  # noqa: BLE001 — any import failure means "no host", fall back to the stub.
    HERMES_PRESENT = False

    class ContextEngine(ABC):  # type: ignore[no-redef]
        """Local stand-in for ``agent.context_engine.ContextEngine`` (pinned from source).

        Carries the token-state attributes the host reads directly, the four abstract members, and the
        optional hooks with the host's documented no-op defaults. Only the surface this port relies on
        is reproduced.
        """

        # Token state the host reads directly.
        last_prompt_tokens: int = 0
        last_completion_tokens: int = 0
        last_total_tokens: int = 0
        threshold_tokens: int = 0
        context_length: int = 0
        compression_count: int = 0
        threshold_percent: float = 0.75
        protect_first_n: int = 3
        protect_last_n: int = 6
        emit_automatic_compaction_status: bool = True

        @property
        @abstractmethod
        def name(self) -> str:
            ...

        @abstractmethod
        def update_from_response(self, usage: Dict[str, Any]) -> None:
            ...

        @abstractmethod
        def should_compress(self, prompt_tokens: int = None) -> bool:  # type: ignore[assignment]
            ...

        @abstractmethod
        def compress(
            self,
            messages: List[Dict[str, Any]],
            current_tokens: Optional[int] = None,
            focus_topic: Optional[str] = None,
            force: bool = False,
            memory_context: str = "",
        ) -> List[Dict[str, Any]]:
            ...

        # --- optional hooks (host defaults) ---

        def select_context(
            self,
            request_messages: List[Dict[str, Any]],
            *,
            conversation_messages: List[Dict[str, Any]] = None,  # type: ignore[assignment]
            incoming_message: Dict[str, Any] = None,  # type: ignore[assignment]
            budget_tokens: int = 0,
        ) -> Optional[List[Dict[str, Any]]]:
            return None

        def on_turn_complete(
            self, messages: List[Dict[str, Any]], usage: Dict[str, Any] = None, **kwargs: Any  # type: ignore[assignment]
        ) -> None:
            return None

        def get_tool_schemas(self) -> List[Dict[str, Any]]:
            return []

        def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs: Any) -> str:
            return json.dumps({"error": f"Unknown context engine tool: {name}"})

        def on_session_start(self, session_id: str, **kwargs: Any) -> None:
            ...

        def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
            ...

        def on_session_reset(self) -> None:
            self.last_prompt_tokens = 0
            self.last_completion_tokens = 0
            self.last_total_tokens = 0
            self.compression_count = 0

        def get_status(self) -> Dict[str, Any]:
            last_prompt = max(self.last_prompt_tokens, 0)
            return {
                "last_prompt_tokens": last_prompt,
                "threshold_tokens": self.threshold_tokens,
                "context_length": self.context_length,
                "usage_percent": min(100, last_prompt / self.context_length * 100) if self.context_length else 0,
                "compression_count": self.compression_count,
            }

        def clone_for_agent(self) -> "ContextEngine":
            return copy.deepcopy(self)

        def update_model(
            self,
            model: str,
            context_length: int,
            base_url: str = "",
            api_key: str = "",
            provider: str = "",
            api_mode: str = "",
        ) -> None:
            self.context_length = context_length
            if not hasattr(self, "_config_threshold_percent"):
                self._config_threshold_percent = self.threshold_percent
            self.threshold_percent = self._config_threshold_percent
            self.threshold_tokens = int(context_length * self.threshold_percent)
