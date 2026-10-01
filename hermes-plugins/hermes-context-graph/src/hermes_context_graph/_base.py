"""``_BaseEngine`` — the token-tracking boilerplate every Hermes engine in this repo shares.

Carries the four abstract ``ContextEngine`` members with budget-aware defaults (``name`` stays
abstract — each engine names itself), the ``update_from_response`` token buckets, ``update_model``
threshold recomputation, and ``get_status``/``on_session_reset`` sentinel handling. A concrete engine
adds only its real seam logic (``select_context`` / ``on_turn_complete`` / ``get_tool_schemas`` /
``handle_tool_call``).

Deliberately duplicated per package rather than placed in ``context-core``: the core must stay
framework-free (no knowledge of Hermes's ``ContextEngine``), exactly as the LangGraph bindings keep
their ``_compat`` shim out of the core.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional

from ._compat import ContextEngine

__all__ = ["BaseEngine"]


class BaseEngine(ContextEngine):
    """Budget/token-tracking base for a Hermes ``ContextEngine``.

    Each practice engine subclasses this and overrides only its real hooks. The three abstract members
    other than ``name`` are given safe, budget-only defaults here: this port saves tokens through
    ``select_context`` rewriting, not through destructive compaction, so ``compress`` is a no-op unless
    the context is over budget, where it falls back to a deterministic tool-result trim.
    """

    #: Default context length assumed before the host calls ``update_model`` (a conservative 200k).
    DEFAULT_CONTEXT_LENGTH: int = 200_000

    def __init__(self) -> None:
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.compression_count = 0
        self.context_length = self.DEFAULT_CONTEXT_LENGTH
        self.threshold_tokens = int(self.context_length * self.threshold_percent)

    @property
    @abstractmethod
    def name(self) -> str:
        """Each engine names itself (the id selected by ``context.engine`` in ``config.yaml``)."""

    # ------------------------------------------------------------------ token state

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        """Track token usage after an LLM call. ``prompt``/``completion``/``total`` always present."""
        if not isinstance(usage, dict):
            return
        self.last_prompt_tokens = int(usage.get("prompt_tokens", self.last_prompt_tokens) or 0)
        self.last_completion_tokens = int(usage.get("completion_tokens", self.last_completion_tokens) or 0)
        self.last_total_tokens = int(
            usage.get("total_tokens", self.last_prompt_tokens + self.last_completion_tokens) or 0
        )

    def should_compress(self, prompt_tokens: int = None) -> bool:  # type: ignore[assignment]
        """Compaction fires only when the live prompt exceeds the model threshold."""
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        return bool(self.threshold_tokens) and int(tokens or 0) > self.threshold_tokens

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: Optional[int] = None,
        focus_topic: Optional[str] = None,
        force: bool = False,
        memory_context: str = "",
    ) -> List[Dict[str, Any]]:
        """Budget fallback. This port's saving is in ``select_context``, so compaction is a last
        resort: when over budget (or forced) trim old tool-result payloads deterministically, else
        return the messages unchanged. Never raises; always returns a valid OpenAI-format list.
        """
        if not messages:
            return messages
        over_budget = force or self.should_compress(current_tokens)
        if not over_budget:
            return messages
        trimmed, n = self.prune_tool_results_only(messages, current_tokens)
        if n:
            self.compression_count += 1
        return trimmed

    def prune_tool_results_only(
        self, messages: List[Dict[str, Any]], current_tokens: int | None = None
    ) -> "tuple[List[Dict[str, Any]], int]":
        """Deterministically shrink the oldest oversized tool-result payloads without an LLM call.

        Keeps the protected head (``protect_first_n``) and tail (``protect_last_n``) intact; replaces an
        older ``role == "tool"`` message's content with a short stub. Returns ``(messages, n_pruned)``.
        """
        if not messages:
            return messages, 0
        head = self.protect_first_n
        tail = len(messages) - self.protect_last_n
        pruned = 0
        out: List[Dict[str, Any]] = []
        for index, message in enumerate(messages):
            if (
                head <= index < tail
                and isinstance(message, dict)
                and message.get("role") == "tool"
                and isinstance(message.get("content"), str)
                and len(message["content"]) > 2_000
            ):
                stub = dict(message)
                stub["content"] = "[older tool result trimmed to fit the context budget]"
                out.append(stub)
                pruned += 1
            else:
                out.append(message)
        return out, pruned
