"""The ``AllThreeEngine``: one Hermes ``ContextEngine`` composing all three practices.

Hermes is **single-select** — ``register_context_engine`` refuses a second engine — so the "all three"
configuration (the source of the benchmark's headline token numbers) cannot be three co-installed
plugins the way LangGraph stacks three middlewares. It is one composed engine.

Composition reuses the three single engines unchanged; nothing is reimplemented:

- ``select_context`` pipelines the message list in the LangGraph compose order:
  **D projects** closed turns into Cards → **B injects** the catalog and folds disclosure exchanges →
  **A rewrites** oversized tool results. Each stage is the same ``context_core`` call the single engine
  makes.
- One **shared relevance store** is handed to D as its ``stash`` so a ``[ref: …]`` the filter minted
  resolves through ``expand_artifact`` as well as ``retrieve_all_context``.
- ``get_tool_schemas`` is the union of the six tools, the two retrieval tools scoped by name so the
  model does not read them as the same job.
- ``handle_tool_call`` dispatches by tool name to the owning practice.
- ``on_turn_complete`` runs A's close then D's indexing, in order.
- The relevance threshold default is the benchmark value ``0.02``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from hermes_context_graph.engine import ContextGraphEngine
from hermes_progressive_tool_disclosure.engine import ProgressiveToolDisclosureEngine
from hermes_relevance_filter.engine import RelevanceFilterEngine

from ._base import BaseEngine

__all__ = ["AllThreeEngine", "register"]

logger = logging.getLogger(__name__)

#: Benchmark default for the relevance threshold in the composed configuration (a distribution
#: position, not an absolute score — see the package README).
_DEFAULT_RELEVANCE_THRESHOLD = 0.02


class AllThreeEngine(BaseEngine):
    """Compose relevance filtering + progressive tool disclosure + context graph in one engine.

    Args:
        tool_specs: The agent's base tools (OpenAI function-tool dicts), passed to the disclosure engine
            for its catalog + index. Required for disclosure to do anything.
        relevance_threshold: Relevance preview threshold (default ``0.02``, the benchmark value).
        max_result_tokens: A's oversized-result gate (default 8000).
        relevance_config: Extra relevance preview tuning merged over the threshold default.
        disclosure_catalog_chars: B catalog line length (default 80), or ``None`` to suppress it.
        graph_kwargs: Extra keyword arguments forwarded to the context-graph engine.
    """

    def __init__(
        self,
        tool_specs: List[Dict[str, Any]] | None = None,
        *,
        relevance_threshold: float = _DEFAULT_RELEVANCE_THRESHOLD,
        max_result_tokens: int = 8_000,
        relevance_config: Dict[str, Any] | None = None,
        disclosure_catalog_chars: int | None = 80,
        graph_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        config = {"relevance_threshold": relevance_threshold}
        if relevance_config:
            config.update(relevance_config)
        self._relevance = RelevanceFilterEngine(
            max_result_tokens=max_result_tokens, config=config, include_retrieval_tool=True
        )
        self._disclosure = ProgressiveToolDisclosureEngine(
            tool_specs=tool_specs, catalog_chars=disclosure_catalog_chars
        )
        # D reads A's store as its stash, so a filter [ref] resolves through expand_artifact too.
        self._graph = ContextGraphEngine(
            stash=self._relevance.stash, include_artifact_tool=True, **(graph_kwargs or {})
        )
        self._tool_owner: Dict[str, Any] = {}
        for owner in (self._relevance, self._disclosure, self._graph):
            for schema in owner.get_tool_schemas():
                self._tool_owner[schema["function"]["name"]] = owner

    @property
    def name(self) -> str:
        return "all-three"

    # ------------------------------------------------------------------ select_context (D -> B -> A)

    def select_context(
        self,
        request_messages: List[Dict[str, Any]],
        *,
        conversation_messages: List[Dict[str, Any]] = None,  # type: ignore[assignment]
        incoming_message: Dict[str, Any] = None,  # type: ignore[assignment]
        budget_tokens: int = 0,
    ) -> Optional[List[Dict[str, Any]]]:
        """Pipeline D (project) -> B (catalog + fold) -> A (rewrite). Fail-open; request-only."""
        current = request_messages
        changed = False
        for stage in (self._graph, self._disclosure, self._relevance):
            try:
                out = stage.select_context(
                    current,
                    conversation_messages=conversation_messages,
                    incoming_message=incoming_message,
                    budget_tokens=budget_tokens,
                )
            except Exception:  # noqa: BLE001 — one stage failing never breaks the request.
                logger.debug("all-three stage %s select_context failed", stage.name, exc_info=True)
                out = None
            if out is not None:
                current = out
                changed = True
        return current if changed else None

    # ------------------------------------------------------------------ on_turn_complete (A then D)

    def on_turn_complete(self, messages: List[Dict[str, Any]], usage: Dict[str, Any] = None, **kwargs: Any) -> None:  # type: ignore[assignment]
        """A's close then D's indexing, in order."""
        self._relevance.on_turn_complete(messages, usage, **kwargs)
        self._graph.on_turn_complete(messages, usage, **kwargs)

    # ------------------------------------------------------------------ tools (union + dispatch)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        schemas: List[Dict[str, Any]] = []
        for owner in (self._relevance, self._disclosure, self._graph):
            schemas.extend(owner.get_tool_schemas())
        return schemas

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs: Any) -> str:
        owner = self._tool_owner.get(name)
        if owner is None:
            import json

            return json.dumps({"error": f"Unknown context engine tool: {name}"})
        return owner.handle_tool_call(name, args, **kwargs)

    # ------------------------------------------------------------------ lifecycle fan-out

    def update_model(self, model: str, context_length: int, base_url: str = "", api_key: str = "", provider: str = "", api_mode: str = "") -> None:  # type: ignore[override]
        super().update_model(model, context_length, base_url, api_key, provider, api_mode)
        for stage in (self._relevance, self._disclosure, self._graph):
            stage.update_model(model, context_length, base_url, api_key, provider, api_mode)

    def on_session_reset(self) -> None:
        super().on_session_reset()
        for stage in (self._relevance, self._disclosure, self._graph):
            stage.on_session_reset()


def register(ctx: Any) -> None:
    """Hermes plugin entry point: register the single composed engine."""
    ctx.register_context_engine(AllThreeEngine())
