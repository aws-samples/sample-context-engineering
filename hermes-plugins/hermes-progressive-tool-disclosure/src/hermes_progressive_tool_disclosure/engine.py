"""The ``ProgressiveToolDisclosureEngine``: Practice B on Hermes's synchronous ``ContextEngine``.

Practice B shows the model a lean one-line-per-tool catalog instead of full schemas, and expands a
tool's full spec on demand. The index and catalog/fold logic live in ``context_core.disclosure``.

Mapping onto the Hermes ABC and the documented compromise:

Strands and LangGraph rewrite the model request's ``tool_specs``/``tools`` directly. Hermes's
``ContextEngine`` does NOT own the agent's base tool catalog — it owns ``get_tool_schemas()`` (its own
tools) and ``select_context()`` (the messages). So B is implemented the **portable** way the repo's
design docs already describe:

- ``select_context(request_messages)`` injects the summarized catalog (one line per tool, via
  ``context_core.disclosure.build_catalog``) into the system message, and folds prior closed
  ``ptd_get_tool_details`` exchanges (``fold_closed_exchanges``) so a loaded spec is not re-sent forever.
- ``get_tool_schemas()`` → ``[ptd_find_tools, ptd_get_tool_details]``.
- ``handle_tool_call()`` → ``ptd_find_tools`` runs the lexical index; ``ptd_get_tool_details`` returns a tool's
  full spec as JSON (and marks it active, so the next ``select_context`` folds and does not re-summarize
  it).

**Documented limitation (see README, non-blocking follow-up):** the ABC exposes no hook to strip the
*base* tool schemas from the provider request, so the model still technically receives the base schemas
Hermes assembled; B steers the model to the catalog + on-demand expansion rather than removing the base
schemas. This matches the LangGraph README's "Portable" note.

The engine is constructed with the agent's ``tool_specs`` (the catalog source) because the ABC provides
no access to the host tool set.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from context_core.disclosure import (
    DEFAULT_CATALOG_CHARS,
    FIND_TOOLS_NAME,
    GET_TOOL_DETAILS_NAME,
    LexicalToolIndex,
    SummaryCache,
    ToolIndex,
    ToolSpec,
    build_catalog,
    fold_closed_exchanges,
)

from ._adapter import hermes_to_neutral_list, neutral_to_hermes_list
from ._base import BaseEngine

__all__ = ["ProgressiveToolDisclosureEngine", "register"]

logger = logging.getLogger(__name__)

_DEFAULT_TOP_K = 5


def _openai_tool_to_spec(tool: Dict[str, Any]) -> Optional[ToolSpec]:
    """Convert an OpenAI function-tool dict to a ``context_core`` ``ToolSpec``."""
    fn = tool.get("function") if isinstance(tool, dict) else None
    if not isinstance(fn, dict) or not fn.get("name"):
        return None
    return {
        "name": fn["name"],
        "description": fn.get("description", ""),
        "inputSchema": {"json": fn.get("parameters", {})},
    }


class ProgressiveToolDisclosureEngine(BaseEngine):
    """Progressive tool disclosure on Hermes's ``ContextEngine`` surface.

    Args:
        tool_specs: The agent's base tools, as OpenAI function-tool dicts
            (``{"type": "function", "function": {"name", "description", "parameters"}}``). These seed the
            catalog and the lexical index. Required — the ABC gives no access to the host tool set.
        catalog_chars: Max characters of one catalog summary line (default 80), or ``None`` to suppress
            the catalog (the two disclosure tools' descriptions are then the only hint others exist).
        index: Tool index for ``ptd_find_tools``. Defaults to the lexical (term-frequency)
            :class:`~context_core.disclosure.LexicalToolIndex`.
        top_k: Max matches ``ptd_find_tools`` returns (default 5).
        always_available: Tool names always kept active (never catalog-only).
    """

    def __init__(
        self,
        tool_specs: List[Dict[str, Any]] | None = None,
        *,
        catalog_chars: int | None = DEFAULT_CATALOG_CHARS,
        index: ToolIndex | None = None,
        top_k: int = _DEFAULT_TOP_K,
        always_available: "frozenset[str] | None" = None,
    ) -> None:
        super().__init__()
        specs: List[ToolSpec] = []
        for tool in tool_specs or []:
            spec = _openai_tool_to_spec(tool)
            if spec is not None:
                specs.append(spec)
        self._specs = specs
        self._catalog_chars = catalog_chars
        self._top_k = max(1, int(top_k))
        self._always_available = frozenset(always_available or ())
        self._index: ToolIndex = index if index is not None else LexicalToolIndex()
        self._summary_cache = SummaryCache()
        self._index_built = False
        # Tools the model has expanded this session; folded (not re-summarized) on the next call.
        self._active: set[str] = set(self._always_available)

    @property
    def name(self) -> str:
        return "progressive-tool-disclosure"

    def _ensure_index(self) -> None:
        if not self._index_built and self._specs:
            result = self._index.build(self._specs)
            # LexicalToolIndex.build is sync; a protocol impl may return an awaitable.
            if result is not None:
                import asyncio

                asyncio.get_event_loop().run_until_complete(result)  # pragma: no cover
            self._index_built = True

    # ------------------------------------------------------------------ select_context

    def select_context(
        self,
        request_messages: List[Dict[str, Any]],
        *,
        conversation_messages: List[Dict[str, Any]] = None,  # type: ignore[assignment]
        incoming_message: Dict[str, Any] = None,  # type: ignore[assignment]
        budget_tokens: int = 0,
    ) -> Optional[List[Dict[str, Any]]]:
        """Inject the catalog into the system message and fold closed disclosure exchanges."""
        try:
            if not self._specs:
                return None
            neutral = hermes_to_neutral_list(list(request_messages))
            folded = fold_closed_exchanges(neutral, active_tool_names=self._active)
            folded_list = list(folded)
            catalog = build_catalog(
                self._specs,
                self._catalog_chars,
                active_tool_names=self._active,
                cache=self._summary_cache,
            )
            changed = folded is not neutral
            if catalog:
                changed = self._inject_catalog(folded_list, catalog) or changed
            if not changed:
                return None
            return neutral_to_hermes_list(folded_list)
        except Exception:  # noqa: BLE001 — fail open.
            logger.debug("disclosure select_context failed; leaving request unchanged", exc_info=True)
            return None

    def _inject_catalog(self, messages: List[Dict[str, Any]], catalog: str) -> bool:
        """Append the catalog block to the first system message, or prepend a system message. True if changed."""
        for message in messages:
            if message.get("role") == "system":
                existing = message.get("content") or []
                message["content"] = [*existing, {"text": catalog}]
                return True
        messages.insert(0, {"role": "system", "content": [{"text": catalog}]})
        return True

    # ------------------------------------------------------------------ tools

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [_find_tools_schema(), _get_tool_details_schema()]

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs: Any) -> str:
        try:
            if name == FIND_TOOLS_NAME:
                return self._find_tools(args.get("need", ""))
            if name == GET_TOOL_DETAILS_NAME:
                return self._get_tool_details(args.get("names", []))
        except Exception as error:  # noqa: BLE001
            logger.debug("disclosure tool %s failed", name, exc_info=True)
            return json.dumps({"error": f"{name} failed: {error}"})
        return json.dumps({"error": f"Unknown context engine tool: {name}"})

    def _find_tools(self, need: str) -> str:
        if not isinstance(need, str) or not need.strip():
            return json.dumps({"error": "need is required"})
        self._ensure_index()
        matches = self._index.search(need, self._top_k)
        if matches is not None and not isinstance(matches, (list, tuple)):
            import asyncio

            matches = asyncio.get_event_loop().run_until_complete(matches)  # pragma: no cover
        found = [{"name": m.name, "score": m.score} for m in matches]
        return json.dumps({"matches": found})

    def _get_tool_details(self, names: Any) -> str:
        if isinstance(names, str):
            names = [names]
        if not isinstance(names, list) or not names:
            return json.dumps({"error": "names must be a non-empty list"})
        by_name = {spec["name"]: spec for spec in self._specs}
        details: Dict[str, Any] = {}
        for n in names:
            spec = by_name.get(n)
            if spec is None:
                details[n] = {"error": "unknown tool"}
                continue
            details[n] = {
                "name": spec["name"],
                "description": spec.get("description", ""),
                "inputSchema": spec.get("inputSchema", {}),
            }
            self._active.add(n)  # keep active so the next call folds, not re-summarizes.
        return json.dumps({"tools": details})

    def on_session_reset(self) -> None:
        super().on_session_reset()
        self._active = set(self._always_available)


def _find_tools_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": FIND_TOOLS_NAME,
            "description": (
                "Search the tool catalog for tools that can do what you need, described in natural "
                "language. Returns matching tool names; call ptd_get_tool_details on them to see full "
                "parameters before using them."
            ),
            "parameters": {
                "type": "object",
                "properties": {"need": {"type": "string", "description": "What you need to do, in natural language."}},
                "required": ["need"],
            },
        },
    }


def _get_tool_details_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": GET_TOOL_DETAILS_NAME,
            "description": (
                "Get the full specification (description + parameters) of one or more tools by name, so "
                "you can call them correctly. Use after ptd_find_tools surfaces a candidate."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "names": {"type": "array", "items": {"type": "string"}, "description": "Tool names to expand."}
                },
                "required": ["names"],
            },
        },
    }


def register(ctx: Any) -> None:
    """Hermes plugin entry point: register the single disclosure engine (no base tools known at import)."""
    ctx.register_context_engine(ProgressiveToolDisclosureEngine())
