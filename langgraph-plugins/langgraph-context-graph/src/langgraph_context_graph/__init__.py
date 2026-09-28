"""LangGraph binding for Practice D — context graph.

The Card model, scan, scoring, matcher and per-call projection live in ``context_core.graph``; this
package holds only the ``create_agent`` middleware (``wrap_model_call`` projecting the message list via
``request.override(messages=…)``, ``wrap_tool_call`` recording a tool return as an addressable artifact),
the ``expand_card``/``expand_artifact``/``find_context`` tools, the serialized-graph state, and the
message adapter. See :mod:`langgraph_context_graph.middleware`.
"""

from .middleware import ContextGraphMiddleware

__all__ = ["ContextGraphMiddleware"]
