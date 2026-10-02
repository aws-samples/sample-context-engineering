"""LangGraph binding for Practice B — progressive tool disclosure.

The lexical index and catalog/fold logic live in ``context_core.disclosure``; this package holds only the
``create_agent`` middleware (``wrap_model_call`` rewriting tools + system prompt + folded messages via
``request.override``), the ``ptd_find_tools``/``ptd_get_tool_details`` tools, and the message adapter.
See :mod:`langgraph_progressive_tool_disclosure.middleware`.
"""

from .middleware import ProgressiveToolDisclosureMiddleware

__all__ = ["ProgressiveToolDisclosureMiddleware"]
