"""LangGraph binding for Practice A — relevance filtering of oversized tool results.

The decision logic lives in ``context_core.relevance``; this package holds only the ``create_agent``
middleware attachment (``wrap_tool_call`` + ``after_agent``), the message adapter, and the
``retrieve_all_context`` tool. See :mod:`langgraph_relevance_filter.middleware`.
"""

from .middleware import RelevanceFilterMiddleware

__all__ = ["RelevanceFilterMiddleware"]
