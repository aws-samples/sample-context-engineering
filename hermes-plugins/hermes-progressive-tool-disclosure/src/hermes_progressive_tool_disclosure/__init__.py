"""Hermes Agent binding for Practice B — progressive tool disclosure.

The lexical index and catalog/fold logic live in ``context_core.disclosure``; this package holds only
the ``ContextEngine`` implementation (``select_context`` catalog injection + fold, ``find_tools`` /
``get_tool_details`` tools), the Hermes OpenAI-message adapter, and the shared token-tracking base.
See :mod:`hermes_progressive_tool_disclosure.engine`.
"""

from .engine import ProgressiveToolDisclosureEngine, register

#: Hermes directory discovery collects an exported ``ContextEngine`` instance named ``engine``.
engine = ProgressiveToolDisclosureEngine()

__all__ = ["ProgressiveToolDisclosureEngine", "engine", "register"]
