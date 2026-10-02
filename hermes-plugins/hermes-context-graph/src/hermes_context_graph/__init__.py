"""Hermes Agent binding for Practice D — context graph.

The Card model, scan, scoring, matcher and per-call projection live in ``context_core.graph``; this
package holds only the ``ContextEngine`` implementation (``select_context`` projecting the message list,
``on_turn_complete`` recording a tool return as an addressable artifact), the
``expand_card`` / ``expand_artifact`` / ``find_context`` tools, the engine-held graph state and
reference store, and the Hermes OpenAI-message adapter. See :mod:`hermes_context_graph.engine`.
"""

from .engine import ContextGraphEngine, register

#: Hermes directory discovery collects an exported ``ContextEngine`` instance named ``engine``.
engine = ContextGraphEngine()

__all__ = ["ContextGraphEngine", "engine", "register"]
