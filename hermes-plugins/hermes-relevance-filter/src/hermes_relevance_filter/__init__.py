"""Hermes Agent binding for Practice A — relevance filtering of oversized tool results.

The decision logic lives in ``context_core.relevance``; this package holds only the
``ContextEngine`` implementation (``on_turn_complete`` detect+store, ``select_context`` rewrite,
``retrieve_all_context`` tool), the Hermes OpenAI-message adapter, and the shared token-tracking base.
See :mod:`hermes_relevance_filter.engine`.

Ship as a Hermes context engine: this module mentions ``register_context_engine`` and ``ContextEngine``
so Hermes's directory discovery recognises it, and both the engine class and a ``register(ctx)`` hook
are exported.
"""

from .engine import RelevanceFilterEngine, register

#: Hermes directory discovery collects an exported ``ContextEngine`` instance named ``engine``.
engine = RelevanceFilterEngine()

__all__ = ["RelevanceFilterEngine", "engine", "register"]
