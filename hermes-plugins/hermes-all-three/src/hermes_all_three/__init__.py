"""Hermes Agent binding: the composed ``ContextEngine`` applying all three practices.

Hermes is single-select, so "all three" is one engine composing relevance filtering, progressive tool
disclosure and the context graph over the unchanged ``context-core``, reusing the three sibling
packages' engines. See :mod:`hermes_all_three.engine`.
"""

from .engine import AllThreeEngine, register

#: Hermes directory discovery collects an exported ``ContextEngine`` instance named ``engine``.
engine = AllThreeEngine()

__all__ = ["AllThreeEngine", "engine", "register"]
