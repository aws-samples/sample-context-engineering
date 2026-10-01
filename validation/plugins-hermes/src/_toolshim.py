"""Framework-neutral replacement for ``langchain_core.tools.tool`` and ``convert_to_openai_tool``.

The Hermes harness runs offline (no LangChain, no Bedrock), so the fat tool suite is declared with a
tiny local decorator instead of LangChain's. It keeps every tool *body* byte-identical to the Strands
and LangGraph harnesses — only the declaration changes — so ``ground_truth.py`` and ``accuracy.py``
remain reusable unchanged.

A decorated tool is callable (so the mock runner can execute it), carries ``.name`` and ``.description``,
and exposes an OpenAI function-tool schema via :func:`convert_to_openai_tool` built from the signature and
the Google-style docstring (name + description + a best-effort parameter object).
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable

__all__ = ["tool", "convert_to_openai_tool", "NeutralTool"]

_PY_TO_JSON = {"str": "string", "int": "integer", "float": "number", "bool": "boolean", "list": "array", "dict": "object"}


class NeutralTool:
    """A callable tool carrying the metadata the harness and the OpenAI schema builder read."""

    def __init__(self, func: Callable[..., Any], name: str | None = None) -> None:
        functools.update_wrapper(self, func)
        self.func = func
        self.name = name or func.__name__
        self.description = (inspect.getdoc(func) or "").strip()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)


def tool(func: Callable[..., Any] | None = None, *, name: str | None = None) -> Any:
    """Decorator: wrap a plain function as a :class:`NeutralTool`. Usable bare or with ``name=``."""
    if func is None:
        return lambda f: NeutralTool(f, name=name)
    return NeutralTool(func, name=name)


def _json_type(annotation: Any) -> str:
    raw = getattr(annotation, "__name__", str(annotation)).lower()
    for key, mapped in _PY_TO_JSON.items():
        if key in raw:
            return mapped
    return "string"


def convert_to_openai_tool(built: Any) -> dict[str, Any]:
    """Build an OpenAI function-tool schema from a :class:`NeutralTool` (or a plain callable)."""
    func = getattr(built, "func", built)
    name = getattr(built, "name", getattr(func, "__name__", "tool"))
    description = getattr(built, "description", (inspect.getdoc(func) or "").strip())
    sig = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for pname, param in sig.parameters.items():
        if pname in {"self", "args", "kwargs"}:
            continue
        properties[pname] = {"type": _json_type(param.annotation)}
        if param.default is inspect.Parameter.empty:
            required.append(pname)
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }
