"""The binding's own contract: what makes this package a thin Strands attachment over ``context_core.relevance``.

- The adapter round-trips every block kind a Strands message can carry, unchanged and in position.
- AWS calls stay attributed to ``strands-agents``, as before the decision logic moved to the core.
- ``src/`` holds only the binding's files, and only ``_compat`` imports ``strands``.
"""

from __future__ import annotations

import ast
import copy
import pathlib
from typing import Any

from hypothesis import given
from hypothesis import strategies as st

import strands_relevance_filter
from strands_relevance_filter import BedrockReranker, S3Store
from strands_relevance_filter._adapter import (
    messages_from_neutral,
    messages_to_neutral,
    result_from_neutral,
    result_to_neutral,
)

SRC = pathlib.Path(strands_relevance_filter.__file__).parent

# ---- every block kind a Strands message can carry ---------------------------------------------------

_text = st.text(max_size=20)
_json = st.recursive(
    st.one_of(st.none(), st.booleans(), st.integers(), _text),
    lambda inner: st.one_of(st.lists(inner, max_size=3), st.dictionaries(_text, inner, max_size=3)),
    max_leaves=6,
)
_result_item = st.one_of(
    st.builds(lambda t: {"text": t}, _text),
    st.builds(lambda j: {"json": j}, _json),
    st.builds(lambda b: {"image": {"format": "png", "source": {"bytes": b}}}, st.binary(max_size=8)),
    st.builds(lambda b: {"document": {"format": "pdf", "name": "doc", "source": {"bytes": b}}}, st.binary(max_size=8)),
)
_tool_result = st.fixed_dictionaries(
    {
        "toolUseId": _text,
        "status": st.sampled_from(["success", "error"]),
        "content": st.lists(_result_item, max_size=3),
    }
)
_block = st.one_of(
    _result_item,
    st.builds(lambda i, n, j: {"toolUse": {"toolUseId": i, "name": n, "input": j}}, _text, _text, _json),
    st.builds(lambda r: {"toolResult": r}, _tool_result),
    st.builds(lambda t, s: {"reasoningContent": {"reasoningText": {"text": t, "signature": s}}}, _text, _text),
    st.just({"cachePoint": {"type": "default"}}),
)
_message = st.builds(
    lambda role, content, metadata: {"role": role, "content": content, **metadata},
    st.sampled_from(["user", "assistant"]),
    st.lists(_block, max_size=4),
    st.one_of(st.just({}), st.builds(lambda u: {"metadata": {"usage": {"inputTokens": u}}}, st.integers(0, 99))),
)


@given(st.lists(_message, max_size=5))
def test_messages_round_trip_unchanged_and_unmutated(messages: list[dict[str, Any]]) -> None:
    before = copy.deepcopy(messages)

    back = messages_from_neutral(messages_to_neutral(messages))

    assert back == before
    assert back is not messages
    assert messages == before


@given(_tool_result)
def test_tool_results_round_trip_unchanged(result: dict[str, Any]) -> None:
    before = copy.deepcopy(result)
    assert result_from_neutral(result_to_neutral(result)) == before


# ---- AWS attribution ------------------------------------------------------------------------------


class _Session:
    def __init__(self) -> None:
        self.tags: list[str] = []

    def client(self, service_name: str, config: Any = None, **kwargs: Any) -> object:
        self.tags.append(config.user_agent_extra)
        return object()


def test_the_exported_aws_classes_keep_the_strands_agents_tag() -> None:
    session = _Session()
    BedrockReranker(model_id="arn:aws:bedrock:us-east-1::foundation-model/x", boto_session=session)
    S3Store(bucket="b", boto_session=session)
    assert session.tags == ["strands-agents", "strands-agents"]


# ---- the binding's shape --------------------------------------------------------------------------


def test_src_holds_only_the_binding_files() -> None:
    assert sorted(p.name for p in SRC.iterdir() if not p.name.startswith("__pycache__")) == [
        "__init__.py",
        "_adapter.py",
        "_compat.py",
        "plugin.py",
        "py.typed",
    ]


def _module_level_imports(path: pathlib.Path) -> set[str]:
    """Top-level module names imported at module level, outside ``if TYPE_CHECKING:``."""
    names: set[str] = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
        elif isinstance(node, ast.Try):
            for inner in node.body:
                if isinstance(inner, ast.ImportFrom) and inner.module:
                    names.add(inner.module.split(".")[0])
    return names


def test_only_compat_imports_strands() -> None:
    offenders = [p.name for p in SRC.glob("*.py") if p.name != "_compat.py" and "strands" in _module_level_imports(p)]
    assert offenders == []
