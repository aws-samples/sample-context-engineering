"""The binding's own contract: a thin Strands attachment over ``context_core.graph``.

- The adapter round-trips every block kind a Strands message can carry, ``tracking_id`` included.
- Importing the package registers the Strands locations of the three optional store bridges.
- The default matcher keeps the ``strands-agents`` AWS user-agent tag.
- ``src/`` holds only the binding's files, and only ``_compat`` imports ``strands``.
"""

from __future__ import annotations

import ast
import copy
import pathlib
from typing import Any

from context_core.graph.store import HOST_SYMBOLS
from hypothesis import given
from hypothesis import strategies as st

import strands_context_graph
from strands_context_graph import ContextGraph, EmbeddingSimilarityMatcher
from strands_context_graph._adapter import messages_from_neutral, messages_to_neutral, result_to_neutral
from strands_context_graph._compat import STRANDS_HOST_SYMBOLS

SRC = pathlib.Path(strands_context_graph.__file__).parent

_text = st.text(max_size=20)
_item = st.one_of(
    st.builds(lambda t: {"text": t}, _text),
    st.builds(lambda j: {"json": j}, st.dictionaries(_text, st.integers(), max_size=3)),
    st.builds(lambda b: {"image": {"format": "png", "source": {"bytes": b}}}, st.binary(max_size=8)),
)
_block = st.one_of(
    _item,
    st.builds(lambda i: {"toolUse": {"toolUseId": i, "name": "f", "input": {}}}, _text),
    st.builds(lambda i, c: {"toolResult": {"toolUseId": i, "status": "success", "content": c}}, _text, st.lists(_item)),
    st.builds(lambda t: {"reasoningContent": {"reasoningText": {"text": t, "signature": "s"}}}, _text),
    st.just({"cachePoint": {"type": "default"}}),
)
_message = st.builds(
    lambda role, content, extra: {"role": role, "content": content, **extra},
    st.sampled_from(["user", "assistant"]),
    st.lists(_block, max_size=4),
    st.fixed_dictionaries({}, optional={"tracking_id": _text, "metadata": st.just({"custom": {"pinned": True}})}),
)


@given(st.lists(_message, max_size=5))
def test_messages_round_trip_unchanged_with_tracking_id(messages: list[dict[str, Any]]) -> None:
    before = copy.deepcopy(messages)

    neutral = messages_to_neutral(messages)
    back = messages_from_neutral(neutral)

    assert back == before
    assert [m.get("tracking_id") for m in neutral] == [m.get("tracking_id") for m in before]
    assert back is not messages
    assert messages == before


def test_a_tool_result_is_handed_over_as_the_same_dict() -> None:
    result = {"toolUseId": "t", "status": "success", "content": [{"text": "x"}]}
    assert result_to_neutral(result) is result


def test_importing_the_package_registers_the_strands_store_bridges() -> None:
    for key, location in STRANDS_HOST_SYMBOLS.items():
        assert HOST_SYMBOLS[key] == location


class _Session:
    def __init__(self) -> None:
        self.tags: list[str] = []

    def client(self, service_name: str, config: Any = None, **kwargs: Any) -> object:
        self.tags.append(config.user_agent_extra)
        return object()


def test_the_default_matcher_keeps_the_strands_agents_tag() -> None:
    session = _Session()
    default = ContextGraph()._matcher_for()
    assert isinstance(default, EmbeddingSimilarityMatcher)

    EmbeddingSimilarityMatcher(boto_session=session)._ensure_client()

    assert session.tags == ["strands-agents"]


def test_src_holds_only_the_binding_files() -> None:
    assert sorted(p.name for p in SRC.iterdir() if p.name != "__pycache__") == [
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
    return names


def test_only_compat_imports_strands() -> None:
    offenders = [p.name for p in SRC.glob("*.py") if p.name != "_compat.py" and "strands" in _module_level_imports(p)]
    assert offenders == []
