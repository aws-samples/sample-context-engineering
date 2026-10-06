"""SDK parity: the core's copies of four Strands rules agree with the SDK originals, while the SDK still ships them.

The binding stops importing ``_is_user_turn``, ``_get_tool_use_ids``, ``is_pinned`` and the injection fold from the
SDK and uses the core's framework-free copies instead (design R2, R3). That trades a hard coupling for a drift risk,
and this file is the guard: on Hypothesis-generated conversations each core function must return what the SDK
function returns. When the SDK renames or drops a symbol the test SKIPS with a message naming it, so the suite says
"the SDK moved" instead of failing for an unrelated reason.

One difference is deliberate and pinned here rather than hidden: the core's fold carries ``tracking_id`` onto the
folded message (the next projection's scan reads Cards off that identity), the SDK's does not. The SDK strips
``tracking_id`` before ``Model.stream``, so the provider request is the same either way.
"""

from __future__ import annotations

import asyncio
import copy
import importlib
from dataclasses import dataclass, field, replace
from typing import Any

import pytest
from context_core.graph.cards import _is_user_turn as core_is_user_turn
from context_core.graph.projection import _fold_into_last_user_message as core_fold
from context_core.graph.removal import _get_tool_use_ids as core_get_tool_use_ids
from context_core.graph.removal import is_pinned as core_is_pinned
from hypothesis import given, settings
from hypothesis import strategies as st


def _sdk(module: str, name: str) -> Any:
    """Import an SDK symbol, or skip the test with the reason it is gone."""
    try:
        return getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError):
        pytest.skip(f"the SDK no longer ships {module}.{name}: re-check the core copy against the new SDK rule")


# ---- conversations --------------------------------------------------------------------------------

_ids = st.sampled_from(["t1", "t2", "t3", ""])
_text_block = st.builds(lambda t: {"text": t}, st.sampled_from(["", "hi", "\nlead", "x y"]))
_use_block = st.builds(lambda i: {"toolUse": {"toolUseId": i, "name": "f", "input": {}}}, _ids)
_result_block = st.builds(
    lambda i: {"toolResult": {"toolUseId": i, "status": "success", "content": [{"text": "r"}]}}, _ids
)
_metadata = st.one_of(
    st.just({}),
    st.builds(lambda p: {"metadata": {"custom": {"pinned": p}}}, st.sampled_from([True, False, "yes"])),
    st.just({"metadata": {"usage": {"inputTokens": 1}}}),
)
_message = st.builds(
    lambda role, content, meta, tid: {"role": role, "content": content, **meta, **tid},
    st.sampled_from(["user", "assistant"]),
    st.lists(st.one_of(_text_block, _use_block, _result_block), max_size=3),
    _metadata,
    st.one_of(st.just({}), st.builds(lambda n: {"tracking_id": f"m{n}"}, st.integers(0, 9))),
)
_conversation = st.lists(_message, max_size=6)


def _without_tracking(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in m.items() if k != "tracking_id"} for m in messages]


# ---- the three predicates -------------------------------------------------------------------------


@settings(deadline=None)
@given(_conversation)
def test_is_user_turn_matches_the_sdk(messages: list[dict[str, Any]]) -> None:
    sdk = _sdk("strands.injection._message_injection", "_is_user_turn")
    assert core_is_user_turn(messages) == sdk(messages)


@settings(deadline=None)
@given(_message)
def test_get_tool_use_ids_matches_the_sdk(message: dict[str, Any]) -> None:
    sdk = _sdk("strands.agent.conversation_manager.compression.pin_message", "_get_tool_use_ids")
    assert core_get_tool_use_ids(message) == sdk(message)


@settings(deadline=None)
@given(_conversation.filter(bool), st.integers(0, 5))
def test_is_pinned_matches_the_sdk(messages: list[dict[str, Any]], index: int) -> None:
    sdk = _sdk("strands.agent.conversation_manager.compression.pin_message", "is_pinned")
    index %= len(messages)
    assert core_is_pinned(messages, index) == sdk(messages, index)


# ---- the fold -------------------------------------------------------------------------------------

_fold_text = st.sampled_from(["<collapsed_turns>block</collapsed_turns>", "\nleading newline", "x"])


@settings(deadline=None)
@given(_conversation, _fold_text)
def test_the_fold_primitive_matches_the_sdk(messages: list[dict[str, Any]], text: str) -> None:
    sdk = _sdk("strands.injection._message_injection", "_fold_into_last_user_message")
    before = copy.deepcopy(messages)

    core_messages, core_blocks = core_fold(messages, text)
    sdk_messages, sdk_blocks = sdk(messages, text)

    assert core_blocks == sdk_blocks
    assert _without_tracking(list(core_messages)) == _without_tracking(list(sdk_messages))
    assert messages == before


@settings(deadline=None)
@given(_conversation, _fold_text)
def test_the_core_fold_keeps_tracking_id_where_the_sdk_drops_it(messages: list[dict[str, Any]], text: str) -> None:
    """The one deliberate difference, pinned: only the folded message's ``tracking_id`` differs."""
    sdk = _sdk("strands.injection._message_injection", "_fold_into_last_user_message")
    core_messages, _ = core_fold(messages, text)
    sdk_messages, _ = sdk(messages, text)

    for original, core_m, sdk_m in zip(messages, core_messages, sdk_messages, strict=True):
        if core_m is original:
            assert sdk_m is original
            continue
        assert core_m.get("tracking_id") == original.get("tracking_id")
        assert "tracking_id" not in sdk_m


@dataclass
class _Agent:
    state: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Context:
    """The two fields of ``InvokeModelContext`` the SDK fold reads and writes, plus the agent it reads."""

    agent: _Agent
    messages: list[dict[str, Any]]
    dynamic_trailing_blocks: int = 0


@settings(max_examples=60, deadline=None)
@given(_conversation, _fold_text, st.integers(0, 2))
def test_the_everyturn_injection_middleware_matches_the_core_fold(
    messages: list[dict[str, Any]], text: str, already: int
) -> None:
    """What the Strands delivery used end to end: ``_create_injection_middleware(trigger="everyTurn")``.

    Its counter adds to whatever ``dynamic_trailing_blocks`` the context already carried, so the binding must add the
    core's count to the incoming value rather than overwrite it.
    """
    create = _sdk("strands.injection._message_injection", "_create_injection_middleware")
    handler = create(lambda _context: text, trigger="everyTurn")
    context = _Context(_Agent(), messages, already)

    folded_context = asyncio.run(handler(context))
    core_messages, core_blocks = core_fold(messages, text)

    expected = replace(context, messages=list(core_messages), dynamic_trailing_blocks=already + core_blocks)
    assert folded_context.dynamic_trailing_blocks == expected.dynamic_trailing_blocks
    assert _without_tracking(list(folded_context.messages)) == _without_tracking(expected.messages)
