"""SDK parity: the core's copies of four Strands rules agree with the SDK originals, while the SDK still ships them.

The binding stops importing ``_is_user_turn``, ``_get_tool_use_ids``, ``is_pinned`` and the injection fold from the
SDK and uses the core's framework-free copies instead (design R2, R3). That trades a hard coupling for a drift risk,
and this file is the guard: on Hypothesis-generated conversations each core function must return what the SDK
function returns. When the SDK renames or drops a symbol the test SKIPS with a message naming it, so the suite says
"the SDK moved" instead of failing for an unrelated reason.

The SDK's fold changed in Strands 1.65, and the core keeps the earlier rule on purpose, so every Strands version
in the supported range produces the same request. Two rules are probed from the installed SDK rather than read off
its version number (:func:`_sdk_fold_rules`), and each test asserts the core against what that SDK does:

- ``tracking_id``. The core's fold always carries it onto the folded message (the next projection's scan reads
  Cards off that identity). The SDK dropped it before 1.65 and keeps it from 1.65. The SDK strips ``tracking_id``
  before ``Model.stream``, so the provider request is the same either way.
- The ``\\n\\n`` separator. The core puts it in front of the folded text whenever the target message has content,
  as the SDK did before 1.65. From 1.65 the SDK puts it there only when the target has a *text* block, so a user
  message holding only a ``toolResult`` gets the text with no separator. The difference is that one prefix and
  nothing else, and it is what the tests normalise when the installed SDK uses the newer rule.
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


@dataclass(frozen=True)
class _FoldRules:
    """How the installed SDK's fold behaves on the two points where Strands 1.65 changed it."""

    separates_non_text_target: bool
    keeps_tracking_id: bool


def _sdk_fold_rules(sdk_fold: Any) -> _FoldRules:
    """Probe the SDK fold once on a user message that holds only a tool result and carries a ``tracking_id``."""
    target = {
        "role": "user",
        "content": [{"toolResult": {"toolUseId": "t1", "status": "success", "content": [{"text": "r"}]}}],
        "tracking_id": "m0",
    }
    folded, _ = sdk_fold([target], "x")
    return _FoldRules(
        separates_non_text_target=folded[0]["content"][-1]["text"].startswith("\n\n"),
        keeps_tracking_id="tracking_id" in folded[0],
    )


def _as_the_sdk_would(
    original: list[dict[str, Any]], folded: list[dict[str, Any]], text: str, rules: _FoldRules
) -> list[dict[str, Any]]:
    """The core's folded list, with the separator rewritten to the installed SDK's rule. Nothing else changes."""
    if rules.separates_non_text_target or text.startswith("\n"):
        return folded
    result = list(folded)
    for index, (before, after) in enumerate(zip(original, folded, strict=True)):
        if after is before:
            continue
        has_text = any("text" in block for block in before["content"])
        if before["content"] and not has_text:
            last = after["content"][-1]
            result[index] = {**after, "content": [*after["content"][:-1], {"text": last["text"].removeprefix("\n\n")}]}
    return result


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
    rules = _sdk_fold_rules(sdk)
    before = copy.deepcopy(messages)

    core_messages, core_blocks = core_fold(messages, text)
    sdk_messages, sdk_blocks = sdk(messages, text)

    assert core_blocks == sdk_blocks
    expected = _as_the_sdk_would(messages, list(core_messages), text, rules)
    assert _without_tracking(expected) == _without_tracking(list(sdk_messages))
    assert messages == before


@settings(deadline=None)
@given(_conversation, _fold_text)
def test_the_core_fold_always_keeps_tracking_id(messages: list[dict[str, Any]], text: str) -> None:
    """The core keeps the folded message's ``tracking_id`` on every SDK; the SDK does so only from 1.65."""
    sdk = _sdk("strands.injection._message_injection", "_fold_into_last_user_message")
    rules = _sdk_fold_rules(sdk)
    core_messages, _ = core_fold(messages, text)
    sdk_messages, _ = sdk(messages, text)

    for original, core_m, sdk_m in zip(messages, core_messages, sdk_messages, strict=True):
        if core_m is original:
            assert sdk_m is original
            continue
        assert core_m.get("tracking_id") == original.get("tracking_id")
        if rules.keeps_tracking_id:
            assert sdk_m.get("tracking_id") == original.get("tracking_id")
        else:
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
    rules = _sdk_fold_rules(_sdk("strands.injection._message_injection", "_fold_into_last_user_message"))
    handler = create(lambda _context: text, trigger="everyTurn")
    context = _Context(_Agent(), messages, already)

    folded_context = asyncio.run(handler(context))
    core_messages, core_blocks = core_fold(messages, text)
    as_sdk = _as_the_sdk_would(messages, list(core_messages), text, rules)

    expected = replace(context, messages=as_sdk, dynamic_trailing_blocks=already + core_blocks)
    assert folded_context.dynamic_trailing_blocks == expected.dynamic_trailing_blocks
    assert _without_tracking(list(folded_context.messages)) == _without_tracking(expected.messages)
