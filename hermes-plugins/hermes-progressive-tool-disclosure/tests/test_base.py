"""Tests for the shared ``BaseEngine`` token/budget mixin (Task 3.2)."""

from __future__ import annotations

from hermes_progressive_tool_disclosure._base import BaseEngine


class _Engine(BaseEngine):
    @property
    def name(self) -> str:
        return "t"


def test_token_buckets_tracked() -> None:
    e = _Engine()
    e.update_from_response({"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150})
    assert e.last_prompt_tokens == 120
    assert e.last_completion_tokens == 30
    assert e.last_total_tokens == 150


def test_update_from_response_total_fallback() -> None:
    e = _Engine()
    e.update_from_response({"prompt_tokens": 10, "completion_tokens": 5})
    assert e.last_total_tokens == 15


def test_update_model_recomputes_threshold() -> None:
    e = _Engine()
    e.update_model("claude", context_length=100_000)
    assert e.context_length == 100_000
    assert e.threshold_tokens == int(100_000 * e.threshold_percent)


def test_should_compress_follows_threshold() -> None:
    e = _Engine()
    e.update_model("m", context_length=1000)  # threshold = 750
    assert e.should_compress(800) is True
    assert e.should_compress(700) is False


def test_get_status_clamps_negative_sentinel() -> None:
    e = _Engine()
    e.update_model("m", context_length=1000)
    e.last_prompt_tokens = -1  # the "compression just ran" sentinel
    status = e.get_status()
    assert status["last_prompt_tokens"] == 0
    assert status["usage_percent"] == 0


def test_get_status_zero_context_length_no_divide() -> None:
    e = _Engine()
    e.context_length = 0
    assert e.get_status()["usage_percent"] == 0


def test_compress_noop_under_budget() -> None:
    e = _Engine()
    e.update_model("m", context_length=1_000_000)
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]
    assert e.compress(msgs, current_tokens=10) == msgs
    assert e.compression_count == 0


def test_compress_trims_old_tool_results_when_over_budget() -> None:
    e = _Engine()
    e.update_model("m", context_length=1000)  # threshold 750
    big = "x" * 5000
    msgs = (
        [{"role": "user", "content": "q"}] * 3  # protected head
        + [{"role": "tool", "tool_call_id": "a", "content": big}]
        + [{"role": "assistant", "content": "mid"}]
        + [{"role": "user", "content": "end"}] * 6  # protected tail
    )
    out = e.compress(msgs, current_tokens=800)
    assert "trimmed" in out[3]["content"]
    assert e.compression_count == 1


def test_on_session_reset_zeroes_counters() -> None:
    e = _Engine()
    e.update_from_response({"prompt_tokens": 99, "completion_tokens": 9, "total_tokens": 108})
    e.compression_count = 4
    e.on_session_reset()
    assert e.last_prompt_tokens == 0
    assert e.last_total_tokens == 0
    assert e.compression_count == 0
