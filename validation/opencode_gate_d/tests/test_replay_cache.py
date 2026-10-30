"""The cache model of the replay tool: checkpoint-only (matches the real bills) against the older ideal-prefix one."""

from __future__ import annotations

from opencode_gate_d.replay import _key_of, _run_cache


def _view(*keys: str, size: int = 100) -> tuple[tuple[str, ...], tuple[int, ...]]:
    return keys, tuple(size for _ in keys)


def test_a_history_that_only_appends_is_read_in_full_and_only_the_new_message_is_written():
    views = [_view("q1"), _view("q1", "a1"), _view("q1", "a1", "q2")]
    run = _run_cache(views, fixed_overhead_tokens=1_000)
    # request 1 writes its message and the prefix; 2 and 3 read everything before the one new message
    assert run == {"read_tokens": 2_000 + 100 + 200, "write_tokens": 1_000 + 100 + 100 + 100}
    assert run == _run_cache(views, fixed_overhead_tokens=1_000, checkpoints=False)


def test_an_edit_in_the_middle_is_a_write_from_the_system_prefix_unless_a_checkpoint_stood_there():
    # The second request replaces "b" by "x", but no earlier request ended right before "b": the only cached prefixes end at
    # the last two messages of request 1 ("c" and "b"), and neither matches.
    views = [_view("a", "b", "c"), _view("a", "x", "c", "d")]
    ideal = _run_cache(views, 0, checkpoints=False)
    real = _run_cache(views, 0)
    assert ideal["read_tokens"] == 100, "the ideal model gives credit for the common prefix 'a'"
    assert real["read_tokens"] == 0 and real["write_tokens"] == 300 + 400, (
        "the real one reads nothing but the system prefix"
    )


def test_an_edit_directly_after_the_message_that_ended_an_earlier_request_keeps_the_cache():
    # The first call of a turn ends at the user question; the next turn replaces everything after the question by a summary.
    views = [
        _view("q1"),  # first call of turn 1: it ends at the question, which is therefore a checkpoint
        _view("q1", "tool-call", "tool-result", "answer"),  # the turn's last call
        _view("q1", "summary", "q2"),  # first call of turn 2: the answer is gone, the summary stands after the question
    ]
    real = _run_cache(views, 0)
    ideal = _run_cache(views, 0, checkpoints=False)
    assert real == ideal == {"read_tokens": 100 + 100, "write_tokens": 100 + 300 + 200}


def test_a_second_edit_one_turn_later_costs_a_rewrite_that_the_ideal_model_does_not_see():
    # What v8 did: the evidence of a turn collapsed when it closed, its answer went one turn later, at a position where no
    # earlier request had ended.
    views = [
        _view("s", "q1"),
        _view("s", "q1", "e1", "f1"),
        _view("s", "q1", "s1", "f1", "q2"),  # evidence of turn 1 collapsed, answer kept
        _view("s", "q1", "s1", "f1", "q2", "e2", "f2"),
        _view("s", "q1", "s1", "q2", "s2", "f2", "q3"),  # the answer of turn 1 goes: no request ended after "s1"
    ]
    real = _run_cache(views, 0)
    ideal = _run_cache(views, 0, checkpoints=False)
    assert real == {"read_tokens": 1_100, "write_tokens": 1_400}
    assert ideal == {"read_tokens": 1_200, "write_tokens": 1_300}


def test_a_message_made_by_the_graph_is_named_by_its_text_too():
    first = {"id": "graph-collapse-a1", "role": "assistant", "parts": [{"type": "text", "text": "one"}]}
    second = {"id": "graph-collapse-a1", "role": "assistant", "parts": [{"type": "text", "text": "two"}]}
    stored = {"id": "msg_1", "role": "user", "parts": [{"type": "text", "text": "q"}]}
    assert _key_of(first) != _key_of(second)
    assert _key_of(first) == _key_of(dict(first))
    assert _key_of(stored) == "msg_1"


def test_calibration_scales_the_conversation_to_the_real_baseline_and_never_the_fixed_prefix():
    from opencode_gate_d.replay import ReplaySummary, RequestReplay, calibrated_dollars

    views = [_view("q1"), _view("q1", "a1"), _view("q1", "a1", "q2")]

    def summary_of(shown: list[tuple[tuple[str, ...], tuple[int, ...]]]) -> ReplaySummary:
        summary = ReplaySummary()
        for index, (keys, tokens) in enumerate(shown, start=1):
            summary.requests.append(
                RequestReplay(
                    index=index, messages=len(keys), operation="unchanged", mode=None, removed=0,
                    received_tokens=sum(tokens), sent_tokens=sum(tokens), block_chars=0, cards=0, pair_split=False,
                    view_tokens=tokens, view_keys=keys,
                )
            )  # fmt: skip
        return summary

    # a real baseline that read and wrote 2x the replay's conversation tokens, plus the fixed prefix on every call
    conversation = 100 + 200 + 300  # what the append-only baseline reads and writes, in replay tokens
    real_cache = 2 * conversation + 1_000 * 2 + 1_000  # prefix read twice and written once
    same = calibrated_dollars(
        summary_of(views), real_cache_tokens=real_cache, real_output_tokens=0, fixed_overhead_tokens=1_000,
        baseline_views=views,
    )  # fmt: skip
    assert (
        same["scale_to_real_tokens"] == 2.0 and same["dollars_change_pct"] == 0.0 and same["tokens_change_pct"] == 0.0
    )

    smaller = [_view("q1"), _view("q1", "s1"), _view("q1", "s1", "q2")]
    cheaper = calibrated_dollars(
        summary_of([(k, tuple(10 if key.startswith("s") else 100 for key in k)) for k, _ in smaller]),
        real_cache_tokens=real_cache, real_output_tokens=0, fixed_overhead_tokens=1_000, baseline_views=views,
    )  # fmt: skip
    assert cheaper["dollars_change_pct"] < 0 and cheaper["tokens_change_pct"] < 0
