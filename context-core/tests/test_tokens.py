"""``approximate_tokens``: the Strands SDK default heuristic, reproduced with no framework import.

The expected values are computed by hand from the SDK's rule (text ``ceil(chars/4)``, JSON ``ceil(dumps/2)``, binary
not counted), so the test pins the rule rather than the implementation. The Strands package carries the parity test
against the SDK function itself.
"""

from __future__ import annotations

import json
import math

from hypothesis import given
from hypothesis import strategies as st

from context_core.relevance import approximate_tokens


def test_text_is_chars_over_four_rounded_up() -> None:
    assert approximate_tokens([{"text": "a" * 9}]) == 3
    assert approximate_tokens([{"text": ""}]) == 0


def test_json_is_serialized_chars_over_two_rounded_up() -> None:
    obj = {"rows": [1, 2, 3]}
    assert approximate_tokens([{"json": obj}]) == math.ceil(len(json.dumps(obj)) / 2)


def test_binary_items_are_not_counted() -> None:
    image = {"image": {"format": "png", "source": {"bytes": b"\x00" * 10_000}}}
    document = {"document": {"format": "pdf", "name": "x", "source": {"bytes": b"\x00" * 10_000}}}
    assert approximate_tokens([image, document]) == 0


def test_an_unserializable_json_counts_zero() -> None:
    assert approximate_tokens([{"json": {"x": object()}}]) == 0


def test_an_item_with_both_keys_counts_both_and_non_dicts_count_nothing() -> None:
    assert approximate_tokens([{"text": "abcd", "json": [1]}, "stray", 7]) == 1 + math.ceil(len("[1]") / 2)


def test_the_8k_to_16k_band_is_where_text_and_json_disagree() -> None:
    """The reason this function exists: the same payload is ~2x the tokens as JSON than as text."""
    rows = [{"id": i, "v": "x" * 20} for i in range(400)]
    as_text = approximate_tokens([{"text": json.dumps(rows)}])
    as_json = approximate_tokens([{"json": rows}])
    assert as_json == 2 * as_text or as_json == 2 * as_text - 1


@given(st.lists(st.one_of(st.text(), st.dictionaries(st.text(max_size=5), st.integers(), max_size=5))))
def test_the_estimate_is_additive_over_items(parts: list) -> None:
    content = [{"text": p} if isinstance(p, str) else {"json": p} for p in parts]
    assert approximate_tokens(content) == sum(approximate_tokens([item]) for item in content)
