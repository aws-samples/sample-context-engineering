"""Property tests for the truncated Description.

Feature: context-graph-plugin, Property 6: The truncated Description is a literal prefix within the ceiling.

Validates: Requirements 4.5.

``compose_description`` documents its contract as verifiable by ``startswith``: strip the trailing ``(+N numeric lines
omitted)`` line the implementation appends by design, and what remains is a literal prefix of the Description the same
Card yields with an unbounded budget. Two claims are asserted over every generated Card and every budget of at least one
token:

- the emitted Description, minus that omission line, is a literal prefix of the untruncated Description — no ellipse, no
  reordering, no line kept out of order;
- its estimated token count never exceeds the budget, measured the way ``compose_description`` filled it.
"""

import re

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from strands_context_graph.describe import _estimate_tokens, compose_description
from strands_context_graph.state import Card, ToolPair

UNBOUNDED = 100_000
"""A budget no generated Description can reach, for comparing against the untruncated form."""

OMISSION = re.compile(r"^\(\+\d+ numeric lines omitted\)$")
"""The trailing line the implementation appends when the selected lines overflow the budget (Requirement 4.6)."""

PROPERTY_SETTINGS = settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

# Text that stays inside what a Description carries: words, digits, currency and separators, and never a newline, which
# would make a numeric line indistinguishable from two.
text_strategy = st.text(
    alphabet=st.sampled_from(list("abcdefghijklmnopqrstuvwxyz 0123456789.,:|$-_")),
    min_size=0,
    max_size=60,
)

name_strategy = st.sampled_from(["http_request", "calculator", "read_file", "render_chart"])

reference_strategy = st.sampled_from(["mem_1_tu-1_0", "mem_2_tu-3_0", "reports/march.png", "data/q1.csv"])

content_type_strategy = st.sampled_from(
    [None, "text/plain", "text/csv; charset=utf-8", "application/json", "image/png"]
)


@st.composite
def card_strategy(draw: st.DrawFn) -> Card:
    """Build a Card carrying only the fields ``describe`` reads, across both kinds and both artifact shapes."""
    names = draw(st.lists(name_strategy, max_size=4))
    pairs = tuple(
        ToolPair(tool_use_id=f"tu-{index}", tool_name=name, tracking_ids=("a", "b"), consumed=True)
        for index, name in enumerate(names)
    )

    return Card(
        title=draw(text_strategy),
        kind=draw(st.sampled_from(["subject", "artifact"])),  # type: ignore[arg-type]
        turn=draw(st.integers(min_value=0, max_value=50)),
        dialogue_ids=(),
        evidence_ids=(),
        pairs=pairs,
        tool_names=frozenset(draw(st.lists(name_strategy, max_size=3))),
        references=tuple(draw(st.lists(reference_strategy, max_size=3))),
        numeric_lines=tuple(draw(st.lists(text_strategy.filter(lambda line: line.strip() != ""), max_size=8))),
        tags=(),
        description="",
        reference=draw(st.one_of(st.none(), reference_strategy)),
        content_type=draw(content_type_strategy),
        size_bytes=draw(st.one_of(st.none(), st.integers(min_value=0, max_value=10**9))),
    )


def _without_omission(description: str, complete: str) -> str:
    """Drop the trailing omission line, which the implementation appends and the untruncated form never carries.

    Only dropped when the Description is not already a prefix of ``complete``: a generated numeric line shaped like the
    omission line would otherwise be stripped as if the implementation had written it.
    """
    if complete.startswith(description):
        return description

    head, separator, last = description.rpartition("\n")

    return head if separator and OMISSION.match(last) else description


@given(card=card_strategy(), description_tokens=st.integers(min_value=1, max_value=200))
@PROPERTY_SETTINGS
def test_the_truncated_description_is_a_prefix_within_the_ceiling(card: Card, description_tokens: int) -> None:
    """Feature: context-graph-plugin, Property 6: The truncated Description is a literal prefix within the ceiling.

    Validates: Requirements 4.5.
    """
    complete = compose_description(card, UNBOUNDED)
    truncated = compose_description(card, description_tokens)

    kept = _without_omission(truncated, complete)

    assert complete.startswith(kept)
    assert _estimate_tokens(truncated) <= description_tokens
