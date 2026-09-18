"""Property tests for construction-time configuration validation.

Feature: context-graph-plugin, Property 11: Invalid configuration fails at construction and registers nothing.

Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 2.17.

``tests/test_plugin_validation.py`` pins the specific branches with hand-picked values; these properties cover the
universal claim over generated ones. Two directions are asserted:

- every invalid value, in any of the nine parameters and in any combination with otherwise valid ones, raises
  ``ValueError`` whose message names the parameter that was wrong, and ``Plugin.__init__`` is never entered, so no hook
  and no tool was ever discovered — which is what "registers nothing on any agent" reduces to at construction time
  (Requirement 2.17);
- every valid configuration constructs, keeps the values it was given, and reaches plugin setup exactly once with an
  empty hook list, since the graph registers what it needs in ``init_agent`` rather than by discovery.

``Plugin.__init__`` is counted by wrapping it for the duration of one example: the entry point is what distinguishes
"validation rejected the configuration" from "an instance exists that merely has not met an agent yet".
"""

import math
from collections.abc import Iterator
from contextlib import contextmanager
from decimal import Decimal
from fractions import Fraction
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from strands.plugins import Plugin

from strands_context_graph.plugin import ContextGraph

PROPERTY_SETTINGS = settings(max_examples=100, deadline=None)

RATIOS = ("expand_threshold", "collapse_floor", "link_threshold")
COUNTS = ("description_tokens", "tags_per_card", "min_cards")


class _ScoringMatcher:
    """Matcher-shaped object that inherits from nothing: the contract is the callable member, not the base class."""

    def score(self, query: str, documents: list[str]) -> list[float]:
        return [0.0] * len(documents)


class _ScoreAsData:
    """Carries ``score`` as a value rather than as an operation."""

    score = 0.5


@contextmanager
def _plugin_setup_calls() -> Iterator[list[object]]:
    """Count entries into ``Plugin.__init__`` for the duration of the block.

    Hook and tool discovery happens there and nowhere else, so an empty count is a direct reading of "nothing was
    registered" that does not depend on an agent existing.
    """
    entered: list[object] = []
    original = Plugin.__init__

    def counting(self: Plugin) -> None:
        entered.append(self)
        original(self)

    Plugin.__init__ = counting  # type: ignore[method-assign]
    try:
        yield entered
    finally:
        Plugin.__init__ = original  # type: ignore[method-assign]


# Values that are not finite reals in [0.0, 1.0]. Bools lead the list: they pass as numbers in Python, so ``True``
# silently meaning ``1.0`` is the failure mode the validator exists for.
invalid_ratios = st.one_of(
    st.sampled_from([True, False, None, "0.5", b"0.5", [0.5], {"v": 0.5}, 1 + 0j]),
    st.sampled_from([math.nan, math.inf, -math.inf, Decimal("NaN")]),
    st.floats(min_value=1.0000001, max_value=1e6, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-1e6, max_value=-0.0000001, allow_nan=False, allow_infinity=False),
)

# Values that are not integers greater than or equal to 1: bools, zero, negatives, floats (even integral ones), text.
invalid_counts = st.one_of(
    st.sampled_from([True, False, None, "3", b"3", [3], 0]),
    st.integers(max_value=0),
    st.floats(min_value=0.5, max_value=100.0, allow_nan=False, allow_infinity=False),
    st.sampled_from([Fraction(3, 2), Decimal("3")]),
)

# ``reuse_ttl_cycles`` floors at zero instead of one, so zero is valid and only negatives and non-integers are not.
invalid_ttls = st.one_of(
    st.sampled_from([True, False, None, "5", [5]]),
    st.integers(max_value=-1),
    st.floats(min_value=0.5, max_value=100.0, allow_nan=False, allow_infinity=False),
)

# ``body_budget`` admits ``None`` as the explicit absence of a ceiling, so ``None`` is absent from this list.
invalid_budgets = st.one_of(
    st.sampled_from([True, False, "4000", [4000], 0]),
    st.integers(max_value=0),
    st.floats(min_value=0.5, max_value=1e4, allow_nan=False, allow_infinity=False),
)

invalid_matchers = st.one_of(
    st.builds(_ScoreAsData),
    st.sampled_from([object(), "matcher", 0.5, [], {"score": lambda q, d: []}]),
)

invalid_names = st.one_of(st.sampled_from(["", 7, b"graph", [], True]))

valid_counts = st.integers(min_value=1, max_value=10_000)
valid_names = st.one_of(st.none(), st.text(min_size=1, max_size=30))
valid_matchers = st.one_of(st.none(), st.builds(_ScoringMatcher))

# A valid pair respects ``collapse_floor <= expand_threshold``; equality is valid, so the floor is drawn up to the
# ceiling rather than below it.
valid_ratio_pairs = st.tuples(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
).map(lambda pair: (max(pair), min(pair)))


@st.composite
def valid_configurations(draw: st.DrawFn) -> dict[str, Any]:
    """A configuration whose every value is inside what the validators accept."""
    expand_threshold, collapse_floor = draw(valid_ratio_pairs)
    return {
        "expand_threshold": expand_threshold,
        "collapse_floor": collapse_floor,
        "link_threshold": draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        "description_tokens": draw(valid_counts),
        "tags_per_card": draw(valid_counts),
        "min_cards": draw(valid_counts),
        "body_budget": draw(st.one_of(st.none(), st.integers(min_value=1, max_value=100_000))),
        "reuse_ttl_cycles": draw(st.integers(min_value=0, max_value=1_000)),
        "matcher": draw(valid_matchers),
        "name": draw(valid_names),
    }


@st.composite
def invalid_configurations(draw: st.DrawFn) -> tuple[dict[str, Any], str]:
    """A configuration with at least one invalid value, paired with the parameter the message must name.

    The rest of the configuration is drawn valid so the assertion is about the injected fault and not about whichever
    parameter the constructor happens to check first.
    """
    configuration = draw(valid_configurations())
    parameter = draw(
        st.sampled_from([*RATIOS, *COUNTS, "body_budget", "reuse_ttl_cycles", "matcher", "name", "_relation"])
    )

    if parameter == "_relation":
        # The one relational check: a floor above the ceiling. Both ends are valid ratios on their own.
        ceiling = draw(st.floats(min_value=0.0, max_value=0.99, allow_nan=False, allow_infinity=False))
        floor = draw(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False).filter(
                lambda value: value > ceiling
            )
        )
        configuration["expand_threshold"] = ceiling
        configuration["collapse_floor"] = floor
        return configuration, "collapse_floor"

    faults = {
        "body_budget": invalid_budgets,
        "reuse_ttl_cycles": invalid_ttls,
        "matcher": invalid_matchers,
        "name": invalid_names,
    }
    fault = faults.get(parameter, invalid_ratios if parameter in RATIOS else invalid_counts)
    configuration[parameter] = draw(fault)

    if parameter in ("expand_threshold", "collapse_floor"):
        # Keep the relation out of the way, so the failure that surfaces is the injected fault rather than the
        # comparison against the other end of the pair.
        configuration["expand_threshold" if parameter == "collapse_floor" else "collapse_floor"] = 0.5

    # The matcher message names the missing member rather than the parameter (Requirement 2.10).
    return configuration, "score" if parameter == "matcher" else parameter


@given(case=invalid_configurations())
@PROPERTY_SETTINGS
def test_invalid_configuration_raises_naming_the_parameter_and_registers_nothing(
    case: tuple[dict[str, Any], str],
) -> None:
    """Feature: context-graph-plugin, Property 11: Invalid configuration fails at construction and registers nothing.

    Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 2.17.
    """
    configuration, named = case

    with _plugin_setup_calls() as entered:
        with pytest.raises(ValueError) as failure:
            ContextGraph(**configuration)

    assert named in str(failure.value)
    # Plugin setup was never entered, so no hook and no tool was discovered for this configuration.
    assert entered == []


@given(configuration=valid_configurations())
@PROPERTY_SETTINGS
def test_valid_configuration_always_constructs_and_keeps_its_values(configuration: dict[str, Any]) -> None:
    """Feature: context-graph-plugin, Property 11: Invalid configuration fails at construction and registers nothing.

    Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 2.17.
    """
    with _plugin_setup_calls() as entered:
        graph = ContextGraph(**configuration)

    assert [id(instance) for instance in entered] == [id(graph)]
    # The graph registers in ``init_agent`` rather than by discovery, so setup leaves the hook list empty.
    assert graph.hooks == []

    for parameter in (*RATIOS, *COUNTS, "body_budget", "reuse_ttl_cycles"):
        assert getattr(graph, f"_{parameter}") == configuration[parameter]
    assert graph._matcher is configuration["matcher"]
    assert graph.name == (configuration["name"] or "strands:context-graph")
    # Nothing resolves a matcher at construction: the default is built on first need, not here (Requirement 2.16).
    assert graph._resolved_matcher is None
