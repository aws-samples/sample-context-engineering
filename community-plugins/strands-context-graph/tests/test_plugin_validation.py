"""Unit tests for construction-time configuration validation.

Every invalid value must be rejected at construction, before the instance can reach an agent, and every message must
name the parameter it is about so the developer does not have to guess which argument was wrong.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any

import pytest

from strands_context_graph.plugin import ContextGraph

_RATIOS = ("expand_threshold", "collapse_floor", "link_threshold")
_COUNTS = ("description_tokens", "tags_per_card", "min_cards")


class _ScoringMatcher:
    """Matcher-shaped object that inherits from nothing: validation is by member, not by type."""

    def score(self, query: str, documents: list[str]) -> list[float]:
        return [0.0] * len(documents)


class _NotAMatcher:
    """Carries ``score`` as data rather than as an operation."""

    score = 0.5


def test_defaults_are_the_documented_ones() -> None:
    graph = ContextGraph()

    assert graph.name == "strands:context-graph"
    assert graph._expand_threshold == 0.55
    assert graph._collapse_floor == 0.45
    assert graph._link_threshold == 0.50
    assert graph._description_tokens == 100
    assert graph._tags_per_card == 5
    assert graph._min_cards == 3
    assert graph._reuse_ttl_cycles == 5
    assert graph._body_budget is None
    # ``matcher=None`` stays observable as the configuration it was; the default is resolved on first need.
    assert graph._matcher is None
    assert graph._resolved_matcher is None


@pytest.mark.parametrize("parameter", _RATIOS)
@pytest.mark.parametrize("value", [True, False, "0.5", None, [0.5], float("nan"), math.inf, -0.01, 1.01])
def test_ratio_rejects_bool_non_numeric_non_finite_and_out_of_range(parameter: str, value: Any) -> None:
    with pytest.raises(ValueError, match=parameter):
        ContextGraph(**{parameter: value})


@pytest.mark.parametrize("parameter", _RATIOS)
@pytest.mark.parametrize("value", [0.0, 1.0, 0, 1, Fraction(1, 2)])
def test_ratio_accepts_any_finite_real_at_the_bounds(parameter: str, value: Any) -> None:
    # ``collapse_floor`` alone would violate the relation at 1.0, so pin the ceiling out of the way, letting the
    # parameter under test override it when it is the ceiling itself.
    graph = ContextGraph(**{"expand_threshold": 1.0, "collapse_floor": 0.0, parameter: value})

    assert getattr(graph, f"_{parameter}") == pytest.approx(float(value))


def test_collapse_floor_above_expand_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="collapse_floor.*expand_threshold"):
        ContextGraph(expand_threshold=0.4, collapse_floor=0.5)


def test_collapse_floor_equal_to_expand_threshold_is_accepted() -> None:
    graph = ContextGraph(expand_threshold=0.5, collapse_floor=0.5)

    assert graph._collapse_floor == graph._expand_threshold


@pytest.mark.parametrize("parameter", _COUNTS)
@pytest.mark.parametrize("value", [True, False, 0, -1, 2.5, "3", None])
def test_count_rejects_bool_non_integer_and_below_one(parameter: str, value: Any) -> None:
    with pytest.raises(ValueError, match=parameter):
        ContextGraph(**{parameter: value})


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "10"])
def test_body_budget_rejects_bool_non_integer_and_below_one(value: Any) -> None:
    with pytest.raises(ValueError, match="body_budget"):
        ContextGraph(body_budget=value)


@pytest.mark.parametrize("value", [None, 1, 4_000])
def test_body_budget_accepts_none_and_positive_integers(value: int | None) -> None:
    assert ContextGraph(body_budget=value)._body_budget == value


@pytest.mark.parametrize("value", [True, False, -1, 1.5, "5", None])
def test_reuse_ttl_cycles_rejects_bool_non_integer_and_negative(value: Any) -> None:
    with pytest.raises(ValueError, match="reuse_ttl_cycles"):
        ContextGraph(reuse_ttl_cycles=value)


def test_reuse_ttl_cycles_accepts_zero() -> None:
    # Zero is a configuration, not an absence: the Fed-Back Note is discarded at the end of the turn that created it.
    assert ContextGraph(reuse_ttl_cycles=0)._reuse_ttl_cycles == 0


def test_matcher_is_validated_by_member() -> None:
    matcher = _ScoringMatcher()

    assert ContextGraph(matcher=matcher)._matcher is matcher


@pytest.mark.parametrize("value", [_NotAMatcher(), object(), "matcher", 0.5])
def test_matcher_without_a_callable_score_is_rejected(value: Any) -> None:
    with pytest.raises(ValueError, match="score"):
        ContextGraph(matcher=value)


@pytest.mark.parametrize("value", ["", 7, b"graph", []])
def test_name_rejects_empty_and_non_string(value: Any) -> None:
    with pytest.raises(ValueError, match="name"):
        ContextGraph(name=value)


def test_name_override_is_adopted() -> None:
    assert ContextGraph(name="my:graph").name == "my:graph"


def test_failed_construction_never_reaches_plugin_setup() -> None:
    """A rejected configuration stops before ``Plugin.__init__``, so hook and tool discovery never runs.

    That is what "registers nothing on any agent" reduces to at construction time: there is no instance to hand to an
    agent, and the base class that would collect its handlers was never entered.
    """
    reached: list[bool] = []

    class _Probe(ContextGraph):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            reached.append(True)

    with pytest.raises(ValueError, match="expand_threshold"):
        _Probe(expand_threshold=1.5)

    assert reached == []
