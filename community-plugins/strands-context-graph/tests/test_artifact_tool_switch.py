"""The artifact tool's switch, and the published tool names the disclosure wiring is derived from.

Both exist because of a measured failure of COMPOSITION, not of either plugin alone.

``include_artifact_tool`` closes an API gap: the sibling ``RelevanceFilter`` has always been able to drop its own
retrieval tool through a constructor parameter, while this plugin could only be stripped from the outside by mutating
``_tools`` -- which is what this repository's own benchmark harness did, in sample code, against a private attribute.

``retrieval_tool_names`` closes the other half. ``ProgressiveToolDisclosure`` reduces an unexposed tool to a catalog
entry with an EMPTY ``inputSchema``, and every retrieval tool here needs arguments, so a hidden one is called with
nothing and cancelled before it runs. The names therefore belong in that plugin's ``always_available``, and a caller who
has to hard-code them will get it wrong the first time a tool is renamed or excluded.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from test_plugin_init_agent import build

from strands_context_graph import ContextGraph

ALL_THREE = {"expand_card", "expand_artifact", "find_context"}
"""What the plugin registers by default (Requirement 1.2)."""


class _Matcher:
    """Scores nothing: wiring never calls the matcher."""

    def score(self, question: str, descriptions: Sequence[str]) -> Sequence[float]:
        """Return one similarity per Description.

        Args:
            question: Ignored.
            descriptions: The Descriptions.

        Returns:
            A fixed similarity each.
        """
        return [0.5] * len(descriptions)


def registered(plugin: ContextGraph) -> set[str]:
    """Wire the plugin to a real agent and return the retrieval tools the agent's registry gained.

    Asserting on the agent's own registry rather than on the plugin's list is what makes this a test of
    registration instead of a test of bookkeeping.

    Args:
        plugin: The plugin to wire.

    Returns:
        Names of this plugin's retrieval tools present in the agent's tool registry.
    """
    agent = build(plugin=plugin)
    return ALL_THREE & set(agent.tool_registry.registry)


# --- the switch ------------------------------------------------------------------------------------


def test_all_three_tools_are_registered_by_default() -> None:
    """The default is unchanged: excluding a tool is opt-in, so an existing caller sees no difference."""
    plugin = ContextGraph(matcher=_Matcher())

    assert registered(plugin) == ALL_THREE
    assert set(plugin.retrieval_tool_names) == ALL_THREE


def test_excluding_the_artifact_tool_drops_exactly_it() -> None:
    """The conversation-facing tools are untouched: they do a job no offloader does."""
    plugin = ContextGraph(matcher=_Matcher(), include_artifact_tool=False)

    assert registered(plugin) == {"expand_card", "find_context"}
    assert set(plugin.retrieval_tool_names) == {"expand_card", "find_context"}


def test_the_exclusion_is_idempotent_across_agents() -> None:
    """One instance wired to two agents must not strip twice and must not restore the tool."""
    plugin = ContextGraph(matcher=_Matcher(), include_artifact_tool=False)
    registered(plugin)

    assert registered(plugin) == {"expand_card", "find_context"}


def test_an_excluded_tool_is_not_advertised_to_the_model() -> None:
    """The half that made the original bug: guidance naming a tool the registry does not hold.

    Measured before the guidance read the registered set, a run produced ``tool not found in registry`` five times for
    exactly this reason -- the model was told to call the tool the harness had removed.
    """
    from strands_context_graph.compaction import guidance

    plugin = ContextGraph(matcher=_Matcher(), include_artifact_tool=False)
    registered(plugin)

    text = guidance(set(plugin.retrieval_tool_names))

    assert "expand_artifact" not in text
    assert "expand_card" in text
    assert "find_context" in text


@pytest.mark.parametrize("bad", [0, 1, "yes", None, ()])
def test_the_switch_rejects_a_non_boolean(bad: object) -> None:
    """Validated like every other constructor argument, and before anything is assigned.

    ``0`` and ``1`` are rejected on purpose: they would work by accident and hide a caller's type error.

    Args:
        bad: A value that is not a bool.
    """
    with pytest.raises(ValueError, match="include_artifact_tool"):
        ContextGraph(include_artifact_tool=bad)  # type: ignore[arg-type]


# --- the published names ---------------------------------------------------------------------------


def test_the_names_are_what_the_disclosure_wiring_needs() -> None:
    """The documented composition: derive the always-available list instead of hard-coding it."""
    plugin = ContextGraph(matcher=_Matcher(), include_artifact_tool=False)
    registered(plugin)

    always_available = [*plugin.retrieval_tool_names, "retrieve_context"]

    assert always_available == ["expand_card", "find_context", "retrieve_context"]


def test_the_names_are_read_at_call_time_not_fixed_at_construction() -> None:
    """A caller that de-registers by hand still gets a truthful answer, which is what the guidance depends on."""
    plugin = ContextGraph(matcher=_Matcher())
    registered(plugin)
    plugin._tools = [tool for tool in plugin._tools if tool.tool_name != "find_context"]

    assert set(plugin.retrieval_tool_names) == {"expand_card", "expand_artifact"}


def test_the_names_are_a_tuple_so_a_caller_cannot_mutate_the_registration() -> None:
    """Published for reading. Handing out the live list would let a caller change what is registered."""
    plugin = ContextGraph(matcher=_Matcher())
    registered(plugin)

    assert isinstance(plugin.retrieval_tool_names, tuple)
