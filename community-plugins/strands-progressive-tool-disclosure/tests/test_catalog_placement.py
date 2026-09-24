"""The catalog's PLACEMENT: in ``tool_specs`` as entries, or in the system prompt as prose.

``catalog_in_system_prompt`` moves the catalog off the tool schema. What has to hold is a partition: a
name gets a full specification in ``tool_specs`` or a line in the prompt block, never both and never
neither. These tests assert that partition, the three shapes ``SystemPrompt`` can arrive in, and the two
cases where the mode must do nothing at all.
"""

from __future__ import annotations

from typing import Any

import pytest
from strands_progressive_tool_disclosure.plugin import (
    FIND_TOOLS_NAME,
    _append_to_system_prompt,
    _catalog_prompt_block,
)

from tests.test_projection import TOOL_NAMES, _names, _project_once

_BASE_PROMPT = "You are a helpful assistant."

# The agent fixture registers a fixed pool, and a name outside it has no specification to project.
_ALPHA, _BETA, _GAMMA, _DELTA, _EPSILON = TOOL_NAMES


def _prompt_text(system_prompt: Any) -> str:
    """Flatten a ``SystemPrompt`` of any shape to one searchable string."""
    if system_prompt is None:
        return ""
    if isinstance(system_prompt, str):
        return system_prompt
    return "\n".join(block.get("text", "") for block in system_prompt)


def _listed_names(system_prompt: Any) -> set[str]:
    """Names the prompt block lists, read back off its ``- name: description`` lines."""
    return {
        line[2:].split(":", 1)[0]
        for line in _prompt_text(system_prompt).splitlines()
        if line.startswith("- ") and ":" in line
    }


def test_the_catalog_leaves_the_tool_schema_and_arrives_in_the_system_prompt() -> None:
    """Only callable tools stay in ``tool_specs``; every other name is listed in the prompt."""
    _, _, context, result = _project_once(
        names=[_ALPHA, _BETA, _GAMMA, _DELTA],
        exposed=[_ALPHA],
        always_available=[_BETA],
        catalog_in_system_prompt=True,
    )

    projected = set(_names(result.tool_specs))
    assert projected == {FIND_TOOLS_NAME, _ALPHA, _BETA}

    # The partition: what left the schema is exactly what the prompt now carries.
    incoming = set(_names(context.tool_specs))
    assert _listed_names(result.system_prompt) == incoming - projected == {_GAMMA, _DELTA}


def test_no_name_is_in_both_places_and_none_is_missing_from_both() -> None:
    """The partition holds for every incoming name, which is the whole correctness claim."""
    names = list(TOOL_NAMES)
    _, _, context, result = _project_once(
        names=names,
        exposed=[_BETA],
        referenced=[_GAMMA],
        always_available=[_DELTA],
        catalog_in_system_prompt=True,
    )

    projected = set(_names(result.tool_specs))
    listed = _listed_names(result.system_prompt)

    assert not (projected & listed), "a name is both callable and listed as unavailable"
    assert projected | listed == set(_names(context.tool_specs))


def test_the_projection_carries_no_catalog_entry_in_this_mode() -> None:
    """An empty ``inputSchema`` is what the mode exists to remove, so none may survive in it."""
    _, _, _, result = _project_once(
        names=[_ALPHA, _BETA, _GAMMA],
        catalog_in_system_prompt=True,
    )

    for spec in result.tool_specs:
        schema = spec["inputSchema"]["json"] if "json" in spec["inputSchema"] else spec["inputSchema"]
        assert schema.get("properties") != {} or spec["name"] == FIND_TOOLS_NAME or "required" in schema


def test_the_block_names_the_search_tool_so_the_rule_is_actionable() -> None:
    """A listing the model cannot act on is worse than no listing: the header must name the way out."""
    _, _, _, result = _project_once(names=[_ALPHA, _BETA], catalog_in_system_prompt=True)
    assert FIND_TOOLS_NAME in _prompt_text(result.system_prompt)


def test_the_operators_prompt_is_preserved_and_keeps_the_opening_position() -> None:
    """The block is appended, never substituted, so a prefix cache checkpoint does not move."""
    _, _, _, result = _project_once(names=[_ALPHA, _BETA], catalog_in_system_prompt=True)
    text = _prompt_text(result.system_prompt)
    assert text.startswith(_BASE_PROMPT)


def test_the_mode_is_inert_when_the_catalog_is_suppressed() -> None:
    """``catalog_tokens=None`` already means there is no catalog, so there is nothing to place."""
    _, _, context, result = _project_once(
        names=[_ALPHA, _BETA],
        catalog_tokens=None,
        catalog_in_system_prompt=True,
    )
    assert _prompt_text(result.system_prompt) == _prompt_text(context.system_prompt)


def test_the_mode_off_leaves_the_system_prompt_identical() -> None:
    """The default must not write to the system prompt at all -- that is the documented contract."""
    _, _, context, result = _project_once(names=[_ALPHA, _BETA], catalog_in_system_prompt=False)
    assert result.system_prompt == context.system_prompt


def test_an_empty_catalog_adds_no_header() -> None:
    """A header promising a list, with no list under it, is a false statement about the tool set."""
    assert _catalog_prompt_block([], set(), 20) == ""
    specs = [{"name": _ALPHA, "description": "Do alpha.", "inputSchema": {"json": {}}}]
    assert _catalog_prompt_block(specs, {_ALPHA}, 20) == ""


@pytest.mark.parametrize(
    ("prompt", "expected_type"),
    [(None, str), ("Existing.", str), ([{"text": "Existing."}], list)],
)
def test_appending_preserves_the_shape_the_prompt_arrived_in(prompt: Any, expected_type: type) -> None:
    """A caller using the list form places cache checkpoints between blocks; flattening would move them."""
    result = _append_to_system_prompt(prompt, "Block.")
    assert isinstance(result, expected_type)
    assert "Block." in _prompt_text(result)
    if prompt is not None:
        assert "Existing." in _prompt_text(result)


def test_appending_an_empty_block_returns_the_prompt_unchanged() -> None:
    """By identity, so a call that places nothing cannot invalidate a cached prefix."""
    prompt = [{"text": "Existing."}]
    assert _append_to_system_prompt(prompt, "") is prompt


def test_a_non_boolean_placement_flag_is_refused() -> None:
    """Construction validates before any handler is registered, so a typo fails loudly and early."""
    from strands_progressive_tool_disclosure import ProgressiveToolDisclosure

    with pytest.raises(ValueError, match="catalog_in_system_prompt"):
        ProgressiveToolDisclosure(catalog_in_system_prompt="yes")  # type: ignore[arg-type]
