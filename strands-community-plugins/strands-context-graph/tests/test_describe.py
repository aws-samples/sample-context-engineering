"""Unit tests for Title, Description and Tag derivation by rule.

Every assertion here is about wording, and the load-bearing ones are substring assertions: a Title is a prefix of the
user message, a numeric line is an exact substring of the text it came from, and a truncated Description is a prefix of
the untruncated one. Where the card came from belongs to ``cards`` and is asserted in its own tests.
"""

import pytest

from strands_context_graph.describe import (
    _is_textual,
    compose_description,
    normalize,
    numeric_lines,
    select_tags,
    tag_candidates,
    title_for,
)
from strands_context_graph.state import Card, ToolPair

UNBOUNDED = 10_000
"""A budget no Description in these tests can reach, for comparing against the untruncated form."""


def card(
    *,
    title: str = "Compare the two quotes",
    kind: str = "subject",
    turn: int = 1,
    pairs: tuple[ToolPair, ...] = (),
    tool_names: frozenset[str] = frozenset(),
    references: tuple[str, ...] = (),
    lines: tuple[str, ...] = (),
    reference: str | None = None,
    content_type: str | None = None,
    size_bytes: int | None = None,
) -> Card:
    """Build a Card carrying only the fields ``describe`` reads."""
    return Card(
        title=title,
        kind=kind,  # type: ignore[arg-type]
        turn=turn,
        dialogue_ids=(),
        evidence_ids=(),
        pairs=pairs,
        tool_names=tool_names,
        references=references,
        numeric_lines=lines,
        tags=(),
        description="",
        reference=reference,
        content_type=content_type,
        size_bytes=size_bytes,
    )


def pair(tool_name: str, tool_use_id: str = "tu-1") -> ToolPair:
    """Build a consumed tool pair naming ``tool_name``."""
    return ToolPair(tool_use_id=tool_use_id, tool_name=tool_name, tracking_ids=("a", "b"), consumed=True)


# --- Title -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "user_text",
    [
        "",
        "short ask",
        "How much did the Sao Paulo warehouse invoice come to last quarter, in total?",
        "x" * 200,
        "   leading whitespace and a very long tail that runs well past the title budget for sure",
    ],
)
def test_a_title_is_always_a_literal_prefix_of_the_user_message(user_text: str) -> None:
    title = title_for(user_text)

    assert user_text.startswith(title)
    assert len(title) <= max(len(user_text), 48)


def test_a_short_message_is_its_own_title() -> None:
    assert title_for("check the balance") == "check the balance"


def test_a_long_message_is_cut_at_a_word_boundary_with_no_ellipse() -> None:
    title = title_for("How much did the Sao Paulo warehouse invoice come to last quarter?")

    assert title == "How much did the Sao Paulo warehouse invoice"
    assert "..." not in title


def test_a_single_token_longer_than_the_budget_is_cut_by_character_count() -> None:
    title = title_for("x" * 100)

    assert title == "x" * 48


# --- Numeric lines -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "balance: 1200",
        "total R$ 1.200,00",
        "amount 1,200.00 USD",
        "BRL1.2mi in receivables",
        "| march | 12 | 480 |",
        "revenue\t2024\t1200",
        "throughput 1.5e3 requests",
        "-42 adjustments",
    ],
)
def test_a_numeric_monetary_or_tabular_line_is_selected(line: str) -> None:
    assert numeric_lines([line]) == (line,)


@pytest.mark.parametrize(
    "line",
    [
        "no numbers at all here",
        "version abc123 of the parser",
        "see v1.2 of the spec",
        "   ",
    ],
)
def test_a_line_carrying_no_standalone_number_is_not_selected(line: str) -> None:
    assert numeric_lines([line]) == ()


def test_selected_lines_are_exact_substrings_of_their_source_text() -> None:
    text = "summary of the run\n    balance: 1.200,00\ntotal: 3\nnothing numeric\n"

    selected = numeric_lines([text])

    assert selected == ("    balance: 1.200,00", "total: 3")
    for line in selected:
        assert line in text


def test_duplicate_lines_are_kept_once_in_order_of_first_appearance() -> None:
    assert numeric_lines(["total: 3\nother: 4", "total: 3\nlast: 5"]) == ("total: 3", "other: 4", "last: 5")


def test_two_runs_over_the_same_texts_select_the_same_lines() -> None:
    texts = ["a: 1\nb: 2", "| x | 3 | 4 |"]

    assert numeric_lines(texts) == numeric_lines(texts)


# --- Description -------------------------------------------------------------------------------------------------


def test_a_subject_description_names_the_subject_the_tools_the_references_and_the_lines() -> None:
    subject = card(
        pairs=(pair("http_request"), pair("http_request", "tu-2"), pair("calculator", "tu-3")),
        references=("mem_1_tu-1_0", "mem_1_tu-1_0", "mem_1_tu-2_0"),
        lines=("total: 1.200,00",),
    )

    description = compose_description(subject, UNBOUNDED)

    assert description == (
        "Compare the two quotes\ntools: http_request (2), calculator (1)\nreferences: mem_1_tu-1_0, mem_1_tu-2_0\n"
        "total: 1.200,00"
    )


def test_an_absent_field_leaves_its_line_out_rather_than_rendering_it_empty() -> None:
    description = compose_description(card(), UNBOUNDED)

    assert description == "Compare the two quotes"


def test_a_tool_named_without_a_pair_is_counted_once() -> None:
    description = compose_description(card(tool_names=frozenset({"calculator", "http_request"})), UNBOUNDED)

    assert "tools: calculator (1), http_request (1)" in description


def test_a_textual_artifact_description_leads_with_the_reference_and_keeps_its_lines() -> None:
    artifact = card(
        title="mem_1_tu-3_0",
        kind="artifact",
        turn=2,
        pairs=(pair("read_file"),),
        lines=("balance: 1200",),
        reference="mem_1_tu-3_0",
        content_type="text/plain",
    )

    description = compose_description(artifact, UNBOUNDED)

    assert description == "reference: mem_1_tu-3_0\ntool: read_file\nturn: 2\nbalance: 1200"


def test_a_non_textual_artifact_description_carries_the_address_and_no_lines() -> None:
    artifact = card(
        title="reports/march.png",
        kind="artifact",
        turn=3,
        pairs=(pair("render_chart"),),
        lines=("balance: 1200",),
        reference="reports/march.png",
        content_type="image/png",
        size_bytes=900,
    )

    description = compose_description(artifact, UNBOUNDED)

    assert description == (
        "file: march.png\ncontent_type: image/png\nsize: 900 bytes\ntool: render_chart\nturn: 3\n"
        "reference: reports/march.png"
    )
    assert "balance: 1200" not in description


def test_overflowing_lines_are_cut_at_line_granularity_and_the_omitted_count_is_recorded() -> None:
    lines = tuple(f"account {index}: 1.200,0{index}" for index in range(11))
    subject = card(lines=lines)

    description = compose_description(subject, 20)

    assert description.endswith("numeric lines omitted)")
    omitted = int(description.rsplit("(+", 1)[1].split(" ", 1)[0])
    kept = description.count("account ")
    assert kept + omitted == len(lines)
    assert omitted > 0


def test_the_truncated_description_is_a_literal_prefix_of_the_untruncated_one() -> None:
    subject = card(lines=tuple(f"account {index}: 1.200,0{index}" for index in range(11)))

    truncated = compose_description(subject, 20)
    complete = compose_description(subject, UNBOUNDED)

    kept = truncated.rsplit("\n", 1)[0]
    assert complete.startswith(kept)
    assert len(truncated) <= 20 * 4


def test_a_header_longer_than_the_budget_is_cut_at_a_boundary_with_no_ellipse() -> None:
    subject = card(title="First sentence ends here. And a second sentence runs past the budget entirely.")

    description = compose_description(subject, 8)

    assert description == "First sentence ends here."
    assert "..." not in description


def test_two_runs_compose_the_same_description_character_for_character() -> None:
    subject = card(
        pairs=(pair("http_request"), pair("calculator", "tu-2")),
        tool_names=frozenset({"calculator", "http_request", "read_file"}),
        references=("mem_1_tu-1_0",),
        lines=("total: 3",),
    )

    assert compose_description(subject, 100) == compose_description(subject, 100)


# --- Tags --------------------------------------------------------------------------------------------------------


def test_candidates_come_from_tool_names_references_and_the_text_scan() -> None:
    subject = card(pairs=(pair("http_request"),), references=("mem_1_tu-1_0",))

    structural, textual = tag_candidates(subject, ["compare the warehouse invoice"])

    assert structural == ("http_request", "mem_1_tu-1_0")
    assert set(textual) == {"compare", "the", "warehouse", "invoice"}


def test_a_word_shorter_than_three_characters_is_not_a_candidate() -> None:
    _structural, textual = tag_candidates(card(), ["an ok id for the run"])

    assert "an" not in textual
    assert "ok" not in textual
    assert "for" in textual


def test_a_structural_candidate_is_not_repeated_as_a_textual_one() -> None:
    subject = card(pairs=(pair("calculator"),))

    structural, textual = tag_candidates(subject, ["run the calculator again"])

    assert structural == ("calculator",)
    assert "calculator" not in textual


def test_structural_candidates_take_every_slot_before_any_textual_one() -> None:
    tags = select_tags(
        candidates={"warehouse": 9, "invoice": 7},
        structural=["http_request", "calculator", "read_file", "mem_1", "mem_2"],
        document_frequency={},
        total_cards=4,
        tags_per_card=5,
        rarity_weight=0.5,
    )

    assert tags == ("http_request", "calculator", "read_file", "mem_1", "mem_2")


def test_textual_candidates_fill_the_slots_the_structural_ones_leave() -> None:
    tags = select_tags(
        candidates={"warehouse": 9, "invoice": 1},
        structural=["http_request"],
        document_frequency={"warehouse": 1, "invoice": 4},
        total_cards=4,
        tags_per_card=3,
        rarity_weight=0.5,
    )

    assert tags == ("http_request", "warehouse", "invoice")


def test_a_rarer_candidate_outranks_a_common_one_of_equal_repetition() -> None:
    tags = select_tags(
        candidates={"common": 5, "rare": 5},
        structural=[],
        document_frequency={"common": 8, "rare": 1},
        total_cards=8,
        tags_per_card=1,
        rarity_weight=1.0,
    )

    assert tags == ("rare",)


def test_ties_break_by_order_of_first_appearance() -> None:
    arguments = {
        "candidates": {"first": 4, "second": 4},
        "structural": [],
        "document_frequency": {"first": 2, "second": 2},
        "total_cards": 4,
        "tags_per_card": 1,
        "rarity_weight": 0.5,
    }

    assert select_tags(**arguments) == ("first",)  # type: ignore[arg-type]
    assert select_tags(**arguments) == select_tags(**arguments)  # type: ignore[arg-type]


def test_a_candidate_that_normalizes_to_whitespace_never_spends_a_slot() -> None:
    tags = select_tags(
        candidates={"warehouse": 2},
        structural=["", "   ", "http_request"],
        document_frequency={},
        total_cards=2,
        tags_per_card=2,
        rarity_weight=0.5,
    )

    assert tags == ("http_request", "warehouse")


def test_the_tag_count_never_exceeds_the_ceiling() -> None:
    tags = select_tags(
        candidates={f"word{index}": index + 1 for index in range(20)},
        structural=["http_request", "calculator"],
        document_frequency={},
        total_cards=6,
        tags_per_card=5,
        rarity_weight=0.5,
    )

    assert len(tags) == 5


# --- Normalization -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("R$ 1.200,00", "1200"),
        ("1.200,00", "1200"),
        ("1,200.00", "1200"),
        ("1200", "1200"),
        ("US$1200", "1200"),
        ("1 200 USD".replace(" ", "\u00a0"), "1200"),
        ("0.500", "0.5"),
        ("-1.200,50", "-1200.5"),
        ("Warehouse.", "warehouse"),
        ("(invoice)", "invoice"),
        ("get_balance", "get_balance"),
        ("2024-03", "2024-03"),
        ("...", ""),
        ("", ""),
    ],
)
def test_a_monetary_or_separator_bearing_number_collapses_onto_its_bare_form(token: str, expected: str) -> None:
    assert normalize(token) == expected


# --- Content types -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "content_type",
    ["text/plain", "text/csv; charset=utf-8", "application/json", "application/vnd.api+json", "APPLICATION/YAML"],
)
def test_a_textual_content_type_is_recognized(content_type: str) -> None:
    assert _is_textual(content_type) is True


@pytest.mark.parametrize("content_type", [None, "", "image/png", "application/octet-stream", "application/pdf"])
def test_an_absent_or_unrecognized_content_type_is_not_text(content_type: str | None) -> None:
    assert _is_textual(content_type) is False
