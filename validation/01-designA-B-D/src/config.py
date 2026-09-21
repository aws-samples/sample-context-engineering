"""Central configuration for the context-strategy validation harness.

Everything that touches AWS or that a run depends on is declared here, so a run is
reproducible from a single file. Model ids were verified against the target account
with ``bedrock list-inference-profiles`` and ``bedrock list-foundation-models``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --- AWS account -------------------------------------------------------------------

ACCOUNT_ID = os.environ.get("VALIDATION_ACCOUNT_ID", "")
"""Account a run must be executing in, or empty to accept whichever one the credentials resolve to.

Empty by default, and read from the environment rather than written here, for two reasons. An
account id is not a secret but it does identify an organisation, and a harness meant to be shared
should not carry one. And pinning it is only useful to the person who owns that account: for anyone
else a hardcoded id turns a working checkout into a preflight failure.

Set it when you want the guarantee — a run that silently used the wrong account would produce
numbers attributed to the wrong place.
"""

REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
"""Region every client is built in. Honours the standard AWS environment variables."""

AWS_PROFILE = os.environ.get("VALIDATION_AWS_PROFILE") or os.environ.get("AWS_PROFILE") or None
"""Named profile to use, or ``None`` to use the default credential chain.

``None`` is the shareable default: credentials come from wherever boto3 normally finds them —
environment variables, an SSO session, an instance or container role — so nothing about one
workstation's setup is recorded here. Set ``VALIDATION_AWS_PROFILE`` to pin a profile.
"""

# --- Models ------------------------------------------------------------------------

AGENT_MODEL_ID = "us.anthropic.claude-opus-4-8"
"""Claude Opus 4.8, cross-region inference profile. Drives the agent under test.

Switch this together with ``Pricing.agent_input_per_mtok`` and ``agent_output_per_mtok`` below, or the
cost column reports one model's rates against another's tokens. The pairs are Opus 4.8 at ``5.00`` /
``25.00`` and ``us.anthropic.claude-haiku-4-5-20251001-v1:0`` at ``1.00`` / ``5.00`` -- read off the
Bedrock pricing page on 2026-09-21.

Absolute accuracy is not comparable across models; the token comparison is, because every
configuration within a run uses the same model. Haiku's 200k context window is also the tighter
constraint — the baseline peaked at 180k input tokens per call over 60 turns, about 10% of headroom —
so a longer script wants Opus or ``us.anthropic.claude-sonnet-4-6`` (1M context, $3/$15).
"""

RERANK_MODEL_ID = "cohere.rerank-v3-5:0"
"""Latest rerank model available in this account. BedrockReranker builds the ARN itself."""

INVOCATION_LOG_GROUP = "/aws/bedrock/modelinvocations"
"""Log group Bedrock delivers model invocation logs to, and the source of truth for tokens.

Configured account-wide with every data modality disabled, so entries carry the request id,
the token counts and the caller's ``requestMetadata`` but no prompts or completions. That is
deliberate: the scenario's payloads are synthetic, but a log group is a second copy of
whatever it records, and token counts are all this harness needs from it.
"""

# --- Thresholds --------------------------------------------------------------------
#
# These are deliberately low relative to the tool payloads: the mocked AWS-doc tools
# return 40k-120k characters, so every one of them blows max_result_tokens and forces
# the offloader (and therefore the reranker) to act. That is the point of the harness.


@dataclass(frozen=True)
class Thresholds:
    """Plugin thresholds shared by every configuration under test."""

    max_result_tokens: int = 4_000
    """Above this, a tool result is offloaded. Payloads are 10-30x this."""

    preview_tokens: int = 800
    """Preview budget. Small enough that chunk selection actually has to choose."""

    chunk_tokens: int = 500
    """Scoring granularity. Small chunks mean more rerank sources and finer recall."""

    relevance_threshold: float = 0.02
    """Cohere rerank v3.5 returns low absolute scores; 0.5 would reject everything.

    Measured on this account: a strong match scores ~0.29, an unrelated chunk ~0.03.
    The threshold is calibrated to that distribution, not to an abstract 'half'.
    """

    catalog_tokens: int = 20
    """Description budget per unexposed tool in the disclosure catalog."""

    ttl_cycles: int = 5
    """Cycles a disclosed tool schema stays resident after its last use."""

    top_k: int = 4
    """Tools exposed per find_tools call."""

    evict_after_cycles: int = 40
    """Keep offloaded content retrievable for the whole run."""


THRESHOLDS = Thresholds()

# --- Paths -------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
"""The harness directory — one level above the ``src`` package, so generated data sits beside the
run scripts instead of inside the source tree."""

CACHE_DIR = ROOT / ".cache"
"""Downloaded AWS documentation, cached so runs are comparable and offline-repeatable."""

RESULTS_DIR = ROOT / "results"
ARTIFACTS_DIR = ROOT / ".artifacts"
SESSIONS_DIR = ROOT / ".sessions"
"""Where a resumed run keeps its sessions.

File-backed on purpose. The graph rides ``agent.state``, so which session manager carries it is not
what a resume measures — and a local directory keeps the figure free of network variance the run
would otherwise read as strategy overhead.
"""
"""Offloader FileStorage root."""

# --- Run control -------------------------------------------------------------------



@dataclass
class RunConfig:
    """One configuration under test."""

    name: str
    disclosure: bool = False
    relevance: bool = False
    graph: bool = False
    label: str = ""
    notes: str = ""
    extra: dict = field(default_factory=dict)


RUN_CONFIGS = {
    "baseline": RunConfig(
        name="baseline",
        label="Baseline (offloader, prefix preview)",
        notes=(
            "ContextOffloader with the default prefix preview. Establishes the cost the "
            "three strategies are measured against, and keeps tool results from "
            "overflowing the window outright."
        ),
    ),
    "disclosure": RunConfig(
        name="disclosure",
        disclosure=True,
        label="Progressive Tool Disclosure only",
        notes="Replaces full tool schemas with a lean catalog plus find_tools.",
    ),
    "relevance": RunConfig(
        name="relevance",
        relevance=True,
        label="Relevance Filtering only",
        notes="Reranker-scored, verbatim preview instead of a positional prefix.",
    ),
    "all": RunConfig(
        name="all",
        disclosure=True,
        relevance=True,
        label="Disclosure + relevance combined",
        notes="Measures interaction effects, not just the sum of the parts.",
    ),
    "graph": RunConfig(
        name="graph",
        graph=True,
        label="Context Graph only",
        notes=(
            "Replaces the linear history with a graph of Cards at three resolutions. "
            "The history-side strategy, measured against the same baseline as the "
            "schema and preview strategies."
        ),
    ),
    "graph-all": RunConfig(
        name="graph-all",
        graph=True,
        disclosure=True,
        relevance=True,
        label="Graph + disclosure + relevance",
        notes=(
            "The full stack. Also the only configuration that exercises "
            "referenced_source, which keeps a tool named only in a Description "
            "from losing its inputSchema."
        ),
    ),
}

GRAPH_VARIANTS: dict[str, dict] = {
    # Each variant changes exactly one thing against the shipped defaults, so a difference in
    # the report is attributable to that one thing.
    # Note: the double-applied link weight is a code defect, not a configuration, so it is
    # measured on a git branch rather than as a variant here.
    "gr-budget-40k": {
        "label": "Graph, 40k body budget",
        "body_budget": 40_000,
        "notes": "A ceiling that actually binds, so the budget-driven step down is exercised.",
    },
    # The three threshold variants are stated on the MEASURED scale, not on an intuitive one.
    # Over an 18-turn session cohere.embed-multilingual-v3 answered 133 scored Cards with a
    # minimum note of 0.346 and a median of 0.563. A floor of 0.15 or 0.30 is below everything
    # the model ever answers, so it would measure nothing at all.
    "gr-eager": {
        "label": "Graph, eager expansion (threshold 0.50)",
        "expand_threshold": 0.50,
        "notes": "Just under the measured median: buys accuracy with tokens.",
    },
    "gr-aggressive": {
        "label": "Graph, aggressive collapse (floor 0.55)",
        "collapse_floor": 0.55,
        "expand_threshold": 0.55,
        "notes": (
            "Floor and threshold together, which removes the Description rung entirely: "
            "a Card is either whole or an address. The most tokens the ladder can save, and "
            "the reading that says whether the middle rung earns its place."
        ),
    },
    "gr-regression": {
        "label": "Graph, off switch (expand_threshold 0.0)",
        "expand_threshold": 0.0,
        "notes": (
            "The regression key. Must produce a context identical to the baseline's, "
            "field by field. A difference here is a bug, not a tradeoff."
        ),
    },
    # Selection: the call addresses a bounded set of Cards instead of all of them. This is what
    # gives the links a job -- with every Card addressed, propagation only breaks ties in a ranking
    # nobody is excluded from, so an edge can never be the reason a Card is reached.
    "gr-select": {
        "label": "Graph, selection (10 recent + top 5 + one hop)",
        "recent_cards": 10,
        "select_top_k": 5,
        "notes": (
            "Addressing every Card grows the call linearly with the conversation and is bounded "
            "by nothing, since body_budget only debits full content. This bounds it, and the "
            "turns left out are announced as searchable rather than hidden."
        ),
    },
    "gr-select-tight": {
        "label": "Graph, tight selection (5 recent + top 3 + one hop)",
        "recent_cards": 5,
        "select_top_k": 3,
        "notes": "The most the selection can save, and the arm where losing information should show.",
    },
    "gr-select-rerank": {
        "label": "Graph, selection + rerank second stage",
        "recent_cards": 10,
        "select_top_k": 5,
        "rerank": True,
        "notes": (
            "Embedding picks the candidates, rerank orders them. Worth a round trip only because "
            "selection makes being wrong cost the answer instead of tokens: the measured note "
            "distribution spans 0.346 to 0.75, which is a score that ranks without discriminating."
        ),
    },
    "gr-select-persist": {
        "label": "Graph, selection + persisted through the session",
        "recent_cards": 10,
        "select_top_k": 5,
        "persist": True,
        "notes": (
            "The graph rides agent.state, so file, S3 and AgentCore Memory all carry it. Measures "
            "what the payload costs a run whose session syncs on every message."
        ),
    },
    # The pair for a 100-turn run. Identical but for `persist`, so a difference between them is the
    # persistence and nothing else -- and the difference only exists across a `--resume-at`, since a
    # long-lived process restores nothing and never runs the load path.
    # A threshold is a position in a distribution, not an absolute value, and propagation strength is
    # part of that distribution: applying the structural weight once rather than twice roughly halves
    # what a hop hands over, which moves every score and collapses the expand/collapse ladder without
    # either threshold being touched. Measured over 133 scored Cards, the two thresholds below place
    # the collapse floor 48% of the way from the minimum to the median and the expansion threshold at
    # 95% -- the positions the ladder needs to have three working rungs.
    "gr-select-recal": {
        "label": "Graph, selection + thresholds calibrated on the score distribution",
        "recent_cards": 10,
        "select_top_k": 5,
        "collapse_floor": 0.52,
        "expand_threshold": 0.64,
        "notes": (
            "A threshold is only calibratable against the distribution of the pair it compares, and "
            "propagation strength is part of that distribution. This is the same constraint that "
            "makes a collapse_floor of 0.15 unreachable, arriving from the other direction."
        ),
    },
    "gr-select-rerank-recal": {
        "label": "Graph, selection + rerank + thresholds calibrated on the score distribution",
        "recent_cards": 10,
        "select_top_k": 5,
        "collapse_floor": 0.52,
        "expand_threshold": 0.64,
        "rerank": True,
        "notes": "The cheapest arm measured, with the ladder at three working rungs.",
    },
    # The control for the calibration above: it pins the thresholds lower in the distribution, so the
    # comparison isolates the two numbers under test instead of confounding them with anything else
    # that differs between two runs.
    "gr-select-lowscale": {
        "label": "Graph, selection + thresholds pinned lower in the distribution",
        "recent_cards": 10,
        "select_top_k": 5,
        "collapse_floor": 0.45,
        "expand_threshold": 0.55,
        "notes": (
            "Run this against gr-select with --repeats 3. One replay cannot settle a token delta: the "
            "agent picks its own tool path and the model-call count moved 50 to 57 between two replays "
            "of the same configuration, taking the token total with it."
        ),
    },
    "gr-long": {
        "label": "Graph, 100-turn selection (20 recent + top 10), derived",
        "recent_cards": 20,
        "select_top_k": 10,
        "persist": False,
        "notes": (
            "The control arm of the resume. After the restart it derives the graph by scan, which "
            "measured 30ms over 18 turns and 2.9s over 200 -- on the critical path of a model call."
        ),
    },
    "gr-long-persist": {
        "label": "Graph, 100-turn selection (20 recent + top 10), persisted",
        "recent_cards": 20,
        "select_top_k": 10,
        "persist": True,
        "notes": (
            "The same arm with the graph stored in agent.state. After the restart it loads instead "
            "of scanning, and the guards drop any Card addressing a message the session no longer has."
        ),
    },
}
"""Graph variants swept to find the token economy that holds accuracy.

The sweep runs as ordinary configurations, so it reuses the whole measurement and reporting
path rather than needing one of its own.
"""

for _variant, _spec in GRAPH_VARIANTS.items():
    RUN_CONFIGS[_variant] = RunConfig(
        name=_variant,
        graph=True,
        # On the full stack, not on the graph alone. Measured, the graph by itself costs 8.5% MORE
        # than the baseline: history is ~11% of the baseline's input and schemas are 58%, so the
        # ceiling on what the graph can save without disclosure is below the fixed cost of its own
        # final block. `graph-all` is the only configuration a graph variant is comparable to.
        disclosure=True,
        relevance=True,
        label=_spec["label"],
        notes=_spec["notes"],
        extra={key: value for key, value in _spec.items() if key not in ("label", "notes")},
    )

# --- Web fetch ---------------------------------------------------------------------

WEB_TARGETS = {
    "aws-lambda-invocation": "https://docs.aws.amazon.com/lambda/latest/dg/lambda-invocation.html",
    "aws-s3-naming": "https://docs.aws.amazon.com/AmazonS3/latest/userguide/bucketnamingrules.html",
    "amazon-home": "https://www.amazon.com/",
}
"""Pages the Playwright tool downloads. Public, read-only, no credentials involved."""

CHARS_PER_TOKEN = 4
"""The same estimate the SDK plugins use, so our numbers line up with their logs."""

# --- Pricing -----------------------------------------------------------------------


@dataclass(frozen=True)
class Pricing:
    """Unit prices used to turn measured usage into a cost comparison.

    **The harness measures units; these are configuration.** Every figure in the cost column is
    ``measured units x a rate declared here``, so a wrong rate is corrected by editing this class and
    re-rendering with ``--report-only``, never by re-running against Bedrock.

    They are not read from the Price List API on purpose. The API in this account carries no usage
    type for the agent model, for ``cohere.embed-multilingual-v3`` or for ``cohere.rerank-v3-5``, so a
    lookup would either fail or silently match a different model — and a cost table that looks
    authoritative and is not is worse than one that says where its numbers came from.

    Attributes:
        agent_input_per_mtok: Agent model input tokens, per million.
        agent_output_per_mtok: Agent model output tokens, per million.
        embedding_per_mtok: Embedding input tokens, per million.
        rerank_per_ksearchunit: Rerank search units, per thousand. One unit is up to 100 documents
            of up to 512 tokens in a single query.
    """

    agent_input_per_mtok: float = 5.00
    agent_output_per_mtok: float = 25.00
    embedding_per_mtok: float = 0.10
    rerank_per_ksearchunit: float = 2.00


PRICING = Pricing()
"""The rates the cost column applies. Edit and re-render; do not re-run."""

# --- Tool suite sizing -------------------------------------------------------------

TARGET_SCHEMA_TOKENS = 63_000
"""Schema budget per model call, matched to the measured session.

That session showed ~63,000 tokens of tool schema on every one of its 33 calls — about
85% of the floor of a call with an empty history — from an MCP banking server plus a web
search gateway, a browser, a code interpreter and a calculator.

This number is the harness's most important calibration, and getting it wrong understates
the strategy under test. The budget has to reach the share of a call the motivating case
showed: a suite carrying 40 tools for ~10,900 tokens of schema puts schema at 41% of
assembled input instead of 85%, six times less than the case the design addresses. Halving
that barely moves the total, which reads as "the saving is small" when it actually means
"the scenario had little schema to save".

The filler suite is generated until this budget is met, so the ratio between schema and
history matches a real agent rather than whatever a hand-written tool list happened to add
up to. Lower it for cheaper runs, but the comparison stops being representative.
"""


def estimate_tokens(text: str) -> int:
    """Return the SDK's character-based token estimate for ``text``."""
    return len(text) // CHARS_PER_TOKEN


DEFAULT_CONFIGURATIONS = (
    "baseline",
    "disclosure",
    "relevance",
    "graph",
    "all",
)
"""What a run compares when no configuration is named.

The strategy comparison, and nothing else. The graph sweep variants are deliberately not
here: folding them in would make a default run launch many configurations in parallel — enough
contention to distort the latency figures and risk throttling, to answer a question nobody asked.
"""

# Declared last so the sweep names, built above, are included.
CONFIGURATIONS = (*DEFAULT_CONFIGURATIONS, "graph-all", *GRAPH_VARIANTS)
"""Every configuration the runner accepts by name.

The sweep variants are selectable with ``--configs`` but never run by default.
"""
