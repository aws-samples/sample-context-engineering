"""Central configuration for the community-plugin validation harness.

Everything that touches AWS or that a run depends on is declared here, so a run is
reproducible from a single file. Model ids were verified against the target account with
``bedrock list-inference-profiles`` and ``bedrock list-foundation-models``.

This harness is the sibling of ``validation/01-designA-B-D``, which measures the same three
strategies as **vended plugins of a forked SDK**. Here they are the three **community
packages** installed alongside an unmodified ``strands-agents``:

======================  ==========================================  ===============================
Strategy                Vended (01-designA-B-D)                     Community (here)
======================  ==========================================  ===============================
Relevance filtering     ``ContextManager`` + ``Offload.relevance``   ``strands_relevance_filter.RelevanceFilter``
Progressive disclosure  ``strands.vended_plugins…``                  ``strands_progressive_tool_disclosure…``
Context graph           ``ContextStrategy(strategy="graph")``        ``strands_context_graph.ContextGraph``
======================  ==========================================  ===============================

Three consequences of that swap are configuration, not detail, and are recorded here because
they change what a run means:

1. **The baseline installs no plugin at all.** The vended harness put its offloader in every
   configuration, the baseline included, so nothing could overflow the window. Here the
   baseline is the unmodified agent: the 60k-250k character tool payloads enter the history
   whole. That is the honest control -- it measures the cost of doing nothing -- and it is
   expected to hit the model's context limit on the heavier lines. A turn that overflows is
   recorded as an error and reported as such, which is a result rather than a harness failure.
2. **The graph is ephemeral.** The community plugin keeps its state in a weakly-keyed map and
   writes nothing to ``agent.state``, so there is no load path and therefore no resume to
   measure. The vended harness's ``--resume-at`` and its ``persist`` variants have no
   counterpart and are gone.
3. **The graph publishes no per-turn telemetry.** The vended plugin emitted one log record per
   turn carrying the resolution ladder, which the vended harness parsed into curves. The
   community plugin emits failures only, so the evidence here is read off the plugin's
   end-of-run state instead. The per-turn ladder curves are unavailable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --- AWS account -------------------------------------------------------------------

ACCOUNT_ID = os.environ.get("VALIDATION_ACCOUNT_ID", "")
"""Account a run must be executing in, or empty to accept whichever one the credentials resolve to.

Empty by default, and read from the environment rather than written here: an account id is not a
secret but it does identify an organisation, and a harness meant to be shared should not carry one.
Set it when you want the guarantee -- a run that silently used the wrong account would produce
numbers attributed to the wrong place.
"""

REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
"""Region every client is built in. Honours the standard AWS environment variables."""

AWS_PROFILE = os.environ.get("VALIDATION_AWS_PROFILE") or os.environ.get("AWS_PROFILE") or None
"""Named profile to use, or ``None`` to use the default credential chain.

``None`` is the shareable default: credentials come from wherever boto3 normally finds them --
environment variables, an SSO session, an instance or container role -- so nothing about one
workstation's setup is recorded here.
"""

# --- Models ------------------------------------------------------------------------

AGENT_MODEL_ID = "us.anthropic.claude-opus-4-8"
"""Claude Opus 4.8, cross-region inference profile. Drives the agent under test.

Switch this together with ``Pricing.agent_input_per_mtok`` and ``agent_output_per_mtok`` below, or
the cost column reports one model's rates against another's tokens.

Absolute accuracy is not comparable across models; the token comparison is, because every
configuration within a run uses the same model.
"""

RERANK_MODEL_ID = "cohere.rerank-v3-5:0"
"""Rerank model the relevance filter scores chunks with. Latest rerank model in the account.

Set explicitly because the community package defaults to ``amazon.rerank-v1:0`` instead, and a
harness that accepted that default would report one model's behaviour under another's name.
``BedrockReranker`` resolves the ARN itself.
"""

EMBED_MODEL_ID = "cohere.embed-multilingual-v3"
"""Embedding model the graph's similarity matcher scores Cards with.

The community matcher already defaults to this, and it is restated here so the three graph
thresholds below stay traceable to the distribution they were calibrated against: they are
calibrated for *this* model and are not portable to another one.
"""

INVOCATION_LOG_GROUP = "/aws/bedrock/modelinvocations"
"""Log group Bedrock delivers model invocation logs to, and the source of truth for tokens.

Configured account-wide with every data modality disabled, so entries carry the request id, the
token counts and the caller's ``requestMetadata`` but no prompts or completions.
"""

# --- Thresholds --------------------------------------------------------------------
#
# Deliberately low relative to the tool payloads: the mocked AWS-doc tools return 40k-120k
# characters, so every one of them blows max_result_tokens and forces the relevance filter (and
# therefore the reranker) to act. That is the point of the harness.


@dataclass(frozen=True)
class Thresholds:
    """Plugin thresholds shared by every configuration under test."""

    # -- relevance filtering ------------------------------------------------------
    max_result_tokens: int = 4_000
    """Above this, a tool result is relevance-filtered. Payloads are 10-30x this."""

    preview_tokens: int = 800
    """Preview budget. Small enough that chunk selection actually has to choose."""

    chunk_tokens: int = 500
    """Scoring granularity. Small chunks mean more rerank sources and finer recall."""

    relevance_threshold: float = 0.02
    """Cohere rerank v3.5 returns low absolute scores; the package default of 0.5 rejects everything.

    Measured on this account: a strong match scores ~0.29, an unrelated chunk ~0.03. The threshold
    is calibrated to that distribution, not to an abstract 'half'. This is the one place the
    community package's default is deliberately overridden on numeric grounds rather than on
    preference.
    """

    # -- progressive tool disclosure ----------------------------------------------
    catalog_tokens: int = 20
    """Description budget per unexposed tool in the disclosure catalog."""

    ttl_cycles: int = 5
    """Cycles a disclosed tool schema stays resident after its last use."""

    top_k: int = 4
    """Tools exposed per find_tools call."""

    # -- context graph -------------------------------------------------------------
    #
    # The package's own defaults, restated rather than omitted: they are calibrated against
    # EMBED_MODEL_ID's score distribution, so a run that changes the embedding model has to
    # revisit all three. Stating them here is what makes that dependency visible.
    expand_threshold: float = 0.55
    """Note at or above which a Card enters the call at full content, budget permitting."""

    collapse_floor: float = 0.45
    """Note below which a Card keeps only its Title."""

    link_threshold: float = 0.50
    """Similarity at or above which two Cards link to each other."""

    description_tokens: int = 100
    """Token ceiling of a Card's Description."""

    min_cards: int = 3
    """Below this many Cards the whole choice is skipped: the only decision is 'send it all'."""


THRESHOLDS = Thresholds()

# --- Paths -------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
"""The harness directory -- one level above the ``src`` package, so generated data sits beside the
run scripts instead of inside the source tree."""

CACHE_DIR = ROOT / ".cache"
"""Downloaded AWS documentation, cached so runs are comparable and offline-repeatable."""

RESULTS_DIR = ROOT / "results"
ARTIFACTS_DIR = ROOT / ".artifacts"
"""Where the relevance filter's FileStore keeps the raw sub-blocks it replaced with a preview."""

SESSIONS_DIR = ROOT / ".sessions"
"""Unused by this harness, kept so the reused modules that reference a path find one.

The community graph is ephemeral -- its state reaches neither ``agent.state`` nor disk -- so there
is no load path to exercise and no resume to measure.
"""

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
        label="Baseline (no plugin)",
        notes=(
            "The unmodified agent. No plugin is installed, so every tool payload enters the "
            "history whole and stays there. This is the control the three strategies are "
            "measured against, and it is expected to reach the model's context limit on the "
            "heavier lines: a turn that overflows is recorded as an error, which is the "
            "measurement rather than a failure of the harness."
        ),
    ),
    "relevance": RunConfig(
        name="relevance",
        relevance=True,
        label="Relevance Filtering only",
        notes=(
            "RelevanceFilter alone: an oversized tool result is stored and replaced by a "
            "reranker-scored, verbatim preview plus a reference the model reads back through "
            "retrieve_context."
        ),
    ),
    "disclosure": RunConfig(
        name="disclosure",
        disclosure=True,
        label="Progressive Tool Disclosure only",
        notes=(
            "ProgressiveToolDisclosure alone: full tool schemas are replaced by a lean catalog "
            "plus find_tools. Note that it does nothing about oversized payloads, so this "
            "configuration carries the same history mass as the baseline."
        ),
    ),
    "graph": RunConfig(
        name="graph",
        graph=True,
        label="Context Graph only",
        notes=(
            "ContextGraph alone: the linear history becomes a graph of Cards entering each call "
            "at Title, Description or Full Content. The history-side strategy, measured against "
            "the same baseline as the other two."
        ),
    ),
    "all": RunConfig(
        name="all",
        disclosure=True,
        relevance=True,
        graph=True,
        label="All three combined",
        notes=(
            "The full stack, and the only configuration that exercises the harness's own "
            "referenced-source bridge: the community graph publishes no accessor for the tools a "
            "stepped-down Card still mentions, so without the bridge the model reads about a "
            "tool whose inputSchema left the call."
        ),
    ),
}

DEFAULT_CONFIGURATIONS = (
    "baseline",
    "relevance",
    "disclosure",
    "graph",
    "all",
)
"""What a run compares when no configuration is named: each strategy alone, plus all three."""

CONFIGURATIONS = DEFAULT_CONFIGURATIONS
"""Every configuration the runner accepts by name."""

# --- Web fetch ---------------------------------------------------------------------

WEB_TARGETS = {
    "aws-lambda-invocation": "https://docs.aws.amazon.com/lambda/latest/dg/lambda-invocation.html",
    "aws-s3-naming": "https://docs.aws.amazon.com/AmazonS3/latest/userguide/bucketnamingrules.html",
    "amazon-home": "https://www.amazon.com/",
}
"""Pages the Playwright tool downloads. Public, read-only, no credentials involved."""

CHARS_PER_TOKEN = 4
"""The same estimate the plugins use, so our numbers line up with their logs."""

# --- Pricing -----------------------------------------------------------------------


@dataclass(frozen=True)
class Pricing:
    """Unit prices used to turn measured usage into a cost comparison.

    **The harness measures units; these are configuration.** Every figure in the cost column is
    ``measured units x a rate declared here``, so a wrong rate is corrected by editing this class
    and re-rendering with ``--report-only``, never by re-running against Bedrock.

    They are not read from the Price List API on purpose: the API in this account carries no usage
    type for the agent model, for the embedding model or for the rerank model, so a lookup would
    either fail or silently match a different model.

    Attributes:
        agent_input_per_mtok: Agent model input tokens, per million.
        agent_output_per_mtok: Agent model output tokens, per million.
        embedding_per_mtok: Embedding input tokens, per million.
        rerank_per_ksearchunit: Rerank search units, per thousand. One unit is up to 100 documents
            of up to 512 tokens in a single query.
    """

    agent_input_per_mtok: float = 15.00
    agent_output_per_mtok: float = 75.00
    embedding_per_mtok: float = 0.10
    rerank_per_ksearchunit: float = 2.00


PRICING = Pricing()
"""The rates the cost column applies. Edit and re-render; do not re-run."""

# --- Tool suite sizing -------------------------------------------------------------

TARGET_SCHEMA_TOKENS = 63_000
"""Schema budget per model call, matched to the measured session.

That session showed ~63,000 tokens of tool schema on every one of its 33 calls -- about 85% of the
floor of a call with an empty history -- from an MCP banking server plus a web search gateway, a
browser, a code interpreter and a calculator.

This is the harness's most important calibration, and getting it wrong understates the strategy
under test: a suite carrying 40 tools for ~10,900 tokens of schema puts schema at 41% of assembled
input instead of 85%. The filler suite is generated until this budget is met, so the ratio between
schema and history matches a real agent rather than whatever a hand-written tool list added up to.
"""


def estimate_tokens(text: str) -> int:
    """Return the character-based token estimate for ``text``."""
    return len(text) // CHARS_PER_TOKEN
