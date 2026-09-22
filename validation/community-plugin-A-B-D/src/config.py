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

AGENT_MODEL_ID = os.environ.get("VALIDATION_AGENT_MODEL_ID") or "us.anthropic.claude-opus-4-8"
"""Model driving the agent under test. Claude Opus 4.8's cross-region profile by default.

Override it with ``VALIDATION_AGENT_MODEL_ID`` rather than by editing this line: the rates the cost
column bills at are looked up from this value in :data:`MODEL_PRICING` below. That lookup is what
removes the footgun this docstring used to warn about -- the rate pair no longer has to be edited in
a second place to stay honest, and a model with no entry is refused at preflight rather than being
billed at another model's price.

Absolute accuracy is not comparable across models; the token comparison is, because every
configuration within a run uses the same model. Haiku is ~15x cheaper per token and a reasonable way
to iterate on the harness itself, but it is a weaker model and its accuracy column sits well below
this one's -- measured on the same 60-turn script, the no-plugin baseline scored 85.8% on Haiku 4.5
against 97.2% here.

**The window is the constraint the relevance arm reaches.** On the corrected 60-turn script, where
every filler turn produces a real tool payload, peak input per call measured on Haiku 4.5 was:
baseline 173,155; relevance 193,185 (plus four calls refused outright at 201,035 against a 200,000
limit); disclosure 161,934; graph 139,917; all three 49,863. Read the completed-turn count and the
error list alongside the token column: an arm that stopped completing turns has a truncated total,
and a percentage taken against it understates the saving.
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

    preview_tokens: int = 2_000
    """Preview budget: how much of an oversized payload survives, as whole verbatim chunks.

    Raised from 800 on measurement. The preview is verbatim -- ``RelevancePreview`` selects whole
    chunks that are exact substrings and inserts only gap markers -- so nothing here is lost to
    paraphrase. What 800 lost was *selection*: at ``chunk_tokens`` of 500 it admitted one or two
    chunks of a payload ten to thirty times its size, and a literal the answer needs (an enum value
    like ``DEGRADED``, an error class like ``MFA_CHALLENGE_TIMEOUT``, a figure like ``907,35``) simply
    was not in the chunks that made the cut.

    That cost compounds with the graph, which is why the number is raised here rather than worked
    around downstream. The graph derives a Card's ``numeric_lines`` from ``agent.messages`` at the turn
    boundary, and by then the message carries the preview, not the payload -- so the preview's budget
    is the ceiling on everything the graph can preserve, and ``expand_card`` cannot return a figure the
    preview dropped. Two cuts in series, and 800 made the first one decide the second.

    2,000 admits four chunks instead of one. The headroom is there: the all-three arm's peak call
    measured 57,259 tokens against roughly 199,000 usable on a 203K window, so 71% of the window was
    going unused while the plugin starved the evidence.
    """

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
    # The graph's knobs are NOT here, because a single set of them is wrong. They depend on whether
    # the relevance filter is also installed, which is a measured result rather than a preference --
    # see GRAPH_ALONE and GRAPH_WITH_RELEVANCE below.
    min_cards: int = 3
    """Below this many Cards the whole choice is skipped: the only decision is 'send it all'."""


THRESHOLDS = Thresholds()


@dataclass(frozen=True)
class GraphTuning:
    """The graph's knobs, as one set. There are two of them, and which one applies is measured.

    A single tuning for the graph is wrong, because what the graph should do depends on whether the
    relevance filter has already acted on the same content. Both sets below were measured on the same
    60-turn script against Haiku 4.5, one replay each.

    Attributes:
        expand_threshold: Note at or above which a Card enters the call at full content.
        collapse_floor: Note below which a Card keeps only its Title.
        link_threshold: Similarity at or above which two Cards link to each other.
        description_tokens: Token ceiling of a Card's Description.
        body_budget: Token ceiling across the Cards at full content, or ``None`` for no ceiling.
        max_retrieval_cycles: Retrieval calls one turn may spend across the graph's three tools, or
            ``None`` for the package's unbounded behaviour. Held here rather than left at the package
            default because the value that is right depends on what else is installed: the measured
            runaway was an ``all`` turn spending 31 tool calls, and a turn that has already had the
            filter compress its evidence has less to gain from a fourth recovery attempt than a turn
            running the graph alone.
    """

    expand_threshold: float
    collapse_floor: float
    link_threshold: float
    description_tokens: int
    body_budget: int | None
    max_retrieval_cycles: int | None = 8


GRAPH_ALONE = GraphTuning(
    expand_threshold=0.62,
    collapse_floor=0.45,
    link_threshold=0.50,
    description_tokens=250,
    body_budget=60_000,
)
"""Tuning for a graph with no relevance filter beside it. Raised off the package's defaults.

One hypothesis: move mass off the full-content rung and onto a Description rich enough to carry the
facts a check asks for.

The package's defaults (0.55 / 100 tokens) used all three rungs -- full 52%, Description 30%, Title
18% -- yet the graph still lost scored turns the no-plugin baseline got right, and not by forgetting
anything. It answered ``42.1%`` where the tool had emitted ``42,1%``. The figures were right and the
*formatting* was not, which this harness scores as wrong on purpose: an assistant that restates a
value in its own format has introduced an error class.

The mechanism is exact. ``compose_description`` copies ``Card.numeric_lines`` verbatim and the budget
decides only HOW MANY of those lines get in, appending ``(+N numeric lines omitted)`` for the rest. At
100 tokens a Card whose turn carried a real payload keeps a handful, so a later turn answering from
that Card re-renders the figure from its own paraphrase -- and that is where the separator flips.
Measured: the graph answered ``S000-allocation`` with no tool call at all, reading an earlier Card.

What these values measured against the defaults, same script, same model: input tokens 9,045,017 ->
8,462,341 (-6.4%), peak call 139,917 -> 119,342 (-15%), weighted accuracy 85.0% -> 89.0%, materially
correct 23/30 -> 24/30, and the ladder shifting from 17/10/6 to 12/15/6 exactly as intended. The token
and ladder moves are mechanical; the one-turn accuracy gain is inside the noise of a single replay.
"""

GRAPH_WITH_RELEVANCE = GraphTuning(
    expand_threshold=0.55,
    collapse_floor=0.45,
    link_threshold=0.50,
    description_tokens=250,
    body_budget=None,
    max_retrieval_cycles=4,
)
"""Tuning for a graph installed alongside the relevance filter.

**The two knobs were calibrated in the wrong order, and this docstring used to record the consequence as
a discovery.** It said the package's ``description_tokens`` of 100 belonged here because raising it "has
little left to preserve" once the filter had replaced the payload with a preview. That observation was
correct and the conclusion drawn from it was not: the preview's budget was 800 tokens against payloads
ten to thirty times that, so what starved the Card was the FIRST cut, and lowering the second one
accepted the loss instead of locating it. Reading the two in series --

    payload -> preview budget -> the message -> the Card's numeric lines -> Description budget

-- the preview is the ceiling on everything downstream. So ``preview_tokens`` was raised to 2,000 first
(see :class:`Thresholds`), and only then is 250 here worth anything: the figures a Card would preserve
now exist in the message it derives from.

The earlier measurement that produced 100 is not wrong, it is conditional. Applying the graph-alone set
with the preview at 800 lost five materially correct turns while moving tokens 1.2%, which is exactly
what a bigger Description budget with nothing left to put in it should do.

``expand_threshold`` stays at 0.55 rather than the graph-alone 0.62 for a reason the preview does not
change: raising it steps Cards off a full-content rung, and with the filter installed that rung is
already the cheap one. *When relevance has compressed the evidence, the graph should fold less, not
more.*

``body_budget`` stays ``None`` here because that is how this arm was measured, and its peak call sat at
49,863 tokens -- below any ceiling worth setting. Stating the measured configuration matters more than
carrying a ceiling that never binds.

``max_retrieval_cycles`` is 4 here against the package's 8, and the distribution is why. Measured on
GLM 4.7 Flash over 60 turns, this arm's retrieval spend per turn was 1 call on 13 turns, 2 on 5, 3 on
2, then 5, 8 and 12 -- so 87% of the turns that retrieved at all finished inside three calls, and the
whole tail above five is two turns. Eight therefore binds on nothing a healthy turn does while leaving
the pathological turn eight real Card rebuilds to spend; four keeps every turn in that 87% untouched
and halves what the outlier costs.

One replay each, so read every quantity here as the direction it points rather than as a settled
number. ``--repeats 3`` is what would settle it.
"""

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
        agent_input_per_mtok: Agent model uncached input tokens, per million.
        agent_output_per_mtok: Agent model output tokens, per million.
        cache_read_per_mtok: Input tokens served from a prompt cache, per million, or ``None`` when
            the model does not support caching on this API surface. Billed against
            ``cacheReadInputTokens``, which Bedrock reports separately from ``inputTokens`` -- so a
            cached run whose cost ignored this field would understate what it spent.
        cache_write_5m_per_mtok: Input tokens written to a 5-minute cache checkpoint, per million.
        cache_write_1h_per_mtok: Input tokens written to a 1-hour cache checkpoint, per million.
        embedding_per_mtok: Embedding input tokens, per million.
        rerank_per_ksearchunit: Rerank search units, per thousand. One unit is up to 100 documents
            of up to 512 tokens in a single query.
    """

    agent_input_per_mtok: float
    agent_output_per_mtok: float
    cache_read_per_mtok: float | None = None
    cache_write_5m_per_mtok: float | None = None
    cache_write_1h_per_mtok: float | None = None
    embedding_per_mtok: float = 0.10
    rerank_per_ksearchunit: float = 2.00


MODEL_PRICING = {
    # -- Amazon ---------------------------------------------------------------------
    # Nova 2 Lite is INFERENCE_PROFILE only in us-east-1 -- the bare model id carries no ON_DEMAND
    # entry, so the invocable ids are the us. and global. profiles, both verified ACTIVE.
    #
    # THE CACHE WRITE IS FREE. From the Price List API for us-east-1:
    # USE1-Nova2.0Lite-cache-write-input-token-count = $0.0000/Mtok, and cache read $0.0825 against a
    # $0.33 input rate (0.25x). Every other family measured here charges ~1.25x input to WRITE, which
    # is what makes a prefix-mutating strategy expensive under caching. On Nova that penalty is zero,
    # so this is the one model where caching and context compression can compose instead of compete.
    # Worth a cache-on/cache-off pair for exactly that reason.
    #
    # Caching is also capped: the card states Nova models cache a maximum of 20K tokens, with a
    # 5-minute TTL and checkpoints in system and messages only. The 1h field is therefore None -- a
    # 1h run against Nova is refused rather than priced at an invented rate.
    #
    # Tool use verified by invocation, not by reading the card: one Converse call carrying a
    # toolConfig returned stopReason=tool_use on both profiles (tmp/probe_tool_use.py).
    "us.amazon.nova-2-lite-v1:0": Pricing(0.33, 2.75, 0.0825, 0.0, None),
    "global.amazon.nova-2-lite-v1:0": Pricing(0.30, 2.50, 0.0750, 0.0, None),
    # -- Anthropic ----------------------------------------------------------------
    "us.anthropic.claude-opus-4-8": Pricing(5.00, 25.00, 0.50, 6.25, 10.00),
    "us.anthropic.claude-opus-5": Pricing(5.00, 25.00, 0.50, 6.25, 10.00),
    "us.anthropic.claude-sonnet-5": Pricing(2.00, 10.00, 0.20, 2.50, 4.00),
    "us.anthropic.claude-fable-5": Pricing(10.00, 50.00, 1.00, 12.50, 20.00),
    "us.anthropic.claude-fable-5-1": Pricing(10.00, 50.00, 0.25, 12.50, 20.00),
    # Cache rates for Haiku 4.5 are NOT on the Bedrock pricing page's Anthropic table; these follow
    # the 0.10x / 1.25x / 2.00x pattern every listed Claude obeys, so treat them as inferred.
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": Pricing(1.00, 5.00, 0.10, 1.25, 2.00),
    # -- OpenAI -------------------------------------------------------------------
    # Short-context rates (<=272k input). Above that the model card doubles them, which this
    # harness never reaches -- its peak call measured 204,439 tokens.
    #
    # These models publish ONE cache-write rate, for a 30-minute checkpoint, so both write fields
    # carry it rather than pretending a 5m/1h split exists. It matters because these models cache
    # IMPLICITLY on the Converse path whether or not the harness asks: measured on the first
    # baseline call of the astra run, usage came back as inputTokens=2, cacheWriteInputTokens=48,583.
    # So a run with --cache off still produces cache traffic here, and still has to be priced.
    "us.openai.gpt-6-astra": Pricing(11.00, 55.00, 1.10, 13.75, 13.75),
    "global.openai.gpt-6-astra": Pricing(10.00, 50.00, 1.00, 12.50, 12.50),
    "us.openai.gpt-5.6-sol": Pricing(4.40, 22.00, 0.44, 5.50, 5.50),
    "global.openai.gpt-5.6-sol": Pricing(4.00, 20.00, 0.40, 5.00, 5.00),
    # Terra and Luna write rates are inferred at the 1.25x every other OpenAI entry obeys; their
    # input, read and output rates are read off the cards.
    "us.openai.gpt-5.6-terra": Pricing(2.20, 13.20, 0.22, 2.75, 2.75),
    "global.openai.gpt-5.6-luna": Pricing(0.20, 1.20, 0.02, 0.25, 0.25),
    # -- Others -------------------------------------------------------------------
    # No cache rates published for these, so the fields stay None and a cached run against one is
    # refused rather than priced with a guess.
    "us.deepseek.r1-v1:0": Pricing(1.35, 5.40),
    # In-region only: the GLM models publish no cross-region inference profile, so the bare model id
    # is the invocable one. Verified with list-foundation-models / list-inference-profiles for GLM 5,
    # and the 4.7 cards state Geo and Global as not supported.
    #
    # None of the three publishes any prompt caching, so the cache fields stay None and a cached run
    # against one is refused rather than priced with a guess. That is the point of running them: in
    # this regime prompt caching is not an alternative to context engineering, it is unavailable.
    #
    # GLM 4.7 and 4.7 Flash cap max output at 4K tokens against GLM 5's 128K. This harness's answers
    # are far below that, but a script with long-form answers would truncate.
    "zai.glm-5": Pricing(1.00, 3.20),
    "zai.glm-4.7": Pricing(0.60, 2.20),
    "zai.glm-4.7-flash": Pricing(0.07, 0.40),
    # Small-window, no-caching models: the regime where these strategies are not an optimisation but
    # the thing that lets a 60-turn conversation finish. None publishes prompt caching, so the cache
    # field stays None on all of them and there is nothing for caching to compete against.
    #
    # Windows, from each model card: Nemotron Nano 9B v2 128K, Nemotron Nano 3 30B 256K, Ministral
    # 3B/8B/14B 128K, Gemma 3 12B 128K. The bare-agent baseline peaks near 200K on this script, so a
    # 128K model is where the overflow contrast is sharpest.
    #
    # Rates are the US East (N. Virginia) / US East (Ohio) / US West (Oregon) Standard-tier rows of
    # the Bedrock pricing page, read 2026-09-22. Two models were left OUT deliberately: gpt-oss-20b
    # /120b and Qwen3 32B publish Standard on-demand rows for Asia Pacific (Sydney) only, so pricing
    # them in us-east-1 would be a guess.
    "nvidia.nemotron-nano-9b-v2": Pricing(0.06, 0.23),
    "nvidia.nemotron-nano-3-30b": Pricing(0.06, 0.24),
    "nvidia.nemotron-super-3-120b": Pricing(0.15, 0.65),
    "mistral.ministral-3-3b-instruct": Pricing(0.10, 0.10),
    "mistral.ministral-3-8b-instruct": Pricing(0.15, 0.15),
    "mistral.ministral-3-14b-instruct": Pricing(0.20, 0.20),
    "mistral.magistral-small-2509": Pricing(0.50, 1.50),
    "mistral.mistral-large-3-675b-instruct": Pricing(0.50, 1.50),
    # Qwen3 Next 80B A3B, 256K window, no prompt caching published. Rates are the us-east-1 Standard
    # tier from the Price List API (USE1-Qwen3Next-80B-A3B-*-tokens-standard). Tool use verified by
    # invocation: stopReason=tool_use on a Converse call carrying a toolConfig.
    "qwen.qwen3-next-80b-a3b": Pricing(0.14, 1.20),
    # NOT ADDED, and deliberately: the Gemma family cannot run this harness. Gemma 3 12B/27B accept a
    # Converse request carrying a toolConfig and then IGNORE it -- measured, the model answers in prose
    # asking to be given the tool, and inputTokens comes back at 27, meaning the tool schema was
    # dropped rather than read. Gemma 4 is bedrock-mantle only: Converse answers "The provided model
    # identifier is invalid", and it is absent from list-foundation-models in us-east-1. A benchmark
    # built entirely on tool payloads has nothing to measure on a model that cannot call a tool.
}
"""Rates per model id, in USD per million tokens, for ``us-east-1`` Standard tier.

Sourced from the Bedrock pricing page's provider tables and, for the OpenAI line, from each model
card -- read on 2026-09-21. Two shapes of price live here that the old single ``Pricing`` could not
express: the OpenAI models are priced per inference profile rather than per region, so the ``us.``
and ``global.`` profiles of one model are separate entries; and a model without published cache
rates carries ``None``, which :func:`pricing_for` turns into a refusal instead of a guess.

**The keys are load-bearing.** ``VALIDATION_AGENT_MODEL_ID`` is looked up here verbatim, so a model
id absent from this map stops the run at preflight. That is deliberate for a harness whose output is
a cost column: silently billing a new model at Opus rates produces a number that looks fine and is
wrong.

These are **list prices**, and a cost column built from them is an estimate of list cost, not of any
particular bill: discounts, commitments and regional fees are account-specific and deliberately not
modelled here. Every comparison in the reports is a ratio between rows priced the same way, so a
uniform difference between list and actual cancels out of it.
"""


def pricing_for(model_id: str) -> Pricing:
    """Return the rates for ``model_id``.

    Args:
        model_id: The Bedrock model id or inference profile the run will invoke.

    Returns:
        The declared rates for that model.

    Raises:
        KeyError: If the model has no declared rates. Raised rather than defaulted, because a cost
            column billed at the wrong model's price is worse than no run at all.
    """
    try:
        return MODEL_PRICING[model_id]
    except KeyError:
        known = "\n  ".join(sorted(MODEL_PRICING))
        raise KeyError(
            f"no declared pricing for model id {model_id!r}. Add it to MODEL_PRICING in "
            f"src/config.py with rates read off the Bedrock pricing page. Known ids:\n  {known}"
        ) from None


PRICING = pricing_for(AGENT_MODEL_ID)
"""The rates the cost column applies, resolved from :data:`AGENT_MODEL_ID` at import.

Correcting a rate is still an edit-and-re-render: fix the entry in :data:`MODEL_PRICING` and re-run
with ``--report-only``, never against Bedrock.
"""


CACHE_TTL = os.environ.get("VALIDATION_CACHE") or None
"""Prompt-cache TTL for the agent's cache checkpoints, or ``None`` to run uncached.

``None`` (the default) is what every published figure in this harness was measured with, so it stays
the default: turning caching on changes the cost column's meaning and must be an explicit choice.
Accepts a Bedrock TTL string -- ``"5m"`` or ``"1h"`` -- set by ``--cache`` or by
``VALIDATION_CACHE``.

One TTL covers all three checkpoints (toolConfig, system, messages) on purpose. Bedrock requires
checkpoint TTLs to be non-increasing in that order and rejects a longer one following a shorter one,
so a single value is the only setting that cannot produce a request-time rejection.

**What this buys, and where.** 5,451,072 of the baseline's 14,346,683 input tokens are tool schema,
identical on all 87 calls, which is the cacheable prefix. It is cacheable only where the prefix is
stable: ``disclosure`` mutates the tool set by design, so each change invalidates the checkpoint and
pays a write at 1.25x, and ``graph`` rewrites the history, so the message checkpoint never reads.
Caching therefore cheapens the control more than the treatments and *narrows* the measured saving of
the strategies. That is what caching does, not a fault in it -- but it means a cached run is a
separate column, not a replacement for the uncached one.
"""

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
