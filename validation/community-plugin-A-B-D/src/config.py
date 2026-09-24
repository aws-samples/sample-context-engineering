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

# --- Sweep overrides ---------------------------------------------------------------
#
# Every tunable below can be overridden from the environment. The defaults are unchanged, so a run
# that sets nothing behaves exactly as the committed configuration does -- the overrides exist so a
# tuning sweep is a list of environment variables rather than a list of commits, which is what makes
# "measure, change one knob, measure again" affordable enough to actually do.
#
# Each override is recorded in the run's metadata (see ``sweep_overrides``), so no result can be read
# without knowing which knobs produced it.

_OVERRIDES_SEEN: dict[str, str] = {}
"""Every override actually read from the environment, in the order the module read it."""


def _env(name: str) -> str | None:
    """Return the raw value of ``name``, remembering that it was set."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    _OVERRIDES_SEEN[name] = raw
    return raw


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    return default if raw is None else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    return default if raw is None else float(raw)


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean knob. Accepts ``1/true/yes/on`` and ``0/false/no/off``, case-insensitively.

    A value that is neither raises rather than falling back to the default: a typo in a sweep variable
    must not silently measure the default configuration under the variant's name.
    """
    raw = _env(name)
    if raw is None:
        return default
    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name}=<{raw!r}> | must be one of 1/true/yes/on or 0/false/no/off")


def _env_opt_int(name: str, default: int | None) -> int | None:
    """Like :func:`_env_int` but accepts ``none`` to mean the package's unbounded behaviour."""
    raw = _env(name)
    if raw is None:
        return default
    if raw.strip().lower() in {"none", "null", "off", "unbounded"}:
        return None
    return int(raw)


def sweep_overrides() -> dict[str, str]:
    """Return the overrides this process read, for the run record."""
    return dict(_OVERRIDES_SEEN)


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

MAX_OUTPUT_TOKENS = _env_int("VALIDATION_MAX_OUTPUT_TOKENS", 4_096)
"""Output-token cap handed to every model under test. Override with ``VALIDATION_MAX_OUTPUT_TOKENS``.

This was a hardcoded 4,096 applied to every model, and it was quietly costing accuracy on the arms
that fold context. When a turn runs past the cap Strands raises, the harness recorded the error, and
the answer was scored as an empty string -- so a model that answered and was interrupted scored the
same as a model that said nothing. Measured on GLM 4.7 Flash: nine truncations in one 60-turn
all-three run, one to two of them on SCORED turns.

The bias is not random. Folded context tells the model to restate figures verbatim, and restating is
what makes an answer long, so the cap fell hardest on exactly the configurations under test. The
partial answer is now recovered from history and scored for what it says, and ``answer_truncated``
marks the turn so a reader can still take the stricter view.

Raising it is also the cheapest lever this harness has on a small model. GLM 4.7 Flash bills output at
$0.40/Mtok: doubling the cap across a 60-turn run costs cents, where a lost scored turn costs a
thirtieth of the accuracy column.

**On a tight-window model that reasoning inverts, and the correction is the point of this note.** The
provider subtracts the requested output cap from the context window before it admits the prompt, so
the cap is not only a ceiling on the answer -- it is a reservation taken out of the window. Measured
on GLM 4.7 Flash (202,752 tokens) with the cap at 8,192, Bedrock refused calls with::

    This model's maximum context length is 202752 tokens. However, you requested 8192 output
    tokens and your prompt contains at least 194561 input tokens, for a total of at least 202753

That same 194,561-token prompt fits with the cap at 4,096. So raising the cap to buy back truncated
answers spends 4,096 tokens of history to do it, and on the arms that do not fold context it converts
calls that would have completed into ``ContextWindowOverflowException``. The lever is genuinely cheap
in dollars and genuinely expensive in window, which is why it belongs in the tight-window
configuration rather than in the default: raise it only as far as the answers actually need, and read
any run that moved it against a run that did not.
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

# --- Window regime -----------------------------------------------------------------
#
# Three of the budgets below are not one number but two, and which pair applies is decided by the
# agent model's context window. That condition is a measurement rather than a preference: a single set
# of values is wrong because the window is what decides whether evidence starves or a larger budget
# only buys a larger bill.

TIGHT_WINDOW_CEILING = 300_000
"""At or below this many tokens, a model is in the tight-window regime.

Not a discovered constant -- a declared boundary, drawn where the measurements change sign. Below it
the window is the binding constraint and evidence starves; above it the window is not reached at all
and the only thing a larger budget buys is a larger bill.

**It is 300K rather than the 250K the class is usually named after, and the reason is a measurement.**
Qwen3 Next has a 256,000-token window -- above 250K -- and measured firmly inside this regime: on the
60-turn script its bare agent lost **92 calls** to context-window overflow and finished 14 turns of 60,
and the graph-alone arm peaked at **131% of the window**. A 250K ceiling would have handed it the
large-window budgets, which are the ones calibrated for a window the conversation never fills.

So the window alone is not really the determinant; the determinant is the window against the payload
mass the workload puts in front of it, and the window is a proxy for it. The ceiling carries headroom
because the proxy is imperfect and the failure is asymmetric: the tight budgets cost tokens on a large
window (measured: +50.2% on Opus 4.8, buying nothing), while the large budgets cost *answers* on a tight
one. Paying the cheaper error is the point of putting the boundary above the highest window measured to
starve rather than at the round number.
"""

CONTEXT_WINDOWS = {
    # Large window: the conversation never approaches the limit on this script.
    "us.anthropic.claude-opus-4-8": 1_000_000,
    "us.anthropic.claude-opus-5": 1_000_000,
    "us.anthropic.claude-sonnet-5": 1_000_000,
    "us.anthropic.claude-fable-5": 1_000_000,
    "us.anthropic.claude-fable-5-1": 1_000_000,
    "us.openai.gpt-6-astra": 1_050_000,
    "global.openai.gpt-6-astra": 1_050_000,
    "us.openai.gpt-5.6-sol": 1_000_000,
    "global.openai.gpt-5.6-sol": 1_000_000,
    "us.amazon.nova-2-lite-v1:0": 1_000_000,
    "global.amazon.nova-2-lite-v1:0": 1_000_000,
    # Tight window: the constraint this harness was extended to measure.
    "zai.glm-5": 200_000,
    "zai.glm-4.7": 202_752,
    "zai.glm-4.7-flash": 202_752,
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": 200_000,
    "global.anthropic.claude-haiku-4-5-20251001-v1:0": 200_000,
    "qwen.qwen3-next-80b-a3b": 256_000,
    "nvidia.nemotron-nano-9b-v2": 128_000,
    "nvidia.nemotron-super-3-120b": 128_000,
}
"""Context window per model id, from each model's Bedrock model card.

Only used to pick a budget regime, never to predict an overflow: the harness measures overflows by
letting them happen. GLM 4.7's 202,752 is the figure Bedrock's own refusal message states, which is
why it is not the round 200K the card implies.

A model absent from this map gets the large-window regime, on the ground that an unknown model is more
likely to be a new frontier model than a small one -- and the regime is overridable, so a wrong guess
costs one environment variable rather than a wrong run.
"""


@dataclass(frozen=True)
class BudgetRegime:
    """The three budgets that differ between window regimes, as one measured set.

    They are held together rather than as three independent knobs because that is how they were
    measured: the Opus 4.8 comparison reverted all three at once, so the aggregate is attributable and
    the individual contributions are not. Splitting them here would claim an attribution the
    measurement does not support.

    Attributes:
        preview_tokens: Relevance-filter preview budget -- how much of an oversized payload survives.
        graph_description_tokens: ``description_tokens`` for a graph installed beside the filter.
        graph_max_retrieval_cycles: Retrieval calls one turn may spend, or ``None`` for unbounded.
    """

    preview_tokens: int
    graph_description_tokens: int
    graph_max_retrieval_cycles: int | None


LARGE_WINDOW = BudgetRegime(
    preview_tokens=800,
    graph_description_tokens=100,
    graph_max_retrieval_cycles=8,
)
"""Budgets for a model whose window the conversation never fills. Measured on Opus 4.8, 1M tokens.

These are the values every published large-window figure was measured with, and reverting to them is
what established that the tight-window pair is not a general improvement. Replaying the five arms on
Opus 4.8 with the tight-window budgets instead moved all three combined from 2,569,888 tokens to
3,858,779 -- **+50.2%** -- for the same 28 of 30 materially correct turns.

The mechanism is visible in the peak call: 49,943 -> 81,065 tokens, while the graph's resolution ladder
barely moved (full 20/9/4 -> 19/8/7). The graph was not folding differently; every rung was carrying
more. On a window this size there was no starvation to cure, so the larger budgets bought nothing and
were billed anyway.

Read the baseline row of that comparison before believing any of it: the baseline runs no plugin, so
none of these values can reach it, and it still drifted +12.5% in tokens and two materially-correct
turns between the two runs. That is the agent choosing a different tool path, and it is the floor below
which nothing in this file is attributable.
"""

TIGHT_WINDOW = BudgetRegime(
    preview_tokens=2_000,
    graph_description_tokens=250,
    graph_max_retrieval_cycles=4,
)
"""Budgets for a model at or below :data:`TIGHT_WINDOW_CEILING`. Measured on GLM 4.7 and GLM 4.7 Flash.

Here the ceiling on the answer is the preview, not the window. ``preview_tokens`` of 800 admitted one
or two chunks of a payload ten to thirty times its size, so a literal the answer needed -- an enum like
``DEGRADED``, an error class like ``MFA_CHALLENGE_TIMEOUT``, a figure like ``907,35`` -- simply was not
in the chunks that made the cut, while the all-three arm's peak call sat at 57,259 tokens against
roughly 199,000 usable. 71% of the window was going unused while the plugin starved the evidence.

``graph_description_tokens`` follows it rather than leading: the graph derives a Card's numeric lines
from the message, and by then the message carries the preview. The two budgets are in series --

    payload -> preview budget -> the message -> the Card's numeric lines -> Description budget

-- so raising the second while the first starves buys nothing, which is the measured reason the
earlier calibration of 100 here was correct *for* a preview of 800 and wrong once it was raised.
"""

WINDOW_REGIME = (_env("VALIDATION_WINDOW_REGIME") or "").strip().lower() or (
    "tight"
    if (CONTEXT_WINDOWS.get(AGENT_MODEL_ID) or TIGHT_WINDOW_CEILING + 1) <= TIGHT_WINDOW_CEILING
    else "large"
)
"""Which regime this run is in: ``tight`` or ``large``, derived from the model unless overridden.

Derived rather than configured so the common case is right without anyone remembering to set it, and
overridable with ``VALIDATION_WINDOW_REGIME`` so the boundary itself can be measured -- running a
large-window model in the tight regime is exactly the experiment that produced the figures in
:data:`LARGE_WINDOW`.
"""

BUDGETS = TIGHT_WINDOW if WINDOW_REGIME == "tight" else LARGE_WINDOW
"""The regime's budgets. Individual values are still overridable one by one from the environment."""

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

    **Regime-dependent, and the default here is not the one that applies.** The value in force comes
    from :data:`BUDGETS` -- 800 on a large window, 2,000 on a tight one -- and the two are documented
    at :data:`LARGE_WINDOW` and :data:`TIGHT_WINDOW`. The literal below is only what the dataclass
    falls back to when constructed with no argument, which the harness never does.

    The preview is verbatim: ``RelevancePreview`` selects whole chunks that are exact substrings and
    inserts only gap markers, so nothing here is lost to paraphrase. What a small budget loses is
    *selection*, and what a large one costs is every downstream rung carrying more.
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
    # The graph's knobs are NOT here: they live in GRAPH_TUNING below, as one set for every arm the
    # graph appears in.
    min_cards: int = 3
    """Below this many Cards the whole choice is skipped: the only decision is 'send it all'."""

    catalog_in_system_prompt: bool = False
    """Place the disclosure catalog in the system prompt instead of in the tool schema.

    With this on, ``toolConfig`` carries only the tools that are callable on the call -- the search tool,
    the always-available ones, the exposed ones and the referenced ones -- and every other name arrives
    as a prose listing under a header stating the rule.

    Two defects motivate it, both located by reading the projection rather than by a run. A catalog entry
    in ``toolConfig`` declares ``{"type": "object", "properties": {}}``, which a model reads as a tool
    that takes no arguments, and the statement that the entry is incomplete lives in the SEARCH tool's
    description -- a different place from the entry being read at the moment of the decision. Measured on
    Opus 4.8 over 60 turns, the combined arm made 22 searches AND 19 premature cancellations: the model
    follows the instruction and guesses at the same time, and each guess is a round trip of about 31,000
    tokens carrying no information.

    Unmeasured. It is off by default because every published figure was measured with the catalog in the
    tool schema, and because a name outside ``toolConfig`` is a name the provider does not know: a model
    that calls one anyway may be refused by the provider before the plugin's guard is reached, which is a
    harder failure than the cancellation it replaces.
    """


THRESHOLDS = Thresholds(
    max_result_tokens=_env_int("VALIDATION_MAX_RESULT_TOKENS", 4_000),
    preview_tokens=_env_int("VALIDATION_PREVIEW_TOKENS", BUDGETS.preview_tokens),
    chunk_tokens=_env_int("VALIDATION_CHUNK_TOKENS", 500),
    relevance_threshold=_env_float("VALIDATION_RELEVANCE_THRESHOLD", 0.02),
    catalog_tokens=_env_int("VALIDATION_CATALOG_TOKENS", 20),
    ttl_cycles=_env_int("VALIDATION_TTL_CYCLES", 5),
    top_k=_env_int("VALIDATION_TOP_K", 4),
    min_cards=_env_int("VALIDATION_MIN_CARDS", 3),
    catalog_in_system_prompt=_env_bool("VALIDATION_CATALOG_IN_SYSTEM_PROMPT", False),
)


@dataclass(frozen=True)
class GraphTuning:
    """The graph's knobs, as ONE set. There used to be two, selected by whether the filter was installed.

    **The split is gone, and the reason is a design argument rather than a new measurement.** It existed
    because the graph "should fold less when relevance has already compressed the evidence" -- which
    treats the two plugins as rivals for one job. They are not. They act at different moments on
    different material:

    - The relevance filter acts on ``AfterToolCallEvent``, on a payload that has not entered the history
      yet. Its job is to decide what of that payload is worth keeping. Whether the survivor then goes to
      the history, to the graph, or nowhere is not its concern.
    - The graph acts at delivery, on a history that already exists. It never sees a payload; it sees
      whatever was written down.

    So the graph's input is *smaller* with the filter installed, not *different in kind*, and a knob that
    decides how aggressively to fold a history has no business reading whether some other plugin trimmed
    that history first. One set, and the filter's output is simply what the graph is given.

    The measurement that justified the split is also confounded, which is what made it safe to drop.
    Applying the graph-alone values with the filter present lost five materially correct turns -- but that
    run had ``preview_tokens`` at 800 against payloads ten to thirty times that, so the Cards were starved
    by the FIRST cut in the chain and a larger Description budget had nothing left to preserve. The
    preview is 2,000 now, so the condition that produced the result no longer holds.

    Unmeasured as a unified set: the values below are the graph's own measured optimum, which was measured
    with no filter beside it. Every knob is env-overridable, so a sweep can settle it without a commit.

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
    reuse_ttl_cycles: int = 5
    """Cycles a Card retrieved by a tool stays elevated for. The package default, restated here so a
    sweep can reach it: on a tight-window model a longer reuse keeps recovered evidence resident
    across the follow-up questions that usually come right after a retrieval, and a shorter one
    stops a single retrieval from pinning content for the rest of the line."""
    tags_per_card: int = 5
    """Tags derived per Card, which is what ``find_context`` matches on. Reachable from a sweep
    because the graph's discovery path is only as good as the tags it searches."""
    neighbors_per_candidate: int = 3
    """``similar`` neighbours ``find_context`` lists under each candidate. ``0`` lists none.

    The edge had no reader before this: measured on the write path, stored with its similarity as the
    weight, omitted from ``_STRUCTURAL_WEIGHTS`` so it propagates no Note, and traversed by no retrieval
    path. It answers what the candidate ranking cannot -- that ranking scores each Description against
    the QUESTION and never against another Description, so two turns covering the same ground in
    different words are invisible to each other in it.

    Unmeasured: every published figure was produced with the edge unread, so a run with this above zero
    is not comparable to them on tokens. ``VALIDATION_GRAPH_NEIGHBORS=0`` reproduces them."""


GRAPH_TUNING = GraphTuning(
    expand_threshold=_env_float("VALIDATION_GRAPH_EXPAND", 0.62),
    collapse_floor=_env_float("VALIDATION_GRAPH_COLLAPSE", 0.45),
    link_threshold=_env_float("VALIDATION_GRAPH_LINK", 0.50),
    description_tokens=_env_int("VALIDATION_GRAPH_DESCRIPTION_TOKENS", BUDGETS.graph_description_tokens),
    body_budget=_env_opt_int("VALIDATION_GRAPH_BODY_BUDGET", 40_000),
    max_retrieval_cycles=_env_opt_int("VALIDATION_GRAPH_MAX_RETRIEVAL_CYCLES", BUDGETS.graph_max_retrieval_cycles),
    reuse_ttl_cycles=_env_int("VALIDATION_GRAPH_REUSE_TTL", 5),
    tags_per_card=_env_int("VALIDATION_GRAPH_TAGS", 5),
    neighbors_per_candidate=_env_int("VALIDATION_GRAPH_NEIGHBORS", 3),
)
"""The graph's tuning, for every arm it appears in. Raised off the package's defaults.

One hypothesis behind the thresholds: move mass off the full-content rung and onto a Description rich
enough to carry the facts a check asks for.

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

``body_budget`` is 40,000, and the combined arm used to run it at ``None`` on the grounds that its "peak
call sat at 49,863 tokens -- below any ceiling worth setting". **That premise is dead.** The same arm on
Opus 4.8 now peaks at 75,000-81,000, and the peak call's composition says the growth is entirely message
mass: 25,598 -> 38,507 tokens of messages against a flat 7,963 of tool schema.

``None`` is not "no ceiling", it is *the step-down turned off*. ``distribute`` only moves a Card down a
rung when the remaining budget cannot fit it (``scoring.py:382``: ``remaining is None or cost <=
remaining``), so with ``None`` every Card at or above ``expand_threshold`` travels at full content however
many of them there are. ``expand_threshold`` cannot substitute for the ceiling -- it is a per-Card
classifier and knows nothing about the total, so the call grows linearly with the conversation and is
bounded by nothing. The vended harness carries a ``gr-budget-40k`` variant for exactly this reason, "a
ceiling that actually binds, so the budget-driven step down is exercised", and the combined arm was the
one running without it. A Card that does not fit steps down ONE rung, to Description, never to Title, so
the ceiling binding costs a Description on the lowest-Note Card of the turn rather than a dropped Card.
``VALIDATION_GRAPH_BODY_BUDGET=none`` restores the measured configuration.

``max_retrieval_cycles`` is 4 against the package's 8, and the distribution is why. Measured on GLM 4.7
Flash over 60 turns, the combined arm's retrieval spend per turn was 1 call on 13 turns, 2 on 5, 3 on 2,
then 5, 8 and 12 -- so 87% of the turns that retrieved at all finished inside three calls, and the whole
tail above five is two turns. Eight therefore binds on nothing a healthy turn does while leaving the
pathological turn eight real Card rebuilds to spend; four keeps every turn in that 87% untouched and
halves what the outlier costs.

One replay each, so read every quantity here as the direction it points rather than as a settled number.
``--repeats 3`` is what would settle it.
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
    "no-disclosure": RunConfig(
        name="no-disclosure",
        disclosure=False,
        relevance=True,
        graph=True,
        label="Relevance + graph (disclosure removed)",
        notes=(
            "Leave-one-out. Tuning three plugins together is only possible once each one's MARGINAL "
            "contribution inside the stack is known, and a single-strategy arm does not give that: "
            "what a plugin buys on its own and what it adds to the other two are different "
            "quantities. This arm prices the schema floor against the round trips disclosure costs "
            "when the model guesses a hidden tool's name instead of searching for it."
        ),
    ),
    "no-relevance": RunConfig(
        name="no-relevance",
        disclosure=True,
        relevance=False,
        graph=True,
        label="Graph + disclosure (relevance removed)",
        notes=(
            "Leave-one-out. Every arm with the graph now keeps its own artifact tool: the "
            "``include_artifact_tool=not config.relevance`` drop is gone, because the filter no "
            "longer registers a retrieval tool for it to collide with."
        ),
    ),
    "no-graph": RunConfig(
        name="no-graph",
        disclosure=True,
        relevance=True,
        graph=False,
        label="Relevance + disclosure (graph removed)",
        notes=(
            "Leave-one-out. The two payload-side strategies without any history folding, which is "
            "what isolates whether the graph is adding recall or only removing tokens."
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

CONFIGURATIONS = DEFAULT_CONFIGURATIONS + (
    "no-disclosure",
    "no-relevance",
    "no-graph",
)
"""Every configuration the runner accepts by name.

The three leave-one-out arms are accepted but not run by default: they answer a tuning question
rather than the comparison the report is built around, and adding them to the default would change
what every published table means.

**What they measured, and it is the most useful thing in this file for anyone tuning the stack.** On
``zai.glm-4.7`` (a 202,752-token window, so the tight-window class), 20 turns of which 18 are scored,
two to three replays each:

======================================  =============  =========  ==============
Configuration                           Total tokens   Δ tokens   Materially correct
======================================  =============  =========  ==============
Baseline, no plugin                         3,733,922         --   12.5/18
All three                                   1,687,068     -54.8%   15.67/18
Graph + disclosure                          1,524,740     -59.2%   15.5/18
Relevance + graph (no disclosure)           3,960,855     +6.1%    14.5/18
Relevance + disclosure (no graph)           4,222,162     +13.1%   14.5/18
======================================  =============  =========  ==============

**The saving is a conjunction, not a sum.** Either PAIR spends more than using no plugin at all, and
only the full stack saves. Drop disclosure and the fixed tool-schema floor rides every call again,
multiplied by the extra retrieval round trips the other two introduce. Drop the graph and the history
never folds, so previews accumulate and the round trips are paid on a conversation that only grows.
Neither pair is a degraded version of the stack; both are worse than doing nothing.

Read the baseline's token column with the error column beside it, which is the whole reason they sit
together: that baseline overflowed the window six times per replay, and a turn that overflows stops
spending. Part of why the pairs look expensive against it is that they finish turns it abandoned. The
comparison that is not contaminated by this is all-three against either pair, and there the full stack
wins on both axes at once.
"""

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
