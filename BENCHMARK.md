# Benchmark — the same agent on six models, with and without prompt caching

The figures on the [landing page](README.md) are one model with prompt caching off. This page is the
whole measurement behind them: **nine runs, five configurations each, 45 cells**, across six models
from four providers, with the caching-on counterpart of every Claude run.

It exists because two of the conclusions flip depending on the model you run, and a reader who only
sees the headline cannot tell which regime they are in.

## What was run

The same **60-turn scripted conversation**, replayed once per configuration. 30 of the 60 turns are
scored against deterministic ground truth; the other 30 are unscored mass, there to grow the history.
The five configurations are the bare agent, each practice alone, and all three together. At list prices
the whole battery is the sum of the Cost column below — about **$2,260**, which is what reproducing it
costs and most of it is the bare-agent arms.

| Model | Model id | Context window | Caching in this benchmark |
|---|---|---:|---|
| Claude Opus 4.8 | `us.anthropic.claude-opus-4-8` | 1M | off **and** on |
| Claude Opus 5 | `us.anthropic.claude-opus-5` | 1M | off **and** on |
| Claude Fable 5 | `us.anthropic.claude-fable-5` | 1M | off **and** on |
| GPT-6 Astra | `us.openai.gpt-6-astra` | 1.05M | implicit, cannot be turned off |
| GPT-5.6 Sol | `us.openai.gpt-5.6-sol` | 1M | implicit, cannot be turned off |
| GLM 5 | `zai.glm-5` | 200K | none published |

Context windows and caching support are from each model's Bedrock model card: [Opus
4.8](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-opus-4-8.html),
[Opus 5](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-opus-5.html),
[Fable 5](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-fable-5.html),
[Astra](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-6-astra.html),
[Sol](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-56-sol.html),
[GLM 5](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-zai-glm-5.html). The three
Claude cards list both Implicit and Explicit Prompt Caching on `bedrock-runtime`; GLM 5's card lists
neither, and also no cross-Region inference profile, which is why it is invoked by its bare model id.

Caching, where it was on, used Bedrock's default **5-minute TTL** with checkpoints placed by the SDK.
Rates come from `MODEL_PRICING` in
[`validation/community-plugin-A-B-D/src/config.py`](validation/community-plugin-A-B-D/src/config.py),
read off the published pricing page and model cards on 2026-09-21. See
[Prices](#prices-and-what-they-are-based-on) below.

## How to read the table

Twelve columns, and the two that decide everything are the cache pair.

| Column | What it is | What to do with it |
|---|---|---|
| **Uncached input** | Input tokens billed at the full input rate | On a cached run this is near zero *by construction*, not because the prompt was small — see the note below |
| **Cache read** | Input tokens served from the cache, billed at ~0.10x input | High is good: the prefix is being reused |
| **Cache write** | Input tokens written to the cache, billed at ~1.25x input | This is the expensive band. Written and never re-read = 25% *more* than not caching at all |
| **read:write** | The ratio of the two | **The single best predictor of cost in this table.** Above ~5 is cheap, below ~2 is expensive, with no exception across the 45 cells |
| **Billed tokens** | The three input columns summed | Compare *within* a run, never across runs — see trap 2 |
| **Cost** | Billed units x the declared rates, auxiliary embedding and rerank calls included | The number to decide on |
| **Accuracy** | Fraction of expectation *weight* met, with partial credit | Read together with the next column, never alone |
| **Correct** | Turns with no *critical* failure, out of 30 scored | The stricter reading; a config can gain accuracy and lose a turn |
| **s/turn** | Mean seconds per turn | Relative only: the five configurations ran concurrently |
| **$/correct** | Cost ÷ correct turns | Puts cost and quality on one axis. A ranking, not a measurement — the denominator is 30 |

🏆 marks the best `$/correct` of that run. `—` in read:write means no cache traffic at all.

**On a cached run, `inputTokens` stops measuring input.** Bedrock reports cached input separately, and
the documented identity is `total input tokens = inputTokens + cacheReadInputTokens +
cacheWriteInputTokens` ([Prompt
caching](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html)). Since the SDK puts
the cache checkpoint at the end of the prompt, every token falls into read or write and the uncached
column keeps only structural residue — 174 to 344 tokens for a 60-turn conversation. A report that
quotes that column alone reports almost nothing; this table sums all three.

### Three traps in this data

**1. A truncated run looks cheap.** An arm that died of window overflow sends fewer tokens because it
stopped working. Cells marked `*` are truncated, and a percentage taken against them is fiction.

**2. Billed tokens are not comparable between runs.** The agent picks its own tool path, and one extra
tool call early in a 60-turn conversation rides along in every later call. Opus 5's baseline billed 35%
*more* tokens with caching on than off for exactly this reason — seven more tool calls, not a caching
artefact. Within a run all five configurations replay the same script, so the comparison holds; across
runs, compare cost and ratios rather than token mass.

**3. One replay.** Accuracy differences of two or three turns are inside the noise floor of a single
replay on a non-deterministic agent. Only the window-regime gap (GLM 5) is large enough to read as a
quality signal; in the eight large-window runs the accuracy column spans 88.2% to 100% while the cost
column spans 48x.

## The table

| Model | Cache | Configuration | Uncached input | Cache read | Cache write | read:write | Billed tokens | Cost | Accuracy | Correct | s/turn | $/correct |
|---|:--:|---|---:|---:|---:|---:|---:|---:|---:|:--:|---:|---:|
| **Opus 4.8** | off | Baseline | 14,346,683 | 0 | 0 | — | 14,346,683 | $72.50 | 96.9% | 28/30 | 11.7 | $2.59 |
|  |  | Relevance | 12,735,943 | 0 | 0 | — | 12,735,943 | $64.44 | 96.9% | 29/30 | 11.7 | $2.22 |
|  |  | Disclosure | 9,427,523 | 0 | 0 | — | 9,427,523 | $48.10 | 94.5% | 27/30 | 12.8 | $1.78 |
|  |  | Graph | 10,345,220 | 0 | 0 | — | 10,345,220 | $52.43 | 93.7% | 27/30 | 10.8 | $1.94 |
|  |  | 🏆 All three | 2,538,860 | 0 | 0 | — | 2,538,860 | $13.35 | 96.1% | 28/30 | 8.7 | $0.48 |
| **Opus 4.8** | **on** | 🏆 Baseline | 174 | 14,842,278 | 215,531 | 68.9 | 15,057,983 | $9.49 | 94.5% | 27/30 | 8.0 | $0.35 |
|  |  | Relevance | 204 | 15,701,512 | 211,522 | 74.2 | 15,913,238 | $10.08 | 94.5% | 27/30 | 10.2 | $0.37 |
|  |  | Disclosure | 202 | 5,578,574 | 3,679,071 | 1.5 | 9,257,847 | $26.65 | 94.5% | 27/30 | 10.0 | $0.99 |
|  |  | Graph | 576,260 | 8,205,206 | 1,521,353 | 5.4 | 10,302,819 | $17.29 | 99.2% | 29/30 | 9.2 | $0.60 |
|  |  | All three | 328,419 | 1,067,190 | 2,253,984 | 0.5 | 3,649,593 | $17.13 | 94.5% | 27/30 | 10.4 | $0.63 |
| **Opus 5** | off | Baseline | 19,700,954 | 0 | 0 | — | 19,700,954 | $100.44 | 89.8% | 24/30 | 24.3 | $4.18 |
|  |  | Relevance | 18,039,650 | 0 | 0 | — | 18,039,650 | $91.88 | 96.9% | 28/30 | 21.4 | $3.28 |
|  |  | Disclosure ⚠ | 646,854 | 0 | 0 | — | 646,854 | $3.25 | 31.5% | 1/30 | 1.8 | *invalid* |
|  |  | Graph | 13,416,010 | 0 | 0 | — | 13,416,010 | $69.54 | 93.7% | 25/30 | 26.0 | $2.78 |
|  |  | 🏆 All three | 5,492,513 | 0 | 0 | — | 5,492,513 | $29.15 | 100.0% | 30/30 | 17.9 | $0.97 |
| **Opus 5** | **on** | Baseline | 196 | 26,321,602 | 285,328 | 92.3 | 26,607,126 | $17.16 | 92.1% | 24/30 | 23.2 | $0.71 |
|  |  | 🏆 Relevance | 200 | 17,343,538 | 237,313 | 73.1 | 17,581,051 | $12.05 | 92.9% | 25/30 | 19.7 | $0.48 |
|  |  | Disclosure | 218 | 10,217,732 | 6,296,888 | 1.6 | 16,514,838 | $46.07 | 92.1% | 26/30 | 18.7 | $1.77 |
|  |  | Graph | 664,269 | 9,508,166 | 2,898,117 | 3.3 | 13,070,552 | $28.34 | 94.5% | 27/30 | 22.0 | $1.05 |
|  |  | All three | 363,512 | 1,526,446 | 3,470,771 | 0.4 | 5,360,729 | $25.96 | 95.3% | 27/30 | 18.3 | $0.96 |
| **Fable 5** | off | Baseline | 21,579,745 | 0 | 0 | — | 21,579,745 | $217.86 | 100.0% | 30/30 | 20.5 | $7.26 |
|  |  | Relevance | 17,227,288 | 0 | 0 | — | 17,227,288 | $174.07 | 97.6% | 29/30 | 18.1 | $6.00 |
|  |  | Disclosure | 14,241,315 | 0 | 0 | — | 14,241,315 | $144.58 | 97.6% | 29/30 | 20.5 | $4.99 |
|  |  | Graph | 12,576,521 | 0 | 0 | — | 12,576,521 | $128.54 | 96.1% | 28/30 | 21.0 | $4.59 |
|  |  | 🏆 All three | 5,249,763 | 0 | 0 | — | 5,249,763 | $55.31 | 96.9% | 28/30 | 22.1 | $1.98 |
| **Fable 5** | **on** | Baseline | 190 | 18,254,571 | 250,858 | 72.8 | 18,505,619 | $23.38 | 99.2% | 30/30 | 16.1 | $0.78 |
|  |  | 🏆 Relevance | 212 | 17,687,881 | 229,005 | 77.2 | 17,917,098 | $22.47 | 98.4% | 29/30 | 15.7 | $0.77 |
|  |  | Disclosure | 248 | 8,001,876 | 6,708,508 | 1.2 | 14,710,632 | $93.96 | 98.4% | 29/30 | 19.3 | $3.24 |
|  |  | Graph | 812,339 | 10,368,477 | 2,528,084 | 4.1 | 13,708,900 | $53.19 | 98.4% | 29/30 | 21.7 | $1.83 |
|  |  | All three | 360,961 | 1,291,253 | 3,567,890 | 0.4 | 5,220,104 | $52.57 | 96.1% | 27/30 | 23.3 | $1.95 |
| **GPT-6 Astra** | implicit | Baseline | 228 | 15,270,589 | 213,127 | 71.7 | 15,483,944 | $21.20 | 92.9% | 26/30 | 10.4 | $0.82 |
|  |  | 🏆 Relevance | 222 | 12,069,886 | 185,578 | 65.0 | 12,255,686 | $17.32 | 88.2% | 23/30 | 10.2 | $0.75 |
|  |  | Disclosure | 252 | 7,810,974 | 5,534,968 | 1.4 | 13,346,194 | $86.18 | 92.9% | 26/30 | 10.3 | $3.31 |
|  |  | Graph | 252 | 1,473,390 | 8,536,045 | 0.2 | 10,009,687 | $120.58 | 91.3% | 25/30 | 10.9 | $4.82 |
|  |  | All three | 292 | 17,588 | 4,051,747 | 0.0 | 4,069,627 | $57.37 | 94.5% | 27/30 | 9.2 | $2.12 |
| **GPT-5.6 Sol** | implicit | 🏆 Baseline | 242 | 15,679,917 | 218,227 | 71.9 | 15,898,386 | $8.82 | 97.6% | 29/30 | 9.0 | $0.30 |
|  |  | Relevance | 276 | 17,145,788 | 208,099 | 82.4 | 17,354,163 | $9.55 | 97.6% | 29/30 | 10.3 | $0.33 |
|  |  | Disclosure | 340 | 11,120,494 | 12,432,437 | 0.9 | 23,553,271 | $73.82 | 96.9% | 28/30 | 10.8 | $2.64 |
|  |  | Graph | 278 | 790,589 | 10,825,077 | 0.1 | 11,615,944 | $60.56 | 91.3% | 25/30 | 9.9 | $2.42 |
|  |  | All three | 344 | 17,861 | 6,811,669 | 0.0 | 6,829,874 | $38.54 | 96.1% | 28/30 | 10.0 | $1.38 |
| **GLM 5** | n/a | Baseline \* | 5,247,478 | 0 | 0 | — | 5,247,478 | $5.26 | 63.8% | 14/30 | 12.7 | *invalid* |
|  |  | Relevance | 19,383,574 | 0 | 0 | — | 19,383,574 | $19.46 | 98.4% | 29/30 | 48.8 | $0.67 |
|  |  | Disclosure \* | 11,309,551 | 0 | 0 | — | 11,309,551 | $11.37 | 74.8% | 20/30 | 30.1 | *invalid* |
|  |  | Graph | 8,884,923 | 0 | 0 | — | 8,884,923 | $8.99 | 85.0% | 23/30 | 28.1 | $0.39 |
|  |  | 🏆 All three | 2,432,543 | 0 | 0 | — | 2,432,543 | $2.49 | 92.1% | 26/30 | 12.0 | $0.10 |

**\* GLM 5 — the two marked cells are truncated, not cheap.** Against a 200K window the bare agent lost
**90 calls** to `ContextWindowOverflowException` and disclosure lost **38**; their token and cost
figures are low because the run stopped answering. Only `relevance`, `graph` and `all` completed the 60
turns, and the comparison is valid between those three.

**⚠ Opus 5 / disclosure, cache off — not a result.** The model never called `find_tools`, so it ran
with no tool schema and answered almost nothing (1 of 30, 1.8s per turn). The same arm with caching on
scored 26 of 30. This is path variance, not a property of the plugin.

## What the numbers say

### 1. With caching off, the full stack wins on every model that fits

Five models, one ranking: all three combined is the cheapest arm, by 72% to 82%, and its accuracy stays
within the replay noise of the bare agent — on Opus 5 it was the *best* arm at 30 of 30. This is the
regime the landing page reports, and the one the practices were designed for.

### 2. With caching on, the ranking inverts — and read:write says why

Cache reads cost about **0.10x** the input rate and cache writes about **1.25x**, a factor of ~12
between the two bands ([Bedrock pricing](https://aws.amazon.com/bedrock/pricing/); the
[prompt-caching page](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html) states
that written tokens "can be billed at a rate that is higher than the standard input token rate", for
implicit and explicit caching alike).

A bare agent only ever *appends* to its prompt, so its prefix is stable and it reads nearly everything
back: read:write of 69 to 92. The compressing plugins rewrite the prompt, so they write continuously
and read little: 0.4 to 1.6. That is the whole inversion. On Opus 4.8 the bare agent falls from $72.50
to $9.49 by switching one flag, while all three combined *rises* from $13.35 to $17.13 — the
recommended configuration becomes the more expensive one.

Expressed as an effective rate — cost divided by billed prompt tokens, which does not move with the
agent's path — the same thing reads as a price band per configuration:

| Model | Cache | Baseline | Relevance | Graph | Disclosure | All three | Uncached input rate |
|---|:--:|---:|---:|---:|---:|---:|---:|
| Opus 4.8 | off | $5.05 | $5.06 | $5.07 | $5.10 | $5.26 | $5.00 |
| Opus 4.8 | **on** | **$0.63** | **$0.63** | $1.68 | $2.88 | $4.69 | $5.00 |
| Opus 5 | **on** | **$0.64** | **$0.69** | $2.17 | $2.79 | $4.84 | $5.00 |
| Fable 5 | **on** | **$1.26** | **$1.25** | $3.88 | $6.39 | $10.07 | $10.00 |
| GPT-6 Astra | implicit | **$1.37** | **$1.41** | $12.05 | $6.46 | $14.10 | $11.00 |
| GPT-5.6 Sol | implicit | **$0.55** | **$0.55** | $5.21 | $3.13 | $5.64 | $4.40 |
| GLM 5 | n/a | $1.00 | $1.00 | $1.01 | $1.01 | $1.02 | $1.00 |

*Two readings validate the method and one is the finding.* The uncached rows land within 1-5% of the
published rate, and GLM 5 within 1% — so the accounting is right. The finding is the right-hand cells:
on Astra and Sol, all three combined pays **28% more per token than not caching at all**, which is
exactly the 1.25x write multiplier arriving on almost every token sent.

### 3. Relevance filtering is the one practice that is compatible with caching

Its effective rate is indistinguishable from the bare agent's on all five caching models, and it is
🏆 in four of the nine runs. The reason is mechanical, and the documentation states the rule: cache
checkpoints are processed `tools` → `system` → `messages`, and "changing content in an earlier section
invalidates the cache for later sections"
([Prompt caching](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html)). So:

| Practice | What it changes in the prompt | Effect on the cache |
|---|---|---|
| Bare agent | appends to the end only | prefix always matches — cheapest possible reads |
| **Relevance filtering** | replaces a payload **as it arrives**, then never touches it | still append-only — cache intact |
| Progressive tool disclosure | edits `toolConfig`, the **first** section | every change invalidates the whole prompt |
| Context graph | removes messages from the **middle** of the history | everything from that point on is new |

Measured on Opus 5 with caching on: 44 of the disclosure arm's 109 calls came back with `cacheRead = 0`
— total rewrites — and those 44 carry 98.4% of its 6.3M cache writes. The disclosure plugin did its job
(the tool schema fell from 62,656 tokens to 5,300-11,606); it is the *editing* that costs, not the
size. The rule that follows is a design criterion: a context plugin is cache-compatible if and only if
its mutations are append-only or confined to the end of the prompt.

### 4. On a small window, the practices stop being an optimisation

GLM 5 is the only model here under 1M tokens, and it reverses the question. Peak input on a single call:
baseline 198,588 against a 200K window, all three combined 43,490. The bare agent lost 90 calls and 16
scored turns to overflow; the full stack lost none. Cost stops being the argument — the argument is that
the conversation finishes.

This is also the regime where caching does not compete: GLM 5's card publishes no caching of either
type, so there is nothing to weigh the plugins against.

### 5. Caching makes the agent cheaper, not better

The model sees an identical prompt whether or not it was served from cache, so accuracy must be
unchanged in principle — and measured, it moves by up to three turns in both directions with no
mechanism, which is the single-replay noise floor. Do not read "cache on, 27 of 30" against "cache off,
28 of 30" as a quality difference.

### 6. Compression and caching optimise the same redundancy

On the caching runs, **amplification** — billed prompt tokens divided by the tokens that were written
to cache exactly once, i.e. how many times the same content was re-sent — is **66x to 83x** for the
bare agent and the relevance filter, and **1.0x to 1.6x** for all three combined. The full stack barely
re-sends anything, which is the point; but it also means the cache has nothing cheap left to re-read
and collects only expensive writes. The two techniques address the same waste from opposite ends, and
only one of them can bill it. That is why they are alternatives rather than a stack — with relevance
filtering the exception, because it compresses once and then stops changing the prompt.

## Prices and what they are based on

Every dollar figure here is `measured units x a published list rate`. Nothing in this document is
derived from an actual bill, and no discount, commitment or account-specific fee is modelled — so read
the cost columns as list cost, and the comparisons between them (which is what the findings rest on) as
ratios between rows priced identically.

| Claim | Status | Source |
|---|---|---|
| Per-model input, output and cache rates | published | [Bedrock pricing](https://aws.amazon.com/bedrock/pricing/) and each model card |
| Cache writes are billed above the input rate, implicit **and** explicit | documented | [Billing for cached tokens](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html) |
| Cache write = **1.25x** input, read = 90% discount, on GPT-5.6 | documented | [Prompt caching for models from OpenAI](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html) |
| Astra's cache write $13.75 against $11.00 input (1.25x) | published | [Astra model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-6-astra.html) |
| Claude cache-read at 0.10x and write at 1.25x / 2.00x input | published for the models listed, **inferred** for Haiku 4.5, whose cache rates are not on the pricing table | `MODEL_PRICING` in [`src/config.py`](validation/community-plugin-A-B-D/src/config.py) |
| Token counts, call counts, accuracy, timings | measured by the harness from the API's own `usage` | the run JSON under `results/` |

Correcting a rate never needs another run: rates live in one map and the report re-renders from the
recorded JSON.

### What is not settled

**Caching on the Converse path for the OpenAI models.** Both cards list Implicit and Explicit Prompt
Caching as *Responses API only*, yet these runs used Converse and came back with cache read and write
counts anyway — measured on Astra's first baseline call as `inputTokens=2,
cacheWriteInputTokens=48,583`. The *rate* is documented (1.25x); what is undocumented is that the
traffic happens at all on this path, so the Astra and Sol columns price a documented rate against
undocumented behaviour.

**The 5-minute TTL is the best case.** Turns in this benchmark land 8-24 seconds apart, so nearly every
call hit a warm cache. A human who pauses longer than the TTL between turns gets the opposite: the
prefix is rewritten each turn at the write rate, which is worse than not caching. A 1-hour TTL exists
for that case — `"ttl": "1h"` on the `cachePoint`, supported by all three Claude models here per the
[supported-models table](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html) — at
a higher write rate, and the pinned botocore in this harness cannot send it. Neither was measured.

**Cross-Region routing stayed put.** Every model was served by exactly one Region for the whole day
(read from `inferenceRegion` in the invocation logs), including Sol and Astra, which were served from
`us-east-2` and `us-west-2` while being called from `us-east-1` — so cache reads do work across a Geo
CRIS hop. The documentation warns that "at times of high demand, these optimizations may lead to
increased cache writes"
([Cross-Region inference](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html)),
and that case did not occur here. Global CRIS was not tested at all.

**These are the community packages.** Every mechanism above was measured on
`validation/community-plugin-A-B-D/` with the three packages installed from this repository. The
forked-SDK vended plugins place their cache checkpoints in their own code and were **not** re-measured;
nothing on this page transfers to them without a run.

## Reproducing this

```bash
cd validation/community-plugin-A-B-D
./run.sh --total-turns 60 --tag myrun                    # cache off
./run.sh --total-turns 60 --cache default --tag myrun-cache
```

`--cache default` uses Bedrock's own TTL, which is 5 minutes. An explicit `--cache 5m` or `--cache 1h`
is rejected by the pinned botocore 1.40, whose Converse model declares `cachePoint` with `type` alone —
so the 1-hour TTL needs a newer botocore and was not measured.

Correcting a rate never requires another run against Bedrock — edit `MODEL_PRICING` and re-render:

```bash
python -m src.run --report-only results/run-myrun.json
```

See [`validation/community-plugin-A-B-D/README.md`](validation/community-plugin-A-B-D/README.md) for
what the harness measures and how scoring works.
