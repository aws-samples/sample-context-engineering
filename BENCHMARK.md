# Benchmark — the same agent on six models, with and without prompt caching

The figures on the [landing page](README.md) are one model with prompt caching off. This page is the
whole measurement behind them: **nine runs, five configurations each, 45 cells**, across six models
from four providers, with the caching-on counterpart of every Claude run — followed by a **second study
of twelve further cells** on the tight-window class, three models from three different vendors, which
asks what happens where the window is the binding constraint rather than the bill.

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

GLM 5 was the only model here under 1M tokens, which made this finding rest on a single model until the
[tight-window study](#a-second-study-the-tight-window-class-on-three-vendors) below added three more from
three different vendors. It reproduces on all three, and sharpens it: the bare agent is not the only arm
that overflows in this class, and which *single* practice keeps a model inside its window depends on the
model.

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

## A second study: the tight-window class, on three vendors

The table above is six models, five of them with a window of 1M tokens, and its argument is cost. This
second study asks the other question: **what happens in the class where the window is the binding
constraint**, and does the answer depend on whose model it is. Three vendors, one class, same 60-turn
script, same 30 scored turns, prompt caching **off** in every row — two of the three models publish no
caching at all, so there is nothing to turn on.

| Vendor | Model | Window |
|---|---|---:|
| Zhipu | `zai.glm-4.7` | 202,752 |
| Anthropic | `us.anthropic.claude-haiku-4-5` | 200,000 |
| Alibaba | `qwen.qwen3-next-80b-a3b` | 256,000 |

One column is new, and it is the one that decides how to read the rest. **Peak input** is the largest
single call the arm sent, as a percentage of that model's window — the one quantity in this study that
truncation cannot contaminate, because it is measured on the calls that *did* go out. **Completed** is
how many of the 60 turns produced an answer at all. The twelve columns of the table above describe what
a run spent and scored; neither of them says whether the conversation survived, and in this class that
is the question.

A cell marked `*` is truncated in the sense of trap 1 — it spent less and scored less because it stopped
working. **In this study the truncated cells are not confined to the baseline**, which is what makes the
accuracy column dangerous here: on GLM 4.7 three of the five arms overflowed.

| Model | Configuration | Peak input | Total tokens | Δ tokens | Accuracy | Correct | Completed | Cost | $/correct |
|---|---|---:|---:|---:|---:|:--:|:--:|---:|---:|
| **GLM 4.7** | Baseline * | 85% | 5,814,709 | — | 65.3% | 14/30 | 22/60 | $3.52 | *$0.25* |
| | Relevance * | 83% | 11,140,007 | +91.6% | 66.1% | 17/30 | 27/60 | $6.75 | *$0.40* |
| | Disclosure * | 85% | 14,153,824 | +143.4% | 73.2% | 20/30 | 40/60 | $8.56 | *$0.43* |
| | Graph | 70% | 13,191,178 | +126.9% | **95.3%** | **27/30** | 60/60 | $7.96 | $0.29 |
| | 🏆 All three | **37%** | 5,916,862 | +1.8% | 83.5% | 22/30 | 59/60 † | $3.60 | $0.16 |
| **Haiku 4.5** | Baseline * | 80% | 6,058,882 | — | 65.3% | 13/30 | 24/60 | $6.09 | *$0.47* |
| | Relevance | 75% | 13,214,041 | +118.1% | **93.7%** | **26/30** | 60/60 | $13.32 | $0.51 |
| | Disclosure | 65% | 12,156,014 | +100.6% | 88.2% | 22/30 | 60/60 | $12.26 | $0.56 |
| | Graph | 50% | 7,894,451 | +30.3% | 79.5% | 18/30 | 60/60 | $7.97 | $0.44 |
| | 🏆 All three | **22%** | 3,260,491 | **−46.2%** | 89.8% | 25/30 | 60/60 | $3.34 | $0.13 |
| **Qwen3 Next** | Baseline * | 117% | 8,231,351 | — | 59.1% | 11/30 | 14/60 | $1.16 | *$0.11* |
| | Relevance | 70% | 21,803,250 | +164.9% | **91.3%** | **26/30** | 60/60 | $3.10 | $0.12 |
| | Disclosure | 83% | 33,004,333 | +301.0% | 86.6% | 23/30 | 60/60 | $4.64 | $0.20 |
| | Graph * | **131%** | 25,119,509 | +205.2% | 55.9% | 10/30 | 16/60 | $3.53 | *$0.35* |
| | 🏆 All three | **25%** | 5,512,454 | **−33.0%** | 81.9% | 23/30 | 59/60 | $0.82 | $0.04 |

Peak input is the estimated prompt of the largest call, which is why it can exceed 100%: that is a call
the provider refused. † GLM 4.7's all-three lost its one turn to a `ReadTimeoutError`, a network failure,
not to the window — that arm overflowed zero times.

One replay per cell, so trap 3 applies in full: differences of two or three turns between two arms that
both completed are inside the noise floor. Measured separately on this class, the same arm replayed three
times moved by up to **two materially-correct turns out of eighteen**. `s/turn` is omitted from this
table rather than reported: the three models ran concurrently *and* several arms died early, so the
figure would compare the pace of a finished conversation against the pace of an abandoned one.

### 7. Only the full stack keeps every vendor inside its window

Peak input for all three combined is **37%, 22% and 25%** of the window against a bare agent's 85%, 80%
and 117%. That is the finding, and `Completed` is its consequence: 59, 60 and 59 turns against 22, 24 and
14. Every arm using **one** plugin fails to control the peak on at least one vendor — relevance and
disclosure both sit at 83-85% on GLM 4.7 and overflow there (54 and 30 calls lost), and the graph alone
reaches **131%** on Qwen3 Next and loses 88. No single practice is sufficient in this class on all three
vendors; the conjunction is, on all three.

### 8. Folding dialogue does not shrink a tool payload

The graph alone is the cleanest arm on GLM 4.7 — 27 of 30, zero errors, the highest accuracy anywhere in
this study — and on Qwen3 Next the same plugin peaks **above the window** and finishes 16 turns of 60.
That is not a contradiction and not noise: in this scenario the mass is the 40k-120k-character
documentation payloads, not the conversation, so folding history compresses the part that was not the
problem. What attacks the payload is the relevance filter, and what removes the fixed schema floor is
disclosure. The three are complementary by construction, which is why the conjunction holds where its
members do not.

**What this study cannot tell you is which single plugin is "best".** An arm that completed 27 of 60
turns and an arm that completed 60 cannot be compared on accuracy at all — the first one's score is a
score of the turns it survived. Relevance filtering looks second from last on GLM 4.7 (17 of 30) and is
the most accurate arm on both other vendors (26 of 30); the GLM figure is its 54 overflows, not its
recall. Read the single-strategy rows for their peak and their completion, and take accuracy from them
only where `Completed` reads 60/60.

### 9. The saving changes sign with the vendor, and that is not a regression

All three combined cuts tokens 46.2% on Haiku and 33.0% on Qwen, but spends **1.8% more** than the bare
agent on GLM 4.7. Read that against `Completed`: the GLM 4.7 baseline answered 22 turns and the full
stack answered 59. Nearly tripling the work delivered for 1.8% more tokens is the result, and the sign
alone gets it backwards. This is trap 1 at full strength — strong enough that the cheapest `$/correct`
cell in the study belongs to a truncated baseline, which is why every truncated ratio here is italicised
rather than competing.

Measured on the same class with two to three replays, the leave-one-out arms say it from the other side:
on GLM 4.7 at 20 turns all three combined is −54.8% tokens, but relevance+graph is **+6.1%** and
relevance+disclosure **+13.1%** — either *pair* spends more than using no plugin at all. The saving is a
conjunction, not a sum. See `CONFIGURATIONS` in
[`src/config.py`](validation/community-plugin-A-B-D/src/config.py).

### 10. The parameters to use on a window up to 250K

These are the values the study above was run with, and the reason to write them down is that **every
alternative tried against them lost**. On a window this size the knobs are not free choices — the
sweep below is what rules them out.

Three of them are the `TIGHT_WINDOW` set and apply *only* in this class; on a 1M window they cost 50%
for nothing, which is finding 11 below. The harness picks the set from the model, so running a
tight-window model needs no configuration at all — the values are spelled out here because a reader
wiring the plugins into their own agent has no harness to pick for them.

```python
from strands import Agent
from strands.agent.conversation_manager import NullConversationManager
from strands_context_graph import ContextGraph, EmbeddingSimilarityMatcher
from strands_progressive_tool_disclosure import ProgressiveToolDisclosure
from strands_relevance_filter import BedrockReranker, FileStore, RelevanceFilter

relevance = RelevanceFilter(
    store=FileStore("./artifacts"),   # file-backed: a reference read hours later must still resolve
    max_result_tokens=4_000,          # above this a payload is filtered; the payloads here are 10-30x it
    config={
        "reranker": BedrockReranker(model_id="cohere.rerank-v3-5:0"),
        "relevance_threshold": 0.02,  # calibrated to THIS reranker -- see the note below
        "chunk_tokens": 500,
        "preview_tokens": 2_000,      # REGIME-DEPENDENT: 2,000 here, 800 on a large window
    },
)

graph = ContextGraph(
    expand_threshold=0.55,       # keep the package default: with the filter installed, fold LESS
    collapse_floor=0.45,
    link_threshold=0.50,
    description_tokens=250,      # REGIME-DEPENDENT: 250 here, 100 on a large window
    body_budget=None,            # never bound in this class: peak sat at 22-37% of the window
    min_cards=3,
    max_retrieval_cycles=4,      # REGIME-DEPENDENT: 4 here, 8 on a large window
    reuse_ttl_cycles=5,
    include_artifact_tool=False,  # REQUIRED when the relevance filter is installed
    matcher=EmbeddingSimilarityMatcher("cohere.embed-multilingual-v3"),
)

disclosure = ProgressiveToolDisclosure(
    catalog_tokens=20,
    ttl_cycles=5,
    top_k=4,
    always_available=[*graph.retrieval_tool_names, "retrieve_context"],
)

agent = Agent(
    model=...,                                    # max_tokens=4096, NOT higher -- see below
    conversation_manager=NullConversationManager(),  # a precondition of the graph, not a preference
    plugins=[relevance, graph, disclosure],       # construct the graph before disclosure
)
```

Five of these are load-bearing in a way a reader would not guess, and each one was measured:

**Cap the model's output at 4,096, and resist raising it.** The provider subtracts the requested output
cap from the window *before* it admits the prompt, so the cap is a reservation taken out of history. At
8,192 on GLM 4.7 Flash, Bedrock refused calls whose prompt was 194,561 tokens — a prompt that fits at
4,096. Raising it to recover truncated answers converts completing calls into overflows, which is the
opposite of what this class needs.

**`include_artifact_tool=False` is required, not optional.** With the relevance filter installed there
are otherwise two retrieval tools over two stores that do not know each other, the model reaches for the
wrong one, and the reference does not resolve.

**`relevance_threshold=0.02` is a position in a distribution, not a number.** On `cohere.rerank-v3-5` a
strong match scores ~0.29 and an unrelated chunk ~0.03; the package default of `0.5` rejects every chunk
that exists. Change the reranker and this value is meaningless until you re-measure its distribution.

**`expand_threshold` stays at the default rather than rising.** When the relevance filter has already
replaced a payload with a preview, the graph should fold *less*, not more — the full-content rung is
already the cheap one. Raising it to 0.65 lost 1.5 materially-correct turns *and* increased tokens.

**`preview_tokens=2_000` is the ceiling on everything downstream.** The graph derives a Card's numeric
lines from the message, and by then the message carries the preview — so `expand_card` cannot return a
figure the preview dropped. Lowering the preview to save tokens silently lowers the Description budget's
value too; these two budgets are in series and the first one decides the second.

#### What was tried against these values and rejected

GLM 4.7, 20 turns of which 18 scored, two to three replays per row, all-three arm, each row changing
only what its name says. The reference row is the block above.

| Variant | Total tokens | Δ vs baseline | Correct | Verdict |
|---|---:|---:|:--:|---|
| Baseline, no plugin * | 3,733,922 | — | 12.5/18 | overflowed 6x per replay |
| **Reference (the values above)** | 1,906,838 | **−48.9%** | **16/18** | nothing beat it |
| `expand_threshold` 0.65 | 2,044,890 | −45.2% | 14.5/18 | lost turns **and** cost tokens |
| `ttl_cycles` 12 | 1,604,010 | −57.0% | 14.5/18 | −8pp tokens for −1.5 turns |
| `ttl_cycles` 8 | 1,397,004 | −62.6% | 13/18 | −14pp tokens for −3 turns |
| `catalog_tokens` 48 + `top_k` 6 | 1,563,844 | −58.1% | 13.5/18 | −9pp tokens for −2.5 turns |
| …plus `ttl_cycles` 8 | 1,764,815 | −52.7% | 15/18 | best challenger, still −1 turn |
| `preview_tokens` 1200 | 1,339,228 | −64.1% | 13.5/18 | the cheapest row, and −2.5 turns |
| `chunk_tokens` 250 | 1,592,800 | −57.3% | 14/18 | finer chunks did not buy selection |
| `relevance_threshold` 0.05 | 1,640,628 | −56.1% | 15/18 | −1 turn, inside the noise |

The shape of that table is the finding: **the trade is monotone and the defaults sit at its knee.** Every
row that spends less scores less, none dominates the reference, and the spread between replays of the
*same* row reached two materially-correct turns — so a variant that gains one turn has not gained
anything. A ninth attempt, raising the output cap to 8,192 and turning on a citable-density rerank prior,
doubled cost for +0.67 turns and is the clearest example of the failure mode this table exists to
prevent.

**What is genuinely open.** These rows are 20 turns, where the window is not yet binding on this model.
The 60-turn study above is where it binds, and there the same defaults hold — but the *single*-plugin
ordering changes between the two lengths, so a value tuned at 20 turns should be confirmed at the length
you actually run. Every knob is reachable from the environment (`VALIDATION_PREVIEW_TOKENS`,
`VALIDATION_GRAPH_EXPAND`, `VALIDATION_TTL_CYCLES`, …) and each run records which ones it read, so
repeating this sweep at another length is a list of variables rather than a list of commits.

### 11. Those parameters are for this class only — on a 1M window they cost 50% for nothing

The values in finding 10 were calibrated where evidence was starving. Replayed on **Opus 4.8**, a 1M
window, caching off, the same five arms with those values against the published run above:

| Configuration | Published | With these parameters | Δ tokens | Correct, then → now |
|---|---:|---:|---:|:--:|
| Baseline (no plugin) | 14,377,382 | 16,178,910 | +12.5% | 28/30 → 26/30 |
| Disclosure | 9,466,084 | 11,690,932 | +23.5% | 27/30 → 28/30 |
| Relevance | 12,766,237 | 15,133,294 | +18.5% | 29/30 → 28/30 |
| Graph | 10,385,714 | 10,043,554 | −3.3% | 27/30 → 24/30 |
| **All three** | **2,569,888** | **3,858,779** | **+50.2%** | 28/30 → 28/30 |

**Read the baseline row first, because it is the yardstick.** The baseline runs no plugin, so no change
in finding 10 can reach it — and it still moved +12.5% in tokens and lost two materially-correct turns.
That is trap 2 and trap 3 measured directly: the agent picked a different tool path, and one extra call
early in a 60-turn conversation rides along in every later one. **Nothing smaller than that is
attributable.** The graph's −3.3% is not a change; the disclosure and relevance rows are barely outside
it; the all-three row, at four times the yardstick, is.

The mechanism is `preview_tokens` and `description_tokens`, and the graph-only arm isolates it. Both runs
of that arm carry the same 5.4x jump in graph links that the live vector index produced (81 → 482), and
its tokens went *down* — so the link count is not the amplifier. What only the all-three arm has is the
filter's preview feeding the Card's Description, and its peak call went **49,943 → 81,065 (+62%)** while
the resolution ladder barely moved (full 20/9/4 → 19/8/7). The graph is not folding differently; every
rung is simply carrying more. Cancellations also went 14 → 24, which is the tightened guessed-call guard
charging a round trip for each invented-argument call it now refuses.

And it bought nothing: 28 of 30 both times. On a window this size there was no starvation to fix.

**So these two budgets are regime-dependent, exactly as the graph's own thresholds already are — and
the harness now treats them that way.** The file carries `GRAPH_ALONE` and `GRAPH_WITH_RELEVANCE`
because one set of graph knobs is wrong; `LARGE_WINDOW` and `TIGHT_WINDOW` in
[`src/config.py`](validation/community-plugin-A-B-D/src/config.py) are the same construction for the
three budgets that differ across window classes:

| Budget | `LARGE_WINDOW` | `TIGHT_WINDOW` |
|---|---:|---:|
| `preview_tokens` | 800 | 2,000 |
| graph `description_tokens` (with relevance) | 100 | 250 |
| graph `max_retrieval_cycles` | 8 | 4 |

They are held as one set rather than three knobs because that is how they were measured: the comparison
above reverted all three at once, so the aggregate is attributable and the individual contributions are
not. The regime is derived from the agent model's declared context window, recorded in every run's
`meta.window_regime`, and overridable with `VALIDATION_WINDOW_REGIME=tight|large` — which is how the
Opus row above was produced.

**The selection ceiling is 300K, not the 250K the class is named after, and that is a measurement
rather than a rounding.** Qwen3 Next's window is 256,000 — above 250K — and it sat firmly in the tight
regime: 92 overflows on the bare agent, and the graph-alone arm peaking at 131% of the window. The real
determinant is not the window but the window against the payload mass in front of it, for which the
window is only a proxy; the ceiling carries headroom because the errors are asymmetric. Tight budgets on
a large window cost tokens (+50.2%, buying nothing). Large budgets on a tight window cost answers.

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

**The two studies are not the same code.** The nine-run battery predates four fixes the tight-window
study was run with, the load-bearing one being that the disclosure plugin used to cancel the model's
first `find_tools` call of every session — `find_tools` is projected unconditionally but never recorded
as exposed, so the guard read a call to it as a call made off a catalog entry and refused the one call
that opens the discovery path. Measured on GLM 4.7 Flash, repairing it took searches from 1 to 7 per run
and materially-correct turns from 6.7 to 9.3 of 18. Do not read a disclosure or all-three cell from the
first table against one from the second; compare within a study.

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

The tight-window study is the same script against one model at a time, caching left off:

```bash
cd validation/community-plugin-A-B-D
VALIDATION_AGENT_MODEL_ID=zai.glm-4.7 ./run.sh --total-turns 60 --tag tw-glm47
VALIDATION_AGENT_MODEL_ID=us.anthropic.claude-haiku-4-5-20251001-v1:0 ./run.sh --total-turns 60 --tag tw-haiku45
VALIDATION_AGENT_MODEL_ID=qwen.qwen3-next-80b-a3b ./run.sh --total-turns 60 --tag tw-qwen3
```

Re-running the parameter sweep needs no code edit: every knob in the block above reads an override off
the environment, defaults unchanged, and each run records under `meta.sweep_overrides` which ones it
read — so a result can never be read without its configuration.

```bash
VALIDATION_PREVIEW_TOKENS=1200 ./run.sh --configs all --total-turns 20 --repeats 3 --tag sweep-preview
VALIDATION_GRAPH_EXPAND=0.65 ./run.sh --configs all --total-turns 20 --repeats 3 --tag sweep-fold
```

The leave-one-out arms behind finding 9 are `no-disclosure`, `no-relevance` and `no-graph`. They are
accepted by `--configs` but deliberately not in the default set, so adding them cannot change what a
published table means:

```bash
./run.sh --configs no-disclosure no-relevance no-graph --total-turns 20 --repeats 3 --tag loo
```

See [`validation/community-plugin-A-B-D/README.md`](validation/community-plugin-A-B-D/README.md) for
what the harness measures and how scoring works.
