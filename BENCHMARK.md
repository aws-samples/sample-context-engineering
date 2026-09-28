# Benchmark

Eleven models, one scripted conversation, five configurations each, organised **by model**: what the model is,
what the plugins and the harness were set to, what came out, what it supports.

Parts 1 and 2 define the script and the parameters every table refers to. Part 5 is what survives across
models; nothing in one model's section generalises on its own.

[1. What a run is](#1-what-a-run-is) · [2. Plugins and parameters](#2-plugins-and-parameters) ·
[3. Prices](#3-prices) · [4. The models](#4-the-models) ·
[5. What holds across models](#5-what-holds-across-models) · [6. Reproducing](#6-reproducing) ·
[7. Not settled](#7-not-settled)

---

## 1. What a run is

60 turns, same prompts in the same order for every configuration, so two configurations differ only in what
the agent was allowed to send. A personal-finance and cloud-ops assistant over **mocked tools**; several
return 40,000–120,000 characters in one result, which is where the mass is — the payloads, not the
conversation. 30 turns are scored (18 hand-written, 12 generated), the other 30 are grounded filler that
lengthens the history.

Scoring is deterministic, never an LLM judge: the tools are mocked, so ground truth comes from their
implementations (`src/ground_truth.py`). Figures must match **as the tools formatted them** — restating
`42,1%` as `42.1%` fails on purpose.

| Column | Meaning |
|---|---|
| **Accuracy** | fraction of expectation *weight* met; partial credit |
| **Correct** | turns with no *critical* failure — the fact the question asked for |
| **Answered** | turns that produced an answer at all |
| **Scored lost** | scored turns never answered |
| **Refused** | calls the provider rejected with `ContextWindowOverflow` |
| **Peak/call** | largest call that went out, and its share of the window |

The arms are: baseline (no plugin — every payload enters the history whole and stays), each practice alone,
all three combined. Columns follow the README — Total tokens · Δ tokens · Accuracy · Correct · Turn · Cost ·
Δ cost. Cache columns appear only for runs that served cache reads (there `inputTokens` stops measuring
input), completion columns only where an arm had calls refused, `Peak/call` for windows ≤ 300K. **Their
absence is information.**

**Three traps.** ① A truncated arm looks cheap — it stopped answering, so it stopped spending: understated
tokens, accuracy bounded by truncation, seconds-per-turn measuring an abandoned conversation. Marked ✝, read
`Answered` first. ② Where the baseline truncated, Δ is measured against the heaviest arm (`← ref`): a
baseline that answered 9 of 60 turns makes every completing arm look like a regression. ③ One replay per
cell, and the agent picks its own tool path — four byte-identical arms moved by up to six correct turns
([the noise floor](#the-noise-floor)), so **nothing under ±6 turns or ±20% of tokens is evidence.**

---

## 2. Plugins and parameters

| | Practice | Package | What it does |
|---|---|---|---|
| **A** | Relevance filtering | [`strands-relevance-filter`](community-plugins/strands-relevance-filter/) | Scores a tool result's chunks against the question, replacing the payload with a preview **before it enters the history** |
| **B** | Progressive tool disclosure | [`strands-progressive-tool-disclosure`](community-plugins/strands-progressive-tool-disclosure/) | Lean tool catalog, full spec on demand, then forgotten — attacks the ~63k schema floor |
| **D** | Context graph | [`strands-context-graph`](community-plugins/strands-context-graph/) | History as a graph with remove/recover, folding each turn to full content, Description or Title |

They compose because they act at different moments: **the filter on a tool result before it enters the
history, the graph on a history that already exists.**

| Plugin | Parameter | Value | Env override |
|---|---|---:|---|
| A | `max_result_tokens` (trigger) | 4,000 | `VALIDATION_MAX_RESULT_TOKENS` |
| A | `chunk_tokens` | 500 | `VALIDATION_CHUNK_TOKENS` |
| A | `preview_tokens` | *regime* — 800 / 2,000 | `VALIDATION_PREVIEW_TOKENS` |
| A | `relevance_threshold` | 0.02 | `VALIDATION_RELEVANCE_THRESHOLD` |
| A | `include_retrieval_tool` | **False** | `VALIDATION_RELEVANCE_RETRIEVAL_TOOL` |
| B | `catalog_tokens` / `ttl_cycles` / `top_k` | 20 / 5 / 4 | `VALIDATION_CATALOG_TOKENS`, … |
| B | `catalog_in_system_prompt` | False | `VALIDATION_CATALOG_IN_SYSTEM_PROMPT` |
| D | `expand_threshold` / `collapse_floor` / `link_threshold` | 0.62 / 0.45 / 0.50 | `VALIDATION_GRAPH_EXPAND`, … |
| D | `description_tokens` | *regime* — 100 / 250 | `VALIDATION_GRAPH_DESCRIPTION_TOKENS` |
| D | `body_budget` | **40,000** | `VALIDATION_GRAPH_BODY_BUDGET` |
| D | `max_retrieval_cycles` | *regime* — 8 / 4 | `VALIDATION_GRAPH_MAX_RETRIEVAL_CYCLES` |
| D | `reuse_ttl_cycles` / `tags_per_card` / `neighbors_per_candidate` / `min_cards` | 5 / 5 / 3 / 3 | `VALIDATION_GRAPH_REUSE_TTL`, … |
| D | `include_artifact_tool` | True, every arm | — |

Two values that look wrong and are not. `relevance_threshold=0.02` is a position in a distribution:
`cohere.rerank-v3-5` scores strong matches around 0.29, so the package default of 0.5 rejects everything. And
`body_budget` is the *only* ceiling on full-content mass — `expand_threshold` is a per-Card classifier that
cannot see the total, so `None` means a long enough conversation sends every Card above it whole.

### The window regime

Picked from the declared window against `TIGHT_WINDOW_CEILING = 300_000`; recorded as `meta.window_regime`.

| Budget | `LARGE_WINDOW` | `TIGHT_WINDOW` |
|---|---:|---:|
| `preview_tokens` | 800 | 2,000 |
| graph `description_tokens` | 100 | 250 |
| graph `max_retrieval_cycles` | 8 | 4 |

One set, not three knobs: the comparison that established them reverted all three at once, so the aggregate
is attributable and the individual contributions are not. The ceiling is 300K rather than 250K because Qwen3
Next's 256,000-token window sits firmly in the tight regime — the determinant is the window against the
payload mass in front of it, and the errors are asymmetric (tight budgets on a large window cost tokens,
large budgets on a tight window cost answers). `preview_tokens` bounds everything downstream — `payload →
preview → message → Card numeric lines → Description budget` — so raising the Description budget while the
preview starves buys nothing.

---

## 3. Prices

`measured units × published list rate` — list cost, not a bill, no discounts modelled. Cost includes what each
strategy spends on its own account: the graph's embeddings (`cohere.embed-multilingual-v3`) and the filter's
rerank calls (`cohere.rerank-v3-5:0`). Rates live in `MODEL_PRICING` in
[`src/config.py`](validation/community-plugin-A-B-D/src/config.py), so a correction re-renders from the
recorded JSON instead of needing another run.

| Claim | Status | Source |
|---|---|---|
| Per-model input, output, cache rates | published | [Bedrock pricing](https://aws.amazon.com/bedrock/pricing/) and each model card |
| Cache writes billed above input rate, implicit **and** explicit | documented | [Billing for cached tokens](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html) |
| Cache write 1.25× input, read −90%, GPT-5.6 | documented | [Prompt caching for OpenAI models](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html) |
| Astra write $13.75 against $11.00 input | published | [Astra model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-6-astra.html) |
| Claude cache read 0.10× / write 1.25–2.00× | published, **except Haiku 4.5** — inferred, not on the pricing table | `MODEL_PRICING` |
| Tokens, calls, accuracy, timings | measured from the API's own `usage` | run JSON under `results/` |

---

## 4. The models

| Model | Window | Caching | Runs shown | Bare agent finished? | Cheapest completing arm |
|---|---:|---|---|:--:|---|
| [Claude Opus 4.8](#41-claude-opus-48) | 1,000,000 | explicit | cache off, cache on | yes | all three |
| [Claude Opus 5](#42-claude-opus-5) | 1,000,000 | explicit | cache off, cache on | yes | all three |
| [Claude Fable 5](#43-claude-fable-5) | 1,000,000 | explicit | cache off, cache on | yes | all three |
| [GPT-6 Astra](#44-gpt-6-astra) | 1,050,000 | implicit | implicit cache | yes | relevance |
| [GPT-5.6 Sol](#45-gpt-56-sol) | 1,000,000 | implicit | implicit cache | yes | relevance |
| [Claude Haiku 4.5](#46-claude-haiku-45) | 200,000 | explicit | cache off | **no** | all three |
| [GLM 5](#47-glm-5) | 200,000 | none | one | **no** | all three |
| [GLM 4.7](#48-glm-47) | 202,752 | none | one | **no** | all three |
| [GLM 4.7 Flash](#49-glm-47-flash) | 202,752 | none | one | **no** | all three |
| [Qwen3 Next 80B](#410-qwen3-next-80b) | 256,000 | none | one | **no** | relevance |
| [Nemotron Nano 9B](#411-nemotron-nano-9b) | 128,000 | none | one | **no** | all three |

Every run: 60 turns / 30 scored · 1 replay · `max_output 4,096` · rerank `cohere.rerank-v3-5:0` · embed
`cohere.embed-multilingual-v3` · region `us-east-1`. Deviations are named in the section.

### 4.1 Claude Opus 4.8

`us.anthropic.claude-opus-4-8` · Anthropic · window **1,000,000** · explicit caching (read $0.50, write
$6.25/5m) · $5.00 in / $25.00 out per Mtok · **large** regime (preview 800, graph desc 100, cycles 8).

**Caching off.**

| Configuration | Total tokens | Δ tokens | Accuracy | Correct | Turn | Cost | Δ cost |
|---|---:|---:|---:|:--:|---:|---:|---:|
| Baseline (no plugin) | 17,897,954 | — | 97.6% | 29/30 | 11.0s | $90.06 | — |
| Progressive Tool Disclosure only | 6,738,122 | −62.4% | 92.9% | 26/30 | **8.1s** | $34.24 | −62.0% |
| Relevance Filtering only | 12,432,470 | −30.5% | 95.3% | 27/30 | 9.8s | $62.77 | −30.3% |
| Context Graph only | 10,511,391 | −41.3% | 94.5% | 27/30 | 10.0s | $53.13 | −41.0% |
| **All three combined** | **2,877,396** | **−83.9%** | **99.2%** | **30/30** | 10.0s | **$15.09** | −83.2% |

**Caching on.**

| Configuration | Uncached | Cache read | Cache write | read:write | Billed tokens | Δ tokens | Accuracy | Correct | Turn | Cost | Δ cost |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|---:|---:|---:|
| Baseline (no plugin) | 184 | 19,276,592 | 279,517 | **69.0** | 19,556,293 | — | **98.4%** | **29/30** | 8.1s | $12.07 | — |
| Progressive Tool Disclosure only | 200 | 5,602,888 | 2,935,685 | 1.9 | 8,538,773 | −56.3% | 92.9% | 26/30 | 8.6s | $22.04 | +82.6% |
| Relevance Filtering only | 186 | 11,742,023 | 183,017 | 64.2 | 11,925,226 | −39.0% | **98.4%** | **29/30** | **7.8s** | **$7.72** | −36.0% |
| Context Graph only | 241,949 | 8,752,948 | 1,586,320 | 5.5 | 10,581,217 | −45.9% | 95.3% | 26/30 | 9.9s | $16.34 | +35.4% |
| **All three combined** | 322,673 | 823,873 | 1,523,424 | **0.5** | 2,669,970 | −86.3% | 97.6% | **29/30** | 9.8s | $12.50 | +3.5% |

### 4.2 Claude Opus 5

`us.anthropic.claude-opus-5` · Anthropic · window **1,000,000** · explicit caching (read $0.50, write
$6.25/5m) · $5.00 in / $25.00 out per Mtok · **large** regime.

**Caching off.**

| Configuration | Total tokens | Δ tokens | Accuracy | Correct | Turn | Cost | Δ cost |
|---|---:|---:|---:|:--:|---:|---:|---:|
| Baseline (no plugin) | 18,817,365 | — | **97.6%** | **28/30** | 20.7s | $95.34 | — |
| Progressive Tool Disclosure only | 19,040,438 | +1.2% | 92.9% | 26/30 | 17.8s | $96.17 | +0.9% |
| Relevance Filtering only | 14,819,944 | −21.2% | 96.9% | **28/30** | 17.7s | $75.24 | −21.1% |
| Context Graph only | 11,345,387 | −39.7% | 95.3% | 26/30 | 19.8s | $58.12 | −39.0% |
| **All three combined** | **4,727,088** | **−74.9%** | 96.9% | 27/30 | **17.2s** | **$24.94** | −73.8% |

**Caching on.**

| Configuration | Uncached | Cache read | Cache write | read:write | Billed tokens | Δ tokens | Accuracy | Correct | Turn | Cost | Δ cost |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|---:|---:|---:|
| Baseline (no plugin) | 192 | 20,305,665 | 281,010 | **72.3** | 20,586,867 | — | **97.6%** | 28/30 | 20.5s | $13.91 | — |
| Progressive Tool Disclosure only | 218 | 10,482,326 | 7,023,275 | 1.5 | 17,505,819 | −15.0% | 91.3% | 25/30 | **13.9s** | $50.45 | +262.7% |
| Relevance Filtering only | 216 | 15,734,826 | 255,567 | 61.6 | 15,990,609 | −22.3% | 96.1% | 28/30 | 16.4s | **$11.01** | −20.8% |
| Context Graph only | 276,386 | 9,489,827 | 2,420,670 | 3.9 | 12,186,883 | −40.8% | 93.7% | 25/30 | 19.4s | $23.13 | +66.3% |
| **All three combined** | 353,942 | 1,118,910 | 2,045,680 | **0.5** | 3,518,532 | −82.9% | **97.6%** | **29/30** | 16.9s | $16.64 | +19.6% |

### 4.3 Claude Fable 5

`us.anthropic.claude-fable-5` · Anthropic · window **1,000,000** · explicit caching (read $1.00, write
$12.50/5m) · **$10.00 in / $50.00 out** per Mtok, the most expensive model here · **large** regime.

**Caching off.**

| Configuration | Total tokens | Δ tokens | Accuracy | Correct | Turn | Cost | Δ cost |
|---|---:|---:|---:|:--:|---:|---:|---:|
| Baseline (no plugin) | 19,172,194 | — | 98.4% | 29/30 | 17.7s | $193.09 | — |
| Progressive Tool Disclosure only | 22,512,071 | +17.4% | 98.4% | 29/30 | 21.9s | $226.88 | +17.5% |
| Relevance Filtering only | 14,444,452 | −24.7% | **99.2%** | **30/30** | **16.1s** | $145.76 | −24.5% |
| Context Graph only | 12,591,615 | −34.3% | 94.5% | 27/30 | 20.1s | $127.83 | −33.8% |
| **All three combined** | **4,461,881** | **−76.7%** | 98.4% | 29/30 | 22.7s | **$47.01** | −75.7% |

**Caching on.**

| Configuration | Uncached | Cache read | Cache write | read:write | Billed tokens | Δ tokens | Accuracy | Correct | Turn | Cost | Δ cost |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|---:|---:|---:|
| Baseline (no plugin) | 198 | 19,700,362 | 271,431 | **72.6** | 19,971,991 | — | 97.6% | 29/30 | **14.2s** | $24.79 | — |
| Progressive Tool Disclosure only | 252 | 10,078,601 | 10,177,268 | 1.0 | 20,256,121 | +1.4% | 97.6% | 29/30 | 18.2s | $139.47 | +462.5% |
| Relevance Filtering only | 204 | 14,013,780 | 223,409 | 62.7 | 14,237,393 | −28.7% | 99.2% | **30/30** | 14.4s | **$18.67** | −24.7% |
| Context Graph only | 297,203 | 10,049,572 | 1,836,828 | 5.5 | 12,183,603 | −39.0% | **100.0%** | **30/30** | 17.7s | $38.23 | +54.2% |
| **All three combined** | 409,642 | 1,790,760 | 2,958,854 | **0.6** | 5,159,256 | −74.2% | 97.6% | 29/30 | 24.8s | $46.40 | +87.1% |

### 4.4 GPT-6 Astra

`us.openai.gpt-6-astra` · OpenAI · window **1,050,000** · **implicit** caching applied by the service, not
requested (read $1.10, write $13.75 = 1.25× input) · $11.00 in / $55.00 out per Mtok · **large** regime ·
served from `us-west-2` while called from `us-east-1`.

| Configuration | Uncached | Cache read | Cache write | read:write | Billed tokens | Δ tokens | Accuracy | Correct | Turn | Cost | Δ cost |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|---:|---:|---:|
| Baseline (no plugin) | 202 | 10,766,018 | 164,828 | **65.3** | 10,931,048 | — | 96.1% | 28/30 | 6.9s | $15.25 | — |
| Progressive Tool Disclosure only | 328 | 7,644,124 | 4,619,239 | 1.7 | 12,263,691 | +12.2% | **97.6%** | **29/30** | 9.2s | $73.50 | +381.9% |
| Relevance Filtering only | 212 | 9,831,881 | 157,816 | 62.3 | 9,989,909 | −8.6% | **97.6%** | **29/30** | **6.8s** | **$14.13** | −7.4% |
| Context Graph only | 252 | 1,477,264 | 7,271,836 | 0.2 | 8,749,352 | −20.0% | **97.6%** | **29/30** | 8.5s | $102.85 | +574.3% |
| **All three combined** | 364 | 18,903 | 3,593,236 | **0.0** | 3,612,503 | −67.0% | **97.6%** | **29/30** | 8.9s | $51.16 | +235.4% |

**Observations.** Implicit caching cannot be turned off.

### 4.5 GPT-5.6 Sol

`us.openai.gpt-5.6-sol` · OpenAI · window **1,000,000** · **implicit** caching (read $0.44, write $5.50) ·
$4.40 in / $22.00 out per Mtok · **large** regime · served from `us-east-2` while called from `us-east-1`.

| Configuration | Uncached | Cache read | Cache write | read:write | Billed tokens | Δ tokens | Accuracy | Correct | Turn | Cost | Δ cost |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|---:|---:|---:|
| Baseline (no plugin) | 210 | 12,208,657 | 179,251 | **68.1** | 12,388,118 | — | **97.6%** | **29/30** | 5.8s | $6.91 | — |
| Progressive Tool Disclosure only | 238 | 4,796,001 | 4,054,147 | 1.2 | 8,850,386 | −28.6% | 86.6% | 25/30 | 6.3s | $24.85 | +259.6% |
| Relevance Filtering only | 220 | 9,675,159 | 151,027 | 64.1 | 9,826,406 | −20.7% | **97.6%** | **29/30** | **5.3s** | **$5.58** | −19.2% |
| Context Graph only | 270 | 952,765 | 9,579,693 | 0.1 | 10,532,728 | −15.0% | 96.9% | 28/30 | 8.1s | $53.96 | +680.9% |
| **All three combined** | 286 | 22,202 | 2,205,615 | **0.0** | 2,228,103 | −82.0% | 77.2% | 21/30 | 8.9s | $12.83 | +85.7% |

**Observations.** Implicit caching cannot be turned off, so there is no uncached control here either. This is
the one model where doing nothing wins on every axis at once — cheapest, most accurate, fastest.
### 4.6 Claude Haiku 4.5

`us.anthropic.claude-haiku-4-5-20251001-v1:0` · Anthropic · window **200,000** · explicit caching (read
$0.10, write $1.25/5m — **inferred**) · $1.00 in / $5.00 out per Mtok · **tight** regime (preview 2,000,
graph desc 250, cycles 4).

| Configuration | Answered | Scored lost | Refused | Total tokens | Δ vs heaviest | Accuracy | Correct | Peak/call | Turn | Cost |
|---|:--:|:--:|---:|---:|---:|---:|:--:|---:|---:|---:|
| Baseline (no plugin) ✝ | **26/60** | **12/30** | **34** | 6,482,160 | −29.7% | 52.8% | 14/30 | 198,745 (**99%**) | 4.0s ✝ | $6.52 |
| Progressive Tool Disclosure only ← ref | 60/60 | 0/30 | 0 | 9,225,960 | — | 85.0% | 21/30 | 136,935 (68%) | 6.2s | $9.31 |
| Relevance Filtering only | 60/60 | 0/30 | 0 | 7,899,828 | −14.4% | 83.5% | 20/30 | 111,389 (56%) | 5.4s | $7.97 |
| Context Graph only | 60/60 | 0/30 | 0 | 8,544,161 | −7.4% | **92.1%** | **25/30** | 115,009 (58%) | 7.8s | $8.64 |
| **All three combined** | 60/60 | 0/30 | 0 | **2,199,431** | **−76.2%** | 89.8% | **25/30** | **36,352 (18%)** | 5.2s | **$2.29** |

### 4.7 GLM 5

`zai.glm-5` · Zhipu · window **200,000** · **no caching published**, so there is nothing to weigh the plugins
against · $1.00 in / $3.20 out per Mtok · **tight** regime (preview 2,000, graph desc 250, cycles 4).

| Configuration | Answered | Scored lost | Refused | Total tokens | Δ vs heaviest | Accuracy | Correct | Peak/call | Turn | Cost |
|---|:--:|:--:|---:|---:|---:|---:|:--:|---:|---:|---:|
| Baseline (no plugin) ✝ | **26/60** | **12/30** | **34** | 8,341,004 | −45.9% | 51.2% | 15/30 | 193,846 (**97%**) | 15.5s ✝ | $8.36 |
| Progressive Tool Disclosure only ← ref | 60/60 | 0/30 | 0 | 15,424,133 | — | **95.3%** | **27/30** | 180,967 (90%) | 35.8s | $15.46 |
| Relevance Filtering only | 60/60 | 0/30 | 0 | 14,092,871 | −8.6% | 89.0% | 26/30 | 165,567 (83%) | 34.4s | $14.16 |
| Context Graph only | 60/60 | 0/30 | 0 | 10,038,122 | −34.9% | 87.4% | 24/30 | 117,486 (59%) | 23.9s | $10.07 |
| **All three combined** | 60/60 | 0/30 | 0 | **2,370,745** | **−84.6%** | 89.8% | 25/30 | **33,695 (17%)** | 15.6s | **$2.42** |

**Observations.** The model that turns the question from cost into completion: the bare agent lost 84 calls
and answered 18 of 60, so its 65.3% is mostly absence, with 14 of its 30 scored turns never attempted. Only
the graph and the full stack finished intact, and they are the only arms peaking below 60% of the window —
every arm at 97% or above lost calls. Cost stops being the argument: the full stack costs $2.96 and answers
everything, the bare agent $6.17 for a quarter of the script. The graph alone is the most accurate arm here
(28/30) and the second cheapest — on *this* model the single plugin that keeps you inside the window is the
graph, which is not true of the next two, and is why more than one vendor is measured.

### 4.8 GLM 4.7

`zai.glm-4.7` · Zhipu · window **202,752** · no caching published · $0.60 in / $2.20 out per Mtok ·
**tight** regime.

| Configuration | Answered | Scored lost | Refused | Total tokens | Δ vs heaviest | Accuracy | Correct | Peak/call | Turn | Cost | $/correct |
|---|:--:|:--:|---:|---:|---:|---:|:--:|---:|---:|---:|---:|
| Baseline (no plugin) ✝ | **20/60** | **14/30** | **40** | 5,755,771 | −65.2% | 48.0% | 13/30 | 193,179 (**95%**) | 16.5s ✝ | $3.47 | *$0.27* |
| Progressive Tool Disclosure only ✝ ← ref | **54/60** | **3/30** | **6** | 16,547,804 | — | 78.0% | 25/30 | 195,933 (**97%**) | 66.8s ✝ | $10.00 | *$0.40* |
| Relevance Filtering only | 60/60 | 0/30 | 0 | 10,899,610 | −34.1% | 89.8% | 26/30 | 127,025 (63%) | 16.2s | $6.57 | $0.25 |
| Context Graph only | 60/60 | 0/30 | 0 | 7,416,591 | −55.2% | 90.5% | 27/30 | 107,480 (53%) | 29.9s | $4.49 | $0.17 |
| **All three combined** | 60/60 | 0/30 | 0 | **3,670,567** | **−77.8%** | **96.1%** | **28/30** | **56,283 (28%)** | 27.8s | **$2.26** | **$0.08** |

† A `ReadTimeoutError`, not the window; that arm refused zero calls. Fifteen other non-overflow errors
(3 baseline, 7 relevance, 5 disclosure, 2 all-three) were output-cap truncations.

**Observations.** Three of five arms truncated, and not just the baseline: relevance lost 54 calls and
disclosure 30, which is what makes the accuracy column dangerous here — relevance looks second-from-last
(17/30) and is the most accurate arm on both other vendors of its study, so that figure is its refusals, not
its recall. The full stack spends 1.8% *more* than the bare agent, and that is the result: the baseline
answered 22 turns and the full stack 59, nearly tripling the work for 1.8% more tokens. It is also why every
truncated `$/correct` is italicised rather than competing. The graph alone is the cleanest arm here (27 of 30,
zero refusals) while the same plugin on Qwen3 Next peaks above the window and answers 16 of 60 — folding
history compresses the part that was not the problem when the mass is in the payloads. At 20 turns with 2–3
replays the leave-one-out arms say it from the other side: all three combined −54.8% tokens, but
relevance+graph **+6.1%** and relevance+disclosure **+13.1%** — either *pair* spends more than no plugin at
all, so **the saving is a conjunction, not a sum.**

### 4.9 GLM 4.7 Flash

`zai.glm-4.7-flash` · Zhipu · window **202,752** · no caching published · **$0.07 in / $0.40 out** per Mtok,
the cheapest model here · **tight** regime.

| Configuration | Answered | Scored lost | Refused | Total tokens | Δ vs heaviest | Accuracy | Correct | Peak/call | Turn | Cost |
|---|:--:|:--:|---:|---:|---:|---:|:--:|---:|---:|---:|
| Baseline (no plugin) ✝ | **26/60** | **12/30** | **34** | 7,295,295 | −38.5% | 49.6% | 13/30 | 188,759 (**93%**) | 7.6s ✝ | $0.51 |
| Progressive Tool Disclosure only | 60/60 | 0/30 | 0 | 10,804,814 | −8.9% | **71.7%** | **18/30** | 189,266 (93%) | 6.8s | $0.76 |
| Relevance Filtering only | 60/60 | 0/30 | 0 | 11,019,518 | −7.1% | 70.1% | 17/30 | 116,327 (57%) | 7.6s | $0.78 |
| Context Graph only ← ref | 60/60 | 0/30 | 0 | 11,858,161 | — | 69.3% | 13/30 | 154,134 (76%) | 16.2s | $0.84 |
| **All three combined** | 60/60 | 0/30 | 0 | **4,187,110** | **−64.7%** | 67.7% | 14/30 | **47,366 (23%)** | 15.2s | **$0.31** |

Ten non-overflow errors (2 baseline, 4 disclosure, 4 graph) were output-cap truncations.

**Observations.** The only model whose reference arm is itself truncated: disclosure spent 27.4M tokens over
237 calls and answered 36 of 60, an erratic tool path where every retry carries the whole history, so treat
the Δ magnitudes as a floor. Relevance filtering is the arm to use here — the only single plugin that
completed, and the highest scorer among completing arms. The full stack completes, is by far the cheapest
($0.30), and scores 10/30, its worst anywhere: a genuine weakness rather than truncation, since it answered
all sixty. Two serial cuts (`preview_tokens` then `description_tokens`) plus a reranker query lacking the
sought literal is an evidence-*selection* loss, and this model is where it bites hardest. Every arm costs
under $2 here, so the interesting column is `Answered`.

### 4.10 Qwen3 Next 80B

`qwen.qwen3-next-80b-a3b` · Alibaba · window **256,000** · no caching published · $0.14 in / $1.20 out per
Mtok · regime **tight** — above 250K and still tight, see [the window regime](#the-window-regime).

| Configuration | Answered | Scored lost | Refused | Total tokens | Δ vs heaviest | Accuracy | Correct | Peak/call | Turn | Cost |
|---|:--:|:--:|---:|---:|---:|---:|:--:|---:|---:|---:|
| Baseline (no plugin) ✝ | **14/60** | **17/30** | **46** | 6,730,068 | −84.9% | 37.8% | 9/30 | 251,450 (**98%**) | 4.7s ✝ | $0.95 |
| Progressive Tool Disclosure only ← ref | 60/60 | 0/30 | 0 | 44,594,408 | — | **86.6%** | 23/30 | 178,170 (70%) | 24.1s | $6.27 |
| Relevance Filtering only | 60/60 | 0/30 | 0 | **13,663,677** | **−69.4%** | 81.1% | 23/30 | 181,676 (71%) | 9.9s | **$1.95** |
| Context Graph only | 60/60 | 0/30 | 0 | 22,648,027 | −49.2% | 80.3% | 22/30 | 144,801 (57%) | 17.4s | $3.21 |
| **All three combined** | 60/60 | 0/30 | 0 | 21,483,823 | −51.8% | **86.6%** | **24/30** | **72,336 (28%)** | 28.8s | $3.09 |

**Observations.** The heaviest payload mass of the battery, and the reason the tight ceiling is 300K: a
256,000-token window and the bare agent still peaks at it and loses 94 calls. Both single plugins that attack
the schema or the payload fail here — disclosure answered 13 of 60, relevance 55 with 10 refusals — while the
graph completes at 53% of the window. The disclosure row is the trap in one line: 5.0M tokens and $0.71 look
efficient until `Answered` reads 13 of 60, the second-cheapest cell of the run and the second-worst outcome.

### 4.11 Nemotron Nano 9B

`nvidia.nemotron-nano-9b-v2` · NVIDIA · window **128,000**, the smallest here · no caching published ·
$0.06 in / $0.23 out per Mtok · **tight** regime.

| Configuration | Answered | Scored lost | Refused | Total tokens | Δ vs heaviest | Accuracy | Correct | Peak/call | Turn | Cost |
|---|:--:|:--:|---:|---:|---:|---:|:--:|---:|---:|---:|
| Baseline (no plugin) ✝ | **12/60** | **19/30** | **48** | 1,823,314 | −75.9% | 27.6% | 5/30 | 125,581 (**98%**) | 4.7s ✝ | $0.11 |
| Progressive Tool Disclosure only ✝ ← ref | **44/60** | **6/30** | **16** | 7,559,453 | — | 60.6% | 16/30 | 121,054 (**95%**) | 19.2s ✝ | $0.47 |
| Relevance Filtering only ✝ | **35/60** | **9/30** | **25** | 5,531,562 | −26.8% | 53.5% | 14/30 | 124,652 (**97%**) | 13.1s ✝ | $0.35 |
| Context Graph only ✝ | **15/60** | **16/30** | **45** | 4,144,929 | −45.2% | 35.4% | 7/30 | 123,516 (**96%**) | 10.8s ✝ | $0.25 |
| **All three combined** | 60/60 | 0/30 | 0 | **3,464,956** | **−54.2%** | **60.6%** | **11/30** | **54,537 (43%)** | 24.9s | **$0.25** |

**Observations.** Four of five arms could not finish, and the only column that orders cleanly is `Refused` —
102, 92, 64, 34, **0**: each practice removes some of the pressure, only the conjunction removes all of it.
The bare agent answered nine turns, so its $0.10 is not a price for this workload but what nine turns cost —
the clearest case for the completion columns anywhere here. The full stack is the only arm that answered all
sixty and still sends 40.8% fewer tokens than the heaviest arm that *tried*; it is also the slowest per turn
(28.6s), the honest trade, since retrieval cycles cost latency and are what keep the conversation inside
128K. Accuracy is low across the board (6–14 of 30) because the model is small — compare arms with each
other, never with a larger model.

---

## 5. What holds across models

### The noise floor

Between two Opus 4.8 runs, four arms executed **byte-identical code** (the only change applied to the
combined arm alone) and all eight runs refused zero calls:

| Opus 4.8 arm, identical code | Correct, run A → run B | Δ tokens |
|---|:--:|---:|
| Baseline (no plugin) | 24/30 → 28/30 | −16.2% |
| Context Graph only | 27/30 → 21/30 | +1.8% |
| Relevance Filtering only | 25/30 → 27/30 | +1.7% |
| Progressive Tool Disclosure only | 27/30 → 29/30 | −5.4% |

Six correct turns moved on the graph arm with no code change, and the plugin-free baseline moved four turns
and 16% of its tokens; on Qwen3 Next the same comparison moved the relevance arm **+79.2%** in tokens for the
same 22 of 30. The agent picks its own tool path, and one extra call early in a 60-turn conversation rides
along in every later one. So the figures that decide something clear that band by a wide margin — −75%
tokens, 102 refused calls against 0, a peak of 38K against 248K — and **the per-arm accuracy ranking within
one model is not one of them.** `--repeats` would resolve it; at ~$254 per Opus replay it is a deliberate
omission.

### Compression and caching optimise the same redundancy

**Amplification** — billed prompt tokens over tokens written to cache once, i.e. how many times the same
content was re-sent — is **66–83×** for the bare agent and the relevance filter, **1.0–1.6×** for all three
combined. The full stack barely re-sends anything, so the cache has nothing cheap left to re-read and
collects only expensive writes. `read:write` is the predictor, legible in every cache table above: 68–92 for
baseline and relevance, 0.0–1.6 for disclosure and the full stack. **The two techniques are alternatives, not
a stack** — relevance filtering excepted, because it compresses once and then stops changing the prompt.

The design rule: **a context plugin is cache-compatible if and only if its mutations are append-only or
confined to the end of the prompt.** Size is not the problem, editing is — disclosure cut the tool schema from
62,656 to 5,300–11,606 tokens and still cost +181% to +737% under caching. Caching also only pays when the
same prefix returns inside the TTL, and turns here land 8–24 seconds apart, a best case: a system prompt per
tenant, a tool set per permission, one-shot fan-out, an A/B prompt split, or a human who pauses longer than
the TTL all pay the write premium and collect no read, and there the plugins are the only lever.

### The window is a proxy for payload mass

| Model | Window | Bare agent answered | Bare peak | Full stack answered | Full stack peak |
|---|---:|:--:|---:|:--:|---:|
| Opus 4.8 / Opus 5 / Fable 5 / Astra / Sol | 1M+ | 60/60 | 19% | 60/60 | 6% |
| Haiku 4.5 | 200K | 60/60 | 98% | 60/60 | 30% |
| GLM 5 | 200K | **18/60** | 97% | 60/60 | 21% |
| GLM 4.7 | 203K | **22/60** | 92% | 59/60 | 51% |
| GLM 4.7 Flash | 203K | **17/60** | 98% | 60/60 | 26% |
| Qwen3 Next | 256K | **13/60** | 97% | 60/60 | 15% |
| Nemotron Nano 9B | 128K | **9/60** | 87% | 60/60 | 57% |

At 1M the argument is cost; below ~250K it is completion. But Haiku 4.5 completed the script at **98% of its
window** while GLM 5, on the same 200K, answered 18 of 60 — so the window is not the variable. It is the
window against the payload mass in front of it, and the window is merely what a model card tells you in
advance.

### No single practice is sufficient

| Model | Single plugin that completed | Best single arm by `Correct` |
|---|---|---|
| GLM 5 | graph | graph, 28/30 |
| GLM 4.7 | graph | graph, 27/30 |
| GLM 4.7 Flash | relevance | relevance, 17/30 |
| Qwen3 Next | graph | relevance, 22/30 (10 refusals) |
| Nemotron Nano 9B | **none** | relevance, 14/30 (64 refusals) |

Every single-plugin arm fails to control the peak on at least one model, and on the smallest window none
completes; the full stack answered **60 of 60 on every model it ran against**. A attacks the payload, B the
fixed schema floor, D the history — different mass, which is why the conjunction holds where its members do
not.

**The regression this document owes the reader.** Removing the filter's `retrieve_context` fixed a real cost
(relevance filtering had been measured at +21.6% on Haiku, and now sits at −5.3%) and cost two specific turns
on Opus 4.8 — `A5-statement` and `R2-cross-reference`, both needing a figure inside a statement payload the
preview cut, with nothing left to ask for it back. `include_retrieval_tool=True` is the way back, at the price
the cache tables show.

---

## 6. Reproducing

```bash
cd validation/community-plugin-A-B-D
./run.sh --total-turns 60 --tag myrun                        # cache off
./run.sh --total-turns 60 --cache default --tag myrun-cache  # Bedrock's own TTL, 5 minutes

VALIDATION_AGENT_MODEL_ID=zai.glm-5 ./run.sh --total-turns 60 --tag tw-glm5
python -m src.run --report-only results/run-myrun.json       # re-render, no credentials needed
```

**One process per model id.** Two concurrent runs against the *same* id share quota and produce
`ServiceUnavailableException`, which contaminates a run without looking like a plugin failure; different ids
have separate quota and are safe in parallel.

Every knob in part 2 reads an environment override, and each run records under `meta.sweep_overrides` which
ones it read, so a result cannot be read without its configuration:

```bash
VALIDATION_PREVIEW_TOKENS=1200 ./run.sh --configs all --total-turns 20 --repeats 3 --tag sweep-preview
VALIDATION_RELEVANCE_RETRIEVAL_TOOL=1 ./run.sh --configs all relevance --total-turns 60 --tag gen1-filter
./run.sh --configs no-disclosure no-relevance no-graph --total-turns 20 --repeats 3 --tag loo
```

`--cache default` uses Bedrock's 5-minute TTL; an explicit `--cache 5m` / `--cache 1h` is rejected by the
pinned botocore 1.40, whose Converse model declares `cachePoint` with `type` alone, so the 1-hour TTL was not
measured. See [`validation/community-plugin-A-B-D/README.md`](validation/community-plugin-A-B-D/README.md) for
what the harness measures and how scoring works.

---

## 7. Not settled

**Replicas.** Every cell in part 4 is one replay against a measured noise floor of ±6 turns. No conclusion
should be read finer than that.

**Caching on the Converse path for the OpenAI models.** Both cards list prompt caching as *Responses API
only*, yet these Converse runs returned cache read and write counts anyway — Astra's first baseline call
measured `inputTokens=2, cacheWriteInputTokens=48,583`. The *rate* is documented; that the traffic happens on
this path at all is not, so the Astra and Sol tables price a documented rate against undocumented behaviour.

**The 5-minute TTL is the best case.** The 1-hour TTL (`"ttl": "1h"` on the `cachePoint`, supported by all
three Claude models) costs more per write and the pinned botocore cannot send it; neither was measured.

**Cross-Region routing stayed put.** One Region per model for the whole day (`inferenceRegion` in the
invocation logs), including Sol and Astra served from `us-east-2` / `us-west-2` while called from `us-east-1`
— so cache reads survive a Geo CRIS hop. The documented high-demand case that increases cache writes did not
occur; Global CRIS was not tested.

**Not every run is the same code.** These runs were measured over several weeks while the packages changed,
and each model's table is its **latest** measurement. Where a model shows a cache-off and a cache-on table,
both come from the same code so the pair is comparable; across models, a difference of a few turns can belong
to the packages having moved rather than to the model.

**These are the community packages**, measured with the three installed from this repository. The forked-SDK
vended plugins place their cache checkpoints in their own code and were **not** re-measured.

**Haiku 4.5's cache rates are inferred** — not on the pricing table; the multipliers used are the family's.
Every other rate is published.
