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

### Parameter generations

Tables in part 4 span three parameter sets; each model section names its own. **A cell from one generation
and a cell from another are two different programs.**

| Gen | Commit | What is different | Runs |
|---|---|---|---|
| **1** | `dd55284` | Filter ships `retrieve_context` (**on**). Graph split in two tunings: alone `0.62` desc 250 body 60,000, with-relevance `0.55` desc 100 body **none**. Preview 800 everywhere. No retrieval ceiling. Graph's `expand_artifact` **dropped** when the filter is installed. | Opus 4.8 ×2, Opus 5 ×2, Fable 5 ×2, Astra, Sol, GLM 5 |
| **2** | `d951195` | Preview **2,000** everywhere. With-relevance desc **250**, cycles 4, live vector index, batched `expand_card`. Disclosure gains the `[+]` sigil, the guessed-call guard, the `find_tools` exemption. Truncated answers scored for what they say. | GLM 4.7, Haiku 4.5, Qwen3 Next |
| **3** | `HEAD` | `retrieve_context` **removed by default**. One unified graph tuning. `body_budget=40,000`. `neighbors_per_candidate=3`. Window regime. `include_artifact_tool=True` everywhere. | Opus 4.8, Haiku 4.5, GLM 5, GLM 4.7 Flash, Qwen3 Next, Nemotron 9B |

Two of those were defects, not tuning: the filter's `retrieve_context` made every retrieval result a
conversation message re-sent on every later call (the mechanism behind relevance filtering costing **+21.6%**
on Haiku in generation 1), and `include_artifact_tool=not relevance_installed` left the combined arm with
**no** artifact tool once the filter stopped registering one.

### What was rejected

GLM 4.7, 20 turns of which 18 scored, 2–3 replays per row, all-three arm, generation 2, each row changing
only what its name says.

| Variant | Total tokens | Δ vs baseline | Correct | Verdict |
|---|---:|---:|:--:|---|
| Baseline, no plugin ✝ | 3,733,922 | — | 12.5/18 | overflowed 6× per replay |
| **Reference (the values above)** | 1,906,838 | **−48.9%** | **16/18** | nothing beat it |
| `expand_threshold` 0.65 | 2,044,890 | −45.2% | 14.5/18 | lost turns **and** cost tokens |
| `ttl_cycles` 12 | 1,604,010 | −57.0% | 14.5/18 | −8pp tokens for −1.5 turns |
| `ttl_cycles` 8 | 1,397,004 | −62.6% | 13/18 | −14pp tokens for −3 turns |
| `catalog_tokens` 48 + `top_k` 6 | 1,563,844 | −58.1% | 13.5/18 | −9pp tokens for −2.5 turns |
| …plus `ttl_cycles` 8 | 1,764,815 | −52.7% | 15/18 | best challenger, still −1 turn |
| `preview_tokens` 1200 | 1,339,228 | −64.1% | 13.5/18 | cheapest row, −2.5 turns |
| `chunk_tokens` 250 | 1,592,800 | −57.3% | 14/18 | finer chunks did not buy selection |
| `relevance_threshold` 0.05 | 1,640,628 | −56.1% | 15/18 | −1 turn, inside the noise |

**The trade is monotone and the defaults sit at its knee.** Every cheaper row scores less, none dominates, and
replays of the *same* row spread by two correct turns — so a variant gaining one turn has gained nothing.
These are 20 turns, where the window does not yet bind; confirm a value at the length you run.

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

| Model | Window | Caching | Generations | Bare agent finished? | Cheapest completing arm |
|---|---:|---|---|:--:|---|
| [Claude Opus 4.8](#41-claude-opus-48) | 1,000,000 | explicit | 1 off, 1 on, 3 off | yes | all three |
| [Claude Opus 5](#42-claude-opus-5) | 1,000,000 | explicit | 1 off, 1 on | yes | all three |
| [Claude Fable 5](#43-claude-fable-5) | 1,000,000 | explicit | 1 off, 1 on | yes | all three |
| [GPT-6 Astra](#44-gpt-6-astra) | 1,050,000 | implicit | 1 | yes | relevance |
| [GPT-5.6 Sol](#45-gpt-56-sol) | 1,000,000 | implicit | 1 | yes | baseline |
| [Claude Haiku 4.5](#46-claude-haiku-45) | 200,000 | explicit | 2, 3 | gen 2 **no**, gen 3 yes | all three |
| [GLM 5](#47-glm-5) | 200,000 | none | 1, 3 | **no** | all three |
| [GLM 4.7](#48-glm-47) | 202,752 | none | 2 | **no** | all three |
| [GLM 4.7 Flash](#49-glm-47-flash) | 202,752 | none | 3 | **no** | all three |
| [Qwen3 Next 80B](#410-qwen3-next-80b) | 256,000 | none | 2, 3 | **no** | all three |
| [Nemotron Nano 9B](#411-nemotron-nano-9b) | 128,000 | none | 3 | **no** | all three |

Every run: 60 turns / 30 scored · 1 replay · `max_output 4,096` · rerank `cohere.rerank-v3-5:0` · embed
`cohere.embed-multilingual-v3` · region `us-east-1`. Deviations are named in the section.

### 4.1 Claude Opus 4.8

`us.anthropic.claude-opus-4-8` · Anthropic · window **1,000,000** · explicit caching (read $0.50, write
$6.25/5m) · $5.00 in / $25.00 out per Mtok · gen 1 off, gen 1 on, gen 3 off ·
[generations 1 and 3](#parameter-generations), **large** regime (preview 800, graph desc 100, cycles 8).

**Generation 1, caching off.**

| Configuration | Total tokens | Δ tokens | Accuracy | Correct | Turn | Cost | Δ cost |
|---|---:|---:|---:|:--:|---:|---:|---:|
| Baseline (no plugin) | 14,377,382 | — | 96.9% | 28/30 | 11.7s | $72.50 | — |
| Progressive Tool Disclosure only | 9,466,084 | −34.2% | 94.5% | 27/30 | 12.8s | $48.10 | −33.7% |
| Relevance Filtering only | 12,766,237 | −11.2% | **96.9%** | **29/30** | 11.7s | $64.45 | −11.1% |
| Context Graph only | 10,385,714 | −27.8% | 93.7% | 27/30 | 10.8s | $52.43 | −27.7% |
| **All three combined** | **2,569,888** | **−82.1%** | 96.1% | 28/30 | **8.7s** | **$13.36** | −81.6% |

**Generation 1, caching on.**

| Configuration | Uncached | Cache read | Cache write | read:write | Billed tokens | Δ tokens | Accuracy | Correct | Turn | Cost | Δ cost |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|---:|---:|---:|
|  Baseline (no plugin) | 174 | 14,842,278 | 215,531 | **68.9** | 15,057,983 | — | 94.5% | 27/30 | **8.0s** | **$9.49** | — |
| Progressive Tool Disclosure only | 202 | 5,578,574 | 3,679,071 | 1.5 | 9,257,847 | −38.4% | 94.5% | 27/30 | 10.0s | $26.65 | +181.0% |
| Relevance Filtering only | 204 | 15,701,512 | 211,522 | **74.2** | 15,913,238 | +5.7% | 94.5% | 27/30 | 10.2s | $10.10 | +6.5% |
| Context Graph only | 576,260 | 8,205,206 | 1,521,353 | 5.4 | 10,302,819 | −31.4% | **99.2%** | **29/30** | 9.2s | $17.29 | +82.3% |
| **All three combined** | 328,419 | 1,067,190 | 2,253,984 | **0.5** | 3,649,593 | −75.5% | 94.5% | 27/30 | 10.4s | $17.14 | +80.7% |

**Generation 3, caching off.**

| Configuration | Total tokens | Δ tokens | Accuracy | Correct | Peak/call | Turn | Cost | Δ cost |
|---|---:|---:|---:|:--:|---:|---:|---:|---:|
| Baseline (no plugin) | 13,647,898 | — | 97.6% | 28/30 | 192,662 (19%) | 11.3s | $68.89 | — |
| Progressive Tool Disclosure only | 9,381,497 | −31.3% | **97.6%** | **29/30** | 139,961 (14%) | 12.6s | $47.74 | −30.7% |
| Relevance Filtering only | 13,447,761 | −1.5% | 89.8% | 27/30 | 189,714 (19%) | 12.2s | $67.96 | −1.4% |
| Context Graph only | 10,254,016 | −24.9% | 85.0% | 21/30 | 149,010 (15%) | 10.1s | $51.84 | −24.7% |
| **All three combined** | **3,348,835** | **−75.5%** | 91.3% | 26/30 | **59,405 (6%)** | 10.6s | **$17.47** | −74.6% |

**Observations.** The reference model for the cost argument: nothing truncates, so every column is comparable
— **−82.1%** and **−75.5%** for the same 28 and 26 of 30, at a fifth of the cost. Caching inverts the
ranking and `read:write` says why: baseline 68.9 at **$9.49**, full stack 0.5 at $17.14, **1.8× more than
doing nothing**; only relevance filtering stays cache-friendly (74.2). Generation 3 shows that generation's
two regressions — relevance at −1.5% / 27-of-30 loses `A5-statement` and `R2-cross-reference`, both needing a
figure inside a statement payload the preview cut with `retrieve_context` gone, and the combined arm loses the
same two. The graph arm's 21/30 is **not** attributable: the identical arm scored 27/30 on another run.

### 4.2 Claude Opus 5

`us.anthropic.claude-opus-5` · Anthropic · window **1,000,000** · explicit caching (read $0.50, write
$6.25/5m) · $5.00 in / $25.00 out per Mtok · gen 1 off, gen 1 on. [Generation 1](#parameter-generations).

**Caching off.**

| Configuration | Total tokens | Δ tokens | Accuracy | Correct | Turn | Cost | Δ cost |
|---|---:|---:|---:|:--:|---:|---:|---:|
| Baseline (no plugin) | 19,778,235 | — | 89.8% | 24/30 | 24.3s | $100.44 | — |
| Progressive Tool Disclosure only ⚠ | 647,368 | −96.7% | 31.5% | 1/30 | 1.8s | $3.25 | −96.8% |
| Relevance Filtering only | 18,106,860 | −8.5% | 96.9% | 28/30 | 21.4s | $91.89 | −8.5% |
| Context Graph only | 13,527,548 | −31.6% | 93.7% | 25/30 | 26.0s | $69.54 | −30.8% |
| **All three combined** | **5,565,805** | **−71.9%** | **100.0%** | **30/30** | 17.9s | **$29.17** | −71.0% |

**Caching on.**

| Configuration | Uncached | Cache read | Cache write | read:write | Billed tokens | Δ tokens | Accuracy | Correct | Turn | Cost | Δ cost |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|---:|---:|---:|
| Baseline (no plugin) | 196 | 26,321,602 | 285,328 | **92.3** | 26,607,126 | — | 92.1% | 24/30 | 23.2s | $17.16 | — |
| Progressive Tool Disclosure only | 218 | 10,217,732 | 6,296,888 | 1.6 | 16,514,838 | −37.9% | 92.1% | 26/30 | 18.7s | $46.07 | +168.5% |
| Relevance Filtering only | 200 | 17,343,538 | 237,313 | **73.1** | 17,581,051 | −33.9% | 92.9% | 25/30 | 19.7s | **$12.06** | −29.7% |
| Context Graph only | 664,269 | 9,508,166 | 2,898,117 | 3.3 | 13,070,552 | −50.7% | 94.5% | 27/30 | 22.0s | $28.34 | +65.2% |
| **All three combined** | 363,512 | 1,526,446 | 3,470,771 | **0.4** | 5,360,729 | −79.6% | **95.3%** | 27/30 | 18.3s | $25.98 | +51.4% |

**Observations.** The full stack was the most accurate arm of the cache-off run — **30 of 30** against the
bare agent's 24, the only perfect score here; one replay, so read it as a path the agent can take. ⚠ The
cache-off disclosure arm is not a result: the model never called `find_tools`, ran with no tool schema and
answered 1 of 30 at 1.8s per turn, while cached the same arm scored 26 of 30 — which is why a −96.7% token
figure sits next to a useless run. The baseline spent 35% more billed tokens with caching than without,
purely because the agent made 7 more tool calls: compare arms within a run, not across runs.

### 4.3 Claude Fable 5

`us.anthropic.claude-fable-5` · Anthropic · window **1,000,000** · explicit caching (read $1.00, write
$12.50/5m) · **$10.00 in / $50.00 out** per Mtok, the most expensive model here · gen 1 off, gen 1 on.
[Generation 1](#parameter-generations).

**Caching off.**

| Configuration | Total tokens | Δ tokens | Accuracy | Correct | Turn | Cost | Δ cost |
|---|---:|---:|---:|:--:|---:|---:|---:|
| Baseline (no plugin) | 21,620,903 | — | **100.0%** | **30/30** | 20.5s | $217.86 | — |
| Progressive Tool Disclosure only | 14,284,626 | −33.9% | 97.6% | 29/30 | 20.5s | $144.58 | −33.6% |
| Relevance Filtering only | 17,263,244 | −20.2% | 97.6% | 29/30 | 18.1s | $174.08 | −20.1% |
| Context Graph only | 12,645,472 | −41.5% | 96.1% | 28/30 | 21.0s | $128.54 | −41.0% |
| **All three combined** | **5,311,719** | **−75.4%** | 96.9% | 28/30 | 22.1s | **$55.32** | −74.6% |

**Caching on.**

| Configuration | Uncached | Cache read | Cache write | read:write | Billed tokens | Δ tokens | Accuracy | Correct | Turn | Cost | Δ cost |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|---:|---:|---:|
| Baseline (no plugin) | 190 | 18,254,571 | 250,858 | **72.8** | 18,505,619 | — | 99.2% | **30/30** | 16.1s | $23.38 | — |
| Progressive Tool Disclosure only | 248 | 8,001,876 | 6,708,508 | 1.2 | 14,710,632 | −20.5% | 98.4% | 29/30 | 19.3s | $93.96 | +301.8% |
| Relevance Filtering only | 212 | 17,687,881 | 229,005 | **77.2** | 17,917,098 | −3.2% | 98.4% | 29/30 | **15.7s** | **$22.48** | −3.9% |
| Context Graph only | 812,339 | 10,368,477 | 2,528,084 | 4.1 | 13,708,900 | −25.7% | 98.4% | 29/30 | 21.7s | $53.19 | +127.5% |
| **All three combined** | 360,961 | 1,291,253 | 3,567,890 | **0.4** | 5,220,104 | −71.5% | 96.1% | 27/30 | 23.3s | $52.58 | +124.9% |

**Observations.** The bare agent is perfect here and the most expensive row in the document — 30 of 30 for
**$217.86** against the full stack's 28 of 30 for **$55.32**, a $162 difference on one conversation, which is
where the saving matters most in absolute dollars. Cache-on inverts as on every Claude model: disclosure at
1.2 costs **+301.8%**, and only relevance filtering (77.2) is cheaper than doing nothing. The two
non-overflow errors in the cache-off run were transport failures, not window failures.

### 4.4 GPT-6 Astra

`us.openai.gpt-6-astra` · OpenAI · window **1,050,000** · **implicit** caching applied by the service, not
requested (read $1.10, write $13.75 = 1.25× input) · $11.00 in / $55.00 out per Mtok · gen 1, served from
`us-west-2` while called from `us-east-1`. [Generation 1](#parameter-generations).

| Configuration | Uncached | Cache read | Cache write | read:write | Billed tokens | Δ tokens | Accuracy | Correct | Turn | Cost | Δ cost |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|---:|---:|---:|
| Baseline (no plugin) | 228 | 15,270,589 | 213,127 | **71.7** | 15,483,944 | — | 92.9% | 26/30 | 10.4s | $21.20 | — |
| Progressive Tool Disclosure only | 252 | 7,810,974 | 5,534,968 | 1.4 | 13,346,194 | −13.8% | 92.9% | 26/30 | 10.3s | $86.18 | +306.5% |
| Relevance Filtering only | 222 | 12,069,886 | 185,578 | **65.0** | 12,255,686 | −20.8% | 88.2% | 23/30 | 10.2s | **$17.33** | −18.3% |
| Context Graph only | 252 | 1,473,390 | 8,536,045 | **0.2** | 10,009,687 | −35.2% | 91.3% | 25/30 | 10.9s | $120.59 | +468.8% |
| **All three combined** | 292 | 17,588 | 4,051,747 | **0.0** | 4,069,627 | −73.5% | **94.5%** | **27/30** | **9.2s** | $57.38 | +170.7% |

**Observations.** Implicit caching cannot be turned off, so there is no uncached control: every row pays the
write premium, and the prefix-mutating plugins pay it on nearly everything — the graph reads 1.47M against
8.54M written and costs **+468.8%**. The full stack sends 73.5% fewer tokens and costs 170.7% more, which is
the caching argument in one row; it is still the most accurate (27 of 30) and fastest arm, so accuracy
objective → unchanged ranking, bill objective → relevance filtering alone.

### 4.5 GPT-5.6 Sol

`us.openai.gpt-5.6-sol` · OpenAI · window **1,000,000** · **implicit** caching (read $0.44, write $5.50) ·
$4.40 in / $22.00 out per Mtok · gen 1, served from `us-east-2` while called from `us-east-1`. Parameters as
[4.1](#41-claude-opus-48), generation 1.

| Configuration | Uncached | Cache read | Cache write | read:write | Billed tokens | Δ tokens | Accuracy | Correct | Turn | Cost | Δ cost |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|---:|---:|---:|
| Baseline (no plugin) | 242 | 15,679,917 | 218,227 | **71.9** | 15,898,386 | — | **97.6%** | **29/30** | **9.0s** | **$8.82** | — |
| Progressive Tool Disclosure only | 340 | 11,120,494 | 12,432,437 | 0.9 | 23,553,271 | +48.0% | 96.9% | 28/30 | 10.8s | $73.82 | +737.2% |
| Relevance Filtering only | 276 | 17,145,788 | 208,099 | **82.4** | 17,354,163 | +9.2% | **97.6%** | **29/30** | 10.3s | $9.57 | +8.5% |
| Context Graph only | 278 | 790,589 | 10,825,077 | **0.1** | 11,615,944 | −26.8% | 91.3% | 25/30 | 9.9s | $60.56 | +586.8% |
| **All three combined** | 344 | 17,861 | 6,811,669 | **0.0** | 6,829,874 | −56.8% | 96.1% | 28/30 | 10.0s | $38.55 | +337.3% |

**Observations.** The one model where doing nothing wins on every axis at once — cheapest, most accurate,
fastest. A 1M window, free implicit caching and a reused prefix leave nothing for compression to buy. The
disclosure arm sent **more** tokens than the bare agent (+48.0%) and wrote 12.4M of them: editing the tool
section invalidates the prefix, so each reveal turn reprocesses everything after it at write rate. Relevance
filtering's +8.5% is the smallest penalty, and the one to consider if something other than cost demands
compression here.

### 4.6 Claude Haiku 4.5

`us.anthropic.claude-haiku-4-5-20251001-v1:0` · Anthropic · window **200,000** · explicit caching (read
$0.10, write $1.25/5m — **inferred**) · $1.00 in / $5.00 out per Mtok · gen 2 off, gen 3 off ·
[generations 2 and 3](#parameter-generations), **tight** regime (preview 2,000, graph desc 250, cycles 4).

**Generation 2.**

| Configuration | Answered | Scored lost | Refused | Total tokens | Δ vs heaviest | Accuracy | Correct | Peak/call | Turn | Cost |
|---|:--:|:--:|---:|---:|---:|---:|:--:|---:|---:|---:|
| Baseline (no plugin) ✝ | **24/60** | **12/30** | **72** | 6,058,882 | −54.1% | 65.3% | 13/30 | 193,683 (97%) | 3.9s ✝ | $6.09 |
| Progressive Tool Disclosure only | 60/60 | 0/30 | 0 | 12,156,014 | −8.0% | 88.2% | 22/30 | 167,060 (84%) | 8.6s | $12.26 |
| Relevance Filtering only ← ref | 60/60 | 0/30 | 0 | 13,214,041 | — | **93.7%** | **26/30** | 177,119 (89%) | 8.6s | $13.32 |
| Context Graph only | 60/60 | 0/30 | 0 | 7,894,451 | −40.3% | 79.5% | 18/30 | 118,442 (59%) | 7.0s | $7.97 |
| **All three combined** | 60/60 | 0/30 | 0 | **3,260,491** | **−75.3%** | 89.8% | 25/30 | **56,545 (28%)** | 5.8s | **$3.34** |

**Generation 3.**

| Configuration | Total tokens | Δ tokens | Accuracy | Correct | Peak/call | Turn | Cost | Δ cost |
|---|---:|---:|---:|:--:|---:|---:|---:|---:|
| Baseline (no plugin) | 12,703,571 | — | **95.3%** | **27/30** | 196,722 (**98%**) | 7.9s | $12.77 | — |
| Progressive Tool Disclosure only | 10,296,950 | −18.9% | 88.2% | 22/30 | 160,920 (80%) | 7.9s | $10.37 | −18.8% |
| Relevance Filtering only | 12,036,064 | −5.3% | 86.6% | 21/30 | 166,779 (83%) | 9.3s | $12.15 | −4.9% |
| Context Graph only | 7,883,137 | −37.9% | 76.4% | 17/30 | 114,003 (57%) | 7.8s | $7.95 | −37.7% |
| **All three combined** | **3,224,286** | **−74.6%** | 84.2% | 20/30 | **60,393 (30%)** | 6.6s | **$3.32** | −74.0% |

**Observations.** Same model, same window, and the bare agent truncated in one run and not the other — 24 of
60 turns in generation 2, all sixty in generation 3 with its largest call at **98% of the window**. No plugin
can reach the baseline, so that is the tool path landing just inside or just outside the window, and the
sharpest argument for `Answered` being a column; accuracy across the two runs is not comparable. This is also
where relevance filtering was first measured costing more than nothing (+21.6%), which motivated removing its
retrieval tool — it worked, the same arm moves to −5.3%. The full stack is the cheapest arm in both runs and
within a cent of itself ($3.34, $3.32) while the baseline moved by a factor of two: its cost does not depend
on how much history there is.

### 4.7 GLM 5

`zai.glm-5` · Zhipu · window **200,000** · **no caching published**, so there is nothing to weigh the plugins
against · $1.00 in / $3.20 out per Mtok · gen 1, gen 3 ·
[generations 1 and 3](#parameter-generations); gen 1 ran preview 800 before the regime existed, gen 3 the
**tight** regime (preview 2,000, graph desc 250, cycles 4).

**Generation 1.**

| Configuration | Answered | Scored lost | Refused | Total tokens | Δ vs heaviest | Accuracy | Correct | Peak/call | Turn | Cost |
|---|:--:|:--:|---:|---:|---:|---:|:--:|---:|---:|---:|
| Baseline (no plugin) ✝ | **15/60** | **15/30** | **90** | 5,252,666 | −72.9% | 63.8% | 14/30 | 198,588 (99%) | 12.7s ✝ | $5.26 |
| Progressive Tool Disclosure only ✝ | **41/60** | **7/30** | **38** | 11,327,573 | −41.6% | 74.8% | 20/30 | 198,249 (99%) | 30.1s ✝ | $11.37 |
| Relevance Filtering only ← ref | 60/60 | 0/30 | 0 | 19,406,514 | — | **98.4%** | **29/30** | 192,751 (96%) | 48.8s | $19.48 |
| Context Graph only | 60/60 | 0/30 | 0 | 8,931,685 | −54.0% | 85.0% | 23/30 | 117,147 (59%) | 28.1s | $8.99 |
| **All three combined** | 60/60 | 0/30 | 0 | **2,457,277** | **−87.3%** | 92.1% | 26/30 | **43,490 (22%)** | 12.0s | **$2.51** |

**Generation 3.**

| Configuration | Answered | Scored lost | Refused | Total tokens | Δ vs heaviest | Accuracy | Correct | Peak/call | Turn | Cost |
|---|:--:|:--:|---:|---:|---:|---:|:--:|---:|---:|---:|
| Baseline (no plugin) ✝ | **18/60** | **14/30** | **84** | 6,150,161 | −58.3% | 65.3% | 14/30 | 193,207 (97%) | 12.6s ✝ | $6.17 |
| Progressive Tool Disclosure only ✝ | **25/60** | **11/30** | **64** | 9,170,600 | −37.8% | 66.1% | 16/30 | 193,113 (97%) | 34.0s ✝ | $9.22 |
| Relevance Filtering only ✝ ← ref | 55/60 | 3/30 | 10 | 14,739,065 | — | 80.3% | 24/30 | 198,011 (99%) | 35.5s | $14.80 |
| Context Graph only | 60/60 | 0/30 | 0 | 7,175,940 | −51.3% | **96.1%** | **28/30** | 102,645 (51%) | 20.5s | $7.21 |
| **All three combined** | 60/60 | 0/30 | 0 | **2,908,717** | **−80.3%** | 81.1% | 21/30 | **41,553 (21%)** | 14.0s | **$2.96** |

**Observations.** The model that turns the question from cost into completion, and it reproduces: in both
generations the bare agent lost 84–90 calls and answered 15–18 of 60, so its 63.8% and 65.3% are mostly
absence with 14–15 of its 30 scored turns never attempted. Only the graph and the full stack finished intact,
and they are the only arms peaking below 60% of the window — every arm at 96% or above lost calls. Cost stops
being the argument: the full stack costs $2.51–$2.96 and answers everything, the bare agent $5.26–$6.17 for a
quarter of the script. The graph alone is the most accurate arm of generation 3 (28/30) and the second
cheapest — on *this* model the single plugin that keeps you inside the window is the graph, which is not true
of the next two, and is why more than one vendor is measured.

### 4.8 GLM 4.7

`zai.glm-4.7` · Zhipu · window **202,752** · no caching published · $0.60 in / $2.20 out per Mtok · [generation 2](#parameter-generations),
**tight** regime.

| Configuration | Answered | Scored lost | Refused | Total tokens | Δ vs heaviest | Accuracy | Correct | Peak/call | Turn | Cost | $/correct |
|---|:--:|:--:|---:|---:|---:|---:|:--:|---:|---:|---:|---:|
| Baseline (no plugin) ✝ | **22/60** | **12/30** | **70** | 5,814,709 | −58.9% | 65.3% | 14/30 | 186,905 (92%) | 18.4s ✝ | $3.52 | *$0.25* |
| Progressive Tool Disclosure only ✝ ← ref | **40/60** | **6/30** | **30** | 14,153,824 | — | 73.2% | 20/30 | 195,178 (96%) | 47.4s ✝ | $8.56 | *$0.43* |
| Relevance Filtering only ✝ | **27/60** | **11/30** | **54** | 11,140,007 | −21.3% | 66.1% | 17/30 | 194,955 (96%) | 36.4s ✝ | $6.75 | *$0.40* |
| Context Graph only | 60/60 | 0/30 | 0 | 13,191,178 | −6.8% | **95.3%** | **27/30** | 176,766 (87%) | 36.4s | $7.96 | $0.29 |
| **All three combined** | 59/60 † | 0/30 | 0 | **5,916,862** | **−58.2%** | 83.5% | 22/30 | **102,796 (51%)** | 23.6s | **$3.60** | **$0.16** |

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
the cheapest model here · [generation 3](#parameter-generations), **tight** regime.

| Configuration | Answered | Scored lost | Refused | Total tokens | Δ vs heaviest | Accuracy | Correct | Peak/call | Turn | Cost |
|---|:--:|:--:|---:|---:|---:|---:|:--:|---:|---:|---:|
| Baseline (no plugin) ✝ | **17/60** | **15/30** | **82** | 5,398,019 | −80.3% | 52.8% | 8/30 | 198,536 (98%) | 7.3s ✝ | $0.38 |
| Progressive Tool Disclosure only ✝ ← ref | **36/60** | **7/30** | **40** | 27,403,577 | — | 58.3% | 11/30 | 198,654 (98%) | 33.0s ✝ | $1.93 |
| Relevance Filtering only | 60/60 | 0/30 | 0 | 17,181,923 | −37.3% | **75.6%** | 17/30 | 187,849 (93%) | 13.4s | $1.22 |
| Context Graph only ✝ | **52/60** | **2/30** | **8** | 18,123,481 | −33.9% | 74.8% | **18/30** | 198,646 (98%) | 29.7s ✝ | $1.28 |
| **All three combined** | 60/60 | 0/30 | 0 | **4,010,248** | **−85.4%** | 55.1% | 10/30 | **53,355 (26%)** | 11.6s | **$0.30** |

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
Mtok · gen 2, gen 3 · regime **tight** — above 250K and still tight, see
[the window regime](#the-window-regime) · [generations 2 and 3](#parameter-generations).

**Generation 2.**

| Configuration | Answered | Scored lost | Refused | Total tokens | Δ vs heaviest | Accuracy | Correct | Peak/call | Turn | Cost |
|---|:--:|:--:|---:|---:|---:|---:|:--:|---:|---:|---:|
| Baseline (no plugin) ✝ | **14/60** | **16/30** | **92** | 8,231,351 | −75.1% | 59.1% | 11/30 | 257,163 (100%) | 6.0s ✝ | $1.16 |
| Progressive Tool Disclosure only ← ref | 60/60 | 0/30 | 0 | 33,004,333 | — | 86.6% | 23/30 | 248,224 (97%) | 19.6s | $4.64 |
| Relevance Filtering only | 60/60 | 0/30 | 0 | 21,803,250 | −33.9% | **91.3%** | **26/30** | 241,712 (94%) | 16.5s | $3.10 |
| Context Graph only ✝ | **16/60** | **15/30** | **88** | 25,119,509 | −23.9% | 55.9% | 10/30 | 257,560 (**101%**) | 13.9s ✝ | $3.53 |
| **All three combined** | 59/60 | 0/30 | 0 | **5,512,454** | **−83.3%** | 81.9% | 23/30 | **92,972 (36%)** | 11.9s | **$0.82** |

**Generation 3.**

| Configuration | Answered | Scored lost | Refused | Total tokens | Δ vs heaviest | Accuracy | Correct | Peak/call | Turn | Cost |
|---|:--:|:--:|---:|---:|---:|---:|:--:|---:|---:|---:|
| Baseline (no plugin) ✝ | **13/60** | **17/30** | **94** | 8,591,570 | −74.1% | 54.3% | 9/30 | 248,552 (97%) | 5.6s ✝ | $1.21 |
| Progressive Tool Disclosure only ✝ | **13/60** | **17/30** | **94** | 5,033,909 | −84.8% | 60.6% | 11/30 | 252,395 (99%) | 4.0s ✝ | $0.71 |
| Relevance Filtering only ✝ ← ref | 55/60 | 3/30 | 10 | 33,149,405 | — | 79.5% | **22/30** | 256,310 (100%) | 19.9s | $4.71 |
| Context Graph only | 60/60 | 0/30 | 0 | 9,727,355 | −70.7% | **81.1%** | 20/30 | 136,350 (53%) | 8.5s | $1.38 |
| **All three combined** | 60/60 | 0/30 | 0 | **4,395,046** | **−86.7%** | 74.0% | 17/30 | **38,496 (15%)** | 10.2s | **$0.68** |

**Observations.** The heaviest payload mass of the battery, and the reason the tight ceiling is 300K: a
256,000-token window and the bare agent still peaks at it and loses 92–94 calls. Which single plugin survives
changes between generations — the graph peaked at **101% of the window** in generation 2 (16 of 60 answered)
and at 53% in generation 3 (all sixty), while disclosure went the other way, 60 of 60 then 13 of 60. Only the
full stack answered nearly everything in both. The generation 3 disclosure row is the trap in one line: 5.0M
tokens and $0.71 look efficient until `Answered` reads 13 of 60 — the second-cheapest cell of the run and the
second-worst outcome.

### 4.11 Nemotron Nano 9B

`nvidia.nemotron-nano-9b-v2` · NVIDIA · window **128,000**, the smallest here · no caching published ·
$0.06 in / $0.23 out per Mtok · [generation 3](#parameter-generations), **tight** regime.

| Configuration | Answered | Scored lost | Refused | Total tokens | Δ vs heaviest | Accuracy | Correct | Peak/call | Turn | Cost |
|---|:--:|:--:|---:|---:|---:|---:|:--:|---:|---:|---:|
| Baseline (no plugin) ✝ | **9/60** | **21/30** | **102** | 1,596,017 | −85.3% | 48.8% | 6/30 | 111,016 (87%) | 4.2s ✝ | $0.10 |
| Progressive Tool Disclosure only ✝ | **14/60** | **16/30** | **92** | 2,298,800 | −78.8% | 59.1% | 9/30 | 118,522 (93%) | 7.8s ✝ | $0.14 |
| Relevance Filtering only ✝ | **28/60** | **11/30** | **64** | 5,073,784 | −53.3% | 66.9% | **14/30** | 125,719 (98%) | 12.4s ✝ | $0.33 |
| Context Graph only ✝ ← ref | **43/60** | **7/30** | **34** | 10,868,671 | — | 59.8% | 11/30 | 124,762 (97%) | 22.8s ✝ | $0.67 |
| **All three combined** | **60/60** | **0/30** | **0** | 6,431,158 | −40.8% | **68.5%** | 12/30 | **73,358 (57%)** | 28.6s | $0.44 |

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
| Haiku 4.5 gen 3 | 200K | 60/60 | 98% | 60/60 | 30% |
| Haiku 4.5 gen 2 | 200K | **24/60** | 97% | 60/60 | 28% |
| GLM 5 | 200K | **15–18/60** | 99% | 60/60 | 21% |
| GLM 4.7 | 203K | **22/60** | 92% | 59/60 | 51% |
| GLM 4.7 Flash | 203K | **17/60** | 98% | 60/60 | 26% |
| Qwen3 Next | 256K | **13–14/60** | 100% | 59–60/60 | 15–36% |
| Nemotron Nano 9B | 128K | **9/60** | 87% | 60/60 | 57% |

At 1M the argument is cost; below ~250K it is completion. But Haiku 4.5 appears twice with the same window
and opposite outcomes, so the window is not the variable — it is the window against the payload mass in front
of it, and the window is merely what a model card tells you in advance.

### No single practice is sufficient

| Model | Single plugin that completed | Best single arm by `Correct` |
|---|---|---|
| GLM 5 | graph (both generations) | graph, 28/30 |
| GLM 4.7 | graph | graph, 27/30 |
| GLM 4.7 Flash | relevance | relevance, 17/30 |
| Qwen3 Next gen 2 | disclosure, relevance | relevance, 26/30 |
| Qwen3 Next gen 3 | graph | relevance, 22/30 (10 refusals) |
| Nemotron Nano 9B | **none** | relevance, 14/30 (64 refusals) |

Every single-plugin arm fails to control the peak on at least one model, and on the smallest window none
completes; the full stack answered **60 of 60 on every model it ran against**. A attacks the payload, B the
fixed schema floor, D the history — different mass, which is why the conjunction holds where its members do
not.

**The regression this document owes the reader.** Removing the filter's `retrieve_context` in generation 3
fixed a real cost (relevance went +21.6% → −5.3% on Haiku) and cost two specific turns on Opus 4.8 —
`A5-statement` and `R2-cross-reference`, both needing a figure inside a statement payload the preview cut,
with nothing left to ask for it back. `include_retrieval_tool=True` is the way back, at the price the cache
tables show.

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

**Generations are not comparable** — part 2 says which code each run used, and the differences between
generations are larger than most deltas here.

**These are the community packages**, measured with the three installed from this repository. The forked-SDK
vended plugins place their cache checkpoints in their own code and were **not** re-measured.

**Haiku 4.5's cache rates are inferred** — not on the pricing table; the multipliers used are the family's.
Every other rate is published.
