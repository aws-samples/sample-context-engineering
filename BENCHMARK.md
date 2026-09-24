# Benchmark

Eleven models, one scripted conversation, five configurations each. This page is organised **by model**:
what the model is, what the plugins were set to, what the harness was set to, what came out, and what that
run does and does not support.

Read part 1 and part 2 first — they define the script and the parameters every table below refers to. Part
4 is the model list. Part 5 is what survives across models; nothing in a single model's section should be
generalised on its own.

- [1. What a run is](#1-what-a-run-is)
- [2. The plugins and their parameters](#2-the-plugins-and-their-parameters)
- [3. Prices](#3-prices)
- [4. The models](#4-the-models)
- [5. What holds across the models](#5-what-holds-across-the-models)
- [6. Reproducing this](#6-reproducing-this)
- [7. What is not settled](#7-what-is-not-settled)

---

## 1. What a run is

### The script

One scripted conversation of **60 turns**, replayed start to finish for each configuration. The same 60
prompts in the same order every time, so two configurations differ only in what the agent was allowed to
send — never in what it was asked.

The scenario is a personal-finance and cloud-operations assistant over **mocked tools**: account listings,
investment positions, portfolio allocations, statement exports, connector diagnostics, CloudWatch metrics,
IAM policies, S3 listings and documentation lookups. Several tools return **40,000–120,000 characters** in
one result, which is what makes the context grow: the payload mass is in the tool results, not in the
conversation.

**Half the script is scored.** 30 of the 60 turns carry weighted expectations — 18 hand-written plus 12
generated. The other 30 are unscored mass, present to lengthen the history, and they are *grounded*: each
makes a real tool call and contributes a real payload.

### How correctness is scored

Deterministically, never by an LLM judge. The tools are mocked, so every factual question has one
computable answer, and ground truth is derived from the tool implementations rather than written by hand
(`src/ground_truth.py`). Expectations are strings the answer must contain, may contain, or must **not**
contain — the last catching the confidently wrong answers a degraded context produces.

| Column | What it means |
|---|---|
| **Accuracy** | fraction of expectation *weight* met across the run. Partial credit: a turn that states the right figure but omits a secondary fact scores between 0 and 1. |
| **Correct** | turns with no *critical* failure — the fact the question was actually asking for. The stricter reading, and why a configuration can gain accuracy while losing a turn. |

Figures must match **as the tools formatted them**. A paraphrased or reformatted number fails even when the
arithmetic is right: an assistant that restates `42,1%` as `42.1%` has introduced an error class, and this
harness treats it as one on purpose. See `src/accuracy.py`.

### The five arms

| Arm | What runs |
|---|---|
| Baseline (no plugin) | nothing. Every tool payload enters the history whole and stays there. The honest control: it measures the cost of doing nothing. |
| Progressive Tool Disclosure only | practice B |
| Relevance Filtering only | practice A |
| Context Graph only | practice D |
| **All three combined** | A + B + D |

Three leave-one-out arms (`no-disclosure`, `no-relevance`, `no-graph`) exist and are reachable from
`--configs`, but are deliberately not in the default set.

### The columns, and the three traps

Column format follows the repository README: **Total tokens · Δ tokens · Accuracy · Correct · Turn · Cost ·
Δ cost**. Three column groups are added only where the data requires them, and their absence is
information too:

- **Cache columns** (`Uncached · Cache read · Cache write · read:write · Billed tokens`) appear only for
  runs that actually served cache reads. In a cached run `inputTokens` stops measuring input, so a
  single token column would be wrong.
- **Completion columns** (`Answered · Scored lost · Refused`) appear only where at least one arm had calls
  refused by the provider. `Answered` is how many of the 60 turns produced an answer at all; `Scored lost`
  how many of the 30 scored turns were never answered; `Refused` how many calls came back
  `ContextWindowOverflow`.
- **Peak/call** appears for every model whose window is ≤ 300K, as an absolute and as a share of the
  window.

**Trap 1 — a truncated arm looks cheap.** An arm that stopped answering also stopped spending. Its token
total is understated (a refused call is not billed), its accuracy is bounded by truncation rather than by
the strategy, and its seconds-per-turn is the pace of an abandoned conversation. Rows like that are marked
✝ and must be read from `Answered` first.

**Trap 2 — the Δ needs the right reference.** Where the baseline itself truncated, **Δ is measured against
the arm that consumed the most tokens** (marked `← ref`), not against the baseline. A baseline that
answered 9 of 60 turns stopped accumulating tokens when it stopped answering, so a delta against it prints
as a regression for the arms that completed the script. The heaviest arm is the closest thing a run has to
the full workload uncompressed.

**Trap 3 — one replay per cell.** Unless a table says otherwise, every cell is a single replay. The agent
chooses its own tool path, so two replays of the same configuration differ on their own. This is measured,
not assumed — see [the noise floor](#the-noise-floor), where four byte-identical arms moved by up to six
materially-correct turns. **Nothing under ±6 turns or ±20% of tokens is evidence at n=1.**

**Peak input per call is the truncation-immune quantity.** It is measured on calls that went out, so it
cannot be flattered by an arm that stopped working. Where it is shown as an *estimated* prompt it can
exceed 100% of the window — that is a call the provider refused.

---

## 2. The plugins and their parameters

Three practices, three installable packages. Nothing here forks the SDK.

| | Practice | Package | What it does |
|---|---|---|---|
| **A** | Relevance filtering | [`strands-relevance-filter`](community-plugins/strands-relevance-filter/) | Scores a tool result's chunks against the question and keeps only what answers it, replacing the payload with a preview **before it enters the history**. |
| **B** | Progressive tool disclosure | [`strands-progressive-tool-disclosure`](community-plugins/strands-progressive-tool-disclosure/) | Sends a lean tool catalog and fetches a tool's full spec on demand, then forgets it — attacking the fixed ~63k schema floor. |
| **D** | Context graph | [`strands-context-graph`](community-plugins/strands-context-graph/) | Reorganises the history as a graph with remove/recover over an immutable log, folding each turn to full content, Description or Title. |

The two act at different moments, which is why they compose: **the filter acts on a tool result before it
enters the history; the graph acts on a history that already exists.** The filter only makes the graph's
input smaller.

### Current parameters (generation 3)

These are the values in `src/config.py` today, and what a reproduction run uses. Every one reads an
override off the environment, and each run records under `meta.sweep_overrides` which ones it read.

| Plugin | Parameter | Value | Env override |
|---|---|---:|---|
| A | `max_result_tokens` (trigger) | 4,000 | `VALIDATION_MAX_RESULT_TOKENS` |
| A | `chunk_tokens` | 500 | `VALIDATION_CHUNK_TOKENS` |
| A | `preview_tokens` | **by regime** — 800 large / 2,000 tight | `VALIDATION_PREVIEW_TOKENS` |
| A | `relevance_threshold` | 0.02 | `VALIDATION_RELEVANCE_THRESHOLD` |
| A | `include_retrieval_tool` | **False** | `VALIDATION_RELEVANCE_RETRIEVAL_TOOL` |
| B | `catalog_tokens` | 20 | `VALIDATION_CATALOG_TOKENS` |
| B | `ttl_cycles` | 5 | `VALIDATION_TTL_CYCLES` |
| B | `top_k` | 4 | `VALIDATION_TOP_K` |
| B | `catalog_in_system_prompt` | False | `VALIDATION_CATALOG_IN_SYSTEM_PROMPT` |
| D | `expand_threshold` | 0.62 | `VALIDATION_GRAPH_EXPAND` |
| D | `collapse_floor` | 0.45 | `VALIDATION_GRAPH_COLLAPSE` |
| D | `link_threshold` | 0.50 | `VALIDATION_GRAPH_LINK` |
| D | `description_tokens` | **by regime** — 100 large / 250 tight | `VALIDATION_GRAPH_DESCRIPTION_TOKENS` |
| D | `body_budget` | **40,000** | `VALIDATION_GRAPH_BODY_BUDGET` |
| D | `max_retrieval_cycles` | **by regime** — 8 large / 4 tight | `VALIDATION_GRAPH_MAX_RETRIEVAL_CYCLES` |
| D | `reuse_ttl_cycles` | 5 | `VALIDATION_GRAPH_REUSE_TTL` |
| D | `tags_per_card` | 5 | `VALIDATION_GRAPH_TAGS` |
| D | `neighbors_per_candidate` | 3 | `VALIDATION_GRAPH_NEIGHBORS` |
| D | `min_cards` | 3 | `VALIDATION_MIN_CARDS` |
| D | `include_artifact_tool` | True, every arm | — |

`relevance_threshold=0.02` is not a low bar by mistake: it is a position in a distribution, and
`cohere.rerank-v3-5` returns strong matches around 0.29, so the package default of 0.5 rejects every chunk.

`body_budget` is the only ceiling on how much travels at full content. `expand_threshold` is a per-Card
*classifier* and cannot see the total, so with `body_budget=None` a long enough conversation sends every
Card above the threshold whole — growth linear in the conversation, bounded by nothing.

### The window regime

Three budgets are not one number but two, and which pair applies is decided by the model's declared window
against `TIGHT_WINDOW_CEILING = 300_000`. Recorded in every run as `meta.window_regime`, overridable with
`VALIDATION_WINDOW_REGIME`.

| Budget | `LARGE_WINDOW` | `TIGHT_WINDOW` |
|---|---:|---:|
| `preview_tokens` | 800 | 2,000 |
| graph `description_tokens` | 100 | 250 |
| graph `max_retrieval_cycles` | 8 | 4 |

They are held as **one set** rather than three knobs because that is how they were measured: the comparison
that established them reverted all three at once, so the aggregate is attributable and the individual
contributions are not.

**The ceiling is 300K, not the 250K the class is named after, and that is a measurement.** Qwen3 Next's
window is 256,000 — above 250K — and it sits firmly in the tight regime: 94 refused calls on the bare
agent, and a graph-alone arm that peaked *above* the window. The real determinant is the window against the
payload mass in front of it, for which the window is only a proxy, and the errors are asymmetric — tight
budgets on a large window cost tokens, large budgets on a tight window cost answers.

**`preview_tokens` is the ceiling on everything downstream.** The graph derives a Card's numeric lines from
the message, and by then the message carries the preview, so `expand_card` cannot return a figure the
preview dropped. The two budgets are in series:

```
payload -> preview budget -> the message -> the Card's numeric lines -> Description budget
```

Raising the second while the first starves buys nothing, which is the measured reason a Description budget
of 100 was correct *for* a preview of 800 and wrong once the preview was raised.

### Parameter generations

The tables in part 4 span three parameter sets. Each model section names the one its run used. **Do not
compare a cell from one generation against a cell from another** — they are different code.

| Generation | Commit | What is different | Runs |
|---|---|---|---|
| **1** | `dd55284` | Filter ships its own `retrieve_context` (**on**). Graph split in two tunings: alone `0.62/0.45/0.50`, desc 250, `body_budget=60,000`; with-relevance `0.55/0.45/0.50`, desc **100**, `body_budget=None`. Preview 800 for every model. No retrieval ceiling. The graph's `expand_artifact` **dropped** whenever the filter is installed. Output cap hardcoded 4,096. | Opus 4.8 ×2, Opus 5 ×2, Fable 5 ×2, Astra, Sol, GLM 5 |
| **2** | `d951195` | Preview **2,000** for every model. Graph with-relevance desc **250**, `max_retrieval_cycles=4`, live vector index, batched `expand_card`, conditional guidance. Disclosure gains the `[+]` catalog sigil, the guessed-call guard and the `find_tools` exemption. A truncated answer is now scored for what it says instead of as silence. | GLM 4.7, Haiku 4.5, Qwen3 Next |
| **3** | `HEAD` | Filter's `retrieve_context` **removed by default**. One unified graph tuning for every arm. `body_budget=40,000`. `neighbors_per_candidate=3` — the `similar` edge finally has a reader. Window regime picks preview / desc / cycles. `include_artifact_tool=True` in every arm. | Opus 4.8, Haiku 4.5, GLM 5, GLM 4.7 Flash, Qwen3 Next, Nemotron Nano 9B |

Two of those changes were defects rather than tuning. The filter's `retrieve_context` was a safety net whose
every result became a conversation message re-sent on every later call — the measured mechanism behind
relevance filtering costing **+21.6%** on Haiku 4.5 in generation 1 instead of saving. And
`include_artifact_tool=not relevance_installed` existed to leave exactly one artifact-retrieval tool when
there were two; once the filter stopped registering one, the same line left the combined arm with **none**.

### What was tried against these values and rejected

GLM 4.7, 20 turns of which 18 scored, two to three replays per row, all-three arm, each row changing only
what its name says. Generation 2 parameters.

| Variant | Total tokens | Δ vs baseline | Correct | Verdict |
|---|---:|---:|:--:|---|
| Baseline, no plugin ✝ | 3,733,922 | — | 12.5/18 | overflowed 6× per replay |
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
row that spends less scores less, none dominates the reference, and the spread between replays of the *same*
row reached two materially-correct turns — so a variant that gains one turn has gained nothing. A ninth
attempt, raising the output cap to 8,192 and turning on a citable-density rerank prior, doubled cost for
+0.67 turns.

These rows are 20 turns, where the window is not yet binding on this model. A value tuned at 20 turns should
be confirmed at the length actually run.

---

## 3. Prices

Every dollar figure is `measured units × a published list rate`. Nothing is derived from a bill, and no
discount, commitment or account-specific fee is modelled — read the cost columns as **list cost**, and the
comparisons between them as ratios between rows priced identically. Rates live in `MODEL_PRICING` in
[`src/config.py`](validation/community-plugin-A-B-D/src/config.py), so correcting one never needs another
run: the report re-renders from the recorded JSON.

Cost includes what each strategy spends **on its own account** — the graph's embedding calls
(`cohere.embed-multilingual-v3`) and the filter's rerank calls (`cohere.rerank-v3-5:0`). Without that, a
strategy that buys its saving with a second model call would rank better than it is.

| Claim | Status | Source |
|---|---|---|
| Per-model input, output and cache rates | published | [Bedrock pricing](https://aws.amazon.com/bedrock/pricing/) and each model card |
| Cache writes are billed above the input rate, implicit **and** explicit | documented | [Billing for cached tokens](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html) |
| Cache write = **1.25×** input, read = 90% discount, on GPT-5.6 | documented | [Prompt caching for models from OpenAI](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html) |
| Astra's cache write $13.75 against $11.00 input (1.25×) | published | [Astra model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-6-astra.html) |
| Claude cache-read at 0.10× and write at 1.25× / 2.00× input | published for the models listed, **inferred** for Haiku 4.5, whose cache rates are not on the pricing table | `MODEL_PRICING` |
| Token counts, call counts, accuracy, timings | measured from the API's own `usage` | the run JSON under `results/` |

---

## 4. The models

| Model | Window | Caching | Generations measured | Bare agent finished? | Cheapest completing arm |
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

---

### 4.1 Claude Opus 4.8

| Field | Value |
|---|---|
| Model id | `us.anthropic.claude-opus-4-8` |
| Vendor | Anthropic |
| Context window | 1,000,000 |
| Caching | explicit; read $0.50, write $6.25 (5m) / $10.00 (1h) per Mtok |
| List price | $5.00 in / $25.00 out per Mtok |
| Runs | generation 1 caching off, generation 1 caching on, generation 3 caching off |

**Plugin parameters.**

| Plugin | Generation 1 | Generation 3 |
|---|---|---|
| A relevance | `chunk 500`, `preview 800`, `threshold 0.02`, retrieval tool **on** | `chunk 500`, `preview 800` (large regime), `threshold 0.02`, retrieval tool **off** |
| B disclosure | `catalog 20`, `ttl 5`, `top_k 4` | same, plus `[+]` sigil and guessed-call guard |
| D graph | alone `0.62/0.45/0.50`, desc 250, body 60,000 · with-relevance `0.55/0.45/0.50`, desc **100**, body **none** · artifact tool dropped in the all arm | unified `0.62/0.45/0.50`, desc 100 (large regime), **body 40,000**, cycles 8, neighbours 3, artifact tool on |

**Validation parameters.** 60 turns / 30 scored · 1 replay · `max_output 4,096` · rerank
`cohere.rerank-v3-5:0` · embed `cohere.embed-multilingual-v3` · region `us-east-1`.

**Results — generation 1, caching off.**

| Configuration | Total tokens | Δ tokens | Accuracy | Correct | Turn | Cost | Δ cost |
|---|---:|---:|---:|:--:|---:|---:|---:|
| Baseline (no plugin) | 14,377,382 | — | 96.9% | 28/30 | 11.7s | $72.50 | — |
| Progressive Tool Disclosure only | 9,466,084 | −34.2% | 94.5% | 27/30 | 12.8s | $48.10 | −33.7% |
| Relevance Filtering only | 12,766,237 | −11.2% | **96.9%** | **29/30** | 11.7s | $64.45 | −11.1% |
| Context Graph only | 10,385,714 | −27.8% | 93.7% | 27/30 | 10.8s | $52.43 | −27.7% |
| **All three combined** | **2,569,888** | **−82.1%** | 96.1% | 28/30 | **8.7s** | **$13.36** | −81.6% |

**Results — generation 1, caching on.**

| Configuration | Uncached | Cache read | Cache write | read:write | Billed tokens | Δ tokens | Accuracy | Correct | Turn | Cost | Δ cost |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|---:|---:|---:|
| 🏆 Baseline (no plugin) | 174 | 14,842,278 | 215,531 | **68.9** | 15,057,983 | — | 94.5% | 27/30 | **8.0s** | **$9.49** | — |
| Progressive Tool Disclosure only | 202 | 5,578,574 | 3,679,071 | 1.5 | 9,257,847 | −38.4% | 94.5% | 27/30 | 10.0s | $26.65 | +181.0% |
| Relevance Filtering only | 204 | 15,701,512 | 211,522 | **74.2** | 15,913,238 | +5.7% | 94.5% | 27/30 | 10.2s | $10.10 | +6.5% |
| Context Graph only | 576,260 | 8,205,206 | 1,521,353 | 5.4 | 10,302,819 | −31.4% | **99.2%** | **29/30** | 9.2s | $17.29 | +82.3% |
| **All three combined** | 328,419 | 1,067,190 | 2,253,984 | **0.5** | 3,649,593 | −75.5% | 94.5% | 27/30 | 10.4s | $17.14 | +80.7% |

**Results — generation 3, caching off.**

| Configuration | Total tokens | Δ tokens | Accuracy | Correct | Peak/call | Turn | Cost | Δ cost |
|---|---:|---:|---:|:--:|---:|---:|---:|---:|
| Baseline (no plugin) | 13,647,898 | — | 97.6% | 28/30 | 192,662 (19%) | 11.3s | $68.89 | — |
| Progressive Tool Disclosure only | 9,381,497 | −31.3% | **97.6%** | **29/30** | 139,961 (14%) | 12.6s | $47.74 | −30.7% |
| Relevance Filtering only | 13,447,761 | −1.5% | 89.8% | 27/30 | 189,714 (19%) | 12.2s | $67.96 | −1.4% |
| Context Graph only | 10,254,016 | −24.9% | 85.0% | 21/30 | 149,010 (15%) | 10.1s | $51.84 | −24.7% |
| **All three combined** | **3,348,835** | **−75.5%** | 91.3% | 26/30 | **59,405 (6%)** | 10.6s | **$17.47** | −74.6% |

**Observations.**

- **This is the reference model for the cost argument.** The window is not a constraint — the bare agent's
  largest call uses 19% of it — so nothing truncates and every column is comparable. The full stack sends
  **−82.1%** (gen 1) and **−75.5%** (gen 3) for the same 28 and 26 of 30, at a fifth of the cost.
- **The bare agent still peaks at 192,662–204,439 tokens on one call**, which is more than some models'
  entire window. That peak is what decides which models this benchmark can run on at all.
- **With caching on the ranking inverts and `read:write` says why.** The baseline re-reads a stable prefix
  at 68.9 reads per write and costs **$9.49**; the full stack rewrites its prefix every turn, lands at 0.5,
  and costs $17.14 — **1.8× more expensive than doing nothing**. Only relevance filtering keeps a
  cache-friendly ratio (74.2), because it compresses once and then stops changing the prompt.
- **Caching makes the agent cheaper, not better.** The model sees an identical prompt either way; accuracy
  moves by up to three turns in both directions with no mechanism, which is the single-replay noise floor.
- **The generation 3 run is where the two regressions of that generation are visible.** Relevance filtering
  drops to −1.5% and 27/30, and its two lost turns are `A5-statement` and `R2-cross-reference` — both asking
  for a redemption figure inside a statement payload. With the preview cutting the passage and
  `retrieve_context` removed, that content is unreachable. The combined arm fails the same two.
- **The graph-only arm's 21/30 is not attributable.** The same arm scored 27/30 on a byte-identical run —
  see [the noise floor](#the-noise-floor).

---

### 4.2 Claude Opus 5

| Field | Value |
|---|---|
| Model id | `us.anthropic.claude-opus-5` |
| Vendor | Anthropic |
| Context window | 1,000,000 |
| Caching | explicit; read $0.50, write $6.25 (5m) / $10.00 (1h) per Mtok |
| List price | $5.00 in / $25.00 out per Mtok |
| Runs | generation 1 caching off, generation 1 caching on |

**Plugin parameters.** Generation 1, as in [4.1](#41-claude-opus-48): filter retrieval tool on, preview 800,
graph split `0.62`/`0.55`, `body_budget` 60,000 alone and none with relevance, artifact tool dropped in the
all arm.

**Validation parameters.** 60 turns / 30 scored · 1 replay · `max_output 4,096` · region `us-east-1`.

**Results — caching off.**

| Configuration | Total tokens | Δ tokens | Accuracy | Correct | Turn | Cost | Δ cost |
|---|---:|---:|---:|:--:|---:|---:|---:|
| Baseline (no plugin) | 19,778,235 | — | 89.8% | 24/30 | 24.3s | $100.44 | — |
| Progressive Tool Disclosure only ⚠ | 647,368 | −96.7% | 31.5% | 1/30 | 1.8s | $3.25 | −96.8% |
| Relevance Filtering only | 18,106,860 | −8.5% | 96.9% | 28/30 | 21.4s | $91.89 | −8.5% |
| Context Graph only | 13,527,548 | −31.6% | 93.7% | 25/30 | 26.0s | $69.54 | −30.8% |
| **All three combined** | **5,565,805** | **−71.9%** | **100.0%** | **30/30** | 17.9s | **$29.17** | −71.0% |

**Results — caching on.**

| Configuration | Uncached | Cache read | Cache write | read:write | Billed tokens | Δ tokens | Accuracy | Correct | Turn | Cost | Δ cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|---:|---:|
| Baseline (no plugin) | 196 | 26,321,602 | 285,328 | **92.3** | 26,607,126 | — | 92.1% | 24/30 | 23.2s | $17.16 | — |
| Progressive Tool Disclosure only | 218 | 10,217,732 | 6,296,888 | 1.6 | 16,514,838 | −37.9% | 92.1% | 26/30 | 18.7s | $46.07 | +168.5% |
| 🏆 Relevance Filtering only | 200 | 17,343,538 | 237,313 | **73.1** | 17,581,051 | −33.9% | 92.9% | 25/30 | 19.7s | **$12.06** | −29.7% |
| Context Graph only | 664,269 | 9,508,166 | 2,898,117 | 3.3 | 13,070,552 | −50.7% | 94.5% | 27/30 | 22.0s | $28.34 | +65.2% |
| **All three combined** | 363,512 | 1,526,446 | 3,470,771 | **0.4** | 5,360,729 | −79.6% | **95.3%** | 27/30 | 18.3s | $25.98 | +51.4% |

**Observations.**

- **The full stack was the most accurate arm of the cache-off run — 30 of 30 against the bare agent's 24.**
  One replay, so read it as a path the agent can take rather than as a property of the stack; but it is the
  only arm in this document that scored perfectly.
- **⚠ The disclosure arm of the cache-off run is not a result.** The model never called `find_tools`, ran
  without a tool schema and answered almost nothing (1 of 30, 1.8s per turn). The same arm with caching on
  scored 26 of 30. This is path variance, and it is why a −96.7% token figure appears next to a useless run.
- With caching on, relevance filtering is the only arm cheaper than doing nothing, at `read:write` 73.1.
- Opus 5 used **35% more billed tokens with caching than without** on the baseline, purely because the
  agent's non-deterministic path made 7 more tool calls. Compare arms within a run, not across runs.

---

### 4.3 Claude Fable 5

| Field | Value |
|---|---|
| Model id | `us.anthropic.claude-fable-5` |
| Vendor | Anthropic |
| Context window | 1,000,000 |
| Caching | explicit; read $1.00, write $12.50 (5m) / $20.00 (1h) per Mtok |
| List price | $10.00 in / $50.00 out per Mtok — the most expensive model here |
| Runs | generation 1 caching off, generation 1 caching on |

**Plugin parameters.** Generation 1, as in [4.1](#41-claude-opus-48).

**Validation parameters.** 60 turns / 30 scored · 1 replay · `max_output 4,096` · region `us-east-1`.

**Results — caching off.**

| Configuration | Total tokens | Δ tokens | Accuracy | Correct | Turn | Cost | Δ cost |
|---|---:|---:|---:|:--:|---:|---:|---:|
| Baseline (no plugin) | 21,620,903 | — | **100.0%** | **30/30** | 20.5s | $217.86 | — |
| Progressive Tool Disclosure only | 14,284,626 | −33.9% | 97.6% | 29/30 | 20.5s | $144.58 | −33.6% |
| Relevance Filtering only | 17,263,244 | −20.2% | 97.6% | 29/30 | 18.1s | $174.08 | −20.1% |
| Context Graph only | 12,645,472 | −41.5% | 96.1% | 28/30 | 21.0s | $128.54 | −41.0% |
| **All three combined** | **5,311,719** | **−75.4%** | 96.9% | 28/30 | 22.1s | **$55.32** | −74.6% |

**Results — caching on.**

| Configuration | Uncached | Cache read | Cache write | read:write | Billed tokens | Δ tokens | Accuracy | Correct | Turn | Cost | Δ cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|---:|---:|
| Baseline (no plugin) | 190 | 18,254,571 | 250,858 | **72.8** | 18,505,619 | — | 99.2% | **30/30** | 16.1s | $23.38 | — |
| Progressive Tool Disclosure only | 248 | 8,001,876 | 6,708,508 | 1.2 | 14,710,632 | −20.5% | 98.4% | 29/30 | 19.3s | $93.96 | +301.8% |
| 🏆 Relevance Filtering only | 212 | 17,687,881 | 229,005 | **77.2** | 17,917,098 | −3.2% | 98.4% | 29/30 | **15.7s** | **$22.48** | −3.9% |
| Context Graph only | 812,339 | 10,368,477 | 2,528,084 | 4.1 | 13,708,900 | −25.7% | 98.4% | 29/30 | 21.7s | $53.19 | +127.5% |
| **All three combined** | 360,961 | 1,291,253 | 3,567,890 | **0.4** | 5,220,104 | −71.5% | 96.1% | 27/30 | 23.3s | $52.58 | +124.9% |

**Observations.**

- **The bare agent is perfect here and the most expensive row in the document: 30 of 30 for $217.86.** This
  is the model where the saving matters most in absolute dollars — the full stack holds 28 of 30 for
  **$55.32**, a $162 difference on one 60-turn conversation.
- The cache-on ranking inverts as on every Claude model: disclosure at `read:write` 1.2 costs **+301.8%**,
  and only relevance filtering (77.2) is cheaper than doing nothing.
- Two non-overflow errors occurred in the cache-off run (one in the relevance arm, one in all-three); both
  were transport failures, not window failures.

---

### 4.4 GPT-6 Astra

| Field | Value |
|---|---|
| Model id | `us.openai.gpt-6-astra` |
| Vendor | OpenAI |
| Context window | 1,050,000 |
| Caching | **implicit** — not requested by the harness, applied by the service; read $1.10, write $13.75 per Mtok (1.25× input) |
| List price | $11.00 in / $55.00 out per Mtok |
| Runs | generation 1, implicit caching (there is no cache-off counterpart to run) |

**Plugin parameters.** Generation 1, as in [4.1](#41-claude-opus-48).

**Validation parameters.** 60 turns / 30 scored · 1 replay · `max_output 4,096` · served from `us-west-2`
via cross-Region inference while called from `us-east-1`.

**Results.**

| Configuration | Uncached | Cache read | Cache write | read:write | Billed tokens | Δ tokens | Accuracy | Correct | Turn | Cost | Δ cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|---:|---:|
| Baseline (no plugin) | 228 | 15,270,589 | 213,127 | **71.7** | 15,483,944 | — | 92.9% | 26/30 | 10.4s | $21.20 | — |
| Progressive Tool Disclosure only | 252 | 7,810,974 | 5,534,968 | 1.4 | 13,346,194 | −13.8% | 92.9% | 26/30 | 10.3s | $86.18 | +306.5% |
| 🏆 Relevance Filtering only | 222 | 12,069,886 | 185,578 | **65.0** | 12,255,686 | −20.8% | 88.2% | 23/30 | 10.2s | **$17.33** | −18.3% |
| Context Graph only | 252 | 1,473,390 | 8,536,045 | **0.2** | 10,009,687 | −35.2% | 91.3% | 25/30 | 10.9s | $120.59 | +468.8% |
| **All three combined** | 292 | 17,588 | 4,051,747 | **0.0** | 4,069,627 | −73.5% | **94.5%** | **27/30** | **9.2s** | $57.38 | +170.7% |

**Observations.**

- **Implicit caching cannot be turned off, so this model has no uncached control.** Every row pays the write
  premium, and the plugins that mutate the prefix pay it on nearly everything they send: the graph reads
  1.47M against 8.54M written (`read:write` 0.2) and costs **+468.8%**.
- **The full stack sends 73.5% fewer tokens and costs 170.7% more.** That is the whole caching argument in
  one row: compression and caching attack the same redundancy, and on a model that caches implicitly the
  service has already collected it.
- The full stack is still the most accurate arm (27 of 30) and the fastest per turn. If accuracy is the
  objective and the bill is not, the ranking is unchanged; if the bill is the objective, use relevance
  filtering alone.

---

### 4.5 GPT-5.6 Sol

| Field | Value |
|---|---|
| Model id | `us.openai.gpt-5.6-sol` |
| Vendor | OpenAI |
| Context window | 1,000,000 |
| Caching | **implicit**; read $0.44, write $5.50 per Mtok (1.25× input) |
| List price | $4.40 in / $22.00 out per Mtok |
| Runs | generation 1, implicit caching |

**Plugin parameters.** Generation 1, as in [4.1](#41-claude-opus-48).

**Validation parameters.** 60 turns / 30 scored · 1 replay · `max_output 4,096` · served from `us-east-2`
via cross-Region inference while called from `us-east-1`.

**Results.**

| Configuration | Uncached | Cache read | Cache write | read:write | Billed tokens | Δ tokens | Accuracy | Correct | Turn | Cost | Δ cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|---:|---:|
| 🏆 Baseline (no plugin) | 242 | 15,679,917 | 218,227 | **71.9** | 15,898,386 | — | **97.6%** | **29/30** | **9.0s** | **$8.82** | — |
| Progressive Tool Disclosure only | 340 | 11,120,494 | 12,432,437 | 0.9 | 23,553,271 | +48.0% | 96.9% | 28/30 | 10.8s | $73.82 | +737.2% |
| Relevance Filtering only | 276 | 17,145,788 | 208,099 | **82.4** | 17,354,163 | +9.2% | **97.6%** | **29/30** | 10.3s | $9.57 | +8.5% |
| Context Graph only | 278 | 790,589 | 10,825,077 | **0.1** | 11,615,944 | −26.8% | 91.3% | 25/30 | 9.9s | $60.56 | +586.8% |
| **All three combined** | 344 | 17,861 | 6,811,669 | **0.0** | 6,829,874 | −56.8% | 96.1% | 28/30 | 10.0s | $38.55 | +337.3% |

**Observations.**

- **This is the one model where doing nothing wins on every axis at once**: cheapest, most accurate, fastest.
  A 1M window, implicit caching the service applies for free, and a conversation whose prefix is reused —
  there is nothing left for a compressing plugin to buy.
- **The disclosure arm sent more tokens than the bare agent (+48.0%) and wrote 12.4M of them.** Editing the
  tool section invalidates the prefix, so each reveal turn reprocesses everything after it at the write rate.
- Relevance filtering costs +8.5%, which is the smallest penalty of the four and the one to consider if a
  reason other than cost demands compression here.

---

### 4.6 Claude Haiku 4.5

| Field | Value |
|---|---|
| Model id | `us.anthropic.claude-haiku-4-5-20251001-v1:0` |
| Vendor | Anthropic |
| Context window | 200,000 |
| Caching | explicit; read $0.10, write $1.25 (5m) / $2.00 (1h) per Mtok — **inferred**, not on the pricing table |
| List price | $1.00 in / $5.00 out per Mtok |
| Runs | generation 2 caching off, generation 3 caching off |

**Plugin parameters.**

| Plugin | Generation 2 | Generation 3 |
|---|---|---|
| A relevance | `chunk 500`, `preview 2,000`, `threshold 0.02`, retrieval tool **on** | same budgets (tight regime), retrieval tool **off** |
| B disclosure | `catalog 20`, `ttl 5`, `top_k 4`, `[+]` sigil, guessed-call guard | same |
| D graph | alone `0.62/0.45/0.50`, desc 250, body 60,000 · with-relevance `0.55/0.45/0.50`, desc 250, body **none**, cycles 4 · artifact tool dropped in the all arm | unified `0.62/0.45/0.50`, desc 250, **body 40,000**, cycles 4, neighbours 3, artifact tool on |

**Validation parameters.** 60 turns / 30 scored · 1 replay · `max_output 4,096` · caching off · regime
**tight** (gen 3 records it; gen 2 predates the field).

**Results — generation 2.**

| Configuration | Answered | Scored lost | Refused | Total tokens | Δ vs heaviest | Accuracy | Correct | Peak/call | Turn | Cost |
|---|:--:|:--:|---:|---:|---:|---:|:--:|---:|---:|---:|
| Baseline (no plugin) ✝ | **24/60** | **12/30** | **72** | 6,058,882 | −54.1% | 65.3% | 13/30 | 193,683 (97%) | 3.9s ✝ | $6.09 |
| Progressive Tool Disclosure only | 60/60 | 0/30 | 0 | 12,156,014 | −8.0% | 88.2% | 22/30 | 167,060 (84%) | 8.6s | $12.26 |
| Relevance Filtering only ← ref | 60/60 | 0/30 | 0 | 13,214,041 | — | **93.7%** | **26/30** | 177,119 (89%) | 8.6s | $13.32 |
| Context Graph only | 60/60 | 0/30 | 0 | 7,894,451 | −40.3% | 79.5% | 18/30 | 118,442 (59%) | 7.0s | $7.97 |
| **All three combined** | 60/60 | 0/30 | 0 | **3,260,491** | **−75.3%** | 89.8% | 25/30 | **56,545 (28%)** | 5.8s | **$3.34** |

**Results — generation 3.**

| Configuration | Total tokens | Δ tokens | Accuracy | Correct | Peak/call | Turn | Cost | Δ cost |
|---|---:|---:|---:|:--:|---:|---:|---:|---:|
| Baseline (no plugin) | 12,703,571 | — | **95.3%** | **27/30** | 196,722 (**98%**) | 7.9s | $12.77 | — |
| Progressive Tool Disclosure only | 10,296,950 | −18.9% | 88.2% | 22/30 | 160,920 (80%) | 7.9s | $10.37 | −18.8% |
| Relevance Filtering only | 12,036,064 | −5.3% | 86.6% | 21/30 | 166,779 (83%) | 9.3s | $12.15 | −4.9% |
| Context Graph only | 7,883,137 | −37.9% | 76.4% | 17/30 | 114,003 (57%) | 7.8s | $7.95 | −37.7% |
| **All three combined** | **3,224,286** | **−74.6%** | 84.2% | 20/30 | **60,393 (30%)** | 6.6s | **$3.32** | −74.0% |

**Observations.**

- **The same model, the same window, two runs, and the bare agent truncated in one and not the other.** In
  generation 2 it lost 72 calls and answered 24 of 60 turns; in generation 3 it answered all sixty with its
  largest call at **98% of the window**. Nothing in the plugins explains that — the baseline runs none of
  them. It is the agent's tool path landing just inside or just outside the window, and it is the sharpest
  illustration of why `Answered` has to be a column.
- **This model is where relevance filtering was first measured costing more than nothing (+21.6%)**, which
  is what motivated removing its retrieval tool in generation 3. It worked: the same arm moves to −5.3%.
- **The full stack is the cheapest arm in both generations and within a cent of itself ($3.34, $3.32)**,
  while the baseline it is compared against moved by a factor of two. The stack's cost is stable because it
  does not depend on how much history there is.
- **Accuracy across the two runs is not comparable.** Generation 2's numbers come from a run where three
  arms scored over 60 answered turns and the baseline over 24.

---

### 4.7 GLM 5

| Field | Value |
|---|---|
| Model id | `zai.glm-5` |
| Vendor | Zhipu |
| Context window | 200,000 |
| Caching | **none published** — the model card lists neither implicit nor explicit caching, so there is nothing to weigh the plugins against |
| List price | $1.00 in / $3.20 out per Mtok |
| Runs | generation 1, generation 3 |

**Plugin parameters.**

| Plugin | Generation 1 | Generation 3 |
|---|---|---|
| A relevance | `preview 800`, retrieval tool **on** | `preview 2,000` (tight regime), retrieval tool **off** |
| B disclosure | `catalog 20`, `ttl 5`, `top_k 4` | same, plus `[+]` sigil and guessed-call guard |
| D graph | alone desc 250 / body 60,000 · with-relevance `0.55`, desc 100, body **none** · artifact tool dropped in the all arm | unified `0.62`, desc 250, **body 40,000**, cycles 4, neighbours 3, artifact tool on |

**Validation parameters.** 60 turns / 30 scored · 1 replay · `max_output 4,096` · no caching · regime
**tight**.

**Results — generation 1.**

| Configuration | Answered | Scored lost | Refused | Total tokens | Δ vs heaviest | Accuracy | Correct | Peak/call | Turn | Cost |
|---|:--:|:--:|---:|---:|---:|---:|:--:|---:|---:|---:|
| Baseline (no plugin) ✝ | **15/60** | **15/30** | **90** | 5,252,666 | −72.9% | 63.8% | 14/30 | 198,588 (99%) | 12.7s ✝ | $5.26 |
| Progressive Tool Disclosure only ✝ | **41/60** | **7/30** | **38** | 11,327,573 | −41.6% | 74.8% | 20/30 | 198,249 (99%) | 30.1s ✝ | $11.37 |
| Relevance Filtering only ← ref | 60/60 | 0/30 | 0 | 19,406,514 | — | **98.4%** | **29/30** | 192,751 (96%) | 48.8s | $19.48 |
| Context Graph only | 60/60 | 0/30 | 0 | 8,931,685 | −54.0% | 85.0% | 23/30 | 117,147 (59%) | 28.1s | $8.99 |
| **All three combined** | 60/60 | 0/30 | 0 | **2,457,277** | **−87.3%** | 92.1% | 26/30 | **43,490 (22%)** | 12.0s | **$2.51** |

**Results — generation 3.**

| Configuration | Answered | Scored lost | Refused | Total tokens | Δ vs heaviest | Accuracy | Correct | Peak/call | Turn | Cost |
|---|:--:|:--:|---:|---:|---:|---:|:--:|---:|---:|---:|
| Baseline (no plugin) ✝ | **18/60** | **14/30** | **84** | 6,150,161 | −58.3% | 65.3% | 14/30 | 193,207 (97%) | 12.6s ✝ | $6.17 |
| Progressive Tool Disclosure only ✝ | **25/60** | **11/30** | **64** | 9,170,600 | −37.8% | 66.1% | 16/30 | 193,113 (97%) | 34.0s ✝ | $9.22 |
| Relevance Filtering only ✝ ← ref | 55/60 | 3/30 | 10 | 14,739,065 | — | 80.3% | 24/30 | 198,011 (99%) | 35.5s | $14.80 |
| Context Graph only | 60/60 | 0/30 | 0 | 7,175,940 | −51.3% | **96.1%** | **28/30** | 102,645 (51%) | 20.5s | $7.21 |
| **All three combined** | 60/60 | 0/30 | 0 | **2,908,717** | **−80.3%** | 81.1% | 21/30 | **41,553 (21%)** | 14.0s | **$2.96** |

**Observations.**

- **This is the model that reverses the question from cost to completion, and it reproduces.** In both
  generations the bare agent lost 84–90 calls and answered 15–18 of 60 turns. Its 63.8% and 65.3% are
  mostly absence: 14–15 of its 30 scored turns were never attempted.
- **Only the graph and the full stack finished intact in both runs**, and they are the only two arms whose
  peak call stays below 60% of the window. Every arm at 96% or above lost calls.
- **Cost stops being the argument.** The full stack costs $2.51–$2.96 and answers everything; the bare agent
  costs $5.26–$6.17 and answers a quarter of the script. The cheap-looking baseline is a truncated log.
- **The graph alone is the most accurate arm of generation 3 (28/30)** and the second cheapest. On this model
  the single-plugin choice that keeps you inside the window is the graph — which is *not* true of the next
  two models, and that is the point of measuring more than one vendor.
- No caching is published for this model, so the cache-versus-compression trade does not arise here at all.

---

### 4.8 GLM 4.7

| Field | Value |
|---|---|
| Model id | `zai.glm-4.7` |
| Vendor | Zhipu |
| Context window | 202,752 |
| Caching | none published |
| List price | $0.60 in / $2.20 out per Mtok |
| Runs | generation 2 |

**Plugin parameters.** Generation 2: `preview 2,000`, filter retrieval tool **on**, disclosure `20/5/4`
with the `[+]` sigil and guessed-call guard, graph alone `0.62/0.45/0.50` desc 250 body 60,000 ·
with-relevance `0.55/0.45/0.50` desc 250 body **none** cycles 4, artifact tool dropped in the all arm.

**Validation parameters.** 60 turns / 30 scored · 1 replay · `max_output 4,096` explicit · no caching.

**Results.**

| Configuration | Answered | Scored lost | Refused | Total tokens | Δ vs heaviest | Accuracy | Correct | Peak/call | Turn | Cost | $/correct |
|---|:--:|:--:|---:|---:|---:|---:|:--:|---:|---:|---:|---:|
| Baseline (no plugin) ✝ | **22/60** | **12/30** | **70** | 5,814,709 | −58.9% | 65.3% | 14/30 | 186,905 (92%) | 18.4s ✝ | $3.52 | *$0.25* |
| Progressive Tool Disclosure only ✝ ← ref | **40/60** | **6/30** | **30** | 14,153,824 | — | 73.2% | 20/30 | 195,178 (96%) | 47.4s ✝ | $8.56 | *$0.43* |
| Relevance Filtering only ✝ | **27/60** | **11/30** | **54** | 11,140,007 | −21.3% | 66.1% | 17/30 | 194,955 (96%) | 36.4s ✝ | $6.75 | *$0.40* |
| Context Graph only | 60/60 | 0/30 | 0 | 13,191,178 | −6.8% | **95.3%** | **27/30** | 176,766 (87%) | 36.4s | $7.96 | $0.29 |
| **All three combined** | 59/60 † | 0/30 | 0 | **5,916,862** | **−58.2%** | 83.5% | 22/30 | **102,796 (51%)** | 23.6s | **$3.60** | **$0.16** |

† The one missing turn was a `ReadTimeoutError`, a network failure, not the window; that arm refused zero
calls. Fifteen other non-overflow errors occurred across the arms (3 baseline, 7 relevance, 5 disclosure,
2 all-three) — output-cap truncations, now scored for what the partial answer says.

**Observations.**

- **Three of five arms truncated, and the truncated set is not just the baseline.** Relevance filtering lost
  54 calls and disclosure 30, which is why the accuracy column is dangerous on this model: relevance looks
  second-from-last here (17/30) and is the most accurate arm on both other vendors of its study. That figure
  is its 54 overflows, not its recall.
- **The full stack spends 1.8% *more* than the bare agent — and that is the result, not a regression.** The
  baseline answered 22 turns and the full stack answered 59. Nearly tripling the work delivered for 1.8%
  more tokens is what happened; the sign alone gets it backwards. It is also why every truncated `$/correct`
  is italicised rather than competing: the cheapest ratio in the table belongs to an arm that quit.
- **The graph alone is the cleanest arm on this model — 27 of 30, zero refusals, the highest accuracy in its
  study** — while the same plugin on Qwen3 Next peaks above the window and answers 16 of 60. Folding history
  compresses the part that was not the problem when the mass is in the payloads.
- Measured on this model with 2–3 replays at 20 turns, the leave-one-out arms say it from the other side:
  all three combined is −54.8% tokens, but relevance+graph is **+6.1%** and relevance+disclosure **+13.1%**
  — either *pair* spends more than using no plugin at all. **The saving is a conjunction, not a sum.**

---

### 4.9 GLM 4.7 Flash

| Field | Value |
|---|---|
| Model id | `zai.glm-4.7-flash` |
| Vendor | Zhipu |
| Context window | 202,752 |
| Caching | none published |
| List price | $0.07 in / $0.40 out per Mtok — the cheapest model here |
| Runs | generation 3 |

**Plugin parameters.** Generation 3, tight regime: `preview 2,000`, filter retrieval tool **off**,
disclosure `20/5/4`, graph unified `0.62/0.45/0.50` desc 250 **body 40,000** cycles 4 neighbours 3,
artifact tool on.

**Validation parameters.** 60 turns / 30 scored · 1 replay · `max_output 4,096` · no caching · regime
**tight**.

**Results.**

| Configuration | Answered | Scored lost | Refused | Total tokens | Δ vs heaviest | Accuracy | Correct | Peak/call | Turn | Cost |
|---|:--:|:--:|---:|---:|---:|---:|:--:|---:|---:|---:|
| Baseline (no plugin) ✝ | **17/60** | **15/30** | **82** | 5,398,019 | −80.3% | 52.8% | 8/30 | 198,536 (98%) | 7.3s ✝ | $0.38 |
| Progressive Tool Disclosure only ✝ ← ref | **36/60** | **7/30** | **40** | 27,403,577 | — | 58.3% | 11/30 | 198,654 (98%) | 33.0s ✝ | $1.93 |
| Relevance Filtering only | 60/60 | 0/30 | 0 | 17,181,923 | −37.3% | **75.6%** | 17/30 | 187,849 (93%) | 13.4s | $1.22 |
| Context Graph only ✝ | **52/60** | **2/30** | **8** | 18,123,481 | −33.9% | 74.8% | **18/30** | 198,646 (98%) | 29.7s ✝ | $1.28 |
| **All three combined** | 60/60 | 0/30 | 0 | **4,010,248** | **−85.4%** | 55.1% | 10/30 | **53,355 (26%)** | 11.6s | **$0.30** |

Ten non-overflow errors (2 baseline, 4 disclosure, 4 graph) — output-cap truncations.

**Observations.**

- **The only model whose reference arm is itself truncated.** Disclosure spent 27.4M tokens over 237 calls
  and still answered 36 of 60 turns, which is what an erratic tool path costs when every retry carries the
  whole history. Its Δ column is therefore measured against a run that did not finish either; treat the
  magnitudes as a floor.
- **Relevance filtering is the arm to use on this model.** It is the only single plugin that completed the
  script and it scores highest among completing arms (17/30).
- **The full stack completes, is by far the cheapest ($0.30), and scores 10/30 — its worst result anywhere.**
  This is a genuine weakness rather than truncation: the arm answered all sixty turns. Two serial cuts
  (`preview_tokens` then `description_tokens`) plus a reranker query that does not contain the sought
  literal is an evidence-*selection* loss, and this model is where it bites hardest.
- **Do not read the price column as a recommendation.** Every arm here costs under $2 because the model is
  15× cheaper per token than Haiku; the interesting column is `Answered`.

---

### 4.10 Qwen3 Next 80B

| Field | Value |
|---|---|
| Model id | `qwen.qwen3-next-80b-a3b` |
| Vendor | Alibaba |
| Context window | 256,000 |
| Caching | none published |
| List price | $0.14 in / $1.20 out per Mtok |
| Runs | generation 2, generation 3 |

**Plugin parameters.**

| Plugin | Generation 2 | Generation 3 |
|---|---|---|
| A relevance | `preview 2,000`, retrieval tool **on** | `preview 2,000` (tight regime), retrieval tool **off** |
| B disclosure | `20/5/4`, `[+]` sigil, guessed-call guard | same |
| D graph | alone desc 250 body 60,000 · with-relevance `0.55` desc 250 body **none** cycles 4 · artifact tool dropped in the all arm | unified `0.62` desc 250 **body 40,000** cycles 4 neighbours 3, artifact tool on |

**Validation parameters.** 60 turns / 30 scored · 1 replay · `max_output 4,096` · no caching · regime
**tight** (256,000 is above 250K and still tight — see [the window regime](#the-window-regime)).

**Results — generation 2.**

| Configuration | Answered | Scored lost | Refused | Total tokens | Δ vs heaviest | Accuracy | Correct | Peak/call | Turn | Cost |
|---|:--:|:--:|---:|---:|---:|---:|:--:|---:|---:|---:|
| Baseline (no plugin) ✝ | **14/60** | **16/30** | **92** | 8,231,351 | −75.1% | 59.1% | 11/30 | 257,163 (100%) | 6.0s ✝ | $1.16 |
| Progressive Tool Disclosure only ← ref | 60/60 | 0/30 | 0 | 33,004,333 | — | 86.6% | 23/30 | 248,224 (97%) | 19.6s | $4.64 |
| Relevance Filtering only | 60/60 | 0/30 | 0 | 21,803,250 | −33.9% | **91.3%** | **26/30** | 241,712 (94%) | 16.5s | $3.10 |
| Context Graph only ✝ | **16/60** | **15/30** | **88** | 25,119,509 | −23.9% | 55.9% | 10/30 | 257,560 (**101%**) | 13.9s ✝ | $3.53 |
| **All three combined** | 59/60 | 0/30 | 0 | **5,512,454** | **−83.3%** | 81.9% | 23/30 | **92,972 (36%)** | 11.9s | **$0.82** |

**Results — generation 3.**

| Configuration | Answered | Scored lost | Refused | Total tokens | Δ vs heaviest | Accuracy | Correct | Peak/call | Turn | Cost |
|---|:--:|:--:|---:|---:|---:|---:|:--:|---:|---:|---:|
| Baseline (no plugin) ✝ | **13/60** | **17/30** | **94** | 8,591,570 | −74.1% | 54.3% | 9/30 | 248,552 (97%) | 5.6s ✝ | $1.21 |
| Progressive Tool Disclosure only ✝ | **13/60** | **17/30** | **94** | 5,033,909 | −84.8% | 60.6% | 11/30 | 252,395 (99%) | 4.0s ✝ | $0.71 |
| Relevance Filtering only ✝ ← ref | 55/60 | 3/30 | 10 | 33,149,405 | — | 79.5% | **22/30** | 256,310 (100%) | 19.9s | $4.71 |
| Context Graph only | 60/60 | 0/30 | 0 | 9,727,355 | −70.7% | **81.1%** | 20/30 | 136,350 (53%) | 8.5s | $1.38 |
| **All three combined** | 60/60 | 0/30 | 0 | **4,395,046** | **−86.7%** | 74.0% | 17/30 | **38,496 (15%)** | 10.2s | **$0.68** |

**Observations.**

- **This model carries the heaviest payload mass of the battery**, and it is why the tight-window ceiling is
  300K rather than 250K: a 256,000-token window and the bare agent still peaks *at* it and loses 92–94
  calls. Window size alone does not decide the regime; window against payload mass does.
- **Which single plugin survives changes between the two generations, and neither answer is the graph's.** In
  generation 2 the graph alone peaked at **101% of the window** and answered 16 of 60; in generation 3 it
  answered all sixty at 53%. Disclosure went the other way — 60 of 60 in generation 2, 13 of 60 in
  generation 3. Only the full stack answered nearly everything in both.
- **The generation 3 disclosure row is the trap this table exists to prevent**: 5.0M tokens and $0.71 look
  like the efficient arm until `Answered` reads 13 of 60 and `Refused` reads 94. It is the second-cheapest
  cell of the run and the second-worst outcome.
- **The full stack's peak call is 15% of the window**, an 85% margin on a model whose bare agent could not
  fit its own consolidation turns.

---

### 4.11 Nemotron Nano 9B

| Field | Value |
|---|---|
| Model id | `nvidia.nemotron-nano-9b-v2` |
| Vendor | NVIDIA |
| Context window | **128,000** — the smallest here |
| Caching | none published |
| List price | $0.06 in / $0.23 out per Mtok |
| Runs | generation 3 |

**Plugin parameters.** Generation 3, tight regime: `preview 2,000`, filter retrieval tool **off**,
disclosure `20/5/4`, graph unified `0.62/0.45/0.50` desc 250 **body 40,000** cycles 4 neighbours 3,
artifact tool on.

**Validation parameters.** 60 turns / 30 scored · 1 replay · `max_output 4,096` · no caching · regime
**tight**.

**Results.**

| Configuration | Answered | Scored lost | Refused | Total tokens | Δ vs heaviest | Accuracy | Correct | Peak/call | Turn | Cost |
|---|:--:|:--:|---:|---:|---:|---:|:--:|---:|---:|---:|
| Baseline (no plugin) ✝ | **9/60** | **21/30** | **102** | 1,596,017 | −85.3% | 48.8% | 6/30 | 111,016 (87%) | 4.2s ✝ | $0.10 |
| Progressive Tool Disclosure only ✝ | **14/60** | **16/30** | **92** | 2,298,800 | −78.8% | 59.1% | 9/30 | 118,522 (93%) | 7.8s ✝ | $0.14 |
| Relevance Filtering only ✝ | **28/60** | **11/30** | **64** | 5,073,784 | −53.3% | 66.9% | **14/30** | 125,719 (98%) | 12.4s ✝ | $0.33 |
| Context Graph only ✝ ← ref | **43/60** | **7/30** | **34** | 10,868,671 | — | 59.8% | 11/30 | 124,762 (97%) | 22.8s ✝ | $0.67 |
| **All three combined** | **60/60** | **0/30** | **0** | 6,431,158 | −40.8% | **68.5%** | 12/30 | **73,358 (57%)** | 28.6s | $0.44 |

**Observations.**

- **Four of the five arms could not finish, and the only column that orders cleanly is `Refused` — 102, 92,
  64, 34, 0.** Each practice removes some of the pressure; only the conjunction removes all of it.
- **The bare agent answered nine turns.** Its $0.10 is not a price for this workload — it is what nine turns
  cost. This is the row that makes the case for the completion columns better than any argument.
- **The full stack is the only arm that answered all sixty**, and against the heaviest arm that *tried* it
  still sends 40.8% fewer tokens. At this window the practices are not an optimisation but the difference
  between a conversation and a truncated log.
- **It is also the slowest arm per turn (28.6s), and that is the honest trade**: retrieval cycles cost
  latency, and they are what keep the conversation inside a 128K window.
- Accuracy here is low across the board (6–14 of 30) because the model is small. Compare the arms with each
  other, never this column with a larger model's.

---

## 5. What holds across the models

### The noise floor

**One replay per cell cannot resolve anything smaller than ±6 turns or ±20% of tokens**, and that is
measured rather than asserted. Between two Opus 4.8 runs, four arms were executing **byte-identical code**
— the only change between the runs applied to the combined arm alone — and all eight of those runs refused
zero calls, so nothing is truncated either:

| Opus 4.8 arm, identical code | Correct, run A → run B | Δ tokens |
|---|:--:|---:|
| Baseline (no plugin) | 24/30 → 28/30 | −16.2% |
| Context Graph only | 27/30 → 21/30 | +1.8% |
| Relevance Filtering only | 25/30 → 27/30 | +1.7% |
| Progressive Tool Disclosure only | 27/30 → 29/30 | −5.4% |

Six materially-correct turns moved on the graph arm with no code change, and the plugin-free baseline moved
four turns and 16% of its tokens. On Qwen3 Next the same comparison moved the relevance arm **+79.2%** in
tokens for the same 22 of 30.

The cause is trap 3: the agent chooses its own tool path, and one extra call early in a 60-turn conversation
is re-sent in every later call. So the figures that decide something here are the ones clearing that band by
a wide margin — −75% tokens, 102 refused calls against 0, a peak call of 38K against 248K. **The per-arm
accuracy ranking within a single model is not one of them.** Resolving the finer comparisons needs
`--repeats`; at Opus list prices one 5-arm replay is ~$254, so it is a deliberate omission.

### Compression and caching optimise the same redundancy

On the caching runs, **amplification** — billed prompt tokens divided by the tokens written to cache exactly
once, i.e. how many times the same content was re-sent — is **66× to 83×** for the bare agent and the
relevance filter, and **1.0× to 1.6×** for all three combined. The full stack barely re-sends anything,
which is the point; but it also means the cache has nothing cheap left to re-read and collects only
expensive writes.

`read:write` is the predictor and it is legible in every cache table above: 68–92 for the baseline and for
relevance filtering, 0.0–1.6 for disclosure and for the full stack. **The two techniques are alternatives,
not a stack** — with relevance filtering the exception, because it compresses once and then stops changing
the prompt.

The design rule that follows: **a context plugin is cache-compatible if and only if its mutations are
append-only or confined to the end of the prompt.** Size is not the problem; editing is. Disclosure cut the
tool schema from 62,656 tokens to 5,300–11,606 and still cost +181% to +737% under caching.

Caching also only pays when the same prefix is re-sent inside the TTL. Turns in this benchmark land 8–24
seconds apart, so nearly every call hit a warm cache — a best case. A system prompt per tenant, a tool set
per user permission, one-shot fan-out, an A/B split across prompt variants, or a human who pauses longer
than the TTL all pay the write premium and collect no read. In those shapes the plugins are the only lever.

### The window is a proxy for payload mass, and below ~250K it decides completion

Ordered by window, the bare agent's outcome:

| Model | Window | Bare agent answered | Bare agent peak | Full stack answered | Full stack peak |
|---|---:|:--:|---:|:--:|---:|
| Opus 4.8 / Opus 5 / Fable 5 / Astra / Sol | 1M+ | 60/60 | 19% | 60/60 | 6% |
| Haiku 4.5 (gen 3) | 200K | 60/60 | 98% | 60/60 | 30% |
| Haiku 4.5 (gen 2) | 200K | **24/60** | 97% | 60/60 | 28% |
| GLM 5 | 200K | **15–18/60** | 99% | 60/60 | 21% |
| GLM 4.7 | 203K | **22/60** | 92% | 59/60 | 51% |
| GLM 4.7 Flash | 203K | **17/60** | 98% | 60/60 | 26% |
| Qwen3 Next | 256K | **13–14/60** | 100% | 59–60/60 | 15–36% |
| Nemotron Nano 9B | 128K | **9/60** | 87% | 60/60 | 57% |

Two readings, and the second is the one to take away. First: **at 1M the argument is cost, below ~250K it is
completion.** Second: Haiku 4.5 appears twice with the same window and opposite outcomes, so the window is
not the variable — it is the window against the payload mass in front of it, and the window is only what
you can read off a model card in advance.

### No single practice is sufficient, and which one to pick depends on the model

| Model | Single plugin that completed | Best single arm by `Correct` |
|---|---|---|
| GLM 5 | graph (both generations) | graph, 28/30 |
| GLM 4.7 | graph | graph, 27/30 |
| GLM 4.7 Flash | relevance | relevance, 17/30 |
| Qwen3 Next gen 2 | disclosure, relevance | relevance, 26/30 |
| Qwen3 Next gen 3 | graph | relevance, 22/30 (with 10 refusals) |
| Nemotron Nano 9B | **none** | relevance, 14/30 (with 64 refusals) |

Every arm using one plugin fails to control the peak on at least one model, and on the smallest window none
of them completes. The full stack answered **60 of 60 on every model it was run against**. That is the
conjunction argument: A attacks the payload, B removes the fixed schema floor, D folds the history, and the
three address different mass.

### The regression this document owes the reader

Removing the filter's `retrieve_context` in generation 3 fixed a real cost (relevance filtering went from
+21.6% to −5.3% on Haiku) and cost two specific turns on Opus 4.8 — `A5-statement` and
`R2-cross-reference`, both asking for a figure inside a statement payload the preview cut, with nothing left
to ask for it back. `include_retrieval_tool=True` is the way back, at the price the cache tables show.

---

## 6. Reproducing this

```bash
cd validation/community-plugin-A-B-D
./run.sh --total-turns 60 --tag myrun                        # cache off
./run.sh --total-turns 60 --cache default --tag myrun-cache  # Bedrock's own TTL, 5 minutes
```

One model at a time, which is how every table in part 4 was produced:

```bash
VALIDATION_AGENT_MODEL_ID=zai.glm-5 ./run.sh --total-turns 60 --tag tw-glm5
VALIDATION_AGENT_MODEL_ID=qwen.qwen3-next-80b-a3b ./run.sh --total-turns 60 --tag tw-qwen3
VALIDATION_AGENT_MODEL_ID=nvidia.nemotron-nano-9b-v2 ./run.sh --total-turns 60 --tag tw-nemo
```

Run **one process per model id**. Two concurrent runs against the *same* id draw on the same quota and
produce `ServiceUnavailableException`, which contaminates a run without looking like a plugin failure;
different ids have separate quota and are safe to run in parallel.

Re-rendering a recorded run needs no AWS credentials at all, which is also how a corrected rate is applied:

```bash
python -m src.run --report-only results/run-myrun.json
```

Every knob in part 2 reads an override off the environment, and each run records under
`meta.sweep_overrides` which ones it read — so a result can never be read without its configuration:

```bash
VALIDATION_PREVIEW_TOKENS=1200 ./run.sh --configs all --total-turns 20 --repeats 3 --tag sweep-preview
VALIDATION_GRAPH_EXPAND=0.65 ./run.sh --configs all --total-turns 20 --repeats 3 --tag sweep-fold
VALIDATION_RELEVANCE_RETRIEVAL_TOOL=1 ./run.sh --configs all relevance --total-turns 60 --tag gen1-filter
```

The leave-one-out arms are accepted by `--configs` but not in the default set, so adding them cannot change
what a published table means:

```bash
./run.sh --configs no-disclosure no-relevance no-graph --total-turns 20 --repeats 3 --tag loo
```

`--cache default` uses Bedrock's own 5-minute TTL. An explicit `--cache 5m` or `--cache 1h` is rejected by
the pinned botocore 1.40, whose Converse model declares `cachePoint` with `type` alone — so the 1-hour TTL
needs a newer botocore and was not measured.

See [`validation/community-plugin-A-B-D/README.md`](validation/community-plugin-A-B-D/README.md) for what
the harness measures and how scoring works.

---

## 7. What is not settled

**Replicas.** Every cell in part 4 is one replay, against a measured noise floor of ±6 turns. No conclusion
here should be read finer than that, and the finer comparisons are simply not resolved.

**Caching on the Converse path for the OpenAI models.** Both cards list Implicit and Explicit Prompt Caching
as *Responses API only*, yet these runs used Converse and came back with cache read and write counts anyway
— measured on Astra's first baseline call as `inputTokens=2, cacheWriteInputTokens=48,583`. The *rate* is
documented (1.25×); what is undocumented is that the traffic happens at all on this path, so the Astra and
Sol tables price a documented rate against undocumented behaviour.

**The 5-minute TTL is the best case.** Turns land 8–24 seconds apart here, so nearly every call hit a warm
cache. A 1-hour TTL exists for the opposite case — `"ttl": "1h"` on the `cachePoint`, supported by all three
Claude models here — at a higher write rate, and the pinned botocore cannot send it. Neither was measured.

**Cross-Region routing stayed put.** Every model was served by exactly one Region for the whole day (read
from `inferenceRegion` in the invocation logs), including Sol and Astra, served from `us-east-2` and
`us-west-2` while being called from `us-east-1` — so cache reads do work across a Geo CRIS hop. The
documentation warns that at times of high demand these optimisations may lead to increased cache writes;
that case did not occur here. Global CRIS was not tested.

**Generations are not comparable.** Part 2's generation table says which code each model's run used. A
disclosure or all-three cell from generation 1 against one from generation 3 is a comparison of two
different programs, and the differences between them are larger than most of the deltas in this document.

**These are the community packages.** Every mechanism here was measured on `validation/community-plugin-A-B-D/`
with the three packages installed from this repository. The forked-SDK vended plugins place their cache
checkpoints in their own code and were **not** re-measured; nothing on this page transfers to them without a
run.

**Haiku 4.5's cache rates are inferred.** They are not on the pricing table; the read/write multipliers used
are the family's. Every other rate is published.
