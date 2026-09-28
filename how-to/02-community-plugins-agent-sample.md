# Running a Strands agent with the community plugins

> **⚠️ Not for production use.** This is a minimal working sample for experimentation and learning.
> It has no guardrails, no error handling and no operational hardening.

A runnable example of the three practices — [relevance filtering](../docs/design/design-a-relevance-filtering.md),
[progressive tool disclosure](../docs/design/design-b-progressive-tool-disclosure.md) and the
[context graph](../docs/design/design-d-context-graph.md) — installed on a Strands agent as **three
standalone community packages**, one at a time and then together.

The difference from [`01-designA-B-D-agent-sample.md`](01-designA-B-D-agent-sample.md) is where the
plugins come from. That guide installs a **forked SDK** and uses the plugins it vends. This one
installs three ordinary packages next to an **unmodified `strands-agents`** from PyPI — no fork, no
pinned commit SHA.

This is the shape the benchmark under
[`validation/community-plugin-A-B-D/`](../validation/community-plugin-A-B-D/README.md) measures.
Start here for the wiring; go there for what it costs.

**Read [the four gotchas](#four-things-that-will-bite-you) before wiring all three together.** Two of
them cost a measured benchmark run its answers, and none of them fails loudly.

## No fork required

The three packages install next to an **unmodified** `strands-agents` from PyPI. Verified against
**1.56.0**: the private middleware seam they couple to
(`strands._middleware.stages.InvokeModelStage`, `strands.injection._message_injection`) is present on
the public release, which is what makes "community plugin" a real claim rather than a repackaging of
the fork. They are not published to PyPI yet, so they install from this repository.

Everything you need to run them is in [Run it in five commands](#run-it-in-five-commands) below.

## Run it in five commands

If you only want to see it work, this is the whole path. It creates an isolated environment, installs
the three packages, and runs the agent below against a real Bedrock model.

### Prerequisites

| | Requirement | Why |
|---|---|---|
| Python | **3.10 or newer** | the packages' `requires-python`; the benchmark harness wants 3.12 |
| `strands-agents` | **>= 1.44.0, < 2.0.0** | the middleware seam the plugins attach to; verified on 1.56.0 |
| `boto3` | >= 1.26 | pulled in automatically; used by the reranker and the embedding matcher |
| AWS CLI | **v2** | only to configure and verify credentials |

Bedrock models that must be **enabled in your account**, in the region you use:

| Model id | Used by |
|---|---|
| `us.anthropic.claude-opus-4-8` | the agent itself (any Converse-capable model works) |
| `cohere.rerank-v3-5:0` | relevance filtering |
| `cohere.embed-multilingual-v3` | the context graph's similarity matcher |

Enable them once under **Amazon Bedrock → Model access** in the console.

### 1. Configure AWS credentials with the AWS CLI

The plugins use `boto3`, which reads the standard credential chain — so whatever the AWS CLI is
configured with is what the agent will use. Nothing is hardcoded and no key is written by this sample.

```bash
# interactive: paste an access key pair, choose a region
aws configure

# or, if your organisation uses IAM Identity Center (SSO)
aws configure sso
aws sso login --profile my-profile
export AWS_PROFILE=my-profile
```

Then verify — both commands must succeed before the agent will run:

```bash
aws sts get-caller-identity                 # who you are: account, ARN
aws bedrock list-foundation-models \
  --region us-east-1 \
  --query "modelSummaries[?contains(modelId,'rerank') || contains(modelId,'embed-multilingual')].modelId" \
  --output table                            # the auxiliary models are reachable
```

Set the region once so you do not have to pass it every time:

```bash
export AWS_DEFAULT_REGION=us-east-1
```

The IAM identity needs `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream` on the three
model ids above, plus `bedrock:Rerank` on `bedrock-agent-runtime` for relevance filtering. Scope it to
those resources rather than using a wildcard.

### 2. Create the environment and install

```bash
git clone https://github.com/aws-samples/sample-context-engineering.git
cd sample-context-engineering

python3 -m venv .venv
source .venv/bin/activate                   # Windows: .venv\Scripts\activate

pip install -e community-plugins/strands-context-graph \
            -e community-plugins/strands-progressive-tool-disclosure \
            -e community-plugins/strands-relevance-filter \
            "strands-agents>=1.44.0,<2.0.0"
```

### 3. Save the agent and run it

Put any of the code blocks from the sections below into `agent.py` — start with
[the starting point](#the-starting-point) to see the problem, then
[all three together](#all-three-together) to see it solved — and run:

```bash
python agent.py
```

**This spends money on Bedrock:** every turn is a real model call, and the sample's tool returns a
~40k-character payload on purpose.

To see the measured numbers instead of the wiring, the benchmark is one command:

```bash
cd validation/community-plugin-A-B-D
./run.sh --total-turns 60 --tag myrun
```

That one costs considerably more — it is 60 turns across five configurations. See
[`validation/community-plugin-A-B-D/README.md`](../validation/community-plugin-A-B-D/README.md) first.

---

## The starting point

An agent with one oversized tool — enough to make the practices visible, because each one acts on a
payload the model would otherwise carry in full for the rest of the conversation.

```python
import boto3
from strands import Agent, tool
from strands.models import BedrockModel

MODEL_ID = "us.anthropic.claude-opus-4-8"
REGION = "us-east-1"
session = boto3.Session(region_name=REGION)

SYSTEM_PROMPT = "You are a financial assistant. Answer from the tools, never from memory."


@tool
def account_statement(account: str, days: int) -> str:
    """Return the transaction history for an account over a number of days."""
    # Stand-in for a real integration: a payload large enough to matter, ~40k characters.
    rows = [f"2026-01-{day % 28 + 1:02d},PURCHASE {day},-{day * 3.17:.2f}" for day in range(900)]
    return "date,description,amount\n" + "\n".join(rows)


def build_model() -> BedrockModel:
    # No temperature: Opus 4.8 rejects the parameter.
    return BedrockModel(boto_session=session, model_id=MODEL_ID, max_tokens=4096)


agent = Agent(model=build_model(), system_prompt=SYSTEM_PROMPT, tools=[account_statement])
print(agent("Which was the largest purchase in the last 90 days on account 0001/12345-6?"))
```

Each section below changes only how that `Agent` is constructed.

## A — Relevance filtering

`RelevanceFilter` attaches to the public `AfterToolCallEvent` hook. When a tool result exceeds
`max_result_tokens`, it writes the raw sub-blocks to a `Store`, then replaces the result with a
**verbatim** preview of the chunks that score highest against the question, headed by a disclaimer that
says how much of the result the excerpt covers, plus a reference the model can pass to the plugin's own
`retrieve_all_context` tool when a question needs the whole result (a maximum, a total, a count).
Those retrieval exchanges are removed from the history when the turn ends.

```python
from strands_relevance_filter import BedrockReranker, FileStore, RelevanceFilter

agent = Agent(
    model=build_model(),
    system_prompt=SYSTEM_PROMPT,
    tools=[account_statement],
    plugins=[
        RelevanceFilter(
            store=FileStore(".artifacts/relevance"),
            max_result_tokens=4_000,        # above this, the result is filtered
            config={
                # The package default is amazon.rerank-v1:0 -- name the model you mean.
                "reranker": BedrockReranker("cohere.rerank-v3-5:0", boto_session=session),
                "relevance_threshold": 0.02,  # see the gotchas: 0.5 rejects everything here
                "chunk_tokens": 500,          # scoring granularity
                "preview_tokens": 800,        # budget for what stays visible
            },
        )
    ],
)
```

Selection is verbatim — chosen chunks reach the model character for character — which is what keeps
monetary and tabular figures exact. Omissions are marked with `[... N lines omitted ...]`, and the
line numbers in those markers are the ones `retrieve_all_context` accepts as a `line_range`.

The tool is registered by `include_retrieval_tool`, which defaults to `True`; off, nothing is stored
and no reference is minted. Its budgets are `pattern` (only the matching lines, with `context_lines`
around each — the cheapest way to aggregate), `line_range`, `max_chunks` (the N most relevant chunks
in document order, from the ranking the excerpt already computed, so no second rerank is charged) and
`max_tokens`, which bounds the response in every mode. Precedence is `line_range`, then `pattern`,
then `max_chunks`.

Construction is inert: the reranker is built on the first filtered result, so a filter that never
fires needs no credentials.

Ranking costs one rerank call per oversized result. That is the trade: a second API call to spend the
same preview budget on the chunks that answer the question.

## B — Progressive tool disclosure

Projects the call's tool list down to what is callable on it: the two plugin tools `find_tools` and
`get_tool_details`, the `always_available` tools, and the schemas already loaded and still live by TTL.
Every other tool is **one line in the system prompt** — its name and a summary of its description, at
most `catalog_chars` characters. A full schema enters the call when the model loads it by name, and
leaves again after `ttl_cycles` idle cycles. Every tool stays callable throughout — only what the call
is *told about* changes.

```python
from strands_progressive_tool_disclosure import ProgressiveToolDisclosure

agent = Agent(
    model=build_model(),
    system_prompt=SYSTEM_PROMPT,
    tools=[account_statement],
    plugins=[
        ProgressiveToolDisclosure(
            catalog_chars=80,    # character limit of each catalog line; None drops the catalog
            summarizer=None,     # None = the agent's own model writes the lines, once per tool
            ttl_cycles=3,        # cycles a loaded schema survives after its last use
            top_k=4,             # tools listed per find_tools call
            # A retrieval tool must never need discovery: the model is told to call it in the text
            # that replaced the payload. Name the ones your setup installs -- but not the relevance
            # filter's retrieve_all_context, which is for whole-result questions and is loaded on demand.
            always_available=[],
        )
    ],
)
```

The flow is **catalog → `get_tool_details([names])` → call**. The model reads the names in the system
prompt, asks for the ones it wants — several in one call — and their full parameters arrive in its tool
list on the next call. `find_tools` is the fallback for a need no listed name fits: it searches and
reports names plus summaries, and loading is still `get_tool_details`' job, so there is one road to a
schema rather than two.

A catalog line is a **summary, not a cut**. A description that already fits `catalog_chars` is used
verbatim and costs nothing; a longer one is summarized once by a model, cached per `(name, description)`
for the life of the plugin, and falls back to a sentence- or word-boundary truncation if the summarizer
fails. Pass your own `summarizer` to avoid the calls entirely. What they cost is reported as
`summary_usage` on the plugin's per-agent state, next to what the catalog saves.

The saving scales with catalog size: with a handful of tools there is little schema to avoid. It pays
off at the tool counts a real agent reaches — the benchmark runs 93 tools for ~63k tokens of schema
per call.

A released schema leaves nothing behind that invites a call. In the messages each call sends, the
exchanges of tools that call does not carry are folded to one sentence — `The tool X was called and the
result was: ...` — and the plugin's own `find_tools` / `get_tool_details` exchanges are dropped. A
`toolUse` with its arguments next to a success is a template the model copies, and a copy of a tool no
longer in the list is a call it cannot make. Only closed turns are folded: the turn in flight passes
through untouched, because a reasoning model rejects a modified latest assistant message.
`agent.messages` is never mutated — the fold is on the per-call copy.

A catalog name is not in `tool_specs` at all, so nothing in the call claims it is callable with no
arguments. If a model calls one anyway, a pre-call guard cancels that call and points it at
`get_tool_details`; nothing is loaded on its behalf, because a recovery that loaded the tool would teach
the model that calling a catalog name directly works. A tool whose parameters are all optional is
exempt — it is callable empty, so the call is not a guess. The measurements that made that path common
(5 `find_tools` searches against 14 premature cancellations over 60 turns) were taken on the **previous
design**, where every undisclosed tool sat in `tool_specs` as a reduced entry with an empty
`inputSchema` that read as "takes no arguments". Those figures do not describe this one, and the
benchmark's `searches` / `loads` / `premature_cancellations` counters are what to read instead.

## D — Context graph

Turns short-term memory into a graph of **Cards** — one per closed turn, derived by deterministic scan
with no model call. Each Card enters a call at one of three resolutions: **Title**, **Description**
(rule-derived) or **Full Content**. A Note is computed per Card each turn from the similarity between
the question and the Card's Description, then propagated one hop along the Links; the Note picks the
resolution. Descending is for budget, never a verdict — a collapsed Card is recoverable through
`expand_card`.

```python
from strands.agent.conversation_manager import NullConversationManager
from strands_context_graph import ContextGraph, EmbeddingSimilarityMatcher

agent = Agent(
    model=build_model(),
    system_prompt=SYSTEM_PROMPT,
    tools=[account_statement],
    # A PRECONDITION, not a suggestion. See the gotchas.
    conversation_manager=NullConversationManager(),
    plugins=[
        ContextGraph(
            expand_threshold=0.55,   # Note at or above which a Card is Full Content
            collapse_floor=0.45,     # Note below which a Card keeps only its Title
            link_threshold=0.50,     # similarity at or above which two Cards link
            description_tokens=100,  # token ceiling of a Description
            min_cards=3,             # below this, the choice is skipped entirely
            matcher=EmbeddingSimilarityMatcher("cohere.embed-multilingual-v3", boto_session=session),
        )
    ],
)
```

The matcher's embedding round is the graph's **only** remote call, at most one per turn, cached by
`(purpose, text)` so an unchanged Description costs nothing next turn.

The three thresholds are calibrated against that matcher's score distribution and are **not
portable**: supply a different embedding model and they mean nothing. `expand_threshold=0.0,
collapse_floor=0.0` is the regression switch — it projects every Card at Full Content, producing a
call identical field for field to one made without the plugin.

Unlike the vended plugin, this one is **ephemeral**: it writes nothing to `agent.state`, so there is
no `persist` option and nothing to restore. A fresh process rebuilds the whole graph by one scan over
the history.

## All three together

```python
from strands.agent.conversation_manager import NullConversationManager
from strands_context_graph import ContextGraph, EmbeddingSimilarityMatcher
from strands_progressive_tool_disclosure import ProgressiveToolDisclosure
from strands_relevance_filter import BedrockReranker, FileStore, RelevanceFilter

graph = ContextGraph(
    matcher=EmbeddingSimilarityMatcher("cohere.embed-multilingual-v3", boto_session=session),
    # Kept ON. The filter's tool is scoped to whole-result questions and is named in the filter's own
    # disclaimer, so the two no longer present as the same job -- see gotcha 1 for the measurement
    # that once motivated dropping this one.
    include_artifact_tool=True,
)


agent = Agent(
    model=build_model(),
    system_prompt=SYSTEM_PROMPT,
    tools=[account_statement],
    conversation_manager=NullConversationManager(),
    plugins=[
        RelevanceFilter(
            store=FileStore(".artifacts/relevance"),
            max_result_tokens=4_000,
            config={
                "reranker": BedrockReranker("cohere.rerank-v3-5:0", boto_session=session),
                "relevance_threshold": 0.02,
                "chunk_tokens": 500,
                "preview_tokens": 800,
            },
        ),
        graph,
        ProgressiveToolDisclosure(
            catalog_chars=80,
            ttl_cycles=3,
            top_k=4,
            # Derived, never hard-coded. A tool that is only in the catalog is not in `tool_specs` at
            # all, so the model has to load it with `get_tool_details` before it can be called -- and
            # every retrieval tool here needs arguments (a Title, a reference, a search need). Making
            # them always available spends no cycle on loading what the folded-context guidance
            # already told the model to call. Reading the names off the plugin keeps this correct if
            # `include_artifact_tool` is ever turned off or a tool is renamed. The filter's
            # retrieve_all_context is left out on purpose: it is loaded from the catalog only for a
            # whole-result question.
            always_available=[*graph.retrieval_tool_names],
        ),
    ],
)
```

The three plugins compose without fighting: the filter acts on tool results as they arrive, the graph
rewrites the per-call message list, and disclosure rewrites the per-call tool list plus the exchanges
of the tools it left out. None of them mutates `agent.messages` or the tool registry.

## Four things that will bite you

### 1. Two retrieval tools, two stores, no bridge

`RelevanceFilter` hands out references that **only its own** `retrieve_all_context` resolves.
`ContextGraph` resolves **its own** references through `expand_artifact`. Nothing bridges the two
stores — the graph's README is explicit that its bridge to another plugin's stash is built entirely on
private symbols and degrades to "answers as prose naming the miss".

When both tools read as the same job, the model reaches for whichever looks right and gets a miss.
Measured, it said so in its own answer:

> "every export's artifact reference has come back **unreachable** … I can't read the stored
> artifacts."

That cost the benchmark two of eighteen scored turns, and dropping the graph's `expand_artifact` took
the full stack from **84.5% / 15-of-18 to 94.4% / 17-of-18**.

**Both tools are installed, and the disambiguation sits in the tool itself rather than in the
wiring.** Three things changed:

- The filter's tool is `retrieve_all_context`, scoped to the one question an excerpt cannot answer —
  one that needs every row. It is not an artifact reader.
- It is registered by default (`include_retrieval_tool=True`), and the excerpt's disclaimer names it
  by name, with the reference and the budgets to pass, so the model is told which call to make rather
  than left to pick.
- It is reached **through the disclosure catalog**, not through `always_available`: a whole-result
  question is rare, so its schema is loaded only when one comes up.

So the graph keeps `expand_artifact` (`include_artifact_tool=True`), alongside `expand_card` and
`find_context`, which reach into the conversation's own turns — a job the filter does not do. Whether
the model still confuses the two is something the benchmark's `all` arm measures; it is not assumed
here. If you see "unreachable reference" answers come back, that is the symptom, and turning
`include_artifact_tool` off is still the one-line way to leave a single retrieval path.

The vended stack never had the ambiguity: relevance lived *inside* the `ContextManager` whose stash
the graph bridged to, so there was one store and one retrieval path.

### 2. `NullConversationManager` is a precondition of the graph

Any other conversation manager edits the **live** message list before the call is assembled, so it can
physically drop what the graph only meant to fold — and raising that Card's resolution back up then
recovers nothing. The plugin emits one `warnings.warn` at wiring time and still registers everything;
that warning is the whole protection you get.

### 3. The relevance threshold is a position in a distribution, not a number

The package default is `0.5`. With `cohere.rerank-v3-5` that **rejects every chunk**: measured on a
live account, a strong match scores ~0.29 and an unrelated chunk ~0.03. The benchmark uses `0.02`
because that is where the distribution actually sits. Change the rerank model and this number has to
be re-derived, not carried over.

The same holds for the graph's three thresholds against its embedding model.

### 4. The graph should fold *less* when the relevance filter is present

The two act on the same content, one after the other. The filter replaces an oversized payload with an
800-token preview on `AfterToolCallEvent`; the graph derives its Card **after** that, so the Card's
numeric lines come from the preview rather than from the raw result.

That inverts the tuning. Values measured as an improvement for the graph alone, applied to all three
together, lost five materially correct turns while moving tokens 1.2%:

| Arm | `expand_threshold` / `description_tokens` | Tokens | Materially correct |
|---|---|---:|:--:|
| graph alone | 0.55 / 100 (defaults) | 9,045,017 | 23/30 |
| graph alone | **0.62 / 250** | **8,462,341** | **24/30** |
| all three | 0.55 / 100 (defaults) | 2,406,570 | **21/30** |
| all three | 0.62 / 250 | 2,377,270 | **16/30** |

*Folding harder buys nothing once the filter has already compressed the evidence — it only costs
recall.* One replay each, on Haiku 4.5, so read the direction rather than the quantity.

If you install both, start from the package defaults for the graph. If you run the graph alone, a
larger `description_tokens` is the cheapest accuracy you can buy: the budget is spent on
`Card.numeric_lines`, which are exact substrings of the turn's own text, so more budget means more
figures survive verbatim instead of being re-rendered from a paraphrase — which is where a `42,1%`
turns into `42.1%` and fails a check that the arithmetic would have passed.

## Prompt caching and these plugins

Read this before you turn prompt caching on, because two of the three plugins get *more* expensive with
it — and the reason is mechanical, not a tuning problem.

Prompt caching and context compression attack the same waste: a growing conversation re-sent on every
call. Caching makes that repetition cheap to re-read; these plugins delete the repetition. Only one of
the two can bill it, so on a large-window model they are **alternatives, not a stack**.

### The rule

| You are running | Do this |
|---|---|
| An Anthropic model (Claude), large window | **Caching on + `RelevanceFilter` only.** Leave the graph and disclosure off. |
| An Anthropic model, and you want maximum compression | **Caching off + all three.** This is the configuration the [landing page](../README.md) measures. |
| A small-window model (under ~250K), or one with no caching | **All three**, caching irrelevant. Completion is the constraint, not cost. |
| An OpenAI model on Bedrock (Astra, Sol) | **Relevance filter only.** Caching there is implicit and cannot be switched off, so the graph and disclosure pay the write premium with no way to avoid it. |

### Why disclosure and the graph fight the cache

The cache only recognises a prompt prefix from its first byte, in order, unaltered. Bedrock processes
cache checkpoints `tools` → `system` → `messages`, and the documentation is explicit that "changing
content in an earlier section invalidates the cache for later sections (for example, modifying `tools`
invalidates the `system` and `messages` caches)" — see [Prompt
caching](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html). Tokens read from
cache are billed at the cache-read rate, and tokens *written* to cache can be billed above the standard
input rate; on Bedrock that write premium is ~1.25x input against a read at ~0.10x.

Which puts each plugin in a different position:

- **`ProgressiveToolDisclosure` edits `toolConfig`** — the very first section. Every reveal invalidates
  the whole prompt, history included. Measured on Opus 5, 44 of 109 calls came back with `cacheRead = 0`
  and those 44 carried 98.4% of the arm's cache writes. Note the irony: the plugin works (schema down
  from 62,656 tokens to ~5,300), and it is that success that triggers the penalty — a small tool set is
  one you change often. Its catalog now sits in the system prompt, which does not add a second source of
  invalidation: a line only enters or leaves the catalog when the same load or expiry already changed
  `toolConfig` ahead of it, and the line's text is cached so it never drifts on its own. The same holds
  for the folded exchanges — the fold is a function of the projection, so it changes only on a call whose
  `toolConfig` changed anyway. Loading several tools in one `get_tool_details` call is therefore cheaper
  than loading them one at a time.
- **`ContextGraph` removes messages from the middle of the history.** Everything after the edit is new.
  Its digest block is appended after the cache point and so is billed as ordinary input, which is the
  right design; the removal is what cannot be made cache-friendly, because the removal *is* the plugin.
- **`RelevanceFilter` replaces a payload as it arrives and then never touches it.** The prompt stays
  append-only, so the prefix keeps matching. Its read:write ratio under caching is 66-83:1, the same
  band as a bare agent.

The design criterion, if you are writing your own context plugin: it is cache-compatible if and only if
its mutations are **append-only or confined to the end of the prompt** (for Strands, the last `user`
message / `dynamic_trailing_blocks`). Anything that edits an earlier message or the tool set is not.

### Do not prune revealed tools to save money

A reasonable-sounding idea that measures badly: removing tools the agent has finished with. With caching
on, removing a tool invalidates the prefix exactly like adding one, so pruning **doubles** the number of
invalidations. Measured on Opus 5: perfect pruning would save $0.16, and a single extra invalidation
costs $1.13. With caching off it is a genuine but small win — ~3% — because the plugin has already cut
the schema to 7% of the prompt mass and the remaining 93% is history. Prune for **window headroom**,
which is a real reason, not for cost.

### When caching does not pay at all

Caching only earns its write premium if the *same* prefix is re-sent inside the TTL. These agentic
shapes never get there, and in all of them the plugins are the only lever:

- **A system prompt per tenant**, or a **tool set per user permission** — every request has a different
  prefix, and the tool set is the worst possible place for a difference.
- **One-shot fan-out** — classifying thousands of documents, one call each, no reuse.
- **A prompt A/B test in production** — two variants, two prefixes, half the traffic each.
- **A human who thinks for longer than the TTL** between turns. Bedrock's default is 5 minutes; a
  1-hour TTL is available on the Claude models at a higher write rate.
- **A prompt below the model's checkpoint minimum** — 1,024 tokens for Opus 4.8 and Sol, 512 for Opus 5
  and Fable 5, 4,096 for Haiku 4.5. Below it the request still succeeds but nothing is cached.

The measured numbers behind all of this, per model and per configuration, are in
[`BENCHMARK.md`](../BENCHMARK.md).

## Where the effect shows up

Not in one turn. The practices act on what a growing conversation carries forward, so a single question
against a single tool shows almost nothing — the schema floor is small and there is no history yet. The
difference appears over dozens of turns with a realistic tool count, which is what the benchmark
replays: **82% fewer tokens for the same 28 of 30 materially correct turns**, at $13.36 against
$72.50, and three seconds faster per turn.

The values above are close to the benchmark's, which tunes them to the case it measures. See
[`validation/community-plugin-A-B-D/src/config.py`](../validation/community-plugin-A-B-D/src/config.py)
for the tuned set and why each value is what it is.

To see the numbers instead of the wiring:

```bash
cd validation/community-plugin-A-B-D
./run.sh --total-turns 60 --tag myrun
```

See [`validation/community-plugin-A-B-D/README.md`](../validation/community-plugin-A-B-D/README.md)
for what is measured and how, and [`docs/design/design.md`](../docs/design/design.md) for why each
practice is shaped the way it is.
