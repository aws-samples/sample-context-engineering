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
**verbatim** preview of the chunks that score highest against the question, plus a reference the model
reads back through the plugin's own `retrieve_context` tool.

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
line numbers in those markers are the ones `retrieve_context` accepts as a `line_range`.

Construction is inert: the reranker is built on the first filtered result, so a filter that never
fires needs no credentials.

Ranking costs one rerank call per oversized result. That is the trade: a second API call to spend the
same preview budget on the chunks that answer the question.

## B — Progressive tool disclosure

Projects the call's tool list down to the `find_tools` search tool, the `always_available` tools, the
schemas already exposed and still live by TTL, the tools the history references, and a ~20-token
catalog entry for everything else. A full schema enters the call when the model searches for it, and
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
            catalog_tokens=20,   # description budget per undisclosed tool; None drops the catalog
            ttl_cycles=5,        # cycles an exposed schema survives after its last use
            top_k=4,             # tools exposed per find_tools call
            # A retrieval tool must never need discovery: the model is told to call it in the text
            # that replaced the payload. Name the ones your setup installs.
            always_available=["retrieve_context"],
        )
    ],
)
```

The saving scales with catalog size: with a handful of tools there is little schema to avoid. It pays
off at the tool counts a real agent reaches — the benchmark runs 93 tools for ~63k tokens of schema
per call.

Expect the model to **skip the search**. Measured over 60 turns: 5 `find_tools` searches against 14
premature cancellations. The catalog names a tool, the model calls it straight away with no arguments,
the plugin's pre-call guard cancels the call and exposes the schema, and the model retries
successfully. Nothing is lost, but each occurrence costs one model cycle.

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
    # The filter below ships its own retrieval tool over its own store, and the graph's
    # `expand_artifact` reads a store that cannot resolve the filter's references. Two plausible
    # tools for one job, and only one of them can answer -- see gotcha 1. Excluding it here is the
    # supported way to say so; earlier revisions of this page reached into `graph._tools`.
    include_artifact_tool=False,
)

# GOTCHA 1: two artifact-retrieval tools, two stores, no bridge. Drop the graph's so the only
# artifact path is the filter's retrieve_context, which is the one that can resolve its references.
graph._tools = [t for t in graph._tools if t.tool_name != "expand_artifact"]


# GOTCHA 2: the community graph publishes no accessor for the tools a stepped-down Card still
# mentions, so the bridge disclosure accepts has to be supplied. Without it the model reads about a
# tool whose inputSchema left the call.
def graph_referenced_tools(agent):
    """Return the tool names of every Card the current turn did not take at full content."""
    state = graph._states.get(agent)
    if state is None or state.choice.full_pass:
        return ()
    selected = state.choice.selected
    names = set()
    for title, card in state.cards.items():
        choice = state.choice.by_title.get(title)
        stepped_down = (selected is not None and title not in selected) or (
            choice is not None and (choice.dialogue != "full" or choice.evidence != "full")
        )
        if stepped_down:
            names.update(card.tool_names)
    return tuple(sorted(names))


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
            catalog_tokens=20,
            ttl_cycles=5,
            top_k=4,
            # Derived, never hard-coded. A tool disclosure has not exposed is reduced to a catalog
            # entry with an EMPTY inputSchema, and every retrieval tool here needs arguments -- a
            # Title, a reference, a search need. So a hidden one is called with nothing, cancelled by
            # the premature-call guard, and only exposed on the retry: the model pays a round trip to
            # learn what the folded-context guidance already told it to do. Reading the names off the
            # plugin keeps this correct when `include_artifact_tool` is false, as it is above.
            always_available=[*graph.retrieval_tool_names, "retrieve_context"],
            referenced_source=graph_referenced_tools,
        ),
    ],
)
```

The three plugins compose without fighting: the filter acts on tool results as they arrive, the graph
rewrites the per-call message list, and disclosure rewrites the per-call tool list. None of them
mutates `agent.messages` or the tool registry.

## Four things that will bite you

### 1. Two retrieval tools for one job, and only one can answer

`RelevanceFilter` hands out references that **only its own** `retrieve_context` resolves.
`ContextGraph` resolves **its own** references through `expand_artifact`. Nothing bridges the two
stores — the graph's README is explicit that its bridge to another plugin's stash is built entirely on
private symbols and degrades to "answers as prose naming the miss".

Installed together, the model reaches for whichever looks right and gets a miss. Measured, it said so
in its own answer:

> "every export's artifact reference has come back **unreachable** … I can't read the stored
> artifacts."

That cost the benchmark two of eighteen scored turns. Dropping the graph's `expand_artifact` took the
full stack from **84.5% / 15-of-18 to 94.4% / 17-of-18**. The graph's other two tools stay:
`expand_card` and `find_context` reach into the conversation's own turns, which the filter does not do.

This is the vended stack's one structural advantage — there, relevance lived *inside* the
`ContextManager` whose stash the graph bridged to, so there was one store and one retrieval path.

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
  one you change often.
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
