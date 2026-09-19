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

**Read [the three gotchas](#three-things-that-will-bite-you) before wiring all three together.** Two
of them cost a measured benchmark run its answers, and neither fails loudly.

## What you need

- **Python 3.10 or newer** (3.12 for the benchmark harness).
- **AWS credentials** for an account with these Bedrock models enabled in `us-east-1`:
  `us.anthropic.claude-opus-4-8`, `cohere.rerank-v3-5:0`, `cohere.embed-multilingual-v3`.
  Credentials resolve through the standard AWS chain.
- **The three packages plus the public SDK.** They are not published to PyPI yet, so install them
  from this repository:

```bash
pip install -e community-plugins/strands-context-graph
pip install -e community-plugins/strands-progressive-tool-disclosure
pip install -e community-plugins/strands-relevance-filter
pip install "strands-agents>=1.44.0,<2.0.0"
```

Each package declares that same SDK range itself. Verified against **`strands-agents` 1.56.0**: the
private middleware seam the plugins couple to (`strands._middleware.stages.InvokeModelStage`) is
present on the public release, which is what makes "no fork required" a real claim.

Running the sample spends on Bedrock: every turn is a real model call.

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
            always_available=["retrieve_context", "expand_card", "find_context"],
            referenced_source=graph_referenced_tools,
        ),
    ],
)
```

The three plugins compose without fighting: the filter acts on tool results as they arrive, the graph
rewrites the per-call message list, and disclosure rewrites the per-call tool list. None of them
mutates `agent.messages` or the tool registry.

## Three things that will bite you

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

## Where the effect shows up

Not in one turn. The practices act on what a growing conversation carries forward, so a single question
against a single tool shows almost nothing — the schema floor is small and there is no history yet. The
difference appears over dozens of turns with a realistic tool count, which is what the benchmark
replays: **83% fewer tokens for the same 17 of 18 materially correct turns**, at $35.88 against
$203.65.

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
