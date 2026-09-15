# Running a Strands agent with the context plugins

> **⚠️ Not for production use.** This is a minimal working sample for experimentation and learning.
> It has no guardrails, no error handling and no operational hardening.

A runnable example of the three practices — [relevance filtering](../docs/design/design-a-relevance-filtering.md),
[progressive tool disclosure](../docs/design/design-b-progressive-tool-disclosure.md) and the
[context graph](../docs/design/design-d-context-graph.md) — installed on a Strands agent, one at a
time and then together.

This is the shape the benchmark under [`validation/01-designA-B-D/`](../validation/01-designA-B-D/README.md)
measures. Start here to see the wiring; go there to see what it costs. To deploy this agent to
AgentCore Runtime, see
[`validation/01-designA-B-D/agentcore/DEPLOY.md`](../validation/01-designA-B-D/agentcore/DEPLOY.md).

## What you need

- **Python 3.12.**
- **AWS credentials** for an account with these Bedrock models enabled in `us-east-1`:
  `us.anthropic.claude-opus-4-8`, `cohere.rerank-v3-5:0`, `cohere.embed-multilingual-v3`.
  Credentials resolve through the standard AWS chain.
- **The SDK**, installed from the public fork that vends the plugins. The commit SHA is an immutable
  reference — the same one the measured results use:

```bash
pip install "strands-agents @ git+https://github.com/scandura/harness-sdk.git@c4083a04d35170b0626cfbdbb5e4926d9b35af20#subdirectory=strands-py"
```

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

## Offloading: the floor to measure against

The plugins act on tool results, so a result too large for the window has to go somewhere first. The
**offloader** writes the payload to storage and leaves a short preview plus a retrieval tool in its
place. On its own it is not one of the three practices — it is the baseline the other three improve on.

```python
from strands.vended_plugins.context_offloader import ContextOffloader, FileStorage

agent = Agent(
    model=build_model(),
    system_prompt=SYSTEM_PROMPT,
    tools=[account_statement],
    plugins=[
        ContextOffloader(
            storage=FileStorage(".artifacts/offload"),
            max_result_tokens=4_000,   # above this, the result is offloaded
            preview_tokens=1_000,      # what stays in the conversation
            evict_after_cycles=40,     # how long the payload stays retrievable
        )
    ],
)
```

The preview is the first 1,000 tokens of the payload — whatever happens to be at the top, which is
rarely the answer.

## A — Relevance filtering

Scores the payload's chunks against the question and keeps the ones that answer it, instead of a
positional slice. It is a context-manager strategy, so it arrives through `context_manager=` rather
than `plugins=`:

```python
from strands.experimental.context_manager import BedrockReranker, ContextManager, Offload
from strands.storage import InMemoryStorage

relevance_strategy = Offload.relevance(
    "tool_results",
    {
        "reranker": BedrockReranker(region_name=REGION),
        "relevance_threshold": 0.5,   # minimum score for a chunk to get in
        "chunk_tokens": 2500,         # chunk size for scoring
        "preview_tokens": 1000,       # budget for what stays visible
    },
).when(threshold=2500)                # only results above this are rewritten

agent = Agent(
    model=build_model(),
    system_prompt=SYSTEM_PROMPT,
    tools=[account_statement],
    context_manager=ContextManager(
        strategies=[relevance_strategy],
        # retrieval_tool registers retrieve_context, so the model can still reach the full payload.
        stash={"storage": InMemoryStorage(), "retrieval_tool": True},
    ),
)
```

**This replaces the offloader — do not install both.** The offloader rewrites a result to a positional
preview as it enters the conversation, so relevance would score the preview instead of the payload. The
manager keeps its own stash, so the full payload is still persisted and retrievable.

`InMemoryStorage` keeps the stash in the process, which is enough for a sample; swap it for a
persistent storage when the payload has to outlive the run.

Ranking costs a rerank call per oversized result. That is the trade: a second API call to spend the
same preview budget on the chunks that answer the question.

## B — Progressive tool disclosure

Sends a lean catalog — one short line per tool — and fetches a tool's full schema only when it is
needed, then forgets it. This attacks the fixed cost paid on every call, which is schema and not
history.

```python
from strands.vended_plugins.progressive_tool_disclosure import ProgressiveToolDisclosure

agent = Agent(
    model=build_model(),
    system_prompt=SYSTEM_PROMPT,
    tools=[account_statement],
    plugins=[
        ContextOffloader(storage=FileStorage(".artifacts/offload"), max_result_tokens=4_000),
        ProgressiveToolDisclosure(
            catalog_tokens=20,   # description budget per undisclosed tool
            ttl_cycles=5,        # cycles a disclosed schema stays resident
            top_k=3,             # tools exposed per find_tools call
            # The retrieval tool must never need discovery: the model is told to call it in the
            # text that replaced the payload. Name the one your setup installs — the offloader
            # registers retrieve_offloaded_content, the context manager registers retrieve_context.
            always_available=["retrieve_offloaded_content"],
        ),
    ],
)
```

The saving scales with catalog size: with a handful of tools there is little schema to avoid. It pays
off at the tool counts a real agent reaches.

## D — Context graph

Reorganizes the conversation into a graph of **cards** — one per turn — selecting what the current turn
needs and collapsing the rest into short descriptions. Collapsed is recoverable, not deleted.

```python
from strands.vended_plugins.context_graph import ContextStrategy

graph = ContextStrategy(strategy="graph", persist=False)

agent = Agent(
    model=build_model(),
    system_prompt=SYSTEM_PROMPT,
    tools=[account_statement],
    plugins=[ContextOffloader(storage=FileStorage(".artifacts/offload")), graph],
)
```

`persist=False` derives the graph by scanning the history on each turn. Set `persist=True` to store it
in `agent.state`, which any session manager then carries — that is what makes the graph survive a
restart, and it is the only case where the load path runs at all.

## All three together

The combination the benchmark calls `graph-all`. Two things it has to get right:

```python
graph = ContextStrategy(strategy="graph", persist=False)

relevance_strategy = Offload.relevance(
    "tool_results",
    {
        "reranker": BedrockReranker(region_name=REGION),
        "relevance_threshold": 0.5,
        "chunk_tokens": 2500,
        "preview_tokens": 1000,
    },
).when(threshold=2500)

agent = Agent(
    model=build_model(),
    system_prompt=SYSTEM_PROMPT,
    tools=[account_statement],
    # No offloader: relevance takes its place.
    plugins=[
        graph,
        ProgressiveToolDisclosure(
            catalog_tokens=20,
            ttl_cycles=5,
            top_k=3,
            always_available=["retrieve_context"],
            # Without this, a card collapsing to a description drops its tool out of the
            # full-schema block, and the model reads about a tool it can no longer call.
            referenced_source=graph.referenced_tool_names,
        ),
    ],
    context_manager=ContextManager(
        strategies=[relevance_strategy],
        stash={"storage": InMemoryStorage(), "retrieval_tool": True},
    ),
)
```

1. **Relevance replaces the offloader** — same rule as section A.
2. **`referenced_source=graph.referenced_tool_names`** — the graph collapses cards, and disclosure has
   to know which tools those cards still mention. The graph introduces the gap, so the graph closes it.

Because relevance is installed, the retrieval tool is `retrieve_context` and that is what
`always_available` names.

## Where the effect shows up

Not in one turn. The practices act on what a growing conversation carries forward, so a single question
against a single tool shows almost nothing — the schema floor is small and there is no history yet. The
difference appears over dozens of turns with a realistic tool count, which is what the benchmark
replays.

The values above are the plugin defaults. The benchmark tunes them to the case it measures — a smaller
chunk size and a much lower relevance threshold, because `cohere.rerank-v3-5` returns low absolute
scores where a 0.5 floor would reject every chunk. See `validation/01-designA-B-D/src/config.py` for the
tuned set and why each value is what it is.

To see the numbers instead of the wiring:

```bash
cd validation/01-designA-B-D
./run.sh --configs baseline graph-all --total-turns 60 --tag myrun
```

See [`validation/01-designA-B-D/README.md`](../validation/01-designA-B-D/README.md) for what is
measured and how, and [`docs/design/design.md`](../docs/design/design.md) for why each practice is
shaped the way it is.
