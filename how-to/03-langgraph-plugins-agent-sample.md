# Running a LangChain / LangGraph agent with the context-engineering middlewares

> **⚠️ Not for production use.** This is a minimal working sample for experimentation and learning.
> It has no guardrails, no error handling and no operational hardening.

A runnable example of the three practices — [relevance filtering](../docs/design/design-a-relevance-filtering.md),
[progressive tool disclosure](../docs/design/design-b-progressive-tool-disclosure.md) and the
[context graph](../docs/design/design-d-context-graph.md) — installed on a LangChain v1 `create_agent`
agent as **three middlewares**, one at a time and then together.

The logic is the same as in the Strands plugins
([`02-community-plugins-agent-sample.md`](02-community-plugins-agent-sample.md)): both are thin bindings
over one framework-agnostic package, [`context-core`](../context-core/). Same defaults, same tool names,
same model-facing text. The places where LangChain forces a difference are listed in
[`langgraph-plugins/README.md`](../langgraph-plugins/README.md#parity-with-the-strands-plugins).

This is the shape the benchmark under [`validation/plugins-langgraph/`](../validation/plugins-langgraph/README.md)
measures. **Read [the things that will bite you](#things-that-will-bite-you) before wiring all three
together** — two of them lose state silently.

## Run it

### Prerequisites

| | Requirement | Why |
|---|---|---|
| Python | **3.10 or newer** (3.12 for the benchmark) | the packages' `requires-python` |
| `langchain` / `langgraph` | **>= 1.0, < 2** | the `create_agent` middleware API; verified on `langchain` 1.4.2 |
| `langchain-aws` | >= 1.7 | `ChatBedrockConverse`, the Bedrock chat model |
| AWS CLI | **v2** | only to configure and verify credentials |

Bedrock models that must be **enabled in your account**, in the region you use:

| Model id | Used by |
|---|---|
| `us.anthropic.claude-opus-4-8` | the agent itself (any Converse-capable model works) |
| `cohere.rerank-v3-5:0` | relevance filtering |
| `cohere.embed-multilingual-v3` | the context graph's similarity matcher |

Credentials come from the standard `boto3` chain, so whatever the AWS CLI is configured with is what the
agent uses. Configure and verify them exactly as in
[step 1 of the Strands guide](02-community-plugins-agent-sample.md#1-configure-aws-credentials-with-the-aws-cli):
`aws configure` (or `aws configure sso`), then `aws sts get-caller-identity`. The identity needs
`bedrock:InvokeModel` on the three model ids above and `bedrock:Rerank` for the filter.

### Install

The packages are not on PyPI yet, so they install from a clone:

```bash
git clone https://github.com/aws-samples/sample-context-engineering.git
cd sample-context-engineering

uv venv --python 3.12 .venv
source .venv/bin/activate                   # Windows: .venv\Scripts\activate

uv pip install -e context-core \
               -e langgraph-plugins/langgraph-relevance-filter \
               -e langgraph-plugins/langgraph-progressive-tool-disclosure \
               -e langgraph-plugins/langgraph-context-graph \
               "langchain>=1.0,<2" "langgraph>=1.0,<2" "langchain-aws>=1.7,<2"
```

Plain `pip install -e ...` works the same way.

Put any code block below into `agent.py` and run `python agent.py`. **This spends money on Bedrock:**
every turn is a real model call, and the sample's tool returns a ~40k-character payload on purpose.

---

## The starting point

An agent with one oversized tool and a checkpointer, so several `invoke` calls on one `thread_id` are one
conversation.

```python
import boto3
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_aws import ChatBedrockConverse
from langgraph.checkpoint.memory import InMemorySaver

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


def build_model() -> ChatBedrockConverse:
    # No temperature: Opus 4.8 rejects the parameter.
    return ChatBedrockConverse(
        client=session.client("bedrock-runtime"), model=MODEL_ID, max_tokens=4096
    )


agent = create_agent(
    model=build_model(),
    tools=[account_statement],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=InMemorySaver(),
)

thread = {"configurable": {"thread_id": "demo"}}
question = "Which was the largest purchase in the last 90 days on account 0001/12345-6?"
result = agent.invoke({"messages": [{"role": "user", "content": question}]}, thread)
print(result["messages"][-1].content)
```

Each section below changes only the `middleware=` list of that `create_agent` call.

## A — Relevance filtering

`RelevanceFilterMiddleware` wraps every tool call (`wrap_tool_call`). When a result exceeds
`max_result_tokens`, it splits the result into chunks, scores them against the user's question with a
reranker, and replaces the `ToolMessage` with a marker, a disclaimer and the best chunks verbatim, plus a
reference to the full content. The payload never enters the history whole. At the end of each run
(`after_agent`) it deletes its own closed `retrieve_all_context` exchanges from the state.

```python
from context_core.relevance import BedrockReranker, FileStore
from langgraph_relevance_filter import RelevanceFilterMiddleware

relevance = RelevanceFilterMiddleware(
    store=FileStore(".artifacts/relevance"),  # full results, readable back by reference
    max_result_tokens=4_000,
    config={
        "reranker": BedrockReranker("cohere.rerank-v3-5:0", boto_session=session),
        "relevance_threshold": 0.02,  # see "things that will bite you", item 3
        "chunk_tokens": 500,
        "preview_tokens": 800,
    },
)

agent = create_agent(
    model=build_model(),
    tools=[account_statement],
    system_prompt=SYSTEM_PROMPT,
    middleware=[relevance],
    checkpointer=InMemorySaver(),
)
```

The middleware registers `retrieve_all_context` (turn it off with `include_retrieval_tool=False`). It is
scoped to the one question an excerpt cannot answer — one that needs every row — and the disclaimer
names it, with the reference to pass.

## B — Progressive tool disclosure

`ProgressiveToolDisclosureMiddleware` rewrites each model call (`wrap_model_call`). Instead of every tool's
full schema, the model gets a one-line catalog in the system message and two tools, `find_tools` and
`get_tool_details`. A tool's full spec is sent only after the model asks for it, for `ttl_cycles` calls,
and closed exchanges of tools the call does not carry are folded out of the messages sent. It pays off with
many tools; with one it only adds calls.

```python
from langgraph_progressive_tool_disclosure import ProgressiveToolDisclosureMiddleware

disclosure = ProgressiveToolDisclosureMiddleware(
    catalog_chars=80,  # longer descriptions are summarized by the agent's own model
    ttl_cycles=3,
    top_k=4,
)

agent = create_agent(
    model=build_model(),
    tools=[account_statement],  # register ALL tools; disclosure decides what the model sees
    system_prompt=SYSTEM_PROMPT,
    middleware=[disclosure],
    checkpointer=InMemorySaver(),
)
```

The catalog summaries cost tokens once per tool; they are counted in `disclosure.summary_usage`. Pass your
own `summarizer=` to write them some other way.

## D — Context graph

`ContextGraphMiddleware` turns the history into Cards, one per closed turn, and at each model call sends
every Card at the resolution the current question needs: full content, a Description, or only its Title.
The rewrite is `request.override(messages=...)`, so it is transient: the persisted state is never trimmed,
and a Card folded too far is one `expand_card` call away. It also records every tool return as an artifact,
readable back through `expand_artifact`, and gives the model `find_context` to search earlier turns.

```python
from context_core.graph import EmbeddingSimilarityMatcher
from langgraph_context_graph import ContextGraphMiddleware

graph = ContextGraphMiddleware(
    matcher=EmbeddingSimilarityMatcher("cohere.embed-multilingual-v3", boto_session=session),
    expand_threshold=0.62,
    collapse_floor=0.45,
    body_budget=40_000,  # the only ceiling on full-content mass
)

agent = create_agent(
    model=build_model(),
    tools=[account_statement],
    system_prompt=SYSTEM_PROMPT,
    middleware=[graph],
    checkpointer=graph_checkpointer(),  # defined in "things that will bite you", item 2
)
```

## All three together

Order is the composition: LangChain nests hooks in list order, first is outermost. The graph projects the
history first, disclosure then rewrites tools and folds, and relevance acts on a tool result before either
sees it. The two plugins touch disjoint fields of the model request (`messages` against `tools` and
`system_message`), which is why they stack.

```python
relevance = RelevanceFilterMiddleware(
    store=FileStore(".artifacts/relevance"),
    max_result_tokens=4_000,
    config={
        "reranker": BedrockReranker("cohere.rerank-v3-5:0", boto_session=session),
        "relevance_threshold": 0.02,
        "chunk_tokens": 500,
        "preview_tokens": 800,
    },
)
graph = ContextGraphMiddleware(
    matcher=EmbeddingSimilarityMatcher("cohere.embed-multilingual-v3", boto_session=session),
    body_budget=40_000,
    stash=relevance.stash,  # a [ref: mem_N_...] the filter mints also resolves via expand_artifact
)
disclosure = ProgressiveToolDisclosureMiddleware(
    catalog_chars=80,
    ttl_cycles=3,
    top_k=4,
    # The graph's retrieval tools must never need discovery: the folded context tells the model to
    # call them. Read the names off the middleware instead of hard-coding them.
    always_available=[each.name for each in graph.tools],
)

agent = create_agent(
    model=build_model(),
    tools=[account_statement],
    system_prompt=SYSTEM_PROMPT,
    middleware=[graph, disclosure, relevance],
    checkpointer=graph_checkpointer(),
)
```

Every hook ships both a sync and an async version, so the stack runs under `agent.invoke` and
`agent.ainvoke`.

## Things that will bite you

### 1. No checkpointer, no conversation

Without a checkpointer each `invoke` starts from an empty history, and the relevance filter's end-of-run
cleanup (a `RemoveMessage` state write) has nowhere to land. Use one, and reuse the same `thread_id` for
every turn of a conversation.

### 2. The checkpointer must be allowed to restore the graph's state

The graph persists its state as dataclasses from `context_core.graph.state`. LangGraph's serializer only
restores types on its allowlist, and the allowlist is keyed by class, not by module: pass a module name and
the list is strict *and* empty, so every restore drops the graph and it silently starts from scratch each
turn. The benchmark caught exactly that. Pass the classes:

```python
import inspect

from context_core.graph import state as graph_state
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


def graph_checkpointer() -> InMemorySaver:
    types = tuple(
        each
        for each in vars(graph_state).values()
        if inspect.isclass(each) and each.__module__ == graph_state.__name__
    )
    return InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=types))
```

### 3. The relevance threshold is a position in a distribution, not a number

The default is `0.5`. With `cohere.rerank-v3-5` that rejects every chunk: a strong match scores ~0.29 and
an unrelated chunk ~0.03, so the benchmark uses `0.02`. Change the rerank model and re-derive it. The same
holds for the graph's thresholds against its embedding model.

### 4. Do not add a summarization or trimming middleware next to the graph

The graph folds the history without deleting it. A middleware that trims `state["messages"]` deletes what
the graph only meant to fold, and raising that Card back up then recovers nothing. LangGraph cannot forbid
the combination, so the graph warns once at construction when it sees one and still registers.

### 5. Hand the graph the filter's `stash`

Strands bridges the two stores through the SDK's context-manager Stash; LangGraph has none. Without
`stash=relevance.stash`, a reference minted by the filter resolves through `retrieve_all_context` but not
through the graph's `expand_artifact`, and the model gets an "unreachable" answer when it picks the wrong
tool.

## Where the effect shows up

In one turn, on one tool, the saving is small. The practices pay off over a conversation, where a payload
the bare agent re-sends on every later call is sent once as a preview. On the 60-turn benchmark with
Opus 4.8 and prompt caching off, the three middlewares together sent **83.5% fewer tokens** than the bare
agent at the same accuracy (29 of 30 scored turns), and cost $13.18 against $76.57. The Strands plugins on
the same model reach −83.9%. Table and method: [`langgraph-plugins/README.md`](../langgraph-plugins/README.md#benchmark).

To reproduce it, see [`validation/plugins-langgraph/README.md`](../validation/plugins-langgraph/README.md).
`python -m src.run --dry-run` builds every arm and calls nothing; a live run needs
`--live --i-understand-this-spends-money` and costs a few hundred dollars at list price.
