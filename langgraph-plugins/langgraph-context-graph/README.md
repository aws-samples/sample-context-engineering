# langgraph-context-graph

LangGraph binding for **Practice D** (context graph). The Card model, deterministic scan, scoring,
similarity matcher and the per-call `project()` entry point live in `context-core`
(`context_core.graph`). This package is the thin interface: a `create_agent` middleware whose
`wrap_model_call` projects the message list into Cards for a single call via
`request.override(messages=…)`, keeping persisted state intact; whose `wrap_tool_call` records each tool
return in the conversation's reference store and derives the artifact Card of any reference the return
names; the three retrieval tools; the serialized-graph state schema; and the `BaseMessage` ↔ neutral
adapter.

Because `request.override(messages=…)` is transient, no persisted state is deleted — the
`NullConversationManager` precondition becomes "do not also install a summarization/pruning middleware",
which the middleware warns about at construction. Verified against `langchain` 1.x.

## The three retrieval tools

| Tool | Reaches | Switch |
| --- | --- | --- |
| `expand_card` | an earlier turn of this conversation, by title | none |
| `expand_artifact` | a stored artifact, by the address a placeholder carried | `include_artifact_tool` |
| `find_context` | earlier turns, searched by description in the model's own words | none |

All three answer with text on every miss and share one `max_retrieval_cycles` budget per turn. The
guidance block folded into a projected call names exactly the tools that are registered, so excluding one
stops it being advertised without a second switch.

`include_artifact_tool=False` leaves `expand_artifact` unregistered. Pass it when a middleware that
offloads tool results is installed beside this one — typically `RelevanceFilterMiddleware` — because each
then ships a retrieval tool over a store the other cannot read, and the model has two plausible tools for
one job. `expand_card` and `find_context` reach back into the conversation's own turns, which is a job no
offloader does, so they have no switch.

## Where an artifact lives

The Card holds the **address** and the store holds the block, so nothing in the graph rots when the
content behind a reference changes. The store is process-local, one per thread id, held on the middleware
instance and never written into agent state: LangGraph copies every state update as it applies it, and the
blocks are the one part of this plugin that is content rather than addresses.

`wrap_tool_call` stores a return's text and JSON blocks under `<tool_call_id>_<block index>`, which is the
key format the relevance filter hands its own store. Its own three tools' answers are skipped. Targeted
reads (`line_range`, `pattern`) are delegated to `context_core.relevance.search._search_content`, which
this package registers into `context_core.graph.store.HOST_SYMBOLS` at import; `"context_manager"` stays
unregistered because LangGraph has no equivalent of the Strands `ContextManager` Stash, so a reference
this binding did not record resolves to prose naming the miss.
