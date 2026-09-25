# context-core

Framework-agnostic core for the three context-engineering practices:

- **relevance** (Practice A) — chunk / rerank / select / verbatim preview of oversized tool results,
  with a pluggable raw-content store.
- **disclosure** (Practice B) — lexical tool index and lean-catalog / closed-exchange-fold logic.
- **graph** (Practice D) — Card model, deterministic scan, scoring, similarity matcher, and the
  per-call projection.

## The neutral contract

`context_core` holds **all decision logic** and imports **no agent framework** — neither `strands` nor
`langchain`/`langgraph`. It operates on a neutral message shape (`context_core.message`): a dict
`{"role": str, "content": [block, ...]}` where a block is a dict such as `{"text": ...}`,
`{"toolUse": ...}`, `{"toolResult": ...}`, or `{"json": ...}`. Each framework binding adapts its native
message type to this shape at the boundary and calls into the core. The contract is machine-checked by
`tests/test_no_framework_import.py`.

## Provenance

The logic here is **recreated from the Strands community plugins** (`community-plugins/strands-*`) used
as **read-only reference**. This package never modifies those packages; it is an independent
implementation that the LangGraph bindings depend on.
