# langgraph-progressive-tool-disclosure

LangGraph binding for **Practice B** (progressive tool disclosure). The lexical tool index and the
lean-catalog / closed-exchange-fold logic live in `context-core` (`context_core.disclosure`). This
package is the thin interface: a `create_agent` middleware whose `wrap_model_call` rewrites the model
call for a single turn — `request.override(tools=…, system_message=…+catalog, messages=…folded)` — plus
the `find_tools` / `get_tool_details` tools and the `BaseMessage` ↔ neutral adapter.

Tested `langchain` / `langgraph` versions are in [Compatibility](#compatibility).

## Install

From a clone of [sample-context-engineering](https://github.com/aws-samples/sample-context-engineering):

```bash
uv pip install -e context-core -e langgraph-plugins/langgraph-progressive-tool-disclosure
```

This pulls in `agent-context-core`, `langchain` and
`langgraph`. Tool search is lexical and local. By default, catalog lines that exceed `catalog_chars` are
summarized by the agent's own model.

## Compatibility

The package declares `langchain>=1.2.12,<2` and `langgraph>=1.1.5,<2`. These are the oldest versions where
the three LangGraph plugin suites and the validation harness all pass.

| `langchain` | `langgraph` | `langchain-core` | Result |
| --- | --- | --- | --- |
| 1.5.0 | 1.2.22 | 1.8.1 | passes |
| 1.4.2 | 1.2.12 | 1.6.5 | passes |
| 1.3.0 | 1.2.22 | 1.8.1 | passes |
| 1.2.12 – 1.2.18 | 1.1.5 / 1.1.10 | 1.8.1 | passes |
| 1.2.10 | 1.0.8 | 1.2.10 | 2 failures |
| 1.0.0 – 1.2.0 | 1.0.x | | 37 failures |

Before `langchain` 1.1, `ModelRequest` has no `system_message` field, which the middleware overrides to add
the catalog. On 1.2.10 with `langgraph` 1.0.8, the full agent does not restrict its tools and a direct call to
a catalog tool is not cancelled. `langgraph` below 1.1.5 could not be tested with `langchain` 1.2.12: those
releases accept a `langgraph-prebuilt` that imports `ExecutionInfo`, which only exists in newer `langgraph`.

## Usage

```python
from langchain.agents import create_agent
from langgraph_progressive_tool_disclosure import ProgressiveToolDisclosureMiddleware

agent = create_agent(
    model="...",
    tools=[...],  # the full tool set; the model sees a lean catalog plus find_tools / get_tool_details
    middleware=[ProgressiveToolDisclosureMiddleware(catalog_chars=80)],
)
```

Source, design notes and benchmarks:
[sample-context-engineering](https://github.com/aws-samples/sample-context-engineering).
