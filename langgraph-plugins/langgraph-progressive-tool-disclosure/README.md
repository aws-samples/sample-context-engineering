# langgraph-progressive-tool-disclosure

LangGraph binding for **Practice B** (progressive tool disclosure). The lexical tool index and the
lean-catalog / closed-exchange-fold logic live in `context-core` (`context_core.disclosure`). This
package is the thin interface: a `create_agent` middleware whose `wrap_model_call` rewrites the model
call for a single turn — `request.override(tools=…, system_message=…+catalog, messages=…folded)` — plus
the `find_tools` / `get_tool_details` tools and the `BaseMessage` ↔ neutral adapter.

Verified against `langchain` 1.x.

## Install

```bash
pip install langgraph-progressive-tool-disclosure
```

This pulls in [`agent-context-core`](https://pypi.org/project/agent-context-core/), `langchain` and
`langgraph`. Tool search is lexical and local. By default, catalog lines that exceed `catalog_chars` are
summarized by the agent's own model.

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
