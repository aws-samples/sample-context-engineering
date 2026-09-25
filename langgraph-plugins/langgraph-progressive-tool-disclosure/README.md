# langgraph-progressive-tool-disclosure

LangGraph binding for **Practice B** (progressive tool disclosure). The lexical tool index and the
lean-catalog / closed-exchange-fold logic live in `context-core` (`context_core.disclosure`). This
package is the thin interface: a `create_agent` middleware whose `wrap_model_call` rewrites the model
call for a single turn — `request.override(tools=…, system_message=…+catalog, messages=…folded)` — plus
the `find_tools` / `get_tool_details` tools and the `BaseMessage` ↔ neutral adapter.

Verified against `langchain` 1.x.
