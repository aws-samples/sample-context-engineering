# strands-relevance-filter

Relevance filtering of oversized tool results for Strands Agents.

`RelevanceFilter` is a community `Plugin` that attaches to the public `AfterToolCallEvent` hook. When a
tool result exceeds `max_result_tokens`, the plugin chunks the result, scores the chunks against a query
built from the latest user question and the tool-call arguments, and rewrites `event.result` into a
budget-bounded preview composed only of verbatim source substrings and gap markers. The raw sub-blocks
are written to a `Store` first, and the reference tokens embedded in the rewrite let the model recover
what was cut through the plugin's `retrieve_context` tool.

No private SDK surface is used: no `ContextManager`, no `_middleware`, no `Stash`.

## Install

```bash
pip install strands-relevance-filter
```

## Development

```bash
hatch fmt      # ruff format + check
hatch test     # pytest across the supported Python matrix
```
