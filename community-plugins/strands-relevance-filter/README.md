# strands-relevance-filter

Relevance filtering of oversized tool results for Strands Agents.

`RelevanceFilter` is a community `Plugin` that attaches to the public `AfterToolCallEvent` hook. When a
tool result exceeds `max_result_tokens`, the plugin chunks the result, scores the chunks against a query
built from the latest user question and the tool-call arguments, and rewrites `event.result` into a
budget-bounded preview composed only of verbatim source substrings and gap markers, headed by a disclaimer
that states how much of the result the excerpt covers. The raw sub-blocks are written to a `Store` first,
and the reference in the rewrite lets the model load the whole result through the plugin's
`retrieve_all_context` tool -- meant only for questions an excerpt cannot answer, those that need every
row (a maximum, a total, a count). Its exchanges are removed from the history when the turn ends. With
`ProgressiveToolDisclosure`, keep the tool out of `always_available` so it is loaded only when needed.

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
