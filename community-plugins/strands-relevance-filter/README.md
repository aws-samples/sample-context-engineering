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

## Retrieving the rest of a result

`retrieve_all_context` takes the `reference` from the excerpt plus one way of bounding the read.
Precedence is `line_range`, then `pattern`, then `max_chunks`; `max_tokens` bounds the response in
every mode.

| Argument | Returns |
|---|---|
| `pattern` (regex) | only the matching lines, numbered, with `context_lines` around each (default 5). The cheapest way to aggregate: match the rows to sum and nothing else |
| `line_range` `{start, end}` | exactly that span, 1-indexed inclusive. The line numbers a `[... N lines omitted ...]` marker reports are the ones to pass here. An `end` past the last line is clamped, `sed`-style |
| `max_chunks` | the N most relevant chunks, rendered in document order with markers for the lines left out. The ranking is the one the filter already computed, so no second rerank is charged. A value at or above the result's chunk count returns all of it |
| `max_tokens` | approximate size limit of the response. Alone, it returns the whole content cut to that budget |
| none of them | the full original content, unbounded |

Without `max_tokens`, a bounded read answers within `max_result_tokens`, so reading content back
never costs more than keeping the original would have.

## Install

```bash
pip install strands-relevance-filter
```

## Development

```bash
hatch fmt      # ruff format + check
hatch test     # pytest across the supported Python matrix
```
