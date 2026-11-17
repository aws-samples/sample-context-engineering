# strands-progressive-tool-disclosure

Progressive Tool Disclosure for Strands Agents. Instead of sending every tool's full specification on
every model call, the plugin sends in `tool_specs` only the tools that are callable on the call —
`find_tools`, `get_tool_details`, the `always_available` tools and the schemas loaded and still live
by TTL — and lists every other tool as one line in the system prompt: its name and a summary of its
description, at most `catalog_chars` characters. The model loads full schemas with
`get_tool_details([names])`, and they leave again after `ttl_cycles` idle cycles. `find_tools`
searches when no catalog name fits.

A tool that left `tool_specs` leaves nothing behind that invites a call: in the messages each call
sends, the closed exchanges of tools that call does not carry are folded to one sentence — `The tool X
was called and the result was: Y` — and the plugin's own `find_tools` / `get_tool_details` exchanges
are dropped. The turn in flight is never folded. The evidence stays, the call shape the model would
copy does not.

The plugin reads the per-call copy the event loop hands to `InvokeModelStage` and never mutates
`agent.tool_registry.registry` or `agent.messages` — every tool stays callable and the stored history
is intact, regardless of what one call shows.

The lexical tool index, the catalog lines and the closed-exchange fold live in `agent-context-core`
(`context_core.disclosure`), the same code the LangGraph and Hermes bindings run. This package keeps what is
specific to Strands: the per-agent state, the TTL, the model summarizer and the cancellation write-back
(`plugin.py`), every SDK import (`_compat.py`), and the message conversion (`_adapter.py`).

## Install

From a clone of [sample-context-engineering](https://github.com/aws-samples/sample-context-engineering):

```bash
uv pip install -e context-core -e strands-community-plugins/strands-progressive-tool-disclosure
```

Two runtime dependencies: `agent-context-core` and `strands-agents`. The default index is standard-library only. The default
summarizer calls the agent's own model once per tool whose description is longer than
`catalog_chars`, on the first projected call, and caches the line; pass a `summarizer` to avoid it.

## Usage

```python
from strands import Agent
from strands_progressive_tool_disclosure import ProgressiveToolDisclosure

agent = Agent(
    tools=[...],
    plugins=[
        ProgressiveToolDisclosure(
            catalog_chars=80,       # summary limit per catalog line; None = no catalog
            ttl_cycles=3,           # cycles a loaded schema survives after its last use
            always_available=[],    # tools that skip discovery and carry full spec every call
        )
    ],
)
```

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `catalog_chars` | `int >= 1 \| None` | `80` | Character limit of a catalog line's summary. `None` drops the catalog entirely, leaving only the two plugin tools as the hint that other tools exist. |
| `summarizer` | `Callable[[ToolSpec, int], str \| Awaitable[str]] \| None` | `None` | Writes a catalog line for a description longer than the limit. `None` uses the agent's model; usage is reported as `summary_usage`. Failures fall back to a boundary cut. |
| `ttl_cycles` | `int >= 1` | `3` | Idle cycles a loaded schema survives after its last use. Every call that runs the tool renews it. |
| `always_available` | `Sequence[str]` | `()` | Names that carry their full specification on every call, skipping the discovery cycle. |
| `index` | `ToolIndex \| None` | `LexicalToolIndex()` | Search implementation behind `find_tools`. Any object exposing `build` and `search`. |
| `top_k` | `int >= 1` | `3` | How many tools a single search lists. |

A call to a catalog name that skipped the load is cancelled with a message pointing at
`get_tool_details`, and nothing is loaded on the model's behalf — a recovery that loaded it would
teach the model that calling a catalog name directly works. A tool whose parameters are all optional
is exempt: it is callable with no arguments, so the call is not a guess.

## Compatibility

| Plugin version | `strands-agents` tested |
| --- | --- |
| 0.2.x | 1.44 – 1.65 |
| 0.1.x | 1.44 – 1.61; fails to import from 1.62 |

## Private-API caveat

The plugin reaches the middleware stage through `InvokeModelStage` and `InvokeModelContext`, and calls
`agent._middleware_registry.add_middleware(InvokeModelStage.Input, handler)`, where
`InvokeModelContext.tool_specs`, `.system_prompt` and `.messages` are mutable, defensively-copied
per-call fields — the three the handler replaces. `BeforeModelCallEvent`, the closest hook, does not
carry `tool_specs` and cannot rewrite them.

The two classes were private (`strands._middleware.stages`) until Strands 1.61 and are public
(`strands.middleware`) from 1.62, which removed the private package without an alias. The
`_middleware_registry` attribute is still underscore-prefixed in every version.

How the package contains it:

- The import is isolated in `_compat.py`, which tries `strands.middleware` first and falls back to
  `strands._middleware`, so the same code runs on both sides of 1.62 and an upstream rename touches one file.
- `pyproject.toml` pins `strands-agents` to a conservative range whose upper bound excludes the next
  major release, so a new major SDK cannot silently pull in a moved seam.
- `tests/test_compat.py` asserts, on every SDK version tested in CI, that `InvokeModelStage.Input` is
  importable, that `InvokeModelContext` has a `tool_specs` field, and that `add_middleware` exists on
  the middleware registry. A moved seam fails there rather than at your runtime.

## License

Apache License 2.0. See [LICENSE](LICENSE).
