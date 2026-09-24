# strands-progressive-tool-disclosure

Progressive Tool Disclosure for Strands Agents. Instead of sending every tool's full specification on
every model call, the plugin sends in `tool_specs` only the tools that are callable on the call —
`find_tools`, `get_tool_details`, the `always_available` tools, the schemas loaded and still live by
TTL, and the tools the history references — and lists every other tool as one line in the system
prompt: its name and a summary of its description, at most `catalog_chars` characters. The model
loads full schemas with `get_tool_details([names])`, and they leave again after `ttl_cycles` idle
cycles. `find_tools` searches when no catalog name fits.

The plugin reads the per-call copy the event loop hands to `InvokeModelStage` and never mutates
`agent.tool_registry.registry` — every tool stays callable regardless of what the call shows.

## Install

```bash
pip install strands-progressive-tool-disclosure
```

One runtime dependency: `strands-agents`. The default index is standard-library only. The default
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
            ttl_cycles=5,           # cycles a loaded schema survives after its last use
            always_available=[],    # tools that skip discovery and carry full spec every call
        )
    ],
)
```

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `catalog_chars` | `int >= 1 \| None` | `80` | Character limit of a catalog line's summary. `None` drops the catalog entirely, leaving only the two plugin tools as the hint that other tools exist. |
| `summarizer` | `Callable[[ToolSpec, int], str \| Awaitable[str]] \| None` | `None` | Writes a catalog line for a description longer than the limit. `None` uses the agent's model; usage is reported as `summary_usage`. Failures fall back to a boundary cut. |
| `ttl_cycles` | `int >= 1` | `5` | Idle cycles a loaded schema survives after its last use. |
| `always_available` | `Sequence[str]` | `()` | Names that carry their full specification on every call, skipping the discovery cycle. |
| `index` | `ToolIndex \| None` | `LexicalToolIndex()` | Search implementation behind `find_tools`. Any object exposing `build` and `search`. |
| `top_k` | `int >= 1` | `3` | How many tools a single search lists. |
| `referenced_source` | `Callable[[Agent], Iterable[str]] \| None` | `None` | Supplemental source of names to keep at full spec, on top of the ones the history references. |

## Private-API caveat

The plugin depends on one private SDK seam: importing `InvokeModelStage` and `InvokeModelContext`
from `strands._middleware.stages`, and calling
`agent._middleware_registry.add_middleware(InvokeModelStage.Input, handler)`, where
`InvokeModelContext.tool_specs` is a mutable, defensively-copied `list[ToolSpec]`. The underscore
prefixes mark these as SDK-internal: they carry no public-API stability guarantee and can change in a
minor release.

It is accepted because there is no public middleware API yet — `BeforeModelCallEvent`, the closest
public hook, does not carry `tool_specs` and cannot rewrite them — and because vended plugins use the
same seam.

How the package contains it:

- The import is isolated in `_compat.py`, so an upstream rename touches one file. That module is also
  the single adoption point when a public middleware API lands.
- `pyproject.toml` pins `strands-agents` to a conservative range whose upper bound excludes the next
  major release, so a new major SDK cannot silently pull in a moved seam.
- `tests/test_compat.py` asserts, on every SDK version tested in CI, that `InvokeModelStage.Input` is
  importable, that `InvokeModelContext` has a `tool_specs` field, and that `add_middleware` exists on
  the middleware registry. A moved seam fails there rather than at your runtime.

## License

Apache License 2.0. See [LICENSE](LICENSE).
