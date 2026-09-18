# strands-progressive-tool-disclosure

Progressive Tool Disclosure for Strands Agents. Instead of sending every tool's full specification on
every model call, the plugin projects the call's tool list down to five ordered blocks: the
`find_tools` search tool, the `always_available` tools, the schemas already exposed and still live by
TTL, the tools the history references, and a ~20-token catalog entry for everything else. A tool's
full schema enters the call when the model searches for it, and leaves again after `ttl_cycles` idle
cycles.

The plugin reads the per-call copy the event loop hands to `InvokeModelStage` and never mutates
`agent.tool_registry.registry` — every tool stays callable regardless of what the call shows.

## Install

```bash
pip install strands-progressive-tool-disclosure
```

One runtime dependency: `strands-agents`. The default index is standard-library only, so the base
install performs no network call.

## Usage

```python
from strands import Agent
from strands_progressive_tool_disclosure import ProgressiveToolDisclosure

agent = Agent(
    tools=[...],
    plugins=[
        ProgressiveToolDisclosure(
            catalog_tokens=20,      # per-tool description budget in the catalog; None = search-only
            ttl_cycles=5,           # cycles an exposed schema survives after its last use
            always_available=[],    # tools that skip discovery and carry full spec every call
        )
    ],
)
```

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `catalog_tokens` | `int >= 1 \| None` | `20` | Token budget of a catalog entry's description. `None` drops the catalog entirely, leaving only the search tool as the hint that other tools exist. |
| `ttl_cycles` | `int >= 1` | `5` | Idle cycles an exposure survives after its last use. |
| `always_available` | `Sequence[str]` | `()` | Names that carry their full specification on every call, skipping the discovery cycle. |
| `index` | `ToolIndex \| None` | `LexicalToolIndex()` | Search implementation. Any object exposing `build` and `search`. |
| `top_k` | `int >= 1` | `3` | How many tools a single search exposes. |
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
