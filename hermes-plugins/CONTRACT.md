# Hermes `ContextEngine` contract (pinned from source)

Pinned from a read-only clone of `NousResearch/hermes-agent` (`pyproject.toml` version `0.0.0`,
`requires-python >=3.11,<3.15`). This note is the implementation reference for the four
`hermes-plugins/` packages; it is not shipped in any wheel.

## 1. The `ContextEngine` ABC — `agent/context_engine.py`

`class ContextEngine(ABC)`. A concrete engine MUST implement the four abstract members and MAY
override the hooks. All methods are **synchronous**.

### Required attributes (plain class/instance attributes, read directly by the host)

```
last_prompt_tokens: int = 0
last_completion_tokens: int = 0
last_total_tokens: int = 0
threshold_tokens: int = 0
context_length: int = 0
compression_count: int = 0
threshold_percent: float = 0.75
protect_first_n: int = 3
protect_last_n: int = 6
emit_automatic_compaction_status: bool = True
```

### Abstract members (MUST implement)

| Member | Signature |
|---|---|
| `name` | `@property -> str` |
| `update_from_response` | `(self, usage: Dict[str, Any]) -> None` |
| `should_compress` | `(self, prompt_tokens: int = None) -> bool` |
| `compress` | `(self, messages, current_tokens=None, focus_topic=None, force=False, memory_context="") -> List[Dict]` |

`usage` always carries `prompt_tokens`/`completion_tokens`/`total_tokens`; optional canonical buckets
`input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_write_tokens`, `reasoning_tokens`.

### Hooks used by this port (default no-op in the base class)

| Member | Signature | Role in port |
|---|---|---|
| `select_context` | `(self, request_messages, *, conversation_messages=None, incoming_message=None, budget_tokens=0) -> List[Dict] \| None` | **per-request message replacement, pre-generation.** Return `None` to leave unchanged. Request-only: MUST NOT be treated as persisted history. The seam for D (projection) and A (history drop) and B (catalog injection). |
| `on_turn_complete` | `(self, messages, usage=None, **kwargs) -> None` | index closed turns. `messages` is a read-only shallow copy. `kwargs` may carry `turn_id`, `task_id`, `api_call_count`, `interrupted`, `failed`, `turn_exit_reason`. |
| `get_tool_schemas` | `(self) -> List[Dict]` | the engine's on-demand tools, merged into the agent tool set. |
| `handle_tool_call` | `(self, name, args, **kwargs) -> str` | **MUST return a JSON string.** `kwargs` carries `messages` (live in-memory list). |
| `on_session_start` | `(self, session_id, **kwargs) -> None` | kwargs may include `hermes_home`, `platform`, `model`. |
| `on_session_end` | `(self, session_id, messages) -> None` | real session boundary only. |
| `on_session_reset` | `(self) -> None` | `/new` or `/reset`. Base resets token counters + `compression_count` to 0 (NOT -1). |
| `get_status` | `(self) -> Dict` | base clamps `-1` sentinel; returns `last_prompt_tokens`, `threshold_tokens`, `context_length`, `usage_percent`, `compression_count`. |
| `clone_for_agent` | `(self) -> ContextEngine` | base = `copy.deepcopy(self)`; override when holding uncopyable state. |
| `update_model` | `(self, model, context_length, base_url="", api_key="", provider="", api_mode="") -> None` | base recomputes `threshold_tokens = int(context_length * threshold_percent)` (imports `agent.context_compressor.resolve_model_threshold`). |

Other base hooks exist (`prune_tool_results_only`, `should_compress_preflight`,
`should_defer_preflight_to_real_usage`, `has_content_to_compress`,
`get_automatic_compaction_status_message`, `should_compress_info`) — safe defaults; the port does not
override them.

## 2. Call sites (verified line numbers in the clone)

- `agent/agent_init.py:2138` — `for _raw_schema in agent.context_compressor.get_tool_schemas():` — the
  **selected engine instance is `agent.context_compressor`**; its tool schemas are merged into the
  agent's catalog at init.
- `agent/conversation_loop.py:1295` — `engine.select_context(request_messages, conversation_messages=..., incoming_message=..., budget_tokens=...)`.
  Runs every turn, **fail-open** (any exception → unmodified request, logged at `:1309`); an invalid
  return (`:1325`) is rejected and the request is left unchanged. Guarded by
  `_engine_overrides_hook(engine, "select_context")` at `:1288` — the hook only runs if the subclass
  **overrides** it (base no-op is skipped).
- `agent/tool_executor.py:1666` — `agent.context_compressor.handle_tool_call(function_name, next_args, messages=messages)`.

## 3. Shipping a context engine as a plugin — `plugins/context_engine/__init__.py`

- A context engine is a **directory** `plugins/context_engine/<name>/` (bundled) or
  `$HERMES_HOME/plugins/<name>/` (user); bundled wins on name collision.
- Discovery is a **text heuristic**: the dir's `__init__.py` must mention `register_context_engine`
  **or** `ContextEngine`.
- Selection: `context.engine: <name>` in `config.yaml` (default `"compressor"`). A user engine needs
  **no** `plugins.enabled` entry to be selectable.
- The engine is extracted from the module by `_loader.instance_from_module` via an `_EngineCollector`
  (`collected_attr="engine"`, `base_cls=ContextEngine`): the module may either
  - **export** a `ContextEngine` subclass/instance (attribute discovered as `engine`), or
  - define `def register(ctx): ctx.register_context_engine(<engine instance>)` — the collector captures
    the instance.
- `plugin.yaml` minimal shape (confirmed from `plugins/browser/*/plugin.yaml`): `name`, `version`,
  `description` (optional `author`, `kind`, and `provides_*` keys exist for other plugin kinds).

### Single-select is a hard constraint

`PluginRegistration.register_context_engine` (`hermes_cli/plugins.py:717`) refuses a second engine:
`if self._manager._context_engine is not None: logger.warning(...); return`. => "all three" MUST be one
composed engine (`hermes-all-three`), not three co-active plugins.

## 4. Message shape

OpenAI chat format: `{"role": "...", "content": "..."}` with assistant `tool_calls`
(`[{"id","type":"function","function":{"name","arguments"}}]`) and tool messages carrying
`tool_call_id`. Provider-specific parts (reasoning, cache-control, opaque replay items) may ride along;
the adapter preserves unknown parts verbatim.

## 5. Dependency pin (Task 1.3)

Clone `pyproject.toml`: `name = "hermes-agent"`, `version = "0.0.0"`, `requires-python = ">=3.11,<3.15"`.

- `hermes-agent` is **not published to PyPI** as a versioned release (version `0.0.0`); it is installed
  from the GitHub source. Our packages therefore declare `hermes-agent` as an **optional** peer in a
  `[project.optional-dependencies] hermes` extra pointing at the git source, NOT a hard
  `install_requires` (a hard dep on an unpublished `0.0.0` would make the wheel uninstallable from an
  index). The runtime hard dependency is `agent-context-core>=0.1.0,<0.2` only; the engine imports
  `agent.context_engine.ContextEngine` lazily inside the module so the package imports without Hermes
  present (tests stub the ABC).
- `requires-python` is set to `>=3.11` to match Hermes (not `>=3.10`).
