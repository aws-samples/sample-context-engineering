"""Offline Hermes harness runner — drive each arm's `ContextEngine`(s) with a deterministic mock model.

This mirrors `validation/plugins-langgraph/`'s runner at the level that matters for the comparison —
the same scenario, the same tools, the same deterministic ground truth and accuracy, the same five arms
— but it runs **offline**: no Bedrock, no AWS credentials, a scripted mock model instead of a live LLM,
and fake reranker/matcher so the relevance and graph engines fire without network.

What it actually measures: the message-shaping each Hermes engine does. For every turn the mock model
emits a tool call to the scenario's expected tool, the harness runs the tool, feeds the result through
the engine seams (`on_turn_complete`), and on the next provider request runs `select_context` — exactly
the path a real Hermes loop takes (`conversation_loop.py:1295`, `tool_executor.py:1666`). The call's
input size is estimated with the harness's own `ceil(chars/4)` heuristic, which is what makes the
baseline (no engine, full payloads in history) and the engine arms comparable offline.

The live Bedrock run (filling the real token/accuracy/cost numbers) is a separate, user-authorized step
(Task 20.2); this module is the offline verification (Task 19) and the templated harness.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from . import accuracy, metrics, scenario, tools
from .config import RunConfig
from .metrics import ModelCallRecord, RunCollector

# Fat deterministic answers come from the scenario's own ground truth checks; the mock model emits a
# plausible answer string the deterministic scorer can grade. The scorer matches verbatim figures, so
# the mock answer concatenates the tool result's salient lines — the same text a real model would quote.


class _MockModel:
    """A scripted, deterministic stand-in for the LLM.

    Per turn it: (1) calls the single tool the scenario turn is about, (2) answers by quoting the tool
    result. No randomness, no network — the point is to exercise the engine seams, not to grade a model.
    """

    def __init__(self, tool_registry: dict[str, Callable[..., Any]]) -> None:
        self._tools = tool_registry

    def tool_for(self, turn: scenario.Turn) -> tuple[str, dict[str, Any]] | None:
        """Pick the tool + args this turn exercises, from its label/prompt (deterministic heuristic)."""
        prompt = turn.prompt.lower()
        if "list" in prompt and "account" in prompt:
            return "list_accounts", {}
        if "position" in prompt:
            return "list_investment_positions", {"account_id": "0001/12345-6"}
        if "distribut" in prompt or "allocation" in prompt:
            return "get_portfolio_allocation", {"account_id": "0001/12345-6"}
        if "yield" in prompt or "project" in prompt:
            return "project_yield", {"account_id": "0001/12345-6"}
        if "connector" in prompt or "sync" in prompt or "log" in prompt:
            return "get_connector_status", {"institution": "FinBank Invest"}
        if "documentation" in prompt or "aws" in prompt:
            return "read_aws_documentation", {"url": "https://docs.aws.amazon.com/lambda/latest/dg/welcome.html"}
        return "list_accounts", {}

    def answer(self, tool_name: str, result: str) -> str:
        """Quote the salient (figure-bearing) lines of the tool result as the turn's answer."""
        lines = [ln for ln in result.splitlines() if any(c.isdigit() for c in ln)]
        return " ".join(lines[:12]) if lines else result[:400]


def _estimate_record(turn_index: int, call_index: int, messages: list[dict], tool_specs: list[dict], system: str) -> ModelCallRecord:
    """Build a ModelCallRecord with real char counts of the shaped request (estimates feed the report)."""
    return ModelCallRecord(
        turn_index=turn_index,
        call_index=call_index,
        message_count=len(messages),
        message_chars=len(json.dumps(messages, ensure_ascii=False)),
        tool_spec_count=len(tool_specs),
        tool_spec_chars=len(json.dumps(tool_specs, ensure_ascii=False)),
        system_prompt_chars=len(system or ""),
        tool_names_sent=[s.get("function", {}).get("name", "?") for s in tool_specs if isinstance(s, dict)],
    )


def run_configuration(config: RunConfig, turns: tuple[scenario.Turn, ...], build_engine: Callable[[RunConfig], Any]) -> RunCollector:
    """Replay the scenario for one arm against the mock model, through its Hermes engine(s)."""
    collector = RunCollector(config.name)
    engine = build_engine(config)  # None for baseline
    tool_registry = {t.name: t for t in tools.all_tools()}
    from ._toolshim import convert_to_openai_tool

    base_tool_schemas = [convert_to_openai_tool(t) for t in tool_registry.values()]
    engine_tool_schemas = engine.get_tool_schemas() if engine is not None else []
    all_tool_schemas = base_tool_schemas + engine_tool_schemas
    model = _MockModel(tool_registry)

    history: list[dict[str, Any]] = [{"role": "system", "content": scenario.SYSTEM_PROMPT}]
    run_started = time.perf_counter()

    for index, turn in enumerate(turns):
        turn_started = time.perf_counter()
        record = collector.begin_turn(index, turn.label, turn.prompt)
        history.append({"role": "user", "content": turn.prompt})

        try:
            # --- model call 1: the request after select_context, model emits a tool call ---
            request = _apply_select_context(engine, history, all_tool_schemas)
            collector.calls.append(_estimate_record(index, collector._call_index, request, all_tool_schemas, scenario.SYSTEM_PROMPT))
            collector._call_index += 1

            picked = model.tool_for(turn)
            tool_name, args = picked
            fn = tool_registry.get(tool_name)
            result = str(fn(**args)) if fn is not None else ""
            call_id = f"t{index}_c0"
            history.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": call_id, "type": "function", "function": {"name": tool_name, "arguments": json.dumps(args)}}]})
            history.append({"role": "tool", "tool_call_id": call_id, "content": result})
            record.tool_calls.append(tool_name)

            # engine observes the finished tool exchange
            if engine is not None:
                engine.on_turn_complete(history)

            # --- model call 2: request after select_context (engine folds/rewrites), model answers ---
            request2 = _apply_select_context(engine, history, all_tool_schemas)
            collector.calls.append(_estimate_record(index, collector._call_index, request2, all_tool_schemas, scenario.SYSTEM_PROMPT))
            collector._call_index += 1

            answer = model.answer(tool_name, result)
            history.append({"role": "assistant", "content": answer})
            record.response_text = answer
            record.response_chars = len(answer)
        except Exception as error:  # noqa: BLE001 — a failing turn is recorded, not fatal.
            record.error = f"{type(error).__name__}: {error}"
            collector.errors.append(record.error)
        finally:
            record.turn_seconds = time.perf_counter() - turn_started

    collector.wall_seconds = time.perf_counter() - run_started
    collector.accuracy = accuracy.score_run([t.to_dict() for t in collector.turns])
    collector.plugin_counters = _plugin_counters(engine)
    return collector


def _apply_select_context(engine: Any, history: list[dict], tool_schemas: list[dict]) -> list[dict]:
    """Return the request the model would see — the engine's select_context output, or history as-is."""
    if engine is None:
        return list(history)
    try:
        out = engine.select_context(list(history))
    except Exception:  # noqa: BLE001 — fail open, matching the ABC contract.
        out = None
    return out if out is not None else list(history)


def _plugin_counters(engine: Any) -> dict[str, Any]:
    """Best-effort engine evidence for the report's plugin-counters block (offline: minimal)."""
    if engine is None:
        return {}
    return {"engine": getattr(engine, "name", "?")}


def run_all(
    names: list[str],
    turns: tuple[scenario.Turn, ...],
    build_engine: Callable[[RunConfig], Any],
    run_configs: dict[str, RunConfig],
    repeats: int = 1,
) -> dict[str, list[RunCollector]]:
    """Run every named arm `repeats` times. Returns name -> list of collectors."""
    out: dict[str, list[RunCollector]] = {}
    for name in names:
        config = run_configs[name]
        out[name] = [run_configuration(config, turns, build_engine) for _ in range(repeats)]
    return out
