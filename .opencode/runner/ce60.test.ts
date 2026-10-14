/**
 * Local tests for the CE-60 host driver. These are offline and start no OpenCode process and make no
 * model call: they exercise argument parsing, the live-config guards, turn resolution and the 60-turn
 * invariant, the strictly-sequential (never parallel) turn loop, the per-arm environment, response
 * parsing and the dry-run plan. Run with `bun test` from `.opencode/runner`.
 */

import { afterAll, describe, expect, test } from "bun:test";
import { spawn } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  attachSessionTelemetry,
  attributeSessionRetrieval,
  buildDiagnosticArtifact,
  classifyTimeoutState,
  inheritedOpencodeEnvKeys,
  isPidAlive,
  listDescendantPids,
  promptToolsMap,
  readSessionStatus,
  reapProcessTree,
  verifyAbort,
  type Env,
  type PythonResult,
  type Turn,
  type TurnObservation,
  buildArmEnv,
  buildFinalizePayload,
  collectSessionTelemetry,
  computeGraphDurability,
  decisionFileName,
  graphEvidencePathForRun,
  DEFAULT_CHUNK_TOKENS,
  DEFAULT_MAX_RESULT_TOKENS,
  DEFAULT_PREVIEW_TOKENS,
  DEFAULT_RELEVANCE_THRESHOLD,
  DEFAULT_RERANK_MODEL_ID,
  EXPECTED_TURN_COUNT,
  extractModelInputTokens,
  extractToolSpecMetrics,
  main,
  METRICS_FILE_ENV,
  metricsPathForRun,
  newRetrievalTracker,
  normalizeTurns,
  parseArgs,
  parsePromptResponse,
  parseSessionMessagesResponse,
  planSummary,
  pluginOptions,
  promptSystem,
  RETRIEVE_CONTEXT_TOOL,
  runDiagnosticFixture,
  runTurnsSequential,
  summarizeMetricsFile,
  summarizeObservability,
  summarizeToolSpecMetrics,
  SYSTEM_PROMPT_MAX_CHARS,
  systemPromptDigest,
  timedOutObservation,
  validateCalibrationOptions,
  validateLiveConfig,
} from "./ce60.ts";

const tmps: string[] = [];
afterAll(() => {
  for (const dir of tmps) rmSync(dir, { recursive: true, force: true });
});

function sixtyTurns(): Turn[] {
  return Array.from({ length: EXPECTED_TURN_COUNT }, (_, i) => ({
    turn_id: `t${i.toString().padStart(2, "0")}`,
    prompt: `prompt ${i}`,
  }));
}

function liveOpts() {
  const statPlan = join(mkTmp(), "stat-plan.json");
  writeFileSync(statPlan, JSON.stringify({ confidence_level: 0.95 }));
  return parseArgs([
    "--execute-live",
    "--model",
    "anthropic/claude-x",
    "--approval-id",
    "appr-1",
    "--budget-ceiling",
    "10",
    "--budget-unit",
    "USD",
    "--stat-plan",
    statPlan,
    "--run-id",
    "run-1",
  ]);
}

function mkTmp(): string {
  const dir = mkdtempSync(join(tmpdir(), "ce60-test-"));
  tmps.push(dir);
  return dir;
}

const fullEnv: Env = { AWS_PROFILE: "p", AWS_REGION: "us-west-2" };

describe("parseArgs", () => {
  test("defaults to a dry run of both arms with bridge calibration defaults", () => {
    const o = parseArgs([]);
    expect(o.dryRun).toBe(true);
    expect(o.executeLive).toBe(false);
    expect(o.arm).toBe("both");
    expect(o.outputDir).toBe("ce60-out");
    expect(o.startupTimeoutMs).toBe(60000);
    expect(o.maxResultTokens).toBe(DEFAULT_MAX_RESULT_TOKENS);
    expect(o.previewTokens).toBe(DEFAULT_PREVIEW_TOKENS);
    expect(o.chunkTokens).toBe(DEFAULT_CHUNK_TOKENS);
    expect(o.relevanceThreshold).toBe(DEFAULT_RELEVANCE_THRESHOLD);
  });

  test("--execute-live flips dry run off", () => {
    const o = parseArgs(["--execute-live"]);
    expect(o.executeLive).toBe(true);
    expect(o.dryRun).toBe(false);
  });

  test("rejects an unknown arm and unknown flags", () => {
    expect(() => parseArgs(["--arm", "A_plus_D"])).toThrow(/--arm must be/);
    expect(() => parseArgs(["--nope"])).toThrow(/unknown argument/);
  });

  test("captures all live flags", () => {
    const o = liveOpts();
    expect(o.model).toBe("anthropic/claude-x");
    expect(o.approvalId).toBe("appr-1");
    expect(o.budgetUnit).toBe("USD");
    expect(o.runId).toBe("run-1");
  });

  test("parses explicit arm-A calibration flags", () => {
    const o = parseArgs([
      "--max-result-tokens",
      "4000",
      "--preview-tokens",
      "800",
      "--chunk-tokens",
      "500",
      "--relevance-threshold",
      "0.02",
    ]);
    expect(o.maxResultTokens).toBe(4000);
    expect(o.previewTokens).toBe(800);
    expect(o.chunkTokens).toBe(500);
    expect(o.relevanceThreshold).toBe(0.02);
  });
});

describe("validateLiveConfig", () => {
  test("reports every missing input on an empty configuration", () => {
    const errors = validateLiveConfig(parseArgs(["--execute-live"]), {});
    expect(errors.some((e) => e.includes("AWS_PROFILE"))).toBe(true);
    expect(errors.some((e) => e.includes("AWS_REGION"))).toBe(true);
    expect(errors.some((e) => e.includes("--model"))).toBe(true);
    expect(errors.some((e) => e.includes("--stat-plan"))).toBe(true);
  });

  test("accepts a complete, valid configuration", () => {
    expect(validateLiveConfig(liveOpts(), fullEnv)).toEqual([]);
  });

  test("rejects invalid calibration ranges and token types", () => {
    const o = parseArgs([
      "--max-result-tokens",
      "0",
      "--preview-tokens",
      "1.5",
      "--chunk-tokens",
      "not-a-number",
      "--relevance-threshold",
      "1.1",
    ]);
    const errors = validateCalibrationOptions(o);
    expect(errors).toEqual(
      expect.arrayContaining([
        "--max-result-tokens must be a positive integer",
        "--preview-tokens must be a positive integer",
        "--chunk-tokens must be a positive integer",
        "--relevance-threshold must be a number within [0, 1]",
      ]),
    );
  });
});

describe("normalizeTurns", () => {
  test("accepts exactly 60 turns, array or wrapped", () => {
    expect(normalizeTurns(sixtyTurns())).toHaveLength(60);
    expect(normalizeTurns({ turns: sixtyTurns() })).toHaveLength(60);
  });

  test("never truncates: a wrong count fails closed", () => {
    expect(() => normalizeTurns(sixtyTurns().slice(0, 59))).toThrow(/exactly 60/);
    expect(() => normalizeTurns([...sixtyTurns(), { turn_id: "t00", prompt: "dup" }])).toThrow(
      /exactly 60|unique/,
    );
  });

  test("rejects an empty prompt", () => {
    const turns = sixtyTurns();
    turns[3].prompt = "   ";
    expect(() => normalizeTurns(turns)).toThrow(/empty prompt/);
  });
});

describe("runTurnsSequential", () => {
  test("runs one turn at a time, in order, never in parallel", async () => {
    const turns = sixtyTurns();
    let inFlight = 0;
    let maxInFlight = 0;
    const seen: string[] = [];
    const send = async (turn: Turn): Promise<TurnObservation> => {
      inFlight++;
      maxInFlight = Math.max(maxInFlight, inFlight);
      await new Promise((r) => setTimeout(r, 1));
      seen.push(turn.turn_id);
      inFlight--;
      return parsePromptResponse(turn.turn_id, { info: {}, parts: [{ type: "text", text: "ok" }] });
    };
    const observations = await runTurnsSequential(turns, send);
    expect(observations).toHaveLength(60);
    expect(maxInFlight).toBe(1);
    expect(seen).toEqual(turns.map((t) => t.turn_id));
  });
});

describe("runArmLive", () => {
  test("collects observations before reading metrics and building the final payload", () => {
    const source = readFileSync(join(import.meta.dir, "ce60.ts"), "utf8");
    const start = source.indexOf("export async function runArmLive(");
    const end = source.indexOf("\nconst TIMEOUT", start);
    const body = source.slice(start, end);
    const metricsFile = body.indexOf("const metricsFile = metricsPathForRun(opts, arm, runId, env);");
    const metricsDirectory = body.indexOf("mkdirSync(dirname(metricsFile), { recursive: true });");
    const startOpencode = body.indexOf("const started = await startOpencode");
    const send = body.indexOf("const send: SendTurn = async");
    const observations = body.indexOf("const observations = await runTurnsSequential(turnsToRun, send);");
    const metrics = body.indexOf("const metricsSummary = summarizeMetricsFile");
    const payload = body.indexOf("const payload = buildFinalizePayload");

    expect(start).toBeGreaterThanOrEqual(0);
    expect(end).toBeGreaterThan(start);
    expect(metricsFile).toBeGreaterThanOrEqual(0);
    expect(metricsDirectory).toBeGreaterThan(metricsFile);
    expect(metricsDirectory).toBeLessThan(startOpencode);
    expect(send).toBeGreaterThanOrEqual(0);
    expect(observations).toBeGreaterThan(send);
    expect(observations).toBeLessThan(metrics);
    expect(observations).toBeLessThan(payload);
  });
});

describe("buildArmEnv", () => {
  test("arm A carries the live context-engineering settings with defaults", () => {
    const env = buildArmEnv("A", { AWS_PROFILE: "p" }, liveOpts());
    expect(env.CONTEXT_EXECUTION_MODE).toBe("live");
    expect(env.CONTEXT_APPROVAL_ID).toBe("appr-1");
    expect(env.CONTEXT_BUDGET_CEILING).toBe("10");
    expect(env.CONTEXT_BUDGET_UNIT).toBe("USD");
    expect(env.CONTEXT_EXECUTION_ALLOWED_MODELS).toBe("anthropic/claude-x");
    expect(env.CONTEXT_EXECUTION_MODEL_ID).toBe("anthropic/claude-x");
    expect(env.CONTEXT_RERANK_MODEL_ID).toBe(DEFAULT_RERANK_MODEL_ID);
    expect(env.CONTEXT_MAX_RESULT_TOKENS).toBe(String(DEFAULT_MAX_RESULT_TOKENS));
    expect(env.CONTEXT_PREVIEW_TOKENS).toBe(String(DEFAULT_PREVIEW_TOKENS));
    expect(env.CONTEXT_CHUNK_TOKENS).toBe(String(DEFAULT_CHUNK_TOKENS));
    expect(env.CONTEXT_RELEVANCE_THRESHOLD).toBe(String(DEFAULT_RELEVANCE_THRESHOLD));
    expect(env.AWS_PROFILE).toBe("p");
  });

  test("arm A propagates explicit calibration values", () => {
    const opts = parseArgs([
      "--max-result-tokens",
      "4000",
      "--preview-tokens",
      "800",
      "--chunk-tokens",
      "500",
      "--relevance-threshold",
      "0.02",
    ]);
    const env = buildArmEnv("A", {}, opts);
    expect(env.CONTEXT_MAX_RESULT_TOKENS).toBe("4000");
    expect(env.CONTEXT_PREVIEW_TOKENS).toBe("800");
    expect(env.CONTEXT_CHUNK_TOKENS).toBe("500");
    expect(env.CONTEXT_RELEVANCE_THRESHOLD).toBe("0.02");
  });

  test("arm A does not propagate an execution model during dry-run configuration", () => {
    const opts = parseArgs(["--model", "anthropic/claude-x"]);
    const env = buildArmEnv("A", {}, opts);
    expect(env.CONTEXT_EXECUTION_MODEL_ID).toBeUndefined();
  });

  test("baseline strips inherited execution opt-in and calibration", () => {
    const env = buildArmEnv(
      "baseline",
      {
        AWS_PROFILE: "p",
        CONTEXT_EXECUTION_MODE: "live",
        CONTEXT_EXECUTION_MODEL_ID: "stale-agent-model",
        CONTEXT_MAX_RESULT_TOKENS: "1",
        CONTEXT_PREVIEW_TOKENS: "2",
        CONTEXT_CHUNK_TOKENS: "3",
        CONTEXT_RELEVANCE_THRESHOLD: "0.1",
      },
      liveOpts(),
    );
    expect(env.CONTEXT_EXECUTION_MODE).toBeUndefined();
    expect(env.CONTEXT_EXECUTION_MODEL_ID).toBeUndefined();
    expect(env.CONTEXT_MAX_RESULT_TOKENS).toBeUndefined();
    expect(env.CONTEXT_PREVIEW_TOKENS).toBeUndefined();
    expect(env.CONTEXT_CHUNK_TOKENS).toBeUndefined();
    expect(env.CONTEXT_RELEVANCE_THRESHOLD).toBeUndefined();
  });
});

describe("parsePromptResponse", () => {
  test("extracts tokens, latency, tool calls and marks a well-formed completion", () => {
    const obs = parsePromptResponse("t01", {
      info: { tokens: { input: 1234 }, time: { created: 100, completed: 180 } },
      parts: [
        { type: "text", text: "the answer" },
        { type: "tool", tool: "read" },
      ],
    });
    expect(obs.host_signal).toBe("success");
    expect(obs.matcher_passed).toBe(true);
    expect(obs.model_input).toBe(1234);
    expect(obs.latency_ms).toBe(80);
    expect(obs.tool_calls).toEqual(["read"]);
    expect(obs.retrieval_status).toBe("host_not_exposed");
  });

  test("an error on the message yields an error signal and no completion", () => {
    const obs = parsePromptResponse("t02", {
      info: { error: { name: "MessageOutputLengthError" }, tokens: {} },
      parts: [],
    });
    expect(obs.host_signal).toBe("error");
    expect(obs.matcher_passed).toBe(false);
    expect(obs.error_code).toBe("MessageOutputLengthError");
  });

  test("maps unknown assistant lifecycle enums to other without retaining error payload", () => {
    const obs = parsePromptResponse("t03b", {
      info: { role: "assistant", finish: "future-finish", error: { name: "SECRET-ERROR-NAME" } },
      parts: [{ type: "text", text: "ok" }],
    });
    expect(obs.assistant?.finish).toBe("other");
    expect(obs.assistant?.error_name).toBe("other");
    expect(JSON.stringify(obs.assistant)).not.toContain("SECRET-ERROR-NAME");
  });

  test("empty text is not a completed turn", () => {
    const obs = parsePromptResponse("t03", { info: {}, parts: [{ type: "text", text: "   " }] });
    expect(obs.matcher_passed).toBe(false);
  });

  test("sums input and cache read/write into model_input, ignoring output/reasoning", () => {
    const obs = parsePromptResponse("t04", {
      info: {
        tokens: { input: 1000, output: 500, reasoning: 200, cache: { read: 300, write: 50 } },
      },
      parts: [{ type: "text", text: "ok" }],
    });
    // input (1000) + cache.read (300) + cache.write (50) = 1350; output/reasoning never added.
    expect(obs.model_input).toBe(1350);
  });

  test("model_input is null (host_not_exposed) when no input-side token field is present", () => {
    const obs = parsePromptResponse("t05", {
      info: { tokens: { output: 500, reasoning: 200 } },
      parts: [{ type: "text", text: "ok" }],
    });
    expect(obs.model_input).toBeNull();
  });

  test("retrieval is host_not_exposed with null counts by default (baseline)", () => {
    const obs = parsePromptResponse("t06", {
      info: {},
      parts: [{ type: "tool", tool: RETRIEVE_CONTEXT_TOOL, state: { status: "completed" } }],
    });
    expect(obs.retrieval_status).toBe("host_not_exposed");
    expect(obs.retrieval_attempts).toBeNull();
    expect(obs.retrieval_successes).toBeNull();
  });

  test("counts retrieve_context attempts and completed successes when exposed (arm A)", () => {
    const obs = parsePromptResponse(
      "t07",
      {
        info: {},
        parts: [
          { type: "tool", tool: RETRIEVE_CONTEXT_TOOL, state: { status: "completed" } },
          { type: "tool", tool: RETRIEVE_CONTEXT_TOOL, state: { status: "error" } },
          { type: "tool", tool: "read", state: { status: "completed" } },
          { type: "text", text: "done" },
        ],
      },
      { retrievalExposed: true },
    );
    expect(obs.retrieval_status).toBe("exposed");
    expect(obs.retrieval_attempts).toBe(2);
    expect(obs.retrieval_successes).toBe(1);
    // Non-retrieval tool calls are still recorded in tool_calls but not counted as retrieval.
    expect(obs.tool_calls).toEqual([RETRIEVE_CONTEXT_TOOL, RETRIEVE_CONTEXT_TOOL, "read"]);
  });

  test("exposes a truthful zero retrieval count for an arm-A turn with no retrieval calls", () => {
    const obs = parsePromptResponse(
      "t08",
      { info: {}, parts: [{ type: "text", text: "no retrieval here" }] },
      { retrievalExposed: true },
    );
    expect(obs.retrieval_status).toBe("exposed");
    expect(obs.retrieval_attempts).toBe(0);
    expect(obs.retrieval_successes).toBe(0);
  });
});

describe("public session telemetry", () => {
  test("discards message content, tool arguments and tool output", () => {
    const response = {
      data: [
        {
          info: {
            role: "user",
            parts: [{ type: "text", text: "PROMPT-CANARY" }],
          },
          parts: [
            { type: "text", text: "RESPONSE-CANARY" },
            {
              type: "tool",
              tool: "read",
              state: { input: { secret: "ARGS-CANARY" }, output: "PAYLOAD-CANARY" },
            },
          ],
        },
        {
          info: { role: "assistant" },
          parts: [
            {
              type: "tool",
              tool: "mcp_ce60-fixture_search",
              state: { input: { query: "ARGS-CANARY" }, output: "PAYLOAD-CANARY" },
            },
            { type: "compaction", summary: "CONTENT-CANARY" },
          ],
        },
      ],
    };

    const telemetry = parseSessionMessagesResponse(response);
    expect(telemetry).toEqual({
      telemetry_status: "ok",
      message_count: 2,
      tool_call_count: 2,
      mcp_tool_call_count: 1,
      session_compacted_observed: true,
      last_message_role: "assistant",
      assistant: {
        present: true,
        completed: false,
        finish: null,
        error_name: null,
        error_status_code: null,
        error_retryable: null,
        step_start_count: 0,
        step_finish_count: 0,
        last_part_types: ["tool", "compaction"],
        last_tool: null,
        last_step_tokens: null,
      },
      tool_names: ["read", "mcp_ce60-fixture_search"],
      retrieval_call_count: 0,
      retrieval_success_count: 0,
    });
    const serialized = JSON.stringify(telemetry);
    expect(serialized).not.toContain("PROMPT-CANARY");
    expect(serialized).not.toContain("RESPONSE-CANARY");
    expect(serialized).not.toContain("ARGS-CANARY");
    expect(serialized).not.toContain("PAYLOAD-CANARY");
    expect(serialized).not.toContain("CONTENT-CANARY");
  });

  test("counts tool and MCP calls from the public messages response", async () => {
    const response = {
      data: [
        {
          info: { role: "assistant" },
          parts: [
            { type: "tool", tool: "read" },
            { type: "tool", tool: "ce60-fixture_list_accounts" },
            { type: "tool", tool: "mcp_ce60-fixture_search" },
          ],
        },
      ],
    };
    const telemetry = await collectSessionTelemetry(
      { session: { messages: async () => response } },
      "session-1",
    );
    expect(telemetry.message_count).toBe(1);
    expect(telemetry.tool_call_count).toBe(3);
    expect(telemetry.mcp_tool_call_count).toBe(2);
    expect(telemetry.tool_names).toEqual([
      "read",
      "ce60-fixture_list_accounts",
      "mcp_ce60-fixture_search",
    ]);
  });

  test("records F002 timeout identity and counts collected before abort", () => {
    const observation = timedOutObservation(
      "F002",
      true,
      {
        telemetry_status: "ok",
        message_count: 8,
        tool_call_count: 4,
        mcp_tool_call_count: 2,
        session_compacted_observed: false,
        tool_names: ["read", "mcp_ce60-fixture_search"],
      },
      600000,
    );
    expect(observation.turn_id).toBe("F002");
    expect(observation.timeout).toBe(true);
    expect(observation.error_code).toBe("TURN_TIMEOUT");
    expect(observation.message_count).toBe(8);
    expect(observation.tool_call_count).toBe(4);
    expect(observation.mcp_tool_call_count).toBe(2);
    expect("model_request_started" in observation).toBe(false);
    expect("model_request_completed" in observation).toBe(false);
    expect(observation.retrieval_status).toBe("collection_failed");
    expect(observation.retrieval_attempts).toBeNull();
    expect(observation.retrieval_successes).toBeNull();

    const summary = summarizeObservability([observation]);
    expect(summary.observability_schema_version).toBe(2);
    expect(summary.telemetry_status).toBe("ok");
    expect(summary.session_message_count_final).toBe(8);
    expect(summary.session_tool_call_count_final).toBe(4);
    expect(summary.session_mcp_tool_call_count_final).toBe(2);
    expect(summary.delta_message_count_sum).toBe(8);
    expect(summary.delta_tool_call_count_sum).toBe(4);
    expect(summary.delta_mcp_tool_call_count_sum).toBe(2);
    expect(summary.delta_tool_call_count_max).toBe(4);
    expect(summary.snapshot_gap_turn_ids).toEqual([]);
    expect(summary.snapshot_regression_turn_ids).toEqual([]);
    expect(summary.compaction_status).toBe("unavailable");
    expect(summary.compaction_reason).toBe("event_channel_not_observed");
    expect(summary.timeout_count).toBe(1);
    expect(summary.timeout_turn_ids).toEqual(["F002"]);
    expect(summary.tool_names).toEqual(["mcp_ce60-fixture_search", "read"]);
  });

  test("converts cumulative snapshots into deltas and keeps the final snapshot explicit", () => {
    const observations = [1, 3, 3, 7].map((count, index) =>
      attachSessionTelemetry(
        parsePromptResponse(`snapshot-${index}`, { info: {}, parts: [{ type: "text", text: "ok" }] }),
        {
          telemetry_status: "ok",
          message_count: count,
          tool_call_count: count,
          mcp_tool_call_count: count,
          session_compacted_observed: false,
          tool_names: [],
        },
      ),
    );
    const summary = summarizeObservability(observations);
    expect(summary.session_message_count_final).toBe(7);
    expect(summary.delta_message_count_sum).toBe(7);
    expect(summary.delta_tool_call_count_sum).toBe(7);
    expect(summary.delta_mcp_tool_call_count_sum).toBe(7);
    expect(summary.delta_tool_call_count_max).toBe(4);
    expect(summary.snapshot_gap_turn_ids).toEqual([]);
    expect(summary.snapshot_regression_turn_ids).toEqual([]);
  });

  test("invalid snapshot gaps do not become zero deltas", () => {
    const first = attachSessionTelemetry(
      parsePromptResponse("snapshot-1", { info: {}, parts: [{ type: "text", text: "ok" }] }),
      {
        telemetry_status: "ok",
        message_count: 1,
        tool_call_count: 1,
        mcp_tool_call_count: 1,
        session_compacted_observed: false,
        tool_names: [],
      },
    );
    const missing = attachSessionTelemetry(
      parsePromptResponse("snapshot-2", { info: {}, parts: [{ type: "text", text: "ok" }] }),
      {
        telemetry_status: "error",
        message_count: null,
        tool_call_count: null,
        mcp_tool_call_count: null,
        session_compacted_observed: null,
        tool_names: [],
      },
    );
    const recovered = attachSessionTelemetry(
      parsePromptResponse("snapshot-3", { info: {}, parts: [{ type: "text", text: "ok" }] }),
      {
        telemetry_status: "ok",
        message_count: 3,
        tool_call_count: 3,
        mcp_tool_call_count: 3,
        session_compacted_observed: false,
        tool_names: [],
      },
    );
    const summary = summarizeObservability([first, missing, recovered]);
    expect(summary.delta_message_count_sum).toBeNull();
    expect(summary.snapshot_gap_turn_ids).toEqual(["snapshot-2"]);
  });

  test("keeps unavailable session counts null instead of inventing zero", () => {
    const telemetry = parseSessionMessagesResponse({ error: { message: "SECRET-ERROR" } });
    expect(telemetry.message_count).toBeNull();
    expect(telemetry.tool_call_count).toBeNull();
    expect(telemetry.mcp_tool_call_count).toBeNull();
    expect(telemetry.session_compacted_observed).toBeNull();
    expect(telemetry.telemetry_status).toBe("error");
    expect(JSON.stringify(telemetry)).not.toContain("SECRET-ERROR");
  });
});

describe("extractModelInputTokens", () => {
  test("input only", () => {
    expect(extractModelInputTokens({ input: 1234 })).toBe(1234);
  });

  test("input plus cache read and write", () => {
    expect(extractModelInputTokens({ input: 100, cache: { read: 20, write: 5 } })).toBe(125);
  });

  test("cache only, no uncached input", () => {
    expect(extractModelInputTokens({ cache: { read: 7, write: 3 } })).toBe(10);
  });

  test("no input-side field yields null, never zero", () => {
    expect(extractModelInputTokens({ output: 9, reasoning: 4 })).toBeNull();
    expect(extractModelInputTokens({})).toBeNull();
  });

  test("non-object yields null", () => {
    expect(extractModelInputTokens(undefined)).toBeNull();
    expect(extractModelInputTokens(null)).toBeNull();
    expect(extractModelInputTokens(42)).toBeNull();
  });
});

describe("buildFinalizePayload", () => {
  test("offline mode carries tuning and no live block", () => {
    const payload = buildFinalizePayload("baseline", "run-1-baseline", parseArgs([]), "1.18.34", [], "offline");
    expect(payload.mode).toBe("offline");
    expect(payload.live).toBeUndefined();
    expect(payload.host).toBe("opencode");
    expect(payload.tuning).toEqual({
      max_result_tokens: DEFAULT_MAX_RESULT_TOKENS,
      preview_tokens: DEFAULT_PREVIEW_TOKENS,
      chunk_tokens: DEFAULT_CHUNK_TOKENS,
      relevance_threshold: DEFAULT_RELEVANCE_THRESHOLD,
    });
    expect((payload.host_provenance as Record<string, unknown>).tool_spec_tokens_status).toBe("host_not_exposed");
  });

  test("live mode binds admission, budget, price, stat plan and safe host tuning provenance", () => {
    const opts = parseArgs([
      ...[
        "--execute-live",
        "--model",
        "anthropic/claude-x",
        "--approval-id",
        "appr-1",
        "--budget-ceiling",
        "10",
        "--budget-unit",
        "USD",
        "--stat-plan",
        join(mkTmp(), "stat-plan.json"),
        "--run-id",
        "run-1",
      ],
      "--max-result-tokens",
      "4000",
      "--preview-tokens",
      "800",
      "--chunk-tokens",
      "500",
      "--relevance-threshold",
      "0.02",
    ]);
    writeFileSync(opts.statPlan as string, JSON.stringify({ confidence_level: 0.95 }));
    const payload = buildFinalizePayload("A", "run-1-A", opts, "1.18.34", [], "live");
    const live = payload.live as Record<string, unknown>;
    const provenance = payload.host_provenance as Record<string, unknown>;
    expect(live.execution_admission_id).toBe("appr-1");
    expect(String(live.admission_policy_digest)).toMatch(/^sha256:[0-9a-f]{64}$/);
    expect(live.budget_ceiling_units).toBe(10);
    expect(String(live.price_provenance)).toContain("appr-1");
    expect(live.stat_plan).toBeDefined();
    expect(payload.tuning).toEqual({
      max_result_tokens: 4000,
      preview_tokens: 800,
      chunk_tokens: 500,
      relevance_threshold: 0.02,
    });
    expect(provenance.tuning).toEqual(payload.tuning);
    expect(provenance.model_provenance).toEqual({
      execution_model_id: "anthropic/claude-x",
      rerank_model_id: DEFAULT_RERANK_MODEL_ID,
    });
    const baselinePayload = buildFinalizePayload("baseline", "run-1-baseline", opts, "1.18.34", [], "live");
    expect((baselinePayload.host_provenance as Record<string, unknown>).model_provenance).toEqual({
      execution_model_id: null,
      rerank_model_id: DEFAULT_RERANK_MODEL_ID,
    });
    expect(JSON.stringify(provenance)).not.toContain("answer_text");
  });
});

describe("planSummary (dry run)", () => {
  test("redacts secrets, reports the turn count and shows calibration", () => {
    const opts = liveOpts();
    opts.maxResultTokens = 4000;
    opts.previewTokens = 800;
    opts.chunkTokens = 500;
    opts.relevanceThreshold = 0.02;
    const plan = planSummary(opts, fullEnv, sixtyTurns());
    expect(plan.turn_count).toBe(60);
    expect(plan.aws_profile_present).toBe(true);
    expect(plan.approval_id_present).toBe(true);
    expect(plan.tuning).toEqual({
      max_result_tokens: 4000,
      preview_tokens: 800,
      chunk_tokens: 500,
      relevance_threshold: 0.02,
    });
    expect(plan.arm_A_env_keys).toEqual(
      expect.arrayContaining([
        "CONTEXT_MAX_RESULT_TOKENS",
        "CONTEXT_PREVIEW_TOKENS",
        "CONTEXT_CHUNK_TOKENS",
        "CONTEXT_RELEVANCE_THRESHOLD",
        "CONTEXT_EXECUTION_MODEL_ID",
        "CONTEXT_RERANK_MODEL_ID",
      ]),
    );
    // The raw approval id is never surfaced in the plan.
    expect(JSON.stringify(plan)).not.toContain("appr-1");
  });
});

describe("main", () => {
  const fakeTurns: PythonResult = {
    code: 0,
    stdout: JSON.stringify({ turns: sixtyTurns(), source: "scenario" }),
    stderr: "",
  };
  const fakePython = async (): Promise<PythonResult> => fakeTurns;

  test("--help returns 0", async () => {
    expect(await main(["--help"], fullEnv, fakePython)).toBe(0);
  });

  test("dry run resolves 60 turns and returns 0 without going live", async () => {
    expect(await main(["--dry-run"], fullEnv, fakePython)).toBe(0);
  });

  test("live without configuration fails closed with exit 2", async () => {
    expect(await main(["--execute-live"], {}, fakePython)).toBe(2);
  });

  test("a bad argument returns exit 2", async () => {
    expect(await main(["--bogus"], fullEnv, fakePython)).toBe(2);
  });
});


describe("MCP metrics and schema overhead", () => {
  test("parses schema/tool overhead separately without changing model_input", () => {
    const info = {
      tokens: {
        input: 100,
        cache: { read: 20, write: 5 },
        tool_spec_tokens: 33,
        tool_spec_count: 4,
      },
    };
    expect(extractModelInputTokens(info.tokens)).toBe(125);
    expect(extractToolSpecMetrics(info)).toEqual({ tool_spec_tokens: 33, tool_spec_count: 4 });
    const observation = parsePromptResponse(
      "overhead",
      { info, parts: [{ type: "text", text: "ok" }] },
    );
    expect(observation.model_input).toBe(125);
    expect(observation.tool_spec_tokens).toBe(33);
    expect(observation.tool_spec_count).toBe(4);
  });

  test("marks SDK 1.18.34 tool overhead as host_not_exposed when absent", () => {
    const observation = parsePromptResponse("not-exposed", {
      info: { tokens: { input: 100, cache: { read: 20, write: 5 } } },
      parts: [{ type: "text", text: "ok" }],
    });
    expect(observation.tool_spec_tokens).toBeNull();
    expect(observation.tool_spec_count).toBeNull();
    const summary = summarizeToolSpecMetrics([observation]);
    expect(summary.tool_spec_tokens).toBeNull();
    expect(summary.tool_spec_tokens_status).toBe("host_not_exposed");
    expect(summary.tool_spec_count_status).toBe("host_not_exposed");
  });

  test("aggregates only sanitized bridge JSONL fields", () => {
    const path = join(mkTmp(), "metrics.jsonl");
    writeFileSync(
      path,
      [
        JSON.stringify({ event: "metrics_ready", schema_version: 1 }),
        JSON.stringify({
          event: "transform",
          tool_class: "list_accounts",
          operation: "previewed",
          input_bytes: 100,
          output_bytes: 40,
          filtered: true,
          source_already_truncated: false,
          fallback: false,
          rerank_attempt: true,
          rerank_result: "success",
          retrieval_attempts: 1,
          retrieval_successes: 1,
          prompt: "SECRET-PROMPT-CANARY",
        }),
        JSON.stringify({
          event: "retrieval",
          retrieval_attempts: 2,
          retrieval_successes: 1,
          arguments: "SECRET-ARGUMENT-CANARY",
        }),
      ].join("\n") + "\n",
    );
    const summary = summarizeMetricsFile(path, true);
    expect(summary.metrics_status).toBe("exposed");
    expect(summary.previewed_count).toBe(1);
    expect(summary.filtered_count).toBe(1);
    expect(summary.input_bytes).toBe(100);
    expect(summary.output_bytes).toBe(40);
    expect(summary.retrieval_attempts).toBe(2);
    expect(summary.retrieval_successes).toBe(1);
    expect(JSON.stringify(summary)).not.toContain("SECRET-");
  });

  test("isolates metrics paths per arm/run and passes the metrics env to arm A", () => {
    const opts = parseArgs(["--output-dir", "/tmp/ce60-output", "--metrics-file", "/tmp/metrics-{arm}-{run_id}.jsonl"]);
    expect(metricsPathForRun(opts, "A", "run-A", {})).toBe("/tmp/metrics-A-run-A.jsonl");
    const env = buildArmEnv("A", {}, opts, "/tmp/metrics-A-run-A.jsonl");
    expect(env[METRICS_FILE_ENV]).toBe("/tmp/metrics-A-run-A.jsonl");
    expect(buildArmEnv("baseline", env, opts)[METRICS_FILE_ENV]).toBeUndefined();
  });

  test("diagnostic fixture mode delegates an offline bridge request without live execution", async () => {
    let capturedArgs: string[] = [];
    let capturedEnv: Env | undefined;
    const fakePython = async (
      args: string[],
      _cwd: string,
      _stdin: string | null,
      childEnv?: Env,
    ): Promise<PythonResult> => {
      capturedArgs = args;
      capturedEnv = childEnv;
      return {
        code: 0,
        stdout: JSON.stringify({ mode: "diagnostic-fixture", operations: ["skipped"] }),
        stderr: "",
      };
    };
    const opts = parseArgs(["--diagnostic-fixture", "--output-dir", mkTmp()]);
    expect(await runDiagnosticFixture(opts, {}, fakePython)).toBe(0);
    expect(capturedArgs).toEqual(["-m", "context_core.opencode_bridge", "--diagnostic-fixture"]);
    expect(capturedEnv?.[METRICS_FILE_ENV]).toContain("diagnostic-fixture-A-metrics.jsonl");
    expect(capturedEnv?.AWS_PROFILE).toBe("offline-fixture");
  });
});

// --- F002 diagnostics: timeout capture, abort/idle, cleanup, tool policy ------------------------

function assistantState(overrides: Record<string, unknown> = {}) {
  return {
    present: true,
    completed: false,
    finish: null,
    error_name: null,
    error_status_code: null,
    error_retryable: null,
    step_start_count: 1,
    step_finish_count: 0,
    last_part_types: ["step-start", "tool"],
    last_tool: null,
    last_step_tokens: null,
    ...overrides,
  } as Parameters<typeof classifyTimeoutState>[0]["assistant"];
}

describe("classifyTimeoutState", () => {
  test("a pending question tool is blocked_question", () => {
    expect(
      classifyTimeoutState({
        telemetry_status: "ok",
        last_message_role: "assistant",
        assistant: assistantState({ last_tool: { name: "question", status: "running" } }),
      }),
    ).toBe("blocked_question");
  });

  test("pending MCP and host tools are separated", () => {
    expect(
      classifyTimeoutState({
        telemetry_status: "ok",
        assistant: assistantState({ last_tool: { name: "ce60-fixture_read_connector_logs", status: "pending" } }),
      }),
    ).toBe("tool_pending_mcp");
    expect(
      classifyTimeoutState({
        telemetry_status: "ok",
        assistant: assistantState({ last_tool: { name: "bash", status: "running" } }),
      }),
    ).toBe("tool_pending_host");
  });

  test("retry, resolution and stream stalls are distinguished", () => {
    expect(
      classifyTimeoutState({ telemetry_status: "ok", assistant: assistantState({ error_retryable: true }) }),
    ).toBe("provider_retry");
    expect(
      classifyTimeoutState({ telemetry_status: "ok", assistant: assistantState({ completed: true }) }),
    ).toBe("prompt_resolution");
    expect(classifyTimeoutState({ telemetry_status: "ok", assistant: assistantState() })).toBe(
      "model_stream_stall",
    );
    expect(
      classifyTimeoutState({ telemetry_status: "ok", last_message_role: "user", assistant: assistantState() }),
    ).toBe("model_stream_stall");
  });

  test("partial telemetry is never guessed", () => {
    expect(
      classifyTimeoutState({
        telemetry_status: "timeout",
        assistant: assistantState({ last_tool: { name: "question", status: "running" } }),
      }),
    ).toBe("unclassified");
    expect(classifyTimeoutState({ telemetry_status: "ok" })).toBe("unclassified");
  });

  test("timeout state lists stay aligned with timeout_turn_ids", () => {
    const observation = timedOutObservation(
      "F002-filler",
      false,
      {
        telemetry_status: "ok",
        last_message_role: "assistant",
        message_count: 40,
        tool_call_count: 20,
        mcp_tool_call_count: 18,
        session_compacted_observed: false,
        assistant: assistantState({ last_tool: { name: "question", status: "running" } }),
        tool_names: ["question"],
      },
      600000,
    );
    observation.timeout_classification = classifyTimeoutState(observation);
    observation.abort_status = "ok";
    observation.idle_status = "idle";
    const summary = summarizeObservability([observation]);
    expect(summary.timeout_turn_ids).toEqual(["F002-filler"]);
    expect(summary.timeout_classifications).toEqual(["blocked_question"]);
    expect(summary.timeout_last_tool_names).toEqual(["question"]);
    expect(summary.timeout_last_tool_statuses).toEqual(["running"]);
    expect(summary.timeout_assistant_states).toEqual(["incomplete"]);
    expect(summary.timeout_abort_statuses).toEqual(["ok"]);
    expect(summary.timeout_idle_statuses).toEqual(["idle"]);
  });
});

describe("abort and idle verification", () => {
  test("an absent status entry for a known session reads as idle", async () => {
    const client = { session: { abort: async () => ({ data: true }), status: async () => ({ data: {} }) } };
    expect(await readSessionStatus(client, "s1")).toBe("idle");
  });

  test("a successful abort that returns to idle is recorded", async () => {
    let calls = 0;
    const client = {
      session: {
        abort: async () => ({ data: true }),
        status: async () => {
          calls++;
          return { data: calls < 2 ? { s1: { type: "busy" } } : { s1: { type: "idle" } } };
        },
      },
    };
    expect(await verifyAbort(client, "s1", { idleTimeoutMs: 2000, pollMs: 10 })).toEqual({
      abort_status: "ok",
      idle_status: "idle",
    });
  });

  test("an abort error and a session that stays busy are never swallowed", async () => {
    const client = {
      session: {
        abort: async () => ({ error: { name: "NotFound" } }),
        status: async () => ({ data: { s1: { type: "busy" } } }),
      },
    };
    expect(await verifyAbort(client, "s1", { idleTimeoutMs: 50, pollMs: 10 })).toEqual({
      abort_status: "error",
      idle_status: "not_idle",
    });
  });

  test("an abort that hangs is reported as timeout", async () => {
    const client = {
      session: {
        abort: () => new Promise<unknown>(() => undefined),
        status: async () => ({ data: {} }),
      },
    };
    expect(await verifyAbort(client, "s1", { abortTimeoutMs: 20, idleTimeoutMs: 20, pollMs: 5 })).toEqual({
      abort_status: "timeout",
      idle_status: "idle",
    });
  });
});

describe("process-tree cleanup", () => {
  test("reaps the root and its re-parentable descendants", async () => {
    const child = spawn("sh", ["-c", "sleep 30 & sleep 30"], { stdio: "ignore" });
    await new Promise((r) => setTimeout(r, 300));
    const descendants = listDescendantPids(child.pid as number);
    expect(descendants.length).toBeGreaterThanOrEqual(1);
    const result = await reapProcessTree(child, 2000);
    expect(result.sigterm_sent).toBe(true);
    expect(result.residual_process_count).toBe(0);
    for (const pid of descendants) expect(isPidAlive(pid)).toBe(false);
  });
});

describe("declared tool policy and environment hygiene", () => {
  test("--disable-tools builds an arm-identical per-prompt tools map", () => {
    const opts = parseArgs(["--disable-tools", "webfetch, question,question"]);
    expect(opts.disabledTools).toEqual(["question", "webfetch"]);
    expect(promptToolsMap(opts)).toEqual({ question: false, webfetch: false });
    expect(promptToolsMap(parseArgs([]))).toBeUndefined();
    expect(validateCalibrationOptions(parseArgs(["--disable-tools", "bad name"]))).toContain(
      "--disable-tools has an invalid tool name: bad name",
    );
  });

  test("inherited OPENCODE_* overrides are stripped from both arms and reported by name only", () => {
    const base = { AWS_PROFILE: "p", OPENCODE_CONFIG_CONTENT: '{"secret":1}', OPENCODE_CONFIG: "/x" };
    expect(inheritedOpencodeEnvKeys(base)).toEqual(["OPENCODE_CONFIG", "OPENCODE_CONFIG_CONTENT"]);
    for (const arm of ["baseline", "A"] as const) {
      const env = buildArmEnv(arm, base, liveOpts());
      expect(env.OPENCODE_CONFIG_CONTENT).toBeUndefined();
      expect(env.OPENCODE_CONFIG).toBeUndefined();
      expect(env.AWS_PROFILE).toBe("p");
    }
  });

  test("--diagnostic-stop-after is parsed and shown in the dry-run plan", () => {
    const opts = parseArgs(["--diagnostic-stop-after", "t18", "--disable-tools", "question"]);
    const plan = planSummary(opts, fullEnv, sixtyTurns());
    expect(plan.diagnostic_stop_after).toBe("t18");
    expect(plan.turns_to_send).toBe(19);
    expect(plan.prompt_tools_map).toEqual({ question: false });
  });

  test("the driver never rewrites the CE-60 session through summarize/fork/revert/delete", () => {
    const source = readFileSync(join(import.meta.dir, "ce60.ts"), "utf8");
    expect(source).not.toMatch(/session\.(summarize|fork|revert|delete|unrevert)\(/);
    expect(source.match(/session\.create\(/g)?.length).toBe(1);
  });
});

describe("diagnostic artifact", () => {
  test("is diagnostic-only, payload-free and records the declared policy", () => {
    const turns = sixtyTurns();
    const opts = parseArgs(["--diagnostic-stop-after", "t01", "--disable-tools", "question"]);
    opts.model = "amazon-bedrock/anthropic.claude-sonnet-5-5";
    const answered = parsePromptResponse("t00", {
      info: { role: "assistant", time: { created: 1, completed: 2 } },
      parts: [{ type: "text", text: "SECRET-ANSWER-CANARY" }],
    });
    const timedOut = timedOutObservation("t01", false, {
      telemetry_status: "ok",
      last_message_role: "assistant",
      message_count: 4,
      tool_call_count: 1,
      mcp_tool_call_count: 0,
      session_compacted_observed: false,
      assistant: assistantState({ last_tool: { name: "question", status: "running" } }),
      tool_names: ["question"],
    });
    timedOut.timeout_classification = "blocked_question";
    const artifact = buildDiagnosticArtifact({
      arm: "baseline",
      runId: "d1-baseline",
      opts,
      hostVersion: "1.18.34",
      turns,
      observations: [answered, timedOut],
      metricsSummary: {},
      removedEnvKeys: [],
    });
    expect(artifact.evidence_class).toBe("diagnostic_truncated");
    expect(artifact.turns_sent).toBe(2);
    expect(artifact.disabled_tools).toEqual(["question"]);
    expect(String(artifact.turn_order_digest)).toMatch(/^sha256:[0-9a-f]{64}$/);
    const serialized = JSON.stringify(artifact);
    expect(serialized).not.toContain("SECRET-ANSWER-CANARY");
    expect(serialized).not.toContain("answer_text");
    expect(serialized).not.toContain("prompt 0");
  });
});

describe("session-derived retrieval attribution", () => {
  const retrieveCall = (status: string) => ({ type: "tool", tool: RETRIEVE_CONTEXT_TOOL, state: { status } });
  const history = (...assistantTurns: Array<Array<Record<string, unknown>>>) => ({
    data: [
      { info: { role: "user" }, parts: [{ type: "text", text: "PROMPT-CANARY" }] },
      ...assistantTurns.map((parts) => ({ info: { role: "assistant" }, parts })),
    ],
  });
  const answered = (text = "ok") =>
    parsePromptResponse("t", { info: { role: "assistant" }, parts: [{ type: "text", text }] }, { retrievalExposed: true });

  test("parseSessionMessagesResponse counts retrieve_context across every step, not only the last", () => {
    const telemetry = parseSessionMessagesResponse(
      history(
        [retrieveCall("completed"), { type: "tool", tool: "ce60-fixture_list_accounts", state: { status: "completed" } }],
        [retrieveCall("error")],
        [{ type: "text", text: "final answer" }],
      ),
    );
    expect(telemetry.retrieval_call_count).toBe(2);
    expect(telemetry.retrieval_success_count).toBe(1);
  });

  test("a retrieval made in a non-final step is attributed to its turn (the prompt response misses it)", () => {
    const telemetry = parseSessionMessagesResponse(
      history([retrieveCall("completed")], [{ type: "text", text: "final answer" }]),
    );
    const tracker = newRetrievalTracker();
    // The prompt response only holds the final message, so the legacy count is zero.
    expect(answered().retrieval_attempts).toBe(0);
    const obs = attributeSessionRetrieval(attachSessionTelemetry(answered(), telemetry), telemetry, tracker, true);
    expect(obs.retrieval_status).toBe("exposed");
    expect(obs.retrieval_attempts).toBe(1);
    expect(obs.retrieval_successes).toBe(1);
    expect(tracker).toEqual({ calls: 1, successes: 1 });
  });

  test("counts are per turn: the second read only attributes the growth since the first", () => {
    const tracker = newRetrievalTracker();
    const first = parseSessionMessagesResponse(history([retrieveCall("completed")], [{ type: "text", text: "a" }]));
    const second = parseSessionMessagesResponse(
      history([retrieveCall("completed")], [{ type: "text", text: "a" }], [retrieveCall("completed")], [retrieveCall("error")]),
    );
    const turn1 = attributeSessionRetrieval(attachSessionTelemetry(answered(), first), first, tracker, true);
    const turn2 = attributeSessionRetrieval(attachSessionTelemetry(answered(), second), second, tracker, true);
    expect([turn1.retrieval_attempts, turn1.retrieval_successes]).toEqual([1, 1]);
    expect([turn2.retrieval_attempts, turn2.retrieval_successes]).toEqual([2, 1]);
  });

  test("a turn with no retrieval is a truthful zero, not a missing value", () => {
    const telemetry = parseSessionMessagesResponse(history([{ type: "text", text: "plain" }]));
    const obs = attributeSessionRetrieval(attachSessionTelemetry(answered(), telemetry), telemetry, newRetrievalTracker(), true);
    expect(obs.retrieval_attempts).toBe(0);
    expect(obs.retrieval_successes).toBe(0);
    expect(obs.retrieval_status).toBe("exposed");
  });

  test("the baseline arm exposes no retrieval and is returned unchanged", () => {
    const telemetry = parseSessionMessagesResponse(history([retrieveCall("completed")]));
    const baseline = parsePromptResponse("t", { info: {}, parts: [] });
    const obs = attributeSessionRetrieval(attachSessionTelemetry(baseline, telemetry), telemetry, newRetrievalTracker(), false);
    expect(obs.retrieval_status).toBe("host_not_exposed");
    expect(obs.retrieval_attempts).toBeNull();
    expect(obs.retrieval_successes).toBeNull();
  });

  test("an unavailable session read fails closed and the next read attributes the missing growth", () => {
    const tracker = newRetrievalTracker();
    const failedRead = {
      telemetry_status: "timeout" as const,
      message_count: null,
      tool_call_count: null,
      mcp_tool_call_count: null,
      session_compacted_observed: null,
      tool_names: [],
    };
    const lost = attributeSessionRetrieval(attachSessionTelemetry(answered(), failedRead), failedRead, tracker, true);
    expect(lost.retrieval_status).toBe("collection_failed");
    expect(lost.retrieval_attempts).toBeNull();
    expect(lost.retrieval_successes).toBeNull();
    expect(tracker).toEqual({ calls: 0, successes: 0 });

    const recovered = parseSessionMessagesResponse(history([retrieveCall("completed")], [{ type: "text", text: "a" }]));
    const next = attributeSessionRetrieval(attachSessionTelemetry(answered(), recovered), recovered, tracker, true);
    expect(next.retrieval_attempts).toBe(1);
    expect(next.retrieval_successes).toBe(1);
  });

  test("implausible movement (counts going backwards, successes above calls) fails closed without poisoning later turns", () => {
    const tracker = { calls: 3, successes: 3 };
    const shrunk = parseSessionMessagesResponse(history([retrieveCall("completed")]));
    const bad = attributeSessionRetrieval(attachSessionTelemetry(answered(), shrunk), shrunk, tracker, true);
    expect(bad.retrieval_status).toBe("collection_failed");
    expect(tracker).toEqual({ calls: 1, successes: 1 });

    const grown = parseSessionMessagesResponse(history([retrieveCall("completed")], [retrieveCall("completed")]));
    const good = attributeSessionRetrieval(attachSessionTelemetry(answered(), grown), grown, tracker, true);
    expect([good.retrieval_attempts, good.retrieval_successes]).toEqual([1, 1]);
  });

  test("cumulative bookkeeping never reaches the payload and no content is retained", () => {
    const telemetry = parseSessionMessagesResponse(history([retrieveCall("completed")]));
    const obs = attributeSessionRetrieval(attachSessionTelemetry(answered(), telemetry), telemetry, newRetrievalTracker(), true);
    expect(Object.keys(obs)).not.toContain("retrieval_call_count");
    expect(Object.keys(obs)).not.toContain("retrieval_success_count");
    expect(JSON.stringify(obs)).not.toContain("PROMPT-CANARY");
  });

  test("the live runner wires the attribution into the arm-A turn loop", () => {
    const source = readFileSync(join(import.meta.dir, "ce60.ts"), "utf8");
    expect(source).toContain("attributeSessionRetrieval(");
    expect(source).toContain("const retrievalTracker = newRetrievalTracker();");
  });
});

describe("arm-identical system prompt (response-language pin)", () => {
  const PIN = "Always answer in English, even when tool results or data are in another language.";

  test("is off by default and parsed from --system-prompt", () => {
    expect(parseArgs([]).systemPrompt).toBeNull();
    const opts = parseArgs(["--system-prompt", PIN]);
    expect(opts.systemPrompt).toBe(PIN);
    expect(promptSystem(opts)).toBe(PIN);
    expect(promptSystem(parseArgs([]))).toBeUndefined();
  });

  test("rejects an empty or oversized system prompt", () => {
    expect(validateCalibrationOptions(parseArgs(["--system-prompt", "   "]))).toContain(
      `--system-prompt must be 1-${SYSTEM_PROMPT_MAX_CHARS} characters`,
    );
    expect(
      validateCalibrationOptions(parseArgs(["--system-prompt", "x".repeat(SYSTEM_PROMPT_MAX_CHARS + 1)])),
    ).toHaveLength(1);
    expect(validateCalibrationOptions(parseArgs(["--system-prompt", PIN]))).toEqual([]);
  });

  test("only a digest of the text reaches the evidence payload", () => {
    const opts = liveOpts();
    opts.systemPrompt = PIN;
    const payload = buildFinalizePayload("A", "run-1", opts, "1.18.34", [], "live");
    expect(String(payload.system_prompt_digest)).toMatch(/^sha256:[0-9a-f]{64}$/);
    expect(payload.system_prompt_digest).toBe(systemPromptDigest(opts));
    expect(JSON.stringify(payload)).not.toContain(PIN);
    const none = buildFinalizePayload("A", "run-1", liveOpts(), "1.18.34", [], "live");
    expect(none.system_prompt_digest).toBe("");
  });

  test("the dry-run plan shows that a system prompt is set without printing it", () => {
    const opts = parseArgs(["--system-prompt", PIN]);
    const plan = planSummary(opts, {}, sixtyTurns());
    expect(plan.system_prompt_set).toBe(true);
    expect(JSON.stringify(plan)).not.toContain(PIN);
    expect(planSummary(parseArgs([]), {}, sixtyTurns()).system_prompt_set).toBe(false);
  });

  test("the live runner sends the same system text with every prompt of both arms", () => {
    const source = readFileSync(join(import.meta.dir, "ce60.ts"), "utf8");
    expect(source).toContain("const systemText = promptSystem(opts);");
    expect(source).toContain("...(systemText !== undefined ? { system: systemText } : {}),");
  });
});

describe("plugin options in the evidence", () => {
  test("default to the scoped preview with the retrieval cleanup on", () => {
    expect(pluginOptions({})).toBe("preview_style:scoped,retrieval_cleanup:on,plugin_rev:2");
  });

  test("record the switches that restore the old behaviour", () => {
    expect(pluginOptions({ CONTEXT_PREVIEW_STYLE: "legacy", CONTEXT_RETRIEVAL_CLEANUP: "0" })).toBe(
      "preview_style:legacy,retrieval_cleanup:off,plugin_rev:2",
    );
    expect(pluginOptions({ CONTEXT_PREVIEW_STYLE: "weird", CONTEXT_RETRIEVAL_CLEANUP: "OFF" })).toBe(
      "preview_style:scoped,retrieval_cleanup:off,plugin_rev:2",
    );
  });

  test("only arm A records them, and only token characters are used", () => {
    const opts = liveOpts();
    const a = buildFinalizePayload("A", "run-1", opts, "1.18.34", [], "live");
    const baseline = buildFinalizePayload("baseline", "run-1", opts, "1.18.34", [], "live");
    expect(String(a.plugin_options)).toMatch(/^[A-Za-z0-9_.:,-]+$/);
    expect(baseline.plugin_options).toBe("");
  });
});

describe("practice D in the runner (opt-in; the A path is unchanged when it is off)", () => {
  test("plugin options say graph:on only when the switch is on, and are byte-identical to before otherwise", () => {
    expect(pluginOptions({})).toBe("preview_style:scoped,retrieval_cleanup:on,plugin_rev:2");
    expect(pluginOptions({ CONTEXT_GRAPH: "off" })).toBe("preview_style:scoped,retrieval_cleanup:on,plugin_rev:2");
    expect(pluginOptions({ CONTEXT_GRAPH: "on" })).toBe("preview_style:scoped,retrieval_cleanup:on,plugin_rev:2,graph:on");
  });

  test("the in-place policy is recorded with what it keeps, and the reference policy adds nothing", () => {
    const plain = "preview_style:scoped,retrieval_cleanup:on,plugin_rev:2,graph:on";
    expect(pluginOptions({ CONTEXT_GRAPH: "on", CONTEXT_GRAPH_MODE: "reference" })).toBe(plain);
    expect(pluginOptions({ CONTEXT_GRAPH: "on", CONTEXT_GRAPH_MODE: "inplace" })).toBe(`${plain},graph_mode:inplace,graph_keep:0,graph_dialogue:0`);
    expect(
      pluginOptions({ CONTEXT_GRAPH: "on", CONTEXT_GRAPH_MODE: "inplace", CONTEXT_GRAPH_KEEP_RECENT: "1", CONTEXT_GRAPH_KEEP_DIALOGUE: "1" }),
    ).toBe(`${plain},graph_mode:inplace,graph_keep:1,graph_dialogue:1`);
    // Only digits are recorded, so a malformed value cannot put anything else in the evidence (the bridge refuses it).
    expect(
      pluginOptions({ CONTEXT_GRAPH: "on", CONTEXT_GRAPH_MODE: "inplace", CONTEXT_GRAPH_KEEP_RECENT: "x;y", CONTEXT_GRAPH_KEEP_DIALOGUE: "" }),
    ).toBe(`${plain},graph_mode:inplace,graph_keep:0,graph_dialogue:0`);
    // The policy is a graph option: with the graph off it says nothing.
    expect(pluginOptions({ CONTEXT_GRAPH_MODE: "inplace" })).toBe("preview_style:scoped,retrieval_cleanup:on,plugin_rev:2");
  });

  test("arm A passes the policy variables on, and the baseline strips them with the rest of the graph's", () => {
    const base = { CONTEXT_GRAPH: "on", CONTEXT_GRAPH_MODE: "inplace", CONTEXT_GRAPH_KEEP_RECENT: "1", AWS_PROFILE: "p" };
    const a = buildArmEnv("A", base, liveOpts(), undefined, "/tmp/run/A-graph-evidence.jsonl");
    expect(a.CONTEXT_GRAPH_MODE).toBe("inplace");
    expect(a.CONTEXT_GRAPH_KEEP_RECENT).toBe("1");
    const baseline = buildArmEnv("baseline", base, liveOpts());
    expect(Object.keys(baseline).filter((key) => key.startsWith("CONTEXT_GRAPH"))).toEqual([]);
  });

  test("arm A passes the switch and its evidence file; the baseline strips every CONTEXT_GRAPH variable", () => {
    const base = { CONTEXT_GRAPH: "on", CONTEXT_GRAPH_EXPAND_THRESHOLD: "0.7", AWS_PROFILE: "p" };
    const a = buildArmEnv("A", base, liveOpts(), undefined, "/tmp/run/A-graph-evidence.jsonl");
    expect(a.CONTEXT_GRAPH).toBe("on");
    expect(a.CONTEXT_GRAPH_EVIDENCE_FILE).toBe("/tmp/run/A-graph-evidence.jsonl");
    expect(a.CONTEXT_GRAPH_EXPAND_THRESHOLD).toBe("0.7");
    const baseline = buildArmEnv("baseline", base, liveOpts());
    expect(Object.keys(baseline).filter((key) => key.startsWith("CONTEXT_GRAPH"))).toEqual([]);
  });

  test("with the switch off arm A carries no graph variable at all", () => {
    const a = buildArmEnv("A", { AWS_PROFILE: "p" }, liveOpts(), undefined, "/tmp/run/A-graph-evidence.jsonl");
    expect(Object.keys(a).filter((key) => key.startsWith("CONTEXT_GRAPH"))).toEqual([]);
    // An explicit off is passed through as the operator wrote it (the plugin reads it as off) and gets no evidence file.
    const off = buildArmEnv("A", { CONTEXT_GRAPH: "no" }, liveOpts(), undefined, "/tmp/x.jsonl");
    expect(off.CONTEXT_GRAPH).toBe("no");
    expect(off.CONTEXT_GRAPH_EVIDENCE_FILE).toBeUndefined();
  });

  test("a run that also ran D never writes a Gate A decision file", () => {
    expect(decisionFileName({})).toBe("gate-a-decision.json");
    expect(decisionFileName({ CONTEXT_GRAPH: "off" })).toBe("gate-a-decision.json");
    expect(decisionFileName({ CONTEXT_GRAPH: "on" })).toBe("d-slice-comparison.json");
    expect(graphEvidencePathForRun(parseArgs(["--output-dir", "/tmp/out"]), "A")).toBe("/tmp/out/A-graph-evidence.jsonl");
  });

  const message = (id: string, role: string, text: string) => ({ info: { id, role, sessionID: "s" }, parts: [{ type: "text", text }] });

  test("durability: a stored session that reproduces every received digest is request-only", async () => {
    const { digestOf, fingerprintTranscript } = await import("../lib/graph.ts");
    const stored = [message("a", "user", "q1"), message("b", "assistant", "r1"), message("c", "user", "q2")];
    const record = (index: number, upto: number) => ({
      request_index: index,
      session_id: "s",
      operation: index === 2 ? "projected" : "unchanged",
      mode: index === 2 ? "chosen" : "delivered",
      received: {
        count: upto,
        first_id: "a",
        last_id: stored[upto - 1].info.id,
        digest: digestOf(fingerprintTranscript(stored.slice(0, upto))),
      },
      removed: index === 2 ? 1 : 0,
      block_chars: index === 2 ? 40 : 0,
      rejected: null,
    });
    const summary = computeGraphDurability({ s: stored }, [record(1, 1), record(2, 2), record(3, 3)]);
    expect((summary.stored_digests as Record<string, string>)["2"]).toBe(record(2, 2).received.digest);
    expect(summary).toMatchObject({
      requests: 3,
      checked: 3,
      stored_prefix_equal: 3,
      mismatched_requests: [],
      stored_graph_synthetic_messages: 0,
      removed_messages_total: 1,
      block_chars_total: 40,
      operations: { unchanged: 2, projected: 1 },
      modes: { delivered: 2, chosen: 1 },
    });
  });

  test("durability: an edited stored message, a stored graph message and an unknown id are all reported", async () => {
    const { digestOf, fingerprintTranscript } = await import("../lib/graph.ts");
    const original = [message("a", "user", "q1"), message("b", "assistant", "r1")];
    const received = {
      count: 2,
      first_id: "a",
      last_id: "b",
      digest: digestOf(fingerprintTranscript(original)),
    };
    const line = (index: number, overrides: Record<string, unknown> = {}) => ({
      request_index: index,
      session_id: "s",
      operation: "unchanged",
      mode: "chosen",
      received,
      removed: 0,
      block_chars: 0,
      rejected: null,
      ...overrides,
    });
    const edited = [message("a", "user", "q1 EDITED"), message("b", "assistant", "r1"), message("msg_zzzgraphabc", "user", "block")];
    const summary = computeGraphDurability({ s: edited }, [line(1), line(2, { received: { ...received, first_id: "nope" } })]);
    expect(summary.mismatched_requests).toEqual([1, 2]);
    expect(summary.stored_prefix_equal).toBe(0);
    expect(summary.stored_graph_synthetic_messages).toBe(1);
    // A request for a session whose messages could not be read is not counted as checked.
    expect(computeGraphDurability({}, [line(1)]).checked).toBe(0);
  });

  test("durability counts the graph tools the model called, from the stored session only", () => {
    const stored = [
      { info: { id: "a", role: "assistant" }, parts: [{ type: "tool", tool: "expand_card" }, { type: "tool", tool: "find_context" }, { type: "tool", tool: "expand_card" }, { type: "tool", tool: "bash" }] },
    ];
    expect(computeGraphDurability({ s: stored }, []).graph_tool_calls).toEqual({ expand_card: 2, find_context: 1 });
  });
});
