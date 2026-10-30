/**
 * CE-60 host driver for OpenCode (verified against the public SDK of @opencode-ai/sdk 1.18.34).
 *
 * This is the real host driver the offline Gate A layer was built to feed. It drives an actual OpenCode
 * server over its PUBLIC SDK — no private API — and never scores or sanitises anything itself: the
 * deterministic oracle, the terminal classification, the sanitiser and the evaluator all live in the
 * Python `opencode_gate_a` package, which this driver shells out to over stdin/stdout.
 *
 * What it does, per arm:
 *   1. starts OpenCode as a persistent child process (`opencode serve`) bound to 127.0.0.1 on an
 *      EPHEMERAL port, with the arm's environment; `baseline` starts in `--pure` mode (no plugins) and
 *      `A` starts normally so `.opencode/plugins/context-engineering.ts` loads;
 *   2. waits for `/global/health` to answer;
 *   3. creates ONE session via `client.session.create`;
 *   4. sends the 60 CE-60 prompts STRICTLY SEQUENTIALLY via `client.session.prompt` — never in
 *      parallel — keeping each raw answer only in memory;
 *   5. hands the in-memory, payload-free per-turn observations to the Python bridge
 *      (`python -m opencode_gate_a.host_driver finalize`), which writes only the manifest, the
 *      sanitized result, the evaluation and an evidence summary;
 *   6. ALWAYS stops the OpenCode process in a `finally`.
 *
 * Safety: `--dry-run` is the DEFAULT. Nothing — no process, no model call — happens without an explicit
 * `--execute-live`, and `--execute-live` additionally demands `AWS_PROFILE`, an AWS region, `--model`,
 * `--approval-id`, `--budget-ceiling`, `--budget-unit`, `--stat-plan` and `--run-id`. Only arms
 * `baseline` and `A` exist here; `A_plus_D` and `native_B` are refused. Practice D is not an arm: with `CONTEXT_GRAPH=on`
 * arm A runs with the context graph on top and the evidence says so (`graph:on` in `plugin_options`); no Gate A decision is
 * written for such a run, because Gate A judges A alone. A raw prompt, answer, tool
 * payload, credential or log is never written to a file by this driver.
 */

import { createOpencodeClient, type OpencodeClient } from "@opencode-ai/sdk";
import { spawn, spawnSync, type ChildProcess } from "node:child_process";
import { createHash } from "node:crypto";
import { digestOf, fingerprintTranscript, parseGraphFlag, parseGraphMode, type GraphMessage } from "../lib/graph.ts";
import { PLUGIN_REVISION } from "../lib/revision.ts";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

// --- Paths -------------------------------------------------------------------------------------

const RUNNER_DIR = dirname(fileURLToPath(import.meta.url));
/** The project root (parent of `.opencode`), used as the OpenCode server working directory. */
export const PROJECT_ROOT = resolve(RUNNER_DIR, "..", "..");
/** The `validation/` directory, where the Python Gate A package is importable. */
export const VALIDATION_DIR = join(PROJECT_ROOT, "validation");

export const ARMS = ["baseline", "A"] as const;
export type Arm = (typeof ARMS)[number];
export const EXPECTED_TURN_COUNT = 60;
export const DEFAULT_RERANK_MODEL_ID = "amazon.rerank-v1:0";
export const DEFAULT_MAX_RESULT_TOKENS = 8000;
export const DEFAULT_PREVIEW_TOKENS = 1000;
export const DEFAULT_CHUNK_TOKENS = 2500;
export const DEFAULT_RELEVANCE_THRESHOLD = 0.5;

/** The exact calibration environment keys that are scoped to arm A. */
export const CALIBRATION_ENV_KEYS = [
  "CONTEXT_MAX_RESULT_TOKENS",
  "CONTEXT_PREVIEW_TOKENS",
  "CONTEXT_CHUNK_TOKENS",
  "CONTEXT_RELEVANCE_THRESHOLD",
] as const;

/** The public retrieval tool arm A registers; its calls are the CE-60 retrieval attempt/success count. */
export const RETRIEVE_CONTEXT_TOOL = "retrieve_context";
/** Environment key carrying the per-arm JSONL metrics path into the bridge. */
export const METRICS_FILE_ENV = "CONTEXT_METRICS_FILE";

/** A process environment map; a local alias so this file needs no @types/node for the NodeJS namespace. */
export type Env = Record<string, string | undefined>;

// --- Types -------------------------------------------------------------------------------------

/** One CE-60 turn: a stable id and the user prompt to send. */
export interface Turn {
  turn_id: string;
  prompt: string;
}

/** Status of one payload-free public session telemetry read. */
export type TelemetryStatus = "ok" | "timeout" | "error" | "not_attempted";

/** Payload-free assistant lifecycle metadata derived from public OpenCode message fields. */
export interface AssistantTelemetry {
  present: boolean | null;
  completed: boolean | null;
  finish: string | null;
  error_name: string | null;
  error_status_code: number | null;
  error_retryable: boolean | null;
  step_start_count: number | null;
  step_finish_count: number | null;
  last_part_types: string[];
  last_tool: { name: string; status: string } | null;
  last_step_tokens: { input: number | null; cache_read: number | null; cache_write: number | null } | null;
}

/** Payload-free classes for a turn that hit the wall timeout. */
export type TimeoutClassification =
  | "blocked_question"
  | "tool_pending_mcp"
  | "tool_pending_host"
  | "provider_retry"
  | "prompt_resolution"
  | "model_stream_stall"
  | "unclassified";

/** Counts and flags derived from the public `session.messages` response only. */
export interface SessionTelemetry {
  /** Whether this telemetry read completed, timed out or failed. */
  telemetry_status: TelemetryStatus;
  /** Role of the newest message in the session history (`user` means no assistant reply started). */
  last_message_role?: "user" | "assistant" | "other" | null;
  /** Number of message envelopes returned by the public session endpoint. */
  message_count: number | null;
  /** Number of tool parts in the returned message envelopes. */
  tool_call_count: number | null;
  /** Number of MCP-named tool parts in the returned message envelopes. */
  mcp_tool_call_count: number | null;
  /** Whether a public compaction part was observed, or `null` when the response was unavailable. */
  session_compacted_observed: boolean | null;
  /** Bounded tool names only; tool arguments and results are never retained. */
  tool_names: string[];
  /**
   * CUMULATIVE `retrieve_context` tool parts in the whole session history at this read. The prompt
   * response only carries the turn's final assistant message, so a retrieval made in an earlier step of
   * the same turn is visible here and nowhere else. Absent when the read did not complete.
   */
  retrieval_call_count?: number | null;
  /** CUMULATIVE `retrieve_context` tool parts whose state is `completed` (same scope as above). */
  retrieval_success_count?: number | null;
}

/** The payload-free, in-memory observation captured for one turn. No prompt or answer is kept here. */
export interface TurnObservation {
  turn_id: string;
  host_signal: "success" | "error" | "timeout";
  matcher_passed: boolean;
  answer_text: string;
  /** Tool names already exposed by the prompt response; never tool arguments or results. */
  tool_calls: string[];
  /** Tool names observed through the public session history; never tool arguments or results. */
  tool_names: string[];
  message_count: number | null;
  tool_call_count: number | null;
  mcp_tool_call_count: number | null;
  session_compacted_observed: boolean | null;
  telemetry_status: TelemetryStatus;
  assistant?: AssistantTelemetry;
  timeout: boolean;
  /** Payload-free classification of a timed-out turn (see {@link classifyTimeoutState}). */
  timeout_classification?: TimeoutClassification;
  /** Result of the post-timeout `session.abort` call. */
  abort_status?: "ok" | "error" | "timeout" | "not_attempted";
  /** Whether the session reported idle after the abort. */
  idle_status?: "idle" | "not_idle" | "unknown";
  model_input: number | null;
  latency_ms: number | null;
  retrieval_status: string;
  /** Observed `retrieve_context` calls for the turn, or `null` when the host exposes no count. */
  retrieval_attempts: number | null;
  /** Observed `retrieve_context` calls that completed, or `null` when the host exposes no count. */
  retrieval_successes: number | null;
  /** Schema/tool overhead tokens when the public host exposes them; otherwise `null`. */
  tool_spec_tokens: number | null;
  /** Schema/tool specification count when the public host exposes it; otherwise `null`. */
  tool_spec_count: number | null;
  error_code: string | null;
}

/** Tuning values applied to the real OpenCode A arm and recorded as safe provenance metadata. */
export interface CalibrationTuning {
  max_result_tokens: number;
  preview_tokens: number;
  chunk_tokens: number;
  relevance_threshold: number;
}

/** Parsed CLI options. `dryRun` defaults to true: the driver does nothing live unless told to. */
export interface Options {
  help: boolean;
  dryRun: boolean;
  executeLive: boolean;
  arm: "baseline" | "A" | "both";
  model: string | null;
  approvalId: string | null;
  budgetCeiling: string | null;
  budgetUnit: string | null;
  statPlan: string | null;
  runId: string | null;
  turnsJson: string | null;
  metricsFile: string | null;
  diagnosticFixture: boolean;
  outputDir: string;
  promptTimeoutMs: number;
  startupTimeoutMs: number;
  /** Stop after this turn id and write a diagnostic artifact instead of Gate A evidence. */
  diagnosticStopAfter: string | null;
  /**
   * Text passed as the per-prompt `system` field, identically for BOTH arms. Used to pin the response
   * language: the 5.5 models sometimes answer the Brazilian-looking fixture in Portuguese from turn 1, in
   * either arm, and the correctness oracle only knows English keywords.
   */
  systemPrompt: string | null;
  /** Tools disabled identically for both arms through the public per-prompt `tools` map. */
  disabledTools: string[];
  maxResultTokens: number;
  previewTokens: number;
  chunkTokens: number;
  relevanceThreshold: number;
}

// --- CLI ---------------------------------------------------------------------------------------

export const HELP_TEXT = `ce60 — CE-60 host driver for OpenCode (baseline vs A), Gate A evidence producer.

USAGE:
  bun .opencode/runner/ce60.ts [options]

By default this performs a DRY RUN: it resolves the 60 turns and prints the plan. It starts no
OpenCode process and makes no model call. Pass --execute-live to actually run (billable).

OPTIONS:
  --help                 Show this help and exit.
  --dry-run              Resolve turns and print the plan only (DEFAULT).
  --execute-live         Actually start OpenCode and send the 60 prompts (requires the live inputs).
  --arm <baseline|A|both>  Which arm(s) to run. Default: both.
  --model <provider/model> Model id (e.g. anthropic/claude-sonnet-4). Required for --execute-live.
  --approval-id <id>     Execution approval id. Required for --execute-live.
  --budget-ceiling <n>   Budget ceiling (number). Required for --execute-live.
  --budget-unit <unit>   Budget unit (e.g. USD). Required for --execute-live.
  --stat-plan <path>     Approved statistical-plan JSON file. Required for --execute-live.
  --run-id <id>          Base run id; arms get <run-id>-baseline / <run-id>-A. Required for --execute-live.
  --turns-json <path>    Explicit 60-turn file (array or {turns:[...]}). Falls back to the CE-60 scenario.
  --metrics-file <path>  Per-arm JSONL metrics path; {arm} and {run_id} are replaced when present.
  --diagnostic-fixture   Validate MCP matching and payload-free metrics offline; starts no host or model.
  --output-dir <path>    Where artefacts are written. Default: ./ce60-out.
  --prompt-timeout-ms <n>  Per-turn wall timeout. Default: 600000.
  --startup-timeout-ms <n> OpenCode startup timeout. Default: 60000.
  --diagnostic-stop-after <turn_id>  Stop after this turn and write a diagnostic artifact (never Gate A).
  --disable-tools <a,b>  Disable these tools for BOTH arms via the per-prompt tools map (e.g. question).
  --system-prompt <text> Send this text as the per-prompt system field for BOTH arms (1-500 chars), e.g. a
                         response-language pin. Only its sha256 is recorded in the evidence.
  --max-result-tokens <n>  Arm-A result-size gate. Default: ${DEFAULT_MAX_RESULT_TOKENS}.
  --preview-tokens <n>   Arm-A preview budget. Default: ${DEFAULT_PREVIEW_TOKENS}.
  --chunk-tokens <n>     Arm-A scoring chunk size. Default: ${DEFAULT_CHUNK_TOKENS}.
  --relevance-threshold <n>  Arm-A minimum score in [0, 1]. Default: ${DEFAULT_RELEVANCE_THRESHOLD}.

ENVIRONMENT (required for --execute-live):
  AWS_PROFILE                      AWS profile for the model/plugin credentials (never a default).
  AWS_REGION | AWS_DEFAULT_REGION  AWS region.
  CONTEXT_EXECUTION_MODEL_ID       The OpenCode agent model id; set from --model for arm A.
  CONTEXT_EXECUTION_ALLOWED_MODELS Agent-model allowlist; arm A contains the chosen --model.
  CONTEXT_RERANK_MODEL_ID          Optional; defaults to ${DEFAULT_RERANK_MODEL_ID} for arm A.

Calibration flags are applied only to arm A as CONTEXT_MAX_RESULT_TOKENS,
CONTEXT_PREVIEW_TOKENS, CONTEXT_CHUNK_TOKENS and CONTEXT_RELEVANCE_THRESHOLD.

Only arms baseline and A are supported. A_plus_D and native_B are refused by design.

Practice D (opt-in, experimental): with CONTEXT_GRAPH=on in the environment, arm A also runs the context graph. The run
records graph:on in plugin_options, writes <arm>-graph-evidence.jsonl and <arm>-graph-durability.json next to the other
artifacts, and writes d-slice-comparison.json INSTEAD of gate-a-decision.json: Gate A judges A alone and is never decided
on a run that also ran D.`;

/**
 * Parse an argv tail into {@link Options}. Unknown flags throw. `--dry-run` is implied unless
 * `--execute-live` is given; passing both keeps the run live but is harmless.
 *
 * @param argv - The argument vector (without the `bun`/script prefix).
 * @returns The parsed options.
 */
export function parseArgs(argv: string[]): Options {
  const opts: Options = {
    help: false,
    dryRun: true,
    executeLive: false,
    arm: "both",
    model: null,
    approvalId: null,
    budgetCeiling: null,
    budgetUnit: null,
    statPlan: null,
    runId: null,
    turnsJson: null,
    metricsFile: null,
    diagnosticFixture: false,
    outputDir: "ce60-out",
    promptTimeoutMs: 600000,
    startupTimeoutMs: 60000,
    diagnosticStopAfter: null,
    systemPrompt: null,
    disabledTools: [],
    maxResultTokens: DEFAULT_MAX_RESULT_TOKENS,
    previewTokens: DEFAULT_PREVIEW_TOKENS,
    chunkTokens: DEFAULT_CHUNK_TOKENS,
    relevanceThreshold: DEFAULT_RELEVANCE_THRESHOLD,
  };
  const next = (i: number, flag: string): string => {
    const value = argv[i + 1];
    if (value === undefined) throw new Error(`missing value for ${flag}`);
    return value;
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    switch (arg) {
      case "--help":
      case "-h":
        opts.help = true;
        break;
      case "--dry-run":
        opts.dryRun = true;
        break;
      case "--execute-live":
        opts.executeLive = true;
        opts.dryRun = false;
        break;
      case "--arm": {
        const value = next(i++, arg);
        if (value !== "baseline" && value !== "A" && value !== "both") {
          throw new Error(`--arm must be baseline, A or both (got ${value})`);
        }
        opts.arm = value;
        break;
      }
      case "--model":
        opts.model = next(i++, arg);
        break;
      case "--approval-id":
        opts.approvalId = next(i++, arg);
        break;
      case "--budget-ceiling":
        opts.budgetCeiling = next(i++, arg);
        break;
      case "--budget-unit":
        opts.budgetUnit = next(i++, arg);
        break;
      case "--stat-plan":
        opts.statPlan = next(i++, arg);
        break;
      case "--run-id":
        opts.runId = next(i++, arg);
        break;
      case "--turns-json":
        opts.turnsJson = next(i++, arg);
        break;
      case "--metrics-file":
        opts.metricsFile = next(i++, arg);
        break;
      case "--diagnostic-fixture":
        opts.diagnosticFixture = true;
        break;
      case "--output-dir":
        opts.outputDir = next(i++, arg);
        break;
      case "--prompt-timeout-ms":
        opts.promptTimeoutMs = Number(next(i++, arg));
        break;
      case "--startup-timeout-ms":
        opts.startupTimeoutMs = Number(next(i++, arg));
        break;
      case "--diagnostic-stop-after":
        opts.diagnosticStopAfter = next(i++, arg);
        break;
      case "--system-prompt":
        opts.systemPrompt = next(i++, arg);
        break;
      case "--disable-tools":
        opts.disabledTools = [
          ...new Set(
            next(i++, arg)
              .split(",")
              .map((value) => value.trim())
              .filter((value) => value !== ""),
          ),
        ].sort();
        break;
      case "--max-result-tokens":
        opts.maxResultTokens = Number(next(i++, arg));
        break;
      case "--preview-tokens":
        opts.previewTokens = Number(next(i++, arg));
        break;
      case "--chunk-tokens":
        opts.chunkTokens = Number(next(i++, arg));
        break;
      case "--relevance-threshold":
        opts.relevanceThreshold = Number(next(i++, arg));
        break;
      default:
        throw new Error(`unknown argument: ${arg}`);
    }
  }
  return opts;
}

/**
 * Return the resolved arm-A tuning values in the snake-case form used by env and provenance records.
 *
 * @param opts - The parsed CLI options.
 * @returns The safe, payload-free tuning metadata.
 */
export function calibrationTuning(opts: Options): CalibrationTuning {
  return {
    max_result_tokens: opts.maxResultTokens,
    preview_tokens: opts.previewTokens,
    chunk_tokens: opts.chunkTokens,
    relevance_threshold: opts.relevanceThreshold,
  };
}

/**
 * Validate the numeric calibration flags against the bridge's configuration contract.
 *
 * @param opts - The parsed CLI options.
 * @returns A list of validation errors; empty when every calibration value is valid.
 */
export function validateCalibrationOptions(opts: Options): string[] {
  const errors: string[] = [];
  const positiveInteger = (value: number, flag: string): void => {
    if (!Number.isInteger(value) || value < 1) {
      errors.push(`${flag} must be a positive integer`);
    }
  };
  positiveInteger(opts.promptTimeoutMs, "--prompt-timeout-ms");
  positiveInteger(opts.startupTimeoutMs, "--startup-timeout-ms");
  if (opts.systemPrompt !== null && (opts.systemPrompt.trim() === "" || opts.systemPrompt.length > SYSTEM_PROMPT_MAX_CHARS)) {
    errors.push(`--system-prompt must be 1-${SYSTEM_PROMPT_MAX_CHARS} characters`);
  }
  for (const tool of opts.disabledTools) {
    if (!/^[A-Za-z0-9_.:-]{1,128}$/.test(tool)) errors.push(`--disable-tools has an invalid tool name: ${tool}`);
  }
  if (opts.diagnosticStopAfter !== null && !/^[A-Za-z0-9_.:-]{1,128}$/.test(opts.diagnosticStopAfter)) {
    errors.push("--diagnostic-stop-after must be a turn id");
  }
  positiveInteger(opts.maxResultTokens, "--max-result-tokens");
  positiveInteger(opts.previewTokens, "--preview-tokens");
  positiveInteger(opts.chunkTokens, "--chunk-tokens");
  if (
    !Number.isFinite(opts.relevanceThreshold) ||
    opts.relevanceThreshold < 0 ||
    opts.relevanceThreshold > 1
  ) {
    errors.push("--relevance-threshold must be a number within [0, 1]");
  }
  return errors;
}

/**
 * Validate the inputs a live run requires, reading AWS credentials from the environment.
 *
 * @param opts - The parsed options.
 * @param env - The process environment to read AWS settings from.
 * @returns A list of human-readable errors; empty when the configuration is complete and valid.
 */
export function validateLiveConfig(opts: Options, env: Env): string[] {
  const errors: string[] = [];
  if (!env.AWS_PROFILE || env.AWS_PROFILE.trim() === "") {
    errors.push("AWS_PROFILE is required (no default profile is ever assumed)");
  }
  if (!(env.AWS_REGION || env.AWS_DEFAULT_REGION)) {
    errors.push("AWS_REGION (or AWS_DEFAULT_REGION) is required");
  }
  const requireFlag = (value: string | null, name: string): void => {
    if (!value || value.trim() === "") errors.push(`${name} is required for --execute-live`);
  };
  requireFlag(opts.model, "--model");
  requireFlag(opts.approvalId, "--approval-id");
  requireFlag(opts.budgetCeiling, "--budget-ceiling");
  requireFlag(opts.budgetUnit, "--budget-unit");
  requireFlag(opts.statPlan, "--stat-plan");
  requireFlag(opts.runId, "--run-id");
  if (opts.model && !opts.model.includes("/")) {
    errors.push("--model must be in provider/model form (e.g. anthropic/claude-sonnet-4)");
  }
  if (opts.budgetCeiling && !Number.isFinite(Number(opts.budgetCeiling))) {
    errors.push("--budget-ceiling must be a number");
  }
  if (opts.statPlan && !existsSync(opts.statPlan)) {
    errors.push(`--stat-plan file not found: ${opts.statPlan}`);
  }
  errors.push(...validateCalibrationOptions(opts));
  return errors;
}

// --- Turn resolution ---------------------------------------------------------------------------

/**
 * Normalise a raw turns value into exactly 60 unique {@link Turn}s, or throw.
 *
 * Accepts either a bare array of `{turn_id, prompt}` or a `{turns:[...]}` wrapper. Never truncates or
 * pads: a count other than 60, a duplicate id or an empty prompt is a hard failure.
 *
 * @param data - The parsed JSON value from a turns file or the Python emitter.
 * @returns The 60 validated turns in order.
 */
export function normalizeTurns(data: unknown): Turn[] {
  const rawList = Array.isArray(data)
    ? data
    : data && typeof data === "object" && Array.isArray((data as { turns?: unknown }).turns)
      ? (data as { turns: unknown[] }).turns
      : null;
  if (!rawList) throw new Error("turns payload must be an array or an object with a turns array");
  const turns: Turn[] = rawList.map((item, index) => {
    if (!item || typeof item !== "object") throw new Error(`turn ${index} is not an object`);
    const { turn_id, prompt } = item as { turn_id?: unknown; prompt?: unknown };
    if (typeof turn_id !== "string" || turn_id.trim() === "") {
      throw new Error(`turn ${index} has an invalid turn_id`);
    }
    if (typeof prompt !== "string" || prompt.trim() === "") {
      throw new Error(`turn ${turn_id} has an empty prompt`);
    }
    return { turn_id, prompt };
  });
  if (turns.length !== EXPECTED_TURN_COUNT) {
    throw new Error(`expected exactly ${EXPECTED_TURN_COUNT} turns, got ${turns.length} (never truncate)`);
  }
  const ids = new Set(turns.map((t) => t.turn_id));
  if (ids.size !== EXPECTED_TURN_COUNT) throw new Error("turn ids must be unique across all 60 turns");
  return turns;
}

/**
 * Resolve the 60 CE-60 turns: from `--turns-json` when given, otherwise from the Python emitter
 * (which reads the community scenario). Fails closed when neither yields exactly 60 valid turns.
 *
 * @param opts - The parsed options.
 * @param runPython - Injected Python runner (overridable in tests).
 * @returns The 60 turns.
 */
export async function resolveTurns(
  opts: Options,
  runPython: PythonRunner = defaultPythonRunner,
): Promise<Turn[]> {
  if (opts.turnsJson) {
    return normalizeTurns(JSON.parse(readFileSync(opts.turnsJson, "utf8")));
  }
  const args = ["-m", "opencode_gate_a.host_driver", "emit-turns"];
  const { code, stdout, stderr } = await runPython(args, VALIDATION_DIR, null);
  if (code !== 0) {
    throw new Error(
      `could not resolve CE-60 turns from the scenario (exit ${code}): ${stderr.trim()}\n` +
        "pass --turns-json with exactly 60 turns",
    );
  }
  return normalizeTurns(JSON.parse(stdout));
}

// --- Environment per arm -----------------------------------------------------------------------

/**
 * Build the environment for one arm's OpenCode process.
 *
 * `baseline` inherits only the base environment (and runs with `--pure`, so no plugin loads either
 * way). `A` additionally carries the context-engineering plugin's live admission settings and the
 * explicit calibration values so the plugin can admit and filter oversized tool results during the run.
 * No secret is minted here: AWS credentials come from `AWS_PROFILE`, which must already be in the base
 * environment.
 *
 * @param arm - The arm being started.
 * @param base - The base environment to extend.
 * @param opts - The parsed options (admission and calibration settings).
 * @returns The environment for the arm's process.
 */
export function inheritedOpencodeEnvKeys(base: Env): string[] {
  return Object.keys(base)
    .filter((key) => key.startsWith("OPENCODE_"))
    .sort();
}

/** Longest accepted `--system-prompt`; it is sent on every prompt, so it is kept short. */
export const SYSTEM_PROMPT_MAX_CHARS = 500;

/** The per-prompt public `tools` map that disables the declared tools, or `undefined` when none. */
export function promptToolsMap(opts: Options): Record<string, boolean> | undefined {
  if (opts.disabledTools.length === 0) return undefined;
  return Object.fromEntries(opts.disabledTools.map((tool) => [tool, false]));
}

/** The per-prompt public `system` field, or `undefined` when no system prompt is declared. */
export function promptSystem(opts: Options): string | undefined {
  return opts.systemPrompt !== null && opts.systemPrompt.trim() !== "" ? opts.systemPrompt : undefined;
}

/**
 * The graph policy as the evidence records it. The reference policy adds nothing, so a v7 record stays what it was; the
 * in-place policy says so and how many closed turns it keeps as they were and how many it keeps the dialogue of, because
 * those numbers are different experiments. The defaults mirror `GraphConfig` in the bridge (0 and 0).
 */
function graphPolicyOptions(env: Env): string {
  if (parseGraphMode(env.CONTEXT_GRAPH_MODE) !== "inplace") return "";
  const count = (value: string | undefined, fallback: string): string => {
    const trimmed = (value ?? "").trim();
    return /^\d+$/.test(trimmed) ? trimmed : fallback;
  };
  const keep = count(env.CONTEXT_GRAPH_KEEP_RECENT, "0");
  const dialogue = count(env.CONTEXT_GRAPH_KEEP_DIALOGUE, "0");
  return `,graph_mode:inplace,graph_keep:${keep},graph_dialogue:${dialogue}`;
}

/**
 * The plugin's fidelity switches and revision as the evidence records them, e.g.
 * `preview_style:scoped,retrieval_cleanup:on,plugin_rev:2`.
 * Read from the same environment the arm-A OpenCode process inherits (`buildArmEnv` copies it), so a result
 * always says which retrieval behaviour produced it. Only token characters are used (`:` not `=`).
 */
export function pluginOptions(env: Env): string {
  const style = (env.CONTEXT_PREVIEW_STYLE ?? "").trim().toLowerCase() === "legacy" ? "legacy" : "scoped";
  const cleanupOff = ["0", "false", "off", "no"].includes((env.CONTEXT_RETRIEVAL_CLEANUP ?? "").trim().toLowerCase());
  const graph = parseGraphFlag(env.CONTEXT_GRAPH) ? `,graph:on${graphPolicyOptions(env)}` : "";
  return `preview_style:${style},retrieval_cleanup:${cleanupOff ? "off" : "on"},plugin_rev:${PLUGIN_REVISION}${graph}`;
}

/** Where arm `arm` of run `runId` writes the graph's per-request evidence (payload-free JSONL). */
export function graphEvidencePathForRun(opts: Options, arm: Arm): string {
  return resolve(opts.outputDir, `${arm}-graph-evidence.jsonl`);
}

/** The file that holds the Gate A-style decision for a run: it is NOT Gate A's when the graph ran. */
export function decisionFileName(env: Env): string {
  return parseGraphFlag(env.CONTEXT_GRAPH) ? "d-slice-comparison.json" : "gate-a-decision.json";
}

/** Digest recorded in the evidence instead of the text; empty when no system prompt is declared. */
export function systemPromptDigest(opts: Options): string {
  const system = promptSystem(opts);
  return system === undefined ? "" : "sha256:" + createHash("sha256").update(system).digest("hex");
}

export function buildArmEnv(
  arm: Arm,
  base: Env,
  opts: Options,
  metricsFile?: string,
  graphEvidenceFile?: string,
): Env {
  const env: Env = { ...base };
  // Inherited OpenCode configuration overrides would change BOTH arms silently; strip them so every
  // policy difference is declared on the CLI and recorded in the manifests.
  for (const key of inheritedOpencodeEnvKeys(base)) delete env[key];
  if (arm === "A") {
    const tuning = calibrationTuning(opts);
    env.CONTEXT_EXECUTION_MODE = "live";
    env.CONTEXT_APPROVAL_ID = opts.approvalId ?? "";
    env.CONTEXT_BUDGET_CEILING = opts.budgetCeiling ?? "";
    env.CONTEXT_BUDGET_UNIT = opts.budgetUnit ?? "";
    env.CONTEXT_EXECUTION_ALLOWED_MODELS = opts.model ?? "";
    if (opts.executeLive && opts.model) {
      env.CONTEXT_EXECUTION_MODEL_ID = opts.model;
    } else {
      delete env.CONTEXT_EXECUTION_MODEL_ID;
    }
    env.CONTEXT_RERANK_MODEL_ID = base.CONTEXT_RERANK_MODEL_ID || DEFAULT_RERANK_MODEL_ID;
    env.CONTEXT_MAX_RESULT_TOKENS = String(tuning.max_result_tokens);
    env.CONTEXT_PREVIEW_TOKENS = String(tuning.preview_tokens);
    env.CONTEXT_CHUNK_TOKENS = String(tuning.chunk_tokens);
    env.CONTEXT_RELEVANCE_THRESHOLD = String(tuning.relevance_threshold);
    const resolvedMetricsFile = metricsFile ?? opts.metricsFile ?? base[METRICS_FILE_ENV];
    if (resolvedMetricsFile) env[METRICS_FILE_ENV] = resolvedMetricsFile;
    // Practice D rides on arm A only when the operator asked for it; the evidence file is per arm and per run.
    if (parseGraphFlag(base.CONTEXT_GRAPH)) {
      env.CONTEXT_GRAPH = "on";
      if (graphEvidenceFile) env.CONTEXT_GRAPH_EVIDENCE_FILE = graphEvidenceFile;
    }
  } else {
    // Baseline: strip inherited opt-ins, calibration and bridge metrics. `--pure` still loads no plugin.
    delete env.CONTEXT_EXECUTION_MODE;
    delete env.CONTEXT_EXECUTION_MODEL_ID;
    delete env[METRICS_FILE_ENV];
    for (const key of CALIBRATION_ENV_KEYS) delete env[key];
    // The baseline loads no plugin; nothing of the graph may leak into it either.
    for (const key of Object.keys(env)) if (key.startsWith("CONTEXT_GRAPH")) delete env[key];
  }
  return env;
}

// --- Prompt response parsing -------------------------------------------------------------------

/**
 * Derive the countable model-input tokens from the SDK `info.tokens` shape.
 *
 * The real `@opencode-ai/sdk` assistant message carries
 * `tokens: { input, output, reasoning, cache: { read, write } }`. The CE-60 model-input metric is the
 * input the model actually had to read: the uncached `input` plus the cache `read` and `write` counts.
 * `output` and `reasoning` are generation, not input, and are deliberately NOT summed. When none of the
 * three input-side fields is present the reading is unavailable — returned as `null` so it is recorded
 * as a collection failure downstream, never approximated from characters.
 *
 * @param tokens - The `info.tokens` value (any shape).
 * @returns The summed input tokens, or `null` when no input-side field is present.
 */
export function extractModelInputTokens(tokens: unknown): number | null {
  if (!tokens || typeof tokens !== "object") return null;
  const t = tokens as { input?: unknown; cache?: unknown };
  const cache = (t.cache && typeof t.cache === "object" ? t.cache : {}) as {
    read?: unknown;
    write?: unknown;
  };
  const candidates = [t.input, cache.read, cache.write];
  let total = 0;
  let present = false;
  for (const value of candidates) {
    if (typeof value === "number" && Number.isFinite(value)) {
      total += value;
      present = true;
    }
  }
  return present ? total : null;
}

/**
 * Extract public schema/tool overhead fields without folding them into model-input tokens.
 *
 * OpenCode SDK 1.18.34 does not expose these fields, so the normal result is two `null`s. Future host
 * versions may expose either snake-case or camel-case names on `info.tokens` or `info` itself; only
 * finite non-negative numbers are accepted, and no character-based estimate is made.
 */
export function extractToolSpecMetrics(info: unknown): {
  tool_spec_tokens: number | null;
  tool_spec_count: number | null;
} {
  if (!info || typeof info !== "object") {
    return { tool_spec_tokens: null, tool_spec_count: null };
  }
  const body = info as Record<string, unknown>;
  const tokens = body.tokens && typeof body.tokens === "object" ? (body.tokens as Record<string, unknown>) : {};
  const readNumber = (...values: unknown[]): number | null => {
    for (const value of values) {
      if (typeof value === "number" && Number.isFinite(value) && value >= 0) return value;
    }
    return null;
  };
  return {
    tool_spec_tokens: readNumber(
      body.tool_spec_tokens,
      body.toolSpecTokens,
      tokens.tool_spec_tokens,
      tokens.toolSpecTokens,
    ),
    tool_spec_count: readNumber(
      body.tool_spec_count,
      body.toolSpecCount,
      tokens.tool_spec_count,
      tokens.toolSpecCount,
    ),
  };
}

const MAX_TOOL_NAME_LENGTH = 128;
const MAX_TOOL_NAMES_PER_TURN = 256;
const MAX_SUMMARY_TOOL_NAMES = 512;
const MCP_TOOL_PREFIXES = ["mcp_", "ce60-fixture_"] as const;

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object";
}

function safeToolName(value: unknown): string | null {
  if (typeof value !== "string" || value.length === 0 || value.length > MAX_TOOL_NAME_LENGTH) {
    return null;
  }
  // OpenCode tool identifiers are names, not free-form payloads. Reject control/whitespace characters
  // before retaining a name so a malformed host response cannot turn telemetry into a text channel.
  return /^[A-Za-z0-9_.:-]+$/.test(value) ? value : null;
}

function assistantTelemetryFromMessage(value: unknown): AssistantTelemetry {
  const empty: AssistantTelemetry = {
    present: null,
    completed: null,
    finish: null,
    error_name: null,
    error_status_code: null,
    error_retryable: null,
    step_start_count: null,
    step_finish_count: null,
    last_part_types: [],
    last_tool: null,
    last_step_tokens: null,
  };
  if (!isRecord(value)) return empty;
  const info = isRecord(value.info) ? value.info : value;
  const parts = Array.isArray(value.parts)
    ? value.parts.filter(isRecord)
    : Array.isArray(info.parts)
      ? info.parts.filter(isRecord)
      : [];
  const role = typeof info.role === "string" ? info.role : null;
  const time = isRecord(info.time) ? info.time : {};
  const completed = typeof time.completed === "number" && Number.isFinite(time.completed);
  const normalizeEnum = (raw: unknown, allowed: readonly string[]): string | null => {
    if (typeof raw !== "string") return null;
    return allowed.includes(raw) ? raw : "other";
  };
  const finish = normalizeEnum(info.finish, ["stop", "tool-calls", "length", "content-filter", "error"]);
  const error = isRecord(info.error) ? info.error : null;
  const errorData = error && isRecord(error.data) ? error.data : null;
  const errorName = normalizeEnum(
    error?.name ?? errorData?.name,
    ["ProviderAuthError", "UnknownError", "MessageOutputLengthError", "MessageAbortedError", "APIError"],
  );
  const errorStatus = errorData?.statusCode ?? errorData?.status;
  const errorStatusCode = typeof errorStatus === "number" && Number.isInteger(errorStatus) && errorStatus >= 0
    ? errorStatus
    : null;
  // SDK 1.18.34 `ApiError.data` names the flag `isRetryable` (verified against types.gen.d.ts).
  const retryable = errorData?.isRetryable ?? errorData?.retryable;
  const partTypes = parts
    .map((part) => (typeof part.type === "string" && /^[A-Za-z0-9_.:-]{1,64}$/.test(part.type) ? part.type : null))
    .filter((partType): partType is string => partType !== null)
    .slice(-5);
  const toolParts = parts.filter((part) => part.type === "tool");
  const lastToolPart = toolParts.length > 0 ? toolParts[toolParts.length - 1] : null;
  const lastToolName = lastToolPart ? safeToolName(lastToolPart.tool) : null;
  const rawToolStatus = lastToolPart && isRecord(lastToolPart.state) ? lastToolPart.state.status : null;
  const lastToolStatus = typeof rawToolStatus === "string" && /^(pending|running|completed|error)$/.test(rawToolStatus)
    ? rawToolStatus
    : null;
  const tokens = isRecord(info.tokens) ? info.tokens : null;
  const cache = tokens && isRecord(tokens.cache) ? tokens.cache : null;
  const tokenValue = (source: Record<string, unknown> | null, key: string): number | null => {
    const value = source?.[key];
    return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : null;
  };
  const lastStepTokens = tokens
    ? {
        input: tokenValue(tokens, "input"),
        cache_read: tokenValue(cache, "read"),
        cache_write: tokenValue(cache, "write"),
      }
    : null;
  const hasTokenValue = lastStepTokens !== null && Object.values(lastStepTokens).some((item) => item !== null);
  return {
    present: role === null ? true : role === "assistant",
    completed: role === "assistant" || role === null ? completed : null,
    finish,
    error_name: errorName,
    error_status_code: errorStatusCode,
    error_retryable: typeof retryable === "boolean" ? retryable : null,
    step_start_count: parts.filter((part) => part.type === "step-start").length,
    step_finish_count: parts.filter((part) => part.type === "step-finish").length,
    last_part_types: partTypes,
    last_tool: lastToolName && lastToolStatus ? { name: lastToolName, status: lastToolStatus } : null,
    last_step_tokens: hasTokenValue ? lastStepTokens : null,
  };
}

/**
 * Classify a timed-out turn from the payload-free state captured before the abort.
 *
 * Pure function over scalars only. Partial telemetry is never guessed: it yields `unclassified`.
 */
export function classifyTimeoutState(observation: {
  telemetry_status: TelemetryStatus;
  last_message_role?: "user" | "assistant" | "other" | null;
  assistant?: AssistantTelemetry;
}): TimeoutClassification {
  if (observation.telemetry_status !== "ok") return "unclassified";
  // The newest message is still the user prompt: no assistant output started for this turn.
  if (observation.last_message_role === "user") return "model_stream_stall";
  const assistant = observation.assistant;
  if (!assistant || assistant.present !== true) return "unclassified";
  const tool = assistant.last_tool;
  if (tool && (tool.status === "pending" || tool.status === "running")) {
    if (tool.name === "question") return "blocked_question";
    return isMcpToolName(tool.name) ? "tool_pending_mcp" : "tool_pending_host";
  }
  if (assistant.error_retryable === true) return "provider_retry";
  if (assistant.completed === true) return "prompt_resolution";
  if (assistant.completed === false) return "model_stream_stall";
  return "unclassified";
}

/** Return whether an observed tool identifier uses one of the public MCP naming forms. */
export function isMcpToolName(toolName: string): boolean {
  return (
    /^mcp(?:[_:.]|$)/i.test(toolName) ||
    MCP_TOOL_PREFIXES.some((prefix) => toolName.startsWith(prefix))
  );
}

/**
 * Reduce a public `session.messages` response to counts, flags and tool names only.
 *
 * The SDK response contains message metadata and parts that may carry text, arguments and tool output.
 * This function intentionally reads only array length, part type, tool name and compaction type; it never
 * copies or serializes any other field.
 */
export function parseSessionMessagesResponse(response: unknown): SessionTelemetry {
  const unavailable: SessionTelemetry = {
    telemetry_status: "error",
    message_count: null,
    tool_call_count: null,
    mcp_tool_call_count: null,
    session_compacted_observed: null,
    tool_names: [],
  };
  const data = isRecord(response) && "data" in response ? response.data : response;
  if (!Array.isArray(data)) return unavailable;

  let toolCallCount = 0;
  let mcpToolCallCount = 0;
  let retrievalCallCount = 0;
  let retrievalSuccessCount = 0;
  let sessionCompactedObserved = false;
  let assistantMessage: unknown = null;
  let lastMessageRole: "user" | "assistant" | "other" | null = null;
  const toolNames: string[] = [];
  const seenNames = new Set<string>();

  for (const item of data) {
    if (!isRecord(item)) continue;
    const info = isRecord(item.info) ? item.info : undefined;
    const parts = Array.isArray(item.parts)
      ? item.parts
      : info && Array.isArray(info.parts)
        ? info.parts
        : [];
    if (info?.role === "assistant") assistantMessage = item;
    lastMessageRole = info?.role === "user" || info?.role === "assistant" ? info.role : "other";
    for (const part of parts) {
      if (!isRecord(part)) continue;
      if (part.type === "compaction") sessionCompactedObserved = true;
      if (part.type !== "tool") continue;
      toolCallCount++;
      if (part.tool === RETRIEVE_CONTEXT_TOOL) {
        // Counted across the whole history: this is the only place a retrieval made in a non-final step
        // of a turn is visible (the prompt response carries the final assistant message alone).
        retrievalCallCount++;
        const state = isRecord(part.state) ? part.state : undefined;
        if (state?.status === "completed") retrievalSuccessCount++;
      }
      const toolName = safeToolName(part.tool);
      if (!toolName) continue;
      if (isMcpToolName(toolName)) mcpToolCallCount++;
      if (!seenNames.has(toolName) && toolNames.length < MAX_TOOL_NAMES_PER_TURN) {
        seenNames.add(toolName);
        toolNames.push(toolName);
      }
    }
  }

  return {
    telemetry_status: "ok",
    message_count: data.length,
    tool_call_count: toolCallCount,
    mcp_tool_call_count: mcpToolCallCount,
    session_compacted_observed: sessionCompactedObserved,
    assistant: assistantTelemetryFromMessage(assistantMessage),
    last_message_role: lastMessageRole,
    tool_names: toolNames,
    retrieval_call_count: retrievalCallCount,
    retrieval_success_count: retrievalSuccessCount,
  };
}

/** Cumulative `retrieve_context` counts at the last session read that completed (arm A only). */
export interface RetrievalTracker {
  calls: number;
  successes: number;
}

/** A tracker for a fresh session: nothing has been retrieved yet. */
export function newRetrievalTracker(): RetrievalTracker {
  return { calls: 0, successes: 0 };
}

/**
 * Attribute `retrieve_context` calls to the turn that made them, from the session history.
 *
 * `parsePromptResponse` only sees the final assistant message of a prompt, so a retrieval made in an
 * earlier step of the same turn was counted as zero and Gate A's "retrieval is exact" rule judged a
 * vacuous 0/0. The session history holds every step, so the turn's count is the growth of the cumulative
 * count since the last successful read.
 *
 * Fail closed, never a fabricated zero: when the session read did not complete, the turn's retrieval is
 * `collection_failed` with `null` counts and the tracker is left alone, so the next successful read
 * attributes the missing growth rather than losing it. When the read completed but the counts are
 * implausible (moved backwards, or successes outran calls) the turn is `collection_failed` too, and the
 * tracker still advances so one bad turn cannot poison the later ones.
 * The baseline arm exposes no retrieval and is returned unchanged.
 *
 * @param observation - The turn observation parsed from the prompt response.
 * @param telemetry - The session read taken right after that prompt.
 * @param tracker - Cumulative counts at the previous successful read; updated in place on success.
 * @param retrievalExposed - Whether this arm exposes a retrieval count (arm A).
 * @returns The observation with session-derived retrieval counts and no cumulative bookkeeping fields.
 */
export function attributeSessionRetrieval(
  observation: TurnObservation,
  telemetry: SessionTelemetry,
  tracker: RetrievalTracker,
  retrievalExposed: boolean,
): TurnObservation {
  // The cumulative fields are bookkeeping for this function only; never carry them into the payload.
  const base = { ...observation } as TurnObservation & Partial<SessionTelemetry>;
  delete base.retrieval_call_count;
  delete base.retrieval_success_count;
  if (!retrievalExposed) return base;

  const failed = (): TurnObservation => ({
    ...base,
    retrieval_status: "collection_failed",
    retrieval_attempts: null,
    retrieval_successes: null,
  });
  const cumulativeCalls = telemetry.retrieval_call_count;
  const cumulativeSuccesses = telemetry.retrieval_success_count;
  if (
    telemetry.telemetry_status !== "ok" ||
    typeof cumulativeCalls !== "number" ||
    typeof cumulativeSuccesses !== "number"
  ) {
    return failed();
  }
  const deltaCalls = cumulativeCalls - tracker.calls;
  const deltaSuccesses = cumulativeSuccesses - tracker.successes;
  // The read completed, so advance the tracker even when the delta is implausible: one bad turn must not
  // poison every later turn.
  tracker.calls = cumulativeCalls;
  tracker.successes = cumulativeSuccesses;
  if (deltaCalls < 0 || deltaSuccesses < 0 || deltaSuccesses > deltaCalls) return failed();
  return {
    ...base,
    retrieval_status: "exposed",
    retrieval_attempts: deltaCalls,
    retrieval_successes: deltaSuccesses,
  };
}

/** Merge public session telemetry into a prompt observation without copying response content. */
export function attachSessionTelemetry(
  observation: TurnObservation,
  telemetry: SessionTelemetry,
  wallLatencyMs: number | null = null,
): TurnObservation {
  const names = [...new Set([...telemetry.tool_names, ...observation.tool_calls])].slice(
    0,
    MAX_TOOL_NAMES_PER_TURN,
  );
  return {
    ...observation,
    ...telemetry,
    tool_names: names,
    latency_ms: observation.latency_ms ?? wallLatencyMs,
    timeout: false,
  };
}

/**
 * Parse one assistant response into the payload-free observation consumed by the Python finalizer.
 *
 * Reads the assistant message's token/timing/error fields and the text/tool parts. The answer text is
 * returned so the Python oracle can score it, but it is never written to a file by this driver.
 *
 * Retrieval is counted from the `retrieve_context` tool parts: each call is an attempt and each call
 * whose tool state is `completed` is a success. These counts only see the FINAL assistant message of the
 * prompt; a live run replaces them with the session-derived counts of {@link attributeSessionRetrieval}.
 * The count is only reported when `retrievalExposed` is
 * set (arm A, where the plugin registers the tool); otherwise retrieval is `host_not_exposed` with
 * `null` counts, never a fabricated zero.
 *
 * @param turnId - The turn id.
 * @param data - The `{info, parts}` body of a successful prompt response.
 * @param options - `retrievalExposed` marks that the host exposes a retrieval count for this arm.
 * @returns The observation for the turn.
 */
export function parsePromptResponse(
  turnId: string,
  data: unknown,
  options: { retrievalExposed?: boolean } = {},
): TurnObservation {
  const retrievalExposed = options.retrievalExposed === true;
  const body = (data ?? {}) as { info?: Record<string, unknown>; parts?: unknown[] };
  const info = (body.info ?? {}) as Record<string, unknown>;
  const parts = Array.isArray(body.parts) ? body.parts : [];

  const texts: string[] = [];
  const toolCalls: string[] = [];
  let retrievalAttempts = 0;
  let retrievalSuccesses = 0;
  for (const part of parts) {
    const p = (part ?? {}) as {
      type?: unknown;
      text?: unknown;
      tool?: unknown;
      state?: { status?: unknown };
    };
    if (p.type === "text" && typeof p.text === "string") {
      texts.push(p.text);
    } else if (p.type === "tool" && typeof p.tool === "string") {
      toolCalls.push(p.tool);
      if (p.tool === RETRIEVE_CONTEXT_TOOL) {
        retrievalAttempts++;
        const status =
          p.state && typeof p.state === "object" ? (p.state as { status?: unknown }).status : undefined;
        if (status === "completed") retrievalSuccesses++;
      }
    }
  }
  const answerText = texts.join("\n");

  const modelInput = extractModelInputTokens(info.tokens);
  const toolSpec = extractToolSpecMetrics(info);

  const time = (info.time ?? {}) as { created?: unknown; completed?: unknown };
  const latencyMs =
    typeof time.created === "number" && typeof time.completed === "number"
      ? time.completed - time.created
      : null;

  const error = info.error as { name?: unknown; data?: { name?: unknown } } | undefined;
  let hostSignal: TurnObservation["host_signal"] = "success";
  let errorCode: string | null = null;
  if (error) {
    const name =
      (typeof error.name === "string" && error.name) ||
      (error.data && typeof error.data.name === "string" && error.data.name) ||
      "HOST_ERROR";
    errorCode = String(name);
    hostSignal = /abort/i.test(errorCode) ? "error" : "error";
  }

  return {
    turn_id: turnId,
    host_signal: hostSignal,
    // A terminal answer is "well-formed" when the host did not error and produced non-empty text.
    // This is the completion matcher; correctness is a separate, oracle-scored axis.
    matcher_passed: hostSignal === "success" && answerText.trim() !== "",
    answer_text: answerText,
    tool_calls: toolCalls,
    tool_names: [...toolCalls],
    message_count: null,
    tool_call_count: null,
    mcp_tool_call_count: null,
    session_compacted_observed: null,
    telemetry_status: "not_attempted",
    assistant: assistantTelemetryFromMessage({ info, parts }),
    timeout: false,
    model_input: modelInput,
    latency_ms: latencyMs,
    retrieval_status: retrievalExposed ? "exposed" : "host_not_exposed",
    retrieval_attempts: retrievalExposed ? retrievalAttempts : null,
    retrieval_successes: retrievalExposed ? retrievalSuccesses : null,
    tool_spec_tokens: toolSpec.tool_spec_tokens,
    tool_spec_count: toolSpec.tool_spec_count,
    error_code: errorCode,
  };
}

// --- Sequential turn execution -----------------------------------------------------------------

/** A function that sends one turn and resolves its observation. */
export type SendTurn = (turn: Turn, index: number) => Promise<TurnObservation>;

/**
 * Send every turn STRICTLY ONE AT A TIME, in order, awaiting each before the next.
 *
 * There is no `Promise.all`/`Promise.race` over the turns on purpose: CE-60 is a single evolving
 * conversation, so overlapping turns would corrupt the very history the benchmark measures. A
 * `maxInFlight` guard proves the invariant to tests.
 *
 * @param turns - The ordered turns.
 * @param send - The per-turn sender.
 * @returns The observations in turn order.
 */
export async function runTurnsSequential(turns: Turn[], send: SendTurn): Promise<TurnObservation[]> {
  const observations: TurnObservation[] = [];
  for (let i = 0; i < turns.length; i++) {
    observations.push(await send(turns[i], i));
  }
  return observations;
}

// --- Python bridge -----------------------------------------------------------------------------

/** Result of running a Python subprocess. */
export interface PythonResult {
  code: number;
  stdout: string;
  stderr: string;
}

/** Runs a Python subprocess with an optional stdin string; resolves its exit code and streams. */
export type PythonRunner = (
  args: string[],
  cwd: string,
  stdin: string | null,
  env?: Env,
) => Promise<PythonResult>;

/** Default Python runner: spawns `python3` (override via the PYTHON env var). */
export const defaultPythonRunner: PythonRunner = (args, cwd, stdin, env) =>
  new Promise<PythonResult>((resolvePromise, reject) => {
    const python = process.env.PYTHON || "python3";
    const child = spawn(python, args, {
      cwd,
      env: env ? { ...process.env, ...env } : process.env,
      stdio: ["pipe", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => (stdout += chunk.toString()));
    child.stderr.on("data", (chunk) => (stderr += chunk.toString()));
    child.on("error", reject);
    child.on("close", (code) => resolvePromise({ code: code ?? 0, stdout, stderr }));
    if (stdin !== null) child.stdin.write(stdin);
    child.stdin.end();
  });

// --- Sanitized metrics and token-overhead summaries ----------------------------------------------

const METRIC_OPERATIONS = new Set(["skipped", "previewed", "baseline", "refused"]);

function nonNegativeInteger(value: unknown): number | null {
  return typeof value === "number" && Number.isInteger(value) && value >= 0 ? value : null;
}

/** Read only the bridge's allow-listed JSONL fields and aggregate a payload-free run summary. */
export function summarizeMetricsFile(
  path: string | null,
  exposedHint = false,
): Record<string, unknown> {
  const summary: Record<string, unknown> = {
    metrics_schema_version: 1,
    metrics_status: path && existsSync(path) ? "exposed" : exposedHint ? "collection_failed" : "host_not_exposed",
    transform_count: 0,
    skipped_count: 0,
    previewed_count: 0,
    baseline_count: 0,
    refused_count: 0,
    filtered_count: 0,
    input_bytes: 0,
    output_bytes: 0,
    source_already_truncated_count: 0,
    fallback_count: 0,
    rerank_attempts: 0,
    rerank_successes: 0,
    rerank_failures: 0,
    retrieval_attempts: 0,
    retrieval_successes: 0,
    metrics_write_failed: false,
    tool_classes: "",
  };
  if (!path || !existsSync(path)) return summary;

  const classes = new Set<string>();
  let invalidRecord = false;
  let contents = "";
  try {
    contents = readFileSync(path, "utf8");
  } catch {
    summary.metrics_status = "collection_failed";
    return summary;
  }
  for (const line of contents.split(/\r?\n/)) {
    if (line.trim() === "") continue;
    let record: unknown;
    try {
      record = JSON.parse(line);
    } catch {
      invalidRecord = true;
      continue;
    }
    if (!record || typeof record !== "object") {
      invalidRecord = true;
      continue;
    }
    const item = record as Record<string, unknown>;
    if (item.event === "retrieval") {
      const attempts = nonNegativeInteger(item.retrieval_attempts);
      const successes = nonNegativeInteger(item.retrieval_successes);
      if (attempts === null || successes === null || successes > attempts) {
        invalidRecord = true;
        continue;
      }
      summary.retrieval_attempts = Math.max(summary.retrieval_attempts as number, attempts);
      summary.retrieval_successes = Math.max(summary.retrieval_successes as number, successes);
      continue;
    }
    if (item.event !== "transform") continue;
    const operation = typeof item.operation === "string" ? item.operation : "";
    const toolClass = typeof item.tool_class === "string" ? item.tool_class : "";
    const inputBytes = nonNegativeInteger(item.input_bytes);
    const outputBytes = nonNegativeInteger(item.output_bytes);
    const retrievalAttempts = nonNegativeInteger(item.retrieval_attempts);
    const retrievalSuccesses = nonNegativeInteger(item.retrieval_successes);
    if (
      !METRIC_OPERATIONS.has(operation) ||
      !/^[A-Za-z0-9][A-Za-z0-9_.-]*$/.test(toolClass) ||
      inputBytes === null ||
      outputBytes === null ||
      retrievalAttempts === null ||
      retrievalSuccesses === null ||
      retrievalSuccesses > retrievalAttempts
    ) {
      invalidRecord = true;
      continue;
    }
    classes.add(toolClass);
    summary.transform_count = (summary.transform_count as number) + 1;
    summary[`${operation}_count`] = (summary[`${operation}_count`] as number) + 1;
    summary.input_bytes = (summary.input_bytes as number) + inputBytes;
    summary.output_bytes = (summary.output_bytes as number) + outputBytes;
    if (item.filtered === true) summary.filtered_count = (summary.filtered_count as number) + 1;
    if (item.source_already_truncated === true) {
      summary.source_already_truncated_count = (summary.source_already_truncated_count as number) + 1;
    }
    if (item.fallback === true) summary.fallback_count = (summary.fallback_count as number) + 1;
    if (item.rerank_attempt === true) summary.rerank_attempts = (summary.rerank_attempts as number) + 1;
    if (item.rerank_result === "success") summary.rerank_successes = (summary.rerank_successes as number) + 1;
    if (item.rerank_result === "failure") summary.rerank_failures = (summary.rerank_failures as number) + 1;
    summary.retrieval_attempts = Math.max(summary.retrieval_attempts as number, retrievalAttempts);
    summary.retrieval_successes = Math.max(summary.retrieval_successes as number, retrievalSuccesses);
  }
  summary.tool_classes = [...classes].sort().join(",");
  if (invalidRecord) summary.metrics_status = "collection_failed";
  return summary;
}

/** Summarize schema/tool overhead separately from conversation/model-input tokens. */
export function summarizeToolSpecMetrics(observations: TurnObservation[]): Record<string, unknown> {
  const tokens = observations.map((observation) => observation.tool_spec_tokens);
  const counts = observations.map((observation) => observation.tool_spec_count);
  const allTokens = observations.length > 0 && tokens.every((value) => value !== null);
  const allCounts = observations.length > 0 && counts.every((value) => value !== null);
  return {
    tool_spec_tokens: allTokens ? tokens.reduce((total, value) => total + (value as number), 0) : null,
    tool_spec_tokens_status: allTokens ? "exposed" : "host_not_exposed",
    tool_spec_count: allCounts ? counts.reduce((total, value) => total + (value as number), 0) : null,
    tool_spec_count_status: allCounts ? "exposed" : "host_not_exposed",
  };
}

/** Aggregate payload-free session snapshots without summing cumulative snapshots. */
export function summarizeObservability(observations: TurnObservation[]): Record<string, unknown> {
  type Snapshot = {
    message_count: number;
    tool_call_count: number;
    mcp_tool_call_count: number;
  };

  const snapshots = observations.map((observation) => {
    if (
      observation.telemetry_status !== "ok" ||
      observation.message_count === null ||
      observation.tool_call_count === null ||
      observation.mcp_tool_call_count === null
    ) {
      return null;
    }
    return {
      message_count: observation.message_count,
      tool_call_count: observation.tool_call_count,
      mcp_tool_call_count: observation.mcp_tool_call_count,
    } satisfies Snapshot;
  });

  const deltaMessageCounts: number[] = [];
  const deltaToolCallCounts: number[] = [];
  const deltaMcpToolCallCounts: number[] = [];
  const snapshotGapTurnIds: string[] = [];
  const snapshotRegressionTurnIds: string[] = [];
  let previous: Snapshot | null = null;

  for (let index = 0; index < observations.length; index++) {
    const snapshot = snapshots[index];
    if (snapshot === null) {
      snapshotGapTurnIds.push(observations[index].turn_id);
      previous = null;
      continue;
    }
    if (previous === null) {
      if (index === 0) {
        deltaMessageCounts.push(snapshot.message_count);
        deltaToolCallCounts.push(snapshot.tool_call_count);
        deltaMcpToolCallCounts.push(snapshot.mcp_tool_call_count);
      }
      previous = snapshot;
      continue;
    }
    const messageDelta = snapshot.message_count - previous.message_count;
    const toolDelta = snapshot.tool_call_count - previous.tool_call_count;
    const mcpToolDelta = snapshot.mcp_tool_call_count - previous.mcp_tool_call_count;
    if (messageDelta < 0 || toolDelta < 0 || mcpToolDelta < 0) {
      snapshotRegressionTurnIds.push(observations[index].turn_id);
      previous = snapshot;
      continue;
    }
    deltaMessageCounts.push(messageDelta);
    deltaToolCallCounts.push(toolDelta);
    deltaMcpToolCallCounts.push(mcpToolDelta);
    previous = snapshot;
  }

  const finalSnapshot = [...snapshots].reverse().find((snapshot): snapshot is Snapshot => snapshot !== null);
  const completeDeltas = snapshotGapTurnIds.length === 0 && snapshotRegressionTurnIds.length === 0;
  const sumOrNull = (values: number[]): number | null => (completeDeltas ? values.reduce((sum, value) => sum + value, 0) : null);
  const maxOrNull = (values: number[]): number | null => (values.length > 0 ? Math.max(...values) : null);
  const timedOut = observations
    .filter((observation) => observation.timeout || observation.host_signal === "timeout")
    .filter((observation) => typeof observation.turn_id === "string" && observation.turn_id.length > 0)
    .slice(0, EXPECTED_TURN_COUNT);
  const timeoutTurnIds = timedOut.map((observation) => observation.turn_id);
  // Parallel, bounded enum lists aligned with timeout_turn_ids: the state captured before each abort.
  const timeoutClassifications = timedOut.map(
    (observation) => observation.timeout_classification ?? classifyTimeoutState(observation),
  );
  const timeoutLastToolNames = timedOut.map((observation) => observation.assistant?.last_tool?.name ?? "none");
  const timeoutLastToolStatuses = timedOut.map(
    (observation) => observation.assistant?.last_tool?.status ?? "none",
  );
  const timeoutAssistantStates = timedOut.map((observation) =>
    observation.assistant?.completed === true
      ? "completed"
      : observation.assistant?.completed === false
        ? "incomplete"
        : "unknown",
  );
  const timeoutAbortStatuses = timedOut.map((observation) => observation.abort_status ?? "not_attempted");
  const timeoutIdleStatuses = timedOut.map((observation) => observation.idle_status ?? "unknown");
  // Non-timeout host errors (e.g. a transient provider 503) keep their structured, payload-free class so
  // an externally caused failure is distinguishable from a plugin or harness failure.
  const erroredTurns = observations.filter(
    (observation) => observation.host_signal === "error" && observation.error_code !== null,
  );
  const errorTurnIds = erroredTurns.map((observation) => observation.turn_id).slice(0, EXPECTED_TURN_COUNT);
  const errorCodes = erroredTurns.map((observation) =>
    SAFE_CODE.test(observation.error_code as string) ? (observation.error_code as string) : "other",
  );
  const errorStatusCodes = erroredTurns.map((observation) =>
    observation.assistant?.error_status_code === null || observation.assistant?.error_status_code === undefined
      ? "none"
      : String(observation.assistant.error_status_code),
  );
  const errorRetryable = erroredTurns.map((observation) =>
    observation.assistant?.error_retryable === true
      ? "retryable"
      : observation.assistant?.error_retryable === false
        ? "not_retryable"
        : "unknown",
  );
  const sessionNotIdleTurnIds = observations
    .filter((observation) => observation.error_code === "SESSION_NOT_IDLE")
    .map((observation) => observation.turn_id)
    .slice(0, EXPECTED_TURN_COUNT);

  const toolNames = new Set<string>();
  for (const observation of observations) {
    for (const name of [...(observation.tool_names ?? []), ...(observation.tool_calls ?? [])]) {
      const safeName = safeToolName(name);
      if (safeName) toolNames.add(safeName);
    }
  }

  const contextDropTurnIds: string[] = [];
  let previousModelInput: number | null = null;
  for (const observation of observations) {
    if (observation.model_input !== null && previousModelInput !== null && observation.model_input < previousModelInput * 0.8) {
      contextDropTurnIds.push(observation.turn_id);
    }
    if (observation.model_input !== null) previousModelInput = observation.model_input;
  }
  const compactedObservedCount = observations.filter(
    (observation) => observation.session_compacted_observed === true,
  ).length;
  const compactionStatus = compactedObservedCount > 0 ? "observed" : "unavailable";
  const compactionReason = compactedObservedCount > 0
    ? null
    : contextDropTurnIds.length > 0
      ? "unexplained_context_drop"
      : "event_channel_not_observed";
  const telemetryStatus = observations.length === 0
    ? "not_attempted"
    : observations.every((observation) => observation.telemetry_status === "ok")
      ? "ok"
      : observations.some((observation) => observation.telemetry_status === "error" || observation.telemetry_status === "timeout")
        ? "partial"
        : "unavailable";

  return {
    observability_schema_version: 2,
    telemetry_status: telemetryStatus,
    session_message_count_final: finalSnapshot?.message_count ?? null,
    session_tool_call_count_final: finalSnapshot?.tool_call_count ?? null,
    session_mcp_tool_call_count_final: finalSnapshot?.mcp_tool_call_count ?? null,
    delta_message_count_sum: sumOrNull(deltaMessageCounts),
    delta_tool_call_count_sum: sumOrNull(deltaToolCallCounts),
    delta_mcp_tool_call_count_sum: sumOrNull(deltaMcpToolCallCounts),
    delta_message_count_max: maxOrNull(deltaMessageCounts),
    delta_tool_call_count_max: maxOrNull(deltaToolCallCounts),
    delta_mcp_tool_call_count_max: maxOrNull(deltaMcpToolCallCounts),
    snapshot_gap_turn_ids: snapshotGapTurnIds,
    snapshot_regression_turn_ids: snapshotRegressionTurnIds,
    context_drop_turn_ids: contextDropTurnIds,
    timeout_count: timeoutTurnIds.length,
    timeout_turn_ids: timeoutTurnIds,
    timeout_classifications: timeoutClassifications,
    timeout_last_tool_names: timeoutLastToolNames,
    timeout_last_tool_statuses: timeoutLastToolStatuses,
    timeout_assistant_states: timeoutAssistantStates,
    timeout_abort_statuses: timeoutAbortStatuses,
    timeout_idle_statuses: timeoutIdleStatuses,
    session_not_idle_turn_ids: sessionNotIdleTurnIds,
    error_turn_ids: errorTurnIds,
    error_codes: errorCodes,
    error_status_codes: errorStatusCodes,
    error_retryable_flags: errorRetryable,
    tool_names: [...toolNames].sort().slice(0, MAX_SUMMARY_TOOL_NAMES),
    compaction_status: compactionStatus,
    compaction_reason: compactionReason,
    compaction_observed_count: compactedObservedCount,
  };
}

/** Derive a deterministic per-arm metrics path, keeping both arms isolated. */
export function metricsPathForRun(opts: Options, arm: Arm, runId: string, env: Env = process.env): string {
  const configured = opts.metricsFile || env[METRICS_FILE_ENV];
  if (configured) {
    return resolve(configured.replaceAll("{arm}", arm).replaceAll("{run_id}", runId));
  }
  return join(resolve(opts.outputDir), `${runId}-${arm}-metrics.jsonl`);
}

/**
 * Build the payload the Python `finalize` bridge consumes for one arm.
 *
 * For a live run the admission/budget/price/stat-plan block is bound here: `admission_policy_digest`
 * is a deterministic sha256 over the approval id and budget, and `price_provenance` is an operator
 * provenance label. The arm-A tuning values are included as numeric, payload-free metadata in both the
 * finalize payload and `host_provenance`; raw turn answers remain confined to the in-memory `turns`
 * handoff and are never copied into provenance.
 *
 * The model provenance records the selected agent model separately from the reranker model; both are
 * identifiers only and never include prompts, answers, credentials or tool-result payloads.
 *
 * @param arm - The arm.
 * @param runId - The arm's run id.
 * @param opts - The parsed options.
 * @param hostVersion - The OpenCode host version string.
 * @param observations - The 60 payload-free observations.
 * @param mode - `offline` or `live`.
 * @param metricsSummary - Sanitized bridge metrics to copy into host provenance.
 * @param rerankModelId - The reranking model id recorded separately from the agent model.
 * @returns The finalize payload.
 */
export function buildFinalizePayload(
  arm: Arm,
  runId: string,
  opts: Options,
  hostVersion: string,
  observations: TurnObservation[],
  mode: "offline" | "live",
  metricsSummary: Record<string, unknown> = {},
  rerankModelId: string = DEFAULT_RERANK_MODEL_ID,
  removedEnvKeys: string[] = [],
): Record<string, unknown> {
  const tuning = calibrationTuning(opts);
  const toolSpecSummary = summarizeToolSpecMetrics(observations);
  const observabilitySummary = summarizeObservability(observations);
  const modelProvenance = {
    execution_model_id: mode === "live" && arm === "A" ? opts.model : null,
    rerank_model_id: rerankModelId,
  };
  const payload: Record<string, unknown> = {
    mode,
    arm,
    run_id: runId,
    host: "opencode",
    host_version: hostVersion,
    model_id: opts.model ?? "unknown",
    output_dir: resolve(opts.outputDir),
    tuning,
    metrics_summary: metricsSummary,
    observability_summary: observabilitySummary,
    // Declared tool policy, identical for both arms; recorded in host evidence for provenance.
    disabled_tools: opts.disabledTools.join(","),
    removed_env_keys: removedEnvKeys.join(","),
    // Only the digest of a declared, arm-identical system prompt is recorded, never its text.
    system_prompt_digest: systemPromptDigest(opts),
    // The plugin's retrieval behaviour (arm A only; the baseline loads no plugin).
    plugin_options: arm === "A" ? pluginOptions(process.env) : "",
    host_provenance: {
      host: "opencode",
      host_version: hostVersion,
      arm,
      tuning,
      model_provenance: modelProvenance,
      observability_summary: observabilitySummary,
      ...toolSpecSummary,
      ...metricsSummary,
    },
    turns: observations,
  };
  if (mode === "live") {
    const statPlan = JSON.parse(readFileSync(opts.statPlan as string, "utf8"));
    const policyDigest =
      "sha256:" +
      createHash("sha256")
        .update(`${opts.approvalId}|${opts.budgetCeiling}|${opts.budgetUnit}`)
        .digest("hex");
    payload.live = {
      execution_admission_id: opts.approvalId,
      admission_policy_digest: policyDigest,
      budget_ceiling_units: Number(opts.budgetCeiling),
      budget_currency_or_unit: opts.budgetUnit,
      price_provenance: `operator-declared:${opts.budgetUnit}:${opts.approvalId}`,
      stat_plan: statPlan,
    };
  }
  return payload;
}

// --- OpenCode process control ------------------------------------------------------------------

/** A started OpenCode server: its base URL and the child process backing it. */
export interface StartedServer {
  url: string;
  proc: ChildProcess;
}

/**
 * Spawn `opencode serve` on 127.0.0.1 with an ephemeral port and resolve once it prints its URL.
 *
 * The `baseline` arm adds `--pure` so no plugin loads; the `A` arm runs normally with the project root
 * as the working directory so `.opencode/plugins/context-engineering.ts` is discovered.
 *
 * @param arm - The arm being started.
 * @param env - The arm's environment.
 * @param timeoutMs - How long to wait for the listening line.
 * @returns The started server handle.
 */
export function startOpencode(
  arm: Arm,
  env: Env,
  timeoutMs = 30000,
): Promise<StartedServer> {
  const args = ["serve", "--hostname=127.0.0.1", "--port=0"];
  if (arm === "baseline") args.push("--pure");
  const proc = spawn("opencode", args, {
    cwd: PROJECT_ROOT,
    env,
    stdio: ["ignore", "pipe", "pipe"],
  });
  return new Promise<StartedServer>((resolvePromise, reject) => {
    let output = "";
    let settled = false;
    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      proc.kill("SIGTERM");
      reject(new Error(`timeout waiting for opencode to start after ${timeoutMs}ms`));
    }, timeoutMs);
    const onData = (chunk: Buffer): void => {
      if (settled) return;
      output += chunk.toString();
      const match = output.match(/opencode server listening on\s+(https?:\/\/[^\s]+)/);
      if (match) {
        settled = true;
        clearTimeout(timer);
        resolvePromise({ url: match[1], proc });
      }
    };
    proc.stdout?.on("data", onData);
    proc.stderr?.on("data", (chunk) => (output += chunk.toString()));
    proc.on("error", (error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      reject(error);
    });
    proc.on("exit", (code) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      reject(new Error(`opencode exited early with code ${code}: ${output}`));
    });
  });
}

/**
 * Poll `/global/health` until the server answers 200 or the deadline passes.
 *
 * @param url - The server base URL.
 * @param timeoutMs - Overall deadline.
 */
export async function waitForHealth(url: string, timeoutMs = 30000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    try {
      const response = await fetch(`${url}/global/health`);
      if (response.ok) return;
    } catch {
      // server not up yet; retry until the deadline
    }
    if (Date.now() > deadline) throw new Error(`opencode /global/health not ready within ${timeoutMs}ms`);
    await new Promise((r) => setTimeout(r, 250));
  }
}

/** Stop an OpenCode child process; safe to call more than once. Prefer {@link reapProcessTree}. */
export function stopOpencode(proc: ChildProcess): void {
  if (!proc.killed) {
    try {
      proc.kill("SIGTERM");
    } catch {
      /* already gone */
    }
  }
}

/** Payload-free result of tearing down one OpenCode process tree. */
export interface CleanupResult {
  tracked_process_count: number;
  sigterm_sent: boolean;
  sigkill_needed: boolean;
  residual_process_count: number;
}

/** List every descendant pid of `rootPid` from one `ps` snapshot (breadth-first). */
export function listDescendantPids(rootPid: number): number[] {
  const snapshot = spawnSync("ps", ["-axo", "pid=,ppid="], { encoding: "utf8" });
  if (snapshot.status !== 0 || typeof snapshot.stdout !== "string") return [];
  const children = new Map<number, number[]>();
  for (const line of snapshot.stdout.split("\n")) {
    const [pidText, ppidText] = line.trim().split(/\s+/);
    const pid = Number(pidText);
    const ppid = Number(ppidText);
    if (!Number.isInteger(pid) || !Number.isInteger(ppid) || pid <= 0) continue;
    const siblings = children.get(ppid) ?? [];
    siblings.push(pid);
    children.set(ppid, siblings);
  }
  const descendants: number[] = [];
  const queue = [...(children.get(rootPid) ?? [])];
  while (queue.length > 0) {
    const pid = queue.shift() as number;
    if (pid === rootPid || descendants.includes(pid)) continue;
    descendants.push(pid);
    queue.push(...(children.get(pid) ?? []));
  }
  return descendants;
}

/** Whether a pid still exists (EPERM means it exists but belongs to someone else). */
export function isPidAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return (error as { code?: string }).code === "EPERM";
  }
}

/**
 * Tear down a child and every descendant: SIGTERM, wait up to `graceMs`, SIGKILL survivors, verify.
 *
 * Descendants (the MCP fixture server, the plugin bridge) are captured BEFORE the root exits, because
 * once the root dies they are re-parented and can no longer be found by parent pid.
 */
export async function reapProcessTree(proc: ChildProcess, graceMs = 10000): Promise<CleanupResult> {
  const rootPid = proc.pid;
  if (rootPid === undefined) {
    return { tracked_process_count: 0, sigterm_sent: false, sigkill_needed: false, residual_process_count: 0 };
  }
  const tracked = [rootPid, ...listDescendantPids(rootPid)];
  const signal = (pid: number, name: "SIGTERM" | "SIGKILL"): void => {
    try {
      process.kill(pid, name);
    } catch {
      /* already gone */
    }
  };
  for (const pid of tracked) if (isPidAlive(pid)) signal(pid, "SIGTERM");
  const deadline = Date.now() + graceMs;
  while (Date.now() < deadline && tracked.some(isPidAlive)) {
    await new Promise((r) => setTimeout(r, 100));
  }
  const survivors = tracked.filter(isPidAlive);
  for (const pid of survivors) signal(pid, "SIGKILL");
  if (survivors.length > 0) await new Promise((r) => setTimeout(r, 300));
  return {
    tracked_process_count: tracked.length,
    sigterm_sent: true,
    sigkill_needed: survivors.length > 0,
    residual_process_count: tracked.filter(isPidAlive).length,
  };
}

/** Minimal structural client used by the abort/idle verification (the real SDK client satisfies it). */
export interface AbortClient {
  session: {
    abort: (options: { path: { id: string } }) => Promise<unknown>;
    status: () => Promise<unknown>;
  };
}

/** Read one session's public status: `idle`, `busy`, `retry` or `unknown`. */
export async function readSessionStatus(
  client: Pick<AbortClient, "session">,
  sessionId: string,
  timeoutMs = 5000,
): Promise<"idle" | "busy" | "retry" | "unknown"> {
  try {
    const response = await withTimeout(client.session.status(), timeoutMs);
    if (response === TIMEOUT) return "unknown";
    const data = isRecord(response) && "data" in response ? response.data : response;
    if (!isRecord(data)) return "unknown";
    const entry = data[sessionId];
    // OpenCode only lists sessions that are not idle; an absent entry for a known session is idle.
    if (entry === undefined) return "idle";
    if (isRecord(entry) && (entry.type === "idle" || entry.type === "busy" || entry.type === "retry")) {
      return entry.type;
    }
    return "unknown";
  } catch {
    return "unknown";
  }
}

/**
 * Abort a timed-out session and verify it actually returned to idle before another prompt is sent.
 *
 * Never swallows the outcome: the abort result and the idle verdict are returned and persisted.
 */
export async function verifyAbort(
  client: AbortClient,
  sessionId: string,
  { abortTimeoutMs = 15000, idleTimeoutMs = 30000, pollMs = 1000 } = {},
): Promise<{ abort_status: "ok" | "error" | "timeout"; idle_status: "idle" | "not_idle" | "unknown" }> {
  let abortStatus: "ok" | "error" | "timeout" = "error";
  try {
    const result = await withTimeout(client.session.abort({ path: { id: sessionId } }), abortTimeoutMs);
    if (result === TIMEOUT) abortStatus = "timeout";
    else abortStatus = isRecord(result) && result.error ? "error" : "ok";
  } catch {
    abortStatus = "error";
  }
  let idleStatus: "idle" | "not_idle" | "unknown" = "unknown";
  const deadline = Date.now() + idleTimeoutMs;
  for (;;) {
    const status = await readSessionStatus(client, sessionId);
    if (status === "idle") {
      idleStatus = "idle";
      break;
    }
    if (status === "busy" || status === "retry") idleStatus = "not_idle";
    if (Date.now() >= deadline) break;
    await new Promise((r) => setTimeout(r, pollMs));
  }
  return { abort_status: abortStatus, idle_status: idleStatus };
}

const SAFE_CODE = /^[A-Za-z0-9_.:-]{1,64}$/;

/**
 * Build the payload-free artifact of a truncated diagnostic run (`--diagnostic-stop-after`).
 *
 * It is explicitly `evidence_class: diagnostic_truncated` and is never fed to Gate A. Only scalars,
 * enums and bounded tool names are copied; prompts and answers are never read here.
 */
export function buildDiagnosticArtifact(params: {
  arm: Arm;
  runId: string;
  opts: Options;
  hostVersion: string;
  turns: Turn[];
  observations: TurnObservation[];
  metricsSummary: Record<string, unknown>;
  removedEnvKeys: string[];
}): Record<string, unknown> {
  const { arm, runId, opts, hostVersion, turns, observations, metricsSummary, removedEnvKeys } = params;
  const turnOrderDigest =
    "sha256:" + createHash("sha256").update(turns.map((turn) => turn.turn_id).join("\n")).digest("hex");
  return {
    evidence_class: "diagnostic_truncated",
    diagnostic_schema_version: 1,
    host: "opencode",
    host_version: hostVersion,
    arm,
    run_id: runId,
    model_id: opts.model,
    stop_after_turn_id: opts.diagnosticStopAfter,
    turns_resolved: turns.length,
    turns_sent: observations.length,
    turn_order_digest: turnOrderDigest,
    prompt_timeout_ms: opts.promptTimeoutMs,
    disabled_tools: [...opts.disabledTools],
    system_prompt_digest: systemPromptDigest(opts),
    removed_env_keys: [...removedEnvKeys],
    turns: observations.map((observation) => ({
      turn_id: observation.turn_id,
      host_signal: observation.host_signal,
      error_code:
        observation.error_code === null ? null : SAFE_CODE.test(observation.error_code) ? observation.error_code : "other",
      latency_ms: observation.latency_ms,
      model_input: observation.model_input,
      telemetry_status: observation.telemetry_status,
      message_count: observation.message_count,
      tool_call_count: observation.tool_call_count,
      mcp_tool_call_count: observation.mcp_tool_call_count,
      retrieval_status: observation.retrieval_status,
      retrieval_attempts: observation.retrieval_attempts,
      retrieval_successes: observation.retrieval_successes,
      tool_names: (observation.tool_names ?? [])
        .map((name) => safeToolName(name))
        .filter((name): name is string => name !== null)
        .slice(0, MAX_TOOL_NAMES_PER_TURN),
      assistant: observation.assistant ?? null,
      timeout_classification: observation.timeout_classification ?? null,
      abort_status: observation.abort_status ?? null,
      idle_status: observation.idle_status ?? null,
    })),
    observability: summarizeObservability(observations),
    metrics: metricsSummary,
  };
}

// --- Live run ----------------------------------------------------------------------------------

/**
 * Run one arm live: start OpenCode, create a session, send 60 prompts sequentially, finalise.
 *
 * The OpenCode process is ALWAYS stopped in `finally`, whether the run succeeded, failed, or threw.
 *
 * @param arm - The arm to run.
 * @param runId - The arm's run id.
 * @param turns - The 60 turns.
 * @param opts - The parsed options.
 * @param env - The process environment.
 * @param runPython - Injected Python runner.
 * @returns The parsed evidence summary emitted by the Python bridge.
 */
export async function runArmLive(
  arm: Arm,
  runId: string,
  turns: Turn[],
  opts: Options,
  env: Env,
  runPython: PythonRunner = defaultPythonRunner,
): Promise<Record<string, unknown>> {
  const metricsFile = metricsPathForRun(opts, arm, runId, env);
  mkdirSync(dirname(metricsFile), { recursive: true });
  const removedEnvKeys = inheritedOpencodeEnvKeys(env);
  // Practice D, when asked for, writes one payload-free record per model request; start each run from an empty file.
  const graphEvidenceFile = arm === "A" && parseGraphFlag(env.CONTEXT_GRAPH) ? graphEvidencePathForRun(opts, arm) : undefined;
  if (graphEvidenceFile) {
    mkdirSync(dirname(graphEvidenceFile), { recursive: true });
    writeFileSync(graphEvidenceFile, "");
  }
  const armEnv = buildArmEnv(arm, env, opts, metricsFile, graphEvidenceFile);
  const toolsMap = promptToolsMap(opts);
  const systemText = promptSystem(opts);
  const stopIndex = opts.diagnosticStopAfter
    ? turns.findIndex((turn) => turn.turn_id === opts.diagnosticStopAfter)
    : -1;
  if (opts.diagnosticStopAfter && stopIndex < 0) {
    throw new Error(`--diagnostic-stop-after ${opts.diagnosticStopAfter} is not a resolved turn id`);
  }
  const turnsToRun = stopIndex >= 0 ? turns.slice(0, stopIndex + 1) : turns;
  const started = await startOpencode(arm, armEnv, opts.startupTimeoutMs);
  try {
    await waitForHealth(started.url);
    const client = createOpencodeClient({ baseUrl: started.url });
    const hostVersion = await readOpencodeVersion();

    const session = await client.session.create({ body: { title: `ce60-${arm}-${runId}` } });
    const sessionId = (session.data as { id?: string } | undefined)?.id;
    if (!sessionId) throw new Error(`could not create an OpenCode session for arm ${arm}`);

    const [providerID, ...rest] = (opts.model as string).split("/");
    const modelID = rest.join("/");

    // Only arm A loads the plugin, which registers the `retrieve_context` tool, so only arm A exposes
    // a CE-60 retrieval count. Baseline runs `--pure` with no such tool: retrieval is host-not-exposed.
    const retrievalExposed = arm === "A";
    // Set when an abort could not return the session to idle: no further prompt is sent into a
    // session of unknown state; the remaining turns are recorded as errors (fail closed).
    let sessionUnusable = false;
    // Cumulative `retrieve_context` parts at the previous successful session read: a turn's retrieval
    // count is the growth since then, so a call made in a non-final step is still attributed to its turn.
    const retrievalTracker = newRetrievalTracker();

    const send: SendTurn = async (turn) => {
      if (sessionUnusable) {
        return erroredObservation(turn.turn_id, "SESSION_NOT_IDLE", retrievalExposed);
      }
      const requestStartedAt = Date.now();
      let result: unknown;
      try {
        result = await withTimeout(
          client.session.prompt({
            path: { id: sessionId },
            body: {
              model: { providerID, modelID },
              ...(toolsMap ? { tools: toolsMap } : {}),
              ...(systemText !== undefined ? { system: systemText } : {}),
              parts: [{ type: "text", text: turn.prompt }],
            },
          }),
          opts.promptTimeoutMs,
        );
      } catch {
        const telemetry = await collectSessionTelemetry(client, sessionId);
        return erroredObservation(
          turn.turn_id,
          "PROMPT_ERROR",
          retrievalExposed,
          telemetry,
          Date.now() - requestStartedAt,
        );
      }

      // Read only sanitized session counts after every prompt, including the timeout path, before any
      // abort call. The prompt response itself remains on the existing in-memory oracle path.
      const telemetry = await collectSessionTelemetry(client, sessionId);
      const wallLatencyMs = Date.now() - requestStartedAt;
      if (result === TIMEOUT) {
        const observation = timedOutObservation(
          turn.turn_id,
          retrievalExposed,
          telemetry,
          wallLatencyMs,
        );
        // The state was captured above, before the abort; classify it, then abort and verify idle.
        observation.timeout_classification = classifyTimeoutState(observation);
        const abort = await verifyAbort(client as unknown as AbortClient, sessionId);
        observation.abort_status = abort.abort_status;
        observation.idle_status = abort.idle_status;
        if (abort.idle_status === "not_idle") sessionUnusable = true;
        console.error(
          `[ce60] ${arm} ${turn.turn_id} timeout class=${observation.timeout_classification} ` +
            `last_tool=${observation.assistant?.last_tool?.name ?? "none"}/${observation.assistant?.last_tool?.status ?? "none"} ` +
            `abort=${abort.abort_status} idle=${abort.idle_status}`,
        );
        return observation;
      }
      const data = (result as { data?: unknown; error?: unknown }).data;
      if (!data) {
        return erroredObservation(
          turn.turn_id,
          "PROMPT_FAILED",
          retrievalExposed,
          telemetry,
          wallLatencyMs,
        );
      }
      return attributeSessionRetrieval(
        attachSessionTelemetry(
          parsePromptResponse(turn.turn_id, data, { retrievalExposed }),
          telemetry,
          wallLatencyMs,
        ),
        telemetry,
        retrievalTracker,
        retrievalExposed,
      );
    };

    const observations = await runTurnsSequential(turnsToRun, send);
    if (graphEvidenceFile) {
      // Read the stored session back and compare it with what each request was handed: this is what shows, rather than
      // asserts, that the graph is request-only. Best effort; a failure is recorded in the file, never raised.
      try {
        const outDir = resolve(opts.outputDir);
        mkdirSync(outDir, { recursive: true });
        const durability = await writeGraphDurability(client, sessionId, graphEvidenceFile, outDir, arm, runId);
        console.error(
          `[ce60] ${arm} graph: requests=${durability.requests} projected=${(durability.operations as Record<string, number>).projected ?? 0} ` +
            `stored_prefix_equal=${durability.stored_prefix_equal}/${durability.checked} synthetic_stored=${durability.stored_graph_synthetic_messages}`,
        );
      } catch (error) {
        console.error(`[ce60] ${arm} graph durability check failed: ${String(error)}`);
      }
    }
    const metricsSummary = summarizeMetricsFile(metricsFile, arm === "A");
    if (opts.diagnosticStopAfter) {
      // Truncated diagnostic: write the payload-free diagnostic artifact; never Gate A evidence.
      const artifact = buildDiagnosticArtifact({
        arm,
        runId,
        opts,
        hostVersion,
        turns,
        observations,
        metricsSummary,
        removedEnvKeys,
      });
      const outDir = resolve(opts.outputDir);
      mkdirSync(outDir, { recursive: true });
      const path = join(outDir, `${arm}-diagnostic.json`);
      writeFileSync(path, JSON.stringify(artifact, null, 2) + "\n");
      const observability = artifact.observability as Record<string, unknown>;
      return {
        arm,
        run_id: runId,
        evidence_class: "diagnostic_truncated",
        turns_sent: observations.length,
        timeout_turn_ids: observability.timeout_turn_ids,
        timeout_classifications: observability.timeout_classifications,
        path,
      };
    }
    const payload = buildFinalizePayload(
      arm,
      runId,
      opts,
      hostVersion,
      observations,
      "live",
      metricsSummary,
      armEnv.CONTEXT_RERANK_MODEL_ID || DEFAULT_RERANK_MODEL_ID,
      removedEnvKeys,
    );
    const { code, stdout, stderr } = await runPython(
      ["-m", "opencode_gate_a.host_driver", "finalize"],
      VALIDATION_DIR,
      JSON.stringify(payload),
    );
    if (code !== 0) throw new Error(`finalize failed for arm ${arm} (exit ${code}): ${stderr.trim()}`);
    return JSON.parse(stdout) as Record<string, unknown>;
  } finally {
    const cleanup = await reapProcessTree(started.proc);
    try {
      const outDir = resolve(opts.outputDir);
      mkdirSync(outDir, { recursive: true });
      writeFileSync(
        join(outDir, `${arm}-cleanup.json`),
        JSON.stringify({ arm, run_id: runId, ...cleanup }, null, 2) + "\n",
      );
    } catch {
      /* the cleanup record is best-effort; the residual count is also logged below */
    }
    if (cleanup.residual_process_count > 0) {
      console.error(`[ce60] WARNING ${arm}: ${cleanup.residual_process_count} residual process(es) after cleanup`);
    }
  }
}

/** One line of the graph's evidence file, as far as the durability check reads it. */
interface GraphEvidenceLine {
  request_index: number;
  session_id: string | null;
  operation: string;
  mode: string | null;
  received: { count: number; first_id: string | null; last_id: string | null; digest: string };
  removed: number;
  block_chars: number;
  rejected: string | null;
}

/**
 * Compare what every graph request was handed with the session as OpenCode stored it.
 *
 * Each record carries the digest of the messages OpenCode gave the hook, taken before anything edited them. The stored
 * session is the host's own copy: if a request had written to it, the stored slice would no longer reproduce that digest.
 * Pure, so it is tested without a host. Counts and booleans only; no message text is kept.
 */
export function computeGraphDurability(
  storedBySession: Record<string, GraphMessage[]>,
  records: GraphEvidenceLine[],
): Record<string, unknown> {
  const operations: Record<string, number> = {};
  const modes: Record<string, number> = {};
  const mismatches: number[] = [];
  // The digest of the stored slice for each request, read back after it ran: "transcript after" in the Gate D evidence.
  const storedDigests: Record<string, string> = {};
  let checked = 0;
  let equal = 0;
  let removed = 0;
  let blockChars = 0;
  let rejected = 0;
  for (const record of records) {
    operations[record.operation] = (operations[record.operation] ?? 0) + 1;
    if (record.mode) modes[record.mode] = (modes[record.mode] ?? 0) + 1;
    removed += record.removed;
    blockChars += record.block_chars;
    if (record.rejected) rejected++;
    const stored = record.session_id ? storedBySession[record.session_id] : undefined;
    if (!stored) continue;
    checked++;
    const ids = stored.map((message) => message?.info?.id);
    const first = ids.indexOf(record.received.first_id ?? undefined);
    const last = ids.indexOf(record.received.last_id ?? undefined);
    const prefix = first >= 0 && last >= first ? stored.slice(first, last + 1) : [];
    const storedDigest = digestOf(fingerprintTranscript(prefix));
    storedDigests[String(record.request_index)] = storedDigest;
    const same = prefix.length === record.received.count && storedDigest === record.received.digest;
    if (same) equal++;
    else mismatches.push(record.request_index);
  }
  let synthetic = 0;
  const graphToolCalls: Record<string, number> = {};
  for (const messages of Object.values(storedBySession)) {
    for (const message of messages) {
      if (typeof message?.info?.id === "string" && message.info.id.includes("zzzgraph")) synthetic++;
      for (const part of message?.parts ?? []) {
        if (part?.type === "tool" && (part.tool === "expand_card" || part.tool === "find_context")) {
          graphToolCalls[part.tool] = (graphToolCalls[part.tool] ?? 0) + 1;
        }
      }
    }
  }
  return {
    requests: records.length,
    checked,
    stored_prefix_equal: equal,
    mismatched_requests: mismatches,
    stored_digests: storedDigests,
    stored_graph_synthetic_messages: synthetic,
    operations,
    modes,
    removed_messages_total: removed,
    block_chars_total: blockChars,
    rejected_requests: rejected,
    graph_tool_calls: graphToolCalls,
  };
}

/** Read the evidence file and the stored session(s), compute the durability summary and write it next to the run. */
export async function writeGraphDurability(
  client: { session: { messages: (options: { path: { id: string } }) => Promise<unknown> } },
  sessionId: string,
  evidencePath: string,
  outDir: string,
  arm: Arm,
  runId: string,
): Promise<Record<string, unknown>> {
  const records: GraphEvidenceLine[] = readFileSync(evidencePath, "utf8")
    .split("\n")
    .filter((line) => line.trim() !== "")
    .map((line) => JSON.parse(line) as GraphEvidenceLine);
  const sessions = new Set<string>([sessionId]);
  for (const record of records) if (record.session_id) sessions.add(record.session_id);
  const storedBySession: Record<string, GraphMessage[]> = {};
  for (const id of sessions) {
    const response = (await client.session.messages({ path: { id } })) as { data?: unknown } | undefined;
    const data = response?.data;
    if (Array.isArray(data)) storedBySession[id] = data as GraphMessage[];
  }
  const summary = {
    arm,
    run_id: runId,
    sessions_checked: Object.keys(storedBySession).length,
    ...computeGraphDurability(storedBySession, records),
  };
  writeFileSync(join(outDir, `${arm}-graph-durability.json`), JSON.stringify(summary, null, 2) + "\n");
  return summary;
}

const TIMEOUT = Symbol("timeout");

/** Race a promise against a wall timeout, resolving to {@link TIMEOUT} when it expires. */
async function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T | typeof TIMEOUT> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const timeout = new Promise<typeof TIMEOUT>((r) => {
    timer = setTimeout(() => r(TIMEOUT), ms);
  });
  try {
    return await Promise.race([promise, timeout]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

/**
 * Read only public session history metadata after a prompt. A failed or timed-out read returns null
 * counts rather than zero, so an unavailable diagnostic is never mistaken for an empty session.
 */
export async function collectSessionTelemetry(
  client: { session: { messages: (options: { path: { id: string } }) => Promise<unknown> } },
  sessionId: string,
  timeoutMs = 5000,
): Promise<SessionTelemetry> {
  try {
    const response = await withTimeout(
      client.session.messages({ path: { id: sessionId } }),
      timeoutMs,
    );
    if (response === TIMEOUT) {
      return {
        telemetry_status: "timeout",
        message_count: null,
        tool_call_count: null,
        mcp_tool_call_count: null,
        session_compacted_observed: null,
        tool_names: [],
      };
    }
    return parseSessionMessagesResponse(response);
  } catch {
    return {
      telemetry_status: "error",
      message_count: null,
      tool_call_count: null,
      mcp_tool_call_count: null,
      session_compacted_observed: null,
      tool_names: [],
    };
  }
}

/**
 * An observation for a turn that exceeded its wall timeout.
 *
 * Counts are collected from the public session endpoint before aborting the session. Missing counts stay
 * null; no timeout path fabricates an empty session.
 */
export function timedOutObservation(
  turnId: string,
  retrievalExposed: boolean,
  telemetry: SessionTelemetry = {
    telemetry_status: "not_attempted",
    message_count: null,
    tool_call_count: null,
    mcp_tool_call_count: null,
    session_compacted_observed: null,
    tool_names: [],
  },
  latencyMs: number | null = null,
): TurnObservation {
  return {
    turn_id: turnId,
    host_signal: "timeout",
    matcher_passed: false,
    answer_text: "",
    tool_calls: [...telemetry.tool_names],
    tool_names: [...telemetry.tool_names],
    ...telemetry,
    timeout: true,
    model_input: null,
    latency_ms: latencyMs,
    retrieval_status: retrievalExposed ? "collection_failed" : "host_not_exposed",
    retrieval_attempts: null,
    retrieval_successes: null,
    tool_spec_tokens: null,
    tool_spec_count: null,
    error_code: "TURN_TIMEOUT",
  };
}

/** An observation for a turn whose prompt call failed outright. */
export function erroredObservation(
  turnId: string,
  code: string,
  retrievalExposed: boolean,
  telemetry: SessionTelemetry = {
    telemetry_status: "not_attempted",
    message_count: null,
    tool_call_count: null,
    mcp_tool_call_count: null,
    session_compacted_observed: null,
    tool_names: [],
  },
  latencyMs: number | null = null,
): TurnObservation {
  return {
    turn_id: turnId,
    host_signal: "error",
    matcher_passed: false,
    answer_text: "",
    tool_calls: [...telemetry.tool_names],
    tool_names: [...telemetry.tool_names],
    ...telemetry,
    timeout: false,
    model_request_started: true,
    model_input: null,
    latency_ms: latencyMs,
    retrieval_status: retrievalExposed ? "collection_failed" : "host_not_exposed",
    retrieval_attempts: null,
    retrieval_successes: null,
    tool_spec_tokens: null,
    tool_spec_count: null,
    error_code: code,
  };
}

/** Read the installed OpenCode version string, falling back to "unknown" on any failure. */
function readOpencodeVersion(): Promise<string> {
  return new Promise<string>((resolvePromise) => {
    try {
      const child = spawn("opencode", ["--version"], { stdio: ["ignore", "pipe", "pipe"] });
      let out = "";
      child.stdout?.on("data", (c) => (out += c.toString()));
      child.on("close", () => resolvePromise(out.trim() || "unknown"));
      child.on("error", () => resolvePromise("unknown"));
    } catch {
      resolvePromise("unknown");
    }
  });
}

// --- Plan (dry run) ----------------------------------------------------------------------------

/**
 * Build the dry-run plan summary. Secrets (AWS, approval id) are redacted to presence booleans.
 *
 * @param opts - The parsed options.
 * @param env - The process environment.
 * @param turns - The resolved turns.
 * @returns A JSON-safe plan object.
 */
export function planSummary(
  opts: Options,
  env: Env,
  turns: Turn[],
): Record<string, unknown> {
  const arms = opts.arm === "both" ? [...ARMS] : [opts.arm];
  return {
    mode: opts.executeLive ? "execute-live" : "dry-run",
    arms,
    turn_count: turns.length,
    model: opts.model,
    output_dir: resolve(opts.outputDir),
    project_root: PROJECT_ROOT,
    retrieval_metric:
      "arm A: counted from retrieve_context tool calls; baseline (--pure): host_not_exposed",
    aws_profile_present: Boolean(env.AWS_PROFILE),
    aws_region_present: Boolean(env.AWS_REGION || env.AWS_DEFAULT_REGION),
    approval_id_present: Boolean(opts.approvalId),
    budget: opts.budgetCeiling ? `${opts.budgetCeiling} ${opts.budgetUnit ?? ""}`.trim() : null,
    stat_plan: opts.statPlan,
    diagnostic_stop_after: opts.diagnosticStopAfter,
    turns_to_send: opts.diagnosticStopAfter
      ? turns.findIndex((turn) => turn.turn_id === opts.diagnosticStopAfter) + 1
      : turns.length,
    disabled_tools: [...opts.disabledTools],
    prompt_tools_map: promptToolsMap(opts) ?? null,
    system_prompt_set: promptSystem(opts) !== undefined,
    system_prompt_digest: systemPromptDigest(opts),
    removed_env_keys: inheritedOpencodeEnvKeys(env),
    tuning: calibrationTuning(opts),
    arm_A_env_keys: [
      "CONTEXT_EXECUTION_MODE",
      "CONTEXT_APPROVAL_ID",
      "CONTEXT_BUDGET_CEILING",
      "CONTEXT_BUDGET_UNIT",
      "CONTEXT_EXECUTION_ALLOWED_MODELS",
      "CONTEXT_EXECUTION_MODEL_ID",
      "CONTEXT_RERANK_MODEL_ID",
      "CONTEXT_MAX_RESULT_TOKENS",
      "CONTEXT_PREVIEW_TOKENS",
      "CONTEXT_CHUNK_TOKENS",
      "CONTEXT_RELEVANCE_THRESHOLD",
    ],
    note: "DRY RUN: no OpenCode process is started and no model is called. Pass --execute-live to run.",
  };
}

/** Run the local bridge diagnostic without resolving turns or starting OpenCode. */
export async function runDiagnosticFixture(
  opts: Options,
  env: Env,
  runPython: PythonRunner = defaultPythonRunner,
): Promise<number> {
  if (opts.executeLive) {
    console.error("--diagnostic-fixture cannot be combined with --execute-live");
    return 2;
  }
  const metricsFile = metricsPathForRun(opts, "A", "diagnostic-fixture", env);
  const diagnosticEnv: Env = {
    ...env,
    AWS_PROFILE: env.AWS_PROFILE || "offline-fixture",
    [METRICS_FILE_ENV]: metricsFile,
    CONTEXT_MAX_RESULT_TOKENS: "100000",
  };
  const result = await runPython(
    ["-m", "context_core.opencode_bridge", "--diagnostic-fixture"],
    PROJECT_ROOT,
    null,
    diagnosticEnv,
  );
  if (result.code !== 0) {
    console.error(result.stderr.trim() || "offline diagnostic fixture failed");
    return 4;
  }
  try {
    console.log(JSON.stringify(JSON.parse(result.stdout), null, 2));
  } catch {
    console.error("offline diagnostic fixture returned invalid JSON");
    return 4;
  }
  return 0;
}

// --- Entry point -------------------------------------------------------------------------------

/**
 * Driver entry point. Returns a process exit code; never calls `process.exit` directly so it stays
 * testable.
 *
 * @param argv - The argument vector tail.
 * @param env - The process environment.
 * @param runPython - Injected Python runner.
 * @returns The exit code.
 */
export async function main(
  argv: string[],
  env: Env = process.env,
  runPython: PythonRunner = defaultPythonRunner,
): Promise<number> {
  let opts: Options;
  try {
    opts = parseArgs(argv);
  } catch (error) {
    console.error(String(error instanceof Error ? error.message : error));
    console.error("\n" + HELP_TEXT);
    return 2;
  }
  if (opts.help) {
    console.log(HELP_TEXT);
    return 0;
  }

  const calibrationErrors = validateCalibrationOptions(opts);
  if (calibrationErrors.length > 0) {
    console.error("invalid calibration configuration:\n  - " + calibrationErrors.join("\n  - "));
    return 2;
  }

  if (opts.diagnosticFixture) {
    return runDiagnosticFixture(opts, env, runPython);
  }

  const turns = await resolveTurns(opts, runPython);
  const arms: Arm[] = opts.arm === "both" ? [...ARMS] : [opts.arm];
  if (opts.diagnosticStopAfter && !turns.some((turn) => turn.turn_id === opts.diagnosticStopAfter)) {
    console.error(`--diagnostic-stop-after ${opts.diagnosticStopAfter} is not one of the resolved turn ids`);
    return 2;
  }

  if (!opts.executeLive) {
    console.log(JSON.stringify(planSummary(opts, env, turns), null, 2));
    return 0;
  }

  const errors = validateLiveConfig(opts, env);
  if (errors.length > 0) {
    console.error("cannot start a live run:\n  - " + errors.join("\n  - "));
    return 2;
  }

  const baseRunId = opts.runId as string;
  const evidences: Record<string, unknown>[] = [];
  for (const arm of arms) {
    const runId = `${baseRunId}-${arm}`;
    console.error(`[ce60] running arm ${arm} (${runId}) ...`);
    const evidence = await runArmLive(arm, runId, turns, opts, env, runPython);
    evidences.push(evidence);
  }

  // A truncated diagnostic is never Gate A evidence, so the decision is only computed for full runs.
  if (!opts.diagnosticStopAfter && arms.includes("baseline") && arms.includes("A")) {
    const outDir = resolve(opts.outputDir);
    const decision = await runPython(
      [
        "-m",
        "opencode_gate_a.runner",
        "gate-a",
        "--baseline-manifest",
        join(outDir, "baseline-manifest.json"),
        "--baseline-eval",
        join(outDir, "baseline-evaluation.json"),
        "--treatment-manifest",
        join(outDir, "A-manifest.json"),
        "--treatment-eval",
        join(outDir, "A-evaluation.json"),
        "--out",
        // With the graph on, arm A is A+D and Gate A (which judges A alone) is not decided on it. The same comparison is
        // still computed, because the metric table is what the D slice is read from, but under a name that says so.
        join(outDir, decisionFileName(env)),
      ],
      VALIDATION_DIR,
      null,
    );
    if (parseGraphFlag(env.CONTEXT_GRAPH)) {
      console.error("[ce60] CONTEXT_GRAPH=on: the comparison below is the D slice (A+D vs baseline), NOT a Gate A decision");
    }
    console.log(decision.stdout.trim());
    if (decision.code !== 0) {
      console.error(decision.stderr.trim());
      return 4;
    }
  }
  console.log(JSON.stringify({ arms, evidence: evidences }, null, 2));
  return 0;
}

// Only run when invoked directly (not when imported by a test).
if (import.meta.main) {
  main(process.argv.slice(2))
    .then((code) => process.exit(code))
    .catch((error) => {
      console.error(String(error instanceof Error ? error.stack : error));
      process.exit(1);
    });
}
