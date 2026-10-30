#!/usr/bin/env bun
/**
 * Real coding session on OpenCode: baseline against the context-engineering plugin. NOT Gate A.
 *
 * Where `ce60.ts` replays 60 fixed banking prompts through MCP fixture tools and feeds Gate A, this driver runs a
 * scripted developer conversation (a few dozen prompts) in a throwaway COPY of the repository, so the agent really
 * reads files, greps, edits, writes and runs commands with OpenCode's own tools. It measures what the model was
 * billed for and writes payload-free facts; it decides nothing and produces no Gate A evidence.
 *
 * What it does, per arm:
 *   1. starts `opencode serve` on 127.0.0.1 (ephemeral port, HTTP basic auth through OpenCode's own
 *      OPENCODE_SERVER_PASSWORD, a random password per run) with the sandbox as working directory; `baseline` adds
 *      `--pure` so no plugin loads, `plugin` loads `.opencode/plugins/context-engineering.ts` from the sandbox;
 *   2. confirms with lsof that the only listener is loopback, and fails closed otherwise;
 *   3. creates ONE session and sends the prompts STRICTLY SEQUENTIALLY with `client.session.prompt`;
 *   4. reads the stored session back once and derives, per assistant message (one model call each), the tokens,
 *      the provider-priced `cost`, the tools used, denied calls and test-run summaries;
 *   5. ALWAYS tears the OpenCode process tree down.
 *
 * The OpenCode process gets a minimal allow-listed environment, not the caller's: the agent has a shell, so nothing the
 * session does not need (tokens, cookies, other cloud variables) is handed to it. AWS access is the profile and the
 * region only. The answer text of a prompt is kept ONLY for turns of kind `probe` (a short factual answer the report
 * scores); everything else is counts, ids and sizes.
 *
 *   bun .opencode/runner/code-session.ts --arm plugin --sandbox /tmp/oc-code/run/plugin \
 *     --model amazon-bedrock/anthropic.claude-sonnet-5-5 --run-id code-v1 --output-dir <dir> --turns <prompts.json>
 */

import { createOpencodeClient } from "@opencode-ai/sdk";
import { spawn, spawnSync, type ChildProcess } from "node:child_process";
import { createHash, randomBytes } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";

import {
  buildArmEnv,
  parseArgs as parseCe60Args,
  pluginOptions,
  reapProcessTree,
  verifyAbort,
  writeGraphDurability,
  type AbortClient,
  type Env,
} from "./ce60.ts";

export const ARMS = ["baseline", "plugin"] as const;
export type CodeArm = (typeof ARMS)[number];

/** Which practices the plugin arm runs: both, D alone (A's coverage emptied) or A alone (graph off). */
export const PLUGIN_MODES = ["a+d", "d-only", "a-only"] as const;
export type PluginMode = (typeof PLUGIN_MODES)[number];

/** Variables the OpenCode process may inherit. Everything else in the caller's environment stays behind. */
const ENV_ALLOWLIST = [
  "PATH",
  "HOME",
  "USER",
  "LOGNAME",
  "SHELL",
  "LANG",
  "LC_ALL",
  "LC_CTYPE",
  "TERM",
  "TMPDIR",
  "TZ",
  "AWS_PROFILE",
  "AWS_REGION",
  "AWS_DEFAULT_REGION",
] as const;

const DEFAULT_PROMPT_TIMEOUT_MS = 900_000;
const DEFAULT_DISABLED_TOOLS = ["question", "webfetch", "task"];
const GRAPH_TOOLS = new Set(["expand_card", "find_context"]);
const RETRIEVE_TOOL = "retrieve_context";
const SAFE_TOOL = /^[A-Za-z0-9_.:-]{1,64}$/;

/** One scripted prompt. `kind: "probe"` keeps its answer text so the report can score it. */
export interface CodeTurn {
  turn_id: string;
  kind: string;
  prompt: string;
}

export interface CodeOptions {
  arm: CodeArm;
  pluginMode: PluginMode;
  sandbox: string;
  model: string;
  runId: string;
  outDir: string;
  turnsPath: string;
  promptTimeoutMs: number;
  startupTimeoutMs: number;
  disableTools: string[];
  python: string;
  approvalId: string;
  budgetCeiling: string;
  budgetUnit: string;
  maxTurns: number | null;
  dryRun: boolean;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function num(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

// --- Arguments and turns -----------------------------------------------------------------------

/**
 * Parse the command line. Every flag takes a value except `--dry-run`; anything unknown is an error, so a typo can
 * never silently run the wrong arm.
 */
export function parseCodeArgs(argv: string[]): CodeOptions {
  const o: CodeOptions = {
    arm: "baseline",
    pluginMode: "a+d",
    sandbox: "",
    model: "",
    runId: "",
    outDir: "",
    turnsPath: "",
    promptTimeoutMs: DEFAULT_PROMPT_TIMEOUT_MS,
    startupTimeoutMs: 60_000,
    disableTools: [...DEFAULT_DISABLED_TOOLS],
    python: "python3.13",
    approvalId: "",
    budgetCeiling: "100",
    budgetUnit: "USD",
    maxTurns: null,
    dryRun: false,
  };
  const next = (i: number, flag: string): string => {
    const value = argv[i + 1];
    if (value === undefined || value.startsWith("--")) throw new Error(`${flag} needs a value`);
    return value;
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    switch (arg) {
      case "--arm": {
        const value = next(i++, arg);
        if (!(ARMS as readonly string[]).includes(value)) throw new Error(`--arm must be one of ${ARMS.join(", ")}`);
        o.arm = value as CodeArm;
        break;
      }
      case "--plugin-mode": {
        const value = next(i++, arg);
        if (!(PLUGIN_MODES as readonly string[]).includes(value)) {
          throw new Error(`--plugin-mode must be one of ${PLUGIN_MODES.join(", ")}`);
        }
        o.pluginMode = value as PluginMode;
        break;
      }
      case "--sandbox":
        o.sandbox = resolve(next(i++, arg));
        break;
      case "--model":
        o.model = next(i++, arg);
        break;
      case "--run-id":
        o.runId = next(i++, arg);
        break;
      case "--output-dir":
        o.outDir = resolve(next(i++, arg));
        break;
      case "--turns":
        o.turnsPath = resolve(next(i++, arg));
        break;
      case "--prompt-timeout-ms":
        o.promptTimeoutMs = Number(next(i++, arg));
        break;
      case "--startup-timeout-ms":
        o.startupTimeoutMs = Number(next(i++, arg));
        break;
      case "--disable-tools":
        o.disableTools = next(i++, arg)
          .split(",")
          .map((tool) => tool.trim())
          .filter((tool) => tool !== "");
        break;
      case "--python":
        o.python = next(i++, arg);
        break;
      case "--approval-id":
        o.approvalId = next(i++, arg);
        break;
      case "--budget-ceiling":
        o.budgetCeiling = next(i++, arg);
        break;
      case "--budget-unit":
        o.budgetUnit = next(i++, arg);
        break;
      case "--max-turns":
        o.maxTurns = Number(next(i++, arg));
        break;
      case "--dry-run":
        o.dryRun = true;
        break;
      default:
        throw new Error(`unknown argument: ${arg}`);
    }
  }
  const missing = (
    [
      ["--sandbox", o.sandbox],
      ["--model", o.model],
      ["--run-id", o.runId],
      ["--output-dir", o.outDir],
      ["--turns", o.turnsPath],
    ] as const
  )
    .filter(([, value]) => value === "")
    .map(([flag]) => flag);
  if (missing.length > 0) throw new Error(`missing required: ${missing.join(", ")}`);
  if (!/^[A-Za-z0-9._-]{1,80}$/.test(o.runId)) throw new Error("--run-id may use letters, digits, '.', '_' and '-'");
  if (!o.model.includes("/")) throw new Error("--model must be <provider>/<model>");
  if (!Number.isInteger(o.promptTimeoutMs) || o.promptTimeoutMs < 10_000) throw new Error("--prompt-timeout-ms is too small");
  if (o.maxTurns !== null && (!Number.isInteger(o.maxTurns) || o.maxTurns < 1)) throw new Error("--max-turns must be >= 1");
  for (const tool of o.disableTools) if (!SAFE_TOOL.test(tool)) throw new Error(`invalid tool name: ${tool}`);
  if (o.arm === "plugin" && o.approvalId === "") o.approvalId = `user-approved-${o.runId}`;
  return o;
}

/** Validate the prompt list: non-empty, unique ids, non-empty prompts. Accepts an array or `{turns: [...]}`. */
export function normalizeCodeTurns(data: unknown): CodeTurn[] {
  const list = Array.isArray(data) ? data : isRecord(data) && Array.isArray(data.turns) ? data.turns : null;
  if (list === null || list.length === 0) throw new Error("the turns file must hold a non-empty array or {turns: [...]}");
  const seen = new Set<string>();
  return list.map((item, index) => {
    if (!isRecord(item)) throw new Error(`turn ${index} is not an object`);
    const turnId = item.turn_id;
    const prompt = item.prompt;
    if (typeof turnId !== "string" || !SAFE_TOOL.test(turnId)) throw new Error(`turn ${index} has no valid turn_id`);
    if (typeof prompt !== "string" || prompt.trim() === "") throw new Error(`turn ${turnId} has an empty prompt`);
    if (seen.has(turnId)) throw new Error(`duplicate turn_id ${turnId}`);
    seen.add(turnId);
    return { turn_id: turnId, kind: typeof item.kind === "string" ? item.kind : "task", prompt };
  });
}

// --- Environment -------------------------------------------------------------------------------

/** The caller's environment reduced to {@link ENV_ALLOWLIST} (and the locale variables). */
export function minimalEnv(source: Env): Env {
  const out: Env = {};
  for (const key of ENV_ALLOWLIST) if (source[key] !== undefined) out[key] = source[key];
  for (const key of Object.keys(source)) if (key.startsWith("LC_") && source[key] !== undefined) out[key] = source[key];
  return out;
}

/**
 * The environment of one arm's OpenCode process.
 *
 * Both arms get the same shell-visible setup (PYTHONPATH on the sandbox's own context-core, no bytecode files), so the
 * agent's commands behave alike. Only the plugin arm gets the plugin's switches, the live admission that lets practice A
 * call the reranker, and the graph's evidence and metrics files. The `d-only` mode keeps the admission (the graph's
 * `find_context` needs it) but names a coverage that no tool can match, so practice A never filters anything.
 */
export function buildCodeEnv(cfg: CodeOptions, source: Env, files: { metrics: string; graphEvidence: string }): Env {
  const base = minimalEnv(source);
  base.PYTHON = cfg.python;
  base.PYTHONPATH = join(cfg.sandbox, "context-core", "src");
  base.PYTHONDONTWRITEBYTECODE = "1";
  const ce60 = parseCe60Args([
    "--execute-live",
    "--model",
    cfg.model,
    "--approval-id",
    cfg.approvalId || "baseline-not-used",
    "--budget-ceiling",
    cfg.budgetCeiling,
    "--budget-unit",
    cfg.budgetUnit,
  ]);
  if (cfg.arm === "baseline") return buildArmEnv("baseline", base, ce60);
  base.CONTEXT_BRIDGE_COMMAND = cfg.python;
  base.CONTEXT_PREVIEW_STYLE = "scoped";
  base.CONTEXT_RETRIEVAL_CLEANUP = "on";
  const graph = cfg.pluginMode !== "a-only";
  if (graph) {
    base.CONTEXT_GRAPH = "on";
    base.CONTEXT_GRAPH_MODE = "inplace";
    base.CONTEXT_GRAPH_KEEP_RECENT = "0";
    base.CONTEXT_GRAPH_KEEP_DIALOGUE = "0";
  }
  if (cfg.pluginMode === "d-only") base.CONTEXT_OPENCODE_TOOLS = "__none__";
  return buildArmEnv("A", base, ce60, files.metrics, graph ? files.graphEvidence : undefined);
}

// --- Process control ---------------------------------------------------------------------------

interface StartedServer {
  url: string;
  proc: ChildProcess;
}

/** Spawn `opencode serve` on loopback in `cwd` and resolve once it prints its URL (same contract as ce60's). */
function startServer(cwd: string, pure: boolean, env: Env, timeoutMs: number): Promise<StartedServer> {
  const args = ["serve", "--hostname=127.0.0.1", "--port=0"];
  if (pure) args.push("--pure");
  const proc = spawn("opencode", args, { cwd, env, stdio: ["ignore", "pipe", "pipe"] });
  return new Promise<StartedServer>((resolvePromise, reject) => {
    let output = "";
    let settled = false;
    const finish = (action: () => void): void => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      action();
    };
    const timer = setTimeout(
      () =>
        finish(() => {
          proc.kill("SIGTERM");
          reject(new Error(`timeout waiting for opencode to start after ${timeoutMs}ms`));
        }),
      timeoutMs,
    );
    proc.stdout?.on("data", (chunk: Buffer) => {
      if (settled) return;
      output += chunk.toString();
      const match = output.match(/opencode server listening on\s+(https?:\/\/[^\s]+)/);
      if (match) finish(() => resolvePromise({ url: match[1], proc }));
    });
    proc.stderr?.on("data", (chunk: Buffer) => (output += chunk.toString()));
    proc.on("error", (error) => finish(() => reject(error)));
    proc.on("exit", (code) => finish(() => reject(new Error(`opencode exited early with code ${code}: ${output.slice(-400)}`))));
  });
}

/** Poll `/global/health` with the basic-auth header until it answers 200. */
async function waitForAuthedHealth(url: string, headers: Record<string, string>, timeoutMs = 30_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    try {
      const response = await fetch(`${url}/global/health`, { headers });
      if (response.ok) return;
    } catch {
      // not up yet
    }
    if (Date.now() > deadline) throw new Error(`opencode /global/health not ready within ${timeoutMs}ms`);
    await new Promise((r) => setTimeout(r, 250));
  }
}

/** Every address the process listens on (`127.0.0.1:54321`), from lsof; `null` when lsof gave no answer. */
export function listenerAddresses(pid: number): string[] | null {
  const result = spawnSync("lsof", ["-nP", "-a", "-p", String(pid), "-iTCP", "-sTCP:LISTEN", "-Fn"], { encoding: "utf8" });
  if (result.error || typeof result.stdout !== "string") return null;
  return result.stdout
    .split("\n")
    .filter((line) => line.startsWith("n"))
    .map((line) => line.slice(1));
}

/** Whether every listener address is loopback. */
export function allLoopback(addresses: string[]): boolean {
  return addresses.length > 0 && addresses.every((a) => a.startsWith("127.0.0.1:") || a.startsWith("[::1]:"));
}

const TIMEOUT = Symbol("timeout");

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

function hostVersion(): string {
  const result = spawnSync("opencode", ["--version"], { encoding: "utf8" });
  return result.status === 0 ? result.stdout.trim().split("\n").pop() ?? "unknown" : "unknown";
}

// --- Reading the stored session ----------------------------------------------------------------

/** One model call: the provider's own usage and price for it, and what it did. Counts only, no text. */
export interface CallRecord {
  input: number;
  cache_read: number;
  cache_write: number;
  output: number;
  reasoning: number;
  model_input: number;
  cost: number;
  ms: number | null;
  finish: string | null;
  error: string | null;
  tools: string[];
}

export interface TestRun {
  passed: number;
  failed: number;
  errors: number;
}

export interface TurnUsage {
  calls: CallRecord[];
  tool_counts: Record<string, number>;
  tool_errors: Record<string, number>;
  denied: number;
  files_written: number;
  test_runs: TestRun[];
  answer_text?: string;
}

function textOf(parts: unknown[]): string {
  const texts: string[] = [];
  for (const part of parts) if (isRecord(part) && part.type === "text" && typeof part.text === "string") texts.push(part.text);
  return texts.join("\n");
}

/** Read a pytest or unittest summary out of a command's output; `null` when the output is not one. */
export function parseTestRun(command: string, output: string): TestRun | null {
  if (!/\b(pytest|unittest)\b/.test(command)) return null;
  const count = (re: RegExp): number => {
    const m = output.match(re);
    return m ? Number(m[1]) : 0;
  };
  const ran = output.match(/Ran (\d+) tests?/);
  if (ran) {
    const failed = output.match(/failures=(\d+)/);
    const errors = output.match(/errors=(\d+)/);
    const bad = (failed ? Number(failed[1]) : 0) + (errors ? Number(errors[1]) : 0);
    return { passed: Math.max(0, Number(ran[1]) - bad), failed: failed ? Number(failed[1]) : 0, errors: errors ? Number(errors[1]) : 0 };
  }
  const passed = count(/(\d+) passed/);
  const failed = count(/(\d+) failed/);
  const errors = count(/(\d+) errors?\b/);
  return passed + failed + errors > 0 ? { passed, failed, errors } : null;
}

/**
 * Group the stored session into per-turn usage.
 *
 * `messages` is the `session.messages` payload (`[{info, parts}]`). A user message opens a turn; an assistant message
 * belongs to the turn of its `parentID` (or, without one, the latest user message). Only counts leave this function,
 * plus the answer text of the turns named in `keepAnswerFor`.
 */
export function summarizeMessages(messages: unknown, keepAnswerFor: ReadonlySet<number> = new Set()): TurnUsage[] {
  const data = isRecord(messages) && "data" in messages ? messages.data : messages;
  const turns: TurnUsage[] = [];
  if (!Array.isArray(data)) return turns;
  const turnOfUser = new Map<string, number>();
  let current = -1;
  for (const item of data) {
    if (!isRecord(item) || !isRecord(item.info)) continue;
    const info = item.info;
    const parts = Array.isArray(item.parts) ? item.parts : [];
    if (info.role === "user") {
      current++;
      if (typeof info.id === "string") turnOfUser.set(info.id, current);
      turns[current] = { calls: [], tool_counts: {}, tool_errors: {}, denied: 0, files_written: 0, test_runs: [] };
      continue;
    }
    if (info.role !== "assistant") continue;
    const parent = typeof info.parentID === "string" ? turnOfUser.get(info.parentID) : undefined;
    const index = parent ?? current;
    const usage = turns[index];
    if (!usage) continue;
    const tokens = isRecord(info.tokens) ? info.tokens : {};
    const cache = isRecord(tokens.cache) ? tokens.cache : {};
    const time = isRecord(info.time) ? info.time : {};
    const error = isRecord(info.error) ? info.error : null;
    const tools: string[] = [];
    for (const part of parts) {
      if (!isRecord(part) || part.type !== "tool" || typeof part.tool !== "string" || !SAFE_TOOL.test(part.tool)) continue;
      tools.push(part.tool);
      usage.tool_counts[part.tool] = (usage.tool_counts[part.tool] ?? 0) + 1;
      if (part.tool === "write" || part.tool === "edit" || part.tool === "patch") usage.files_written++;
      const state = isRecord(part.state) ? part.state : {};
      if (state.status === "error") {
        usage.tool_errors[part.tool] = (usage.tool_errors[part.tool] ?? 0) + 1;
        if (typeof state.error === "string" && /denied|not allowed|permission|rejected/i.test(state.error)) usage.denied++;
      }
      if (part.tool === "bash" && isRecord(state.input) && typeof state.input.command === "string") {
        const run = parseTestRun(state.input.command, typeof state.output === "string" ? state.output : "");
        if (run) usage.test_runs.push(run);
      }
    }
    const input = num(tokens.input);
    const cacheRead = num(cache.read);
    const cacheWrite = num(cache.write);
    usage.calls.push({
      input,
      cache_read: cacheRead,
      cache_write: cacheWrite,
      output: num(tokens.output),
      reasoning: num(tokens.reasoning),
      model_input: input + cacheRead + cacheWrite,
      cost: num(info.cost),
      ms: typeof time.created === "number" && typeof time.completed === "number" ? time.completed - time.created : null,
      finish: typeof info.finish === "string" ? info.finish : null,
      error: error ? String(error.name ?? "error").slice(0, 64) : null,
      tools,
    });
    if (keepAnswerFor.has(index)) {
      const text = textOf(parts);
      if (text.trim() !== "") usage.answer_text = text;
    }
  }
  return turns;
}

export function totalsOf(turns: TurnUsage[]): Record<string, number> {
  const totals = { model_calls: 0, input: 0, cache_read: 0, cache_write: 0, output: 0, reasoning: 0, model_input: 0, cost: 0 };
  for (const turn of turns) {
    for (const call of turn.calls) {
      totals.model_calls++;
      totals.input += call.input;
      totals.cache_read += call.cache_read;
      totals.cache_write += call.cache_write;
      totals.output += call.output;
      totals.reasoning += call.reasoning;
      totals.model_input += call.model_input;
      totals.cost += call.cost;
    }
  }
  return totals;
}

// --- The run -----------------------------------------------------------------------------------

interface TurnRecord {
  turn_id: string;
  kind: string;
  host_signal: "success" | "error" | "timeout";
  error_code: string | null;
  wall_ms: number;
  abort?: { abort_status: string; idle_status: string };
}

async function runSession(cfg: CodeOptions, turns: CodeTurn[], source: Env): Promise<number> {
  mkdirSync(cfg.outDir, { recursive: true });
  const files = {
    metrics: join(cfg.outDir, "A-metrics.jsonl"),
    graphEvidence: join(cfg.outDir, "A-graph-evidence.jsonl"),
  };
  if (cfg.arm === "plugin") {
    writeFileSync(files.metrics, "");
    if (cfg.pluginMode !== "a-only") writeFileSync(files.graphEvidence, "");
  }
  const env = buildCodeEnv(cfg, source, files);
  const password = randomBytes(24).toString("hex");
  env.OPENCODE_SERVER_PASSWORD = password;
  const headers = { Authorization: "Basic " + Buffer.from(`opencode:${password}`).toString("base64") };

  const started = await startServer(cfg.sandbox, cfg.arm === "baseline", env, cfg.startupTimeoutMs);
  const startedAt = new Date().toISOString();
  const records: TurnRecord[] = [];
  const turnsPlanned = cfg.maxTurns === null ? turns.length : Math.min(cfg.maxTurns, turns.length);
  let usage: TurnUsage[] = [];
  let listener: string[] | null = null;
  let graphDurability: Record<string, unknown> | null = null;
  try {
    listener = listenerAddresses(started.proc.pid ?? -1);
    if (listener === null || !allLoopback(listener)) {
      throw new Error(`refusing to continue: the server is not loopback-only (${JSON.stringify(listener)})`);
    }
    await waitForAuthedHealth(started.url, headers);
    const client = createOpencodeClient({ baseUrl: started.url, headers });
    const created = await client.session.create({ body: { title: `code-${cfg.runId}-${cfg.arm}` } });
    const sessionId = (created.data as { id?: string } | undefined)?.id;
    if (!sessionId) throw new Error("could not create an OpenCode session");
    const [providerID, ...rest] = cfg.model.split("/");
    const modelID = rest.join("/");
    const toolsMap = Object.fromEntries(cfg.disableTools.map((tool) => [tool, false]));
    const toRun = cfg.maxTurns === null ? turns : turns.slice(0, cfg.maxTurns);
    const keepAnswerFor = new Set<number>();
    toRun.forEach((turn, index) => {
      if (turn.kind === "probe") keepAnswerFor.add(index);
    });

    for (const [index, turn] of toRun.entries()) {
      const t0 = Date.now();
      let record: TurnRecord = { turn_id: turn.turn_id, kind: turn.kind, host_signal: "success", error_code: null, wall_ms: 0 };
      let result: unknown;
      try {
        result = await withTimeout(
          client.session.prompt({
            path: { id: sessionId },
            body: { model: { providerID, modelID }, tools: toolsMap, parts: [{ type: "text", text: turn.prompt }] },
          }),
          cfg.promptTimeoutMs,
        );
      } catch (error) {
        record = { ...record, host_signal: "error", error_code: "PROMPT_ERROR" };
        console.error(`[code] ${cfg.arm} ${turn.turn_id} prompt call threw: ${String(error).slice(0, 160)}`);
        result = null;
      }
      if (result === TIMEOUT) {
        const abort = await verifyAbort(client as unknown as AbortClient, sessionId);
        record = { ...record, host_signal: "timeout", error_code: "PROMPT_TIMEOUT", abort };
        console.error(`[code] ${cfg.arm} ${turn.turn_id} TIMEOUT abort=${abort.abort_status} idle=${abort.idle_status}`);
        if (abort.idle_status === "not_idle") {
          record.wall_ms = Date.now() - t0;
          records.push(record);
          console.error("[code] session did not return to idle; stopping (the remaining turns are not sent)");
          break;
        }
      } else if (result !== null) {
        const body: { data?: unknown; error?: unknown } = isRecord(result) ? result : {};
        const info = isRecord(body.data) && isRecord(body.data.info) ? body.data.info : null;
        if (!info || isRecord(body.error)) {
          record = { ...record, host_signal: "error", error_code: "PROMPT_FAILED" };
        } else if (isRecord(info.error)) {
          record = { ...record, host_signal: "error", error_code: String(info.error.name ?? "HOST_ERROR").slice(0, 64) };
        }
      }
      record.wall_ms = Date.now() - t0;
      records.push(record);
      const live = totalsOf(summarizeMessages(await client.session.messages({ path: { id: sessionId } })));
      console.error(
        `[code] ${cfg.arm} ${String(index + 1).padStart(2)}/${toRun.length} ${turn.turn_id} ${record.host_signal}` +
          ` ${(record.wall_ms / 1000).toFixed(0)}s calls=${live.model_calls} input=${live.model_input} cost=$${live.cost.toFixed(3)}`,
      );
      // Fail closed: a turn that did not succeed leaves the session in a state the other arm never saw (a half-written
      // file, an aborted call), so the arm stops here and the caller reruns it from a fresh sandbox.
      if (record.host_signal !== "success") {
        console.error(`[code] ${cfg.arm} stopping at ${turn.turn_id}: ${record.error_code ?? record.host_signal}`);
        break;
      }
    }

    usage = summarizeMessages(await client.session.messages({ path: { id: sessionId } }), keepAnswerFor);
    if (cfg.arm === "plugin" && cfg.pluginMode !== "a-only") {
      try {
        graphDurability = await writeGraphDurability(client, sessionId, files.graphEvidence, cfg.outDir, "A", cfg.runId);
      } catch (error) {
        console.error(`[code] graph durability check failed: ${String(error).slice(0, 200)}`);
      }
    }
  } finally {
    const cleanup = await reapProcessTree(started.proc);
    writeFileSync(
      join(cfg.outDir, `${cfg.arm}-cleanup.json`),
      JSON.stringify({ arm: cfg.arm, run_id: cfg.runId, ...cleanup }, null, 2) + "\n",
    );
    if (cleanup.residual_process_count > 0) console.error(`[code] WARNING ${cleanup.residual_process_count} residual process(es)`);
  }

  const turnRows = records.map((record, index) => ({ ...record, ...(usage[index] ?? { calls: [] }) }));
  const out = {
    schema: "code-session/v1",
    arm: cfg.arm,
    plugin_mode: cfg.arm === "plugin" ? cfg.pluginMode : null,
    run_id: cfg.runId,
    model: cfg.model,
    host: "opencode",
    host_version: hostVersion(),
    plugin_options: cfg.arm === "plugin" ? pluginOptions(env) : null,
    turns_sha256: createHash("sha256").update(readFileSync(cfg.turnsPath)).digest("hex"),
    prompt_timeout_ms: cfg.promptTimeoutMs,
    disabled_tools: cfg.disableTools,
    listener,
    started_at: startedAt,
    finished_at: new Date().toISOString(),
    turns_planned: turnsPlanned,
    complete: records.length === turnsPlanned && records.every((r) => r.host_signal === "success"),
    turns: turnRows,
    totals: totalsOf(usage),
    graph_durability: graphDurability,
  };
  writeFileSync(join(cfg.outDir, `${cfg.arm}-code-session.json`), JSON.stringify(out, null, 2) + "\n");
  const failed = records.filter((r) => r.host_signal !== "success").length;
  console.error(`[code] ${cfg.arm} done: ${records.length}/${turnsPlanned} turns, ${failed} not successful`);
  return failed === 0 && records.length === turnsPlanned ? 0 : 3;
}

export async function main(argv: string[], source: Env = process.env): Promise<number> {
  let cfg: CodeOptions;
  try {
    cfg = parseCodeArgs(argv);
  } catch (error) {
    console.error(String(error instanceof Error ? error.message : error));
    return 2;
  }
  let turns: CodeTurn[];
  try {
    turns = normalizeCodeTurns(JSON.parse(readFileSync(cfg.turnsPath, "utf8")));
  } catch (error) {
    console.error(`cannot read the turns: ${String(error instanceof Error ? error.message : error)}`);
    return 2;
  }
  if (cfg.dryRun) {
    const env = buildCodeEnv(cfg, source, { metrics: join(cfg.outDir, "A-metrics.jsonl"), graphEvidence: join(cfg.outDir, "A-graph-evidence.jsonl") });
    console.log(
      JSON.stringify(
        {
          mode: "dry-run",
          arm: cfg.arm,
          plugin_mode: cfg.arm === "plugin" ? cfg.pluginMode : null,
          sandbox: cfg.sandbox,
          model: cfg.model,
          turns: turns.map((t) => `${t.turn_id}:${t.kind}`),
          disabled_tools: cfg.disableTools,
          env_keys: Object.keys(env).sort(),
          aws_profile_present: Boolean(env.AWS_PROFILE),
          aws_region_present: Boolean(env.AWS_REGION || env.AWS_DEFAULT_REGION),
        },
        null,
        2,
      ),
    );
    return 0;
  }
  if (!source.AWS_PROFILE || !(source.AWS_REGION || source.AWS_DEFAULT_REGION)) {
    console.error("AWS_PROFILE and AWS_REGION are required (no default profile is assumed)");
    return 2;
  }
  return runSession(cfg, turns, source);
}

if (import.meta.main) {
  main(process.argv.slice(2))
    .then((code) => process.exit(code))
    .catch((error) => {
      console.error(String(error instanceof Error ? (error.stack ?? error.message) : error));
      process.exit(1);
    });
}
