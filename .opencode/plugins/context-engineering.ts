/**
 * Context-engineering plugin for OpenCode (verified against the public plugin API of 1.18.34).
 *
 * This file is a thin, loadable interface. All decision logic — admission, capability, relevance
 * filtering, baseline preservation and the closed refusal envelope — lives in the Python bridge
 * (`context_core.opencode_bridge`) and the neutral `context-core`. The plugin only:
 *
 *   1. spawns the bridge as a persistent child process that speaks newline-delimited JSON (JSONL);
 *   2. reacts to the public `tool.execute.after` hook, defensively reads the tool name, the session id
 *      and the result text, asks the bridge to relevance-filter it, and substitutes the result **only**
 *      when the bridge answers `previewed` or `baseline`;
 *   3. reacts to the session lifecycle through the public `event` hook (`session.created`,
 *      `session.compacted`, `session.deleted`);
 *   4. registers the public `retrieve_context` tool (via the official `tool` helper from
 *      `@opencode-ai/plugin`) so the agent can read back what a preview replaced — all of it, or only the
 *      rows it needs (`pattern`, `line_range`, `max_chunks`). The bridge authorizes the owning session and
 *      returns the preserved baseline slice or a closed refusal;
 *   5. keeps a retrieval for the answer in flight: through the experimental
 *      `experimental.chat.messages.transform` hook it marks the completed `retrieve_context` results of
 *      earlier turns as cleared (OpenCode then sends `[Old tool result content cleared]`), in the messages sent
 *      to the model only, as the reference Strands filter removes them from its history.
 *      `CONTEXT_RETRIEVAL_CLEANUP=0` turns this off and `CONTEXT_PREVIEW_STYLE=legacy` restores the old
 *      one-line hint, so the change can be measured against the old behaviour;
 *   6. keeps the retention lease of every session it has seen alive with a periodic `heartbeat`, because a
 *      reference that is read back turns after it was stored must outlive the lease's heartbeat window;
 *   7. OPT-IN, off unless `CONTEXT_GRAPH=on`: practice D (context graph). In the same experimental hook it asks the
 *      bridge which earlier messages to leave out of the request and what summary to append as one trailing synthetic
 *      user message, and it registers the `expand_card` and `find_context` tools. Request-only: the stored session is
 *      never written. With the flag off nothing of this is registered or run (see `lib/graph.ts`).
 *
 * It uses only public API: the `tool.execute.after` and `event` hooks, the `tool` helper, the SDK client
 * (`client.session.messages`, `client.app.log`) and Bun's child-process API. No private OpenCode API is
 * touched. A refusal or a partial payload is never injected into the result; on anything other than
 * `previewed`/`baseline` the original result is left exactly as the tool produced it.
 *
 * Configuration is entirely environment-driven and read by the bridge, not here. See `.opencode/README.md`.
 */

import { tool, type Hooks, type PluginInput } from "@opencode-ai/plugin";
import { BridgeClient } from "../lib/bridge-transport.ts";
import { composeFilteredResult, parsePreviewStyle, parseRetrievalCleanup } from "../lib/excerpt.ts";
import {
  EXPAND_CARD_DESCRIPTION,
  EXPAND_CARD_RESULT_DESCRIPTION,
  EXPAND_CARD_TOOL,
  EvidenceLog,
  FIND_CONTEXT_DESCRIPTION,
  FIND_CONTEXT_TOOL,
  GRAPH_REVISION,
  GraphHook,
  parseGraphFlag,
  parseGraphMode,
} from "../lib/graph.ts";
import { clearClosedRetrievalResults, type HistoryMessage } from "../lib/history.ts";
import { PLUGIN_REVISION } from "../lib/revision.ts";
import {
  classifyOutputShape,
  describeUnreadableToolOutput,
  readOutputText,
  writeOutputText,
} from "../lib/tool-output.ts";

const SERVICE = "context-engineering";

// The public retrieval tool this plugin registers. Its calls are what the CE-60 host driver counts as
// retrieval attempts/successes; it only exists when the plugin is loaded (arm A, never baseline --pure).
const RETRIEVE_CONTEXT_TOOL = "retrieve_context";

// Fidelity with the reference filter, both on by default and both switchable so the change is measurable.
const PREVIEW_STYLE = parsePreviewStyle(process.env.CONTEXT_PREVIEW_STYLE);
const RETRIEVAL_CLEANUP = parseRetrievalCleanup(process.env.CONTEXT_RETRIEVAL_CLEANUP);

// Practice D is opt-in and independent of the two switches above. Off, the plugin registers the same hooks and the same
// tool as before and the graph code never runs.
const GRAPH_ENABLED = parseGraphFlag(process.env.CONTEXT_GRAPH);
// Which policy the bridge runs. The plugin needs it for one thing: what it tells the model `expand_card` does.
const GRAPH_MODE = parseGraphMode(process.env.CONTEXT_GRAPH_MODE);
const GRAPH_EVIDENCE_FILE = (process.env.CONTEXT_GRAPH_EVIDENCE_FILE || "").trim() || null;
// Scoring the conversation against an embedding model takes longer than a relevance-filter call.
const GRAPH_TIMEOUT_MS = Number(process.env.CONTEXT_GRAPH_TIMEOUT_MS || "30000");

type JsonObject = Record<string, unknown>;
type ToolExecuteAfterHook = NonNullable<Hooks["tool.execute.after"]>;
type ToolExecuteAfterInput = Parameters<ToolExecuteAfterHook>[0];
type ToolExecuteAfterOutput = Parameters<ToolExecuteAfterHook>[1];

/** Mint a single-use nonce for one admission request; `crypto.randomUUID` when available. */
function freshNonce(): string {
  const uuid = (globalThis as { crypto?: { randomUUID?: () => string } }).crypto?.randomUUID?.();
  return uuid ?? `nonce-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

// Execution admission is opt-in and never defaults to `live`. The plugin only attempts a `live`
// admission when the operator set CONTEXT_EXECUTION_MODE=live AND supplied an approval id, a budget
// ceiling and a budget unit. Anything short of that leaves the plugin with no admission receipt, so a
// covered, oversized result is refused by the bridge rather than billing a Bedrock call. The actor
// class defaults to `maintainer`; the model allow-list and AWS profile live on the bridge side only.
const EXECUTION_MODE = (process.env.CONTEXT_EXECUTION_MODE || "").trim().toLowerCase();
const APPROVAL_ID = (process.env.CONTEXT_APPROVAL_ID || "").trim();
const APPROVAL_ACTOR = (process.env.CONTEXT_APPROVAL_ACTOR || "maintainer").trim();
const BUDGET_CEILING = (process.env.CONTEXT_BUDGET_CEILING || "").trim();
const BUDGET_UNIT = (process.env.CONTEXT_BUDGET_UNIT || "").trim();
const LIVE_ADMISSION_REQUESTED =
  EXECUTION_MODE === "live" && APPROVAL_ID !== "" && BUDGET_CEILING !== "" && BUDGET_UNIT !== "";

/**
 * Pull the newest user message's text from the session, to use as the relevance query.
 *
 * Uses the public SDK `client.session.messages({ path: { id } })`. Any failure or unexpected shape
 * yields `""`, which the bridge treats as "no query" and preserves the baseline rather than scoring.
 */
async function latestUserText(client: PluginInput["client"], sessionID: string): Promise<string> {
  try {
    const result = await client.session.messages({ path: { id: sessionID } });
    const items: unknown = result?.data ?? result;
    if (!Array.isArray(items)) return "";
    for (let i = items.length - 1; i >= 0; i--) {
      const entry = items[i] as JsonObject;
      const info = (entry?.info ?? entry) as JsonObject;
      if (info?.role !== "user") continue;
      const parts = (entry?.parts ?? info?.parts) as unknown;
      const texts: string[] = [];
      if (Array.isArray(parts)) {
        for (const part of parts) {
          const p = part as JsonObject;
          if (p?.type === "text" && typeof p.text === "string") texts.push(p.text);
        }
      }
      if (texts.length > 0) return texts.join("\n");
    }
    return "";
  } catch {
    return "";
  }
}

/** The bridge's retention lease needs a sign of life within 300 s; renew well inside that window. */
const HEARTBEAT_INTERVAL_MS = 120_000;

/** Options the model may pass to narrow a retrieval; forwarded to the bridge untouched, validated there. */
const READ_OPTIONS = ["pattern", "line_range", "context_lines", "max_chunks", "max_tokens"] as const;

interface RetrieveArgs {
  reference_id?: unknown;
  pattern?: unknown;
  line_range?: unknown;
  context_lines?: unknown;
  max_chunks?: unknown;
  max_tokens?: unknown;
}

/** The tool description; it is the instruction the model reads, so it names when NOT to use the tool. */
function retrievalToolDescription(cleanup: boolean): string {
  return (
    "Read back a tool result that was cut down to an excerpt. Use it ONLY when the excerpt cannot answer the " +
    "question, typically because the answer needs every row: a maximum, minimum, total, count, average, ranking " +
    "or a comparison across the whole result. Do NOT use it to re-read a passage the excerpt already shows. " +
    "Pass the reference_id from the excerpt notice, and prefer the narrowest option: `pattern` returns only the " +
    "matching lines with context (the cheapest way to aggregate over specific rows), `line_range` returns an " +
    "exact span using the line numbers of the \"[... N lines omitted ...]\" markers, `max_chunks` returns the " +
    "most relevant chunks. With no option it returns the whole original." +
    (cleanup
      ? " What this tool returns is cleared from the conversation after the turn that uses it (it then shows as " +
        "[Old tool result content cleared]), so state the figures you relied on in your answer, and do not " +
        "retrieve again just to re-check an earlier answer."
      : "")
  );
}

const ContextEngineeringPlugin = async ({ client }: PluginInput) => {
  const log = (level: string, message: string, extra?: JsonObject) => {
    // Structured logging only: IDs, operations and sizes — never the result text or the query payload.
    try {
      void client.app.log({ body: { service: SERVICE, level, message, extra: extra ?? {} } });
    } catch {
      /* logging must never throw into the agent loop */
    }
  };

  const bridge = new BridgeClient(log);

  // Sessions this plugin has seen. While one is open its retention lease is renewed on a schedule, so a
  // reference stored in a preview still resolves when the model reads it back many turns (and minutes) later.
  const liveSessions = new Set<string>();
  const heartbeat = setInterval(() => {
    for (const id of liveSessions) void bridge.request({ type: "heartbeat", session_id: id });
  }, HEARTBEAT_INTERVAL_MS);
  (heartbeat as { unref?: () => void }).unref?.();
  log("info", "plugin loaded", {
    revision: PLUGIN_REVISION,
    preview_style: PREVIEW_STYLE,
    retrieval_cleanup: RETRIEVAL_CLEANUP,
    ...(GRAPH_ENABLED ? { graph: true, graph_revision: GRAPH_REVISION, graph_mode: GRAPH_MODE } : {}),
  });
  // Announce the bridge once; the response is advisory and never blocks tool execution.
  void bridge.request({ type: "startup" }).then((response) => {
    if (response) log("info", "bridge started", { operation: String(response.operation ?? "") });
  });

  // The admitted live receipt, or null. A null receipt means a covered, oversized result is refused by
  // the bridge (no Bedrock call), which is the intended default when live was not explicitly requested.
  let admissionId: string | null = null;

  // The graph half of the transform hook. Built only when the flag is on, so a graph-off plugin has nothing of it.
  const graphHook = GRAPH_ENABLED
    ? new GraphHook({
        request: (payload) => bridge.request(payload, { timeoutMs: GRAPH_TIMEOUT_MS }),
        evidence: new EvidenceLog(GRAPH_EVIDENCE_FILE),
        admissionId: () => admissionId,
        log,
      })
    : null;

  if (LIVE_ADMISSION_REQUESTED) {
    // Acquire the receipt before any tool runs, so the first oversized result can be filtered. The
    // request carries no credential; the bridge resolves AWS only after it has admitted the request and
    // only when a transform then presents this receipt.
    const admission = await bridge.request({
      type: "admit",
      execution_mode: "live",
      approval_id: APPROVAL_ID,
      actor_class: APPROVAL_ACTOR,
      budget_ceiling: BUDGET_CEILING,
      budget_unit: BUDGET_UNIT,
      nonce: freshNonce(),
      preflight_requested: true,
    });
    if (admission && admission.operation === "admitted" && typeof admission.admission_id === "string") {
      admissionId = admission.admission_id;
      // Log only the receipt identity and policy digest — never the approval id, budget or nonce.
      log("info", "live admission admitted", {
        admission_id: admissionId,
        policy_digest: String(admission.policy_digest ?? ""),
      });
    } else {
      log("warn", "live admission not granted; results will not be filtered", {
        operation: String(admission?.operation ?? "none"),
      });
    }
  } else {
    log("debug", "live admission not requested; results will not be filtered");
  }

  // The two retrieval tools of practice D. Registered only when the graph is on, so a graph-off plugin offers the model
  // exactly the tool set it offered before. Every failure is a sentence the model can act on, never an exception: a
  // thrown error would report the tool broken rather than the request unanswerable.
  const graphTools = (): Record<string, ReturnType<typeof tool>> => {
    if (!GRAPH_ENABLED) return {};
    const unavailable = (name: string): string =>
      `${name} | the context graph is unavailable right now | answer from what the summary and the messages already ` +
      "give you, and state plainly which part you could not verify";
    const ask = async (type: string, name: string, sessionID: string, extra: JsonObject): Promise<string> => {
      const request: JsonObject = { type, session_id: sessionID, ...extra };
      if (admissionId !== null) request.admission_id = admissionId;
      const response = await bridge.request(request, { timeoutMs: GRAPH_TIMEOUT_MS });
      if (response && response.operation === "graph_result" && typeof response.text === "string") return response.text;
      log("info", "graph tool unavailable", { tool: name, operation: String(response?.operation ?? "none") });
      return unavailable(name);
    };
    return {
      [EXPAND_CARD_TOOL]: tool({
        description: GRAPH_MODE === "inplace" ? EXPAND_CARD_RESULT_DESCRIPTION : EXPAND_CARD_DESCRIPTION,
        args: {
          titles: tool.schema
            .array(tool.schema.string())
            .describe("The titles of the earlier turns you need, copied exactly as they were shown to you."),
        },
        execute: async (args: { titles?: unknown }, context: { sessionID?: string }) => {
          const sessionID = typeof context?.sessionID === "string" ? context.sessionID : "";
          if (!sessionID) return unavailable(EXPAND_CARD_TOOL);
          const titles = Array.isArray(args?.titles)
            ? args.titles.filter((title): title is string => typeof title === "string")
            : [];
          return ask("graph_expand_card", EXPAND_CARD_TOOL, sessionID, { titles });
        },
      }),
      [FIND_CONTEXT_TOOL]: tool({
        description: FIND_CONTEXT_DESCRIPTION,
        args: {
          need: tool.schema.string().describe("What you are looking for, in your own words."),
          tag: tool.schema.string().optional().describe("Only consider earlier turns carrying this tag."),
        },
        execute: async (args: { need?: unknown; tag?: unknown }, context: { sessionID?: string }) => {
          const sessionID = typeof context?.sessionID === "string" ? context.sessionID : "";
          if (!sessionID) return unavailable(FIND_CONTEXT_TOOL);
          const extra: JsonObject = { need: typeof args?.need === "string" ? args.need : "" };
          if (typeof args?.tag === "string" && args.tag.trim() !== "") extra.tag = args.tag;
          return ask("graph_find_context", FIND_CONTEXT_TOOL, sessionID, extra);
        },
      }),
    };
  };

  return {
    event: async ({ event }: { event: { type: string; properties?: JsonObject } }) => {
      const type = event?.type;
      const properties = (event?.properties ?? {}) as JsonObject;
      const info = (properties.info ?? properties) as JsonObject;
      const sessionID =
        (typeof info.id === "string" && info.id) ||
        (typeof properties.sessionID === "string" && properties.sessionID) ||
        null;

      if (type === "session.created" && sessionID) {
        log("debug", "session created");
      } else if (type === "session.compacted" && sessionID) {
        log("debug", "session compacted");
      } else if (type === "session.deleted" && sessionID) {
        // Discard the session namespace in the bridge; fire-and-forget, result left unchanged.
        liveSessions.delete(sessionID);
        void bridge.request({ type: "shutdown", session_id: sessionID });
        log("debug", "session deleted");
      }
    },

    "tool.execute.after": async (
      input: ToolExecuteAfterInput,
      output: ToolExecuteAfterOutput,
    ) => {
      // Defensive extraction: a missing field just means "nothing to filter", never a thrown error.
      const toolName = typeof input?.tool === "string" ? input.tool : null;
      const sessionID = typeof input?.sessionID === "string" ? input.sessionID : null;
      if (sessionID) liveSessions.add(sessionID);
      const text = readOutputText(output);
      if (text === null) {
        // Keep this diagnosis payload-free: keys and value types explain the host shape without logging
        // args, content, or any serialized result object.
        log("debug", "tool output text unavailable", describeUnreadableToolOutput(input, output));
        return;
      }
      // A direct string is accepted by readOutputText defensively, but only the public object output
      // can be safely mutated with a filtered result.
      if (
        !toolName ||
        !sessionID ||
        !output ||
        typeof output !== "object" ||
        Array.isArray(output)
      ) {
        return;
      }

      const query = await latestUserText(client, sessionID);
      const request: JsonObject = {
        type: "transform",
        tool: toolName,
        session_id: sessionID,
        query,
        text,
      };
      // Attach the live admission receipt when one was granted. Without it the bridge refuses an
      // oversized result rather than resolving credentials, and the result is left unchanged.
      if (admissionId !== null) request.admission_id = admissionId;
      const response = await bridge.request(request);
      if (!response) return;

      const operation = typeof response.operation === "string" ? response.operation : "";
      // Substitute ONLY on a successful preview or a preserved baseline, and only with the string the
      // bridge returned. A refusal, a skip or any partial payload leaves the result untouched.
      if ((operation === "previewed" || operation === "baseline") && typeof response.content === "string") {
        let substituted = response.content;
        // On a preview the bridge preserved the integral original and returned its reference. The notice
        // that travels with the excerpt says what it is, how much of the original it covers, which
        // questions it cannot answer and the narrowest call that reaches the rest. It carries only the
        // opaque reference id and sizes the model already sees — never the preserved content or a hash.
        const referenceId = typeof response.reference_id === "string" ? response.reference_id : null;
        if (operation === "previewed") {
          substituted = composeFilteredResult({
            preview: response.content,
            referenceId,
            originalText: text,
            excerpt: (response.excerpt ?? null) as Parameters<typeof composeFilteredResult>[0]["excerpt"],
            retrievalTool: RETRIEVE_CONTEXT_TOOL,
            style: PREVIEW_STYLE,
            cleanup: RETRIEVAL_CLEANUP,
          });
        }
        // Write into the field OpenCode actually reads for this result shape. MCP results are rebuilt from
        // `content[].text` after this hook, so writing only `output.output` would be silently discarded.
        const shape = classifyOutputShape(output);
        const written = writeOutputText(output, substituted);
        if (!written) {
          log("warn", "result not filtered: output shape not writable", { shape });
          return;
        }
        log("info", "result filtered", {
          operation,
          shape,
          bytes: new TextEncoder().encode(substituted).byteLength,
          retrievable: referenceId !== null,
        });
      }
    },

    // A retrieval is for the answer in flight. Once its turn is closed, its result is cleared in the
    // messages sent to the model (the stored session is untouched), so the original a model fetched back
    // is not re-sent with every later prompt. The call stays visible: deleting it made the model doubt its
    // own earlier answers and re-run the retrieval every turn. Experimental hook: any failure leaves the
    // messages as they are, because history cleanup must never break the agent loop.
    "experimental.chat.messages.transform": async (
      _input: unknown,
      output: { messages?: HistoryMessage[] },
    ) => {
      // Graph first half: record what arrived, before anything edits the array in flight. A no-op with the flag off.
      const graphRun = graphHook && Array.isArray(output?.messages) ? graphHook.begin(output.messages) : null;
      if (RETRIEVAL_CLEANUP && Array.isArray(output?.messages)) {
        try {
          const cleared = clearClosedRetrievalResults(output.messages, RETRIEVE_CONTEXT_TOOL);
          if (cleared > 0) log("debug", "closed retrieval results cleared in the model-visible history", { cleared });
        } catch {
          /* leave the messages untouched */
        }
      }
      // Graph second half: ask the bridge about the messages as the model would now see them, and apply the answer to
      // this request's array only. It never throws; a failure leaves the messages as they are.
      if (graphRun) await graphRun.finish();
    },

    // A public tool the agent can call to read back what a preview replaced. The
    // decision logic lives in the bridge: it authorizes the owning session, enforces the namespace and
    // lease, and returns the preserved baseline or a closed refusal. This tool exists only when the
    // plugin is loaded (arm A), so baseline `--pure` never offers it.
    tool: {
      ...graphTools(),
      [RETRIEVE_CONTEXT_TOOL]: tool({
        description: retrievalToolDescription(RETRIEVAL_CLEANUP),
        args: {
          reference_id: tool.schema
            .string()
            .describe("The reference_id shown in the notice of a filtered result."),
          pattern: tool.schema
            .string()
            .optional()
            .describe("Regex or keyword. Returns only the matching lines with line numbers and context."),
          line_range: tool.schema
            .object({
              start: tool.schema.number().int().min(1).describe("First line, 1-indexed."),
              end: tool.schema.number().int().min(1).describe("Last line, inclusive."),
            })
            .optional()
            .describe("Return exactly this span of lines. Takes precedence over pattern."),
          context_lines: tool.schema
            .number()
            .int()
            .min(0)
            .optional()
            .describe("Lines before and after each pattern match (default 5)."),
          max_chunks: tool.schema
            .number()
            .int()
            .min(1)
            .optional()
            .describe("Return this many chunks, most relevant first, shown in document order."),
          max_tokens: tool.schema
            .number()
            .int()
            .min(1)
            .optional()
            .describe("Approximate size limit of the response, in tokens."),
        },
        execute: async (args: RetrieveArgs, context: { sessionID?: string }) => {
          const referenceId = typeof args?.reference_id === "string" ? args.reference_id : "";
          const sessionID = typeof context?.sessionID === "string" ? context.sessionID : "";
          if (!referenceId || !sessionID) {
            // Closed failure: no reference or no session means nothing to authorize or retrieve.
            throw new Error("retrieve_context: a reference_id and an active session are required");
          }
          liveSessions.add(sessionID);
          const request: JsonObject = { type: "retrieve", session_id: sessionID, reference_id: referenceId };
          for (const option of READ_OPTIONS) {
            if (args[option] !== undefined && args[option] !== null) request[option] = args[option];
          }
          const response = await bridge.request(request);
          if (response && response.operation === "retrieved" && typeof response.content === "string") {
            log("info", "context retrieved", {
              bytes: new TextEncoder().encode(response.content).byteLength,
              read_mode: String(response.read_mode ?? "full"),
            });
            return response.content;
          }
          // A malformed read option from the authorized owner is correctable: say how (the message never
          // quotes content), so the model can fix the call instead of falling back to a full read.
          if (response && response.operation === "invalid_request" && typeof response.message === "string") {
            log("info", "context retrieve invalid request");
            throw new Error(`retrieve_context: ${response.message}`);
          }
          // A refusal (or an unavailable bridge) surfaces only a stable, payload-free message: no
          // content, reference, size or hash is ever leaked back through the error.
          log("info", "context retrieve refused");
          throw new Error("retrieve_context: no authorized content is available for that reference");
        },
      }),
    },
  };
};

export default ContextEngineeringPlugin;
