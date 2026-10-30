/**
 * Practice D (context graph) on the OpenCode host: the plugin side of `context_core.opencode_graph`.
 *
 * OpenCode keeps the transcript. Just before a model call, the experimental `experimental.chat.messages.transform` hook
 * hands the plugin the messages about to be sent. This module serializes them for the bridge, applies the answer to that
 * array only, and writes down a payload-free record of what it did.
 *
 * What the answer can do, and what it can never do:
 *  - it can leave messages OUT of this one request (a splice on the array the hook was given), and append ONE trailing
 *    synthetic user message carrying the text the graph folds in;
 *  - it cannot write to the session. The array is a per-request copy OpenCode rebuilds from storage on the next call, so
 *    the stored transcript is never touched. The evidence below is how that is shown rather than asserted: each request
 *    records a digest of what arrived, and a later read of the stored session must reproduce it.
 *
 * Every failure here leaves the messages exactly as they were. A projection is an optimization of one request; it is never
 * allowed to break the agent loop.
 *
 * The synthetic message is a trailing user message, not text spliced into a real one. A tool loop ends in a tool result,
 * and the provider merges consecutive user-side messages, so the block lands after the tool result the way the Strands
 * plugin folds it into the last user message, and the stable prefix before it stays byte-identical from call to call.
 *
 * Two policies, chosen on the bridge side (`CONTEXT_GRAPH_MODE`), and this module applies whichever answer comes back:
 *  - `reference`: leave messages out and append the one trailing block, as above;
 *  - `inplace`: for each old closed turn, replace the assistant side of the turn by ONE synthetic assistant message at the
 *    position of the first message it replaces. Nothing is appended at the end and no user message is ever removed. The
 *    point is the provider's prompt cache: an edit that always lands where a turn has just closed never changes the start
 *    of the request, where the reference policy's re-decision every turn does.
 */

import { createHash } from "node:crypto";
import { appendFileSync } from "node:fs";

/** Bump when the wire shape or the application rules change, so a run record says which graph binding produced it. */
export const GRAPH_REVISION = 2;

/** `reference` re-decides what to show every turn; `inplace` collapses each old turn once, where it is. */
export type GraphMode = "reference" | "inplace";

/** Read `CONTEXT_GRAPH_MODE` the way the bridge does; anything but `inplace` is the reference policy. */
export function parseGraphMode(value: string | undefined): GraphMode {
  return (value ?? "").trim().toLowerCase() === "inplace" ? "inplace" : "reference";
}

/** What OpenCode sends the model for a tool result it has marked compacted; written into `output` on the wire. */
export const CLEARED_OUTPUT = "[Old tool result content cleared]";

/** Tool names the graph registers when it is on. They must match `context_core.opencode_graph.RETRIEVAL_TOOLS`. */
export const EXPAND_CARD_TOOL = "expand_card";
export const FIND_CONTEXT_TOOL = "find_context";

/** Model-facing descriptions. They must match `context_core.graph.retrieval` word for word; a test compares them. */
export const EXPAND_CARD_DESCRIPTION =
  "Bring back the full content of one or more earlier turns, by their titles. Earlier turns may reach you as a " +
  "title and a short description instead of their messages. When a description is not enough to answer, call this " +
  "with the titles exactly as they were shown and those turns arrive in full for the rest of this turn. Ask for " +
  "every turn you need in ONE call: a list of titles costs one retrieval where the same titles one at a time cost " +
  "one each, and each extra call grows the conversation you are about to reason over.";

/** The `inplace` policy returns a recalled turn as the tool result, so it says so. Same parity rule as the two above. */
export const EXPAND_CARD_RESULT_DESCRIPTION =
  "Bring back the full content of one or more earlier turns, by their titles. Earlier turns may reach you as a " +
  "title and a short description instead of their messages. When a description is not enough to answer, call this " +
  "with the titles exactly as they were shown and the messages of those turns arrive in the result of this call. " +
  "Ask for every turn you need in ONE call: a list of titles costs one retrieval where the same titles one at a " +
  "time cost one each, and each extra call grows the conversation you are about to reason over.";

export const FIND_CONTEXT_DESCRIPTION =
  "Find earlier turns of this conversation that match what you need, described in your words. Use this when you " +
  "suspect the conversation already covered something but you cannot see it in what reached you. Describe the need, " +
  "not a title.";

/** The shapes this module reads. Structural, so an SDK type change cannot break the build. */
export interface GraphPart {
  type?: string;
  text?: string;
  synthetic?: boolean;
  ignored?: boolean;
  tool?: string;
  callID?: string;
  state?: {
    status?: string;
    input?: unknown;
    output?: unknown;
    error?: unknown;
    time?: { compacted?: number; [key: string]: unknown } | null;
    [key: string]: unknown;
  } | null;
  [key: string]: unknown;
}

export interface GraphMessage {
  info?: { id?: string; role?: string; sessionID?: string; agent?: unknown; model?: unknown; [key: string]: unknown } | null;
  parts?: GraphPart[] | null;
  [key: string]: unknown;
}

/** One OpenCode message as the bridge reads it (`context_core.opencode_graph.to_neutral`). */
export interface WireMessage {
  id: string;
  role: "user" | "assistant";
  parts: WirePart[];
  /** An assistant message the host wrote as a compaction summary: the only copy of what it summarizes. */
  summary?: true;
}

export type WirePart =
  | { type: "text"; text: string; synthetic?: true; ignored?: true }
  | { type: "tool"; tool: string; callID: string; status: string; input: unknown; output?: string; error?: string };

/** Accept the same spellings the bridge does; anything else is off, so a typo never turns a feature on. */
export function parseGraphFlag(value: string | undefined): boolean {
  return ["1", "true", "yes", "on"].includes((value ?? "").trim().toLowerCase());
}

function asText(value: unknown): string {
  if (typeof value === "string") return value;
  if (value === null || value === undefined) return "";
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

/**
 * Serialize the transcript for the bridge.
 *
 * Only what the neutral view needs travels: text, and tool calls with their status, input and result. A tool result the
 * host has marked compacted is sent as the placeholder the model would see, so the graph describes what the model sees.
 * Calls that are still pending or running travel without a result and are left out of the neutral view.
 */
export function serializeTranscript(messages: GraphMessage[]): WireMessage[] {
  const wire: WireMessage[] = [];
  for (const message of messages) {
    const id = message?.info?.id;
    const role = message?.info?.role;
    if (typeof id !== "string" || id === "" || (role !== "user" && role !== "assistant")) continue;
    const parts: WirePart[] = [];
    for (const part of message.parts ?? []) {
      if (part?.type === "text" && typeof part.text === "string") {
        const text: WirePart = { type: "text", text: part.text };
        if (part.synthetic === true) text.synthetic = true;
        if (part.ignored === true) text.ignored = true;
        parts.push(text);
      } else if (part?.type === "tool" && typeof part.callID === "string") {
        const state = part.state ?? {};
        const status = typeof state.status === "string" ? state.status : "unknown";
        const tool: WirePart = {
          type: "tool",
          tool: typeof part.tool === "string" ? part.tool : "",
          callID: part.callID,
          status,
          input: state.input ?? {},
        };
        if (status === "completed") {
          tool.output = typeof state.time?.compacted === "number" ? CLEARED_OUTPUT : asText(state.output);
        } else if (status === "error") {
          tool.error = asText(state.error);
        }
        parts.push(tool);
      }
    }
    const entry: WireMessage = { id, role, parts };
    if (role === "assistant" && message.info?.summary === true) entry.summary = true;
    wire.push(entry);
  }
  return wire;
}

/**
 * A view of the transcript for digests: every field a durable edit could change, and nothing the host or the A path
 * legitimately marks in flight (`state.time` carries the `compacted` marker the retrieval cleanup sets on the array).
 */
export function fingerprintTranscript(messages: GraphMessage[]): unknown[] {
  return messages.map((message) => ({
    id: message?.info?.id ?? null,
    role: message?.info?.role ?? null,
    parts: (message?.parts ?? []).map((part) => {
      if (part?.type === "text") {
        return { type: "text", text: part.text ?? null, synthetic: part.synthetic === true, ignored: part.ignored === true };
      }
      if (part?.type === "tool") {
        const state = part.state ?? {};
        return {
          type: "tool",
          tool: part.tool ?? null,
          callID: part.callID ?? null,
          status: state.status ?? null,
          input: state.input ?? null,
          output: state.output ?? null,
          error: state.error ?? null,
        };
      }
      return { type: part?.type ?? null };
    }),
  }));
}

/** JSON with object keys sorted at every depth, so equal values always serialize to equal bytes. */
export function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value) ?? "null";
  if (Array.isArray(value)) return `[${value.map((item) => canonicalJson(item)).join(",")}]`;
  const record = value as Record<string, unknown>;
  const keys = Object.keys(record)
    .filter((key) => record[key] !== undefined)
    .sort();
  return `{${keys.map((key) => `${JSON.stringify(key)}:${canonicalJson(record[key])}`).join(",")}}`;
}

/** SHA-256 of the canonical JSON, hex. */
export function digestOf(value: unknown): string {
  return createHash("sha256").update(canonicalJson(value)).digest("hex");
}

/** The session the messages belong to, read from the messages themselves: the hook input carries no session id. */
export function sessionIdOf(messages: GraphMessage[]): string | null {
  for (const message of messages) {
    const id = message?.info?.sessionID;
    if (typeof id === "string" && id !== "") return id;
  }
  return null;
}

/** One `inplace` collapse: the message whose position the summary takes, and the summary. */
export interface Collapse {
  atId: string;
  text: string;
}

/** The bridge's answer to `graph_project`, validated. */
export interface Projection {
  removedIds: string[];
  finalBlock: string | null;
  /** Empty under the reference policy. Under `inplace` every collapse's `atId` is one of `removedIds`. */
  collapses: Collapse[];
}

/** Read a `projected` answer, or `null` for anything else (unchanged, skipped, error, malformed). */
export function parseProjection(response: Record<string, unknown> | null): Projection | null {
  if (!response || response.operation !== "projected") return null;
  const ids = response.removed_ids;
  if (!Array.isArray(ids) || !ids.every((id) => typeof id === "string")) return null;
  const block = response.final_block;
  if (block !== null && block !== undefined && typeof block !== "string") return null;
  const finalBlock = typeof block === "string" && block.trim() !== "" ? block : null;
  const collapses: Collapse[] = [];
  const raw = response.collapses;
  if (raw !== undefined && raw !== null) {
    if (!Array.isArray(raw)) return null;
    for (const item of raw) {
      if (!item || typeof item !== "object") return null;
      const { at_id: atId, text } = item as Record<string, unknown>;
      if (typeof atId !== "string" || atId === "" || typeof text !== "string" || text.trim() === "") return null;
      collapses.push({ atId, text });
    }
  }
  if (ids.length === 0 && finalBlock === null && collapses.length === 0) return null;
  return { removedIds: ids as string[], finalBlock, collapses };
}

/** Why an answer was not applied; recorded in the evidence, never raised. */
export type Rejection =
  | "removes_last_message"
  | "removes_first_user"
  | "removes_everything"
  | "unknown_ids"
  | "removes_user_message"
  | "bad_collapse";

/**
 * Check an answer against the array it is about to edit. The bridge already guarantees these properties; checking them
 * here too means a bug on either side leaves the request alone instead of corrupting it.
 *
 * An answer with collapses is held to more: a collapse replaces assistant messages only, so it may never remove a user
 * message, and each summary has to take the place of a message that is itself being removed, once.
 */
export function checkProjection(messages: GraphMessage[], projection: Projection): Rejection | null {
  const removed = new Set(projection.removedIds);
  const roles = new Map<string, string | undefined>();
  for (const message of messages) {
    const id = message?.info?.id;
    if (typeof id === "string") roles.set(id, message?.info?.role);
  }
  for (const id of removed) if (!roles.has(id)) return "unknown_ids";
  const last = messages[messages.length - 1]?.info?.id;
  if (typeof last === "string" && removed.has(last)) return "removes_last_message";
  const firstUser = messages.find((message) => message?.info?.role === "user")?.info?.id;
  if (typeof firstUser === "string" && removed.has(firstUser)) return "removes_first_user";
  if (removed.size > 0 && messages.every((message) => typeof message?.info?.id === "string" && removed.has(message.info.id))) {
    return "removes_everything";
  }
  if (projection.collapses.length > 0) {
    const seen = new Set<string>();
    for (const { atId, text } of projection.collapses) {
      if (!removed.has(atId) || seen.has(atId) || text.trim() === "" || roles.get(atId) !== "assistant") return "bad_collapse";
      seen.add(atId);
    }
    for (const id of removed) if (roles.get(id) !== "assistant") return "removes_user_message";
  }
  return null;
}

/** The trailing synthetic user message that carries the graph's text. It exists only in the request. */
export function buildSyntheticMessage(messages: GraphMessage[], sessionID: string, text: string, now: number): GraphMessage {
  const template = [...messages].reverse().find((message) => message?.info?.role === "user") ?? messages[messages.length - 1];
  const tag = createHash("sha256").update(text).digest("hex").slice(0, 16);
  const id = `msg_zzzgraph${tag}`;
  const info: NonNullable<GraphMessage["info"]> = {
    id,
    role: "user",
    sessionID,
    time: { created: now },
  };
  if (template?.info?.agent !== undefined) info.agent = template.info.agent;
  if (template?.info?.model !== undefined) info.model = template.info.model;
  return {
    info,
    parts: [{ id: `prt_zzzgraph${tag}`, sessionID, messageID: id, type: "text", text, synthetic: true }],
  };
}

/**
 * The assistant message an `inplace` collapse puts where the first message of the turn's assistant side was.
 *
 * It is made from the message it replaces, so the host sees an assistant message of the same session, model and parent, and
 * differs only where it has to: a new id, the summary as its one (synthetic) text part, and none of the replaced message's
 * error, summary and finish markers, which describe a step that no longer is. The id depends on the position and the text,
 * so the same collapse is the same message on every request.
 */
export function buildCollapseMessage(replaced: GraphMessage, sessionID: string, text: string, now: number): GraphMessage {
  const atId = typeof replaced?.info?.id === "string" ? replaced.info.id : "";
  const tag = createHash("sha256").update(`${atId}\0${text}`).digest("hex").slice(0, 16);
  const id = `msg_zzzgraphc${tag}`;
  const { error: _error, summary: _summary, finish: _finish, ...carried } = (replaced?.info ?? {}) as Record<string, unknown>;
  const time =
    carried.time && typeof carried.time === "object" ? { ...(carried.time as Record<string, unknown>) } : { created: now };
  const info = { ...carried, id, role: "assistant", sessionID, time } as NonNullable<GraphMessage["info"]>;
  return {
    info,
    parts: [{ id: `prt_zzzgraphc${tag}`, sessionID, messageID: id, type: "text", text, synthetic: true }],
  };
}

export interface Applied {
  removed: number;
  /** Summaries put in place of removed messages (`inplace`). */
  inserted: number;
  insertedIds: string[];
  appended: boolean;
  syntheticId: string | null;
}

/**
 * Apply an answer to the array the hook was given, in place: take out the removed messages (putting a summary where a
 * collapse says), then append the block. The array keeps its identity (OpenCode holds the same reference); only its
 * contents change, and only for this request.
 */
export function applyProjection(messages: GraphMessage[], projection: Projection, sessionID: string, now: number = Date.now()): Applied {
  const removedSet = new Set(projection.removedIds);
  const summaryAt = new Map(projection.collapses.map((collapse) => [collapse.atId, collapse.text] as const));
  let removed = 0;
  const insertedIds: string[] = [];
  if (removedSet.size > 0) {
    const kept: GraphMessage[] = [];
    for (const message of messages) {
      const id = message?.info?.id;
      if (typeof id === "string" && removedSet.has(id)) {
        removed++;
        const summary = summaryAt.get(id);
        if (summary !== undefined) {
          const replacement = buildCollapseMessage(message, sessionID, summary, now);
          insertedIds.push(String(replacement.info?.id));
          kept.push(replacement);
        }
        continue;
      }
      kept.push(message);
    }
    messages.length = 0;
    for (const message of kept) messages.push(message);
  }
  let syntheticId: string | null = null;
  if (projection.finalBlock !== null) {
    const synthetic = buildSyntheticMessage(messages, sessionID, projection.finalBlock, now);
    syntheticId = synthetic.info?.id ?? null;
    messages.push(synthetic);
  }
  return { removed, inserted: insertedIds.length, insertedIds, appended: syntheticId !== null, syntheticId };
}

/** One request's record. Counts, ids and digests; never message text, tool output, titles or the block. */
export interface GraphEvidence {
  ts: number;
  request_index: number;
  session_id: string | null;
  operation: string;
  mode: string | null;
  received: { count: number; first_id: string | null; last_id: string | null; digest: string };
  sent: { count: number; digest: string; synthetic_id: string | null };
  removed: number;
  removed_ids: string[];
  /** Summaries put in place of removed messages (`inplace`), and their ids. Both zero/empty under the reference policy. */
  inserted: number;
  inserted_ids: string[];
  appended: boolean;
  /** Characters of text the graph added to the request: the trailing block, or the summaries. */
  block_chars: number;
  block_digest: string | null;
  rejected: Rejection | null;
  elapsed_ms: number;
  stats: Record<string, unknown> | null;
}

/** Append-only JSONL writer for the evidence. A failed write is swallowed: evidence must never break the agent loop. */
export class EvidenceLog {
  constructor(private readonly path: string | null) {}

  get enabled(): boolean {
    return this.path !== null && this.path !== "";
  }

  append(record: GraphEvidence): void {
    if (!this.enabled) return;
    try {
      appendFileSync(this.path as string, `${JSON.stringify(record)}\n`);
    } catch {
      /* evidence is best effort */
    }
  }
}

/** The pieces of a bridge `graph_*` call the hook needs, so the hook logic is testable without a process. */
export type GraphRequest = (payload: Record<string, unknown>) => Promise<Record<string, unknown> | null>;

export interface GraphHookDeps {
  request: GraphRequest;
  evidence: EvidenceLog;
  admissionId: () => string | null;
  /** Called with the transcript just before the A path edits the array in place; defaults to the real fingerprint. */
  log?: (level: string, message: string, extra?: Record<string, unknown>) => void;
  now?: () => number;
}

/**
 * The graph half of the transform hook, in two steps around the A path's own edit of the same array.
 *
 * `begin` runs FIRST and records what arrived, before any in-flight edit (the A path marks closed retrieval results
 * cleared on the array). `finish` runs after, serializes what the model would see, asks the bridge and applies the answer.
 */
export class GraphHook {
  private requestIndex = 0;

  constructor(private readonly deps: GraphHookDeps) {}

  begin(messages: GraphMessage[]): GraphRun | null {
    if (!Array.isArray(messages)) return null;
    return new GraphRun(this.deps, ++this.requestIndex, messages);
  }
}

export class GraphRun {
  private readonly startedAt: number;
  private readonly receivedCount: number;
  private readonly firstId: string | null;
  private readonly lastId: string | null;
  private readonly receivedDigest: string;

  constructor(
    private readonly deps: GraphHookDeps,
    private readonly index: number,
    private readonly messages: GraphMessage[],
  ) {
    this.startedAt = (deps.now ?? Date.now)();
    this.receivedCount = messages.length;
    const ids = messages.map((message) => message?.info?.id).filter((id): id is string => typeof id === "string");
    this.firstId = ids[0] ?? null;
    this.lastId = ids[ids.length - 1] ?? null;
    this.receivedDigest = digestOf(fingerprintTranscript(messages));
  }

  async finish(): Promise<void> {
    const { messages } = this;
    const now = this.deps.now ?? Date.now;
    let operation = "skipped";
    let mode: string | null = null;
    let stats: Record<string, unknown> | null = null;
    let rejected: Rejection | null = null;
    let applied: Applied = { removed: 0, inserted: 0, insertedIds: [], appended: false, syntheticId: null };
    let removedIds: string[] = [];
    let blockChars = 0;
    let blockDigest: string | null = null;
    const sessionID = sessionIdOf(messages);

    try {
      if (sessionID !== null) {
        const payload: Record<string, unknown> = {
          type: "graph_project",
          session_id: sessionID,
          messages: serializeTranscript(messages),
        };
        const admission = this.deps.admissionId();
        if (admission !== null) payload.admission_id = admission;
        const response = await this.deps.request(payload);
        operation = response === null ? "bridge_unavailable" : String(response.operation ?? "error");
        stats = (response?.stats ?? null) as Record<string, unknown> | null;
        mode = typeof stats?.mode === "string" ? stats.mode : null;
        const projection = parseProjection(response);
        if (projection !== null) {
          rejected = checkProjection(messages, projection);
          if (rejected === null) {
            applied = applyProjection(messages, projection, sessionID, now());
            removedIds = projection.removedIds;
            // The text the graph added to the request: the trailing block, or the summaries. Only its size and digest are kept.
            const added = projection.finalBlock !== null ? [projection.finalBlock] : projection.collapses.map((collapse) => collapse.text);
            blockChars = added.reduce((total, text) => total + text.length, 0);
            blockDigest = added.length === 0 ? null : createHash("sha256").update(added.join("\n")).digest("hex");
          } else {
            operation = "rejected";
            this.deps.log?.("warn", "graph answer not applied", { rejected });
          }
        }
      }
    } catch (error) {
      operation = "error";
      this.deps.log?.("warn", "graph hook failed; the request goes unchanged", { reason: String(error) });
    }

    this.deps.evidence.append({
      ts: now(),
      request_index: this.index,
      session_id: sessionID,
      operation,
      mode,
      received: { count: this.receivedCount, first_id: this.firstId, last_id: this.lastId, digest: this.receivedDigest },
      sent: { count: messages.length, digest: digestOf(fingerprintTranscript(messages)), synthetic_id: applied.syntheticId },
      removed: applied.removed,
      removed_ids: removedIds,
      inserted: applied.inserted,
      inserted_ids: applied.insertedIds,
      appended: applied.appended,
      block_chars: blockChars,
      block_digest: blockDigest,
      rejected,
      elapsed_ms: now() - this.startedAt,
      stats,
    });
  }
}
