import { describe, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { clearClosedRetrievalResults, type HistoryMessage } from "./history.ts";
import {
  CLEARED_OUTPUT,
  EvidenceLog,
  GraphHook,
  applyProjection,
  buildCollapseMessage,
  buildSyntheticMessage,
  canonicalJson,
  checkProjection,
  digestOf,
  fingerprintTranscript,
  parseGraphFlag,
  parseGraphMode,
  parseProjection,
  serializeTranscript,
  sessionIdOf,
  type GraphEvidence,
  type GraphMessage,
} from "./graph.ts";

const SESSION = "ses_test";
const CANARY = "SECRET-PAYLOAD-CANARY";

let counter = 0;
const text = (value: string, extra: Record<string, unknown> = {}) => ({ type: "text", text: value, ...extra });
const call = (id: string, output: string, extra: Record<string, unknown> = {}, time: Record<string, unknown> | null = { start: 1, end: 2 }) => ({
  type: "tool",
  tool: "read_docs",
  callID: id,
  state: { status: "completed", input: { q: id }, output, time, ...extra },
});
const user = (id: string, ...parts: unknown[]): GraphMessage => ({
  info: { id, role: "user", sessionID: SESSION, agent: "build", model: { providerID: "p", modelID: "m" } },
  parts: parts as GraphMessage["parts"],
});
const assistant = (id: string, ...parts: unknown[]): GraphMessage => ({
  info: { id, role: "assistant", sessionID: SESSION },
  parts: parts as GraphMessage["parts"],
});

/** Three finished turns and an open one: u0 a0 f0 | u1 a1 f1 | u2 a2 f2 | u3 */
function conversation(): GraphMessage[] {
  const messages: GraphMessage[] = [];
  for (let turn = 0; turn < 3; turn++) {
    messages.push(user(`u${turn}`, text(`question ${turn}`)));
    messages.push(assistant(`a${turn}`, { type: "step-start" }, text(`reading ${turn}`), call(`c${turn}`, `ledger ${turn} ${CANARY}`)));
    messages.push(assistant(`f${turn}`, text(`answer ${turn}`)));
  }
  messages.push(user("u3", text("the open question")));
  return messages;
}

describe("parseGraphFlag", () => {
  test("only an explicit on switches the graph on", () => {
    for (const on of ["on", "1", "true", "yes", " ON "]) expect(parseGraphFlag(on)).toBe(true);
    for (const off of [undefined, "", "off", "0", "false", "no", "maybe", "enabled"]) expect(parseGraphFlag(off)).toBe(false);
  });
});

describe("serializeTranscript", () => {
  test("keeps ids, roles, text and tool calls with their status, input and result", () => {
    const wire = serializeTranscript(conversation());
    expect(wire.map((message) => message.id)).toEqual(["u0", "a0", "f0", "u1", "a1", "f1", "u2", "a2", "f2", "u3"]);
    expect(wire[1]).toEqual({
      id: "a0",
      role: "assistant",
      parts: [
        { type: "text", text: "reading 0" },
        { type: "tool", tool: "read_docs", callID: "c0", status: "completed", input: { q: "c0" }, output: `ledger 0 ${CANARY}` },
      ],
    });
  });

  test("a result the host marked compacted travels as the placeholder the model would see", () => {
    const messages = [user("u0", text("q")), assistant("a0", call("c0", "BIG", {}, { start: 1, end: 2, compacted: 3 }))];
    const part = serializeTranscript(messages)[1].parts[0];
    expect(part).toMatchObject({ type: "tool", output: CLEARED_OUTPUT });
  });

  test("an errored call carries its error text, and a running call carries no result", () => {
    const messages = [
      user("u0", text("q")),
      assistant(
        "a0",
        { type: "tool", tool: "bash", callID: "bad", state: { status: "error", input: {}, error: "boom" } },
        { type: "tool", tool: "bash", callID: "live", state: { status: "running", input: {} } },
      ),
    ];
    const parts = serializeTranscript(messages)[1].parts;
    expect(parts[0]).toMatchObject({ status: "error", error: "boom" });
    expect(parts[1]).toMatchObject({ status: "running" });
    expect(parts[1]).not.toHaveProperty("output");
  });

  test("synthetic and ignored text keep their flags, reasoning and step markers are left out, junk is skipped", () => {
    const messages = [
      user("u0", text("real"), text("plumbing", { synthetic: true }), text("hidden", { ignored: true }), { type: "file", url: "x" }),
      assistant("a0", { type: "reasoning", text: "thinking" }, { type: "step-start" }),
      { info: { id: "x", role: "system" }, parts: [] } as GraphMessage,
      { info: null } as GraphMessage,
    ];
    const wire = serializeTranscript(messages);
    expect(wire.map((message) => message.id)).toEqual(["u0", "a0"]);
    expect(wire[0].parts).toEqual([
      { type: "text", text: "real" },
      { type: "text", text: "plumbing", synthetic: true },
      { type: "text", text: "hidden", ignored: true },
    ]);
    expect(wire[1].parts).toEqual([]);
  });
});

describe("canonical digests", () => {
  test("object key order never changes the bytes, undefined is omitted, arrays keep their order", () => {
    expect(canonicalJson({ b: 1, a: { d: [3, 2], c: undefined } })).toBe('{"a":{"d":[3,2]},"b":1}');
    expect(digestOf({ a: 1, b: 2 })).toBe(digestOf({ b: 2, a: 1 }));
    expect(digestOf([1, 2])).not.toBe(digestOf([2, 1]));
    expect(digestOf({ a: 1 })).toMatch(/^[0-9a-f]{64}$/);
  });

  test("the fingerprint sees an edited result and a dropped message but ignores the in-flight compaction marker", () => {
    const original = conversation();
    const base = digestOf(fingerprintTranscript(original));

    const cleared = conversation();
    clearClosedRetrievalResults(cleared as HistoryMessage[], "read_docs");
    expect((cleared[1].parts![2].state!.time as { compacted?: number }).compacted).toBeTypeOf("number");
    expect(digestOf(fingerprintTranscript(cleared))).toBe(base);

    const edited = conversation();
    edited[1].parts![2].state!.output = "changed";
    expect(digestOf(fingerprintTranscript(edited))).not.toBe(base);

    expect(digestOf(fingerprintTranscript(original.slice(1)))).not.toBe(base);
  });
});

describe("parseProjection and checkProjection", () => {
  test("only a projected answer with well-formed ids is applied", () => {
    expect(parseProjection({ operation: "projected", removed_ids: ["a0"], final_block: "block" })).toEqual({
      removedIds: ["a0"],
      finalBlock: "block",
      collapses: [],
    });
    expect(parseProjection({ operation: "projected", removed_ids: [], final_block: "block" })).toEqual({
      removedIds: [],
      finalBlock: "block",
      collapses: [],
    });
    expect(parseProjection({ operation: "projected", removed_ids: ["a0"], final_block: null })).toEqual({
      removedIds: ["a0"],
      finalBlock: null,
      collapses: [],
    });
    for (const bad of [
      null,
      { operation: "unchanged" },
      { operation: "projected", removed_ids: "a0", final_block: "b" },
      { operation: "projected", removed_ids: [1], final_block: "b" },
      { operation: "projected", removed_ids: ["a0"], final_block: 3 },
      { operation: "projected", removed_ids: [], final_block: "   " },
    ]) {
      expect(parseProjection(bad as Record<string, unknown> | null)).toBeNull();
    }
  });

  test("rejects an answer that would remove the question, the opening message or everything, or name unknown ids", () => {
    const messages = conversation();
    const answer = (ids: string[]) => ({ removedIds: ids, finalBlock: null, collapses: [] });
    expect(checkProjection(messages, answer(["a0", "f0"]))).toBeNull();
    expect(checkProjection(messages, answer([]))).toBeNull();
    expect(checkProjection(messages, answer(["nope"]))).toBe("unknown_ids");
    expect(checkProjection(messages, answer(["u3"]))).toBe("removes_last_message");
    expect(checkProjection(messages, answer(["u0"]))).toBe("removes_first_user");
    const single = [user("only", text("q"))];
    expect(checkProjection(single, answer(["only"]))).not.toBeNull();
  });
});

describe("applyProjection", () => {
  test("splices the removed messages out in place and keeps the array identity", () => {
    const messages = conversation();
    const same = messages;
    const applied = applyProjection(messages, { removedIds: ["a0", "f0", "a1"], finalBlock: null, collapses: [] }, SESSION, 1000);
    expect(applied).toEqual({ removed: 3, inserted: 0, insertedIds: [], appended: false, syntheticId: null });
    expect(messages).toBe(same);
    expect(messages.map((message) => message.info!.id)).toEqual(["u0", "u1", "f1", "u2", "a2", "f2", "u3"]);
  });

  test("appends one trailing synthetic user message carrying the block, with the agent and model of the last user turn", () => {
    const messages = conversation();
    const applied = applyProjection(
      messages,
      { removedIds: ["a0"], finalBlock: "<collapsed_turns>x</collapsed_turns>", collapses: [] },
      SESSION,
      1234,
    );
    const last = messages[messages.length - 1];
    expect(applied.appended).toBe(true);
    expect(last.info).toMatchObject({
      id: applied.syntheticId,
      role: "user",
      sessionID: SESSION,
      agent: "build",
      model: { providerID: "p", modelID: "m" },
      time: { created: 1234 },
    });
    expect(String(last.info!.id)).toMatch(/^msg_zzzgraph[0-9a-f]{16}$/);
    expect(last.parts).toEqual([
      {
        id: expect.stringMatching(/^prt_zzzgraph/),
        sessionID: SESSION,
        messageID: last.info!.id,
        type: "text",
        text: "<collapsed_turns>x</collapsed_turns>",
        synthetic: true,
      },
    ]);
    expect(messages.filter((message) => message.info!.role === "user").length).toBe(5);
  });

  test("the synthetic id depends on the text, so the same block is the same message and a different one is not", () => {
    const messages = conversation();
    const one = buildSyntheticMessage(messages, SESSION, "same", 1);
    const two = buildSyntheticMessage(messages, SESSION, "same", 2);
    const other = buildSyntheticMessage(messages, SESSION, "other", 1);
    expect(one.info!.id).toBe(two.info!.id);
    expect(one.info!.id).not.toBe(other.info!.id);
  });

  test("the session id is read from the messages, since the hook input carries none", () => {
    expect(sessionIdOf(conversation())).toBe(SESSION);
    expect(sessionIdOf([{ info: { id: "x" } } as GraphMessage])).toBeNull();
    expect(sessionIdOf([])).toBeNull();
  });
});

describe("GraphHook (the two halves around the A path's own edit)", () => {
  function harness(respond: (payload: Record<string, unknown>) => Promise<Record<string, unknown> | null>, admission: string | null = null) {
    const records: GraphEvidence[] = [];
    const sent: Record<string, unknown>[] = [];
    const logs: string[] = [];
    const evidence = new EvidenceLog("/dev/null");
    evidence.append = (record) => void records.push(record);
    Object.defineProperty(evidence, "enabled", { get: () => true });
    const hook = new GraphHook({
      request: async (payload) => {
        sent.push(payload);
        return respond(payload);
      },
      evidence,
      admissionId: () => admission,
      log: (_level, message) => void logs.push(message),
      now: () => 5000,
    });
    return { hook, records, sent, logs };
  }

  test("applies the answer, and the record holds counts, ids and digests but never a word of the conversation", async () => {
    const { hook, records, sent } = harness(async () => ({
      operation: "projected",
      removed_ids: ["a0", "f0"],
      final_block: `<collapsed_turns>${CANARY}-BLOCK</collapsed_turns>`,
      stats: { mode: "chosen", cards: 3 },
    }), "adm-1");
    const messages = conversation();
    const pristine = digestOf(fingerprintTranscript(messages));

    const run = hook.begin(messages)!;
    clearClosedRetrievalResults(messages as HistoryMessage[], "read_docs"); // the A path edits the array in flight
    await run.finish();

    expect(sent).toHaveLength(1);
    expect(sent[0]).toMatchObject({ type: "graph_project", session_id: SESSION, admission_id: "adm-1" });
    expect((sent[0].messages as { id: string }[]).map((message) => message.id)).toContain("u3");
    expect(messages.map((message) => message.info!.id).slice(0, 3)).toEqual(["u0", "u1", "a1"]);
    expect(messages[messages.length - 1].info!.role).toBe("user");

    expect(records).toHaveLength(1);
    const record = records[0];
    expect(record).toMatchObject({
      request_index: 1,
      session_id: SESSION,
      operation: "projected",
      mode: "chosen",
      removed: 2,
      removed_ids: ["a0", "f0"],
      appended: true,
      rejected: null,
      received: { count: 10, first_id: "u0", last_id: "u3", digest: pristine },
    });
    expect(record.sent.digest).not.toBe(pristine);
    expect(record.sent.count).toBe(9);
    expect(record.block_chars).toBeGreaterThan(0);
    const serialized = JSON.stringify(record);
    for (const forbidden of [CANARY, "question 0", "reading 0", "the open question", "collapsed_turns"]) {
      expect(serialized).not.toContain(forbidden);
    }
  });

  test("an unchanged answer leaves the messages and the digest alone", async () => {
    const { hook, records } = harness(async () => ({ operation: "unchanged", stats: { mode: "delivered" } }));
    const messages = conversation();
    const before = digestOf(fingerprintTranscript(messages));
    const run = hook.begin(messages)!;
    await run.finish();
    expect(digestOf(fingerprintTranscript(messages))).toBe(before);
    expect(records[0]).toMatchObject({ operation: "unchanged", mode: "delivered", removed: 0, appended: false });
    expect(records[0].sent.digest).toBe(records[0].received.digest);
  });

  test("an unavailable bridge leaves the request unchanged and says so", async () => {
    const { hook, records } = harness(async () => null);
    const messages = conversation();
    await hook.begin(messages)!.finish();
    expect(messages).toHaveLength(10);
    expect(records[0]).toMatchObject({ operation: "bridge_unavailable", removed: 0, appended: false });
  });

  test("a request that throws leaves the request unchanged and never throws into the hook", async () => {
    const { hook, records, logs } = harness(async () => {
      throw new Error("pipe closed");
    });
    const messages = conversation();
    await expect(hook.begin(messages)!.finish()).resolves.toBeUndefined();
    expect(messages).toHaveLength(10);
    expect(records[0].operation).toBe("error");
    expect(logs).toContain("graph hook failed; the request goes unchanged");
  });

  test("an unsafe answer is rejected whole: nothing is removed and nothing is appended", async () => {
    const { hook, records, logs } = harness(async () => ({ operation: "projected", removed_ids: ["a0", "u3"], final_block: "block" }));
    const messages = conversation();
    await hook.begin(messages)!.finish();
    expect(messages).toHaveLength(10);
    expect(records[0]).toMatchObject({ operation: "rejected", rejected: "removes_last_message", removed: 0, appended: false });
    expect(logs).toContain("graph answer not applied");
  });

  test("a transcript with no session id is skipped without asking the bridge", async () => {
    const { hook, records, sent } = harness(async () => ({ operation: "projected", removed_ids: [], final_block: "x" }));
    const messages = [{ info: { id: "u0", role: "user" }, parts: [text("q")] }] as GraphMessage[];
    await hook.begin(messages)!.finish();
    expect(sent).toHaveLength(0);
    expect(records[0]).toMatchObject({ operation: "skipped", session_id: null });
  });

  test("request indexes count up per hook, and a begin on a non-array is refused", async () => {
    const { hook, records } = harness(async () => ({ operation: "unchanged" }));
    await hook.begin(conversation())!.finish();
    await hook.begin(conversation())!.finish();
    expect(records.map((record) => record.request_index)).toEqual([1, 2]);
    expect(hook.begin(undefined as unknown as GraphMessage[])).toBeNull();
  });
});

describe("EvidenceLog", () => {
  test("appends one JSON line per record and swallows a failing path", () => {
    const dir = mkdtempSync(join(tmpdir(), "graph-evidence-"));
    try {
      const path = join(dir, "evidence.jsonl");
      const log = new EvidenceLog(path);
      const record = { request_index: 1 } as unknown as GraphEvidence;
      log.append(record);
      log.append({ ...record, request_index: 2 });
      const lines = readFileSync(path, "utf8").trim().split("\n");
      expect(lines.map((line) => JSON.parse(line).request_index)).toEqual([1, 2]);
      expect(() => new EvidenceLog(join(dir, "missing", "x.jsonl")).append(record)).not.toThrow();
      expect(new EvidenceLog(null).enabled).toBe(false);
      expect(() => new EvidenceLog(null).append(record)).not.toThrow();
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

// --- the `inplace` policy -------------------------------------------------------------------------------------------

/** An assistant message as OpenCode stores it, with the fields a collapse must not carry over. */
const stored = (id: string, extra: Record<string, unknown> = {}, ...parts: unknown[]): GraphMessage => ({
  info: {
    id,
    role: "assistant",
    sessionID: SESSION,
    parentID: "u0",
    providerID: "amazon-bedrock",
    modelID: "claude",
    mode: "build",
    time: { created: 10, completed: 20 },
    tokens: { input: 1, output: 2 },
    ...extra,
  },
  parts: parts as GraphMessage["parts"],
});

const plan = (...atIds: string[]) => atIds.map((atId) => ({ atId, text: `<collapsed_turn>\nsummary of ${atId}\n</collapsed_turn>` }));

describe("parseGraphMode", () => {
  test("only an explicit inplace selects the in-place policy, as on the bridge side", () => {
    expect(parseGraphMode("inplace")).toBe("inplace");
    expect(parseGraphMode(" InPlace ")).toBe("inplace");
    for (const other of [undefined, "", "reference", "sticky", "in-place"]) expect(parseGraphMode(other)).toBe("reference");
  });
});

describe("serializeTranscript and the compaction summary", () => {
  test("an assistant message the host wrote as a compaction summary is flagged, nothing else is", () => {
    const messages = [user("u0", text("q")), assistant("a0", text("a")), stored("s0", { summary: true }, text("summary of the session"))];
    const wire = serializeTranscript(messages);
    expect(wire.map((message) => message.summary === true)).toEqual([false, false, true]);
    expect(wire[0]).not.toHaveProperty("summary");
    expect(wire[1]).not.toHaveProperty("summary");
    const asUser = [{ info: { id: "u", role: "user", summary: true, sessionID: SESSION }, parts: [text("q")] }] as GraphMessage[];
    expect(serializeTranscript(asUser)[0]).not.toHaveProperty("summary");
  });
});

describe("parseProjection with collapses", () => {
  const answer = (collapses: unknown, extra: Record<string, unknown> = {}) => ({
    operation: "projected",
    removed_ids: ["a0", "f0"],
    final_block: null,
    collapses,
    ...extra,
  });

  test("reads the summaries in order, and an answer made only of collapses and removed ids is an answer", () => {
    expect(parseProjection(answer([{ at_id: "a0", text: "S0" }, { at_id: "a1", text: "S1" }]))).toEqual({
      removedIds: ["a0", "f0"],
      finalBlock: null,
      collapses: [{ atId: "a0", text: "S0" }, { atId: "a1", text: "S1" }],
    });
    expect(parseProjection(answer([], { removed_ids: [] }))).toBeNull();
    expect(parseProjection(answer(null))).toMatchObject({ collapses: [] });
  });

  test("any malformed collapse makes the whole answer unusable, so half of it is never applied", () => {
    for (const bad of ["a0", {}, [null], [{ at_id: "a0" }], [{ at_id: "", text: "S" }], [{ at_id: "a0", text: "  " }], [{ at_id: 1, text: "S" }], [{ at_id: "a0", text: 3 }]]) {
      expect(parseProjection(answer(bad))).toBeNull();
    }
  });
});

describe("checkProjection with collapses", () => {
  const answer = (removedIds: string[], collapses: { atId: string; text: string }[]) => ({ removedIds, finalBlock: null, collapses });

  test("accepts assistant messages replaced once each by a summary", () => {
    expect(checkProjection(conversation(), answer(["a0", "f0", "a1", "f1"], plan("a0", "a1")))).toBeNull();
  });

  test("never lets a collapse remove a user message", () => {
    expect(checkProjection(conversation(), answer(["a0", "u1", "f0"], plan("a0")))).toBe("removes_user_message");
  });

  test("a summary has to take the place of a message that is itself removed, once, and of an assistant message", () => {
    const messages = conversation();
    expect(checkProjection(messages, answer(["a0", "f0"], plan("a1")))).toBe("bad_collapse");
    expect(checkProjection(messages, answer(["a0", "f0"], [...plan("a0"), ...plan("a0")]))).toBe("bad_collapse");
    expect(checkProjection(messages, answer(["a0", "f0"], plan("u0")))).toBe("bad_collapse");
    expect(checkProjection(messages, answer(["a0", "f0"], [{ atId: "a0", text: " " }]))).toBe("bad_collapse");
    expect(checkProjection(messages, answer([], plan("a0")))).toBe("bad_collapse");
  });

  test("keeps the checks it always had", () => {
    const messages = conversation();
    expect(checkProjection(messages, answer(["nope"], plan("nope")))).toBe("unknown_ids");
    expect(checkProjection(messages, answer(["a0", "u3"], plan("a0")))).toBe("removes_last_message");
  });
});

describe("applyProjection with collapses", () => {
  test("puts each summary where the first message it replaces was and takes the rest of the turn out", () => {
    const messages = conversation();
    const same = messages;
    const applied = applyProjection(
      messages,
      { removedIds: ["a0", "f0", "a1", "f1"], finalBlock: null, collapses: plan("a0", "a1") },
      SESSION,
      1000,
    );

    expect(messages).toBe(same);
    expect(messages.map((message) => message.info!.role)).toEqual(["user", "assistant", "user", "assistant", "user", "assistant", "assistant", "user"]);
    expect(messages.map((message) => message.info!.id).filter((id) => !String(id).includes("zzzgraph"))).toEqual(["u0", "u1", "u2", "a2", "f2", "u3"]);
    expect(applied.removed).toBe(4);
    expect(applied.inserted).toBe(2);
    expect(applied.insertedIds).toEqual([messages[1].info!.id, messages[3].info!.id]);
    expect(applied.appended).toBe(false);
    expect(messages[messages.length - 1].info!.id).toBe("u3");
  });

  test("the summary is an assistant message made from the one it replaces, without the markers that describe a step that is gone", () => {
    const replaced = stored(
      "a0",
      { error: { name: "X" }, summary: true, finish: "tool-calls", cost: 3 },
      text("reading"),
    );
    const message = buildCollapseMessage(replaced, SESSION, "<collapsed_turn>S</collapsed_turn>", 999);

    expect(String(message.info!.id)).toMatch(/^msg_zzzgraphc[0-9a-f]{16}$/);
    expect(message.info).toEqual({
      id: message.info!.id,
      role: "assistant",
      sessionID: SESSION,
      parentID: "u0",
      providerID: "amazon-bedrock",
      modelID: "claude",
      mode: "build",
      cost: 3,
      time: { created: 10, completed: 20 },
      tokens: { input: 1, output: 2 },
    });
    expect(message.parts).toEqual([
      {
        id: expect.stringMatching(/^prt_zzzgraphc[0-9a-f]{16}$/),
        sessionID: SESSION,
        messageID: message.info!.id,
        type: "text",
        text: "<collapsed_turn>S</collapsed_turn>",
        synthetic: true,
      },
    ]);
    expect((message.info!.time as object) !== (replaced.info!.time as object)).toBe(true);
    expect(replaced.info).toMatchObject({ error: { name: "X" }, summary: true, finish: "tool-calls" });
  });

  test("a replaced message without a time still gets one, and the id depends on both the position and the text", () => {
    const bare: GraphMessage = { info: { id: "a0", role: "assistant", sessionID: SESSION }, parts: [] };
    expect(buildCollapseMessage(bare, SESSION, "S", 77).info!.time).toEqual({ created: 77 });
    const one = buildCollapseMessage(bare, SESSION, "S", 1);
    expect(buildCollapseMessage(bare, SESSION, "S", 2).info!.id).toBe(one.info!.id);
    expect(buildCollapseMessage(bare, SESSION, "T", 1).info!.id).not.toBe(one.info!.id);
    expect(buildCollapseMessage({ ...bare, info: { ...bare.info, id: "a1" } }, SESSION, "S", 1).info!.id).not.toBe(one.info!.id);
  });

  test("the stored messages are never modified, only the array that holds them for this request", () => {
    const messages = conversation();
    const before = digestOf(fingerprintTranscript(messages));
    const originals = [...messages];
    applyProjection(messages, { removedIds: ["a0", "f0"], finalBlock: null, collapses: plan("a0") }, SESSION, 1);
    expect(digestOf(fingerprintTranscript(originals))).toBe(before);
    expect(messages).not.toContain(originals[1]);
  });

  test("a collapse can sit beside the trailing block of the reference policy without either hiding the other", () => {
    const messages = conversation();
    const applied = applyProjection(messages, { removedIds: ["a0", "f0"], finalBlock: "block", collapses: plan("a0") }, SESSION, 1);
    expect(applied).toMatchObject({ removed: 2, inserted: 1, appended: true });
    expect(messages[messages.length - 1].info!.role).toBe("user");
  });
});

describe("GraphHook with an in-place answer", () => {
  test("applies the summaries, and the record counts them without keeping a word of them", async () => {
    const records: GraphEvidence[] = [];
    const evidence = new EvidenceLog("/dev/null");
    evidence.append = (record) => void records.push(record);
    Object.defineProperty(evidence, "enabled", { get: () => true });
    const hook = new GraphHook({
      request: async () => ({
        operation: "projected",
        removed_ids: ["a0", "f0", "a1", "f1"],
        final_block: null,
        collapses: [
          { at_id: "a0", text: `<collapsed_turn>${CANARY}-ONE</collapsed_turn>` },
          { at_id: "a1", text: `<collapsed_turn>${CANARY}-TWO</collapsed_turn>` },
        ],
        stats: { mode: "chosen", policy: "inplace", cards: 3, collapsed_cards: 2 },
      }),
      evidence,
      admissionId: () => null,
      now: () => 5000,
    });
    const messages = conversation();
    const pristine = digestOf(fingerprintTranscript(messages));

    await hook.begin(messages)!.finish();

    expect(messages.map((message) => message.info!.role)).toEqual(["user", "assistant", "user", "assistant", "user", "assistant", "assistant", "user"]);
    const record = records[0];
    expect(record).toMatchObject({
      operation: "projected",
      removed: 4,
      inserted: 2,
      appended: false,
      rejected: null,
      received: { count: 10, digest: pristine },
      sent: { count: 8, synthetic_id: null },
    });
    expect(record.inserted_ids).toHaveLength(2);
    expect(record.block_chars).toBe(`<collapsed_turn>${CANARY}-ONE</collapsed_turn>`.length * 2);
    expect(record.block_digest).toMatch(/^[0-9a-f]{64}$/);
    expect(JSON.stringify(record)).not.toContain(CANARY);
  });

  test("an answer that would remove a user message is rejected whole", async () => {
    const records: GraphEvidence[] = [];
    const evidence = new EvidenceLog("/dev/null");
    evidence.append = (record) => void records.push(record);
    Object.defineProperty(evidence, "enabled", { get: () => true });
    const hook = new GraphHook({
      request: async () => ({
        operation: "projected",
        removed_ids: ["a0", "u1"],
        final_block: null,
        collapses: [{ at_id: "a0", text: "S" }],
      }),
      evidence,
      admissionId: () => null,
    });
    const messages = conversation();
    await hook.begin(messages)!.finish();
    expect(messages).toHaveLength(10);
    expect(records[0]).toMatchObject({ operation: "rejected", rejected: "removes_user_message", removed: 0, inserted: 0 });
  });
});
