/**
 * Testable transport seam for the OpenCode context-engineering plugin.
 *
 * This module owns bridge process startup, JSONL request/response correlation and the compatibility
 * adapter for Bun's FileSink-shaped stdin. The plugin entrypoint imports only BridgeClient, keeping
 * its runtime module surface to the single default plugin export OpenCode expects.
 */

// How the bridge is launched. Overridable through the environment so an install can point at its own
// interpreter or an installed console script, but never carries a credential or an inline secret.
export function resolveBridgeCommand(env: {
  CONTEXT_BRIDGE_COMMAND?: string;
  PYTHON?: string;
} = process.env): string {
  return env.CONTEXT_BRIDGE_COMMAND || env.PYTHON || "python3";
}

const BRIDGE_COMMAND = resolveBridgeCommand();
const BRIDGE_ARGS = (process.env.CONTEXT_BRIDGE_ARGS || "-m context_core.opencode_bridge")
  .split(" ")
  .filter((part) => part.length > 0);

// Bound the wait on the bridge so a stuck child never blocks the agent loop. A late response is
// discarded; the original result is kept unchanged.
const REQUEST_TIMEOUT_MS = Number(process.env.CONTEXT_BRIDGE_TIMEOUT_MS || "15000");

type JsonObject = Record<string, unknown>;

type BridgeProcess = {
  stdin: unknown;
  stdout: ReadableStream<Uint8Array>;
};

type BridgeSpawnOptions = {
  stdin: "pipe";
  stdout: "pipe";
  stderr: "inherit";
};

/** The narrow process factory seam used by the transport unit tests. */
export type BridgeSpawner = (command: string[], options: BridgeSpawnOptions) => BridgeProcess;

type BridgeFileSink = {
  write: (data: Uint8Array) => unknown;
  flush?: () => unknown;
};

type BridgeStream = {
  getWriter?: () => { write: (data: Uint8Array) => unknown };
};

type BridgeWriter = {
  write: (data: Uint8Array) => Promise<void>;
};

/**
 * Adapt Bun's FileSink and the stream-shaped stdin used by other runtimes to one async writer.
 *
 * Bun.spawn({ stdin: "pipe" }) returns a FileSink in Bun 1.4, so it has `write`/`flush` rather
 * than WritableStream's `getWriter`. The stream branch remains a defensive compatibility fallback.
 */
function createBridgeWriter(stdin: unknown): BridgeWriter | null {
  const candidate = stdin as BridgeFileSink & BridgeStream;
  if (typeof candidate.write === "function") {
    return {
      write: async (data: Uint8Array) => {
        await candidate.write(data);
        if (typeof candidate.flush === "function") await candidate.flush();
      },
    };
  }
  if (typeof candidate.getWriter === "function") {
    const writer = candidate.getWriter();
    if (!writer || typeof writer.write !== "function") return null;
    return {
      write: async (data: Uint8Array) => {
        await writer.write(data);
      },
    };
  }
  return null;
}

/**
 * A persistent child process speaking JSONL, with request/response correlation by `id`.
 *
 * One request is written per line; one response is read per line. Pending requests are keyed by a
 * monotonic correlation id and resolved as their matching response line arrives. The client is created
 * lazily on the first request so a session that never triggers filtering spawns nothing.
 */
export class BridgeClient {
  private proc: BridgeProcess | null = null;
  private writer: BridgeWriter | null = null;
  private readonly spawn: BridgeSpawner;
  private readonly pending = new Map<string, { resolve: (value: JsonObject) => void; timer: Timer }>();
  private counter = 0;
  private buffer = "";
  private readonly encoder = new TextEncoder();

  constructor(
    private readonly log: (level: string, message: string, extra?: JsonObject) => void,
    options: { spawn?: BridgeSpawner } = {},
  ) {
    this.spawn =
      options.spawn ??
      ((command, spawnOptions) => Bun.spawn(command, spawnOptions) as unknown as BridgeProcess);
  }

  /** Spawn the bridge if it is not already running. Safe to call repeatedly. */
  private ensureStarted(): boolean {
    if (this.proc) return true;
    try {
      // Deliberately omit an `env` override: Bun inherits the OpenCode process environment, including
      // the model configuration needed by the bridge. Requests carry only admission metadata, never
      // credentials, model secrets or tool payloads.
      const proc = this.spawn([BRIDGE_COMMAND, ...BRIDGE_ARGS], {
        stdin: "pipe",
        stdout: "pipe",
        stderr: "inherit",
      });
      const writer = createBridgeWriter(proc.stdin);
      if (!writer) throw new TypeError("bridge stdin has no supported writer");
      this.proc = proc;
      this.writer = writer;
      void this.readLoop();
      return true;
    } catch (error) {
      this.log("error", "bridge spawn failed", { reason: String(error) });
      this.proc = null;
      this.writer = null;
      return false;
    }
  }

  /** Continuously read stdout, split into JSONL lines, and resolve pending requests by id. */
  private async readLoop(): Promise<void> {
    if (!this.proc) return;
    const reader = (this.proc.stdout as ReadableStream<Uint8Array>).getReader();
    const decoder = new TextDecoder();
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        this.buffer += decoder.decode(value, { stream: true });
        let newline = this.buffer.indexOf("\n");
        while (newline >= 0) {
          const line = this.buffer.slice(0, newline).trim();
          this.buffer = this.buffer.slice(newline + 1);
          if (line.length > 0) this.dispatch(line);
          newline = this.buffer.indexOf("\n");
        }
      }
    } catch (error) {
      this.log("warn", "bridge read loop ended", { reason: String(error) });
    } finally {
      this.failAllPending();
    }
  }

  /** Resolve the pending request named by a single response line, if it parses to an object. */
  private dispatch(line: string): void {
    let parsed: unknown;
    try {
      parsed = JSON.parse(line);
    } catch {
      return;
    }
    if (!parsed || typeof parsed !== "object") return;
    const response = parsed as JsonObject;
    const id = typeof response.id === "string" ? response.id : null;
    if (id === null) return;
    const entry = this.pending.get(id);
    if (!entry) return;
    this.pending.delete(id);
    clearTimeout(entry.timer);
    entry.resolve(response);
  }

  private failAllPending(): void {
    for (const [, entry] of this.pending) {
      clearTimeout(entry.timer);
      entry.resolve({ operation: "error" });
    }
    this.pending.clear();
  }

  /**
   * Send one request and await its response, returning `null` on spawn/transport failure or timeout.
   * A `null` result tells the caller to leave the tool result unchanged — never to inject anything.
   *
   * `options.timeoutMs` overrides the default wait for one request. The graph's projection scores the conversation
   * against an embedding model, which takes longer than a relevance-filter call, so it asks for more time; every other
   * request keeps the default.
   */
  async request(payload: JsonObject, options: { timeoutMs?: number } = {}): Promise<JsonObject | null> {
    if (!this.ensureStarted() || !this.writer) return null;
    const id = `req-${++this.counter}`;
    const line = JSON.stringify({ ...payload, id, idempotency_key: id }) + "\n";
    const promise = new Promise<JsonObject>((resolve) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        resolve({ operation: "error" });
      }, options.timeoutMs ?? REQUEST_TIMEOUT_MS);
      this.pending.set(id, { resolve, timer });
    });
    try {
      await this.writer.write(this.encoder.encode(line));
    } catch (error) {
      const entry = this.pending.get(id);
      if (entry) {
        clearTimeout(entry.timer);
        this.pending.delete(id);
      }
      this.log("warn", "bridge write failed", { reason: String(error) });
      return null;
    }
    const response = await promise;
    return response.operation === "error" ? null : response;
  }
}
