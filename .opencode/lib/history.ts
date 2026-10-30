/**
 * Keep a retrieval for the answer in flight, not for every later turn.
 *
 * The reference Strands relevance filter removes the exchanges of its retrieval tool from the history once the
 * invocation ends: a retrieval can return the whole of a large tool result, and kept in the history it is re-sent
 * with every later prompt, which is exactly the cost the preview was meant to remove. The OpenCode port had no
 * such step, so a preview that the model re-expanded left the preview AND the full original in the session for
 * good. This module is that step for OpenCode, applied to the messages OpenCode is about to send to the model
 * (the `experimental.chat.messages.transform` hook), never to the stored session.
 *
 * It CLEARS the result instead of deleting the exchange. Deleting the call left the model looking at earlier
 * answers that quoted figures with no trace of where they came from; in a live run it concluded "I claimed to
 * have read content I never retrieved" and re-ran the same retrieval every turn (23 retrievals for 2 previews),
 * including in turns about unrelated tools. OpenCode already has the right idea for old results: a completed
 * tool part marked `state.time.compacted` reaches the model as `[Old tool result content cleared]`. Marking the
 * part keeps the call visible, so the history stays truthful and well-formed (no message is dropped, no
 * reasoning is rewritten), while the large result is gone.
 *
 * Rules:
 *  - only parts of the named tool are touched, and only in turns that are already closed: everything before the
 *    newest user message. The retrieval a turn is still using stays;
 *  - only completed calls are cleared; a call that is pending, running or failed is left alone (a failure
 *    carries a short message, not content).
 */

/** The shapes this module reads. Structural, so a SDK type change cannot break the build. */
export interface HistoryPart {
  type?: string;
  tool?: string;
  state?: { status?: string; time?: { compacted?: number; [key: string]: unknown } | null; [key: string]: unknown } | null;
  [key: string]: unknown;
}

export interface HistoryMessage {
  info?: { role?: string; error?: unknown; [key: string]: unknown } | null;
  parts?: HistoryPart[] | null;
  [key: string]: unknown;
}

function isClearableExchange(part: HistoryPart, toolName: string): boolean {
  if (part.type !== "tool" || part.tool !== toolName || part.state?.status !== "completed") return false;
  const time = part.state.time;
  return time !== null && typeof time === "object" && typeof time.compacted !== "number";
}

/**
 * Mark the completed calls of `toolName` in every closed turn as cleared, in place.
 *
 * @param messages - The messages about to be sent to the model, oldest first. Mutated.
 * @param toolName - The retrieval tool whose results are cleared.
 * @param now - The timestamp recorded in `state.time.compacted` (injected for tests).
 * @returns How many tool results were cleared.
 */
export function clearClosedRetrievalResults(
  messages: HistoryMessage[],
  toolName: string,
  now: number = Date.now(),
): number {
  let lastUser = -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i]?.info?.role === "user") {
      lastUser = i;
      break;
    }
  }
  if (lastUser <= 0) return 0;

  let cleared = 0;
  for (let i = 0; i < lastUser; i++) {
    const message = messages[i];
    if (message?.info?.role !== "assistant" || !Array.isArray(message.parts)) continue;
    for (const part of message.parts) {
      if (!isClearableExchange(part, toolName)) continue;
      // isClearableExchange proved `state.time` is an object.
      (part.state!.time as { compacted?: number }).compacted = now;
      cleared++;
    }
  }
  return cleared;
}
