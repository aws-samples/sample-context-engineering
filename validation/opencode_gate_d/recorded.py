"""Read a recorded OpenCode session as the wire transcript the graph binding takes.

The Gate D contract suite replays real sessions, not only synthetic ones. OpenCode keeps every session in a SQLite file; this
module reads a COPY of it (never the live file), and rebuilds, message by message, the same serialized shape the TypeScript
plugin sends to the bridge (``.opencode/lib/graph.ts``, ``serializeTranscript``). Keeping the two in step is a test:
``test_recorded.py`` runs both over one fixture and compares the bytes.

Nothing here keeps a payload: the transcripts are returned to the caller, and the callers write digests and counts only.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

__all__ = [
    "CLEARED_OUTPUT",
    "DEFAULT_STORE",
    "copied_store",
    "find_session",
    "request_prefixes",
    "serialize_message",
    "wire_transcript",
]

CLEARED_OUTPUT = "[Old tool result content cleared]"
"""What OpenCode sends the model for a tool result it has marked compacted. Must equal ``CLEARED_OUTPUT`` in graph.ts."""

DEFAULT_STORE = Path.home() / ".local/share/opencode"
_DB_FILES = ("opencode.db", "opencode.db-wal", "opencode.db-shm")


@contextmanager
def copied_store(source: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Open a read-only connection on a temporary COPY of the OpenCode store, and delete the copy afterwards.

    The live database may be open in a running OpenCode; reading a copy never contends with it and can never write to it.
    The copy is large (hundreds of MB after a few campaigns), so it is always removed.
    """
    src = source or DEFAULT_STORE
    tmp = Path(tempfile.mkdtemp(prefix="oc-store-copy-"))
    try:
        for name in _DB_FILES:
            if (src / name).exists():
                shutil.copy(src / name, tmp / name)
        connection = sqlite3.connect(f"file:{tmp / 'opencode.db'}?mode=ro", uri=True)
        try:
            yield connection
        finally:
            connection.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def find_session(connection: sqlite3.Connection, title: str) -> str | None:
    """The newest session with exactly this title, or ``None``."""
    row = connection.execute(
        "select id from session where title = ? order by time_created desc limit 1", (title,)
    ).fetchone()
    return None if row is None else str(row[0])


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def serialize_message(
    role: str, message_id: str, parts: Sequence[dict[str, Any]], *, summary: bool = False
) -> dict[str, Any]:
    """One OpenCode message in the wire shape, exactly as ``serializeTranscript`` writes it.

    ``summary`` is the host's own marker on an assistant message that is a compaction summary; only that role carries it.
    """
    wire_parts: list[dict[str, Any]] = []
    for part in parts:
        kind = part.get("type")
        if kind == "text" and isinstance(part.get("text"), str):
            text: dict[str, Any] = {"type": "text", "text": part["text"]}
            if part.get("synthetic") is True:
                text["synthetic"] = True
            if part.get("ignored") is True:
                text["ignored"] = True
            wire_parts.append(text)
        elif kind == "tool" and isinstance(part.get("callID"), str):
            state = part.get("state") or {}
            status = state.get("status") if isinstance(state.get("status"), str) else "unknown"
            tool: dict[str, Any] = {
                "type": "tool",
                "tool": part.get("tool") if isinstance(part.get("tool"), str) else "",
                "callID": part["callID"],
                "status": status,
                "input": state.get("input") if state.get("input") is not None else {},
            }
            if status == "completed":
                compacted = isinstance((state.get("time") or {}).get("compacted"), (int, float))
                tool["output"] = CLEARED_OUTPUT if compacted else _as_text(state.get("output"))
            elif status == "error":
                tool["error"] = _as_text(state.get("error"))
            wire_parts.append(tool)
    entry: dict[str, Any] = {"id": message_id, "role": role, "parts": wire_parts}
    if summary and role == "assistant":
        entry["summary"] = True
    return entry


def wire_transcript(connection: sqlite3.Connection, session_id: str) -> list[dict[str, Any]]:
    """The whole session as wire messages, oldest first. Messages other than user and assistant are skipped."""
    parts_by_message: dict[str, list[dict[str, Any]]] = {}
    for message_id, data in connection.execute(
        "select message_id, data from part where session_id = ? order by time_created, id", (session_id,)
    ):
        parts_by_message.setdefault(str(message_id), []).append(json.loads(data))
    wire: list[dict[str, Any]] = []
    for message_id, data in connection.execute(
        "select id, data from message where session_id = ? order by time_created, id", (session_id,)
    ):
        info = json.loads(data)
        role = info.get("role")
        if role in ("user", "assistant"):
            wire.append(
                serialize_message(
                    role,
                    str(message_id),
                    parts_by_message.get(str(message_id), []),
                    summary=info.get("summary") is True,
                )
            )
    return wire


def request_prefixes(wire: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """The transcript each model request saw: everything before the assistant message that request produced.

    OpenCode builds the request from the stored messages created before the assistant message it is about to write, so the
    prefix ending just before each assistant message is the hook's input at that request. A trailing user message with no
    assistant message after it (a request that never completed) is not a request this can attribute and is left out.
    """
    return [list(wire[:index]) for index, message in enumerate(wire) if message["role"] == "assistant" and index > 0]
