"""Reading a recorded session, and keeping the Python serializer in step with the TypeScript one."""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
from pathlib import Path

import pytest

from opencode_gate_d.recorded import (
    CLEARED_OUTPUT,
    copied_store,
    find_session,
    request_prefixes,
    serialize_message,
    wire_transcript,
)

REPO = Path(__file__).resolve().parents[3]

# OpenCode-shaped parts: exactly the fields the store keeps, including ones the wire must drop.
PARTS = {
    "m1": [
        {"type": "text", "text": "hello"},
        {"type": "text", "text": "plumbing", "synthetic": True},
        {"type": "file", "url": "x"},
    ],
    "m2": [
        {"type": "step-start"},
        {"type": "reasoning", "text": "thinking"},
        {"type": "text", "text": "let me look"},
        {
            "type": "tool",
            "tool": "read",
            "callID": "c1",
            "state": {"status": "completed", "input": {"p": 1}, "output": "BODY", "time": {"start": 1, "end": 2}},
        },
        {
            "type": "tool",
            "tool": "read",
            "callID": "c2",
            "state": {
                "status": "completed",
                "input": {"p": 2},
                "output": "OLD",
                "time": {"start": 1, "end": 2, "compacted": 9},
            },
        },
        {
            "type": "tool",
            "tool": "bash",
            "callID": "c3",
            "state": {"status": "error", "input": {"cmd": "x"}, "error": "boom"},
        },
        {"type": "tool", "tool": "bash", "callID": "c4", "state": {"status": "running", "input": {}}},
    ],
    "m3": [{"type": "text", "text": "done", "ignored": True}],
}
ROLES = {"m1": "user", "m2": "assistant", "m3": "assistant"}


def build_store(directory: Path) -> Path:
    connection = sqlite3.connect(directory / "opencode.db")
    connection.executescript(
        """
        create table session (id text primary key, title text, time_created integer);
        create table message (id text primary key, session_id text, time_created integer, data text);
        create table part (id text primary key, message_id text, session_id text, time_created integer, data text);
        """
    )
    connection.execute("insert into session values ('ses_1', 'a title', 1)")
    connection.execute("insert into session values ('ses_2', 'a title', 2)")
    for index, (message_id, parts) in enumerate(PARTS.items()):
        connection.execute(
            "insert into message values (?, 'ses_1', ?, ?)",
            (message_id, index, json.dumps({"role": ROLES[message_id]})),
        )
        for number, part in enumerate(parts):
            connection.execute(
                "insert into part values (?, ?, 'ses_1', ?, ?)",
                (f"{message_id}p{number}", message_id, number, json.dumps(part)),
            )
    connection.execute("insert into message values ('sys', 'ses_1', 9, ?)", (json.dumps({"role": "system"}),))
    connection.commit()
    connection.close()
    return directory


def test_the_newest_session_with_a_title_is_found_and_a_missing_one_is_none(tmp_path):
    with copied_store(build_store(tmp_path)) as connection:
        assert find_session(connection, "a title") == "ses_2"
        assert find_session(connection, "nope") is None


def test_the_store_is_read_from_a_copy_that_is_deleted_afterwards(tmp_path):
    store = build_store(tmp_path)
    with copied_store(store) as connection:
        copy = Path(connection.execute("pragma database_list").fetchone()[2]).parent
        assert copy != store and copy.exists()
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("insert into session values ('x', 'y', 3)")
    assert not copy.exists()
    assert (store / "opencode.db").exists()


def test_the_wire_keeps_what_the_binding_reads_and_drops_the_rest(tmp_path):
    with copied_store(build_store(tmp_path)) as connection:
        wire = wire_transcript(connection, "ses_1")
    assert [message["id"] for message in wire] == ["m1", "m2", "m3"], "system messages are not part of the transcript"
    assert wire[0]["parts"] == [
        {"type": "text", "text": "hello"},
        {"type": "text", "text": "plumbing", "synthetic": True},
    ]
    tools = [part for part in wire[1]["parts"] if part["type"] == "tool"]
    assert [(t["callID"], t["status"]) for t in tools] == [
        ("c1", "completed"),
        ("c2", "completed"),
        ("c3", "error"),
        ("c4", "running"),
    ]
    assert tools[0]["output"] == "BODY" and tools[1]["output"] == CLEARED_OUTPUT and tools[2]["error"] == "boom"
    assert "output" not in tools[3]
    assert wire[2]["parts"] == [{"type": "text", "text": "done", "ignored": True}]


def test_each_assistant_message_defines_one_request_prefix(tmp_path):
    with copied_store(build_store(tmp_path)) as connection:
        wire = wire_transcript(connection, "ses_1")
    prefixes = request_prefixes(wire)
    assert [[m["id"] for m in prefix] for prefix in prefixes] == [["m1"], ["m1", "m2"]]


@pytest.mark.skipif(shutil.which("bun") is None, reason="bun is not installed")
def test_python_and_typescript_serialize_the_same_transcript_identically():
    messages = [
        {"info": {"id": message_id, "role": ROLES[message_id], "sessionID": "ses_1"}, "parts": parts}
        for message_id, parts in PARTS.items()
    ]
    script = (
        "import { serializeTranscript } from './.opencode/lib/graph.ts';"
        f"const messages = {json.dumps(messages)};"
        "console.log(JSON.stringify(serializeTranscript(messages)));"
    )
    done = subprocess.run(["bun", "-e", script], cwd=REPO, capture_output=True, text=True, check=True, timeout=60)
    expected = [serialize_message(ROLES[message_id], message_id, parts) for message_id, parts in PARTS.items()]
    assert json.loads(done.stdout) == expected


def test_a_compaction_summary_is_flagged_on_the_wire_by_both_serializers(tmp_path):
    parts = [{"type": "text", "text": "summary of the earlier work"}]
    connection = sqlite3.connect(tmp_path / "opencode.db")
    connection.executescript(
        """
        create table session (id text primary key, title text, time_created integer);
        create table message (id text primary key, session_id text, time_created integer, data text);
        create table part (id text primary key, message_id text, session_id text, time_created integer, data text);
        """
    )
    connection.execute("insert into session values ('ses_1', 't', 1)")
    for index, (message_id, data) in enumerate(
        [
            ("s1", {"role": "assistant", "summary": True}),
            ("s2", {"role": "assistant"}),
            ("s3", {"role": "user", "summary": True}),
        ]
    ):
        connection.execute("insert into message values (?, 'ses_1', ?, ?)", (message_id, index, json.dumps(data)))
        connection.execute(
            "insert into part values (?, ?, 'ses_1', 0, ?)", (f"{message_id}p", message_id, json.dumps(parts[0]))
        )
    connection.commit()
    connection.close()

    with copied_store(tmp_path) as store:
        wire = wire_transcript(store, "ses_1")
    assert [message.get("summary") for message in wire] == [True, None, None], (
        "only an assistant message carries the flag"
    )
    assert serialize_message("assistant", "x", parts, summary=True)["summary"] is True
    assert "summary" not in serialize_message("user", "x", parts, summary=True)

    if shutil.which("bun") is None:
        pytest.skip("bun is not installed")
    messages = [
        {"info": {"id": "s1", "role": "assistant", "sessionID": "s", "summary": True}, "parts": parts},
        {"info": {"id": "s2", "role": "assistant", "sessionID": "s"}, "parts": parts},
        {"info": {"id": "s3", "role": "user", "sessionID": "s", "summary": True}, "parts": parts},
    ]
    script = (
        "import { serializeTranscript } from './.opencode/lib/graph.ts';"
        f"const messages = {json.dumps(messages)};"
        "console.log(JSON.stringify(serializeTranscript(messages)));"
    )
    done = subprocess.run(["bun", "-e", script], cwd=REPO, capture_output=True, text=True, check=True, timeout=60)
    assert json.loads(done.stdout) == wire
