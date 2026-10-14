"""What did the relevance-filter preview actually do to the context? (payload-free)

Gate A reports a median over 60 turns, which moves with every size the agent happens to request. This
script looks only at the plugin's own decisions in one repetition and answers a narrower question: for
each oversized result the plugin replaced with a preview, did the model go back and fetch the full original?

Sources, sizes and counts only (no prompt, answer, argument or result text is read or printed):
  * the bridge's payload-free metrics JSONL: for every ``previewed`` transform, the original and the
    preview size, in order;
  * the persisted arm-A OpenCode session: the preview markers (what the model actually saw), the
    ``retrieve_context`` calls and the size each returned.

A preview that the model re-expands is not a saving while the retrieval stays in the history: the context then
holds the preview AND the full result. Two net effects are reported per preview:

* ``in_turn_net_bytes`` = ``preview + retrieved - original``: what the context holds while the turn that made
  the retrieval is still open (always counted, it is what that turn's model call pays);
* ``steady_state_net_bytes``: what every LATER turn carries. With the plugin's retrieval cleanup on the
  retrieved content is dropped from the history once its turn closes, so this is ``preview - original``;
  with it off (the v3/v4 plugin) it equals the in-turn value.

A read is ``full`` when it returned at least 90 % of the original, ``partial`` when it returned less (a
pattern, a line range, some chunks), and ``none`` when the model never went back.

Usage: python3 preview_outcomes.py <run_id> <run_dir> [--summary]
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any

PREVIEW_MARKER = "[context-engineering]"
RETRIEVE_TOOL = "retrieve_context"
DB_FILES = ("opencode.db", "opencode.db-wal", "opencode.db-shm")


def read_previews(metrics_file: Path) -> list[dict[str, Any]]:
    """Return the ``previewed`` transforms in order, with their original and preview sizes."""
    previews: list[dict[str, Any]] = []
    for line in metrics_file.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("event") == "transform" and record.get("operation") == "previewed":
            previews.append(
                {
                    "tool_class": str(record.get("tool_class", "unknown")),
                    "original_bytes": int(record["input_bytes"]),
                    "bridge_preview_bytes": int(record["output_bytes"]),
                }
            )
    return previews


def read_session(run_id: str, store: Path) -> dict[str, list[dict[str, Any]]]:
    """Return the persisted previews and retrieve_context results of the arm-A session, in order."""
    tmp = Path(tempfile.mkdtemp())
    try:
        for name in DB_FILES:
            if (store / name).exists():
                shutil.copy(store / name, tmp / name)
        con = sqlite3.connect(f"file:{tmp / 'opencode.db'}?mode=ro", uri=True)
        row = con.execute(
            "select id from session where title = ? order by time_created desc limit 1", (f"ce60-A-{run_id}-A",)
        ).fetchone()
        if not row:
            raise LookupError(f"arm-A session ce60-A-{run_id}-A not found")
        tool_parts: dict[str, list[dict[str, Any]]] = {}
        for message_id, data in con.execute(
            "select message_id, data from part where session_id=? order by time_created", (row[0],)
        ):
            part = json.loads(data)
            if part.get("type") == "tool":
                tool_parts.setdefault(message_id, []).append(part)
        turn = 0
        persisted: list[dict[str, Any]] = []
        retrieved: list[dict[str, Any]] = []
        for message_id, data in con.execute(
            "select id, data from message where session_id=? order by time_created", (row[0],)
        ):
            role = json.loads(data).get("role")
            if role == "user":
                turn += 1
            if role != "assistant":
                continue
            for part in tool_parts.get(message_id, []):
                output = (part.get("state") or {}).get("output")
                size = len(output.encode("utf-8")) if isinstance(output, str) else 0
                if part.get("tool") == RETRIEVE_TOOL:
                    retrieved.append({"turn": turn, "bytes": size})
                elif isinstance(output, str) and PREVIEW_MARKER in output:
                    persisted.append({"turn": turn, "tool": str(part.get("tool")), "bytes": size})
        con.close()
        return {"persisted": persisted, "retrieved": retrieved}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def plugin_options(run_dir: Path) -> dict[str, str]:
    """The plugin switches the run recorded (``plugin_options`` in the arm-A evidence); empty for v3/v4 runs."""
    try:
        evidence = json.loads((run_dir / "A-evidence.json").read_text(encoding="utf-8"))
        raw = str(evidence.get("host_evidence", {}).get("plugin_options", ""))
    except (OSError, ValueError):
        return {}
    return dict(item.split(":", 1) for item in raw.split(",") if ":" in item)


FULL_READ_SHARE = 0.9


def analyze(run_id: str, run_dir: Path, *, store: Path | None = None) -> dict[str, Any]:
    """Join the bridge's preview records with the session to classify each preview's outcome."""
    metrics_files = sorted(run_dir.glob("*-A-metrics.jsonl"))
    if not metrics_files:
        raise LookupError(f"no arm-A metrics JSONL in {run_dir}")
    previews = read_previews(metrics_files[0])
    session = read_session(run_id, store or Path.home() / ".local/share/opencode")
    persisted, retrieved = session["persisted"], list(session["retrieved"])
    options = plugin_options(run_dir)
    cleanup = options.get("retrieval_cleanup") == "on"

    results: list[dict[str, Any]] = []
    for index, preview in enumerate(previews):
        seen = persisted[index] if index < len(persisted) else None
        turn = seen["turn"] if seen else None
        # The model reads back in the turn that received the preview or a later one, but before the next
        # turn that received another preview: take the first retrieval in that window not yet claimed.
        later = [q["turn"] for q in persisted[index + 1 :] if turn is not None and q["turn"] > turn]
        upper = min(later) if later else None
        match = next(
            (
                r
                for r in retrieved
                if turn is not None and r["turn"] >= turn and (upper is None or r["turn"] < upper)
            ),
            None,
        )
        if match is not None:
            retrieved.remove(match)
        original = preview["original_bytes"]
        shown = seen["bytes"] if seen else preview["bridge_preview_bytes"]
        back = match["bytes"] if match else 0
        read = "none" if match is None else ("full" if back >= FULL_READ_SHARE * original else "partial")
        in_turn = shown + back - original
        results.append(
            {
                "turn_ordinal": turn,
                "tool_class": preview["tool_class"],
                "original_bytes": original,
                "preview_bytes": shown,
                "read": read,
                "re_expanded": read == "full",
                "retrieved_bytes": back,
                "in_turn_net_bytes": in_turn,
                "steady_state_net_bytes": (shown - original) if cleanup else in_turn,
            }
        )
    in_turn_total = sum(item["in_turn_net_bytes"] for item in results)
    steady_total = sum(item["steady_state_net_bytes"] for item in results)
    reads = {kind: sum(1 for item in results if item["read"] == kind) for kind in ("full", "partial", "none")}
    summary = (
        f"previews={len(results)} full_reads={reads['full']} partial_reads={reads['partial']} unread={reads['none']} "
        f"original={sum(i['original_bytes'] for i in results)}B shown={sum(i['preview_bytes'] for i in results)}B "
        f"retrieved={sum(i['retrieved_bytes'] for i in results)}B in_turn_net={in_turn_total:+d}B "
        f"steady_net={steady_total:+d}B cleanup={'on' if cleanup else 'off'} "
        f"({'net saving' if steady_total < 0 else 'no net saving'})"
    )
    return {
        "run_id": run_id,
        "plugin_options": options,
        "previews": results,
        "unmatched_persisted_previews": max(0, len(persisted) - len(previews)),
        "unmatched_retrievals": len(retrieved),
        "total": {
            "previews": len(results),
            "full_reads": reads["full"],
            "partial_reads": reads["partial"],
            "unread": reads["none"],
            "re_expanded": reads["full"],
            "in_turn_net_bytes": in_turn_total,
            "steady_state_net_bytes": steady_total,
        },
        "summary": summary,
    }


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("--")]
    if len(args) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        report = analyze(args[0], Path(args[1]))
    except (LookupError, OSError, ValueError, KeyError) as error:
        print(f"preview-outcomes unavailable: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(report["summary"] if "--summary" in argv else json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
