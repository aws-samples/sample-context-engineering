"""Acceptance check for one CE-60 paired repetition (used by the campaign scripts).

Reads only the payload-free artifacts written by the runner and prints a one-line summary.

Exit codes:
    0  clean: both arms 60/60, zero timeouts, no residual processes, Gate A is go/no_go.
    1  hard failure: anything else, including a stalled turn, an unrecovered non-provider error,
       residual processes, or a not_evaluable Gate A decision.
    3  the only failures are retryable provider errors (HTTP 5xx / isRetryable): the repetition may be
       repeated once.

Plugin effect is verified separately from the persisted OpenCode session (see ``plugin_effect``). The v4
campaign adds ``v4_conditions`` (oracle version and retrieval-attribution consistency).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

CLEAN, HARD, RETRYABLE_ONLY = 0, 1, 3
DECISION_FILE_ENV = "CE60_DECISION_FILE"
"""Name of the paired-comparison file. Gate A's own is the default; a run that also ran practice D writes the D slice's under
another name (``d-slice-comparison.json``), so its comparison can never be read as a Gate A decision."""
PREVIEW_MARKER = "[context-engineering]"
DB_FILES = ("opencode.db", "opencode.db-wal", "opencode.db-shm")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def arm_summary(out: Path, arm: str) -> tuple[str, bool, bool]:
    """Return (summary, hard_failure, retryable_only_failure) for one arm."""
    evidence = read_json(out / f"{arm}-evidence.json")
    host = evidence["host_evidence"]
    observability = host.get("observability", {})
    completed = evidence["turns_completed"]
    timeouts = observability.get("timeout_count")
    codes = observability.get("error_codes") or []
    retryable = observability.get("error_retryable_flags") or []
    residual = read_json(out / f"{arm}-cleanup.json")["residual_process_count"]

    hard = timeouts != 0 or residual != 0
    retryable_only = False
    if completed != 60:
        only_retryable_api = bool(codes) and all(c == "APIError" for c in codes) and all(
            flag == "retryable" for flag in retryable
        )
        if only_retryable_api and len(retryable) == len(codes):
            retryable_only = True
        else:
            hard = True
    text = (
        f"{arm}: completed={completed} correct={evidence['correctness_numerator']}/"
        f"{evidence['correctness_denominator']} mi_median={evidence['model_input_tokens_median']} "
        f"timeouts={timeouts} errors={codes} retryable={retryable} residual={residual}"
    )
    return text, hard, retryable_only


def check(out: Path) -> tuple[int, str]:
    gate = read_json(out / os.environ.get(DECISION_FILE_ENV, "gate-a-decision.json"))
    outcome = gate["outcome"]
    parts = [f"gate={outcome} reasons={','.join(gate['reasons']) or '-'}"]
    hard = False
    retryable_only = False
    for arm in ("baseline", "A"):
        text, arm_hard, arm_retryable = arm_summary(out, arm)
        parts.append(text)
        hard = hard or arm_hard
        retryable_only = retryable_only or arm_retryable
    host = read_json(out / "A-evidence.json")["host_evidence"]
    parts.append(
        f"A transforms={host.get('transform_count')} previewed={host.get('previewed_count')} "
        f"bytes {host.get('input_bytes')}->{host.get('output_bytes')}"
    )
    if outcome not in ("go", "no_go") and not retryable_only:
        hard = True
    if hard:
        return HARD, "; ".join(parts)
    return (RETRYABLE_ONLY if retryable_only else CLEAN), "; ".join(parts)


def plugin_effect(run_id: str) -> tuple[bool, str]:
    """Count preview markers persisted in the arm-A session (sizes/booleans only, never payload)."""
    src = Path.home() / ".local/share/opencode"
    tmp = Path(tempfile.mkdtemp())
    try:
        for name in DB_FILES:
            if (src / name).exists():
                shutil.copy(src / name, tmp / name)
        con = sqlite3.connect(f"file:{tmp / 'opencode.db'}?mode=ro", uri=True)
        row = con.execute("select id from session where title = ? order by time_created desc limit 1", (f"ce60-A-{run_id}-A",)).fetchone()
        if not row:
            return False, f"plugin-effect: arm-A session ce60-A-{run_id}-A not found"
        marked = 0
        largest_marked = 0
        for (data,) in con.execute("select data from part where session_id=?", (row[0],)):
            part = json.loads(data)
            if part.get("type") != "tool":
                continue
            output = (part.get("state") or {}).get("output")
            if isinstance(output, str) and PREVIEW_MARKER in output:
                marked += 1
                largest_marked = max(largest_marked, len(output.encode("utf-8")))
        return marked >= 1, f"plugin-effect: persisted previews={marked} largest_preview_bytes={largest_marked}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# Portuguese function words; three or more in one answer mark it as Portuguese. The oracle only knows English
# keywords, so a session that drifts into Portuguese fails its checks in whichever arm it happens in. The
# scenario itself contains one Portuguese prompt (its answer is legitimately Portuguese), hence the allowance.
_PT_WORDS = re.compile(r"\b(não|nao|você|voce|está|esta|conector|para|com|uma|são|dos|das|foi|mais|pelo|pela)\b", re.I)
_PT_MIN_HITS = 3
MAX_PORTUGUESE_ANSWERS_PER_ARM = 3


def language_drift(run_id: str, *, store: Path | None = None) -> dict[str, int | None]:
    """Count Portuguese final answers per arm from the persisted OpenCode sessions (counts only).

    Returns ``{"baseline": n, "A": n}``; a value is ``None`` when that arm's session cannot be found.
    """
    src = store or Path.home() / ".local/share/opencode"
    tmp = Path(tempfile.mkdtemp())
    counts: dict[str, int | None] = {}
    try:
        for name in DB_FILES:
            if (src / name).exists():
                shutil.copy(src / name, tmp / name)
        con = sqlite3.connect(f"file:{tmp / 'opencode.db'}?mode=ro", uri=True)
        for arm in ("baseline", "A"):
            row = con.execute(
                "select id from session where title = ? order by time_created desc limit 1",
                (f"ce60-{arm}-{run_id}-{arm}",),
            ).fetchone()
            if not row:
                counts[arm] = None
                continue
            texts: dict[str, list[str]] = {}
            for message_id, data in con.execute(
                "select message_id, data from part where session_id=? order by time_created", (row[0],)
            ):
                part = json.loads(data)
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    texts.setdefault(message_id, []).append(part["text"])
            turns: list[str] = []
            for message_id, data in con.execute(
                "select id, data from message where session_id=? order by time_created", (row[0],)
            ):
                role = json.loads(data).get("role")
                if role == "user":
                    turns.append("")
                elif role == "assistant" and turns and message_id in texts:
                    turns[-1] = "\n".join(texts[message_id])  # the last assistant text of the turn wins
            counts[arm] = sum(1 for text in turns if len(_PT_WORDS.findall(text)) >= _PT_MIN_HITS)
        con.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return counts


def v4_conditions(
    out: Path,
    run_id: str | None = None,
    *,
    store: Path | None = None,
    expect_plugin_options: str | None = None,
) -> tuple[bool, str]:
    """Conditions that make a repetition a valid v4 repetition, on top of the shared acceptance.

    * Both arms must have been scored by the v4 oracle (the evidence records which oracle judged them).
    * With a ``run_id``, neither arm may have drifted into Portuguese (the response language is pinned to
      English; more than ``MAX_PORTUGUESE_ANSWERS_PER_ARM`` Portuguese answers means the pin failed and the
      English-only oracle would be judging the language, not the plugin).
    * With ``expect_plugin_options``, arm A must have run with exactly those plugin switches (for example
      ``preview_style:scoped,retrieval_cleanup:on``), so a repetition can never be attributed to a behaviour
      it did not run with.
    * Arm A must have a retrieval count for every turn. A turn without one means the session read failed
      and "retrieval is exact" would be judged on partial data, so the repetition is not valid.
    * Arm A's per-turn retrieval attribution is compared with the bridge's own run-level counters. They
      should agree because both count every ``retrieve_context`` call; a difference is reported, never
      hidden, but it is not a hard failure by itself (a call that fails before it reaches the bridge is
      legitimately seen by the session and not by the bridge, and Gate A counts it as a failed retrieval).
    """
    problems: list[str] = []
    versions = []
    for arm in ("baseline", "A"):
        version = read_json(out / f"{arm}-evidence.json")["host_evidence"].get("oracle_version")
        versions.append(str(version))
        if version != "v4":
            problems.append(f"{arm} scored by oracle {version!r}, expected 'v4'")
    host = read_json(out / "A-evidence.json")["host_evidence"]
    bridge_attempts = host.get("retrieval_attempts")
    turns = read_json(out / "A-sanitized.json")["turn_results"]
    evaluated = sum(turn.get("retrieval_attempts") or 0 for turn in turns)
    unavailable = [turn["turn_id"] for turn in turns if turn.get("retrieval_attempts") is None]
    state = "consistent" if bridge_attempts == evaluated and not unavailable else "MISMATCH"
    summary = (
        f"v4: oracle={'/'.join(versions)} retrieval bridge={bridge_attempts} evaluated={evaluated} "
        f"unavailable_turns={len(unavailable)} ({state})"
    )
    if expect_plugin_options is not None:
        recorded = host.get("plugin_options")
        summary += f" plugin_options={recorded}"
        if recorded != expect_plugin_options:
            problems.append(f"arm A ran with plugin options {recorded!r}, expected {expect_plugin_options!r}")
    if run_id is not None:
        drift = language_drift(run_id, store=store)
        summary += f" language_pt_answers={drift}"
        for arm, count in drift.items():
            if count is None:
                problems.append(f"{arm} session not found for the language check")
            elif count > MAX_PORTUGUESE_ANSWERS_PER_ARM:
                problems.append(f"{arm} answered in Portuguese in {count} turns (pin failed)")
    if unavailable:
        problems.append(f"{len(unavailable)} arm-A turn(s) have no retrieval count: {', '.join(unavailable[:5])}")
    if problems:
        return False, f"{summary}; " + "; ".join(problems)
    return True, summary


MAX_GRAPH_UNAVAILABLE_FRACTION = 0.05
"""Share of graph requests that may end in ``error`` or ``bridge_unavailable`` before the run stops counting as D evidence."""


def graph_conditions(out: Path) -> tuple[bool, str]:
    """Conditions that make an A+D repetition valid D evidence, on top of the shared acceptance.

    Reads the two files the runner writes for a run that ran practice D (payload-free: counts, ids and digests):

    * every graph request must have been checked against the stored session and reproduce its digest (the graph is
      request-only), and no graph message may be in the stored session;
    * no answer of the bridge may have been rejected by the plugin's own safety check;
    * the graph must have engaged (``projected`` at least once), otherwise the run measured nothing about D;
    * requests that ended in ``error`` or ``bridge_unavailable`` may not exceed ``MAX_GRAPH_UNAVAILABLE_FRACTION``.
    """
    durability = read_json(out / "A-graph-durability.json")
    requests = int(durability["requests"])
    operations = durability.get("operations") or {}
    problems: list[str] = []
    if requests < 1:
        problems.append("no graph request was recorded")
    if durability["checked"] != requests:
        problems.append(f"only {durability['checked']} of {requests} graph requests could be checked against the session")
    if durability["stored_prefix_equal"] != durability["checked"]:
        problems.append(f"stored session differs from what requests were handed: {durability['mismatched_requests'][:5]}")
    if durability["stored_graph_synthetic_messages"] != 0:
        problems.append(f"{durability['stored_graph_synthetic_messages']} graph message(s) were stored in the session")
    if durability["rejected_requests"] != 0:
        problems.append(f"{durability['rejected_requests']} bridge answer(s) were rejected by the plugin's safety check")
    if int(operations.get("projected", 0)) < 1:
        problems.append("the graph never engaged (no projected request)")
    unavailable = int(operations.get("error", 0)) + int(operations.get("bridge_unavailable", 0))
    if requests and unavailable / requests > MAX_GRAPH_UNAVAILABLE_FRACTION:
        problems.append(f"{unavailable} of {requests} graph requests ended in error or bridge_unavailable")
    summary = (
        f"graph: requests={requests} projected={operations.get('projected', 0)} unchanged={operations.get('unchanged', 0)} "
        f"unavailable={unavailable} removed_messages={durability['removed_messages_total']} "
        f"block_chars={durability['block_chars_total']} tool_calls={durability.get('graph_tool_calls')} "
        f"stored_prefix_equal={durability['stored_prefix_equal']}/{durability['checked']}"
    )
    if problems:
        return False, f"{summary}; " + "; ".join(problems)
    return True, summary


def main(argv: list[str]) -> int:
    if len(argv) >= 3 and argv[1] == "graph":
        try:
            ok, text = graph_conditions(Path(argv[2]))
        except (OSError, KeyError, ValueError) as error:
            print(f"graph: unreadable artifacts ({type(error).__name__}: {error})")
            return HARD
        print(text)
        return CLEAN if ok else HARD
    if len(argv) >= 3 and argv[1] == "v4":
        try:
            ok, text = v4_conditions(
                Path(argv[2]),
                argv[3] if len(argv) >= 4 else None,
                expect_plugin_options=argv[4] if len(argv) >= 5 else None,
            )
        except (OSError, KeyError, ValueError) as error:
            print(f"v4: unreadable artifacts ({type(error).__name__}: {error})")
            return HARD
        print(text)
        return CLEAN if ok else HARD
    if len(argv) >= 3 and argv[1] == "effect":
        ok, text = plugin_effect(argv[2])
        print(text)
        return CLEAN if ok else HARD
    if len(argv) >= 2:
        try:
            code, text = check(Path(argv[1]))
        except (OSError, KeyError, ValueError) as error:
            # Missing or malformed artifacts are a hard failure, never a traceback or a pass.
            print(f"hard failure: unreadable artifacts ({type(error).__name__}: {error})")
            return HARD
        print(text)
        return code
    print(
        "usage: check_repetition.py <run_dir> | check_repetition.py effect <run_id> | "
        "check_repetition.py v4 <run_dir> [<run_id> [<plugin_options>]] | check_repetition.py graph <run_dir>",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
