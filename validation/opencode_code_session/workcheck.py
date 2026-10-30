#!/usr/bin/env python3.13
"""Check what a coding session left in its sandbox. Static and payload-free: nothing the agent wrote is executed.

``git status`` in the sandbox (a fresh repository, one commit) says exactly which files the agent created, changed or
removed. The deliverable checks are deliberately shallow and identical for both arms: do the expected files exist, do the
JSON files parse, do the Python files compile (``compile()``, no execution), does the marketplace list the four packages,
did the top-level README gain the section, and did the agent leave the existing packages alone.

The output keeps file names, sizes and line counts. It is a measurement of the session, not input to this repository's
implementation: the files themselves are deleted with the sandbox.

Usage: python3.13 validation/opencode_code_session/workcheck.py --sandbox <dir> --out <file.json>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROMPTS = Path(__file__).resolve().parent / "prompts.json"
_PROTECTED = ("context-core/", "hermes-plugins/", "langgraph-plugins/", "community-plugins/", ".opencode/", "docs/", "validation/")


def _git_status(sandbox: Path) -> list[tuple[str, str]]:
    result = subprocess.run(
        ["git", "-C", str(sandbox), "status", "--porcelain", "--untracked-files=all"],
        capture_output=True,
        text=True,
        check=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        if len(line) > 3:
            rows.append((line[:2], line[3:].strip().strip('"')))
    return rows


def _lines(path: Path) -> int:
    try:
        return len(path.read_text(encoding="utf-8", errors="replace").splitlines())
    except OSError:
        return 0


def check(sandbox: Path, expected: dict) -> dict:
    if not expected:  # the smoke prompts carry no deliverable list
        expected = {"marketplace": "claude-code-plugins/.claude-plugin/marketplace.json", "packages": [], "per_package": [], "scripts": {}, "readme_mention": "claude-code-plugins"}
    status = _git_status(sandbox)
    created = sorted(p for code, p in status if code == "??")
    modified = sorted(p for code, p in status if code != "??" and "D" not in code)
    deleted = sorted(p for code, p in status if "D" in code)

    json_files = [p for p in created if p.endswith(".json")]
    py_files = [p for p in created if p.endswith(".py")]
    json_ok = 0
    json_bad: list[str] = []
    for rel in json_files:
        try:
            json.loads((sandbox / rel).read_text(encoding="utf-8"))
            json_ok += 1
        except (OSError, ValueError):
            json_bad.append(rel)
    py_ok = 0
    py_bad: list[str] = []
    for rel in py_files:
        try:
            compile((sandbox / rel).read_text(encoding="utf-8"), rel, "exec")
            py_ok += 1
        except (OSError, SyntaxError, ValueError):
            py_bad.append(rel)

    base = expected["marketplace"].split("/")[0]
    wanted: list[str] = [expected["marketplace"]]
    for package in expected["packages"]:
        wanted += [f"{base}/{package}/{leaf}" for leaf in expected["per_package"]]
        wanted += [f"{base}/{package}/{leaf}" for leaf in expected["scripts"].get(package, [])]
    present = [p for p in wanted if (sandbox / p).is_file() and (sandbox / p).stat().st_size > 0]

    marketplace_text = ""
    market = sandbox / expected["marketplace"]
    if market.is_file():
        marketplace_text = market.read_text(encoding="utf-8", errors="replace")
    listed = [package for package in expected["packages"] if package in marketplace_text]

    readme = sandbox / "README.md"
    readme_ok = readme.is_file() and expected["readme_mention"] in readme.read_text(encoding="utf-8", errors="replace")

    touched_existing = [p for p in modified + deleted if p != "README.md" and p.startswith(_PROTECTED)]
    new_files = [{"path": p, "bytes": (sandbox / p).stat().st_size, "lines": _lines(sandbox / p)} for p in created if (sandbox / p).is_file()]

    return {
        "created_files": len(created),
        "created_lines": sum(f["lines"] for f in new_files),
        "created_bytes": sum(f["bytes"] for f in new_files),
        "created_under_plugins_dir": sum(1 for p in created if p.startswith(base + "/")),
        "modified_tracked": modified,
        "deleted_tracked": deleted,
        "touched_existing_packages": touched_existing,
        "expected_files": len(wanted),
        "expected_present": len(present),
        "expected_missing": sorted(set(wanted) - set(present)),
        "json_files": len(json_files),
        "json_ok": json_ok,
        "json_bad": json_bad,
        "python_files": len(py_files),
        "python_compile_ok": py_ok,
        "python_bad": py_bad,
        "marketplace_lists": len(listed),
        "marketplace_expected": len(expected["packages"]),
        "readme_section_present": readme_ok,
        "files": new_files,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sandbox", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=PROMPTS)
    args = parser.parse_args(argv)
    expected = json.loads(args.prompts.read_text(encoding="utf-8"))["expected_files"]
    result = check(args.sandbox.resolve(), expected)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = {k: v for k, v in result.items() if k not in {"files", "expected_missing"}}
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
