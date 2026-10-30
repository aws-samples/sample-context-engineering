#!/usr/bin/env python3.13
"""Build identical, disposable sandboxes of this repository for the real coding-session test.

The agent under test gets a shell and write access, so it never works in the real tree. This script copies the
repository into ``<dest>/<arm>/`` (one copy per arm, byte-identical), leaving out everything the agent has no business
seeing (``.git``, ``.env``, local agent and AI-DLC configuration, virtual environments, caches, benchmark results) and
scanning what is left for credential-looking files and strings. It then:

* writes a short, neutral ``AGENTS.md`` (the repository's own one is the AI-DLC onboarding block, which would only
  distract a coding agent);
* writes ``.opencode/opencode.json`` with the permission policy: everything the task needs is allowed, anything outside
  the sandbox, any network or credential tool, package installs, ``git push`` and the interactive ``question`` tool are
  denied (a denied call fails fast; an ``ask`` would block a headless session);
* makes the copy a fresh git repository with one commit, so ``git status`` later shows exactly what the agent changed.

Nothing here talks to a model or to AWS. The files the agent writes in a sandbox are throwaway: they are scored by
``workcheck.py`` and deleted, and are not part of this repository's implementation.

Usage: python3.13 validation/opencode_code_session/prepare_sandbox.py --dest /tmp/oc-code/<run> [--arms baseline plugin]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Directories never copied, wherever they appear. `results` is only skipped under validation/ (the measured campaigns).
_SKIP_DIRS = {
    ".git",
    ".kiro",
    ".vscode",
    ".idea",
    "aidlc",
    "bkp",
    "tmp",
    ".hypothesis",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    ".cache",
}
_SKIP_FILES = {".env", ".DS_Store", "AGENTS.md", "aidlc.settings.local.json"}

_SECRET_NAMES = re.compile(r"(^\.env($|\.)|\.pem$|\.p12$|\.pfx$|^id_(rsa|dsa|ecdsa|ed25519)|^credentials|^\.netrc$|^\.npmrc$|^\.pypirc$)")
_SECRET_CONTENT = {
    "aws access key id": re.compile(rb"\b(AKIA|ASIA)[0-9A-Z]{16}\b"),
    "private key block": re.compile(rb"-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
    "aws secret key assignment": re.compile(rb"aws_secret_access_key\s*[=:]\s*\S{16,}", re.IGNORECASE),
}

AGENTS_MD = """# Notes for contributors

- Python is `python3.13`. Run tests with `python3.13 -m pytest <path>`. The shared core, `context_core`, is already importable.
- Do not modify the existing packages (`context-core`, `hermes-plugins`, `langgraph-plugins`, `community-plugins`, `.opencode`) unless asked.
- Keep changes focused.
"""

_BASH_DENY = [
    "curl *", "wget *", "ssh *", "scp *", "sftp *", "nc *", "ncat *", "telnet *", "ftp *",
    "sudo *", "su *", "doas *",
    "aws *", "aws-vault *", "saml2aws *", "kinit*", "ssh-add*", "security *",
    "pip *", "pip3 *", "python* -m pip *", "uv *", "uvx *", "npm *", "npx *", "pnpm *", "yarn *",
    "bun add *", "bun install*", "bun x *", "bunx *", "brew *", "gem *", "cargo install*", "go install*",
    "git push*", "git remote*", "git clone*", "git fetch*", "git pull*", "git config*",
    "env", "printenv*", "export*", "set",
    "open *", "osascript*", "launchctl*", "killall *", "pkill *", "kill *", "shutdown*", "reboot*",
    "chmod -R *", "chown*", "dd *", "mkfs*", "diskutil*",
    "rm -rf /*", "rm -rf ~*", "rm -rf $HOME*",
]

OPENCODE_CONFIG = {
    "$schema": "https://opencode.ai/config.json",
    "autoupdate": False,
    "share": "disabled",
    # A Bedrock stream that goes quiet in the middle of a tool call is waited on for 15 minutes by default (seen once:
    # a `write` call pending for ten minutes with no output). Three minutes of silence between chunks is far beyond any
    # real generation gap and turns the stall into an error the runner can see.
    "provider": {"amazon-bedrock": {"options": {"chunkTimeout": 180000}}},
    "permission": {
        "*": "allow",
        "question": "deny",
        "doom_loop": "deny",
        "external_directory": "deny",
        "webfetch": "deny",
        "task": "deny",
        "bash": {"*": "allow", **{pattern: "deny" for pattern in _BASH_DENY}},
    },
}


def _ignore(src_root: Path):
    def ignore(directory: str, names: list[str]) -> set[str]:
        relative = Path(directory).resolve().relative_to(src_root)
        skipped: set[str] = set()
        for name in names:
            unwanted_file = name in _SKIP_FILES or name.endswith((".pyc", ".pyo"))
            # `results` is only skipped under validation/ (the measured campaigns).
            campaign_results = name == "results" and relative.parts[:1] == ("validation",)
            # The test must not be visible to the agent being tested (the probe answers are in prompts.json).
            harness = (relative.parts == ("validation",) and name == "opencode_code_session") or (
                relative.parts == (".opencode", "runner") and name == "code-session.ts"
            )
            if unwanted_file or name in _SKIP_DIRS or campaign_results or harness:
                skipped.add(name)
        return skipped

    return ignore


def _scan_for_secrets(root: Path) -> list[str]:
    """Return ``path: reason`` for every credential-looking file or string. Never returns a matched value."""
    findings: list[str] = []
    for path in root.rglob("*"):
        parts = path.relative_to(root).parts
        if ".git" in parts or parts[:2] == (".opencode", "node_modules") or not path.is_file():
            continue
        if _SECRET_NAMES.search(path.name):
            findings.append(f"{path.relative_to(root)}: file name looks like a credential")
            continue
        try:
            if path.stat().st_size > 2_000_000:
                continue
            data = path.read_bytes()
        except OSError:
            continue
        for reason, pattern in _SECRET_CONTENT.items():
            if pattern.search(data):
                findings.append(f"{path.relative_to(root)}: contains a {reason}")
    return findings


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def build_master(src: Path, master: Path) -> dict[str, object]:
    """Copy ``src`` into ``master`` and prepare it. Returns a payload-free description."""
    if master.exists():
        raise SystemExit(f"{master} already exists; choose a fresh --dest")
    shutil.copytree(src, master, ignore=_ignore(src), symlinks=False)
    # The plugin's own dependencies are needed to load it; copy them back (they are skipped everywhere else).
    modules = src / ".opencode" / "node_modules"
    if modules.is_dir():
        shutil.copytree(modules, master / ".opencode" / "node_modules", symlinks=True)

    findings = _scan_for_secrets(master)
    if findings:
        shutil.rmtree(master, ignore_errors=True)
        raise SystemExit("credential-looking content found, sandbox not created:\n  " + "\n  ".join(findings))

    (master / "AGENTS.md").write_text(AGENTS_MD, encoding="utf-8")
    (master / ".opencode" / "opencode.json").write_text(json.dumps(OPENCODE_CONFIG, indent=2) + "\n", encoding="utf-8")

    _git(master, "init", "-q", "-b", "main")
    _git(master, "add", "-A")
    _git(master, "-c", "user.name=sandbox", "-c", "user.email=sandbox@example.invalid", "commit", "-q", "-m", "sandbox snapshot")
    tracked = int(_git(master, "rev-list", "--count", "HEAD"))
    files = len(_git(master, "ls-files").splitlines())
    size = sum(p.stat().st_size for p in master.rglob("*") if p.is_file() and ".git" not in p.parts)
    return {"tree": _git(master, "rev-parse", "HEAD^{tree}"), "commits": tracked, "tracked_files": files, "bytes": size}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--src", type=Path, default=ROOT, help="repository to copy (default: this one)")
    parser.add_argument("--dest", type=Path, required=True, help="directory that will hold the per-arm sandboxes")
    parser.add_argument("--arms", nargs="+", default=["baseline", "plugin"])
    args = parser.parse_args(argv)

    dest = args.dest.resolve()
    master = dest / "_master"
    dest.mkdir(parents=True, exist_ok=True)
    info = build_master(args.src.resolve(), master)
    for arm in args.arms:
        shutil.copytree(master, dest / arm, symlinks=True)
    shutil.rmtree(master)
    manifest = {"src": "<repository>", "arms": args.arms, **info}
    (dest / "sandbox-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    json.dump(manifest, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
