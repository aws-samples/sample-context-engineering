"""Fetch, pin, verify and install the real AWS documentation pages the CE-60 fixture needs.

Why this exists. The CE-60 documentation tools (``read_aws_documentation`` / ``search_aws_documentation``) serve
pages from ``validation/community-plugin-A-B-D/.cache/<key>.txt``. That directory is gitignored and was never
populated on this machine, so the offline MCP fixture fell back to a 7.9 KB stub ("AWS documentation content
unavailable." x200) in BOTH arms. The model then answered the documentation turns from memory and the
benchmark had almost no filterable documentation mass. The fixture's own docstring says the pages must be
byte-identical across arms, so the fix is to populate the cache once and PIN it.

Subcommands:
    fetch   --out DIR            Download the pages (no stub fallback), write ``DIR/<key>.txt`` and ``MANIFEST.json``.
    verify  --dir DIR [--manifest FILE]
                                 Fail unless every page is present, real (not a stub) and matches its pinned digest.
    install --from DIR --to DIR  Copy verified pages into the fixture cache; refuses to overwrite a different file.
    probe                        Start the fixture MCP server and require it to serve real (non-stub) Lambda docs.

The pages are public AWS documentation fetched with a plain HTTPS GET; nothing from this repository, no
credential and no user data is sent. Page text is gitignored (``*.txt``); only ``MANIFEST.json`` is tracked, so a
run records which exact pages it saw.

Exit codes: 0 ok, 1 verification/installation failure, 2 usage error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STUB_MARKERS = ("[offline stub for", "AWS documentation content unavailable.")
MIN_PAGE_CHARS = 1_500
"""Floor for a real page. The shortest legitimate page in the corpus is an index of about 2 KB; an access error,
a redirect shell or an empty body is well under 1 KB. The offline stub is caught by its marker text, not by its
length (it is 7.9 KB)."""

COMPOSED_PAGES: dict[str, tuple[str, ...]] = {
    "lambda_invocation": (
        "https://docs.aws.amazon.com/lambda/latest/dg/lambda-invocation.html",
        "https://docs.aws.amazon.com/lambda/latest/dg/invocation-async.html",
        "https://docs.aws.amazon.com/lambda/latest/dg/invocation-async-configuring.html",
        "https://docs.aws.amazon.com/lambda/latest/dg/invocation-async-error-handling.html",
    ),
}
"""Declared fixture repair. The corpus key ``lambda_invocation`` points at ``lambda-invocation.html``, which was
a single long page when the C1 turn and its oracle were written ("retries the function twice" sat mid-page).
AWS has since split the asynchronous-invocation material into separate pages and the overview is now about
3 KB with no retry semantics. Pinning only the old URL would silently reproduce "the docs do not contain the
answer". So this one key is the overview followed by the three pages that now carry that material. Every
other key is exactly its corpus URL. The sources are recorded in the manifest."""

USER_AGENT = "strands-context-validation/1.0"
MANIFEST_NAME = "MANIFEST.json"
SCHEMA_VERSION = 1


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_stub(text: str) -> bool:
    """Return whether ``text`` is the offline stub or an unusably short page."""
    return len(text) < MIN_PAGE_CHARS or any(marker in text for marker in STUB_MARKERS)


def manifest_digest(pages: dict[str, dict[str, Any]]) -> str:
    """Digest over the pinned (key, url, sha256) triples, independent of fetch time."""
    material = "\n".join(
        f"{key}|{'+'.join(pages[key].get('sources') or [pages[key]['url']])}|{pages[key]['sha256']}"
        for key in sorted(pages)
    )
    return sha256_text(material)


def _community_corpus(repo_root: Path) -> Any:
    """Import the fixture's corpus module (page list and HTML-to-text) without touching its cache."""
    sys.path.insert(0, str(repo_root / "validation" / "community-plugin-A-B-D"))
    try:
        from src import corpus  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
    return corpus


def fetch(out_dir: Path, repo_root: Path, *, getter: Any = None) -> dict[str, Any]:
    """Download every corpus page into ``out_dir`` and write the manifest.

    Args:
        out_dir: Destination directory (created).
        repo_root: Repository root, used to read the corpus page list.
        getter: Optional ``(url) -> str`` returning raw HTML; defaults to an ``httpx`` GET. Injected by tests.

    Returns:
        The manifest dict.

    Raises:
        RuntimeError: when a page cannot be downloaded or is not a real page. Nothing is replaced by a stub.
    """
    corpus = _community_corpus(repo_root)
    if getter is None:
        import httpx

        def getter(url: str) -> str:  # noqa: F811 - default network getter
            response = httpx.get(
                url, timeout=30.0, follow_redirects=True, headers={"User-Agent": USER_AGENT}
            )
            response.raise_for_status()
            return response.text

    out_dir.mkdir(parents=True, exist_ok=True)
    pages: dict[str, dict[str, Any]] = {}
    for key, url in sorted(corpus._DOC_SOURCES.items()):
        sources = COMPOSED_PAGES.get(key, (url,))
        texts: list[str] = []
        for source in sources:
            try:
                part = corpus._html_to_text(getter(source))
            except Exception as error:  # noqa: BLE001 - surface any transport/HTTP failure as one error
                raise RuntimeError(f"could not download {key} ({source}): {type(error).__name__}: {error}") from error
            if is_stub(part):
                raise RuntimeError(f"{key} ({source}) is not a real page ({len(part)} chars)")
            texts.append(part)
        text = "\n\n".join(texts)
        (out_dir / f"{key}.txt").write_text(text, encoding="utf-8")
        pages[key] = {
            "url": url,
            "sources": list(sources),
            "sha256": sha256_text(text),
            "chars": len(text),
            "bytes": len(text.encode("utf-8")),
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "fetched_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "user_agent": USER_AGENT,
        "pages": pages,
        "manifest_digest": manifest_digest(pages),
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def verify(directory: Path, manifest_path: Path | None = None) -> list[str]:
    """Return the list of problems (empty when the directory matches the pinned manifest).

    Args:
        directory: Directory holding ``<key>.txt`` pages.
        manifest_path: The manifest to check against; defaults to ``directory/MANIFEST.json``.
    """
    path = manifest_path or directory / MANIFEST_NAME
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return [f"manifest unreadable: {path} ({type(error).__name__})"]
    pages = manifest.get("pages")
    if not isinstance(pages, dict) or not pages:
        return ["manifest has no pages"]
    problems: list[str] = []
    if manifest.get("manifest_digest") != manifest_digest(pages):
        problems.append("manifest digest does not match its pages")
    for key, entry in sorted(pages.items()):
        file = directory / f"{key}.txt"
        if not file.is_file():
            problems.append(f"{key}: missing")
            continue
        text = file.read_text(encoding="utf-8")
        if is_stub(text):
            problems.append(f"{key}: stub or too short ({len(text)} chars)")
        elif sha256_text(text) != entry.get("sha256"):
            problems.append(f"{key}: content differs from the pinned digest")
    return problems


def install(source: Path, target: Path) -> list[str]:
    """Copy verified pages into ``target``; never overwrite a different existing file.

    Returns:
        The list of problems (empty on success). Nothing is copied when verification of ``source`` fails.
    """
    problems = verify(source)
    if problems:
        return problems
    manifest = json.loads((source / MANIFEST_NAME).read_text(encoding="utf-8"))
    target.mkdir(parents=True, exist_ok=True)
    for key in sorted(manifest["pages"]):
        src_file, dst_file = source / f"{key}.txt", target / f"{key}.txt"
        if dst_file.exists() and dst_file.read_bytes() != src_file.read_bytes():
            # An existing stub from an earlier offline run is the only difference we are allowed to replace.
            if not is_stub(dst_file.read_text(encoding="utf-8")):
                problems.append(f"{key}: refusing to overwrite a different real page at {dst_file}")
                continue
        shutil.copyfile(src_file, dst_file)
    return problems or verify(target, source / MANIFEST_NAME)


def probe_mcp(repo_root: Path) -> list[str]:
    """Start the fixture's MCP server and confirm it serves REAL documentation, not the offline stub.

    This is the end-to-end proof the campaign preflight needs: the cache can verify and the server can still
    read a different place. It asks for the Lambda topic, as the C1 turn does, and requires real prose that
    contains the documented retry sentence. No model, no credential and no network are involved.
    """
    import subprocess

    server = repo_root / "validation" / "opencode_mcp_server.py"
    python = os.environ.get("PYTHON", "python3")
    proc = subprocess.Popen(
        [python, str(server)],
        cwd=repo_root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    def rpc(method: str, params: dict[str, Any] | None = None, request_id: int | None = None) -> Any:
        assert proc.stdin is not None and proc.stdout is not None
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        if request_id is not None:
            message["id"] = request_id
        proc.stdin.write(json.dumps(message) + "\n")
        proc.stdin.flush()
        return json.loads(proc.stdout.readline()) if request_id is not None else None

    problems: list[str] = []
    try:
        rpc("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "probe", "version": "0"}}, 1)
        rpc("notifications/initialized")
        reply = rpc(
            "tools/call",
            {"name": "read_aws_documentation", "arguments": {"topic": "lambda", "include_related": True, "min_chars": 1000}},
            2,
        )
        text = reply["result"]["content"][0]["text"]
        if any(marker in text for marker in STUB_MARKERS):
            problems.append("MCP server returned the offline stub for topic 'lambda'")
        elif "two more times" not in text.lower():
            problems.append("MCP server's Lambda page does not contain the documented retry sentence")
        elif len(text) < 20_000:
            problems.append(f"MCP server's Lambda payload is unexpectedly small ({len(text)} chars)")
    except Exception as error:  # noqa: BLE001 - any protocol failure is a failed probe
        problems.append(f"MCP probe failed: {type(error).__name__}: {error}")
    finally:
        try:
            proc.stdin and proc.stdin.close()
        except OSError:
            pass
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            proc.kill()
    return problems


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="docs_cache", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fetch_p = sub.add_parser("fetch")
    fetch_p.add_argument("--out", required=True)
    verify_p = sub.add_parser("verify")
    verify_p.add_argument("--dir", required=True)
    verify_p.add_argument("--manifest", default=None)
    install_p = sub.add_parser("install")
    install_p.add_argument("--from", dest="source", required=True)
    install_p.add_argument("--to", dest="target", required=True)
    sub.add_parser("probe")
    args = parser.parse_args(argv[1:])

    if args.command == "fetch":
        try:
            manifest = fetch(Path(args.out), _repo_root())
        except RuntimeError as error:
            print(f"fetch failed: {error}", file=sys.stderr)
            return 1
        sizes = {key: entry["chars"] for key, entry in manifest["pages"].items()}
        print(json.dumps({"pages": len(sizes), "chars": sizes, "manifest_digest": manifest["manifest_digest"]}))
        return 0
    if args.command == "verify":
        problems = verify(Path(args.dir), Path(args.manifest) if args.manifest else None)
    elif args.command == "probe":
        problems = probe_mcp(_repo_root())
    else:
        problems = install(Path(args.source), Path(args.target))
    if problems:
        for problem in problems:
            print(f"docs-cache: {problem}", file=sys.stderr)
        return 1
    print(f"docs-cache: {args.command} ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
