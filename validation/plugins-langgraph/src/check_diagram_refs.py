"""Task 10.3 checker: verify every file:line reference in the LangGraph sequence diagrams resolves.

Each diagram documents ONE binding and cites source with bare or partly-qualified filenames
(`middleware.py:NN`, `relevance/preview.py:NN`), resolved against that document's declared source roots —
its own package `src/` tree plus `context-core/src/context_core/` and the harness `src/`. This checker
mirrors that resolution per-document, then confirms each referenced line exists in the resolved file. It
prints the count checked and any failures, the audit standard the Strands diagrams are held to.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path("/local/home/scandura/sample-context-engineering")
DIAGRAMS = REPO / "docs/design/sequence/langgraph"
CORE = REPO / "context-core/src/context_core"
HARNESS = REPO / "validation/plugins-langgraph/src"

# Per-document search roots: the doc's own binding package src dir, then the core and harness trees.
PKG = REPO / "langgraph-plugins"
DOC_ROOTS: dict[str, list[Path]] = {
    "sequence-a-relevance.md": [
        PKG / "langgraph-relevance-filter/src/langgraph_relevance_filter",
        CORE / "relevance",
        CORE,
        HARNESS,
    ],
    "sequence-b-disclosure.md": [
        PKG / "langgraph-progressive-tool-disclosure/src/langgraph_progressive_tool_disclosure",
        CORE / "disclosure",
        CORE,
        HARNESS,
    ],
    "sequence-d-context-graph.md": [
        PKG / "langgraph-context-graph/src/langgraph_context_graph",
        PKG / "langgraph-context-graph",
        CORE / "graph",
        CORE,
        HARNESS / "..",
        HARNESS,
        PKG,
        REPO,
    ],
    "sequence-combined.md": [
        PKG / "langgraph-relevance-filter/src",
        PKG / "langgraph-progressive-tool-disclosure/src",
        PKG / "langgraph-context-graph/src",
        PKG / "langgraph-relevance-filter/src/langgraph_relevance_filter",
        PKG / "langgraph-progressive-tool-disclosure/src/langgraph_progressive_tool_disclosure",
        PKG / "langgraph-context-graph/src/langgraph_context_graph",
        HARNESS / "..",
        HARNESS / "../tests",
        CORE,
        HARNESS,
        REPO,
    ],
}

REF = re.compile(r"`?([A-Za-z0-9_][A-Za-z0-9_./-]*\.py):(\d+)(?:-(\d+))?`?")
_line_cache: dict[Path, int] = {}


def _resolve(path_str: str, roots: list[Path]) -> Path | None:
    for root in roots:
        candidate = (root / path_str)
        if candidate.is_file():
            return candidate
    # Fully-qualified repo-relative path.
    direct = REPO / path_str
    return direct if direct.is_file() else None


def _line_count(path: Path) -> int:
    if path not in _line_cache:
        _line_cache[path] = sum(1 for _ in path.open("rb"))
    return _line_cache[path]


def main() -> int:
    total = 0
    failures: list[str] = []
    per_doc: dict[str, tuple[int, int]] = {}

    for doc in sorted(DIAGRAMS.glob("*.md")):
        roots = DOC_ROOTS.get(doc.name, [REPO])
        text = doc.read_text(encoding="utf-8")
        d_total = 0
        d_fail = 0
        for m in REF.finditer(text):
            path_str, start_s, end_s = m.group(1), m.group(2), m.group(3)
            d_total += 1
            total += 1
            resolved = _resolve(path_str, roots)
            if resolved is None:
                failures.append(f"{doc.name}: unresolved {path_str}:{start_s}")
                d_fail += 1
                continue
            n = _line_count(resolved)
            start, end = int(start_s), int(end_s) if end_s else int(start_s)
            if start < 1 or end > n or start > end:
                failures.append(f"{doc.name}: {path_str}:{start_s}{'-' + end_s if end_s else ''} out of range ({n} lines)")
                d_fail += 1
        per_doc[doc.name] = (d_total, d_fail)

    for name, (t, f) in per_doc.items():
        print(f"{name}: {t} refs, {f} failing")
    print(f"TOTAL: {total} refs checked, {len(failures)} failing")
    for f in failures[:40]:
        print("  FAIL:", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
