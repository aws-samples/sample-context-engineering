"""The machine-checked neutral contract: importing the core must not pull in an agent framework.

``context_core`` holds all decision logic and MUST NOT depend on ``strands`` or ``langchain``/
``langgraph``. This test imports every ``context_core`` submodule in a subprocess with a clean
interpreter and asserts none of the forbidden top-level packages ended up in ``sys.modules``. It runs in
a subprocess so a framework imported by some *other* test in the session cannot mask a real leak here.
"""

from __future__ import annotations

import subprocess
import sys

FORBIDDEN = ("strands", "langchain", "langgraph", "langchain_core")

_PROBE = """
import importlib, pkgutil, sys
import context_core

# Import every submodule of context_core so a leak anywhere is caught, not just at the top level.
for mod in pkgutil.walk_packages(context_core.__path__, context_core.__name__ + "."):
    importlib.import_module(mod.name)

forbidden = __FORBIDDEN__
leaked = sorted(
    name
    for name in sys.modules
    for bad in forbidden
    if name == bad or name.startswith(bad + ".")
)
if leaked:
    print("LEAKED:" + ",".join(leaked))
    raise SystemExit(1)
print("CLEAN")
"""


def test_core_imports_no_agent_framework() -> None:
    """Every context_core submodule imports with no strands/langchain/langgraph in sys.modules."""
    probe = _PROBE.replace("__FORBIDDEN__", repr(FORBIDDEN))
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"neutral-contract probe failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "CLEAN" in result.stdout, result.stdout
