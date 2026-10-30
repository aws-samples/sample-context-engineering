"""Deterministic stdio MCP fixture server for the CE-60 OpenCode harness.

The server exposes the canonical eighteen ``CORE_TOOLS`` from the CE-60 fixture
without importing OpenCode internals or making network/provider calls. Tool
schemas and descriptions come directly from the public Strands ``tool_spec``
property. Documentation payloads reuse the fixture corpus cache when available;
when a cache entry is absent, this process supplies the fixture's deterministic
offline stub instead of allowing the corpus loader to access the network.

The process implements the MCP JSON-RPC methods needed by OpenCode:
``initialize``, ``notifications/initialized``, ``tools/list`` and
``tools/call``. JSON-RPC responses are written only to stdout. Structured logs
are written to stderr and never include tool arguments or result contents.
"""

from __future__ import annotations

import copy
import inspect
import json
import logging
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, TextIO

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
"""Absolute repository root used to locate the canonical CE-60 fixture."""

FIXTURE_ROOT: Final[Path] = PROJECT_ROOT / "validation" / "community-plugin-A-B-D"
"""Directory containing the canonical CE-60 validation fixture package."""

MCP_PROTOCOL_VERSION: Final[str] = "2024-11-05"
"""MCP protocol version returned during initialization."""

SERVER_NAME: Final[str] = "ce60-deterministic-fixture"
"""Stable server identifier reported to MCP clients."""

SERVER_VERSION: Final[str] = "1.0.0"
"""Stable fixture server version reported to MCP clients."""

EXPECTED_CORE_TOOL_COUNT: Final[int] = 18
"""The number of canonical CE-60 domain tools exposed by this fixture."""


class _UnknownToolError(ValueError):
    """Signal a tools/call request for a name outside the closed fixture set."""


class _StructuredFormatter(logging.Formatter):
    """Render log records as compact JSON without sensitive request data."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize a log record and its explicitly safe fields.

        Args:
            record: Standard-library log record to serialize.

        Returns:
            A JSON object containing the log level, event and safe fields.
        """
        fields = getattr(record, "structured", {})
        payload: dict[str, Any] = {
            "event": record.getMessage(),
            "level": record.levelname,
        }
        if isinstance(fields, Mapping):
            payload.update({str(key): value for key, value in fields.items()})
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _build_logger() -> logging.Logger:
    """Create the server logger with a stderr-only structured handler.

    Returns:
        A configured logger that cannot contaminate the stdout JSON-RPC stream.
    """
    logger = logging.getLogger(SERVER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_StructuredFormatter())
        logger.addHandler(handler)
    return logger


LOGGER: Final[logging.Logger] = _build_logger()
"""Server logger; stdout remains reserved for JSON-RPC responses."""


def _offline_corpus_fetch(key: str, url: str) -> str:
    """Read a cached corpus page or return its deterministic offline stub.

    The canonical corpus loader normally downloads a missing page. That behavior
    is appropriate for the live benchmark but is not allowed for this local MCP
    fixture. The fallback mirrors the canonical corpus fallback and deliberately
    does not write a cache file, so repeated calls remain side-effect free.

    Args:
        key: Canonical corpus page key.
        url: Canonical source URL recorded in the existing fixture.

    Returns:
        Cached page text or deterministic offline fixture text.
    """
    cache_dir = _CORPUS.CACHE_DIR
    cache_path = cache_dir / f"{key}.txt"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    return f"[offline stub for {key}] {url}\n" + ("AWS documentation content unavailable.\n" * 200)


def _configure_offline_corpus() -> None:
    """Prevent corpus reads from making network requests in this process."""
    _CORPUS._fetch = _offline_corpus_fetch


def _load_fixture_tools() -> tuple[Any, ...]:
    """Load the canonical CE-60 tools and enforce the expected fixture size.

    Returns:
        The ordered tuple of the eighteen canonical tools.

    Raises:
        RuntimeError: If the fixture cannot be imported or its tool count drifts.
    """
    try:
        tools = tuple(_CORE_TOOLS)
    except Exception as error:  # pragma: no cover - startup failure is environment-specific.
        LOGGER.error(
            "fixture_import_failed",
            extra={"structured": {"error_type": type(error).__name__}},
        )
        raise RuntimeError("the CE-60 tool fixture could not be imported") from error
    if len(tools) != EXPECTED_CORE_TOOL_COUNT:
        LOGGER.error(
            "fixture_tool_count_mismatch",
            extra={
                "structured": {
                    "expected": EXPECTED_CORE_TOOL_COUNT,
                    "actual": len(tools),
                }
            },
        )
        raise RuntimeError("the CE-60 tool fixture does not contain eighteen core tools")
    return tools


def _schema_from_tool_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a Strands tool schema wrapper to the MCP JSON Schema object.

    Args:
        spec: Public Strands tool specification.

    Returns:
        A deep copy of the JSON Schema accepted by the tool.
    """
    raw_schema = spec.get("inputSchema", {})
    if isinstance(raw_schema, Mapping) and "json" in raw_schema:
        raw_schema = raw_schema["json"]
    if not isinstance(raw_schema, Mapping):
        return {}
    return copy.deepcopy(dict(raw_schema))


def _tool_definition(tool: Any) -> tuple[dict[str, Any], Any]:
    """Build one MCP tool definition from a public Strands tool object.

    Args:
        tool: Decorated Strands tool exposing ``tool_spec`` and ``__call__``.

    Returns:
        A tuple containing the MCP definition and the callable tool object.

    Raises:
        TypeError: If the public tool specification is incomplete.
    """
    spec = getattr(tool, "tool_spec", None)
    if not isinstance(spec, Mapping):
        raise TypeError("a CE-60 tool did not expose a public tool specification")
    name = spec.get("name")
    description = spec.get("description", "")
    if not isinstance(name, str) or not name:
        raise RuntimeError("a CE-60 tool specification has no name")
    if not isinstance(description, str):
        description = str(description)
    definition = {
        "name": name,
        "description": description,
        "inputSchema": _schema_from_tool_spec(spec),
    }
    return definition, tool


# Importing the fixture after the path is known keeps this single-file server runnable from the
# repository root, from OpenCode's configured cwd, or by a subprocess launched from another cwd.
if str(FIXTURE_ROOT) not in sys.path:
    sys.path.insert(0, str(FIXTURE_ROOT))
try:
    from src import corpus as _CORPUS  # type: ignore[import-not-found]
    from src.tools import CORE_TOOLS as _CORE_TOOLS  # type: ignore[import-not-found]
except ImportError as error:  # pragma: no cover - exercised only by a broken local install.
    raise RuntimeError(
        "install the CE-60 fixture dependencies before starting the local MCP server"
    ) from error

_configure_offline_corpus()
_TOOL_PAIRS: Final[tuple[tuple[dict[str, Any], Any], ...]] = tuple(
    _tool_definition(tool) for tool in _load_fixture_tools()
)
_TOOL_DEFINITIONS: Final[tuple[dict[str, Any], ...]] = tuple(
    definition for definition, _ in _TOOL_PAIRS
)
_TOOL_REGISTRY: Final[dict[str, Any]] = {
    definition["name"]: tool for definition, tool in _TOOL_PAIRS
}


def _json_response(request_id: Any, result: Mapping[str, Any]) -> dict[str, Any]:
    """Build a successful JSON-RPC response envelope.

    Args:
        request_id: JSON-RPC request identifier.
        result: Protocol result object.

    Returns:
        A JSON-RPC 2.0 response dictionary.
    """
    return {"jsonrpc": "2.0", "id": request_id, "result": dict(result)}


def _json_error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    """Build a JSON-RPC error envelope without echoing request data.

    Args:
        request_id: JSON-RPC request identifier.
        code: JSON-RPC error code.
        message: Stable, non-sensitive error message.

    Returns:
        A JSON-RPC 2.0 error response dictionary.
    """
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def _tool_error(text: str) -> dict[str, Any]:
    """Build an MCP tool execution error result.

    Args:
        text: Stable non-sensitive error text.

    Returns:
        An MCP ``CallToolResult`` with ``isError`` set.
    """
    return {
        "content": [{"type": "text", "text": text}],
        "isError": True,
    }


def _tool_success(text: str) -> dict[str, Any]:
    """Build an MCP successful tool execution result.

    Args:
        text: Exact deterministic fixture output.

    Returns:
        An MCP ``CallToolResult`` containing the output as text.
    """
    return {
        "content": [{"type": "text", "text": text}],
        "isError": False,
    }


def _serialize_tool_output(output: Any) -> str:
    """Convert a fixture result to the text content required by MCP.

    Args:
        output: Value returned by a canonical fixture tool.

    Returns:
        UTF-8-safe deterministic text for the MCP content block.
    """
    if isinstance(output, str):
        return output
    return json.dumps(output, ensure_ascii=False, sort_keys=True)


class FixtureMcpServer:
    """Handle the deterministic CE-60 MCP request/notification subset."""

    def __init__(self) -> None:
        """Initialize an uninitialized MCP server state."""
        self._initialized = False

    def handle_message(self, message: Any) -> dict[str, Any] | None:
        """Handle one parsed JSON-RPC message.

        Notifications produce no response, as required by JSON-RPC. Unknown
        methods return a standard JSON-RPC method-not-found error. Handler
        failures are converted to non-sensitive protocol errors and logged by
        type only.

        Args:
            message: Parsed JSON value received from stdin.

        Returns:
            A JSON-RPC response dictionary, or ``None`` for notifications.
        """
        if not isinstance(message, Mapping):
            return _json_error(None, -32600, "Invalid Request")

        has_id = "id" in message
        request_id = message.get("id")
        method = message.get("method")
        params = message.get("params", {})
        if not isinstance(method, str):
            return _json_error(request_id if has_id else None, -32600, "Invalid Request")

        if not has_id:
            self._handle_notification(method, params)
            return None

        try:
            if method == "initialize":
                return _json_response(request_id, self._initialize())
            if method == "tools/list":
                return _json_response(request_id, self._list_tools())
            if method == "tools/call":
                return _json_response(request_id, self._call_tool(params))
            if method == "ping":
                return _json_response(request_id, {})
            return _json_error(request_id, -32601, "Method not found")
        except _UnknownToolError:
            return _json_error(request_id, -32602, "Unknown tool")
        except (TypeError, ValueError):
            return _json_error(request_id, -32602, "Invalid params")
        except Exception as error:  # pragma: no cover - defensive protocol boundary.  # noqa: BLE001
            LOGGER.error(
                "request_failed",
                extra={"structured": {"method": method, "error_type": type(error).__name__}},
            )
            return _json_error(request_id, -32603, "Internal error")

    def _handle_notification(self, method: str, params: Any) -> None:
        """Handle a JSON-RPC notification without writing a response.

        Args:
            method: Notification method name.
            params: Notification parameters, which are intentionally not logged.
        """
        if method == "notifications/initialized":
            self._initialized = True
            LOGGER.info("mcp_initialized", extra={"structured": {"server": SERVER_NAME}})

    def _initialize(self) -> dict[str, Any]:
        """Return the MCP initialization result and mark the session initialized.

        Returns:
            MCP initialization result with tools capability metadata.
        """
        self._initialized = True
        return {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        }

    def _list_tools(self) -> dict[str, Any]:
        """Return all eighteen canonical tool definitions in fixture order.

        Returns:
            MCP ``ListToolsResult`` containing deep-copied definitions.
        """
        return {"tools": copy.deepcopy(list(_TOOL_DEFINITIONS))}

    def _call_tool(self, params: Any) -> dict[str, Any]:
        """Invoke one canonical tool with JSON arguments.

        Args:
            params: MCP ``tools/call`` parameters.

        Returns:
            MCP ``CallToolResult`` for the requested tool.

        Raises:
            TypeError: If the request shape is invalid.
        _UnknownToolError: If the requested name is outside the fixture set.
        """
        if not isinstance(params, Mapping):
            raise TypeError("tool call parameters must be an object")
        name = params.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("tool name is required")
        if name not in _TOOL_REGISTRY:
            LOGGER.info("tool_refused", extra={"structured": {"reason": "unknown_tool"}})
            raise _UnknownToolError from None

        arguments = params.get("arguments", {})
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, Mapping):
            raise TypeError("tool arguments must be an object")

        tool = _TOOL_REGISTRY[name]
        try:
            output = tool(**dict(arguments))
            if inspect.isawaitable(output):
                raise TypeError("asynchronous tools are not supported by this fixture")
        except Exception as error:  # noqa: BLE001 - fixture failures become safe MCP tool errors.
            LOGGER.error(
                "tool_execution_failed",
                extra={
                    "structured": {
                        "tool": name,
                        "error_type": type(error).__name__,
                    }
                },
            )
            return _tool_error("Tool execution failed")

        text = _serialize_tool_output(output)
        LOGGER.info(
            "tool_completed",
            extra={"structured": {"tool": name, "result_chars": len(text)}},
        )
        return _tool_success(text)


def _write_response(output: TextIO, response: Mapping[str, Any]) -> None:
    """Write one compact JSON-RPC response and flush it immediately.

    Args:
        output: Text stream connected to the MCP client's stdout.
        response: JSON-RPC response object.
    """
    output.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")))
    output.write("\n")
    output.flush()


def run_stdio(input_stream: TextIO = sys.stdin, output_stream: TextIO = sys.stdout) -> int:
    """Serve newline-delimited JSON-RPC messages until stdin closes.

    Args:
        input_stream: Input stream containing one JSON-RPC value per line.
        output_stream: Output stream for JSON-RPC responses.

    Returns:
        Process exit code, zero for a clean end-of-stream.
    """
    server = FixtureMcpServer()
    for raw_line in input_stream:
        line = raw_line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            _write_response(output_stream, _json_error(None, -32700, "Parse error"))
            continue
        response = server.handle_message(message)
        if response is not None:
            _write_response(output_stream, response)
    return 0


def main() -> int:
    """Run the fixture MCP server over the process stdio streams.

    Returns:
        Process exit code.
    """
    return run_stdio()


if __name__ == "__main__":
    raise SystemExit(main())
