"""Contract tests for the deterministic OpenCode MCP fixture server.

The tests exercise the server as a local JSONL subprocess. They deliberately avoid
OpenCode, AWS credentials, network access, and imports from the server implementation
so that the public MCP stream remains the contract under test.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, cast

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
SERVER_COMMAND: Final[tuple[str, str]] = (
    "python3.13",
    "validation/opencode_mcp_server.py",
)
EXPECTED_CORE_TOOL_NAMES: Final[tuple[str, ...]] = (
    "list_accounts",
    "list_investment_positions",
    "list_investment_transactions",
    "get_portfolio_allocation",
    "project_yield",
    "get_connector_status",
    "read_connector_logs",
    "force_connector_sync",
    "rotate_connector_credentials",
    "open_support_case",
    "read_aws_documentation",
    "search_aws_documentation",
    "describe_lambda_function",
    "list_s3_objects",
    "query_dynamodb_table",
    "get_cloudwatch_metrics",
    "describe_iam_role",
    "estimate_aws_cost",
)
LARGE_RESULT_BYTES: Final[int] = 40_000
LOGGER = logging.getLogger(__name__)
JsonObject = dict[str, Any]


def _json_line(message: Mapping[str, Any]) -> bytes:
    """Encode one JSON-RPC message as a UTF-8 JSONL line.

    Args:
        message: JSON-RPC request or notification to send to the fixture.

    Returns:
        The compact UTF-8 encoded message followed by a newline.
    """
    return json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"


def _run_fixture() -> tuple[list[JsonObject], bytes, bytes]:
    """Run the fixture through the complete MCP request sequence.

    Returns:
        Parsed JSON-RPC responses, raw stdout, and raw stderr from the subprocess.

    Raises:
        AssertionError: If the process exits unsuccessfully or emits invalid stdout.
        subprocess.TimeoutExpired: If the local fixture does not terminate promptly.
    """
    messages: tuple[Mapping[str, Any], ...] = (
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "fixture-contract-test", "version": "1.0.0"},
            },
        },
        {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        },
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "list_accounts", "arguments": {}},
        },
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "list_accounts", "arguments": {}},
        },
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "list_investment_transactions",
                "arguments": {"account_id": "0001/12345-6", "days": 120},
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "list_investment_transactions",
                "arguments": {"account_id": "0001/12345-6", "days": 120},
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {"name": "not_a_core_tool", "arguments": {}},
        },
    )
    request_bytes = b"".join(_json_line(message) for message in messages)

    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("AWS_")
    }
    environment["AWS_EC2_METADATA_DISABLED"] = "true"
    completed = subprocess.run(
        list(SERVER_COMMAND),
        cwd=PROJECT_ROOT,
        env=environment,
        input=request_bytes,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")

    stdout_lines = completed.stdout.splitlines()
    assert stdout_lines, "the fixture did not emit any JSON-RPC response"
    assert completed.stdout.endswith(b"\n"), "stdout must be newline-delimited JSON"
    assert all(line.strip() for line in stdout_lines), "stdout contains a blank control line"

    responses = [_decode_response(line) for line in stdout_lines]
    LOGGER.debug(
        "mcp_fixture_exchange_complete",
        extra={
            "structured": {
                "response_count": len(responses),
                "stdout_bytes": len(completed.stdout),
                "stderr_bytes": len(completed.stderr),
            }
        },
    )
    return responses, completed.stdout, completed.stderr


def _decode_response(line: bytes) -> JsonObject:
    """Decode and validate one response line without accepting control messages.

    Args:
        line: Raw stdout line emitted by the fixture server.

    Returns:
        The decoded JSON-RPC response object.
    """
    assert line.startswith(b"{"), f"stdout contains a non-JSON line: {line[:80]!r}"
    decoded = json.loads(line.decode("utf-8"))
    assert isinstance(decoded, dict), "stdout JSON values must be response objects"
    response = cast(JsonObject, decoded)
    assert response.get("jsonrpc") == "2.0"
    assert "id" in response
    assert "method" not in response
    assert "params" not in response
    assert ("result" in response) ^ ("error" in response)
    return response


def _response_by_id(responses: list[JsonObject]) -> dict[int, JsonObject]:
    """Index responses by their integer JSON-RPC identifiers.

    Args:
        responses: Decoded responses from one fixture process.

    Returns:
        Responses keyed by their request identifiers.
    """
    indexed: dict[int, JsonObject] = {}
    for response in responses:
        request_id = response["id"]
        assert isinstance(request_id, int) and not isinstance(request_id, bool)
        indexed[request_id] = response
    return indexed


def _tool_text(response: JsonObject) -> str:
    """Extract a successful MCP text content block from a response.

    Args:
        response: JSON-RPC response for a ``tools/call`` request.

    Returns:
        The exact text returned by the fixture tool.
    """
    result = response.get("result")
    assert isinstance(result, dict)
    assert result.get("isError") is False
    content = result.get("content")
    assert isinstance(content, list) and len(content) == 1
    block = content[0]
    assert isinstance(block, dict) and block.get("type") == "text"
    text = block.get("text")
    assert isinstance(text, str)
    return text


def test_mcp_fixture_stdio_contract() -> None:
    """Validate handshake, tool catalogue, deterministic outputs, and refusal behavior."""
    responses, stdout, _stderr = _run_fixture()
    response_map = _response_by_id(responses)
    assert set(response_map) == set(range(1, 8))
    assert len(responses) == 7, "notifications/initialized must not produce a response"

    initialize_result = response_map[1]["result"]
    assert isinstance(initialize_result, dict)
    assert initialize_result["protocolVersion"] == "2024-11-05"
    assert initialize_result["serverInfo"] == {
        "name": "ce60-deterministic-fixture",
        "version": "1.0.0",
    }

    list_result = response_map[2]["result"]
    assert isinstance(list_result, dict)
    tools = list_result.get("tools")
    assert isinstance(tools, list)
    tool_names = tuple(
        tool["name"]
        for tool in tools
        if isinstance(tool, dict) and isinstance(tool.get("name"), str)
    )
    assert len(tools) == len(EXPECTED_CORE_TOOL_NAMES) == 18
    assert tool_names == EXPECTED_CORE_TOOL_NAMES
    assert len(set(tool_names)) == 18

    accounts_first = _tool_text(response_map[3])
    accounts_second = _tool_text(response_map[4])
    assert accounts_first.encode("utf-8") == accounts_second.encode("utf-8")
    accounts = json.loads(accounts_first)
    assert isinstance(accounts, dict)
    assert accounts["include_closed"] is False
    assert [account["id"] for account in accounts["accounts"]] == [
        "0001/12345-6",
        "0341/98765-4",
        "0260/55512-0",
        "0077/31415-9",
        "0033/27182-8",
    ]

    transactions_first = _tool_text(response_map[5])
    transactions_second = _tool_text(response_map[6])
    first_transaction_bytes = transactions_first.encode("utf-8")
    second_transaction_bytes = transactions_second.encode("utf-8")
    assert len(first_transaction_bytes) > LARGE_RESULT_BYTES
    assert first_transaction_bytes == second_transaction_bytes
    assert first_transaction_bytes.startswith(b"filters: category=None min_amount=None\n")
    assert b"STATEMENT \xe2\x80\x94 account 0001/12345-6" in first_transaction_bytes

    unknown_error = response_map[7].get("error")
    assert unknown_error == {"code": -32602, "message": "Unknown tool"}

    assert b"notifications/initialized" not in stdout
    assert b"mcp_initialized" not in stdout
    assert b"tool_completed" not in stdout
    assert b"tool_refused" not in stdout
