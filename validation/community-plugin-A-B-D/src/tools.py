"""Mocked tool suite for the validation harness.

Two things about this suite are deliberate.

**The schemas are fat.** Every tool carries a multi-paragraph description and several
documented parameters, because that is what makes ``tool_specs`` expensive in a real
agent and it is the cost Progressive Tool Disclosure exists to remove. A suite of
thin one-line tools would make the disclosure strategy look pointless for reasons that
have nothing to do with the strategy.

**The payloads are real and oversized.** The documentation tools return cached AWS
prose in the 20k-120k character range, well past ``max_result_tokens``, so the
offloader always engages and the preview strategy is always the thing under test.
AWS doc pages also open with navigation boilerplate, which is precisely where a
positional prefix preview spends its budget.

No tool performs a mutating action. ``open_support_case`` and friends return
acknowledgements, so the agent can be pushed through long tool-calling sequences with
no side effects to clean up.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Any

from strands import tool

from . import corpus
from .config import TARGET_SCHEMA_TOKENS

# --- Synthetic financial fixtures ---------------------------------------------------
#
# Mirrors the measured session: an investment question, a connector that
# breaks, and a return to the original subject. Values carry thousand and decimal
# separators on purpose — that is what the offloader classifies as protected content
# and refuses to paraphrase.

_ACCOUNTS = [
    {"id": "0001/12345-6", "institution": "FinBank Invest", "type": "investment", "balance": "R$ 47.832,15"},
    {"id": "0341/98765-4", "institution": "TestBank", "type": "checking", "balance": "R$ 12.409,88"},
    {"id": "0260/55512-0", "institution": "NeoBank", "type": "checking", "balance": "R$ 3.187,42"},
    {"id": "0077/31415-9", "institution": "MidBank", "type": "savings", "balance": "R$ 21.650,00"},
    {"id": "0033/27182-8", "institution": "SampleBank", "type": "investment", "balance": "R$ 88.204,73"},
]

_POSITIONS = [
    {"account": "0001/12345-6", "instrument": "CDB FinBank 2028", "quantity": "12", "unit": "R$ 1.204,50", "total": "R$ 14.454,00"},
    {"account": "0001/12345-6", "instrument": "Tesouro IPCA+ 2029", "quantity": "8", "unit": "R$ 3.017,89", "total": "R$ 24.143,12"},
    {"account": "0001/12345-6", "instrument": "Fundo FinBank Absoluto", "quantity": "1.204,338", "unit": "R$ 7,66", "total": "R$ 9.235,03"},
    {"account": "0033/27182-8", "instrument": "LCA SampleBank 2027", "quantity": "40", "unit": "R$ 1.102,44", "total": "R$ 44.097,60"},
    {"account": "0033/27182-8", "instrument": "Tesouro Selic 2027", "quantity": "15", "unit": "R$ 2.940,47", "total": "R$ 44.107,05"},
]


def _synthetic_statement(account_id: str, days: int) -> str:
    """Build a long, line-oriented statement.

    Line-oriented because the offloader chunks on line boundaries and hands the model
    1-indexed line numbers it can feed straight back into ``line_range``.
    """
    start = date(2026, 8, 26) - timedelta(days=days)
    lines = [
        f"STATEMENT — account {account_id}",
        f"period: {start.isoformat()} to 2026-08-26",
        "date,description,category,amount,balance_after",
    ]
    descriptions = [
        "PIX RECEBIDO",
        "CDB RESGATE PARCIAL",
        "TARIFA MANUTENCAO",
        "APLICACAO TESOURO IPCA+",
        "DEBITO CARTAO",
        "RENDIMENTO FUNDO ABSOLUTO",
    ]
    categories = ["transfer", "investment", "fee", "investment", "purchase", "yield"]

    balance = 47832.15
    for offset in range(days * 6):
        day = start + timedelta(days=offset // 6)
        amount = ((offset * 37) % 900) + 12.35
        if offset % 3 == 0:
            amount = -amount
        balance += amount
        lines.append(
            ",".join(
                [
                    day.isoformat(),
                    descriptions[offset % 6],
                    categories[offset % 6],
                    _brl(amount),
                    _brl(balance),
                ]
            )
        )
    return "\n".join(lines)


def _brl(value: float) -> str:
    """Format a number in Brazilian convention: thousands with '.', decimals with ','.

    Each field is formatted on its own rather than by rewriting the finished line. Swapping
    separators across a whole CSV row turns its commas into dots and corrupts the descriptions
    ("PIX" becomes "PI."), which makes the statement unparseable and its numbers
    untrustworthy — exactly the property the offloader's protected-content guard exists to
    preserve.
    """
    return f"R$ {value:,.2f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")


# --- Investment domain --------------------------------------------------------------


@tool
def list_accounts(
    institution: str | None = None,
    account_type: str | None = None,
    include_closed: bool = False,
) -> str:
    """List every financial account linked to the current customer profile.

    Returns one row per account with its institution, product type and last known
    consolidated balance. Balances reflect the most recent successful synchronization
    with the institution and may lag the institution's own statement by up to one
    business day when a connector is degraded.

    Args:
        institution: Restrict to a single institution by display name, for example
            "FinBank Invest" or "TestBank". Matching is case-insensitive and partial.
            Omit to list accounts across every linked institution.
        account_type: Restrict to one product family. Accepted values are "checking",
            "savings" and "investment". Omit to include every product family.
        include_closed: When true, accounts closed within the retention window are
            listed with a closure date. Defaults to false, which lists active accounts
            only.
    """
    rows = _ACCOUNTS
    if institution:
        rows = [r for r in rows if institution.lower() in r["institution"].lower()]
    if account_type:
        rows = [r for r in rows if r["type"] == account_type]
    return json.dumps({"accounts": rows, "include_closed": include_closed}, ensure_ascii=False, indent=2)


@tool
def list_investment_positions(
    account_id: str,
    as_of_date: str | None = None,
    instrument_class: str | None = None,
    include_accrued_yield: bool = True,
) -> str:
    """List the consolidated investment positions held in one account.

    Each position reports the instrument name, held quantity, unit price used for the
    valuation and the resulting total. Unit prices for fixed-income instruments are the
    institution's own marked price, not a market mid, so they will not tie out against
    a public quote feed.

    Args:
        account_id: Account identifier in the institution's own format, for example
            "0001/12345-6". Required — positions are never aggregated across accounts
            because valuation dates differ per institution.
        as_of_date: Valuation date in ISO-8601 form. Defaults to the latest
            consolidated position, which is what the customer sees in the app.
        instrument_class: Restrict to one class. Accepted values are "fixed_income",
            "equity", "fund" and "treasury". Omit to include every class.
        include_accrued_yield: When true, each fixed-income position carries the yield
            accrued since acquisition as a separate field. Defaults to true.
    """
    rows = [p for p in _POSITIONS if p["account"] == account_id]
    return json.dumps(
        {
            "account": account_id,
            "as_of": as_of_date or "2026-08-26",
            "instrument_class": instrument_class,
            "accrued_yield_included": include_accrued_yield,
            "positions": rows,
        },
        ensure_ascii=False,
        indent=2,
    )


@tool
def list_investment_transactions(
    account_id: str,
    days: int = 30,
    category: str | None = None,
    min_amount: float | None = None,
) -> str:
    """Export the full transaction ledger for one account as a CSV statement.

    The result is a line-oriented CSV with a header row, one row per movement, and a
    running balance column. Long periods produce very large results: ninety days of a
    moderately active account is on the order of one hundred thousand characters.

    Args:
        account_id: Account identifier in the institution's own format.
        days: Size of the lookback window in days, counted back from today. Larger
            windows produce proportionally larger results.
        category: Restrict to one movement category. Accepted values are "transfer",
            "investment", "fee", "purchase" and "yield". Omit for every category.
        min_amount: Drop movements whose absolute value is below this threshold, in the
            account's own currency. Omit to include every movement.
    """
    body = _synthetic_statement(account_id, max(1, min(days, 120)))
    return (
        f"filters: category={category} min_amount={min_amount}\n"
        f"{body}"
    )


@tool
def get_portfolio_allocation(account_id: str, group_by: str = "instrument_class") -> str:
    """Summarize how one account's invested capital is distributed.

    Args:
        account_id: Account identifier in the institution's own format.
        group_by: Dimension to aggregate on. Accepted values are "instrument_class",
            "issuer", "maturity_bucket" and "index". Defaults to "instrument_class".
    """
    return json.dumps(
        {
            "account": account_id,
            "group_by": group_by,
            "allocation": _allocation_of(account_id),
        },
        ensure_ascii=False,
        indent=2,
    )


_ASSET_CLASS_PREFIXES = (("Tesouro", "treasury"), ("Fundo", "fund"), ("CDB", "fixed_income"), ("LCA", "fixed_income"))


def _parse_brl(value: str) -> float:
    """Parse "R$ 14.454,00" into 14454.0."""
    return float(value.replace("R$", "").strip().replace(".", "").replace(",", "."))


def _format_brl(value: float) -> str:
    """Format 14454.0 as "R$ 14.454,00"."""
    return "R$ " + f"{value:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")


def _allocation_of(account_id: str) -> list[dict[str, str]]:
    """Group the account's positions by asset class, so allocation and positions never disagree."""
    buckets: dict[str, float] = {}
    for position in _POSITIONS:
        if position["account"] != account_id:
            continue
        bucket = next(
            (name for prefix, name in _ASSET_CLASS_PREFIXES if position["instrument"].startswith(prefix)),
            "other",
        )
        buckets[bucket] = buckets.get(bucket, 0.0) + _parse_brl(position["total"])

    total = sum(buckets.values())
    return [
        {
            "bucket": bucket,
            "share": f"{value / total * 100:.1f}".replace(".", ",") + "%",
            "value": _format_brl(value),
        }
        for bucket, value in sorted(buckets.items(), key=lambda item: -item[1])
    ]


@tool
def project_yield(account_id: str, horizon_months: int = 12, scenario: str = "base") -> str:
    """Project the account's expected yield over a horizon under a rate scenario.

    Args:
        account_id: Account identifier in the institution's own format.
        horizon_months: Projection horizon in months, between 1 and 120.
        scenario: Rate scenario to apply. Accepted values are "base", "hawkish" and
            "dovish". Defaults to "base", which tracks the current forward curve.
    """
    return json.dumps(
        {
            "account": account_id,
            "horizon_months": horizon_months,
            "scenario": scenario,
            "projected_value": "R$ 51.204,77",
            "projected_yield": "7,06%",
        },
        ensure_ascii=False,
        indent=2,
    )


# --- Connector domain: the branch subject -------------------------------------------


@tool
def get_connector_status(institution: str, verbose: bool = False) -> str:
    """Report the health of the data connector for one institution.

    Args:
        institution: Institution display name, for example "FinBank Invest".
        verbose: When true, includes the last ten synchronization attempts with their
            individual outcomes and latencies. Defaults to false.
    """
    return json.dumps(
        {
            "institution": institution,
            "state": "DEGRADED",
            "last_success": "2026-08-24T03:11:42Z",
            "last_error": "MFA_CHALLENGE_TIMEOUT",
            "consecutive_failures": 7,
            "verbose": verbose,
        },
        ensure_ascii=False,
        indent=2,
    )


@tool
def read_connector_logs(institution: str, lines: int = 500, level: str = "ERROR") -> str:
    """Read the raw connector log tail for one institution.

    Log tails are large and repetitive by nature: a degraded connector retrying every
    thirty seconds produces thousands of near-identical lines, of which only a handful
    carry the actual failure.

    Args:
        institution: Institution display name.
        lines: Number of trailing log lines to return, between 1 and 5000.
        level: Minimum severity to include. Accepted values are "DEBUG", "INFO",
            "WARN" and "ERROR". Defaults to "ERROR".
    """
    count = max(1, min(lines, 5000))
    out = [f"connector={institution} level={level} lines={count}"]
    for index in range(count):
        stamp = f"2026-08-2{(index % 5) + 1}T0{index % 10}:{index % 60:02d}:11Z"
        if index % 97 == 0:
            out.append(f"{stamp} ERROR MFA_CHALLENGE_TIMEOUT session=sess-{index:05d} retry_in=30s")
        elif index % 31 == 0:
            out.append(f"{stamp} ERROR TLS_HANDSHAKE_RESET peer=api.{institution.lower().replace(' ', '')}.com")
        else:
            out.append(f"{stamp} WARN  retry attempt={index % 7} backoff=30s pool=connector-worker-{index % 4}")
    return "\n".join(out)


@tool
def force_connector_sync(institution: str, full_refresh: bool = False, timeout_seconds: int = 120) -> str:
    """Trigger an out-of-band synchronization for one institution's connector.

    Args:
        institution: Institution display name.
        full_refresh: When true, discards cached cursors and re-reads the entire
            available history. Slower and heavier on the institution's API.
        timeout_seconds: How long to wait for the institution to respond, between 10
            and 600 seconds.
    """
    return json.dumps(
        {
            "institution": institution,
            "accepted": True,
            "full_refresh": full_refresh,
            "timeout_seconds": timeout_seconds,
            "job_id": "sync-7f3a9c21",
            "note": "queued behind 2 pending jobs",
        },
        ensure_ascii=False,
        indent=2,
    )


@tool
def rotate_connector_credentials(institution: str, notify_customer: bool = True) -> str:
    """Rotate the stored credentials used by one institution's connector.

    Args:
        institution: Institution display name.
        notify_customer: When true, sends the customer a notification explaining that
            re-authentication is required. Defaults to true.
    """
    return json.dumps(
        {"institution": institution, "rotated": True, "notify_customer": notify_customer},
        ensure_ascii=False,
        indent=2,
    )


@tool
def open_support_case(institution: str, summary: str, severity: str = "medium") -> str:
    """Open a support case with the platform team about a connector.

    Args:
        institution: Institution display name.
        summary: One-paragraph description of the observed failure.
        severity: Case severity. Accepted values are "low", "medium", "high" and
            "critical". Defaults to "medium".
    """
    return json.dumps(
        {"case_id": "CASE-20260826-0042", "institution": institution, "severity": severity, "summary": summary},
        ensure_ascii=False,
        indent=2,
    )


# --- AWS documentation domain: the oversized payloads -------------------------------


@tool
def read_aws_documentation(
    topic: str,
    include_related: bool = True,
    min_chars: int = 60_000,
) -> str:
    """Read the full AWS documentation for one topic, as plain text.

    Results are the complete page body including navigation chrome, breadcrumbs and
    footer boilerplate, exactly as the documentation site serves it. Pages routinely
    exceed one hundred thousand characters when related pages are included.

    Args:
        topic: Topic key to read. Accepted values are "lambda", "s3", "dynamodb",
            "vpc", "iam", "bedrock", "rds", "eks" and "cloudwatch".
        include_related: When true, appends the related pages for the same service.
            Defaults to true, which roughly triples the result size.
        min_chars: Lower bound on the returned size in characters. The page sequence is
            repeated as an appendix until the bound is met, which is how a small page
            is made representative of a large one.
    """
    groups = {
        "lambda": ("lambda_invocation", "lambda_foundation"),
        "s3": ("s3_naming", "s3_security"),
        "dynamodb": ("dynamodb_capacity", "dynamodb_partition"),
        "vpc": ("vpc_subnets",),
        "iam": ("iam_best_practices",),
        "bedrock": ("bedrock_inference",),
        "rds": ("rds_backups",),
        "eks": ("eks_networking",),
        "cloudwatch": ("cloudwatch_alarms",),
    }
    keys = groups.get(topic.lower().strip(), ("lambda_invocation",))
    if not include_related:
        keys = keys[:1]
    return corpus.concatenated(*keys, min_chars=max(0, min(min_chars, 400_000)))


@tool
def search_aws_documentation(query: str, max_results: int = 10, service: str | None = None) -> str:
    """Search the AWS documentation index and return matching page summaries.

    Args:
        query: Free-text search phrase.
        max_results: Maximum number of pages to return, between 1 and 50.
        service: Restrict to one service by short name, for example "lambda" or "s3".
            Omit to search across every service.
    """
    catalog = corpus.load_all()
    hits = []
    needles = [term for term in query.lower().split() if len(term) > 3]
    for key, text in catalog.items():
        if service and service.lower() not in key:
            continue
        score = sum(text.lower().count(term) for term in needles)
        hits.append({"page": key, "term_hits": score, "chars": len(text)})
    hits.sort(key=lambda item: -item["term_hits"])
    return json.dumps({"query": query, "results": hits[: max(1, min(max_results, 50))]}, indent=2)


@tool
def describe_lambda_function(function_name: str, qualifier: str | None = None, include_env: bool = False) -> str:
    """Describe the configuration of one Lambda function.

    Args:
        function_name: Function name or full ARN.
        qualifier: Version number or alias name. Omit for $LATEST.
        include_env: When true, includes environment variable keys. Values are always
            redacted. Defaults to false.
    """
    return json.dumps(
        {
            "FunctionName": function_name,
            "Qualifier": qualifier or "$LATEST",
            "Runtime": "python3.12",
            "MemorySize": 1024,
            "Timeout": 30,
            "EnvironmentKeys": ["CONNECTOR_POOL", "LOG_LEVEL"] if include_env else [],
        },
        indent=2,
    )


@tool
def list_s3_objects(bucket: str, prefix: str | None = None, max_keys: int = 100) -> str:
    """List objects in one S3 bucket.

    Args:
        bucket: Bucket name.
        prefix: Key prefix to restrict the listing. Omit to list from the root.
        max_keys: Maximum number of keys to return, between 1 and 1000.
    """
    keys = [f"{prefix or 'statements/'}2026-08-{day:02d}.csv" for day in range(1, min(max_keys, 28) + 1)]
    return json.dumps({"Bucket": bucket, "KeyCount": len(keys), "Keys": keys}, indent=2)


@tool
def query_dynamodb_table(
    table_name: str,
    partition_key: str,
    sort_key_prefix: str | None = None,
    limit: int = 25,
    consistent_read: bool = False,
) -> str:
    """Query one DynamoDB table by partition key.

    Args:
        table_name: Table name.
        partition_key: Partition key value to match exactly.
        sort_key_prefix: Sort key prefix to narrow the result set. Omit to return every
            item under the partition key.
        limit: Maximum number of items to return, between 1 and 1000.
        consistent_read: When true, uses a strongly consistent read at twice the
            capacity cost. Defaults to false.
    """
    items = [
        {"pk": partition_key, "sk": f"{sort_key_prefix or 'evt#'}{index:04d}", "status": "ok"}
        for index in range(min(limit, 50))
    ]
    return json.dumps({"Table": table_name, "ConsistentRead": consistent_read, "Items": items}, indent=2)


@tool
def get_cloudwatch_metrics(namespace: str, metric_name: str, period_seconds: int = 300, statistic: str = "Average") -> str:
    """Fetch one CloudWatch metric series.

    Args:
        namespace: Metric namespace, for example "AWS/Lambda".
        metric_name: Metric name, for example "Duration".
        period_seconds: Aggregation period in seconds. Must be a multiple of 60.
        statistic: Aggregation statistic. Accepted values are "Average", "Sum",
            "Minimum", "Maximum" and "SampleCount".
    """
    points = [{"t": f"2026-08-26T{hour:02d}:00:00Z", "v": round(120.5 + hour * 3.7, 2)} for hour in range(24)]
    return json.dumps(
        {"Namespace": namespace, "MetricName": metric_name, "Period": period_seconds, "Statistic": statistic, "Datapoints": points},
        indent=2,
    )


@tool
def describe_iam_role(role_name: str, include_inline_policies: bool = True) -> str:
    """Describe one IAM role and the policies attached to it.

    Args:
        role_name: Role name, without path.
        include_inline_policies: When true, includes inline policy documents alongside
            attached managed policy ARNs. Defaults to true.
    """
    return json.dumps(
        {
            "RoleName": role_name,
            "AttachedPolicies": ["arn:aws:iam::aws:policy/ReadOnlyAccess"],
            "InlinePolicies": ["connector-secrets-read"] if include_inline_policies else [],
        },
        indent=2,
    )


@tool
def estimate_aws_cost(service: str, region: str = "us-east-1", monthly_units: int = 1000) -> str:
    """Estimate the monthly on-demand cost of one service at a given usage level.

    Args:
        service: Service short name, for example "lambda" or "dynamodb".
        region: AWS region code. Defaults to "us-east-1".
        monthly_units: Billable units per month, in the service's own unit.
    """
    return json.dumps({"service": service, "region": region, "monthly_units": monthly_units, "estimate_usd": round(monthly_units * 0.0000167, 4)}, indent=2)


# --- Filler tools: the schema tax the disclosure strategy removes -------------------
#
# A real agent accumulates dozens of tools it rarely calls, and every one of them is
# billed on every model call. These reproduce that tax with the same shape as the tools
# above: a paragraph of description and several documented parameters.


def _make_filler(name: str, subject: str, verb: str) -> Any:
    """Build one filler tool with an MCP-scale schema footprint.

    Eleven documented parameters rather than five. That is not padding for its own sake:
    it is the shape an MCP server wrapping a cloud API actually exposes, and it is what
    makes the schema budget per call land near the 63,000 tokens measured on the real
    session. A suite of thin tools would understate the cost Progressive Tool Disclosure
    removes and make the strategy look less useful for reasons that are the harness's
    fault rather than the strategy's.
    """

    def implementation(
        target: str,
        region: str = "us-east-1",
        dry_run: bool = False,
        max_items: int = 50,
        tags_filter: str | None = None,
        next_token: str | None = None,
        include_deleted: bool = False,
        sort_by: str = "name",
        sort_order: str = "asc",
        output_format: str = "json",
        propagate_tags: bool = False,
    ) -> str:
        return json.dumps(
            {
                "operation": name,
                "target": target,
                "region": region,
                "dry_run": dry_run,
                "max_items": max_items,
                "tags_filter": tags_filter,
                "next_token": next_token,
                "include_deleted": include_deleted,
                "sort_by": sort_by,
                "sort_order": sort_order,
                "output_format": output_format,
                "propagate_tags": propagate_tags,
                "result": f"{verb} {subject} completed",
            },
            indent=2,
        )

    implementation.__name__ = name
    implementation.__doc__ = f"""{verb.capitalize()} {subject} in the current account.

    This operation is part of the {subject} management surface. It resolves the target,
    validates that the caller is entitled to act on it, and applies the change. Results
    are eventually consistent: a subsequent read may not reflect the change for up to
    thirty seconds, and a read issued against a different availability zone may lag
    further.

    Throttling applies per account and per region. When the request is throttled the
    operation returns a retryable error and the caller is expected to back off
    exponentially with jitter rather than retrying immediately.

    Args:
        target: Identifier of the {subject} to act on. Accepts a bare name or a full
            ARN. Names are resolved within the current account and region. When a bare
            name matches more than one resource the call fails rather than guessing.
        region: AWS region code in which to resolve the target, for example "us-east-1"
            or "sa-east-1". Defaults to "us-east-1".
        dry_run: When true, validates the request and reports what would change without
            applying it. Useful for confirming entitlements before a mutating call.
            Defaults to false.
        max_items: Maximum number of {subject} entries to act on in one call, between 1
            and 500. Requests above the ceiling are clamped rather than rejected.
            Defaults to 50.
        tags_filter: Restrict the operation to resources carrying this tag, in
            "Key=Value" form. Multiple pairs may be comma-separated, in which case a
            resource must carry every pair to match. Omit to act on every resource.
        next_token: Continuation token from a previous truncated response. Omit for the
            first page. Tokens expire after fifteen minutes.
        include_deleted: When true, includes resources deleted within the retention
            window, each annotated with its deletion timestamp. Defaults to false.
        sort_by: Field to order results by. Accepted values are "name", "created_at",
            "updated_at" and "status". Defaults to "name".
        sort_order: Direction of the ordering. Accepted values are "asc" and "desc".
            Defaults to "asc".
        output_format: Shape of the returned payload. Accepted values are "json" for the
            full structure and "summary" for one line per resource. Defaults to "json".
        propagate_tags: When true, applies the operation's tag changes to resources
            dependent on the target. Has no effect on read-only operations. Defaults to
            false.
    """
    return tool(implementation)


_FILLER_SPECS = [
    ("list_ecs_services", "ECS services", "list"),
    ("describe_ecs_task_definition", "ECS task definitions", "describe"),
    ("list_step_functions", "Step Functions state machines", "list"),
    ("describe_sqs_queue", "SQS queues", "describe"),
    ("list_sns_subscriptions", "SNS subscriptions", "list"),
    ("describe_kinesis_stream", "Kinesis streams", "describe"),
    ("list_glue_jobs", "Glue jobs", "list"),
    ("describe_athena_workgroup", "Athena workgroups", "describe"),
    ("list_secrets", "Secrets Manager secrets", "list"),
    ("describe_kms_key", "KMS keys", "describe"),
    ("list_eventbridge_rules", "EventBridge rules", "list"),
    ("describe_apigateway_stage", "API Gateway stages", "describe"),
    ("list_codebuild_projects", "CodeBuild projects", "list"),
    ("describe_ecr_repository", "ECR repositories", "describe"),
    ("list_ssm_parameters", "SSM parameters", "list"),
    ("describe_elasticache_cluster", "ElastiCache clusters", "describe"),
    ("list_route53_records", "Route 53 records", "list"),
    ("describe_cloudfront_distribution", "CloudFront distributions", "describe"),
    ("list_waf_rules", "WAF rules", "list"),
    ("describe_efs_filesystem", "EFS file systems", "describe"),
]

_FILLER_DOMAINS = [
    "primary", "replica", "staging", "canary", "batch", "edge", "archive", "audit",
    "sandbox", "reporting", "ingest", "egress",
]
"""Suffixes used to grow the suite.

A real account has the same operation across many named environments, so widening this way
keeps the schemas plausible instead of inventing services that do not exist. It also gives
the lexical index genuinely similar tool names to discriminate between, which is the
harder and more realistic search problem.
"""


def _generate_filler_tools(target_schema_tokens: int) -> list[Any]:
    """Generate filler tools until their combined schema meets the token budget.

    Sized against a budget rather than a fixed count so the harness stays calibrated to
    the measured real session if the schema shape changes. Returns whole tools, so the
    budget is met or slightly exceeded, never split mid-tool.
    """
    tools: list[Any] = []
    accumulated = 0

    for suffix in ["", *_FILLER_DOMAINS]:
        for name, subject, verb in _FILLER_SPECS:
            if accumulated >= target_schema_tokens:
                return tools
            full_name = f"{name}_{suffix}" if suffix else name
            full_subject = f"{subject} ({suffix})" if suffix else subject
            built = _make_filler(full_name, full_subject, verb)
            accumulated += len(json.dumps(built.tool_spec, ensure_ascii=False)) // 4
            tools.append(built)

    return tools


_CORE_SCHEMA_TOKENS = 4_000
"""Measured contribution of the core tools plus the vended retrieval and search tools.

Subtracted from the budget so the filler targets only the remainder and the total lands on
TARGET_SCHEMA_TOKENS rather than overshooting it.
"""

FILLER_TOOLS = _generate_filler_tools(max(0, TARGET_SCHEMA_TOKENS - _CORE_SCHEMA_TOKENS))

CORE_TOOLS = [
    list_accounts,
    list_investment_positions,
    list_investment_transactions,
    get_portfolio_allocation,
    project_yield,
    get_connector_status,
    read_connector_logs,
    force_connector_sync,
    rotate_connector_credentials,
    open_support_case,
    read_aws_documentation,
    search_aws_documentation,
    describe_lambda_function,
    list_s3_objects,
    query_dynamodb_table,
    get_cloudwatch_metrics,
    describe_iam_role,
    estimate_aws_cost,
]


def all_tools() -> list[Any]:
    """Return the full suite: eighteen core tools plus twenty filler tools."""
    from .web import fetch_web_page

    return [*CORE_TOOLS, fetch_web_page, *FILLER_TOOLS]


def account_ids() -> tuple[str, ...]:
    """Return the account identifiers the fixture actually holds, in listing order.

    Exposed so the scenario's filler turns can ask about accounts that exist. They used to ask about
    a running index -- "account 11" -- which matches nothing here, so the agent had nothing to ground
    on: measured, 36 of 42 filler turns made no tool call at all, and the full stack answered 36 of
    them by inventing a sixth account and mislabelling two institutions.

    Read from the fixture rather than restated in the scenario, so a changed fixture cannot leave the
    prompts asking about an account that no longer exists.
    """
    return tuple(account["id"] for account in _ACCOUNTS)


def account_records() -> tuple[dict[str, str], ...]:
    """Return a copy of every fixture account: ``id``, ``institution``, ``type`` and ``balance``.

    The scored filler needs more than the id: a prompt that names the wrong institution for an account,
    or asks for the yield of a savings account, has a false premise, and a careful model answers it by
    correcting the premise instead of calling the tool -- which the expectation scores as a failure.
    """
    return tuple(dict(account) for account in _ACCOUNTS)
