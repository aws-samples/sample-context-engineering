# Deploying the runtime with the `agentcore` CLI

> **⚠️ Not for production use.** This deployment guide is provided for **experimentation and
> benchmark reproduction only**. The IAM roles, permissions, and configurations below are minimal
> examples — not production-ready baselines. Do not use them in a production environment without an
> independent security review, least-privilege hardening, and operational readiness assessment
> appropriate to your workload.

This is the **manual, do-it-yourself** path to run the strategies on **Amazon Bedrock AgentCore
Runtime**, using the official `agentcore` CLI. 

> **Paid + account-changing.** Deploying creates an ECR repository and image, an IAM execution role,
> an AgentCore Memory resource and an AgentCore Runtime, and running the agent spends on Bedrock.
> `agentcore destroy` removes them. The token benchmark in `validation/` runs without any of this.

## Prerequisites

- **AWS credentials** for an account with these Bedrock models enabled in your region
  (`us-east-1` by default): `us.anthropic.claude-opus-4-8`, `cohere.rerank-v3-5:0`,
  `cohere.embed-multilingual-v3`. Resolved through the standard AWS chain (SSO session, exported
  keys, an instance role) or a named profile.
- **The `agentcore` CLI.** Two exist — pick one:
  - **`@aws/agentcore` (npm) — recommended.** The current, supported CLI: `npm install -g @aws/agentcore@1`.
  - **`bedrock-agentcore-starter-toolkit` (pip).** Already in this repo's venv, but it prints a
    deprecation notice pointing at the npm CLI. Fine for a quick try; do not build on it.
- **Docker** only if you choose a local build. The default cloud path (below) builds in CodeBuild and
  needs no local Docker.
- **X-Ray Transaction Search** enabled on the account (one-time, account-wide) if you want traces —
  it is a prerequisite, not something the deploy creates.

## 1. Point the requirements at the git branch (no wheel)

The container installs the forked SDK from the public branch, pinned to a **commit SHA** for
reproducibility (not the moving branch tip). Create `validation/agentcore/requirements-runtime.txt`:

```
# Forked Strands SDK, from the public branch, pinned to a commit for reproducibility.
strands-agents @ git+https://github.com/scandura/harness-sdk.git@c4083a04d35170b0626cfbdbb5e4926d9b35af20#subdirectory=strands-py
bedrock-agentcore==1.22.0
playwright==1.55.0
html2text==2024.2.26
boto3==1.40.0
```

The SDK is pinned to a full commit SHA (an immutable reference), not a branch name — the value above is
the commit the results were measured against; replace it with another commit SHA if you deploy a
different version. This is what removes the `hatch build` step: the SDK is fetched from git at container
build time, so no wheel is produced or copied.

## 2. Configure the agent

From the repo root. `configure` records the entrypoint, the requirements file, and the execution
role; `--requirements-file` is the key that ties the runtime to the git-based requirements above.

```bash
agentcore configure \
  --entrypoint validation/agentcore/runtime.py \
  --name context_strategy_validation \
  --requirements-file validation/agentcore/requirements-runtime.txt
```

Let it auto-create the ECR repository and the execution role, or pass `--ecr` / `--execution-role`
to reuse existing ones. The execution role needs, at minimum: Bedrock `InvokeModel` + `Rerank` on the
model ARNs above, and the `bedrock-agentcore` memory read/append actions on the memory ARN from step 3.

## 3. Create the Memory resource

Short-term only (no long-term extraction strategies), matching what the harness measures:

```bash
agentcore memory create --name context_strategy_validation
# note the memory id it prints — you pass it to the runtime as AGENTCORE_MEMORY_ID
```

## 4. Deploy

The default (no flags) builds an ARM64 container in the cloud with CodeBuild and deploys it — **no
local Docker required**:

```bash
agentcore deploy --agent context_strategy_validation
```

Two alternatives when you need them:

```bash
agentcore deploy --local          # run the container locally (needs Docker/Finch/Podman)
agentcore deploy --local-build    # build locally, deploy to the cloud runtime
```

Pass the strategy configuration and the memory id to the runtime as environment variables (the
entrypoint `runtime.py` reads them): `CONTEXT_STRATEGY` (e.g. `graph-all` or `baseline`),
`AGENTCORE_MEMORY_ID` (from step 3), and optionally `AGENTCORE_ACTOR_ID` (a role/workload name, never
a person). Set them via your `configure`/`deploy` environment or the CLI's env options.

## 5. Invoke and check

```bash
agentcore status  --agent context_strategy_validation           # config + runtime + endpoint state
agentcore invoke  --agent context_strategy_validation '{"prompt": "Como estão meus investimentos hoje?"}'
agentcore obs     --agent context_strategy_validation           # spans / traces / logs
```

To observe the deployed runtime, invoke it and read its traces through the CLI:

```bash
agentcore invoke --agent context_strategy_validation '{"prompt": "Como estão meus investimentos hoje?"}'
agentcore obs    --agent context_strategy_validation   # spans / traces / logs (2–3 min to reach CloudWatch)
agentcore status --agent context_strategy_validation   # runtime ARN, log group, endpoint state
```

Spans take 2–3 minutes to reach CloudWatch after invocation.

## 6. Tear down

```bash
agentcore destroy --agent context_strategy_validation --dry-run          # preview first
agentcore destroy --agent context_strategy_validation --delete-ecr-repo  # remove image + repo
agentcore memory  delete <memory-id>                                     # remove the memory resource
```

`destroy` removes the runtime, the ECR images, the CodeBuild project and the execution role (only if
no other agent uses it). Use `--dry-run` before the real run.

## Scope and limitations

This deployment path exists solely to **reproduce benchmark measurements** on AgentCore Runtime. It is
not a production deployment reference:

- **IAM roles** — The auto-created execution role has the minimum permissions to run the benchmark.
  Production deployments should scope resource ARNs, add condition keys, and apply permission
  boundaries per your organization's security standards.
- **No operational infrastructure** — No CloudWatch alarms, auto-scaling, health checks, or disaster
  recovery configuration is included.
- **No Bedrock Guardrails** — The harness invokes models without content filtering or PII masking.
  Production workloads should configure [Amazon Bedrock Guardrails](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html).
- **Tear-down is manual** — You must run `agentcore destroy` (step 6) to remove paid resources.
  There is no automated cleanup.

---

