# Deploying the agent with the AgentCore CLI

> **⚠️ Not for production use.** This deployment guide is provided for **experimentation and
> benchmark reproduction only**. The IAM roles, permissions, and configurations below are minimal
> examples — not production-ready baselines. Do not use them in a production environment without an
> independent security review, least-privilege hardening, and operational readiness assessment
> appropriate to your workload.

Run the three context practices as a real, deployed agent on **Amazon Bedrock AgentCore Runtime**,
using the official `agentcore` CLI.

> **Paid + account-changing.** Deploying provisions an AgentCore Runtime endpoint and its supporting
> resources through AWS CDK, and running the agent spends on Bedrock. Step 8 tears it down. The token
> benchmark in `validation/01-designA-B-D/` runs without any of this.

The CLI answers *how the strategies behave* as a deployed agent. It does not measure token cost — that
is the benchmark's job, and the two are independent.

## Prerequisites

- **Node.js 20 or later.** The CLI ships as an npm package.
- **Python 3.10 or later** for the agent code.
- **AWS credentials**, resolved through the standard AWS chain (SSO session, exported keys, a named
  profile), for an account with these Bedrock models enabled in your region (`us-east-1` by default):
  `us.anthropic.claude-opus-4-8`, `cohere.rerank-v3-5:0`, `cohere.embed-multilingual-v3`.
- **IAM permissions** to call the AgentCore APIs and to assume the CDK bootstrap roles the deploy uses.
- **Docker** only if you choose the `Container` build type. The default `CodeZip` build needs no Docker.

Install the CLI:

```bash
npm install -g @aws/agentcore
agentcore --version
```

> If `--version` errors instead of printing a version, an older `agentcore` from the
> `bedrock-agentcore-starter-toolkit` pip package is shadowing the npm one on your `PATH`. That package
> is deprecated. Run `pip uninstall bedrock-agentcore-starter-toolkit`, open a new terminal, retry.

## 1. Create the project

The CLI scaffolds the project — it does not deploy a loose script, so there is no entrypoint path to
pass. Run it outside this repo, or in a directory of your own:

```bash
agentcore create \
  --project-name ContextStrategies \
  --name ContextAgent \
  --language Python \
  --framework Strands \
  --model-provider Bedrock \
  --memory short-term \
  --build CodeZip
```

`agentcore create` with no flags runs an interactive wizard instead. It produces:

```
ContextStrategies/
├── agentcore/
│   ├── agentcore.json      agents, memory stores, gateways — what gets provisioned
│   ├── aws-targets.json    accounts and regions to deploy to
│   └── cdk/
└── app/
    └── ContextAgent/
        ├── main.py         the entrypoint you edit in step 2
        └── pyproject.toml  the dependencies
```

## 2. Install the strategies in `main.py`

This is the step that matters: the entrypoint is where you choose which practices the deployed agent
runs. Take the wiring from
[`how-to/01-designA-B-D-agent-sample.md`](../../../how-to/01-designA-B-D-agent-sample.md) — it shows each
plugin alone and the three combined — and build the agent inside the generated `main.py`.

Read the strategy choice from the environment so one deployment can be redeployed as any configuration
rather than hardcoding one:

```python
import os

STRATEGY = os.environ.get("CONTEXT_STRATEGY", "graph-all")
```

## 3. Declare the dependencies

Add the forked SDK to `app/ContextAgent/pyproject.toml`, pinned to an immutable **commit SHA** rather
than a moving branch tip. [`requirements.txt`](requirements.txt) beside this file carries the same pins
and is the list to copy from:

```toml
dependencies = [
  "strands-agents @ git+https://github.com/scandura/harness-sdk.git@c4083a04d35170b0626cfbdbb5e4926d9b35af20#subdirectory=strands-py",
  "bedrock-agentcore==1.22.0",
  "boto3==1.40.0",
]
```

The SHA above is the commit the benchmark results were measured against — replace it to deploy a
different version. Nothing is built locally: the SDK is fetched from git when the artifact is packaged.

## 4. Test locally, before spending on a deploy

```bash
cd ContextStrategies
agentcore dev
```

`agentcore dev` creates the virtualenv, installs the dependencies, starts a local server with hot
reload, and opens the agent inspector in your browser so you can chat with the agent and read its
traces. Model calls are real and billed; the runtime is not yet provisioned.

## 5. Deploy

Preview first — this shows what CDK would provision, and changes nothing:

```bash
agentcore deploy --dry-run
```

Then deploy:

```bash
agentcore deploy
```

It packages the code, synthesizes and provisions through CDK, creates the Runtime endpoint, and wires up
CloudWatch logging and observability. The first deploy takes a few minutes while CDK bootstraps the
account; later ones are faster. `agentcore deploy --diff` shows the CDK diff of a subsequent change.

## 6. Memory

`--memory short-term` in step 1 already declared it. To add memory to a project created without it:

```bash
agentcore add memory
agentcore deploy          # provisions what add wrote into agentcore.json
```

Short-term only is what the benchmark's scenario matches: a single session, nothing extracted across
sessions.

## 7. Invoke and observe

```bash
agentcore status                                        # deployed resources and their state
agentcore invoke --prompt "How are my investments today?"
agentcore invoke --prompt "And the statement?" --session-id abc123   # continue the conversation
agentcore logs --since 15m                              # runtime logs
agentcore traces                                        # spans, 2-3 min to reach CloudWatch
```

`--session-id` is what makes a multi-turn conversation, which is the only condition under which these
practices do anything: they act on accumulated context, so a single prompt shows nothing.

## 8. Tear down

Paid resources stay until you remove them. There is no single destroy command — you remove the resource
from the project config, then deploy to reconcile:

```bash
agentcore remove          # pick the runtime/memory to remove from agentcore.json
agentcore deploy          # CDK deprovisions what was removed
agentcore status          # confirm nothing is left deployed
```

`agentcore status --state deployed` lists anything still provisioned. Check it before you walk away.

## Scope and limitations

This deployment path exists solely to **run the practices as a deployed agent**. It is not a production
deployment reference:

- **IAM roles** — The roles CDK creates carry the minimum permissions to run the agent. Production
  deployments should scope resource ARNs, add condition keys, and apply permission boundaries per your
  organization's security standards.
- **No operational infrastructure** — No alarms, auto-scaling, health checks, or disaster recovery
  configuration is included.
- **No Bedrock Guardrails** — The agent invokes models without content filtering or PII masking.
  Production workloads should configure [Amazon Bedrock Guardrails](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html).
- **Tear-down is manual** — Step 8 is not automatic. Unremoved resources keep costing.

For the full command surface, see the
[AgentCore CLI reference](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agentcore-cli-reference.html).
