"""``user_agent_extra`` on the three AWS-backed classes: default ``context-core``, overridable, extended not replaced.

A binding passes its own tag (the Strands plugins pass ``strands-agents``), so AWS-side attribution of its users'
calls does not change when the binding moves onto the core. Offline: a fake session records the client config.
"""

from __future__ import annotations

from typing import Any

import pytest
from botocore.config import Config as BotocoreConfig

from context_core.graph.matcher import EmbeddingSimilarityMatcher
from context_core.relevance.reranker import BedrockReranker
from context_core.relevance.store import S3Store


class FakeSession:
    """Records the ``config`` every ``client()`` call receives, and returns a stand-in client."""

    def __init__(self) -> None:
        self.configs: list[BotocoreConfig] = []

    def client(self, service_name: str, config: BotocoreConfig | None = None, **kwargs: Any) -> Any:
        self.configs.append(config)
        return object()

    @property
    def tag(self) -> str:
        (config,) = self.configs
        return config.user_agent_extra


def build_reranker(session: FakeSession, **kwargs: Any) -> None:
    BedrockReranker(model_id="arn:aws:bedrock:us-east-1::foundation-model/x", boto_session=session, **kwargs)


def build_store(session: FakeSession, **kwargs: Any) -> None:
    S3Store(bucket="b", boto_session=session, **kwargs)


def build_matcher(session: FakeSession, **kwargs: Any) -> None:
    # The matcher opens its client on first need, so force it.
    EmbeddingSimilarityMatcher(boto_session=session, **kwargs)._ensure_client()


BUILDERS = pytest.mark.parametrize("build", [build_reranker, build_store, build_matcher], ids=["reranker", "s3", "matcher"])


@BUILDERS
def test_the_default_tag_is_context_core(build) -> None:
    session = FakeSession()
    build(session)
    assert session.tag == "context-core"


@BUILDERS
def test_a_binding_can_pass_its_own_tag(build) -> None:
    session = FakeSession()
    build(session, user_agent_extra="strands-agents")
    assert session.tag == "strands-agents"


@pytest.mark.parametrize("build", [build_reranker, build_store], ids=["reranker", "s3"])
def test_a_caller_config_tag_is_extended_with_the_binding_tag(build) -> None:
    session = FakeSession()
    build(session, boto_client_config=BotocoreConfig(user_agent_extra="my-app"), user_agent_extra="strands-agents")
    assert session.tag == "my-app strands-agents"
