"""Provider 层测试：JSON 提取/修复、LLM 调用预算、Stub embedding、critic 轮次桩。"""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, ValidationError

from deepdossier.llm.embed import embed_question
from deepdossier.llm.provider import (
    LLMError,
    OpenAICompatProvider,
    StubEmbeddingClient,
    StubProvider,
    build_embedder,
    extract_json,
    parse_structured,
)


class Demo(BaseModel):
    title: str
    count: int


def test_extract_json_plain():
    assert extract_json('{"title": "a", "count": 1}') == '{"title": "a", "count": 1}'


def test_extract_json_with_fence_and_noise():
    raw = '好的，以下是结果：\n```json\n{"title": "a", "count": 2}\n```\n希望有帮助'
    assert parse_structured(raw, Demo) == Demo(title="a", count=2)


def test_parse_structured_rejects_garbage():
    with pytest.raises((ValueError, ValidationError, json.JSONDecodeError)):
        parse_structured("完全不是 JSON", Demo)


async def test_stub_provider_plan_and_report():
    p = StubProvider()
    plan = await p.complete(system="s", user="多智能体企业落地", schema=type(_plan_stub()))
    assert plan.title and len(plan.subquestions) == 2

    report = await p.complete(system="s", user="测试问题\n其他", schema=type(_report_stub()))
    assert "调研报告" in report.markdown and "[1]" in report.markdown
    assert report.sources and report.sources[0].n == 1


def _plan_stub():
    from deepdossier.agents.schemas import ResearchPlan

    return ResearchPlan(title="t", subquestions=[])


def _report_stub():
    from deepdossier.agents.schemas import Report

    return Report(title="t", markdown="m")


def _critique_stub():
    from deepdossier.agents.schemas import Critique

    return Critique(sufficient=True, gaps=[], followup_queries=[])


async def test_stub_critic_rounds():
    p = StubProvider()
    c1 = await p.complete(system="s", user="当前轮次: 1", schema=type(_critique_stub()))
    assert c1.sufficient is False and c1.followup_queries
    c2 = await p.complete(system="s", user="当前轮次: 2", schema=type(_critique_stub()))
    assert c2.sufficient is True


async def test_openai_compat_retry_then_raise():
    """不可恢复错误（连接失败）应退避重试后抛 LLMError，而不是挂死。"""
    provider = OpenAICompatProvider(
        base_url="http://127.0.0.1:9/v1",  # 不可达端口
        api_key="k", model="m", max_retries=1,
    )
    with pytest.raises(LLMError):
        await provider.complete(system="s", user="u", schema=Demo)


async def test_llm_budget_exhausted():
    """调用预算耗尽时立即失败，不再请求 API。"""

    class NeverCalledProvider(OpenAICompatProvider):
        async def complete(self, **kw):  # pragma: no cover - 预算检查应先触发
            raise AssertionError("不应发起真实调用")

    p = OpenAICompatProvider(
        base_url="http://127.0.0.1:9/v1", api_key="k", model="m", max_retries=0, max_calls=1,
    )
    p._usage["calls"] = 1  # 模拟已用尽
    with pytest.raises(LLMError, match="预算"):
        await p.complete(system="s", user="u")


# ---------------- embedding ----------------

async def test_stub_embedding_deterministic():
    e = StubEmbeddingClient()
    v1 = await e.embed("多智能体")
    v2 = await e.embed("多智能体")
    assert v1 == v2 and len(v1) == 16


async def test_embed_question_cache(store):
    e = StubEmbeddingClient()
    v1 = await embed_question(e, store, "缓存问题")
    assert v1 is not None
    assert store.get_query_vec("缓存问题") == v1
    # 第二次命中缓存（embedder 不再被调用 —— 用会抛错的桩验证）
    class Boom:
        async def embed(self, text):
            raise AssertionError("不应重复 embedding")

    v2 = await embed_question(Boom(), store, "缓存问题")
    assert v2 == v1


async def test_embed_question_none_embedder(store):
    assert await embed_question(None, store, "任何问题") is None


def test_build_embedder_disabled(settings):
    assert build_embedder(settings) is None
    settings.embedding_model = "text-embedding-3-small"
    assert build_embedder(settings) is not None
