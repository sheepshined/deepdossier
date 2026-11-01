"""图级路由测试：direct 路由必须真的绕过 planner/researcher（回归防线）。"""

from __future__ import annotations

from deepdossier.agents.schemas import QuestionRoute, ResearchPlan, SourceCuration
from deepdossier.llm.provider import StubProvider
from deepdossier.runner import run_research
from deepdossier.store.runs import RunStore
from deepdossier.tools.search import StubRetriever


class _DirectRouteProvider:
    """路由判 direct；若 planner 被执行则立即失败（断言绕过检索流水线）。"""

    def __init__(self) -> None:
        self.calls = 0
        self._usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    @property
    def usage(self) -> dict:
        return dict(self._usage)

    async def complete(self, *, system, user, schema=None, **kw):
        self.calls += 1
        self._usage["calls"] += 1
        if schema is QuestionRoute:
            return QuestionRoute(route="direct", reason="闲聊")
        if schema is ResearchPlan:
            raise AssertionError("direct 路由不应进入 planner")
        if schema is SourceCuration:
            return SourceCuration(keep=[0])
        return "我是 DeepDossier，一个深度调研智能体。"


async def test_direct_route_skips_pipeline(settings):
    store = RunStore(settings.db_path)
    run_id = RunStore.new_run_id()
    store.create_run(run_id, "你是做什么的？")
    result = await run_research(
        "你是做什么的？", settings, store, run_id,
        provider=_DirectRouteProvider(), retriever=StubRetriever(),
    )
    assert result["status"] == "completed"
    assert result["report_md"].startswith("# 直接回答")
    assert result["sources"] == []
    run = store.get_run(run_id)
    assert run["status"] == "completed"


class _AlwaysPlanProvider(StubProvider):
    """路由判 research，其余走 Stub 正常流水线。"""

    async def complete(self, *, system, user, schema=None, **kw):
        if schema is QuestionRoute:
            return QuestionRoute(route="research", reason="研究型")
        return await super().complete(system=system, user=user, schema=schema, **kw)


async def test_research_route_runs_pipeline(settings):
    store = RunStore(settings.db_path)
    run_id = RunStore.new_run_id()
    store.create_run(run_id, "研究型问题")
    result = await run_research(
        "研究型问题", settings, store, run_id,
        provider=_AlwaysPlanProvider(), retriever=StubRetriever(),
    )
    assert result["status"] == "completed"
    assert result["report_md"]  # 走到了 writer
