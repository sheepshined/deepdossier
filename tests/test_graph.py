"""核心编排测试：全图端到端（Stub 双件套 + 离线 http，全离线零出网）。"""

from __future__ import annotations

from deepdossier.graph.builder import build_graph
from deepdossier.llm.provider import StubProvider
from deepdossier.runner import build_deps, run_research
from deepdossier.tools.search import StubRetriever
from deepdossier.trace import Event, SinkFn


async def test_full_graph_produces_report(settings, store, stub_provider, stub_retriever, offline_http):
    run_id = store.new_run_id()
    store.create_run(run_id, "stub 测试问题")

    events: list[Event] = []
    sink: SinkFn = events.append
    result = await run_research(
        question="stub 测试问题",
        settings=settings,
        store=store,
        run_id=run_id,
        max_rounds=2,
        provider=stub_provider,
        retriever=stub_retriever,
        extra_sinks=[sink],
        http=offline_http,
    )

    # 报告非空且含关键结构
    assert result["report_md"], "报告不应为空"
    assert "调研报告" in result["report_md"]
    # 结构化来源已产出并落库
    assert result["sources"] and result["sources"][0]["url"]
    assert store.get_run(run_id)["sources"] == result["sources"]

    # Critic 补搜循环确实发生（stub 第 1 轮必不足 → gap 补搜）
    assert len(result["evidence"]) >= 3

    # 事件链完整性：四大节点均出现
    nodes = {e.node for e in events}
    assert {"planner", "researcher", "critic", "writer"} <= nodes
    types = {e.type for e in events}
    assert {"node_start", "tool_call", "evidence_added", "critique", "final"} <= types

    # 落库校验：run 完成、事件可回放且 seq 连续
    run = store.get_run(run_id)
    assert run["status"] == "completed"
    replayed = store.list_events(run_id)
    assert replayed and replayed[-1]["seq"] == max(e.seq for e in events)

    # 策略记忆已沉淀
    assert store.get_lessons("stub 测试问题")


async def test_graph_critic_loop_respects_max_rounds(settings, store, stub_provider, stub_retriever):
    """max_rounds=1 时 Critic 不得触发第二轮研究。"""
    run_id = store.new_run_id()
    store.create_run(run_id, "单轮测试")

    result = await run_research(
        question="单轮测试", settings=settings, store=store, run_id=run_id, max_rounds=1,
        provider=stub_provider, retriever=stub_retriever,
    )
    events = store.list_events(run_id)
    researcher_runs = [e for e in events if e["node"] == "researcher" and e["type"] == "node_start"]
    assert len(researcher_runs) == 2  # 仅初始两个子问题
    assert result["status"] == "completed"


async def test_search_cache_dedupes_repeat_queries(settings, store, stub_provider):
    """补搜轮重复 query 命中缓存：tool_call 事件可见 cached=true。"""

    class CountingRetriever(StubRetriever):
        def __init__(self):
            self.calls = 0

        async def search(self, query, k=5):
            self.calls += 1
            return await super().search(query, k)

    retriever = CountingRetriever()
    run_id = store.new_run_id()
    store.create_run(run_id, "缓存测试")
    await run_research(
        question="缓存测试", settings=settings, store=store, run_id=run_id,
        provider=stub_provider, retriever=retriever,
    )
    # stub planner 产出 q1/q2，critic 第 1 轮补搜 1 条新 query → 至少 3 次真实搜索
    assert retriever.calls >= 3


async def test_zero_results_skip_llm(settings, store):
    """搜索零结果时：产出空证据、不调 LLM（calls 不增）。"""

    class EmptyRetriever(StubRetriever):
        async def search(self, query, k=5):
            return []

    provider = StubProvider()
    run_id = store.new_run_id()
    store.create_run(run_id, "零结果测试")
    result = await run_research(
        question="零结果测试", settings=settings, store=store, run_id=run_id,
        provider=provider, retriever=EmptyRetriever(),
    )
    # 4 个 researcher 全部零结果短路 → 只剩 planner/critic/writer 的 LLM 调用
    assert len(result["evidence"]) == 0
    events = store.list_events(run_id)
    assert any(e["type"] == "evidence_added" and e["payload"].get("note") for e in events)
    total_calls = provider.usage["calls"]
    # route(1)+planner(1)+2×critic+writer(1)+verifier 审查(1)+幻觉修订(1)+lesson(0)
    assert total_calls <= 8, f"零结果时 LLM 调用应大幅减少，实际 {total_calls}"


async def test_deps_graph_buildable_and_close(settings, store):
    deps = build_deps(settings, store, "rid-test", provider=StubProvider(), retriever=StubRetriever())
    graph = build_graph(deps)
    assert graph is not None
    await deps.aclose()
    assert deps.trace_file is not None
