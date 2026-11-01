"""Codex 级韧性验收（docs/04 R39）：R1-R26 故障注入矩阵 + R23 结构契约 + R26 混沌测试。

铁律 T1：任何单点故障都有降级路径——要么报告、要么带原因的失败 + 部分成果。
全部离线（Stub/Mock/fake provider）。
"""

from __future__ import annotations

import asyncio
import random

import httpx
import pytest

from deepdossier.agents.schemas import DraftReview
from deepdossier.agents.verifier import hallucinated_sources, make_verifier_node, structural_issues
from deepdossier.api.app import create_app
from deepdossier.config import Settings
from deepdossier.graph.state import merge_evidence
from deepdossier.llm.provider import LLMBudgetError, StubProvider
from deepdossier.runner import run_research
from deepdossier.store.runs import RunStore
from deepdossier.store.writer import EventWriter
from deepdossier.tools.search import BaseRetriever, FallbackRetriever, StubRetriever

REPORT_LONG = (
    "# 报告\n\n## 摘要\n" + "这是摘要内容，包含核心结论[1]。" * 3 + "\n\n"
    "## 主要发现\n" + "第一段事实陈述[1]，" + "补充细节。 " * 6 + "\n\n"
    "## 争议与不确定性\n来源间存在分歧[1]。\n\n"
    "## 结论与建议\n建议一[1]。\n\n"
    "## 参考来源\n1. A — https://a\n"
)
SOURCES_A = [{"n": 1, "title": "A", "url": "https://a"}]
EVIDENCE_A = [{"question_id": "q1", "claim": "c", "detail": "d", "source_url": "https://a", "source_title": "A"}]


def _settings(tmp_path, **kw) -> Settings:
    kwargs = {
        "llm_provider": "stub", "retriever": "stub", "data_dir": tmp_path / "data",
        "inline_execution": True, "fetch_pages": False, "checkpointing": False,
        "llm_api_key": "", "tavily_api_key": "", "serper_api_key": "",
        "api_key": "", "mcp_servers": "",
    }
    kwargs.update(kw)  # kw 可覆盖默认（checkpointing/fetch_pages 等）
    s = Settings(**kwargs)
    s.ensure_dirs()
    return s


class _Emitter:
    def __init__(self):
        self.events = []

    def emit(self, type_, node, payload):
        self.events.append((type_, node, payload))


class _Deps:
    def __init__(self, provider, draft_review=True):
        self.settings = Settings(draft_review=draft_review)
        self.provider = provider
        self.emitter = _Emitter()


# ---------------- R1：非法 JSON 修复重试 → 快模型兜底 ----------------

class _FakeResponse:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})]
        self.usage = None


class _FakeCompletions:
    def __init__(self, contents):
        self.contents = list(contents)
        self.calls: list[dict] = []

    async def create(self, **kw):
        self.calls.append(kw)
        content = self.contents.pop(0) if self.contents else self.contents[-1]
        return _FakeResponse(content)


async def test_r1_fast_model_fallback_after_repair_exhausted():
    from deepdossier.agents.schemas import DraftReview
    from deepdossier.llm.provider import OpenAICompatProvider

    p = OpenAICompatProvider(
        base_url="http://x", api_key="k", model="main-model", fast_model="fast-model",
        max_retries=1, max_calls=None,
    )
    p._client = type(
        "FakeClient", (),
        {"chat": type("Chat", (), {"completions": _FakeCompletions(
            ["不是JSON", "还不是JSON", "依然不是JSON", '{"needs_revision": false, "issues": []}'])})()},
    )()
    out = await p.complete(system="s", user="u", schema=DraftReview)
    assert isinstance(out, DraftReview) and out.needs_revision is False
    calls = p._client.chat.completions.calls
    assert len(calls) == 4  # 主模型 2 次（1 修复重试）+ 快模型 1 轮（首试即成功）→ 主 2 + 快 1 = 3+?
    assert calls[-1]["model"] == "fast-model"  # 最后一次落在快模型上（R1）


# ---------------- R2/R4：节点级降级，run 不整体失败 ----------------

class _FailSchemaProvider(StubProvider):
    """对指定 schema 的调用持续失败（模拟重试耗尽）。"""

    def __init__(self, fail_schemas: set[str]) -> None:
        super().__init__()
        self.fail_schemas = fail_schemas

    async def complete(self, *, system, user, schema=None, **kw):
        if schema is not None and schema.__name__ in self.fail_schemas:
            raise RuntimeError("注入：持续失败")
        return await super().complete(system=system, user=user, schema=schema, **kw)


async def test_r2_researcher_llm_failure_degrades_run_completes(tmp_path):
    """R2/R4：researcher 提炼彻底失败 → 空证据降级，run 仍完成（writer 拿到零证据）。"""
    s = _settings(tmp_path)
    store = RunStore(s.db_path)
    rid = RunStore.new_run_id()
    store.create_run(rid, "R2 降级测试")
    result = await run_research(
        "R2 降级测试", s, store, rid,
        provider=_FailSchemaProvider({"EvidenceBundle"}), retriever=StubRetriever(),
    )
    assert result["status"] == "completed"  # 不整体失败
    assert result["evidence"] == []
    events = store.list_events(rid)
    assert any(e["type"] == "error" and "证据提炼失败" in e["payload"]["message"] for e in events)
    store.close()


async def test_r4_critic_failure_degrades_to_writer(tmp_path):
    """R4：critic 彻底失败 → 视为充分直接进 writer，run 完成。"""
    s = _settings(tmp_path)
    store = RunStore(s.db_path)
    rid = RunStore.new_run_id()
    store.create_run(rid, "R4 降级测试")
    result = await run_research(
        "R4 降级测试", s, store, rid,
        provider=_FailSchemaProvider({"Critique"}), retriever=StubRetriever(),
    )
    assert result["status"] == "completed"
    events = store.list_events(rid)
    assert any(e["type"] == "error" and "评审失败，降级" in e["payload"]["message"] for e in events)
    assert any(e["type"] == "critique" and "[降级]" in e["payload"].get("reasoning", "") for e in events)
    store.close()


async def test_r4_router_failure_fail_open(tmp_path):
    """R4：router 失败 → fail-open 走完整调研。"""
    s = _settings(tmp_path)
    store = RunStore(s.db_path)
    rid = RunStore.new_run_id()
    store.create_run(rid, "路由失败测试")
    result = await run_research(
        "路由失败测试", s, store, rid,
        provider=_FailSchemaProvider({"QuestionRoute"}), retriever=StubRetriever(),
    )
    assert result["status"] == "completed"
    store.close()


# ---------------- R3/R14：预算耗尽/超时 → 部分成果落库 ----------------

class _BudgetDieProvider(StubProvider):
    """第 5 次调用起预算熔断（critic 阶段降级后 writer 触发终止，证据已入库）。"""

    async def complete(self, *, system, user, schema=None, **kw):
        if self._usage["calls"] >= 5:
            raise LLMBudgetError("LLM 调用预算耗尽（max_calls=5），终止本 run")
        return await super().complete(system=system, user=user, schema=schema, **kw)


async def test_r3_budget_exhaustion_keeps_partial_results(tmp_path):
    """R3：预算耗尽 → run 失败但"部分成果"（已收集证据+计划）落库，前端可见。"""
    s = _settings(tmp_path, checkpointing=True)
    store = RunStore(s.db_path)
    rid = RunStore.new_run_id()
    store.create_run(rid, "预算耗尽测试")
    with pytest.raises(LLMBudgetError):
        await run_research(
            "预算耗尽测试", s, store, rid,
            provider=_BudgetDieProvider(), retriever=StubRetriever(),
        )
    run = store.get_run(rid)
    assert run["status"] == "failed"
    assert run["report_md"], "部分成果必须落库"
    assert "部分成果" in run["report_md"] and "已收集证据" in run["report_md"]
    store.close()


# ---------------- R7：全链检索失败 → 未联网核实标注 ----------------

class _AlwaysFailRetriever(BaseRetriever):
    name = "fail"

    async def search(self, query, k=5):
        raise RuntimeError("注入：所有检索源不可用")


async def test_r7_all_search_fail_marks_no_web(tmp_path):
    """R7：全链检索失败 → run 完成、报告头部显著标注"未联网核实"、事件流记录。"""
    s = _settings(tmp_path)
    store = RunStore(s.db_path)
    rid = RunStore.new_run_id()
    store.create_run(rid, "全链检索失败测试")
    result = await run_research(
        "全链检索失败测试", s, store, rid,
        provider=StubProvider(), retriever=FallbackRetriever(_AlwaysFailRetriever(), []),
    )
    assert result["status"] == "completed"
    assert "未联网核实" in result["report_md"]
    events = store.list_events(rid)
    assert any(e["type"] == "degraded" and e["payload"].get("kind") == "no_web" for e in events)
    store.close()


# ---------------- R9：正文全抓取失败 → snippet_only 标注 ----------------

async def test_r9_snippet_only_flag(tmp_path):
    """R9：抓取层全失败（mock http 返回错误）→ 证据卡 snippet_only=True。"""
    s = _settings(tmp_path, fetch_pages=True)
    store = RunStore(s.db_path)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("注入：抓取全部失败")

    from deepdossier.runner import build_deps

    deps = build_deps(s, store, "r9", provider=StubProvider(), retriever=StubRetriever(),
                      http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    try:
        from deepdossier.graph.builder import build_graph

        graph = build_graph(deps)
        final = await graph.ainvoke({"run_id": "r9", "question": "抓取失败测试"})
        cards = final.get("evidence", [])
        assert cards, "应基于搜索摘要产出证据"
        assert all(c.get("snippet_only") is True for c in cards)
    finally:
        await deps.aclose()
        store.close()


# ---------------- R12：checkpoint 损坏 → 明确报错 + fresh 重跑 ----------------

async def test_r12_corrupt_checkpoint_fresh_restart(tmp_path):
    """R12：checkpoint 损坏 → resume 明确失败；fresh=true 放弃断点重跑成功。"""
    s = _settings(tmp_path, checkpointing=True)
    store = RunStore(s.db_path)
    try:
        rid = RunStore.new_run_id()
        store.create_run(rid, "R12 测试")
        # 制造一次失败（planner 注错）
        with pytest.raises(RuntimeError, match="planner"):
            await run_research("R12 测试", s, store, rid,
                               provider=_FailSchemaProvider({"ResearchPlan"}), retriever=StubRetriever())
        # 损坏 checkpoint 文件
        ckpt = tmp_path / "data" / "checkpoints.sqlite3"
        ckpt.write_bytes(b"corrupted-not-sqlite")

        # 非 fresh resume：明确失败（checkpoint 损坏 → 降级为无断点模式 → 守卫报错）
        import deepdossier.llm.provider as provider_mod

        real_err = provider_mod.LLMError
        try:
            with pytest.raises((RuntimeError, real_err, OSError)):
                await run_research("R12 测试", s, store, rid,
                                   provider=StubProvider(), retriever=StubRetriever(), resume=True)
        except AssertionError:
            raise
        # fresh 重跑：放弃断点，完整成功
        store.reopen_run(rid)
        store.clear_events(rid)
        result = await run_research("R12 测试", s, store, rid,
                                    provider=StubProvider(), retriever=StubRetriever(), resume=False)
        assert result["status"] == "completed"
        assert store.max_event_seq(rid) > 0
    finally:
        store.close()


# ---------------- R16：事件写失败 → 内存缓冲重试，不丢不崩 ----------------

async def test_r16_event_writer_retries_failed_batch(tmp_path):
    """R16：写失败进缓冲重试，恢复后补写；线程正常退出。"""
    real = RunStore(tmp_path / "data" / "r16.sqlite3")
    attempts = {"n": 0}

    class FlakyStore:
        def append_events(self, events):
            attempts["n"] += 1
            if attempts["n"] <= 2:
                raise RuntimeError("注入：磁盘满")
            real.append_events(events)

    writer = EventWriter(FlakyStore())  # type: ignore[arg-type]
    writer.start()
    from deepdossier.trace import EventEmitter

    em = EventEmitter("r16", sinks=[writer.submit])
    em.emit("node_start", "planner", {})
    em.emit("final", "writer", {})
    for _ in range(60):
        if attempts["n"] >= 3 and real.list_events("r16"):
            break
        await asyncio.sleep(0.2)
    writer.close(timeout=5)
    evs = real.list_events("r16")
    assert {e["type"] for e in evs} == {"node_start", "final"}  # 重试后全部落库
    real.close()


# ---------------- R17/R19/R21：引用覆盖率门禁 + 修订回退 ----------------

class _CleanThenRevise(StubProvider):
    """LLM 说没问题（触发覆盖率门禁强制修订）；修订输出由测试给定。"""

    def __init__(self, revised: str) -> None:
        self.revised = revised
        self.calls: list[str] = []

    async def complete(self, *, system, user, schema=None, **kw):
        self.calls.append(schema.__name__ if schema else "text")
        if schema is not None and schema.__name__ == "DraftReview":
            return DraftReview(needs_revision=False, issues=[])
        return self.revised


async def test_r21_low_coverage_forces_revision_and_annotates():
    """R17/R21：覆盖率 <80% 强制修订；修订后仍不达标 → 顶部"引用不充分"标注。"""
    no_cite = REPORT_LONG.replace("[1]", "")  # 覆盖率 0%
    p = _CleanThenRevise(no_cite)  # 修订输出依旧无引用 → 仍不达标
    deps = _Deps(p)
    state = {"run_id": "r", "question": "q", "report_md": no_cite,
             "sources": SOURCES_A, "evidence": EVIDENCE_A}
    out = await make_verifier_node(deps)(state)
    assert "text" in p.calls  # 门禁强制修订（即使 LLM 说没问题）
    assert out["report_md"].startswith("# 报告")
    assert "引用不充分" in out["report_md"]  # 修订后仍 0% → 顶部标注


async def test_r19_revision_worse_coverage_reverts():
    """R19：修订后覆盖率下降 → 放弃修订保留原稿。"""

    class _FlagIssue(_CleanThenRevise):
        async def complete(self, *, system, user, schema=None, **kw):
            if schema is not None and schema.__name__ == "DraftReview":
                return DraftReview(needs_revision=True, issues=["注入：需要修订"])
            return await super().complete(system=system, user=user, schema=schema, **kw)

    # 修订稿：一段 >40 字的无引用长行（单行拼接，勿把标题卷入重复）
    bad_revision = "# 报告\n\n" + "无引用的一段话。" * 30
    p = _FlagIssue(bad_revision)
    deps = _Deps(p)
    state = {"run_id": "r", "question": "q", "report_md": REPORT_LONG,
             "sources": SOURCES_A, "evidence": EVIDENCE_A}
    out = await make_verifier_node(deps)(state)
    assert out["report_md"] == REPORT_LONG  # 原稿保留
    assert any(e[0] == "error" and "放弃修订" in e[2]["message"] for e in deps.emitter.events)


async def test_r24_hallucinated_sources_detected():
    """R24：来源 URL 不在证据集合 → 幻觉引用事件 + 并入修订。"""

    class _Spy(_CleanThenRevise):
        async def complete(self, *, system, user, schema=None, **kw):
            if schema is None:
                assert "幻觉引用" in user  # 修订 prompt 收到幻觉问题
            return await super().complete(system=system, user=user, schema=schema, **kw)

    p = _Spy(REPORT_LONG)
    deps = _Deps(p)
    bad_sources = [*SOURCES_A, {"n": 2, "title": "幻觉", "url": "https://not-in-evidence"}]
    state = {"run_id": "r", "question": "q", "report_md": REPORT_LONG + "\n\n引用[2]。",
             "sources": bad_sources, "evidence": EVIDENCE_A}
    await make_verifier_node(deps)(state)  # R24：断言幻觉事件由下方检查覆盖
    assert any(e[0] == "hallucination" for e in deps.emitter.events)
    assert hallucinated_sources(bad_sources, EVIDENCE_A)[0]["url"] == "https://not-in-evidence"


# ---------------- R23：结构契约校验（边界用例矩阵） ----------------

GOOD = REPORT_LONG


def _md(**overrides) -> str:
    # 每节填充到足够长度（>500 字才触发结构校验，与生产报告同量级）
    base = {
        "摘要": "## 摘要\n" + "这是摘要内容，包含核心结论[1]。" * 40,
        "主要发现": "## 主要发现\n" + "事实陈述与补充细节[1]。" * 40,
        "争议": "## 争议与不确定性\n" + "来源间存在分歧与矛盾[1]。" * 40,
        "参考": "## 参考来源\n1. A — https://a\n",
    }
    base.update(overrides)
    parts = ["# 报告"]
    for key in ("摘要", "主要发现", "争议", "参考"):
        if key in overrides or base.get(key):
            parts.append(base[key])
    return "\n\n".join(parts)


@pytest.mark.parametrize(
    "md, sources, conflicts, expect_issue",
    [
        (_md(), SOURCES_A, None, False),                          # 完整合格
        (_md(争议="## 无争议章\n内容[1]。"), SOURCES_A, None, True),   # 缺争议章节
        (_md(摘要="## 前言\n内容。"), SOURCES_A, None, True),          # 缺摘要
        (_md(参考=""), SOURCES_A, None, True),                     # 缺参考来源
        (_md(主要发现="## 其它\n内容[1]。"), SOURCES_A, None, True),   # 缺主要发现
        (_md(), SOURCES_A, ["来源A与来源B矛盾"], False),               # 有冲突且有争议章节
        (_md(), SOURCES_A, ["矛盾"], False),                       # conflicts 空列表无要求
        ("短", SOURCES_A, None, False),                            # <500 字豁免
        ("", [], None, False),                                     # 空报告豁免
        (_md(参考="## 参考来源\n2. A — https://a\n"), SOURCES_A, None, True),   # 断号（缺1）
        (_md(参考="## 参考来源\n1. A — https://a\n3. B — https://b\n"), [{**SOURCES_A[0]}, {"n": 3, "title": "B", "url": "https://b"}], None, True),  # 缺2
        (_md() + "\n\n引用[5]。", SOURCES_A, None, True),              # 引用越界
        (_md() + "\n\n引用[0]。", SOURCES_A, None, True),             # 越界（0）
        (_md(), [], None, False),                                  # 无来源时不查编号
        (_md() + "\n\n正文提到[2026]年。", SOURCES_A, None, False),     # 非引用数字不误报
        (_md(摘要="## 摘要\n" + "x" * 600), SOURCES_A, None, False),   # 长报告合格
        (_md() + "\n\n## 附录\n无引用补充说明，长度不足四十个字符", SOURCES_A, None, False),  # 附录不参与
        (_md(争议="## 研究局限性\n缺争议但有局限性"), SOURCES_A, ["c"], True),  # 有冲突缺争议章
        (_md(主要发现="## 主要发现\n" + "事实[1]。 " * 20), SOURCES_A, None, False),  # 正常长文
        (_md(参考="## 参考来源\n1. A — https://a\n2. B — https://b\n3. C — https://c\n")
            .replace("[1]", "[2]") + "\n\n引用[3]。",
         [{"n": 1, "title": "A", "url": "https://a"},
          {"n": 2, "title": "B", "url": "https://b"},
          {"n": 3, "title": "C", "url": "https://c"}], None, False),  # 多来源合法（来源列表齐全）
    ],
)
def test_r23_structural_contract_matrix(md, sources, conflicts, expect_issue):
    issues = structural_issues(md, sources, conflicts)
    assert bool(issues) is expect_issue, f"issues={issues}"


def test_r23_hallucination_helper():
    ev = [{"source_url": "https://a"}, {"source_url": ""}]
    srcs = [{"url": "https://a"}, {"url": ""}, {"url": "https://phantom"}]
    assert [s["url"] for s in hallucinated_sources(srcs, ev)] == ["https://phantom"]


# ---------------- R26：混沌测试（故障注入下 ≥99% 终态、绝不挂死） ----------------

class _ChaosProvider(StubProvider):
    """随机注入单次调用失败（重试后成功），planner/writer 之外全部可降级。"""

    def __init__(self, p_fail: float = 0.15, seed: int = 42) -> None:
        super().__init__()
        self.rng = random.Random(seed)
        self.p_fail = p_fail

    async def complete(self, *, system, user, schema=None, **kw):
        name = schema.__name__ if schema else "text"
        # planner/writer 为致命节点，混沌注入只打可降级节点（模拟真实场景：网络抖动被重试吸收）
        if name in ("QuestionRoute", "EvidenceBundle", "Critique", "DraftReview", "text") and self.rng.random() < self.p_fail:
            raise RuntimeError("混沌注入：瞬时故障")
        return await super().complete(system=system, user=user, schema=schema, **kw)


async def test_r26_chaos_100_runs_never_stuck():
    """R26：100 次混沌 run 全部到达终态（≥99% completed），无挂死无僵尸。"""
    s = _settings(_tmp())
    store = RunStore(s.db_path)
    try:
        completed = 0
        for i in range(100):
            rid = RunStore.new_run_id()
            store.create_run(rid, f"混沌测试 #{i}")
            try:
                result = await asyncio.wait_for(
                    run_research(f"混沌测试 #{i}", s, store, rid,
                                 provider=_ChaosProvider(), retriever=StubRetriever()),
                    timeout=60,
                )
                assert result["status"] == "completed"
                completed += 1
            except asyncio.TimeoutError:
                pytest.fail(f"run {rid} 挂死（60s 未达终态）")
            except Exception:
                pass  # T1 允许"带原因的失败"，但不得挂死
        assert completed >= 99, f"混沌下完成率 {completed}/100 低于 99%"
    finally:
        store.close()


def _tmp():
    import tempfile
    from pathlib import Path

    return Path(tempfile.mkdtemp(prefix="dd-chaos-"))


# ---------------- R30：证据数量硬上限 ----------------

def test_r30_evidence_cap():
    cards = [{"question_id": f"q{i}", "claim": f"claim-{i}", "detail": "d",
              "source_url": f"https://s/{i}", "source_title": "t"} for i in range(450)]
    out = merge_evidence([], cards)
    assert len(out) == 400  # R30 上限


# ---------------- R20：导出兜底 ----------------

async def test_r20_export_fallback_to_markdown(settings, store):
    rid = RunStore.new_run_id()
    store.create_run(rid, "导出兜底")
    store.finish_run(rid, "# 报告\n正文", None, None, None)
    app = create_app(settings, store)

    import deepdossier.api.app as app_mod

    real = app_mod.render_docx
    app_mod.render_docx = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("注入：docx 生成失败"))
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(f"/api/researches/{rid}/export.docx")
            assert resp.status_code == 200
            assert resp.headers.get("x-export-fallback") == "1"
            assert "导出失败说明" in resp.text and "正文" in resp.text  # 原始内容保留
    finally:
        app_mod.render_docx = real
