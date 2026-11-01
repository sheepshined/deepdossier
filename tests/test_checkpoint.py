"""断点续跑（SqliteSaver checkpoint）测试：中途注错 → resume → 验证恢复与 seq 续接。"""

from __future__ import annotations

import pytest

from deepdossier.config import Settings
from deepdossier.llm.provider import LLMError, StubProvider
from deepdossier.runner import run_research
from deepdossier.store.runs import RunStore
from deepdossier.tools.search import StubRetriever


class _MidwayFailProvider(StubProvider):
    """router 正常，planner 失败（planner 为致命节点；researcher 失败会按 R2/R4 降级不再致命）。

    证据提炼阶段注错已无法让 run 失败（R2/R4 降级），故选择 planner 注错：
    失败发生在 router 之后 → checkpoint 已落，续跑可恢复。
    """

    async def complete(self, *, system, user, schema=None, **kw):
        if schema is not None and schema.__name__ == "ResearchPlan":
            raise LLMError("注错：planner 规划失败")
        return await super().complete(system=system, user=user, schema=schema, **kw)


@pytest.fixture()
def ckpt_settings(tmp_path) -> Settings:
    s = Settings(
        llm_provider="stub",
        retriever="stub",
        data_dir=tmp_path / "data",
        inline_execution=True,
        fetch_pages=False,
        checkpointing=True,
        llm_api_key="",
        tavily_api_key="",
        serper_api_key="",
        api_key="",
        mcp_servers="",
    )
    s.ensure_dirs()
    return s


async def test_checkpoint_resume_roundtrip(ckpt_settings, tmp_path):
    store = RunStore(ckpt_settings.db_path)
    try:
        await _roundtrip(store, ckpt_settings, tmp_path)
    finally:
        store.close()


async def _roundtrip(store, ckpt_settings, tmp_path):
    run_id = RunStore.new_run_id()
    store.create_run(run_id, "断点续跑测试问题")

    # 1) 首次运行：researcher 阶段注错 → run 失败，但 router/planner 已 checkpoint
    with pytest.raises(LLMError, match="planner 规划失败"):
        await run_research(
            "断点续跑测试问题", ckpt_settings, store, run_id,
            provider=_MidwayFailProvider(), retriever=StubRetriever(),
        )
    run = store.get_run(run_id)
    assert run["status"] == "failed"
    seq_before = store.max_event_seq(run_id)
    assert seq_before > 0  # planner 等节点事件已落库
    assert (tmp_path / "data" / "checkpoints.sqlite3").exists()

    # 2) 续跑：换回正常 provider，从 checkpoint 继续
    result = await run_research(
        "断点续跑测试问题", ckpt_settings, store, run_id,
        provider=StubProvider(), retriever=StubRetriever(), resume=True,
    )
    assert result["status"] == "completed"
    assert "调研报告" in result["report_md"]

    # 3) 事件 seq 从断点前最大值续接，且没有覆盖历史事件
    events = store.list_events(run_id)
    seqs = [e["seq"] for e in events]
    assert len(seqs) == len(set(seqs))  # 无覆盖
    assert max(seqs) > seq_before  # 有新增
    assert events[0]["seq"] == 1  # 历史事件完好


async def test_resume_requires_checkpointing(settings):
    """checkpointing 关闭时 resume 必须显式报错（而非静默丢状态）。"""
    store = RunStore(settings.db_path)
    try:
        run_id = RunStore.new_run_id()
        store.create_run(run_id, "问题")
        with pytest.raises(RuntimeError, match="checkpointing"):
            await run_research(
                "问题", settings, store, run_id,
                provider=StubProvider(), retriever=StubRetriever(), resume=True,
            )
    finally:
        store.close()
