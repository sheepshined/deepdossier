"""端到端运行器：CLI 与 API 共用的入口。

职责：组装依赖（事件三落点）→ 执行图 → 落报告与结构化来源 → 沉淀策略记忆。

关键顺序保证：deps.aclose()（排空事件写线程）必须先于 store.finish_run()，
否则 SSE 可能在事件落库前看到终态而提前断流。

断点续跑：启用 checkpointing 时 LangGraph 每个超级步落一次 checkpoint
（AsyncSqliteSaver，独立 checkpoints.sqlite3）；run 失败/超时后可 resume ——
从最近 checkpoint 继续，事件 seq 从已有最大值续接，不会覆盖历史。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import httpx

from deepdossier.graph.builder import build_graph
from deepdossier.graph.deps import ResearchDeps
from deepdossier.llm.embed import embed_question
from deepdossier.llm.provider import BaseLLMProvider
from deepdossier.store.writer import EventWriter
from deepdossier.tools.search import BaseRetriever
from deepdossier.trace import EventEmitter, JsonlTrace, SinkFn

if TYPE_CHECKING:
    from deepdossier.config import Settings
    from deepdossier.store.runs import RunStore

logger = logging.getLogger("deepdossier.runner")

LESSON_SYSTEM = (
    "你是研究方法教练。用一两句话总结本次调研中最有效的搜索策略与最大盲区，"
    "供未来同类调研参考。直接输出结论文字。"
)

# 进程级 checkpointer 缓存已废弃：aiosqlite 连接与事件循环绑定，跨 loop 复用会死锁。
# 改为每次 run 独立开关连接（本地 sqlite，开销可忽略）。


async def open_checkpointer(settings: Settings):
    """为单次 run 打开 AsyncSqliteSaver；用完必须 close_checkpointer。关闭/失败返回 None。"""
    if not settings.checkpointing:
        return None
    conn = None
    try:
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        conn = await aiosqlite.connect(str(settings.data_dir / "checkpoints.sqlite3"))
        saver = AsyncSqliteSaver(conn)
        await saver.setup()  # 建表（幂等），顺带立即暴露初始化问题
        logger.info("checkpointer 就绪: %s", settings.data_dir / "checkpoints.sqlite3")
        return saver
    except Exception:  # noqa: BLE001 — checkpoint 不可用时降级为无断点模式
        # R16 卫生：初始化失败必须关闭连接线程（aiosqlite 非守护线程会让进程退出挂起）
        if conn is not None:
            with contextlib.suppress(Exception):
                await conn.close()
        logger.exception("checkpointer 初始化失败，降级为无断点模式")
        return None


async def close_checkpointer(saver) -> None:
    if saver is None:
        return
    with contextlib.suppress(Exception):
        await saver.conn.close()


class _EventedProvider(BaseLLMProvider):
    """provider 事件代理：每次 LLM 调用发射 llm_call 事件（F25 全程可观测）。

    事件含 schema 名、快慢模型标记与本次 token 增量；耗时超过阈值发 slow_node 事件（R28）；
    usage/aclose 透传。
    """

    def __init__(self, inner: BaseLLMProvider, emitter, slow_call_seconds: float = 60.0) -> None:
        self._inner = inner
        self._emitter = emitter
        self._slow_seconds = slow_call_seconds

    @property
    def usage(self) -> dict[str, int]:
        return self._inner.usage

    async def aclose(self) -> None:
        close = getattr(self._inner, "aclose", None)
        if close:
            await close()

    async def complete(self, *, system, user, schema=None, temperature=None, max_tokens=None, fast=False):
        tag = schema.__name__ if schema is not None else "text"
        before = dict(self._inner.usage or {})
        t0 = time.monotonic()
        out = await self._inner.complete(
            system=system, user=user, schema=schema,
            temperature=temperature, max_tokens=max_tokens, fast=fast,
        )
        elapsed = time.monotonic() - t0
        after = dict(self._inner.usage or {})
        payload = {
            "schema": tag,
            "fast": fast,
            "prompt_chars": len(system) + len(user),
            "calls": after.get("calls", 0) - before.get("calls", 0),
            "prompt_tokens": after.get("prompt_tokens", 0) - before.get("prompt_tokens", 0),
            "completion_tokens": after.get("completion_tokens", 0) - before.get("completion_tokens", 0),
        }
        self._emitter.emit("llm_call", "provider", payload)
        if elapsed > self._slow_seconds:
            self._emitter.emit(
                "slow_node", "provider",
                {"schema": tag, "elapsed_seconds": round(elapsed, 1), "threshold": self._slow_seconds},
            )
        return out


def build_deps(
    settings: Settings,
    store: RunStore,
    run_id: str,
    provider: BaseLLMProvider | None = None,
    retriever: BaseRetriever | None = None,
    max_rounds: int | None = None,
    extra_sinks: list[SinkFn] | None = None,
    http: httpx.AsyncClient | None = None,
    on_events_persisted: Callable[[str], None] | None = None,
    resume: bool = False,
) -> ResearchDeps:
    """组装一次 run 的全部依赖（provider/retriever/http 可注入以便测试）。"""
    from deepdossier.llm.provider import build_embedder, build_provider
    from deepdossier.tools.search import build_retriever

    writer = EventWriter(store, on_persisted=on_events_persisted)
    writer.start()
    trace_file = JsonlTrace(settings.traces_dir / f"{run_id}.jsonl")
    emitter = EventEmitter(
        run_id,
        sinks=[writer.submit, trace_file, *(extra_sinks or [])],
        seq_start=store.max_event_seq(run_id) if resume else 0,
    )
    final_provider = provider or build_provider(settings)
    evented = _EventedProvider(final_provider, emitter, slow_call_seconds=settings.slow_call_seconds)
    return ResearchDeps(
        settings=settings,
        provider=evented,
        retriever=retriever or build_retriever(settings, provider=evented),
        http=http if http is not None else httpx.AsyncClient(timeout=30),
        store=store,
        emitter=emitter,
        max_rounds=max_rounds or settings.max_research_rounds,
        embedder=build_embedder(settings),
        event_writer=writer,
        trace_file=trace_file,
        owns_http=http is None,
        owns_provider=provider is None,
        owns_retriever=retriever is None,
    )


async def _partial_from_checkpoint(checkpointer, run_id: str, question: str) -> str | None:
    """R3/R14：run 失败/超时时从最近 checkpoint 抢救已收集的状态，生成"部分成果"报告。

    让用户在失败时也拿得回证据，而不是两手空空。无 checkpoint/无内容时返回 None。
    """
    if checkpointer is None:
        return None
    try:
        tpl = await checkpointer.aget_tuple({"configurable": {"thread_id": run_id}})
        if not tpl:
            return None
        vals = (tpl.checkpoint or {}).get("channel_values", {}) or {}
        evidence = vals.get("evidence") or []
        plan = vals.get("plan") or {}
        if not evidence and not plan:
            return None
        lines = ["# 部分成果（研究中断前已收集）", ""]
        title = plan.get("title")
        if title:
            lines += [f"**调研主题**：{title}", ""]
        subqs = plan.get("subquestions") or []
        if subqs:
            lines += ["**研究计划**：", ""]
            lines += [f"- [{s.get('id')}] {s.get('question')}（{s.get('perspective', '')}）" for s in subqs]
            lines.append("")
        if evidence:
            lines += [f"**已收集证据（{len(evidence)} 条）**：", ""]
            for c in evidence[:80]:
                lines.append(
                    f"- [{c.get('question_id')}] {c.get('claim', '')}"
                    f"（来源：{c.get('source_title') or c.get('source_url', '')}）"
                )
            lines.append("")
        lines += [
            "> 本 run 因中断未能产出完整报告；以上为中断前已落盘的证据与计划。"
            "可点击「从断点续跑」恢复研究。"
        ]
        return "\n".join(lines)
    except Exception:  # noqa: BLE001 — 抢救失败不影响原错误路径
        logger.warning("run %s 部分成果抢救失败", run_id, exc_info=True)
        return None


async def run_research(
    question: str,
    settings: Settings,
    store: RunStore,
    run_id: str,
    max_rounds: int | None = None,
    provider: BaseLLMProvider | None = None,
    retriever: BaseRetriever | None = None,
    extra_sinks: list[SinkFn] | None = None,
    http: httpx.AsyncClient | None = None,
    on_events_persisted: Callable[[str], None] | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """执行一次完整研究。run 记录由调用方先创建（API 需要立即返回 run_id）。

    resume=True 时：不重建初始状态，从最近 checkpoint 继续（thread_id=run_id）。
    """
    # checkpointer 先于 deps 打开：resume 守卫失败时不会创建事件写线程/trace 文件
    checkpointer = await open_checkpointer(settings)
    if resume and checkpointer is None:
        raise RuntimeError("断点续跑不可用（checkpointing 已关闭或 checkpoint 初始化失败，如文件损坏）；可通过 fresh=true 放弃断点重跑")
    deps = build_deps(
        settings, store, run_id,
        provider=provider, retriever=retriever, max_rounds=max_rounds,
        extra_sinks=extra_sinks, http=http, on_events_persisted=on_events_persisted,
        resume=resume,
    )
    logger.info("run %s 开始%s: %s", run_id, "（断点续跑）" if resume else "", question)
    config = {"configurable": {"thread_id": run_id}} if checkpointer is not None else None
    try:
        graph = build_graph(deps, checkpointer=checkpointer)
        if resume:
            final: dict[str, Any] = await graph.ainvoke(None, config=config)
        else:
            final = await graph.ainvoke({"run_id": run_id, "question": question}, config=config)
        report_md = final.get("report_md", "")
        sources = final.get("sources", [])
        deps.emitter.emit("done", "runner", {})

        # 沉淀策略记忆（借鉴 TradingAgents 反思记忆）；直接回答（无来源）不沉淀
        if report_md and sources:
            with contextlib.suppress(Exception):
                query_vec = await embed_question(deps.embedder, deps.store, question)
                lesson = await deps.provider.complete(
                    system=LESSON_SYSTEM,
                    user=f"调研问题：{question}\n本轮共 {len(final.get('evidence', []))} 条证据，"
                    f"执行了 {final.get('round', 1)} 轮，评审结论：{final.get('critique', {}).get('reasoning', '')}",
                    fast=True,
                )
                if isinstance(lesson, str) and lesson.strip():
                    await asyncio.to_thread(
                        store.add_lesson, question, lesson.strip()[:300], query_vec
                    )

        await deps.aclose()  # 先排空事件，再置终态（SSE 依赖该顺序）
        await asyncio.to_thread(
            store.finish_run, run_id, report_md or None, None, deps.provider.usage, sources
        )
        logger.info("run %s 完成（%d 条证据）", run_id, len(final.get("evidence", [])))
        if final.get("route") == "direct":
            # N5：Web 端区分"直接回答"与"完整调研"两类 run
            with contextlib.suppress(Exception):
                await asyncio.to_thread(store.update_run_params, run_id, {"kind": "direct"})
        return {
            "run_id": run_id,
            "status": "completed",
            "report_md": report_md,
            "sources": sources,
            "evidence": final.get("evidence", []),
        }
    except asyncio.CancelledError:
        logger.warning("run %s 被取消/超时", run_id)
        with contextlib.suppress(Exception):
            deps.emitter.emit("error", "runner", {"message": "run cancelled or timed out"})
            await deps.aclose()
        with contextlib.suppress(Exception):
            partial = await _partial_from_checkpoint(checkpointer, run_id, question)
            await asyncio.to_thread(
                store.finish_run, run_id, partial, "cancelled or timed out", deps.provider.usage
            )
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("run %s 失败", run_id)
        with contextlib.suppress(Exception):
            deps.emitter.emit("error", "runner", {"message": str(e)[:500]})
            await deps.aclose()
        with contextlib.suppress(Exception):
            partial = await _partial_from_checkpoint(checkpointer, run_id, question)
        await asyncio.to_thread(store.finish_run, run_id, partial, str(e)[:1000], deps.provider.usage)
        raise
    finally:
        await close_checkpointer(checkpointer)
