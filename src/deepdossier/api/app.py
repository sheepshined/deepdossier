"""FastAPI 服务：创建研究任务 + SSE 事件流 + 报告读取。

工程要点：
- 鉴权覆盖所有 /api/* 端点（含 events），常量时间比较（secrets.compare_digest）
- TaskManager 管理后台 run：并发上限、硬超时、优雅关停（替代裸 create_task）
- SSE：EventHub 即时唤醒（事件落库回调）+ SQLite 回放（断线重连）双通道
- 静态首页用 importlib.resources 读包内资源，源码运行与 pip 安装/Docker 行为一致
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import secrets
import time
from contextlib import asynccontextmanager
from importlib import resources
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from deepdossier import __version__
from deepdossier.api.hub import EventHub
from deepdossier.api.ratelimit import RateLimiter
from deepdossier.api.tasks import TaskManager
from deepdossier.config import Settings, load_settings
from deepdossier.report_docx import render_docx
from deepdossier.report_export import render_export_html
from deepdossier.runner import run_research
from deepdossier.store.runs import RunStore

logger = logging.getLogger("deepdossier.api")


class CreateResearchRequest(BaseModel):
    # min_length=1：问题路由（router）本身就是为"你好/你是谁"这类短输入设计的
    question: str = Field(min_length=1, max_length=2000)
    max_rounds: int | None = Field(default=None, ge=1, le=5)


class CompareRequest(BaseModel):
    run_id_a: str = Field(min_length=1)
    run_id_b: str = Field(min_length=1)


def _web_index_path() -> str | None:
    """包内 web/index.html 的真实文件系统路径；兼容源码运行与 pip 安装/Docker 布局。"""
    try:
        res = resources.files("deepdossier") / "web" / "index.html"
        path = Path(str(res))
        return str(path) if path.is_file() else None
    except Exception:  # noqa: BLE001 — 静态资源缺失时退化为 JSON 响应
        return None


def create_app(settings: Settings | None = None, store: RunStore | None = None) -> FastAPI:
    settings = settings or load_settings()
    store = store or RunStore(settings.db_path)

    hub = EventHub()
    taskman = TaskManager(
        store,
        max_concurrent=settings.max_concurrent_runs,
        run_timeout=settings.run_timeout_seconds,
    )
    limiter = RateLimiter(settings.rate_limit_per_minute) if settings.rate_limit_per_minute > 0 else None

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        # ---- 启动 ----
        hub.bind_loop(asyncio.get_running_loop())
        stale = store.fail_stale_running("interrupted by server restart")  # F20：清理僵尸 running
        store.prune(settings.retention_days, settings.max_lessons)  # 每次启动执行保留策略
        with contextlib.suppress(Exception):
            import shutil

            free_mb = shutil.disk_usage(settings.data_dir).free / 1e6
            if free_mb < 500:
                logger.warning("磁盘剩余空间不足 %.0fMB（<500MB），事件/报告写入可能失败，请清理磁盘", free_mb)
        logger.info(
            "DeepDossier %s 启动（retriever=%s, provider=%s, 主模型=%s, 快模型=%s%s）",
            __version__, settings.retriever, settings.llm_provider,
            settings.llm_model,
            f"{settings.fast_model}（未单独配置，自动回退主模型）" if not settings.llm_fast_model else settings.fast_model,
            f"，清理僵尸 run {stale} 个" if stale else "",
        )
        if not settings.llm_api_key and settings.llm_provider == "openai_compatible":
            logger.warning(
                "LLM API key 为空——真实调研将失败（401）。请确认 .env 位置（服务工作目录或项目根）与 DOSSIER_LLM_API_KEY 配置。"
            )
        yield
        # ---- 关停：取消后台任务、关闭共享 provider 与存储连接 ----
        await taskman.shutdown()
        with contextlib.suppress(Exception):
            await compare_provider.aclose()
        with contextlib.suppress(Exception):
            store.close()

    app = FastAPI(title="DeepDossier", version=__version__, description="深度调研智能体", lifespan=_lifespan)
    app.state.settings = settings
    app.state.store = store
    app.state.hub = hub
    app.state.taskman = taskman
    # 应用级共享 provider：报告对比等非 run 生命周期的 LLM 调用复用（关停时统一释放）
    from deepdossier.llm.provider import build_provider

    compare_provider = build_provider(settings)
    app.state.compare_provider = compare_provider

    limiter = RateLimiter(settings.rate_limit_per_minute) if settings.rate_limit_per_minute > 0 else None

    if settings.cors_origin_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origin_list,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["X-API-Key", "Content-Type"],
        )

    # ---- 鉴权 ----

    async def _auth(x_api_key: str | None = Header(default=None)) -> None:
        if settings.api_key:
            given = x_api_key or ""
            if not (secrets.compare_digest(given, settings.api_key)):
                raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")

    # ---- 创建任务 ----

    @app.post("/api/researches", status_code=202, dependencies=[Depends(_auth)])
    async def create_research(
        body: CreateResearchRequest,
        request: Request,
        x_api_key: str | None = Header(default=None),
    ):
        if limiter is not None:
            client_key = x_api_key or (request.client.host if request.client else "anon")
            if not limiter.allow(client_key):
                raise HTTPException(status_code=429, detail="rate limit exceeded")
        if taskman.active_count >= settings.max_concurrent_runs * 8:
            raise HTTPException(status_code=503, detail="server overloaded, retry later")

        run_id = store.new_run_id()
        store.create_run(run_id, body.question, params={"max_rounds": body.max_rounds})

        async def _job() -> None:
            await run_research(
                question=body.question,
                settings=settings,
                store=store,
                run_id=run_id,
                max_rounds=body.max_rounds,
                on_events_persisted=hub.notify_persisted,
            )

        if settings.inline_execution:
            await _job()
            return {"run_id": run_id, "status": "completed"}
        queued_before = taskman.waiting_count
        taskman.spawn(run_id, _job)
        resp = {"run_id": run_id, "status": "accepted", **taskman.stats()}
        if queued_before > 0 or taskman.active_count > taskman.max_concurrent:
            resp["queued"] = True
        return resp

    @app.post("/api/researches/{run_id}/resume", status_code=202, dependencies=[Depends(_auth)])
    async def resume_research(run_id: str, fresh: bool = False):
        """断点续跑：从最近 checkpoint 恢复失败的 run。

        fresh=true（R12）：放弃断点重新完整跑一遍——checkpoint 损坏/不可用时的退路。
        """
        run = store.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="run not found")
        if run["status"] == "running":
            raise HTTPException(status_code=409, detail="run is already running")
        if run["status"] != "failed":
            raise HTTPException(status_code=409, detail="only failed runs can be resumed")
        if fresh:
            store.clear_events(run_id)  # 事件从 0 重放，放弃断点
        elif not settings.checkpointing:
            raise HTTPException(status_code=409, detail="checkpointing is disabled")
        store.reopen_run(run_id)

        params = run.get("params") or {}

        async def _job() -> None:
            await run_research(
                question=run["question"],
                settings=settings,
                store=store,
                run_id=run_id,
                max_rounds=params.get("max_rounds"),
                resume=bool(fresh is False and settings.checkpointing),
                on_events_persisted=hub.notify_persisted,
            )

        taskman.spawn(run_id, _job)
        return {"run_id": run_id, "status": "resumed", "fresh": fresh}

    # ---- 查询 ----

    @app.get("/api/researches/{run_id}", dependencies=[Depends(_auth)])
    async def get_research(run_id: str):
        run = store.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="run not found")
        return run

    @app.get("/api/researches/{run_id}/export", dependencies=[Depends(_auth)])
    async def export_report(run_id: str):
        """报告独立 HTML 导出：浏览器直开可读，Ctrl+P 即可存 PDF。"""
        run = store.get_run(run_id)
        if not run or not run.get("report_md"):
            raise HTTPException(status_code=404, detail="report not found")
        try:
            html = render_export_html(run, version=__version__)
        except Exception as e:  # noqa: BLE001 — R20：渲染失败降级为 Markdown + 错误说明
            logger.exception("HTML 导出失败 run=%s", run_id)
            return _markdown_fallback(run, f"HTML 渲染失败: {type(e).__name__}")
        return Response(
            content=html,
            media_type="text/html; charset=utf-8",
            headers={"Content-Disposition": f'inline; filename="deepdossier-{run_id}.html"'},
        )

    @app.get("/api/researches/{run_id}/export.docx", dependencies=[Depends(_auth)])
    async def export_report_docx(run_id: str):
        """报告 Word 导出（.docx）。"""
        run = store.get_run(run_id)
        if not run or not run.get("report_md"):
            raise HTTPException(status_code=404, detail="report not found")
        try:
            data = render_docx(run, version=__version__)
        except Exception as e:  # noqa: BLE001 — R20：生成失败降级为 Markdown + 错误说明
            logger.exception("DOCX 导出失败 run=%s", run_id)
            return _markdown_fallback(run, f"DOCX 生成失败: {type(e).__name__}")
        return Response(
            content=data,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f'attachment; filename="deepdossier-{run_id}.docx"'},
        )

    @app.get("/api/runs", dependencies=[Depends(_auth)])
    async def list_runs():
        return store.list_runs()

    @app.post("/api/compare", dependencies=[Depends(_auth)])
    async def compare_reports(body: CompareRequest):
        """两份报告的 LLM 对比评审；同一对 run 的结果持久化缓存，不重复计费。"""
        if body.run_id_a == body.run_id_b:
            raise HTTPException(status_code=400, detail="cannot compare a run with itself")
        run_a = store.get_run(body.run_id_a)
        run_b = store.get_run(body.run_id_b)
        for r, tag in ((run_a, "a"), (run_b, "b")):
            if not r or not r.get("report_md"):
                raise HTTPException(status_code=404, detail=f"run {tag} not found or has no report")

        cached = store.get_comparison(body.run_id_a, body.run_id_b)
        if cached:
            verdict = cached
        else:
            from deepdossier.agents.prompts import COMPARE_SYSTEM, COMPARE_USER

            def _meta(r: dict) -> str:
                dur = (r.get("finished_at") or 0) - (r.get("created_at") or 0) if r.get("finished_at") else 0
                return f"{time.strftime('%m-%d %H:%M', time.localtime(r.get('created_at')))} · {len(r.get('sources') or [])} 来源 · {dur:.0f}s"

            verdict = await app.state.compare_provider.complete(
                system=COMPARE_SYSTEM,
                user=COMPARE_USER.format(
                    question=run_a["question"],
                    meta_a=_meta(run_a),
                    meta_b=_meta(run_b),
                    report_a=run_a["report_md"][:12000],
                    report_b=run_b["report_md"][:12000],
                ),
                temperature=0.2,
            )
            if not isinstance(verdict, str) or not verdict.strip():
                raise HTTPException(status_code=502, detail="empty comparison verdict")
            store.save_comparison(body.run_id_a, body.run_id_b, verdict)

        return {
            "verdict_md": verdict,
            "cached": cached is not None,
            "a": {"id": body.run_id_a, "question": run_a["question"], "created_at": run_a["created_at"], "sources": len(run_a.get("sources") or [])},
            "b": {"id": body.run_id_b, "question": run_b["question"], "created_at": run_b["created_at"], "sources": len(run_b.get("sources") or [])},
        }

    def _markdown_fallback(run: dict, reason: str) -> Response:
        """R20：导出异常时返回原始 Markdown + 明确错误说明（200，不裸 500）。"""
        body = (
            "# 导出失败说明\n\n"
            f"报告导出组件出现异常（{reason}），以下为原始 Markdown 报告内容，未受影响：\n\n---\n\n"
            + run.get("report_md", "")
        )
        return Response(
            content=body,
            media_type="text/markdown; charset=utf-8",
            headers={"X-Export-Fallback": "1"},
        )

    # ---- SSE 事件流 ----

    @app.get("/api/researches/{run_id}/events", dependencies=[Depends(_auth)])
    async def stream_events(run_id: str, after_seq: int = 0):
        if not store.get_run(run_id):
            raise HTTPException(status_code=404, detail="run not found")

        async def gen():
            cursor = after_seq
            wakeup = hub.register(run_id)
            try:
                while True:
                    events = await asyncio.to_thread(store.list_events, run_id, cursor)
                    for ev in events:
                        cursor = ev["seq"]
                        yield f"id: {ev['seq']}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"
                    run = await asyncio.to_thread(store.get_run, run_id)
                    finished = run and run["status"] != "running"
                    if finished and not events:
                        yield 'data: {"type": "stream_end"}\n\n'
                        return
                    if not finished:
                        # 等新事件通知（最多 5s 醒一次兜底轮询，防通知丢失）
                        with contextlib.suppress(TimeoutError):
                            await asyncio.wait_for(wakeup.wait(), timeout=5.0)
                        wakeup.clear()
            finally:
                hub.unregister(run_id, wakeup)

        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    # ---- 基础 ----

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "version": __version__}

    @app.get("/")
    async def index():
        path = _web_index_path()
        if path:
            return FileResponse(path)
        return {"name": "DeepDossier", "version": __version__, "docs": "/docs"}

    return app
