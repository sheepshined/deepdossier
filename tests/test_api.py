"""API 层测试（inline_execution=true，全离线）。"""

from __future__ import annotations

import httpx

from deepdossier.api.app import create_app
from deepdossier.llm.provider import StubProvider
from deepdossier.tools.search import StubRetriever


def _patched_app(settings, store):
    """注入 stub 双件套（inline 模式下 API 直接可用）。"""
    import deepdossier.runner as runner_mod

    real_build_deps = runner_mod.build_deps

    def patched(settings_, store_, run_id, **kw):
        kw.setdefault("provider", StubProvider())
        kw.setdefault("retriever", StubRetriever())
        return real_build_deps(settings_, store_, run_id, **kw)

    runner_mod.build_deps = patched
    try:
        return create_app(settings, store)
    finally:
        runner_mod.build_deps = real_build_deps


async def test_create_and_stream_research(settings, store):
    app = _patched_app(settings, store)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # 创建任务（inline 执行，返回时已完成）
        resp = await client.post("/api/researches", json={"question": "API 测试问题", "max_rounds": 2})
        assert resp.status_code == 202
        run_id = resp.json()["run_id"]

        # 读取 run（usage/sources 已反序列化）
        run = (await client.get(f"/api/researches/{run_id}")).json()
        assert run["status"] == "completed"
        assert "调研报告" in run["report_md"]
        assert run["sources"] and run["sources"][0]["url"]
        assert run["usage"] is not None and run["usage"]["calls"] > 0

        # SSE 事件流可回放
        async with client.stream("GET", f"/api/researches/{run_id}/events") as sse:
            body = (await sse.aread()).decode("utf-8")
        assert "node_start" in body and "stream_end" in body

        # 健康检查 & 首页（包内静态资源）
        assert (await client.get("/healthz")).json()["status"] == "ok"
        index = await client.get("/")
        assert index.status_code == 200
        assert "Deep" in index.text  # HTML 首页而非 JSON 退化

        # 列表
        runs = (await client.get("/api/runs")).json()
        assert any(r["id"] == run_id for r in runs)


async def test_api_auth_all_endpoints(settings, store):
    """鉴权必须覆盖全部 /api/* 端点（含 events —— 此前的 P0 疏漏回归）。"""
    settings.api_key = "secret123"
    app = _patched_app(settings, store)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # 无 key：全部 401
        assert (await client.post("/api/researches", json={"question": "鉴权测试问题"})).status_code == 401
        assert (await client.get("/api/researches/xxx")).status_code == 401
        assert (await client.get("/api/researches/xxx/events")).status_code == 401
        assert (await client.get("/api/runs")).status_code == 401
        # 错 key：401
        bad = {"X-API-Key": "wrong"}
        assert (await client.get("/api/runs", headers=bad)).status_code == 401
        # 对 key：通过
        ok = {"X-API-Key": "secret123"}
        assert (
            await client.post("/api/researches", json={"question": "鉴权测试问题"}, headers=ok)
        ).status_code == 202


async def test_rate_limit(settings, store):
    settings.rate_limit_per_minute = 2
    app = _patched_app(settings, store)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.post("/api/researches", json={"question": "限流第一问"})).status_code == 202
        assert (await client.post("/api/researches", json={"question": "限流第二问"})).status_code == 202
        assert (await client.post("/api/researches", json={"question": "限流第三问"})).status_code == 429


async def test_unknown_run_404(settings, store):
    app = _patched_app(settings, store)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.get("/api/researches/nope")).status_code == 404
        assert (await client.get("/api/researches/nope/events")).status_code == 404


async def test_cors_enabled(settings, store):
    settings.cors_origins = "https://example.com"
    app = _patched_app(settings, store)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.options(
            "/api/runs",
            headers={"Origin": "https://example.com", "Access-Control-Request-Method": "GET"},
        )
        assert resp.headers.get("access-control-allow-origin") == "https://example.com"


def test_web_index_path_resolves_from_package():
    """包内静态资源可解析（此前 Docker/pip 安装下退化为 JSON 的 P0 bug 回归）。

    另有 CI 的 docker job 在真实镜像里做端到端验证；本地以 wheel 打包检查兜底：
    `pip wheel .` 后 deepdossier/web/index.html 必须存在于分发包内。
    """
    from pathlib import Path

    from deepdossier.api.app import _web_index_path

    path = _web_index_path()
    assert path, "包内未找到 web/index.html（检查 pyproject 的 package-data）"
    assert Path(path).is_file() and path.endswith("index.html")
