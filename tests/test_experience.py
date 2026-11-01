"""体验类验收标准自动化（文档 2.3 节）：首事件延迟 < 3s、断线重连事件零丢失。"""

from __future__ import annotations

import json
import time

import httpx
import pytest

from deepdossier.api.app import create_app
from deepdossier.config import Settings
from deepdossier.store.runs import RunStore


@pytest.fixture()
def bg_settings(tmp_path) -> Settings:
    """后台执行模式（inline=False）：POST 立即返回，run 在事件循环内异步执行。"""
    s = Settings(
        llm_provider="stub",
        retriever="stub",
        data_dir=tmp_path / "data",
        inline_execution=False,
        fetch_pages=False,
        checkpointing=False,
        mcp_servers="",
    )
    s.ensure_dirs()
    return s


async def test_first_event_latency_under_3s(bg_settings):
    """验收线：从发起请求到看到第一个 Agent 事件 < 3 秒（stub 本地）。"""
    store = RunStore(bg_settings.db_path)
    try:
        app = create_app(bg_settings, store)
        transport = httpx.ASGITransport(app=app)
        # ASGITransport 不会自动跑 lifespan；显式执行以启用 EventHub 唤醒通道（与生产 uvicorn 一致）
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                t0 = time.monotonic()
                resp = await client.post("/api/researches", json={"question": "体验延迟测试问题"})
                assert resp.status_code == 202
                run_id = resp.json()["run_id"]

                first_at = None
                async with client.stream("GET", f"/api/researches/{run_id}/events") as stream:
                    async for line in stream.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        if first_at is None:
                            first_at = time.monotonic() - t0
                        ev = json.loads(line[6:])
                        if ev.get("type") in ("node_start", "route"):
                            break
                assert first_at is not None, "未收到任何事件"
                assert first_at < 3.0, f"首事件延迟 {first_at:.2f}s 超出 3s 验收线"
    finally:
        store.close()


async def test_reconnect_seq_continuity(bg_settings):
    """验收线：SSE 断线重连后事件零丢失——seq 全程严格递增、无重复，最终 run 完成。"""
    store = RunStore(bg_settings.db_path)
    try:
        app = create_app(bg_settings, store)
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post("/api/researches", json={"question": "断线重连测试问题"})
                assert resp.status_code == 202
                run_id = resp.json()["run_id"]

                seqs: list[int] = []
                last_seq = 0

                async def read_events(after: int, max_take: int | None) -> None:
                    """读取事件直到 stream_end 或取满 max_take 条（模拟断线）。"""
                    nonlocal last_seq
                    taken = 0
                    async with client.stream(
                        "GET", f"/api/researches/{run_id}/events?after_seq={after}"
                    ) as stream:
                        async for line in stream.aiter_lines():
                            if not line.startswith("data: "):
                                continue
                            ev = json.loads(line[6:])
                            if ev.get("type") == "stream_end":
                                break
                            if "seq" in ev:
                                seqs.append(ev["seq"])
                                last_seq = max(last_seq, ev["seq"])
                                taken += 1
                                if max_take is not None and taken >= max_take:
                                    break

                # 第一次连接：只取 2 条就"断开"
                await read_events(0, max_take=2)
                assert seqs, "首次连接未收到事件"
                # 重连：after_seq 续传直到 stream_end
                await read_events(last_seq, max_take=None)
                assert len(seqs) >= 2, "重连后未收到后续事件"
                assert seqs == sorted(seqs) and len(seqs) == len(set(seqs)), "seq 存在丢失/重复/乱序"

                # run 最终完成（stub 全链路）
                run = {"status": "running"}
                for _ in range(100):
                    run = (await client.get(f"/api/researches/{run_id}")).json()
                    if run["status"] != "running":
                        break
                    time.sleep(0.05)
                assert run["status"] == "completed"
    finally:
        store.close()
