"""TaskManager 与 RateLimiter 单元测试。"""

from __future__ import annotations

import asyncio

from deepdossier.api.ratelimit import RateLimiter
from deepdossier.api.tasks import TaskManager


async def test_task_manager_runs_and_cleans_up(store):
    tm = TaskManager(store, max_concurrent=2, run_timeout=10)
    done = asyncio.Event()

    async def job():
        done.set()

    store.create_run("t1", "q")
    tm.spawn("t1", job)
    await done.wait()
    # done_callback 在事件循环下一拍移除引用
    await asyncio.sleep(0.05)
    assert tm.active_count == 0


async def test_task_manager_timeout_marks_failed(store):
    tm = TaskManager(store, max_concurrent=1, run_timeout=0.2)
    store.create_run("t-slow", "q")

    async def slow():
        await asyncio.sleep(10)

    tm.spawn("t-slow", slow)
    await asyncio.sleep(0.5)
    run = store.get_run("t-slow")
    assert run["status"] == "failed"
    assert "timed out" in run["error"]


async def test_task_manager_concurrency_cap(store):
    """并发不超过 max_concurrent。"""
    tm = TaskManager(store, max_concurrent=2, run_timeout=10)
    running = 0
    peak = 0
    lock = asyncio.Lock()

    async def job():
        nonlocal running, peak
        async with lock:
            running += 1
            peak = max(peak, running)
        await asyncio.sleep(0.1)
        async with lock:
            running -= 1

    for i in range(6):
        store.create_run(f"t{i}", "q")
        tm.spawn(f"t{i}", job)
    await asyncio.sleep(0.6)
    assert peak <= 2


async def test_task_manager_shutdown_cancels(store):
    tm = TaskManager(store, max_concurrent=2, run_timeout=10)
    store.create_run("t-kill", "q")
    cancelled = asyncio.Event()

    async def hang():
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    tm.spawn("t-kill", hang)
    await asyncio.sleep(0.05)
    await tm.shutdown()
    assert cancelled.is_set()
    run = store.get_run("t-kill")
    assert run["status"] == "failed"


def test_rate_limiter_burst_and_refill():
    rl = RateLimiter(per_minute=3)
    assert rl.allow("k")
    assert rl.allow("k")
    assert rl.allow("k")
    assert not rl.allow("k")           # 桶空
    assert not rl.allow("k")
    assert rl.allow("other")           # key 之间隔离
