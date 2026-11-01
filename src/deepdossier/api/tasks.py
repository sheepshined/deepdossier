"""后台研究任务管理器：替代裸 asyncio.create_task。

修复三件事：
- 持有任务强引用（裸 fire-and-forget 的任务可能被 GC）
- 全局信号量限制并发 run 数（防止 N 请求 = N 张图同时跑）
- 单 run 硬超时；超时/取消时把 run 置为 failed，不会永久卡在 running
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from deepdossier.store.runs import RunStore

logger = logging.getLogger("deepdossier.tasks")


class TaskManager:
    def __init__(self, store: RunStore, max_concurrent: int = 4, run_timeout: float = 600.0) -> None:
        self._store = store
        self._sem = asyncio.Semaphore(max_concurrent)
        self._timeout = run_timeout
        self._tasks: dict[str, asyncio.Task] = {}
        self._max_concurrent = max_concurrent
        self._waiting = 0  # R15：排队中的任务数（信号量等待）

    @property
    def active_count(self) -> int:
        return len(self._tasks)

    @property
    def waiting_count(self) -> int:
        return self._waiting

    @property
    def max_concurrent(self) -> int:
        return self._max_concurrent

    def stats(self) -> dict[str, int]:
        """R15：队列状态（前端展示"排队中 N/M"）。"""
        return {
            "active": self.active_count,
            "waiting": self._waiting,
            "max_concurrent": self._max_concurrent,
        }

    def spawn(self, run_id: str, factory: Callable[[], Coroutine[Any, Any, None]]) -> None:
        """factory 在信号量获取后才创建并执行协程。"""
        task = asyncio.create_task(self._guarded(run_id, factory), name=f"research-{run_id}")
        self._tasks[run_id] = task
        task.add_done_callback(lambda _t: self._tasks.pop(run_id, None))

    async def _guarded(self, run_id: str, factory: Callable[[], Coroutine[Any, Any, None]]) -> None:
        try:
            self._waiting += 1
            try:
                await self._sem.acquire()
            finally:
                self._waiting -= 1
            try:
                await asyncio.wait_for(factory(), timeout=self._timeout)
            finally:
                self._sem.release()
        except asyncio.TimeoutError:
            logger.error("run %s 超时（%.0fs），标记失败", run_id, self._timeout)
            await asyncio.to_thread(
                self._store.finish_run, run_id, None, f"run timed out after {self._timeout:.0f}s", None
            )
        except asyncio.CancelledError:
            # run_research 的 CancelledError 分支已落终态，这里兜底幂等
            await asyncio.to_thread(
                self._store.finish_run, run_id, None, "cancelled by server shutdown", None
            )
        except Exception:  # noqa: BLE001 — run_research 已自行落 failed 并 re-raise
            logger.exception("后台 run %s 异常退出", run_id)

    async def shutdown(self) -> None:
        tasks = list(self._tasks.values())
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("任务管理器已关闭（取消 %d 个运行中任务）", len(tasks))
