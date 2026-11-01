"""SSE 事件通知中心：事件落库后即时唤醒订阅连接。

此前 SSE 每 0.5s 轮询 SQLite；现在 EventWriter 落库成功后回调 notify_persisted，
经 call_soon_threadsafe 唤醒该 run 的所有订阅者（断线重连仍靠 SQLite 回放）。
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger("deepdossier.hub")


class EventHub:
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._subs: dict[str, list[asyncio.Event]] = {}

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def register(self, run_id: str) -> asyncio.Event:
        ev = asyncio.Event()
        self._subs.setdefault(run_id, []).append(ev)
        return ev

    def unregister(self, run_id: str, ev: asyncio.Event) -> None:
        subs = self._subs.get(run_id)
        if subs and ev in subs:
            subs.remove(ev)
            if not subs:
                self._subs.pop(run_id, None)

    def notify_persisted(self, run_id: str) -> None:
        """事件落库回调（可能在写线程中调用）；无 loop（如测试直连）时静默跳过。"""
        if self._loop is None:
            return

        def _set() -> None:
            for ev in self._subs.get(run_id, []):
                ev.set()

        try:
            self._loop.call_soon_threadsafe(_set)
        except RuntimeError:  # loop 已关闭（进程退出）
            pass
