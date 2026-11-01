"""后台事件写线程：emit() 只入队，SQLite 批量落库不阻塞事件循环。

落库成功后回调 on_persisted(run_id)，供 SSE EventHub 即时唤醒订阅者。
R16：落库失败不丢事件——失败批次进入有界内存缓冲（≤2000 条），后续循环重试；
缓冲满则丢弃最旧并告警。写线程永不拖垮主流程。
"""

from __future__ import annotations

import collections
import logging
import queue
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepdossier.store.runs import RunStore
    from deepdossier.trace import Event

logger = logging.getLogger("deepdossier.events")

_RETRY_BUFFER_MAX = 2000


class EventWriter(threading.Thread):
    """单 run 一个实例；close() 排空队列并 join，保证事件先于终态落库。"""

    def __init__(
        self,
        store: RunStore,
        on_persisted: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(daemon=True, name="dd-event-writer")
        self._store = store
        self._on_persisted = on_persisted
        self._q: queue.SimpleQueue = queue.SimpleQueue()
        self._closed = False
        self._retry: collections.deque = collections.deque()

    def submit(self, event: Event) -> None:
        if not self._closed:
            self._q.put(event)

    def _persist(self, batch: list) -> bool:
        try:
            self._store.append_events(batch)
        except Exception:  # noqa: BLE001 — R16：写失败进入重试缓冲，主流程不受影响
            logger.exception("事件批量落库失败（%d 条），进入内存重试缓冲", len(batch))
            self._retry.extend(batch)
            while len(self._retry) > _RETRY_BUFFER_MAX:
                self._retry.popleft()
                logger.warning("事件重试缓冲溢出，丢弃最旧事件")
            return False
        if self._on_persisted:
            for run_id in dict.fromkeys(e.run_id for e in batch):
                try:
                    self._on_persisted(run_id)
                except Exception:  # noqa: BLE001
                    pass
        return True

    def run(self) -> None:
        stop = False
        while not stop:
            # R16：先重试之前失败的批次；仍失败则短暂退避（不阻塞新事件入队）
            if self._retry:
                batch = list(self._retry)
                self._retry.clear()
                if not self._persist(batch):  # 失败批次由 _persist 放回缓冲
                    if self._closed:
                        break  # 关停时 store 持续故障：放弃重试退出，不卡 close()
                    time.sleep(1.0)
                continue
            item = self._q.get()
            if item is None:
                break
            batch: list = [item]
            while True:
                try:
                    nxt = self._q.get_nowait()
                except queue.Empty:
                    break
                if nxt is None:
                    stop = True
                    break
                batch.append(nxt)
            self._persist(batch)

    def close(self, timeout: float = 5.0) -> None:
        self._closed = True
        self._q.put(None)
        self.join(timeout)
        if self.is_alive():
            logger.warning("事件写线程未能及时退出（超时 %.1fs）", timeout)
