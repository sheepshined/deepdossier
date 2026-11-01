"""进程内令牌桶限流（按 API key 或客户端 IP 维度）。"""

from __future__ import annotations

import threading
import time


class RateLimiter:
    _MAX_BUCKETS = 4096  # 防按 IP 刷 key 撑爆内存；超限清理 5 分钟未活跃的桶

    def __init__(self, per_minute: int) -> None:
        self._per_minute = per_minute
        self._buckets: dict[str, tuple[float, float]] = {}  # key → (tokens, last_ts)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            if len(self._buckets) > self._MAX_BUCKETS:
                stale = [k for k, (_, last) in self._buckets.items() if now - last > 300]
                for k in stale:
                    self._buckets.pop(k, None)
            tokens, last = self._buckets.get(key, (float(self._per_minute), now))
            tokens = min(float(self._per_minute), tokens + (now - last) * self._per_minute / 60.0)
            ok = tokens >= 1.0
            if ok:
                tokens -= 1.0
            self._buckets[key] = (tokens, now)
            return ok
