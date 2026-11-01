"""事件与追踪体系。

所有 Agent 节点通过 EventEmitter 发事件，落点到：
- EventWriter（后台线程批量写 SQLite events 表，见 store/writer.py）
- JsonlTrace（单句柄 JSONL 全链路 trace，自托管，借鉴 OpenAI Agents SDK tracing 思想）
- rich Console（CLI 实时显示）
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Event:
    seq: int
    run_id: str
    type: str  # node_start | node_end | llm_call | tool_call | evidence_added | critique | final | error | done
    node: str
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "run_id": self.run_id,
            "type": self.type,
            "node": self.node,
            "payload": self.payload,
            "ts": self.ts,
        }


SinkFn = Callable[[Event], None]


class EventEmitter:
    """每个 run 一个实例；seq 单调递增，供断点恢复与 SSE 增量拉取。

    seq_start：断点续跑时从已有最大 seq 续接（events 表主键是 (run_id, seq)，
    从 0 重来会覆盖历史事件）。
    """

    def __init__(self, run_id: str, sinks: list[SinkFn] | None = None, seq_start: int = 0) -> None:
        self.run_id = run_id
        self._sinks = list(sinks or [])
        self._seq = seq_start

    def add_sink(self, sink: SinkFn) -> None:
        self._sinks.append(sink)

    def emit(self, type: str, node: str, payload: dict[str, Any] | None = None) -> Event:
        self._seq += 1
        event = Event(seq=self._seq, run_id=self.run_id, type=type, node=node, payload=payload or {})
        for sink in self._sinks:
            try:
                sink(event)
            except Exception:  # 观测层故障不阻断主流程
                pass
        return event


class JsonlTrace:
    """单 run 的 JSONL trace：构造时打开句柄、close 时关闭，避免每事件 open/close。"""

    def __init__(self, path: Path) -> None:
        self._f = open(path, "a", encoding="utf-8")

    def __call__(self, event: Event) -> None:
        self._f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        self._f.flush()

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:  # noqa: BLE001 — 清理路径不抛
            pass


def console_sink(verbose: bool = True) -> SinkFn:
    """CLI 控制台输出（面向用户的事件流，非服务日志）。"""

    ICONS = {
        "node_start": "▶",
        "node_end": "✔",
        "llm_call": "🧠",
        "tool_call": "🔍",
        "evidence_added": "📎",
        "critique": "⚖",
        "final": "📄",
        "error": "✖",
        "done": "🏁",
    }

    def _sink(event: Event) -> None:
        if not verbose:
            return
        icon = ICONS.get(event.type, "·")
        summary = ""
        p = event.payload
        if event.type == "node_start":
            summary = str(p.get("detail", ""))
        elif event.type == "evidence_added":
            summary = p.get("question_id", "")
        elif event.type == "error":
            summary = str(p.get("message", ""))[:120]
        print(f"  {icon} [{event.node}] {event.type} {summary}".rstrip())

    return _sink
