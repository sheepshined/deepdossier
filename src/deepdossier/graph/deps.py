"""节点共享依赖容器（每个 run 构建一份）。

包含：LLM provider / 检索器 / http 客户端（可注入）/ 事件发射器 /
EventWriter 后台写线程 / JSONL trace / 检索与抓取缓存（补搜轮增量执行）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import httpx

from deepdossier.llm.provider import BaseEmbedder, BaseLLMProvider
from deepdossier.tools.search import BaseRetriever
from deepdossier.trace import EventEmitter, JsonlTrace, SinkFn

if TYPE_CHECKING:
    from deepdossier.config import Settings
    from deepdossier.store.runs import RunStore
    from deepdossier.store.writer import EventWriter


@dataclass
class ResearchDeps:
    settings: Settings
    provider: BaseLLMProvider
    retriever: BaseRetriever
    http: httpx.AsyncClient
    store: RunStore
    emitter: EventEmitter
    max_rounds: int = 2
    embedder: BaseEmbedder | None = None
    event_writer: EventWriter | None = None
    trace_file: JsonlTrace | None = None
    owns_http: bool = True
    extra_sinks: list[SinkFn] = field(default_factory=list)
    # 单 run 级缓存：query → 搜索结果；url → (正文, 错误)
    search_cache: dict = field(default_factory=dict)
    fetch_cache: dict = field(default_factory=dict)

    async def aclose(self) -> None:
        """释放资源；事件写线程先排空落库，保证事件先于终态可读（SSE 依赖此顺序）。"""
        close_retriever = getattr(self.retriever, "aclose", None)  # MCP 检索器持有会话
        if close_retriever:
            import contextlib

            with contextlib.suppress(Exception):
                await close_retriever()
        if self.event_writer:
            self.event_writer.close()
        if self.trace_file:
            self.trace_file.close()
        if self.owns_http:
            await self.http.aclose()
