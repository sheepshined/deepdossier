"""MCP 数据源检索器（对齐 GPT-Researcher 的 hybrid research）。

两类 MCP 用法：
- search 型：单工具 `tool`（默认 search），入参 {query_key: 查询词}，返回结果列表 →
  无 keywords 时挂在 web 降级链尾（通用补充）；有 keywords 时按查询词选择性并行调用。
- tool 型（领域工具，如天气）：`call_chain` 多步调用，参数支持
  {query}=原始查询词、{prev}=上一步结果、{"$llm": "参数提取指令"}（用快模型从查询词提取，
  如"提取城市名"）→ 结果映射为单条 SearchResult（snippet=最终输出）。

配置（DOSSIER_MCP_SERVERS，兼容两种格式）：
  标准格式（Claude Desktop 风格，名字取自键）：
    {"mcpServers": {"weather-mcp": {"command": "...", "args": ["test.py"], "cwd": "..."}}}
  扁平数组：
    [{"name":"x","transport":"stdio","command":"...","args":[...],"cwd":"...","keywords":["天气"],
      "call_chain":[{"tool":"query_weather","args":{"city_code":{"$llm":"提取城市名"},"mode":1}}]},
     {"name":"y","transport":"http","url":"http://host/mcp","tool":"search"}]
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from collections.abc import Callable
from contextlib import AsyncExitStack
from typing import Any

from deepdossier.tools.search import BaseRetriever, SearchResult

logger = logging.getLogger("deepdossier.mcp")

_LLM_DIRECTIVE = "$llm"


def _results_from_text(text: str) -> list[SearchResult]:
    """把 MCP 工具返回的文本容错地映射为 SearchResult 列表。"""
    text = (text or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return [SearchResult(title=text[:80], url="", snippet=text[:500])]

    if isinstance(data, str):
        return [SearchResult(title=data[:80], url="", snippet=data[:500])]
    if isinstance(data, dict):
        items = data.get("results") or data.get("data") or [data]
    elif isinstance(data, list):
        items = data
    else:
        return []

    out: list[SearchResult] = []
    for it in items:
        if isinstance(it, str):
            out.append(SearchResult(title=it[:80], url="", snippet=it[:500]))
        elif isinstance(it, dict):
            url = str(it.get("url") or it.get("link") or "")
            title = str(it.get("title") or it.get("name") or url or "MCP 结果")[:200]
            snippet = str(
                it.get("snippet") or it.get("description") or it.get("content") or it.get("text") or ""
            )[:500]
            if url or snippet:
                out.append(SearchResult(title=title, url=url, snippet=snippet))
    return out


def _results_from_tool_result(result: Any) -> list[SearchResult]:
    out: list[SearchResult] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            out.extend(_results_from_text(text))
    return out


def _tool_result_text(result: Any) -> str:
    parts = [b.text for b in getattr(result, "content", None) or [] if isinstance(getattr(b, "text", None), str)]
    return "\n".join(parts).strip()


class McpRetriever(BaseRetriever):
    """单个 MCP Server 的检索封装：懒连接、跨查询复用会话、可 aclose。

    call_chain 非空时走"领域工具"模式（多步调用，{prev} 串联，$llm 提取参数）；
    否则走 search 模式（单工具，query_key 直传查询词）。
    """

    def __init__(
        self,
        name: str,
        config: dict[str, Any],
        session_factory=None,
        llm_fn: Callable[[str], Any] | None = None,
    ) -> None:
        self._cfg_name = name
        self.name = f"mcp:{name}"
        self._config = config
        self._session_factory = session_factory or self._default_factory
        self._llm_fn = llm_fn
        self.keywords: list[str] = [str(k).lower() for k in (config.get("keywords") or [])]
        self._call_chain: list[dict[str, Any]] = config.get("call_chain") or []
        self._session: Any = None
        self._stack: AsyncExitStack | None = None
        self._lock = asyncio.Lock()

    async def _default_factory(self, stack: AsyncExitStack, cfg: dict[str, Any]):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from mcp.client.streamable_http import streamablehttp_client

        transport = cfg.get("transport", "stdio")
        if transport == "http":
            read, write, _ = await stack.enter_async_context(streamablehttp_client(cfg["url"]))
        else:
            env = dict(os.environ)
            env.update({k: str(v) for k, v in (cfg.get("env") or {}).items()})
            params = StdioServerParameters(
                command=cfg["command"], args=list(cfg.get("args") or []),
                env=env, cwd=cfg.get("cwd"),
            )
            read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session

    async def _ensure_session(self):
        async with self._lock:
            if self._session is None:
                self._stack = AsyncExitStack()
                try:
                    self._session = await self._session_factory(self._stack, self._config)
                except Exception:
                    with contextlib.suppress(Exception):
                        await self._stack.aclose()
                    self._stack = None
                    raise
            return self._session

    # ---- 参数解析：{query} / {prev} / {"$llm": 指令} ----

    async def _resolve_args(self, args: dict[str, Any], query: str, prev: str) -> dict[str, Any]:
        resolved: dict[str, Any] = {}
        for k, v in args.items():
            if isinstance(v, dict) and _LLM_DIRECTIVE in v:
                if self._llm_fn is None:
                    raise RuntimeError(f"{self.name} 配置了 $llm 参数提取但检索器未接入 LLM")
                raw = await self._llm_fn(str(v[_LLM_DIRECTIVE]) + f"\n\n待处理文本：{query}")
                value = raw.strip() if isinstance(raw, str) else str(raw)
                resolved[k] = value.strip("\"'“”")
            elif isinstance(v, str):
                resolved[k] = v.replace("{query}", query).replace("{prev}", prev)
            else:
                resolved[k] = v
        return resolved

    async def _search_via_call_chain(self, session, query: str) -> list[SearchResult]:
        prev = ""
        for step in self._call_chain:
            args = await self._resolve_args(step.get("args") or {}, query, prev)
            result = await session.call_tool(step["tool"], args)
            prev = _tool_result_text(result)
        if not prev:
            return []
        # 领域工具输出（如天气文本）作为单条高价值摘要
        return [SearchResult(title=f"{self._cfg_name} 领域数据", url="", snippet=prev[:500])]

    async def search(self, query: str, k: int = 5) -> list[SearchResult]:
        session = await self._ensure_session()
        if self._call_chain:
            return (await self._search_via_call_chain(session, query))[:k]
        tool = self._config.get("tool", "search")
        query_key = self._config.get("query_key", "query")
        result = await session.call_tool(tool, {query_key: query})
        return _results_from_tool_result(result)[:k]

    async def aclose(self) -> None:
        if self._stack is not None:
            with contextlib.suppress(Exception):
                await self._stack.aclose()
        self._session = None
        self._stack = None


def parse_mcp_servers(raw: str) -> list[tuple[str, dict[str, Any]]]:
    """解析 DOSSIER_MCP_SERVERS：兼容标准 {"mcpServers": {...}} 与扁平数组；非法项跳过。"""
    if not raw or not raw.strip():
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("DOSSIER_MCP_SERVERS 不是合法 JSON，忽略 MCP 数据源")
        return []

    entries: list[tuple[str, dict[str, Any]]] = []
    if isinstance(data, dict) and isinstance(data.get("mcpServers"), dict):
        entries = [(str(k), v) for k, v in data["mcpServers"].items() if isinstance(v, dict)]
    elif isinstance(data, list):
        entries = [
            (str(c.get("name") or f"server{i}"), c)
            for i, c in enumerate(data)
            if isinstance(c, dict)
        ]
    else:
        return []

    servers: list[tuple[str, dict[str, Any]]] = []
    for name, cfg in entries:
        transport = cfg.get("transport", "stdio")
        if transport == "http" and not cfg.get("url"):
            logger.warning("MCP server %s 缺 url，跳过", name)
            continue
        if transport != "http" and not cfg.get("command"):
            logger.warning("MCP server %s 缺 command，跳过", name)
            continue
        if not cfg.get("call_chain") and not cfg.get("tool"):
            logger.warning("MCP server %s 未指定 tool 或 call_chain，无法调用，跳过", name)
            continue
        servers.append((name, cfg))
    return servers
