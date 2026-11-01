"""可插拔搜索 Retriever 层。

默认 DuckDuckGo（免 API key，开箱即用）；So360（免 key，国内网络推荐）、
Bing（免 key，国际网络）、Tavily / Serper 可选；StubRetriever 供离线测试与演示。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from abc import ABC, abstractmethod

import httpx
from pydantic import BaseModel

logger = logging.getLogger("deepdossier.search")


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str = ""
    via: str = ""  # 贡献该结果的引擎/数据源（检索透明化），如 so360 / mcp:weather-mcp


class BaseRetriever(ABC):
    name: str = "base"

    @abstractmethod
    async def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """返回前 k 条结果；失败抛异常由调用方决定降级策略。"""

    async def aclose(self) -> None:  # noqa: B027 — 默认无持久资源需要释放
        return None


class DuckDuckGoRetriever(BaseRetriever):
    name = "duckduckgo"

    async def search(self, query: str, k: int = 5) -> list[SearchResult]:
        def _run() -> list[SearchResult]:
            from ddgs import DDGS

            raw = DDGS().text(query, max_results=k)
            return [
                SearchResult(title=r.get("title", ""), url=r.get("href", ""), snippet=r.get("body", ""))
                for r in raw
            ]

        return await asyncio.to_thread(_run)


class So360Retriever(BaseRetriever):
    """360 搜索（so.com）网页检索：免 key，国内直连可达、中文相关性好。

    结果的 data-mdurl 属性携带真实目标 URL（缺失时回退为站内跳转链接）。
    """

    name = "so360"

    _UA = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client  # 测试可注入 MockTransport

    async def search(self, query: str, k: int = 5) -> list[SearchResult]:
        own = self._client is None
        client = self._client or httpx.AsyncClient(
            timeout=20,
            follow_redirects=True,
            headers={"User-Agent": self._UA, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"},
        )
        try:
            resp = await client.get("https://www.so.com/s", params={"q": query, "rn": max(k, 10)})
            resp.raise_for_status()
            return _parse_so360_html(resp.text, k)
        finally:
            if own:
                await client.aclose()


def _parse_so360_html(html: str, k: int) -> list[SearchResult]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    out: list[SearchResult] = []
    for li in soup.select("li.res-list")[:k]:
        a = li.select_one("h3 a[href]")
        if not a:
            continue
        snippet_el = li.select_one("p.res-desc, p.res-rich") or li.select_one("p")
        out.append(
            SearchResult(
                title=a.get_text(" ", strip=True),
                url=a.get("data-mdurl") or a["href"],
                snippet=snippet_el.get_text(" ", strip=True)[:500] if snippet_el else "",
            )
        )
    return out


class BingRetriever(BaseRetriever):
    """Bing 网页检索：免 key，国内网络直连可达（DDG/Brave/Google 常超时）。

    解析 cn.bing.com 返回的 b_algo 自然结果；需浏览器 UA，否则被反爬拦截。
    """

    name = "bing"

    _UA = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client  # 测试可注入 MockTransport

    async def search(self, query: str, k: int = 5) -> list[SearchResult]:
        own = self._client is None
        client = self._client or httpx.AsyncClient(
            timeout=20,
            follow_redirects=True,  # cn.bing.com 会 302 到区域站
            headers={"User-Agent": self._UA, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"},
        )
        try:
            resp = await client.get("https://cn.bing.com/search", params={"q": query, "count": max(k, 10)})
            resp.raise_for_status()
            return _parse_bing_html(resp.text, k)
        finally:
            if own:
                await client.aclose()


def _parse_bing_html(html: str, k: int) -> list[SearchResult]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    out: list[SearchResult] = []
    for li in soup.select("li.b_algo")[:k]:
        a = li.select_one("h2 a[href]")
        if not a:
            continue
        snippet_el = li.select_one(".b_caption p") or li.select_one("p")
        out.append(
            SearchResult(
                title=a.get_text(" ", strip=True),
                url=a["href"],
                snippet=snippet_el.get_text(" ", strip=True)[:500] if snippet_el else "",
            )
        )
    return out


class TavilyRetriever(BaseRetriever):
    name = "tavily"

    def __init__(self, api_key: str, client: httpx.AsyncClient | None = None) -> None:
        self.api_key = api_key
        self._client = client  # 测试可注入 MockTransport

    async def search(self, query: str, k: int = 5) -> list[SearchResult]:
        own = self._client is None
        client = self._client or httpx.AsyncClient(timeout=20)
        try:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={"api_key": self.api_key, "query": query, "max_results": k},
            )
            resp.raise_for_status()
            return [
                SearchResult(title=r.get("title", ""), url=r.get("url", ""), snippet=r.get("content", "")[:500])
                for r in resp.json().get("results", [])
            ]
        finally:
            if own:
                await client.aclose()


class SerperRetriever(BaseRetriever):
    name = "serper"

    def __init__(self, api_key: str, client: httpx.AsyncClient | None = None) -> None:
        self.api_key = api_key
        self._client = client

    async def search(self, query: str, k: int = 5) -> list[SearchResult]:
        own = self._client is None
        client = self._client or httpx.AsyncClient(timeout=20)
        try:
            resp = await client.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": self.api_key},
                json={"q": query, "num": k},
            )
            resp.raise_for_status()
            return [
                SearchResult(title=r.get("title", ""), url=r.get("link", ""), snippet=r.get("snippet", ""))
                for r in resp.json().get("organic", [])
            ]
        finally:
            if own:
                await client.aclose()


class StubRetriever(BaseRetriever):
    """离线桩：按 query 生成两条确定性结果。"""

    name = "stub"

    async def search(self, query: str, k: int = 5) -> list[SearchResult]:
        q = query[:40]
        return [
            SearchResult(
                title=f"关于「{q}」的权威分析",
                url="https://example.com/stub-1",
                snippet=f"该来源系统性梳理了 {q} 的背景、现状与关键数据……",
            ),
            SearchResult(
                title=f"「{q}」的最新进展报道",
                url="https://example.org/stub-2",
                snippet=f"近期公开资料显示 {q} 出现了若干值得关注的变化……",
            ),
        ][:k]


async def _aclose_retrievers(retrievers: list[BaseRetriever]) -> None:
    """逐个释放检索器资源，避免同一对象被包装层重复关闭。"""
    seen: set[int] = set()
    for retriever in retrievers:
        if id(retriever) in seen:
            continue
        seen.add(id(retriever))
        with contextlib.suppress(Exception):
            await retriever.aclose()


class FallbackRetriever(BaseRetriever):
    """主检索器异常或零结果时依次降级到备选引擎（网络环境差异的兜底）。

    语义：任一引擎返回非空结果即成功；全部"干净地"零结果视为真无结果返回 []；
    全链异常则抛出最后一个异常，交由上层记录错误事件。
    """

    def __init__(self, primary: BaseRetriever, fallbacks: list[BaseRetriever]) -> None:
        self.primary = primary
        self.fallbacks = fallbacks
        self.name = primary.name

    async def search(self, query: str, k: int = 5) -> list[SearchResult]:
        last_err: Exception | None = None
        for engine in [self.primary, *self.fallbacks]:
            try:
                results = await engine.search(query, k)
            except Exception as e:  # noqa: BLE001 — 降级链的核心就是吞掉单引擎异常
                last_err = e
                continue
            if results:
                for r in results:
                    if not r.via:
                        r.via = engine.name
                return results
        if last_err is not None:
            raise last_err
        return []

    async def aclose(self) -> None:
        await _aclose_retrievers([self.primary, *self.fallbacks])


class HybridRetriever(BaseRetriever):
    """web 检索链 + 带关键词的领域 MCP 并行补充（GPT-Researcher 式 hybrid research）。

    查询词命中某 MCP 的 keywords（不区分大小写）才并行调用它；未配置 keywords 的
    MCP 不进入本层（仍留在降级链尾）。web 结果在前，MCP 结果去重后追加。
    """

    def __init__(
        self,
        chain: BaseRetriever,
        conditional: list[tuple[BaseRetriever, list[str]]],
    ) -> None:
        self.chain = chain
        self.conditional = conditional
        self.name = chain.name

    async def search(self, query: str, k: int = 5) -> list[SearchResult]:
        q = query.lower()
        matched = [r for r, kws in self.conditional if any(kw in q for kw in kws)]
        if not matched:
            return await self.chain.search(query, k)
        tasks = [self.chain.search(query, k), *[r.search(query, k) for r in matched]]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        web = outcomes[0] if not isinstance(outcomes[0], BaseException) else []
        if isinstance(outcomes[0], BaseException):
            logger.warning("web 检索链失败（MCP 补充仍可用）: %s", outcomes[0])
        for r in web:
            if not r.via:
                r.via = self.chain.name  # 非 FallbackRetriever 链不会自带指纹
        merged = list(web)
        seen = {r.url for r in merged if r.url}
        for outcome, retriever in zip(outcomes[1:], matched, strict=True):
            if isinstance(outcome, BaseException):
                logger.warning("MCP 数据源 %s 调用失败，跳过: %s", retriever.name, outcome)
                continue
            if outcome:
                logger.info("MCP 数据源命中: %s 贡献 %d/%d 条（query=%s）", retriever.name, len(outcome), len(merged), query[:40])
            for r in outcome:
                if not r.via:
                    r.via = retriever.name
                if not r.url or r.url not in seen:
                    if r.url:
                        seen.add(r.url)
                    merged.append(r)
        return merged[: k * 2]

    async def aclose(self) -> None:
        await _aclose_retrievers([self.chain, *[r for r, _ in self.conditional]])


def _build_single(kind: str, settings) -> BaseRetriever:
    if kind == "tavily" and settings.tavily_api_key:
        return TavilyRetriever(settings.tavily_api_key)
    if kind == "serper" and settings.serper_api_key:
        return SerperRetriever(settings.serper_api_key)
    if kind == "stub":
        return StubRetriever()
    if kind == "so360":
        return So360Retriever()
    if kind == "bing":
        return BingRetriever()
    if kind == "duckduckgo":
        return DuckDuckGoRetriever()
    # 配置了无 key 的付费 retriever 时降级到 DDG
    return DuckDuckGoRetriever()


def build_retriever(settings, provider=None) -> BaseRetriever:
    """构建检索器：web 主链 + 降级链 + MCP 数据源。

    MCP 分流：配置 keywords 的走 HybridRetriever 选择性并行补充（领域工具/数据源）；
    未配置 keywords 的挂降级链尾（通用 search 型补充）。
    provider 用于 MCP 领域工具的 $llm 参数提取（如"从查询提取城市名"）。
    """
    mcp_retrievers = _build_mcp_retrievers(settings, provider)
    conditional = [(r, r.keywords) for r in mcp_retrievers if getattr(r, "keywords", None)]
    plain_mcp = [r for r in mcp_retrievers if not getattr(r, "keywords", None)]

    if settings.retriever == "mcp":
        if not mcp_retrievers:
            logger.warning("retriever=mcp 但未配置 mcp_servers，降级到 DDG")
            return DuckDuckGoRetriever()
        return FallbackRetriever(mcp_retrievers[0], [*mcp_retrievers[1:], _build_single("duckduckgo", settings)])

    primary = _build_single(settings.retriever, settings)
    fallbacks: list[BaseRetriever] = []
    for kind in [x.strip() for x in (settings.retriever_fallbacks or "").split(",") if x.strip()]:
        if kind == settings.retriever or kind == primary.name:
            continue
        fb = _build_single(kind, settings)
        # key 缺失时 _build_single 会静默降级为 DDG：非用户所选类型则跳过
        if not isinstance(fb, DuckDuckGoRetriever) or kind == "duckduckgo":
            if not any(isinstance(fb, type(f)) for f in fallbacks):
                fallbacks.append(fb)
    fallbacks.extend(plain_mcp)  # 通用 search 型 MCP：链尾兜底补充
    chain = FallbackRetriever(primary, fallbacks) if fallbacks else primary
    return HybridRetriever(chain, conditional) if conditional else chain


def _build_mcp_retrievers(settings, provider=None) -> list:
    from deepdossier.tools.mcp_source import McpRetriever, parse_mcp_servers

    llm_fn = None
    if provider is not None:
        async def llm_fn(prompt: str):  # noqa: E306 — $llm 参数提取用快模型
            return await provider.complete(
                system="你是参数提取器。根据指令从文本中提取信息，只输出答案本身，不要任何解释。",
                user=prompt,
                fast=True,
            )

    out = []
    for name, cfg in parse_mcp_servers(getattr(settings, "mcp_servers", "") or ""):
        out.append(McpRetriever(name, cfg, llm_fn=llm_fn))
    return out
