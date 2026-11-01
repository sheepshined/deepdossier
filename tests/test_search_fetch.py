"""搜索与抓取层测试。"""

from __future__ import annotations

import httpx

from deepdossier.tools.fetch import extract_text, fetch_readable
from deepdossier.tools.search import (
    BingRetriever,
    FallbackRetriever,
    HybridRetriever,
    SerperRetriever,
    So360Retriever,
    StubRetriever,
    TavilyRetriever,
    build_retriever,
)


async def test_stub_retriever():
    r = await StubRetriever().search("测试查询", k=5)
    assert len(r) == 2 and r[0].url.startswith("https://")


def test_extract_text_strips_scripts():
    html = "<html><head><title>T</title></head><body><script>bad()</script><p>hello world</p></body></html>"
    title, text = extract_text(html)
    assert title == "T" and "hello world" in text and "bad()" not in text


async def test_fetch_readable_with_mock():
    html = "<html><head><title>T</title></head><body><p>" + "内容足够长。" * 50 + "</p></body></html>"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=html, headers={"content-type": "text/html"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    text, err = await fetch_readable(client, "https://x.example.com")
    assert err == "" and text is not None and "内容足够长" in text


async def test_fetch_readable_skips_binary():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"\x00\x01", headers={"content-type": "application/octet-stream"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    text, err = await fetch_readable(client, "https://x.example.com/a.bin")
    assert text is None and "非文本" in err


async def test_tavily_retriever_with_mock_transport():
    def handler(request: httpx.Request) -> httpx.Response:
        body = {"results": [{"title": "t", "url": "https://a", "content": "c" * 600}]}
        return httpx.Response(200, json=body)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    r = TavilyRetriever(api_key="k", client=client)
    results = await r.search("q", k=3)
    assert len(results) == 1 and results[0].url == "https://a"


async def test_bing_retriever_with_mock_transport():
    html = """
    <html><body><ol id="b_results">
      <li class="b_algo"><h2><a href="https://a.example">标题A</a></h2>
        <div class="b_caption"><p>摘要A内容</p></div></li>
      <li class="b_algo"><h2><a href="https://b.example">标题B</a></h2><p>摘要B内容</p></li>
      <li class="b_algo"><h2><a>无链接项应被跳过</a></h2><p>x</p></li>
    </ol></body></html>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        assert "cn.bing.com" in str(request.url)
        return httpx.Response(200, text=html)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    r = BingRetriever(client=client)
    results = await r.search("测试查询", k=5)
    assert [x.url for x in results] == ["https://a.example", "https://b.example"]
    assert results[0].title == "标题A" and "摘要A" in results[0].snippet


async def test_so360_retriever_with_mock_transport():
    html = """
    <html><body><ul>
      <li class="res-list"><h3><a href="https://www.so.com/link?m=abc"
        data-mdurl="https://real.example/a">结果A</a></h3><p class="res-desc">摘要A内容</p></li>
      <li class="res-list"><h3><a href="https://www.so.com/link?m=def">结果B无mdurl</a></h3>
        <p class="res-rich">摘要B内容</p></li>
      <li class="res-list"><h3><span>无链接项应被跳过</span></h3><p>x</p></li>
    </ul></body></html>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        assert "so.com" in str(request.url)
        return httpx.Response(200, text=html)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    r = So360Retriever(client=client)
    results = await r.search("测试查询", k=5)
    # data-mdurl 优先（真实 URL），缺失时回退到站内跳转链接
    assert [x.url for x in results] == ["https://real.example/a", "https://www.so.com/link?m=def"]
    assert results[0].title == "结果A" and "摘要A" in results[0].snippet


async def test_serper_retriever_with_mock_transport():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-API-KEY"] == "k"
        return httpx.Response(200, json={"organic": [{"title": "t", "link": "https://b", "snippet": "s"}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    r = SerperRetriever(api_key="k", client=client)
    results = await r.search("q", k=3)
    assert results[0].url == "https://b"


# ---------------- FallbackRetriever ----------------

class _ErrRetriever(StubRetriever):
    name = "err"

    async def search(self, query: str, k: int = 5):
        raise RuntimeError("引擎挂了")


class _EmptyRetriever(StubRetriever):
    name = "empty"

    async def search(self, query: str, k: int = 5):
        return []


class _CountRetriever(StubRetriever):
    name = "counting"
    calls = 0

    async def search(self, query: str, k: int = 5):
        self.calls += 1
        return await super().search(query, k)


async def test_fallback_on_exception():
    ok = StubRetriever()
    r = FallbackRetriever(_ErrRetriever(), [ok])
    results = await r.search("q", k=5)
    assert len(results) == 2 and r.name == "err"


async def test_fallback_on_empty():
    ok = StubRetriever()
    r = FallbackRetriever(_EmptyRetriever(), [ok])
    results = await r.search("q", k=5)
    assert len(results) == 2


async def test_fallback_all_clean_empty_returns_empty():
    r = FallbackRetriever(_EmptyRetriever(), [_EmptyRetriever()])
    assert await r.search("q") == []


async def test_fallback_all_error_raises():
    import pytest

    r = FallbackRetriever(_ErrRetriever(), [_ErrRetriever()])
    with pytest.raises(RuntimeError, match="引擎挂了"):
        await r.search("q")


def test_build_retriever_dedupes_and_skips_keyless():
    from deepdossier.config import Settings

    s = Settings(
        retriever="so360", retriever_fallbacks="so360, bing, tavily, duckduckgo", mcp_servers=""
    )
    r = build_retriever(s)
    assert isinstance(r, FallbackRetriever)
    # so360 自身被跳过；tavily 无 key 静默降级为 DDG 也被跳过；bing 与 duckduckgo 保留
    assert [f.name for f in r.fallbacks] == ["bing", "duckduckgo"]

    s2 = Settings(retriever="so360", retriever_fallbacks="", mcp_servers="")
    assert not isinstance(build_retriever(s2), FallbackRetriever)


async def test_via_annotation_in_chains():
    """降级链与混合并行都标注 via 引擎指纹（检索透明化）。"""
    r = FallbackRetriever(_ErrRetriever(), [StubRetriever()])
    results = await r.search("q")
    assert all(x.via == "stub" for x in results)


async def test_hybrid_sets_via_for_mcp_results():
    from mcp.types import CallToolResult, TextContent

    from deepdossier.tools.mcp_source import McpRetriever

    class _Sess:
        async def call_tool(self, tool, args):
            import json as _j
            return CallToolResult(content=[TextContent(type="text", text=_j.dumps(
                [{"title": "M", "url": "https://m/1"}], ensure_ascii=False))])

    async def factory(stack, cfg):
        return _Sess()

    mcp = McpRetriever("w", {"keywords": ["天气"], "call_chain": [{"tool": "t", "args": {}}]},
                       session_factory=factory)
    hybrid = HybridRetriever(StubRetriever(), [(mcp, mcp.keywords)])
    try:
        results = await hybrid.search("北京天气")
        mcp_results = [x for x in results if x.title == "w 领域数据"]
        assert mcp_results and mcp_results[0].via == "mcp:w"
        web_results = [x for x in results if x.url.startswith("https://example")]
        assert web_results and all(x.via == "stub" for x in web_results)
    finally:
        await mcp.aclose()


def test_engine_breakdown_counts():
    from deepdossier.agents.researcher import _engine_breakdown
    from deepdossier.tools.search import SearchResult

    rs = [SearchResult(title="a", url="u", via="so360"), SearchResult(title="b", url="v", via="so360"),
          SearchResult(title="c", url="w", via="mcp:w")]
    assert _engine_breakdown(rs) == {"so360": 2, "mcp:w": 1}
    assert _engine_breakdown([SearchResult(title="x", url="y")]) == {}
