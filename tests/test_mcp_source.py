"""MCP 数据源检索器测试：fake session 注入，零网络零子进程。"""

from __future__ import annotations

import json
from contextlib import AsyncExitStack

from mcp.types import CallToolResult, TextContent

from deepdossier.config import Settings
from deepdossier.tools.mcp_source import (
    McpRetriever,
    _results_from_text,
    parse_mcp_servers,
)
from deepdossier.tools.search import (
    FallbackRetriever,
    HybridRetriever,
    StubRetriever,
    build_retriever,
)

# ---------------- 结果映射 ----------------

def test_map_json_list_of_dicts():
    text = json.dumps([
        {"title": "T1", "url": "https://a", "description": "D1"},
        {"name": "T2", "link": "https://b", "content": "D2"},
    ])
    rs = _results_from_text(text)
    assert [(r.title, r.url) for r in rs] == [("T1", "https://a"), ("T2", "https://b")]


def test_map_plain_text_and_empty():
    rs = _results_from_text("这是一段纯文本 MCP 结果")
    assert len(rs) == 1 and rs[0].snippet.startswith("这是一段")
    assert _results_from_text("") == []
    assert _results_from_text("   ") == []


def test_map_dict_with_results_key():
    text = json.dumps({"results": [{"title": "X", "url": "https://x"}]})
    assert _results_from_text(text)[0].url == "https://x"


# ---------------- 配置解析（两种格式） ----------------

def test_parse_standard_mcp_servers_format():
    raw = json.dumps({"mcpServers": {
        "weather-mcp": {"command": "python.exe", "args": ["test.py"], "cwd": "E:/x"},
    }})
    servers = parse_mcp_servers(raw)
    # 标准格式无 tool/call_chain 会被跳过（无法猜测调用方式）
    assert servers == []


def test_parse_standard_format_with_call_chain():
    raw = json.dumps({"mcpServers": {
        "weather-mcp": {"command": "python.exe", "args": ["test.py"],
                        "call_chain": [{"tool": "query_weather", "args": {"city_code": "北京"}}]},
    }})
    servers = parse_mcp_servers(raw)
    assert [n for n, _ in servers] == ["weather-mcp"]
    assert servers[0][1].get("cwd") is None  # 无 cwd 时为 None，有则透传


def test_parse_flat_array_and_validation():
    raw = json.dumps([
        {"name": "a", "transport": "http", "url": "http://x", "tool": "search"},
        {"name": "bad-http"},                      # 缺 url → 跳过
        {"name": "no-tool", "command": "npx"},     # 无 tool/call_chain → 跳过
        {"bad": 1},
    ])
    assert [n for n, _ in parse_mcp_servers(raw)] == ["a"]
    assert parse_mcp_servers("") == []
    assert parse_mcp_servers("not-json") == []


# ---------------- 会话与检索 ----------------

class _FakeSession:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple] = []

    async def call_tool(self, tool: str, args: dict):
        self.calls.append((tool, args))
        return CallToolResult(content=[TextContent(type="text", text=self.responses.pop(0))])


def _factory(session: _FakeSession):
    async def factory(stack: AsyncExitStack, cfg: dict):
        return session

    return factory


async def test_mcp_retriever_search_roundtrip():
    payload = json.dumps([{"title": "M1", "url": "https://m/1", "snippet": "s1"}])
    r = McpRetriever("corp", {"transport": "http", "url": "http://x/mcp", "tool": "search"},
                     session_factory=_factory(_FakeSession([payload])))
    try:
        results = await r.search("查询词", k=5)
        assert results[0].url == "https://m/1" and r.name == "mcp:corp"
    finally:
        await r.aclose()


async def test_mcp_retriever_reuses_session():
    sess = _FakeSession([json.dumps([{"title": "M", "url": "https://m"}])] * 2)

    r = McpRetriever("corp", {}, session_factory=_factory(sess))
    try:
        await r.search("q1")
        await r.search("q2")
        assert len(sess.calls) == 2  # 会话复用
        assert sess.calls[0][1] == {"query": "q1"}
    finally:
        await r.aclose()


async def test_call_chain_with_llm_extraction():
    """领域工具链：$llm 提取城市名 → query_weather(city, mode=1) → 文本作摘要。"""

    async def llm_fn(prompt: str):
        assert "城市" in prompt
        assert "北京明天天气" in prompt  # 原始查询词已注入
        return "北京"

    sess = _FakeSession(["📍 北京市\n天气: 多云 温度: 22°C"])
    r = McpRetriever(
        "weather", {"keywords": ["天气"], "call_chain": [
            {"tool": "query_weather", "args": {"city_code": {"$llm": "提取城市名"}, "mode": 1}},
        ]},
        session_factory=_factory(sess), llm_fn=llm_fn,
    )
    try:
        results = await r.search("北京明天天气如何")
        assert len(results) == 1 and "22°C" in results[0].snippet
        assert sess.calls == [("query_weather", {"city_code": "北京", "mode": 1})]
    finally:
        await r.aclose()


async def test_call_chain_uses_prev_placeholder():
    """{prev} 串联：第一步输出作为第二步参数（模拟 adcode→weather）。"""
    sess = _FakeSession(["110000", "天气: 晴 25°C"])
    r = McpRetriever(
        "weather", {"call_chain": [
            {"tool": "get_adcode", "args": {"city": "{query}"}},
            {"tool": "query_weather", "args": {"city_code": "{prev}", "mode": 1}},
        ]},
        session_factory=_factory(sess),
    )
    try:
        results = await r.search("北京")
        assert sess.calls[1] == ("query_weather", {"city_code": "110000", "mode": 1})
        assert "25°C" in results[0].snippet
    finally:
        await r.aclose()


# ---------------- 与检索链集成 ----------------

def test_build_retriever_keyword_mcp_goes_hybrid():
    s = Settings(
        retriever="so360", retriever_fallbacks="bing",
        mcp_servers=json.dumps([{"name": "w", "command": "x", "keywords": ["天气", "气温"],
                                 "call_chain": [{"tool": "t", "args": {}}]}]),
    )
    r = build_retriever(s)
    assert isinstance(r, HybridRetriever)
    assert r.conditional[0][1] == ["天气", "气温"]


def test_build_retriever_plain_mcp_goes_fallback_tail():
    s = Settings(
        retriever="so360", retriever_fallbacks="",
        mcp_servers=json.dumps([{"name": "s1", "transport": "http", "url": "http://x", "tool": "search"}]),
    )
    r = build_retriever(s)
    assert isinstance(r, FallbackRetriever)
    assert [f.name for f in r.fallbacks] == ["mcp:s1"]


async def test_hybrid_retriever_keyword_gating():
    """未命中关键词 → 只走 web 链；命中 → 并行调用 MCP 并合并。"""
    payload = json.dumps([{"title": "FROM_WEATHER_MCP", "url": "", "snippet": "22°C 多云"}], ensure_ascii=False)

    class _CountWeb(StubRetriever):
        name = "web"

    web = _CountWeb()
    mcp = McpRetriever(
        "weather", {"keywords": ["天气"], "call_chain": [{"tool": "t", "args": {}}]},
        session_factory=_factory(_FakeSession([payload])),
    )
    hybrid = HybridRetriever(web, [(mcp, mcp.keywords)])
    try:
        plain = await hybrid.search("大模型企业选型")   # 无天气关键词
        assert len(plain) == 2 and all("FROM_WEATHER_MCP" not in r.title for r in plain)

        hit = await hybrid.search("北京天气怎么样")      # 命中
        assert any(r.title == "weather 领域数据" and "22°C" in r.snippet for r in hit)  # MCP 结果合并进
        assert any(r.title.startswith("关于") for r in hit)  # web 结果也在
    finally:
        await mcp.aclose()
