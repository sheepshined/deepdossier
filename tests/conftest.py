"""测试公共夹具：全离线（Stub LLM + Stub 检索 + Mock 抓取），临时数据目录。

保真要点：fetch_pages 关闭且注入 mock http —— 全图测试不出网。
"""

from __future__ import annotations

import httpx
import pytest

from deepdossier.config import Settings
from deepdossier.llm.provider import StubProvider
from deepdossier.store.runs import RunStore
from deepdossier.tools.search import StubRetriever


@pytest.fixture()
def settings(tmp_path) -> Settings:
    s = Settings(
        llm_provider="stub",
        retriever="stub",
        data_dir=tmp_path / "data",
        inline_execution=True,
        fetch_pages=False,
        max_research_rounds=2,
        checkpointing=False,  # 默认关闭；断点续跑测试在 test_checkpoint.py 单独开启
        mcp_servers="",  # 隔离本机 .env 的 MCP 配置，测试不依赖环境
    )
    s.ensure_dirs()
    return s


@pytest.fixture()
def store(settings):
    s = RunStore(settings.db_path)
    yield s
    s.close()  # 避免测试进程残留 unclosed sqlite 警告


@pytest.fixture()
def stub_provider() -> StubProvider:
    return StubProvider()


@pytest.fixture()
def stub_retriever() -> StubRetriever:
    return StubRetriever()


@pytest.fixture()
def mock_http() -> httpx.AsyncClient:
    """抓取层 Mock：任何 http(s) URL 返回可解析 HTML。"""
    html = (
        "<html><head><title>Mock Page</title></head><body><script>x</script>"
        "<p>" + "DeepDossier 正文内容测试。" * 20 + "</p></body></html>"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=html, headers={"content-type": "text/html"})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.fixture()
def offline_http() -> httpx.AsyncClient:
    """强断言 Mock：任何网络请求都直接失败，保证测试零出网。"""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"测试不允许真实网络请求: {request.url}")

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))
