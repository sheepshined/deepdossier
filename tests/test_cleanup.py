"""run 结束后的资源释放回归测试。"""

from __future__ import annotations

import httpx

from deepdossier.graph.deps import ResearchDeps
from deepdossier.llm.provider import BaseEmbedder, StubProvider
from deepdossier.runner import build_deps
from deepdossier.tools.search import FallbackRetriever, HybridRetriever, StubRetriever
from deepdossier.trace import EventEmitter


class _ClosableProvider(StubProvider):
    def __init__(self):
        self.closed = False

    async def aclose(self):
        self.closed = True


class _ClosableEmbedder(BaseEmbedder):
    def __init__(self):
        self.closed = False

    async def embed(self, text: str):
        return [0.0]

    async def aclose(self):
        self.closed = True


class _ClosableRetriever(StubRetriever):
    def __init__(self):
        self.closed = False

    async def aclose(self):
        self.closed = True


def _http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200)))


async def test_retriever_wrappers_close_nested_retrievers(settings, store):
    primary = _ClosableRetriever()
    fallback = _ClosableRetriever()
    conditional = _ClosableRetriever()
    retriever = HybridRetriever(
        FallbackRetriever(primary, [fallback]), [(conditional, ["气象"])]
    )
    provider = _ClosableProvider()
    embedder = _ClosableEmbedder()
    deps = ResearchDeps(
        settings=settings,
        provider=provider,
        retriever=retriever,
        http=_http_client(),
        store=store,
        emitter=EventEmitter("cleanup-owned"),
        embedder=embedder,
    )

    await deps.aclose()

    assert provider.closed
    assert embedder.closed
    assert primary.closed
    assert fallback.closed
    assert conditional.closed


async def test_build_deps_does_not_close_injected_resources(settings, store):
    provider = _ClosableProvider()
    retriever = _ClosableRetriever()
    deps = build_deps(
        settings,
        store,
        "cleanup-injected",
        provider=provider,
        retriever=retriever,
    )

    await deps.aclose()

    assert not provider.closed
    assert not retriever.closed
