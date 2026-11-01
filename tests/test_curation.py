"""来源质量过滤（source curation）测试：全离线，fake provider 注入。"""

from __future__ import annotations

from deepdossier.agents.researcher import curate_sources
from deepdossier.agents.schemas import SourceCuration
from deepdossier.tools.search import SearchResult


def _results(n: int) -> list[SearchResult]:
    return [
        SearchResult(title=f"结果{i}", url=f"https://example.com/{i}", snippet=f"摘要{i}")
        for i in range(n)
    ]


class _FakeEmitter:
    def __init__(self) -> None:
        self.events: list[tuple] = []

    def emit(self, type_, node, payload):
        self.events.append((type_, node, payload))


class _Deps:
    """curate_sources 只依赖 settings / provider / emitter 三个属性。"""

    def __init__(self, provider, source_curation=True):
        from deepdossier.config import Settings

        self.settings = Settings(source_curation=source_curation)
        self.provider = provider
        self.emitter = _FakeEmitter()


class _KeepProvider:
    """总是保留给定编号的结果。"""

    def __init__(self, keep: list[int]) -> None:
        self.keep = keep
        self.calls = 0

    async def complete(self, *, system, user, schema=None, **kw):
        self.calls += 1
        assert schema is SourceCuration
        return SourceCuration(keep=self.keep)


class _BoomProvider:
    async def complete(self, *, system, user, schema=None, **kw):
        raise RuntimeError("过滤模型故障")


async def test_curate_filters_to_kept_indices():
    p = _KeepProvider([2, 0, 9])  # 9 越界应被丢弃
    deps = _Deps(p)
    out = await curate_sources(deps, "查询", _results(5))
    assert [r.url for r in out] == ["https://example.com/0", "https://example.com/2"]
    assert p.calls == 1
    types = [e[0] for e in deps.emitter.events]
    assert "curate" in types


async def test_curate_fail_open_keeps_all():
    deps = _Deps(_BoomProvider())
    out = await curate_sources(deps, "查询", _results(5))
    assert len(out) == 5  # 过滤失败不阻断研究
    assert any(e[0] == "error" for e in deps.emitter.events)


async def test_curate_all_irrelevant_keeps_floor_one():
    deps = _Deps(_KeepProvider([]))
    out = await curate_sources(deps, "查询", _results(5))
    assert len(out) == 1  # 兜底保留第一条，避免子问题断粮


async def test_curate_disabled_returns_unchanged():
    p = _KeepProvider([0])
    deps = _Deps(p, source_curation=False)
    out = await curate_sources(deps, "查询", _results(5))
    assert len(out) == 5 and p.calls == 0


async def test_curate_skips_single_result():
    p = _KeepProvider([0])
    deps = _Deps(p)
    out = await curate_sources(deps, "查询", _results(1))
    assert len(out) == 1 and p.calls == 0
