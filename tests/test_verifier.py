"""草稿级事实审查（Verifier）测试：fake provider，全离线。"""

from __future__ import annotations

from deepdossier.agents.schemas import DraftReview
from deepdossier.agents.verifier import make_verifier_node
from deepdossier.config import Settings

REPORT = (
    "# 报告\n\n## 摘要\n核心结论[1]。\n\n## 参考来源\n1. A — https://a\n"
)
SOURCES = [{"n": 1, "title": "A", "url": "https://a"}]
EVIDENCE = [{"question_id": "q1", "claim": "c", "detail": "d", "source_url": "https://a", "source_title": "A"}]
STATE = {"run_id": "r", "question": "q", "report_md": REPORT, "sources": SOURCES, "evidence": EVIDENCE}


class _FakeEmitter:
    def __init__(self) -> None:
        self.events: list[tuple] = []

    def emit(self, type_, node, payload):
        self.events.append((type_, node, payload))


class _Deps:
    def __init__(self, provider, draft_review=True):
        self.settings = Settings(draft_review=draft_review)
        self.provider = provider
        self.emitter = _FakeEmitter()


class _ApproveProvider:
    def __init__(self) -> None:
        self.calls: list = []

    async def complete(self, *, system, user, schema=None, **kw):
        self.calls.append(schema.__name__ if schema else "text")
        if schema is not None and schema.__name__ == "DraftReview":
            return DraftReview(needs_revision=False, issues=[])
        return "无关输出"


class _ReviseProvider(_ApproveProvider):
    async def complete(self, *, system, user, schema=None, **kw):
        self.calls.append(schema.__name__ if schema else "text")
        if schema is not None and schema.__name__ == "DraftReview":
            return DraftReview(needs_revision=True, issues=["摘要存在无引用支撑的断言"])
        return REPORT.replace("核心结论[1]。", "核心结论[1]，补充限定[1]。")


class _BoomProvider:
    async def complete(self, *, system, user, schema=None, **kw):
        raise RuntimeError("审查模型故障")


async def test_verifier_passes_clean_draft():
    p = _ApproveProvider()
    out = await make_verifier_node(_Deps(p))(STATE)
    assert out["report_md"] == REPORT
    assert p.calls == ["DraftReview"]  # 无修订则无第二次调用


async def test_verifier_revises_and_remerges():
    p = _ReviseProvider()
    out = await make_verifier_node(_Deps(p))(STATE)
    assert "补充限定" in out["report_md"]
    assert p.calls == ["DraftReview", "text"]  # 审查 + 一次修订
    # 修订稿重新经过引用合并，sources 保持唯一
    assert out["sources"] == [{"n": 1, "title": "A", "url": "https://a"}]


async def test_verifier_fail_open_on_review_error():
    p = _BoomProvider()
    deps = _Deps(p)
    out = await make_verifier_node(deps)(STATE)
    assert out["report_md"] == REPORT  # 审查失败原样放行
    assert any(e[0] == "error" for e in deps.emitter.events)


async def test_verifier_skipped_when_disabled():
    p = _ApproveProvider()
    out = await make_verifier_node(_Deps(p, draft_review=False))(STATE)
    assert out["report_md"] == REPORT and p.calls == []


async def test_verifier_rejects_shrunk_revision():
    class _ShrinkProvider(_ReviseProvider):
        async def complete(self, *, system, user, schema=None, **kw):
            self.calls.append(schema.__name__ if schema else "text")
            if schema is not None and schema.__name__ == "DraftReview":
                return DraftReview(needs_revision=True, issues=["x"])
            return "# 报告\n\n（截断）"  # 异常缩水

    p = _ShrinkProvider()
    out = await make_verifier_node(_Deps(p))(STATE)
    assert out["report_md"] == REPORT  # 缩水稿被拒，保留原稿
