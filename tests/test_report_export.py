"""报告导出与引用合并测试（全离线）。"""

from __future__ import annotations

import httpx

from deepdossier.agents.schemas import SourceRef
from deepdossier.agents.writer import merge_citations
from deepdossier.api.app import create_app
from deepdossier.report_export import render_export_html
from deepdossier.store.runs import RunStore

# ---------------- 引用合并 ----------------

def test_merge_citations_dedupes_and_renumbers():
    sources = [
        SourceRef(n=1, title="A", url="https://a"),
        SourceRef(n=2, title="B", url="https://b"),
        SourceRef(n=3, title="A重复", url="https://a"),
        SourceRef(n=4, title="C", url="https://c"),
    ]
    md = "论断一[1]与论断二[3]，再看[2]和[4]，普通文本 [2026] 不受影响。"
    new_md, merged = merge_citations(md, sources)
    assert [s.n for s in merged] == [1, 2, 3]
    assert [s.url for s in merged] == ["https://a", "https://b", "https://c"]
    # 重复来源 [3] 已重定向到 [1]
    assert "论断一[1]与论断二[1]" in new_md
    assert "[2026]" in new_md  # 非引用数字原样保留


def test_merge_citations_no_duplicates_is_stable():
    sources = [SourceRef(n=1, title="A", url="https://a")]
    md = "引用[1]"
    new_md, merged = merge_citations(md, sources)
    assert new_md == "引用[1]" and len(merged) == 1


# ---------------- 导出渲染 ----------------

def _fake_run() -> dict:
    return {
        "id": "abc123",
        "question": "测试问题？",
        "status": "completed",
        "report_md": "# 测试报告\n\n核心结论[1]。\n\n| 维度 | 值 |\n|---|---|\n| A | 1 |\n",
        "error": None,
        "usage": {"calls": 5, "prompt_tokens": 1000, "completion_tokens": 500},
        "sources": [{"n": 1, "title": "来源一", "url": "https://example.com/a"}],
        "created_at": 1700000000.0,
        "finished_at": 1700000060.0,
    }


def test_render_export_html_contains_report_and_meta():
    html = render_export_html(_fake_run(), version="0.2.0")
    assert "测试报告" in html
    assert "测试问题？" in html
    assert "1 分 0 秒" in html
    assert "5 次调用" in html
    assert "DeepDossier" in html
    # markdown 以 JSON 内嵌（防 script 注入），原文不裸露在 HTML 标签里
    assert '"# 测试报告' in html


def test_render_export_html_escapes_script_injection():
    run = _fake_run()
    run["report_md"] = "# T\n</script><script>alert(1)</script>"
    html = render_export_html(run)
    # JSON 转义后 </script> 被拆开，不会提前终止脚本标签
    assert "<\\/script>" in html or "<\\/script" in html


async def test_export_endpoint(settings, store):
    run_id = RunStore.new_run_id()
    store.create_run(run_id, "导出测试")
    store.finish_run(run_id, "报告正文", None, {"calls": 1}, [{"n": 1, "title": "s", "url": "https://x"}])
    app = create_app(settings, store)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(f"/api/researches/{run_id}/export")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "deepdossier-" in resp.headers.get("content-disposition", "")

        # 无报告的 run 导出 404
        run_id2 = RunStore.new_run_id()
        store.create_run(run_id2, "无报告")
        assert (await client.get(f"/api/researches/{run_id2}/export")).status_code == 404
        assert (await client.get("/api/researches/不存在/export")).status_code == 404


# ---------------- DOCX 导出 ----------------

def test_render_docx_structure():
    import io

    from docx import Document

    from deepdossier.report_docx import render_docx

    run = _fake_run()
    run["report_md"] = (
        "# 标题\n\n段落带**粗体**与引用[1]。\n\n"
        "- 要点甲\n- 要点乙\n\n"
        "| 维度 | 值 |\n|---|---|\n| A | 1 |\n\n"
        "> 引用块内容\n"
    )
    data = render_docx(run, version="0.2.0")
    doc = Document(io.BytesIO(data))
    texts = [p.text for p in doc.paragraphs]
    assert any("标题" in t for t in texts)
    assert any("要点甲" in t for t in texts)
    assert any("引用块内容" in t for t in texts)
    assert len(doc.tables) == 1 and doc.tables[0].rows[1].cells[0].text == "A"


async def test_export_docx_endpoint(settings, store):
    run_id = RunStore.new_run_id()
    store.create_run(run_id, "DOCX 导出测试")
    store.finish_run(run_id, "# 报告\n正文", None, None, None)
    app = create_app(settings, store)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(f"/api/researches/{run_id}/export.docx")
        assert resp.status_code == 200
        assert "wordprocessingml" in resp.headers["content-type"]
        assert (await client.get("/api/researches/none/export.docx")).status_code == 404


# ---------------- run 参数持久化与续跑守卫 ----------------

def test_create_run_persists_params(settings, store):
    run_id = RunStore.new_run_id()
    store.create_run(run_id, "参数持久化", params={"max_rounds": 3})
    run = store.get_run(run_id)
    assert run["params"] == {"max_rounds": 3}
    assert store.get_run(RunStore.new_run_id() + "x") is None


async def test_resume_guards(settings, store):
    """completed/running run 不可续跑；checkpointing 关闭时 failed run 续跑被拒。"""
    import httpx

    from deepdossier.api.app import create_app

    rid_done = RunStore.new_run_id()
    store.create_run(rid_done, "已完成")
    store.finish_run(rid_done, "报告", None, None, None)

    rid_failed = RunStore.new_run_id()
    store.create_run(rid_failed, "已失败")
    store.finish_run(rid_failed, None, "boom", None, None)

    app = create_app(settings, store)  # conftest settings: checkpointing=False
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r1 = await client.post(f"/api/researches/{rid_done}/resume")
        assert r1.status_code == 409 and "only failed" in r1.json()["detail"]
        r2 = await client.post(f"/api/researches/{rid_failed}/resume")
        assert r2.status_code == 409 and "checkpointing is disabled" in r2.json()["detail"]


async def test_compare_endpoint_and_cache(settings, store):
    """对比端点：生成 → 持久化 → 第二次命中缓存（provider 只调一次）。"""
    import httpx

    from deepdossier.api.app import create_app
    from deepdossier.llm.provider import StubProvider

    ids = []
    for q in ["对比测试A", "对比测试B"]:
        rid = RunStore.new_run_id()
        store.create_run(rid, q)
        store.finish_run(rid, f"# 报告{q}\n结论[1]", None, {"calls": 1}, [{"n": 1, "title": "s", "url": "https://x"}])
        ids.append(rid)

    calls = {"n": 0}

    class _CountingStub(StubProvider):
        async def complete(self, *, system, user, schema=None, **kw):
            calls["n"] += 1
            return f"## 结论一致性\n两份报告一致（第 {calls['n']} 次调用）。"

    app = create_app(settings, store)
    app.state.compare_provider = _CountingStub()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r1 = await client.post("/api/compare", json={"run_id_a": ids[0], "run_id_b": ids[1]})
        assert r1.status_code == 200 and r1.json()["cached"] is False
        r2 = await client.post("/api/compare", json={"run_id_b": ids[1], "run_id_a": ids[0]})  # 方向无关
        assert r2.status_code == 200 and r2.json()["cached"] is True
        assert calls["n"] == 1  # 缓存生效，provider 只调用一次
        # 参数校验
        assert (await client.post("/api/compare", json={"run_id_a": ids[0], "run_id_b": ids[0]})).status_code == 400
        assert (await client.post("/api/compare", json={"run_id_a": "nope", "run_id_b": ids[1]})).status_code == 404
