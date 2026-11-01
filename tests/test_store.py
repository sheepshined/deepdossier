"""存储层测试：记忆检索（中文 n-gram / 向量）、去重归约、保留策略、事件批量写。"""

from __future__ import annotations

import time

from deepdossier.graph.state import merge_evidence
from deepdossier.store.runs import RunStore, _bigrams

# ---------------- lessons：中文 n-gram 检索（旧实现的 P0 bug 回归） ----------------

def test_lessons_chinese_fuzzy_match(store):
    """中文问题无空格：措辞不同但主题相同的历史经验必须能命中（旧 LIKE 实现失败）。"""
    store.add_lesson("2026年新能源汽车市场规模和增速如何？", "优先查询行业白皮书与统计局数据")
    store.add_lesson("如何给孩子选择编程课程？", "对比课程时先明确孩子年龄段")

    hits = store.get_lessons("新能源汽车的市场规模怎么样？")
    assert hits == ["优先查询行业白皮书与统计局数据"], "应命中新能源汽车那条经验且仅此一条"


def test_lessons_irrelevant_not_matched(store):
    store.add_lesson("如何给孩子选择编程课程？", "对比课程时先明确孩子年龄段")
    assert store.get_lessons("火星探测器轨道计算") == []


def test_lessons_embedding_cosine(store):
    """带向量时走余弦相似度：相同向量高相似命中，零向量不命中。"""
    q = "多智能体系统如何落地企业？"
    vec = [0.1, 0.2, 0.3]
    store.add_lesson(q, "企业落地先做 PoC", question_vec=vec)
    assert store.get_lessons(q, query_vec=vec) == ["企业落地先做 PoC"]
    assert store.get_lessons(q, query_vec=[0.0, 0.0, 0.0]) == []


def test_bigrams_chinese():
    assert _bigrams("多智能体") == {"多智", "智能", "能体"}


# ---------------- 归约器：证据去重 ----------------

def test_merge_evidence_dedupes_same_claim_and_source():
    card = {"claim": "市场规模为 100 亿美元。", "source_url": "https://a"}
    assert merge_evidence([card], [dict(card)]) == [card]


def test_merge_evidence_keeps_different_claims():
    a = {"claim": "市场规模为 100 亿美元。", "source_url": "https://a"}
    b = {"claim": "增速为 15%。", "source_url": "https://a"}
    assert len(merge_evidence([a], [b])) == 2


def test_merge_evidence_claim_normalization():
    """标点/空白差异的 claim 视为同一条。"""
    a = {"claim": "市场规模为100亿美元", "source_url": "https://a"}
    b = {"claim": "市场规模为 100 亿美元。", "source_url": "https://a"}
    assert len(merge_evidence([a], [b])) == 1


def test_merge_evidence_none_inputs():
    assert merge_evidence(None, None) == []
    assert merge_evidence(None, [{"claim": "x", "source_url": "u"}]) == [{"claim": "x", "source_url": "u"}]


# ---------------- 保留策略 ----------------

def test_prune_old_runs_and_events(store):
    store.create_run("old", "旧问题")
    store.create_run("new", "新问题")
    store.finish_run("old", report_md="done")
    # 把 old 的创建时间拨回 40 天前
    with store._lock:  # noqa: SLF001 — 测试直接操作内部连接模拟过期
        store._conn.execute("UPDATE runs SET created_at = ? WHERE id = ?", (time.time() - 40 * 86400, "old"))
        store._conn.commit()

    from deepdossier.trace import EventEmitter

    emitter = EventEmitter("old")
    emitter._seq = 3
    store.append_events([emitter.emit("node_start", "planner", {})])

    store.prune(retention_days=30, max_lessons=500)
    assert store.get_run("old") is None
    assert store.list_events("old") == []
    assert store.get_run("new") is not None  # running 状态不清理


def test_prune_caps_lessons(store):
    for i in range(10):
        store.add_lesson(f"问题{i}", f"经验{i}")
    store.prune(retention_days=30, max_lessons=3)
    with store._lock:  # noqa: SLF001
        n = store._conn.execute("SELECT COUNT(*) AS n FROM lessons").fetchone()["n"]
    assert n == 3


# ---------------- run / 事件基础 ----------------

def test_run_sources_roundtrip(store):
    store.create_run("r1", "q")
    store.finish_run("r1", report_md="# R", sources=[{"n": 1, "title": "S", "url": "https://s"}])
    run = store.get_run("r1")
    assert run["sources"] == [{"n": 1, "title": "S", "url": "https://s"}]
    assert run["status"] == "completed"


def test_append_events_batch(store):
    from deepdossier.trace import EventEmitter

    emitter = EventEmitter("r2")
    events = [emitter.emit("node_start", "planner", {}), emitter.emit("node_end", "planner", {"x": 1})]
    store.append_events(events)
    listed = store.list_events("r2")
    assert [e["seq"] for e in listed] == [1, 2]
    assert listed[1]["payload"] == {"x": 1}


def test_migrations_idempotent(tmp_path):
    """重复建库（含旧 schema 缺列）不报错。"""
    p = tmp_path / "migrate.sqlite3"
    RunStore(p)
    RunStore(p)  # 二次初始化：ALTER 已存在列被吞掉
