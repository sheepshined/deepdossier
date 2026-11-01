"""SQLite 运行存储：runs / events / lessons（策略记忆）/ embeddings 缓存。

要点：
- WAL 模式 + EventWriter 后台写线程，async 上下文不阻塞
- lessons 检索：有 embedding 时用余弦相似度，否则退化为字符 bigram Jaccard
  （旧实现按空格分词 + LIKE，对中文完全失效，本版修正）
- prune()：runs/events 按 retention_days 清理、lessons 按 max_lessons 封顶
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from deepdossier.trace import Event

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',   -- running | completed | failed
    report_md TEXT,
    error TEXT,
    usage_json TEXT,
    sources_json TEXT,
    created_at REAL NOT NULL,
    finished_at REAL
);
CREATE TABLE IF NOT EXISTS events (
    run_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    type TEXT NOT NULL,
    node TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    ts REAL NOT NULL,
    PRIMARY KEY (run_id, seq)
);
CREATE TABLE IF NOT EXISTS lessons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    lesson TEXT NOT NULL,
    question_vec_json TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS embeddings (
    text_hash TEXT PRIMARY KEY,
    vec_json TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS comparisons (
    pair_key TEXT PRIMARY KEY,                -- 两个 run_id 排序后拼接，方向无关
    run_a TEXT NOT NULL,
    run_b TEXT NOT NULL,
    verdict_md TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, seq);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at);
"""
# 旧库升级（CREATE TABLE IF NOT EXISTS 不会为旧表补列）
_MIGRATIONS = [
    "ALTER TABLE runs ADD COLUMN sources_json TEXT",
    "ALTER TABLE runs ADD COLUMN params_json TEXT",
    "ALTER TABLE lessons ADD COLUMN question_vec_json TEXT",
]

# 相似度阈值：余弦（语义）与 bigram Jaccard（字面）量纲不同，分别设定
_COSINE_MIN = 0.45
_JACCARD_MIN = 0.15


def _bigrams(text: str) -> set[str]:
    """字符 bigram（去空白、小写）；中文无空格分词问题由此绕开。"""
    norm = re.sub(r"\s+", "", text.lower())
    return {norm[i : i + 2] for i in range(len(norm) - 1)} or ({norm} if norm else set())


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class RunStore:
    """线程安全的 SQLite 封装（标准库实现，零运维；后续可平滑换 Postgres）。"""

    def __init__(self, db_path: Path | str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            for stmt in _MIGRATIONS:
                try:
                    self._conn.execute(stmt)
                except sqlite3.OperationalError:  # 列已存在
                    pass
            self._conn.commit()

    # ---- runs ----

    @staticmethod
    def new_run_id() -> str:
        return uuid.uuid4().hex[:12]

    def create_run(self, run_id: str, question: str, params: dict[str, Any] | None = None) -> None:
        """创建 run；params 记录创建时的请求参数（如 max_rounds），断点续跑时恢复。"""
        with self._lock:
            self._conn.execute(
                "INSERT INTO runs (id, question, status, created_at, params_json) VALUES (?, ?, 'running', ?, ?)",
                (run_id, question, time.time(), json.dumps(params, ensure_ascii=False) if params else None),
            )
            self._conn.commit()

    def finish_run(
        self,
        run_id: str,
        report_md: str | None = None,
        error: str | None = None,
        usage: dict[str, Any] | None = None,
        sources: list[dict[str, Any]] | None = None,
    ) -> None:
        status = "completed" if error is None else "failed"
        with self._lock:
            self._conn.execute(
                "UPDATE runs SET status=?, report_md=?, error=?, usage_json=?, sources_json=?, finished_at=? "
                "WHERE id=?",
                (
                    status,
                    report_md,
                    error,
                    json.dumps(usage) if usage else None,
                    json.dumps(sources, ensure_ascii=False) if sources else None,
                    time.time(),
                    run_id,
                ),
            )
            self._conn.commit()

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if not row:
            return None
        run = dict(row)
        run["usage"] = json.loads(run.pop("usage_json") or "null")
        run["sources"] = json.loads(run.pop("sources_json") or "null")
        run["params"] = json.loads(run.pop("params_json") or "null")
        return run

    def update_run_params(self, run_id: str, extra: dict[str, Any]) -> None:
        """合并写入 run 参数（如断点续跑恢复、direct/research 类型标记）。"""
        with self._lock:
            row = self._conn.execute("SELECT params_json FROM runs WHERE id=?", (run_id,)).fetchone()
            params = json.loads(row["params_json"]) if row and row["params_json"] else {}
            params.update(extra)
            self._conn.execute(
                "UPDATE runs SET params_json=? WHERE id=?",
                (json.dumps(params, ensure_ascii=False), run_id),
            )
            self._conn.commit()

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, question, status, created_at, finished_at, params_json "
                "FROM runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        out = []
        for r in rows:
            run = dict(r)
            run["params"] = json.loads(run.pop("params_json") or "null")
            out.append(run)
        return out

    # ---- events ----

    def append_event(self, event: Event) -> None:
        self.append_events([event])

    def append_events(self, events: list[Event]) -> None:
        if not events:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO events (run_id, seq, type, node, payload_json, ts) VALUES (?,?,?,?,?,?)",
                [
                    (
                        e.run_id,
                        e.seq,
                        e.type,
                        e.node,
                        json.dumps(e.payload, ensure_ascii=False),
                        e.ts,
                    )
                    for e in events
                ],
            )
            self._conn.commit()

    def list_events(self, run_id: str, after_seq: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT seq, type, node, payload_json, ts FROM events WHERE run_id=? AND seq>? ORDER BY seq",
                (run_id, after_seq),
            ).fetchall()
        return [
            {
                "seq": r["seq"],
                "type": r["type"],
                "node": r["node"],
                "payload": json.loads(r["payload_json"]),
                "ts": r["ts"],
            }
            for r in rows
        ]

    def max_event_seq(self, run_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT MAX(seq) AS m FROM events WHERE run_id=?", (run_id,)
            ).fetchone()
        return int(row["m"] or 0)

    def reopen_run(self, run_id: str) -> None:
        """断点续跑前把失败 run 重新置为 running。"""
        with self._lock:
            self._conn.execute(
                "UPDATE runs SET status='running', error=NULL, finished_at=NULL WHERE id=?",
                (run_id,),
            )
            self._conn.commit()

    def clear_events(self, run_id: str) -> int:
        """R12 fresh-restart：放弃断点重跑前清空旧事件（seq 从 0 重新开始）。"""
        with self._lock:
            cur = self._conn.execute("DELETE FROM events WHERE run_id=?", (run_id,))
            self._conn.commit()
            return cur.rowcount

    def fail_stale_running(self, reason: str) -> int:
        """服务启动时清理上次进程残留的 running 状态（F20：重启后无僵尸 run）。"""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE runs SET status='failed', error=?, finished_at=? WHERE status='running'",
                (reason, time.time()),
            )
            self._conn.commit()
            return cur.rowcount

    # ---- lessons（策略记忆）----

    def add_lesson(self, question: str, lesson: str, question_vec: list[float] | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO lessons (question, lesson, question_vec_json, created_at) VALUES (?,?,?,?)",
                (question, lesson, json.dumps(question_vec) if question_vec else None, time.time()),
            )
            self._conn.commit()

    def get_lessons(
        self,
        question: str,
        limit: int = 3,
        query_vec: list[float] | None = None,
    ) -> list[str]:
        """按相似度检索历史经验：优先语义（embedding 余弦），否则字符 bigram Jaccard。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT question, lesson, question_vec_json FROM lessons ORDER BY id DESC LIMIT 2000"
            ).fetchall()
        if not rows:
            return []

        q_grams = _bigrams(question)
        scored: list[tuple[float, str]] = []
        for r in rows:
            row_vec = json.loads(r["question_vec_json"]) if r["question_vec_json"] else None
            if query_vec and row_vec:
                score = _cosine(query_vec, row_vec)
                threshold = _COSINE_MIN
            else:
                l_grams = _bigrams(r["question"])
                union = q_grams | l_grams
                score = len(q_grams & l_grams) / len(union) if union else 0.0
                threshold = _JACCARD_MIN
            if score >= threshold:
                scored.append((score, r["lesson"]))
        scored.sort(key=lambda x: -x[0])
        return [lesson for _, lesson in scored[:limit]]

    # ---- embeddings 缓存（问题文本 → 向量，跨 run 复用）----

    @staticmethod
    def _text_hash(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def get_query_vec(self, text: str) -> list[float] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT vec_json FROM embeddings WHERE text_hash=?", (self._text_hash(text),)
            ).fetchone()
        return json.loads(row["vec_json"]) if row else None

    def set_query_vec(self, text: str, vec: list[float]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO embeddings (text_hash, vec_json, created_at) VALUES (?,?,?)",
                (self._text_hash(text), json.dumps(vec), time.time()),
            )
            self._conn.commit()

    # ---- 报告对比（结果持久化：同一对 run 不重复计费） ----

    @staticmethod
    def _pair_key(a: str, b: str) -> str:
        return "|".join(sorted([a, b]))

    def get_comparison(self, run_a: str, run_b: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT verdict_md FROM comparisons WHERE pair_key=?",
                (self._pair_key(run_a, run_b),),
            ).fetchone()
        return row["verdict_md"] if row else None

    def save_comparison(self, run_a: str, run_b: str, verdict_md: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO comparisons (pair_key, run_a, run_b, verdict_md, created_at) VALUES (?,?,?,?,?)",
                (self._pair_key(run_a, run_b), run_a, run_b, verdict_md, time.time()),
            )
            self._conn.commit()

    # ---- 保留策略 ----

    def prune(self, retention_days: int = 30, max_lessons: int = 500) -> None:
        """清理过期 runs/events，lessons 数量封顶；不清理 running 状态的 run。"""
        cutoff = time.time() - retention_days * 86400
        with self._lock:
            old_ids = [
                r["id"]
                for r in self._conn.execute(
                    "SELECT id FROM runs WHERE created_at < ? AND status != 'running'", (cutoff,)
                ).fetchall()
            ]
            if old_ids:
                self._conn.executemany("DELETE FROM events WHERE run_id = ?", [(i,) for i in old_ids])
                self._conn.executemany("DELETE FROM runs WHERE id = ?", [(i,) for i in old_ids])
            n_lessons = self._conn.execute("SELECT COUNT(*) AS n FROM lessons").fetchone()["n"]
            if n_lessons > max_lessons:
                self._conn.execute(
                    "DELETE FROM lessons WHERE id IN "
                    "(SELECT id FROM lessons ORDER BY id ASC LIMIT ?)",
                    (n_lessons - max_lessons,),
                )
            self._conn.execute("DELETE FROM embeddings WHERE created_at < ?", (cutoff,))
            self._conn.commit()

    def close(self) -> None:
        self._conn.close()
