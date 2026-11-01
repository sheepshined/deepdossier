"""LangGraph 研究状态定义。

evidence 用自定义归约器 merge_evidence：承接 Researcher 并行扇出写入的同时，
按 (归一化 claim, source_url) 去重——补搜轮与并行研究员产出重复证据时自动合并。
其余字段单点写入。plan/critique/evidence 均以 dict 形式存储，保证可序列化。
"""

from __future__ import annotations

import re
from typing import Annotated, TypedDict


def _claim_key(card: dict) -> tuple[str, str]:
    norm = re.sub(r"[\s。．，,\.!！?？:：;；、~～\-—]+", "", str(card.get("claim", "")).lower())
    return (norm, str(card.get("source_url", "")))


_MAX_EVIDENCE_CARDS = 400  # R30 内存护栏：单 run 证据卡上限，超出丢弃新增并保留先到者


def merge_evidence(existing: list[dict] | None, new: list[dict] | None) -> list[dict]:
    """证据归约：累加 + 按 (claim, source_url) 去重 + 数量硬上限（R30）。"""
    existing = list(existing or [])
    new = new or []
    if not new or len(existing) >= _MAX_EVIDENCE_CARDS:
        return existing
    seen = {_claim_key(e) for e in existing}
    for card in new:
        if len(existing) >= _MAX_EVIDENCE_CARDS:
            break
        key = _claim_key(card)
        if key not in seen:
            seen.add(key)
            existing.append(card)
    return existing


class ResearchState(TypedDict, total=False):
    run_id: str
    question: str
    route: str         # Router 路由决策：research | direct（必须在 schema 声明，否则状态被丢弃）
    plan: dict          # ResearchPlan.model_dump()
    round: int          # 当前研究轮次，从 1 开始
    evidence: Annotated[list[dict], merge_evidence]  # EvidenceCard.model_dump() 列表
    critique: dict      # Critique.model_dump()
    pending_followups: list[str]  # Critic 决定的待补搜查询（非空时路由器扇出）
    report_md: str
    sources: list[dict]  # Writer 产出的结构化参考来源（与 markdown 引用编号一致）
