"""Writer：汇总证据成带引用的 Markdown 报告。

Critic 发现的来源矛盾（conflicts）会显式传入 prompt，落到"争议与不确定性"一节——
此前该信息在状态里从未被消费，本版接通。

引用合并：LLM 时常为同一 URL 生成多个编号（实测 39% 重复率），后处理按 URL 去重、
重排编号并同步改写正文中的 [n] 上标，保证正文与参考来源表严格一致。

降级呈现（T1）：证据为空（全链检索失败）→ 报告头部显著标注"未联网核实"；
证据缺口/低置信来源 → 传入"研究局限性"章节要求（R7/R8/R13）。
"""

from __future__ import annotations

import json
import re

from deepdossier.agents.prompts import WRITER_SYSTEM, WRITER_USER
from deepdossier.agents.schemas import Report, SourceRef
from deepdossier.graph.deps import ResearchDeps
from deepdossier.graph.state import ResearchState

_NO_WEB_NOTICE = (
    "\n\n> ⚠️ **未联网核实声明**：本次调研所有检索源均不可用，"
    "以下内容完全基于模型内部知识生成，**未经联网核实**，时效性与事实准确性无法保证，请谨慎采信。\n"
)


def _limitations_block(state: ResearchState) -> str:
    """R8/R13：证据缺口（零证据子问题 + Critic 未充分结论）→ 局限性章节要求。"""
    lines: list[str] = []
    evidence_qids = {c.get("question_id") for c in state.get("evidence", [])}
    for sq in (state.get("plan") or {}).get("subquestions", []):
        if sq.get("id") not in evidence_qids:
            lines.append(f"- 子问题「{sq.get('question')}」（{sq.get('perspective', '')}）未获得任何证据")
    critique = state.get("critique") or {}
    if critique and not critique.get("sufficient", True):
        for g in critique.get("gaps", []) or []:
            if g:
                lines.append(f"- {g}")
        if critique.get("reasoning") and critique.get("reasoning", "").startswith("[降级]"):
            lines.append(f"- {critique['reasoning']}")
    if not lines:
        return "（无）"
    return "\n".join(lines)


def merge_citations(markdown: str, sources: list[SourceRef]) -> tuple[str, list[SourceRef]]:
    """按 URL 去重来源并重排编号，同步改写正文中的 [n] 引用。

    编号按原编号顺序保持稳定；正文中无对应来源的 [n] 原样保留（可能是普通文本）。
    """
    mapping: dict[int, int] = {}
    by_url: dict[str, int] = {}
    merged: list[SourceRef] = []
    for s in sorted(sources, key=lambda x: x.n):
        if s.url in by_url:
            mapping[s.n] = by_url[s.url]
            continue
        new_n = len(merged) + 1
        by_url[s.url] = new_n
        mapping[s.n] = new_n
        merged.append(SourceRef(n=new_n, title=s.title, url=s.url))

    def _sub(m: re.Match[str]) -> str:
        new = mapping.get(int(m.group(1)))
        return f"[{new}]" if new is not None else m.group(0)

    return re.sub(r"\[(\d+)\]", _sub, markdown), merged


def format_evidence(state: ResearchState) -> str:
    """把证据集合渲染为给 Writer/Verifier 的文本（两处共用同一格式）。"""
    return "\n".join(
        f"[{c['question_id']}] 主张: {c['claim']}\n  细节: {c['detail']}\n"
        f"  来源: {c['source_title']} | {c['source_url']} | 置信度 {c.get('confidence', 0.7)}"
        for c in state.get("evidence", [])
    ) or "（无证据）"


def _prepend_after_title(markdown: str, block: str) -> str:
    """在报告一级标题之后插入声明块（无标题时置于开头）。"""
    lines = markdown.split("\n")
    if lines and lines[0].lstrip().startswith("#"):
        return "\n".join([lines[0], block, *lines[1:]])
    return block + "\n" + markdown


def citation_coverage(report_md: str) -> tuple[float, int]:
    """引用覆盖率：疑似事实段落（>40 字的正文行，排除标题/列表/来源区）中有 [n] 引用的比例。

    返回 (覆盖率 0-1, 无引用的段落数)；粗略启发式，供质量事件与 UI 展示。
    """
    lines = [
        ln.strip()
        for ln in report_md.splitlines()
        if len(ln.strip()) > 40 and not ln.strip().startswith(("#", "-", "*", ">", "|", "```"))
    ]
    # 截掉文末参考来源区（列表行已被排除，但来源标题行仍可能是长正文行）
    src_idx = next((i for i, ln in enumerate(lines) if "参考来源" in ln), None)
    if src_idx is not None:
        lines = lines[:src_idx]
    if not lines:
        return 1.0, 0
    bare = [ln for ln in lines if not re.search(r"\[\d+\]", ln)]
    return 1 - len(bare) / len(lines), len(bare)


def make_writer_node(deps: ResearchDeps):
    async def writer_node(state: ResearchState) -> dict:
        emitter = deps.emitter
        emitter.emit("node_start", "writer", {})

        evidence_desc = format_evidence(state)

        conflicts = [c for c in (state.get("critique") or {}).get("conflicts", []) if c]
        conflicts_desc = "\n".join(f"- {c}" for c in conflicts) or "（无）"

        report: Report = await deps.provider.complete(  # type: ignore[assignment]
            system=WRITER_SYSTEM.format(language=deps.settings.report_language),
            user=WRITER_USER.format(
                question=state["question"],
                plan=json.dumps(state.get("plan", {}), ensure_ascii=False, indent=1),
                evidence=evidence_desc,
                conflicts=conflicts_desc,
                limitations=_limitations_block(state),
            ),
            schema=Report,
            fast=False,
            temperature=0.4,
        )
        report.markdown, sources = merge_citations(report.markdown, report.sources)

        # R7：全链检索失败（零证据）→ 报告头部显著标注"未联网核实"，事件流记录
        if not state.get("evidence"):
            report.markdown = _prepend_after_title(report.markdown, _NO_WEB_NOTICE)
            emitter.emit("degraded", "writer", {"kind": "no_web", "reason": "所有检索源不可用，产出纯模型知识报告"})

        coverage, bare = citation_coverage(report.markdown)
        emitter.emit("quality", "writer", {"citation_coverage": round(coverage, 3), "bare_paragraphs": bare})

        emitter.emit(
            "final",
            "writer",
            {"title": report.title, "sources": len(sources)},
        )
        return {
            "report_md": report.markdown,
            "sources": [s.model_dump() for s in sources],
        }

    return writer_node
