"""Verifier：草稿级事实审查（draft → verify → revise）+ 质量硬门禁（T2）。

四道防线：
1. LLM 事实审查：幻觉引用、无支撑断言、编号错位（按需单轮修订，fail-open）
2. 程序化结构契约（R23/R25/R18）：必需章节、引用编号连续/越界、conflicts 必须有争议章节
3. 引用覆盖率门禁（R17/R21）：<80% 强制修订一轮；修订后仍 <80% → 报告顶部标注"引用不充分"
4. 修订回退护栏（R19）：修订后覆盖率下降或异常缩水 → 放弃修订保留原稿
"""

from __future__ import annotations

import re

from deepdossier.agents.prompts import REVISE_SYSTEM, REVISE_USER, VERIFIER_SYSTEM, VERIFIER_USER
from deepdossier.agents.schemas import DraftReview, Report, SourceRef
from deepdossier.agents.writer import citation_coverage, format_evidence, merge_citations
from deepdossier.graph.deps import ResearchDeps
from deepdossier.graph.state import ResearchState

COVERAGE_GATE = 0.80  # R21：低于此线强制修订；修订后仍不达标 → 顶部降级标注

_COVERAGE_NOTICE = (
    "\n\n> ⚠️ **引用不充分**：本报告引用覆盖率为 {cov:.0%}（低于 80% 门禁），"
    "部分结论缺少引用支撑，采信前请自行核实。\n"
)


def structural_issues(report_md: str, sources: list, conflicts: list | None = None) -> list[str]:
    """程序化校验报告结构契约（R23）。

    校验项：必需章节（摘要/主要发现/争议与不确定性/参考来源）、正文引用编号不越界、
    参考来源编号无断号、Critic conflicts 非空时必须有争议章节（R25）。
    仅对成篇报告（>500 字）生效。
    """
    if len(report_md.strip()) < 500:
        return []
    issues: list[str] = []
    for section in ("摘要", "主要发现", "争议与不确定性", "参考来源"):
        if section not in report_md:
            issues.append(f"缺少必需章节「{section}」")
    if conflicts and "争议与不确定性" not in report_md:
        issues.append("Critic 发现了来源矛盾，但报告缺少「争议与不确定性」章节（R25）")
    if sources:
        max_n = len(sources)
        cited = {int(x) for x in re.findall(r"\[(\d+)\]", report_md)}
        # 越界判定：n<1，或略超来源数（≤max_n+50；更大的数视为年份等普通文本，避免 [2026] 误报）
        over = sorted(n for n in cited if n < 1 or (n > max_n and n <= max_n + 50))
        if over:
            issues.append(f"正文引用编号超出来源范围（来源共 {max_n} 条）：{over[:5]}")
        src_section = report_md.split("参考来源", 1)[-1]
        listed = {int(x) for x in re.findall(r"^\s*(\d+)\.\s", src_section, re.M)}
        missing = sorted(set(range(1, max_n + 1)) - listed)
        if missing:
            issues.append(f"参考来源列表缺少编号：{missing[:5]}")
    return issues


def hallucinated_sources(sources: list[dict], evidence: list[dict]) -> list[dict]:
    """R24：来源 URL 不在证据集合中 → 疑似幻觉引用（LLM 凭空造来源）。"""
    evidence_urls = {e.get("source_url") for e in evidence if e.get("source_url")}
    return [s for s in sources if s.get("url") and s["url"] not in evidence_urls]


def make_verifier_node(deps: ResearchDeps):
    async def verifier_node(state: ResearchState) -> dict:
        emitter = deps.emitter
        emitter.emit("node_start", "verifier", {})

        report_md = state.get("report_md", "")
        sources = state.get("sources", [])
        evidence = state.get("evidence", [])
        if not report_md:
            emitter.emit("node_end", "verifier", {"skipped": True})
            return {"report_md": report_md, "sources": sources}

        cov_before, _ = citation_coverage(report_md)
        # 程序化契约：结构 + 幻觉引用（不依赖 LLM，draft_review 关闭也生效）
        structural = structural_issues(
            report_md, sources, (state.get("critique") or {}).get("conflicts")
        )
        halluc = hallucinated_sources(sources, evidence)
        if halluc:
            emitter.emit(
                "hallucination", "verifier",
                {"sources": [s.get("url") for s in halluc][:5], "count": len(halluc)},
            )

        # R17/R21 门禁：覆盖率 <80% 强制修订；draft_review 关闭时仅标注不修订
        gate_issues: list[str] = []
        if cov_before < COVERAGE_GATE:
            gate_issues.append(f"引用覆盖率仅 {cov_before:.0%}，低于 {COVERAGE_GATE:.0%} 门禁")

        if not deps.settings.draft_review:
            notice = _COVERAGE_NOTICE.format(cov=cov_before) if cov_before < COVERAGE_GATE else ""
            emitter.emit("node_end", "verifier", {"skipped": True, "coverage_gate": cov_before < COVERAGE_GATE})
            return {"report_md": (report_md + notice) if notice else report_md, "sources": sources}

        issues: list[str] = []
        review: DraftReview | None = None
        try:
            review = await deps.provider.complete(  # type: ignore[assignment]
                system=VERIFIER_SYSTEM,
                user=VERIFIER_USER.format(evidence=format_evidence(state), draft=report_md[:12000]),
                schema=DraftReview,
                fast=False,
                temperature=0.2,
            )
            issues = [i for i in (review.issues or []) if i][:8]
        except Exception as e:  # noqa: BLE001 — LLM 审查失败放行原稿（fail-open）
            emitter.emit("error", "verifier", {"message": f"草稿审查失败: {str(e)[:150]}"})

        issues.extend(structural[:4])
        issues.extend(gate_issues)
        if halluc:
            issues.append(
                "以下参考来源未出现在证据集合中（疑似幻觉引用），请核对或剔除："
                + "；".join(f"{s.get('title') or s.get('url')}" for s in halluc[:4])
            )

        # 任一来源（LLM/结构契约/覆盖率门禁/幻觉检测）发现问题即进入修订
        needs_revision = bool(issues)
        emitter.emit(
            "draft_review",
            "verifier",
            {"needs_revision": needs_revision, "issues": issues[:8]},
        )

        if not needs_revision:
            emitter.emit("node_end", "verifier", {"revised": False, "issues": len(issues)})
            return {"report_md": report_md, "sources": sources}

        # ---- 单轮修订 ----
        try:
            revised = await deps.provider.complete(
                system=REVISE_SYSTEM,
                user=REVISE_USER.format(
                    draft=report_md[:12000],
                    issues="\n".join(f"- {x}" for x in issues[:10]),
                    evidence=format_evidence(state)[:8000],
                ),
                temperature=0.4,
            )
        except Exception as e:  # noqa: BLE001 — 修订失败放行原稿
            emitter.emit("error", "verifier", {"message": f"草稿修订失败，原样放行: {str(e)[:150]}"})
            emitter.emit("node_end", "verifier", {"revised": False, "issues": len(issues)})
            return {"report_md": report_md, "sources": sources}

        revised_md = revised.markdown if isinstance(revised, Report) else str(revised)
        if len(revised_md.strip()) < len(report_md.strip()) * 0.5:
            emitter.emit("error", "verifier", {"message": "修订输出异常缩水，保留原稿"})
            emitter.emit("node_end", "verifier", {"revised": False, "issues": len(issues)})
            return {"report_md": report_md, "sources": sources}

        new_md, new_sources = merge_citations(revised_md, [SourceRef(**s) for s in sources])
        cov_after, _ = citation_coverage(new_md)
        # R19 回退护栏：修订后覆盖率下降 → 放弃修订保留原稿
        if cov_after < cov_before:
            emitter.emit(
                "error", "verifier",
                {"message": f"修订后引用覆盖率下降（{cov_before:.0%}→{cov_after:.0%}），放弃修订保留原稿"},
            )
            emitter.emit("node_end", "verifier", {"revised": False, "reverted": True})
            return {"report_md": report_md, "sources": sources}

        # R21：修订后仍不达标 → 顶部降级标注（诚实交付而非静默放行）
        if cov_after < COVERAGE_GATE:
            new_md += _COVERAGE_NOTICE.format(cov=cov_after)

        emitter.emit("quality", "verifier", {"citation_coverage": round(cov_after, 3)})
        emitter.emit("node_end", "verifier", {"revised": True, "issues": len(issues)})
        return {"report_md": new_md, "sources": [s.model_dump() for s in new_sources]}

    return verifier_node
