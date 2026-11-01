"""Researcher：单条查询的检索执行器。

由 LangGraph Send 并行扇出调用；每个实例负责一条查询：
搜索 → 来源过滤(curate) → 抓正文 → LLM 提炼 EvidenceCard。并行度由子问题/补搜查询数量决定。
（并行模式借鉴 GPT-Researcher，来源逐条标注与相关性过滤借鉴其 source-tracking/curation）

工程要点：
- 单 run 级 search/fetch 缓存：补搜轮与并行研究员不重复请求同一 URL（失败也缓存，避免反复撞反爬）
- 来源过滤 fail-open：过滤调用本身失败时保留全部结果，不因审核环节阻断研究
- 搜索零结果时直接产出空证据、跳过 LLM 调用（省钱省时）
"""

from __future__ import annotations

from deepdossier.agents.prompts import (
    CURATION_SYSTEM,
    CURATION_USER,
    RESEARCHER_SYSTEM,
    RESEARCHER_USER,
)
from deepdossier.agents.schemas import EvidenceBundle, SourceCuration
from deepdossier.graph.deps import ResearchDeps
from deepdossier.tools.fetch import fetch_readable
from deepdossier.tools.search import SearchResult

# Send 载荷键
KEY_QUERY = "query"
KEY_QUESTION_ID = "question_id"


def _engine_breakdown(results: list) -> dict[str, int]:
    """按 via 统计各引擎贡献（检索透明化）；无标注时返回空 dict。"""
    from collections import Counter

    return dict(Counter(r.via for r in results if getattr(r, "via", "")))


async def curate_sources(deps: ResearchDeps, query: str, results: list[SearchResult]) -> list[SearchResult]:
    """来源质量过滤：轻量模型按相关性筛掉低价值结果（GPT-Researcher 式 curation）。

    fail-open：过滤调用失败 → 保留全部；全判无关 → 兜底保留第一条，避免子问题断粮。
    """
    if not deps.settings.source_curation or len(results) <= 1:
        return results
    try:
        results_desc = "\n".join(
            f"{i}. {r.title} | {r.url}\n   {r.snippet[:150]}" for i, r in enumerate(results)
        )
        out: SourceCuration = await deps.provider.complete(  # type: ignore[assignment]
            system=CURATION_SYSTEM,
            user=CURATION_USER.format(query=query, n=len(results), results=results_desc),
            schema=SourceCuration,
            fast=True,
        )
        keep = sorted({i for i in out.keep if isinstance(i, int) and 0 <= i < len(results)})
        if not keep:
            keep = [0]
        filtered = [results[i] for i in keep]
        if len(filtered) < len(results):
            deps.emitter.emit(
                "curate",
                "researcher",
                {"query": query, "before": len(results), "after": len(filtered)},
            )
        return filtered
    except Exception as e:  # noqa: BLE001 — 审核环节失败不阻断研究
        deps.emitter.emit(
            "error",
            "researcher",
            {"message": f"来源过滤失败，保留全部 {len(results)} 条: {str(e)[:150]}"},
        )
        return results


def make_researcher_node(deps: ResearchDeps):
    async def researcher_node(state: dict) -> dict:
        emitter = deps.emitter
        query: str = state[KEY_QUERY]
        question_id: str = state[KEY_QUESTION_ID]
        emitter.emit("node_start", "researcher", {"question_id": question_id, "query": query})

        # 1) 搜索（单 run 缓存）
        cached = deps.search_cache.get(query)
        if cached is not None:
            results = cached
        else:
            try:
                results = await deps.retriever.search(query, deps.settings.search_top_k)
            except Exception as e:  # noqa: BLE001 — 搜索失败降级为无结果继续
                results = []
                emitter.emit("error", "researcher", {"message": f"搜索失败({question_id}): {e}"})
            deps.search_cache[query] = results
        emitter.emit(
            "tool_call",
            "researcher",
            {
                "tool": deps.retriever.name,
                "query": query,
                "results": len(results),
                "cached": cached is not None,
                "engines": _engine_breakdown(results),  # 检索透明化：各引擎/数据源贡献
            },
        )
        # F11 降级可见：实际贡献引擎不含主检索器时显式标记
        breakdown = _engine_breakdown(results)
        if results and breakdown and deps.settings.retriever not in breakdown:
            emitter.emit(
                "fallback",
                "researcher",
                {"expected": deps.settings.retriever, "used": list(breakdown), "query": query},
            )

        # 2) 来源质量过滤（可配置关闭；fail-open）
        results = await curate_sources(deps, query, results)
        if not results:
            emitter.emit(
                "evidence_added",
                "researcher",
                {"question_id": question_id, "count": 0, "note": "搜索零结果，跳过 LLM 提炼"},
            )
            return {"evidence": []}

        # 3) 抓正文（取前 N 条，单页失败跳过；fetch_pages=false 时离线跳过）
        page_parts: list[str] = []
        if deps.settings.fetch_pages:
            for r in results[: deps.settings.max_sources_per_question]:
                if not r.url:
                    continue  # MCP 等数据源可能只给摘要不给 URL
                if r.url in deps.fetch_cache:
                    text, err = deps.fetch_cache[r.url]
                else:
                    text, err = await fetch_readable(deps.http, r.url, max_chars=3000)
                    deps.fetch_cache[r.url] = (text, err)
                if text:
                    page_parts.append(f"### 来源: {r.title}\nURL: {r.url}\n{text}")
                elif err:
                    emitter.emit("error", "researcher", {"message": f"抓取失败 {r.url}: {err}"})

        search_desc = "\n".join(f"- {r.title} | {r.url}\n  {r.snippet[:200]}" for r in results)
        pages_desc = (
            "\n\n".join(page_parts)
            if page_parts
            else ("（离线模式：仅依据搜索摘要判断）" if not deps.settings.fetch_pages else "（正文抓取失败，仅依据搜索摘要判断）")
        )

        # 4) 提炼证据卡片；R2/R4 节点级降级：提炼彻底失败 → 空证据，整 run 不失败
        try:
            bundle: EvidenceBundle = await deps.provider.complete(  # type: ignore[assignment]
                system=RESEARCHER_SYSTEM,
                user=RESEARCHER_USER.format(
                    question=state["original_question"],
                    question_id=question_id,
                    sub_question=query,
                    search_results=search_desc,
                    pages=pages_desc,
                ),
                schema=EvidenceBundle,
                fast=True,
            )
        except Exception as e:  # noqa: BLE001
            emitter.emit(
                "error",
                "researcher",
                {"message": f"证据提炼失败（{question_id}），该子问题降级为无证据: {str(e)[:150]}"},
            )
            return {"evidence": []}
        cards = bundle.evidence
        # R9：有搜索结果但正文全部抓取失败 → 本次提炼仅基于搜索摘要，证据卡显式标注
        snippet_only = bool(results) and not page_parts
        for c in cards:
            c.question_id = question_id  # 与状态中的子问题 id 严格对齐
            if snippet_only:
                c.snippet_only = True

        emitter.emit(
            "evidence_added",
            "researcher",
            {"question_id": question_id, "count": len(cards), "snippet_only": snippet_only},
        )
        return {"evidence": [c.model_dump() for c in cards]}

    return researcher_node
