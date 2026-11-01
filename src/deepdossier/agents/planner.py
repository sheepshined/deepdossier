"""Planner：多视角拆解调研问题（借鉴 STORM 视角引导提问）+ 注入策略记忆（借鉴 TradingAgents 反思记忆）。

记忆检索：配置 embedding 时用语义相似度（余弦），否则退化为字符 bigram Jaccard（中文友好）。
"""

from __future__ import annotations

import asyncio

from deepdossier.agents.prompts import PLANNER_SYSTEM, PLANNER_USER
from deepdossier.agents.schemas import ResearchPlan
from deepdossier.graph.deps import ResearchDeps
from deepdossier.graph.state import ResearchState
from deepdossier.llm.embed import embed_question


def make_planner_node(deps: ResearchDeps):
    async def planner_node(state: ResearchState) -> dict:
        emitter = deps.emitter
        emitter.emit("node_start", "planner", {"detail": state["question"]})

        query_vec = await embed_question(deps.embedder, deps.store, state["question"])
        lessons = await asyncio.to_thread(
            deps.store.get_lessons, state["question"], 3, query_vec
        )
        plan: ResearchPlan = await deps.provider.complete(  # type: ignore[assignment]
            system=PLANNER_SYSTEM,
            user=PLANNER_USER.format(
                question=state["question"],
                lessons="\n".join(f"- {x}" for x in lessons) or "（无）",
            ),
            schema=ResearchPlan,
            fast=True,
        )
        plan.subquestions = plan.subquestions[: deps.settings.max_subquestions]

        emitter.emit(
            "node_end",
            "planner",
            {"title": plan.title, "subquestions": [sq.model_dump() for sq in plan.subquestions]},
        )
        return {"plan": plan.model_dump(), "round": 1}

    return planner_node
