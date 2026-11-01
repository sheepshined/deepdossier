"""Critic：批判性评审，证据不足时触发定向补搜循环（借鉴 TradingAgents 对抗辩论机制）。"""

from __future__ import annotations

import json

from deepdossier.agents.prompts import CRITIC_SYSTEM, CRITIC_USER
from deepdossier.agents.schemas import Critique
from deepdossier.graph.deps import ResearchDeps
from deepdossier.graph.state import ResearchState


def make_critic_node(deps: ResearchDeps):
    async def critic_node(state: ResearchState) -> dict:
        emitter = deps.emitter
        emitter.emit("node_start", "critic", {"round": state.get("round", 1)})

        evidence_desc = "\n".join(
            f"[{c['question_id']}] {c['claim']} (置信度 {c.get('confidence', 0.7)}, 来源 {c['source_url']})"
            for c in state.get("evidence", [])
        ) or "（尚无证据）"

        # R4 节点级降级：Critic 彻底失败 → 视为证据充分直接进 Writer，整 run 不失败
        try:
            critique: Critique = await deps.provider.complete(  # type: ignore[assignment]
                system=CRITIC_SYSTEM,
                user=CRITIC_USER.format(
                    plan=json.dumps(state.get("plan", {}), ensure_ascii=False, indent=1),
                    round=state.get("round", 1),
                    evidence=evidence_desc,
                ),
                schema=Critique,
                fast=True,
            )
        except Exception as e:  # noqa: BLE001 — 降级为"已充分"，保留已收集证据继续成稿
            emitter.emit("error", "critic", {"message": f"评审失败，降级为直接成稿: {str(e)[:150]}"})
            critique = Critique(
                sufficient=True,
                gaps=[],
                conflicts=[],
                followup_queries=[],
                reasoning="[降级] Critic 调用失败，跳过批判环节直接成稿",
            )
        emitter.emit("critique", "critic", critique.model_dump())

        # 决策收口在节点内：sufficient / 轮次用尽 / 无补搜词 任一成立即停止循环，
        # 路由器只负责按 pending_followups 扇出（纯函数，避免 round 递增时序陷阱）。
        current_round = state.get("round", 1)
        followups = critique.followup_queries[:3]
        can_continue = not critique.sufficient and current_round < deps.max_rounds and bool(followups)

        updates: dict = {
            "critique": critique.model_dump(),
            "pending_followups": followups if can_continue else [],
        }
        if can_continue:
            updates["round"] = current_round + 1

        emitter.emit(
            "node_end",
            "critic",
            {
                "sufficient": critique.sufficient,
                "round": current_round,
                "followups": followups if can_continue else [],
                "next_round": current_round + 1 if can_continue else None,
            },
        )
        return updates

    return critic_node
