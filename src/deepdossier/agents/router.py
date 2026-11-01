"""Router：开头问题路由 + 直接回答节点。

路由意义：闲聊/系统询问/无需检索的问题不进检索流水线（此前"你是做什么的"会被
拆解成 5 个子问题空转一轮），路由到 direct 分支用快模型直接作答，省时省钱省检索配额。
路由 fail-open：分类调用失败一律走 research（完整流水线是安全默认）。
"""

from __future__ import annotations

from deepdossier.agents.prompts import DIRECT_SYSTEM, DIRECT_USER, ROUTE_SYSTEM, ROUTE_USER
from deepdossier.agents.schemas import QuestionRoute
from deepdossier.graph.deps import ResearchDeps
from deepdossier.graph.state import ResearchState


def make_router_node(deps: ResearchDeps):
    async def router_node(state: ResearchState) -> dict:
        emitter = deps.emitter
        emitter.emit("node_start", "router", {"detail": state["question"]})

        if not deps.settings.question_routing:
            emitter.emit("route", "router", {"route": "research", "reason": "路由已关闭"})
            return {"route": "research"}

        out: QuestionRoute | None = None
        try:
            out = await deps.provider.complete(  # type: ignore[assignment]
                system=ROUTE_SYSTEM,
                user=ROUTE_USER.format(question=state["question"]),
                schema=QuestionRoute,
                fast=True,
            )
            route = out.route if out.route in ("research", "direct") else "research"
        except Exception as e:  # noqa: BLE001 — 路由失败 fail-open 走完整调研
            emitter.emit("error", "router", {"message": f"路由分类失败，走完整调研: {str(e)[:150]}"})
            route = "research"

        reason = out.reason.strip()[:200] if out is not None else ""
        emitter.emit("route", "router", {"route": route, "reason": reason})
        return {"route": route}

    return router_node


def make_direct_answer_node(deps: ResearchDeps):
    async def direct_answer_node(state: ResearchState) -> dict:
        emitter = deps.emitter
        emitter.emit("node_start", "direct_answer", {"detail": state["question"]})

        # R36：直接回答跟随报告语言配置
        system = DIRECT_SYSTEM
        if deps.settings.report_language != "zh":
            system += "\nPlease answer in " + deps.settings.report_language + "."
        text = await deps.provider.complete(
            system=system,
            user=DIRECT_USER.format(question=state["question"]),
            fast=True,
        )
        answer = text if isinstance(text, str) else str(text)
        md = (
            f"# 直接回答\n\n{answer.strip()}\n\n"
            "---\n\n*本输入被问题路由识别为简单询问，未启动联网调研流程；"
            "如需就该主题生成带引用的深度报告，请换用研究型提问。*"
        )
        emitter.emit("final", "direct_answer", {"title": "直接回答"})
        return {"report_md": md, "sources": []}

    return direct_answer_node
