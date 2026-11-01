"""LangGraph 编排组装。

图结构：
    START → router ─(direct)→ direct_answer → END
              │(research)
              ▼
           planner →(Send 扇出)→ researcher×N → critic
                        ▲                      │ insufficient & 轮次未满
                        └──── 定向补搜(Send) ←──┘
                                               │ sufficient / 轮次用尽
                                               ▼
                                             writer → verifier ─(修订后)→ END
    （verifier：草稿级事实审查，有问题则单轮修订；可经 draft_review 关闭）
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from deepdossier.agents.critic import make_critic_node
from deepdossier.agents.planner import make_planner_node
from deepdossier.agents.researcher import KEY_QUERY, KEY_QUESTION_ID, make_researcher_node
from deepdossier.agents.router import make_direct_answer_node, make_router_node
from deepdossier.agents.verifier import make_verifier_node
from deepdossier.agents.writer import make_writer_node
from deepdossier.graph.deps import ResearchDeps
from deepdossier.graph.state import ResearchState


def _research_send(run_id: str, question: str, query: str, question_id: str, round_no: int) -> Send:
    return Send(
        "researcher",
        {
            "run_id": run_id,
            "original_question": question,
            KEY_QUERY: query,
            KEY_QUESTION_ID: question_id,
            "round": round_no,
        },
    )


def _fan_out_after_planner(state: ResearchState) -> list[Send]:
    plan = state["plan"]
    return [
        _research_send(state["run_id"], state["question"], sq["question"], sq["id"], state.get("round", 1))
        for sq in plan["subquestions"]
    ]


def _route_after_critic(state: ResearchState) -> str | list[Send]:
    """纯路由：Critic 已把补搜决策写入 pending_followups，这里只做扇出。"""
    pending = state.get("pending_followups") or []
    if not pending:
        return "writer"
    return [
        _research_send(state["run_id"], state["question"], q, f"gap{i + 1}", state.get("round", 1))
        for i, q in enumerate(pending)
    ]


def build_graph(deps: ResearchDeps, checkpointer=None):
    g = StateGraph(ResearchState)
    g.add_node("router", make_router_node(deps))
    g.add_node("direct_answer", make_direct_answer_node(deps))
    g.add_node("planner", make_planner_node(deps))
    g.add_node("researcher", make_researcher_node(deps))
    g.add_node("critic", make_critic_node(deps))
    g.add_node("writer", make_writer_node(deps))
    g.add_node("verifier", make_verifier_node(deps))

    g.add_edge(START, "router")
    g.add_conditional_edges("router", _route_after_router, ["planner", "direct_answer"])
    g.add_conditional_edges("planner", _fan_out_after_planner, ["researcher"])
    g.add_edge("researcher", "critic")
    g.add_conditional_edges("critic", _route_after_critic, ["researcher", "writer"])
    g.add_edge("writer", "verifier")
    g.add_edge("verifier", END)
    g.add_edge("direct_answer", END)
    return g.compile(checkpointer=checkpointer)


def _route_after_router(state: ResearchState) -> str:
    return "direct_answer" if state.get("route") == "direct" else "planner"
