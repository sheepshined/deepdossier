"""问题路由测试：路由节点 + 直接回答节点（全离线 fake provider）。"""

from __future__ import annotations

from deepdossier.agents.router import make_direct_answer_node, make_router_node
from deepdossier.agents.schemas import QuestionRoute
from deepdossier.config import Settings


class _FakeEmitter:
    def __init__(self) -> None:
        self.events: list[tuple] = []

    def emit(self, type_, node, payload):
        self.events.append((type_, node, payload))


class _Deps:
    def __init__(self, provider, question_routing=True):
        self.settings = Settings(question_routing=question_routing)
        self.provider = provider
        self.emitter = _FakeEmitter()


class _RouteProvider:
    def __init__(self, route: str = "research", reason: str = "r") -> None:
        self.route = route
        self.reason = reason
        self.calls = 0

    async def complete(self, *, system, user, schema=None, **kw):
        self.calls += 1
        if schema is QuestionRoute:
            return QuestionRoute(route=self.route, reason=self.reason)
        return "这是对简单询问的直接回答。"


class _BoomProvider:
    async def complete(self, *, system, user, schema=None, **kw):
        raise RuntimeError("模型故障")


async def test_router_direct():
    deps = _Deps(_RouteProvider("direct", "闲聊"))
    out = await make_router_node(deps)({"run_id": "r1", "question": "你是做什么的？"})
    assert out["route"] == "direct"
    assert any(e[0] == "route" and e[2]["route"] == "direct" for e in deps.emitter.events)


async def test_router_research_and_invalid_value_fails_closed_to_research():
    deps = _Deps(_RouteProvider("research"))
    out = await make_router_node(deps)({"run_id": "r1", "question": "企业落地路径？"})
    assert out["route"] == "research"

    deps2 = _Deps(_RouteProvider("不知道"))  # 模型输出非法值 → 回落 research
    out2 = await make_router_node(deps2)({"run_id": "r1", "question": "任意"})
    assert out2["route"] == "research"


async def test_router_fail_open_on_error():
    deps = _Deps(_BoomProvider())
    out = await make_router_node(deps)({"run_id": "r1", "question": "任意"})
    assert out["route"] == "research"
    assert any(e[0] == "error" for e in deps.emitter.events)


async def test_router_disabled():
    p = _RouteProvider("direct")
    deps = _Deps(p, question_routing=False)
    out = await make_router_node(deps)({"run_id": "r1", "question": "你好"})
    assert out["route"] == "research"  # 关闭时一律完整调研
    assert p.calls == 0  # 未触发分类调用


async def test_direct_answer_node_renders_markdown():
    deps = _Deps(_RouteProvider("direct"))
    out = await make_direct_answer_node(deps)({"run_id": "r1", "question": "你是谁？"})
    assert out["sources"] == []
    assert out["report_md"].startswith("# 直接回答")
    assert "这是对简单询问的直接回答。" in out["report_md"]
    assert "问题路由" in out["report_md"]
