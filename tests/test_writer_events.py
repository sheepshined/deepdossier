"""Writer conflicts 接入 + EventWriter 线程行为测试。"""

from __future__ import annotations

import asyncio

from deepdossier.agents.schemas import Critique
from deepdossier.store.writer import EventWriter
from deepdossier.trace import EventEmitter


async def test_writer_receives_conflicts(settings, store, stub_provider, stub_retriever):
    """Critic 产出的 conflicts 必须进入 Writer 的 prompt（此前的设计断层回归）。

    用一个记录 prompt 的 Provider 代理 StubProvider：critic 第 1 轮固定产出
    conflicts=["来源A与来源B在增速上矛盾"]，随后断言 writer 收到的 user prompt 包含该矛盾。
    """
    captured: dict[str, str] = {}

    class SpyProvider(stub_provider.__class__):
        async def complete(self, *, system, user, schema=None, **kw):
            if schema is not None and schema.__name__ == "Critique":
                result: Critique = await super().complete(system=system, user=user, schema=schema, **kw)
                result.conflicts = ["来源A与来源B在市场增速数据上存在矛盾"]
                return result
            if schema is not None and schema.__name__ == "Report":
                captured["writer_user"] = user
            return await super().complete(system=system, user=user, schema=schema, **kw)

    from deepdossier.runner import run_research

    run_id = store.new_run_id()
    store.create_run(run_id, "矛盾传递测试")
    await run_research(
        question="矛盾传递测试", settings=settings, store=store, run_id=run_id,
        provider=SpyProvider(), retriever=stub_retriever,
    )
    assert "来源A与来源B在市场增速数据上存在矛盾" in captured.get("writer_user", ""), (
        "Critic 发现的矛盾必须显式传入 Writer prompt"
    )


async def test_event_writer_batches_and_notifies(store):
    persisted: list[str] = []
    writer = EventWriter(store, on_persisted=lambda rid: persisted.append(rid))
    writer.start()

    emitter = EventEmitter("rw", sinks=[writer.submit])
    for i in range(5):
        emitter.emit("node_start", "planner", {"i": i})
    writer.close()

    events = store.list_events("rw")
    assert [e["seq"] for e in events] == [1, 2, 3, 4, 5]
    assert persisted, "落库回调未触发（SSE 通知依赖它）"


async def test_event_writer_failure_does_not_lose_loop(store):
    """store 写失败不影响 close 退出（观测层故障不阻断主流程）。"""

    class BrokenStore:
        def append_events(self, events):
            raise RuntimeError("db exploded")

    writer = EventWriter(BrokenStore())  # type: ignore[arg-type]
    writer.start()
    emitter = EventEmitter("rx", sinks=[writer.submit])
    emitter.emit("node_start", "planner", {})
    await asyncio.sleep(0.1)
    writer.close(timeout=2)
    assert not writer.is_alive()
