"""事实一致性抽查基准执行器（R22）。

用法：
    python scripts/benchmark_check.py --stub   # CI：stub 模式验证流水线完整性 + 报告结构
    python scripts/benchmark_check.py --real   # 手动：真实 LLM/检索跑全部基准题，输出事实命中评审

真实模式对每题跑完整调研，再用 LLM 逐条判定关键事实点是否被报告覆盖，输出命中率汇总。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

BENCH = Path(__file__).parent.parent / "benchmarks" / "fact_benchmark.json"

JUDGE_SYSTEM = (
    "你是事实核查员。给你一份调研报告与一组关键事实点，逐条判断报告是否覆盖/支持该事实点。"
    '输出 JSON：{"hits": [{"point": 0, "covered": true, "evidence": "报告中的原句片段"}]}'
)


def load_benchmark() -> list[dict]:
    data = json.loads(BENCH.read_text(encoding="utf-8"))
    assert data.get("version") == 1 and len(data["questions"]) >= 10, "基准集结构不完整"
    for q in data["questions"]:
        assert q.get("question") and len(q.get("fact_points", [])) >= 3, f"{q.get('id')} 缺事实点"
    return data["questions"]


async def run_stub(q: dict) -> bool:
    """CI 用：stub 模式跑通全链路 + 报告结构校验。"""
    from deepdossier.config import load_settings
    from deepdossier.runner import run_research
    from deepdossier.store.runs import RunStore

    settings = load_settings(llm_provider="stub", retriever="stub", fetch_pages=False,
                             data_dir=Path("./data"))
    store = RunStore(settings.db_path)
    run_id = store.new_run_id()
    store.create_run(run_id, q["question"])
    try:
        result = await run_research(q["question"], settings, store, run_id)
        md = result["report_md"]
        ok = result["status"] == "completed" and bool(md)
        return ok
    finally:
        store.close()


async def run_real(questions: list[dict]) -> None:
    """手动用：真实配置跑基准题 + LLM 判定事实命中。"""
    from deepdossier.config import load_settings
    from deepdossier.llm.provider import build_provider
    from deepdossier.runner import run_research
    from deepdossier.store.runs import RunStore

    settings = load_settings()
    provider = build_provider(settings)
    store = RunStore(settings.db_path)
    total_hits, total_points = 0, 0
    try:
        for q in questions:
            run_id = store.new_run_id()
            store.create_run(run_id, q["question"])
            print(f"[{q['id']}] 调研中: {q['question'][:40]}…")
            try:
                result = await run_research(q["question"], settings, store, run_id)
            except Exception as e:  # noqa: BLE001
                print(f"  失败: {e}")
                continue
            judge_user = (
                "关键事实点：\n" + "\n".join(f"{i}. {p}" for i, p in enumerate(q["fact_points"]))
                + f"\n\n报告：\n{result['report_md'][:12000]}"
            )
            verdict = await provider.complete(system=JUDGE_SYSTEM, user=judge_user, fast=True)
            try:
                hits = json.loads(verdict).get("hits", [])
            except json.JSONDecodeError:
                print("  评审输出不可解析，跳过")
                continue
            covered = sum(1 for h in hits if h.get("covered"))
            total_hits += covered
            total_points += len(q["fact_points"])
            print(f"  命中 {covered}/{len(q['fact_points'])}")
    finally:
        store.close()
        close = getattr(provider, "aclose", None)
        if close:
            await close()
    if total_points:
        print(f"\n事实命中率：{total_hits}/{total_points} = {total_hits / total_points:.0%}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stub", action="store_true", help="CI 模式：stub 流水线完整性")
    parser.add_argument("--real", action="store_true", help="真实模式：事实命中评审（消耗 token）")
    parser.add_argument("--only", type=str, default="", help="只跑指定 id 的题目（real 模式）")
    args = parser.parse_args()

    questions = load_benchmark()
    if args.only:
        questions = [q for q in questions if q["id"] == args.only]

    if args.real:
        asyncio.run(run_real(questions))
    else:
        ok = asyncio.run(run_stub(questions[0]))
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
