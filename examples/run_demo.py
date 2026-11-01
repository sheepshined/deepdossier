"""离线全链路演示（等价于 `deepdossier demo`）：Stub LLM + Stub 搜索，无需任何 API key。

用法：
    python examples/run_demo.py
"""

import asyncio

from rich.console import Console
from rich.markdown import Markdown

from deepdossier.config import load_settings
from deepdossier.runner import run_research
from deepdossier.store.runs import RunStore
from deepdossier.trace import console_sink

console = Console()


async def main() -> None:
    settings = load_settings(llm_provider="stub", retriever="stub", fetch_pages=False)
    store = RunStore(settings.db_path)
    question = "多智能体系统在企业的落地路径与风险？"
    run_id = store.new_run_id()
    store.create_run(run_id, question)

    result = await run_research(
        question, settings, store, run_id, extra_sinks=[console_sink()]
    )
    console.rule("[bold cyan]DeepDossier 离线演示报告[/]")
    console.print(Markdown(result["report_md"]))
    console.print(f"[muted]run_id: {run_id}，事件与 trace 已落库 {settings.data_dir}[/]")


if __name__ == "__main__":
    asyncio.run(main())
