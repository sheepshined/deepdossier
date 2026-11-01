"""CLI：research（执行研究）/ serve（启动 API）/ demo（离线演示）。"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown

from deepdossier.config import load_settings
from deepdossier.store.runs import RunStore
from deepdossier.trace import console_sink

app = typer.Typer(name="deepdossier", help="DeepDossier — 深度调研智能体", no_args_is_help=True)
console = Console()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


@app.command()
def research(
    question: str = typer.Argument(..., help="调研问题"),
    depth: int | None = typer.Option(None, "--depth", "-d", min=1, max=5, help="最大补搜轮数"),
    output: Path | None = typer.Option(None, "--output", "-o", help="报告输出 Markdown 路径"),
    run_id: str | None = typer.Option(
        None, "--run-id", help="指定 run_id（便于日志对账；同名 run 会覆盖展示，非断点续跑）"
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="不打印实时事件"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="DEBUG 日志"),
) -> None:
    """执行一次深度调研并输出 Markdown 报告。"""
    _setup_logging(verbose)
    settings = load_settings()
    store = RunStore(settings.db_path)
    rid = run_id or store.new_run_id()
    if not store.get_run(rid):
        store.create_run(rid, question)

    sinks = [] if quiet else [console_sink()]
    with console.status("DeepDossier 研究中…", spinner="dots"):
        result = asyncio.run(
            _run(settings, store, question, rid, depth, sinks)
        )
    report = result["report_md"]
    console.rule(f"[bold cyan]{result.get('run_id')}[/]")
    console.print(Markdown(report))
    if output:
        Path(output).write_text(report, encoding="utf-8")
        console.print(f"[green]报告已写入[/] {output}")


async def _run(settings, store, question, run_id, depth, sinks):
    from deepdossier.runner import run_research

    return await run_research(
        question=question, settings=settings, store=store, run_id=run_id,
        max_rounds=depth, extra_sinks=sinks,
    )


@app.command()
def serve(
    host: str | None = typer.Option(None, "--host"),
    port: int | None = typer.Option(None, "--port", min=1, max=65535),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="DEBUG 日志"),
) -> None:
    """启动 Web 服务（API + 时间线前端）。"""
    _setup_logging(verbose)
    import uvicorn

    settings = load_settings()
    uvicorn.run(
        "deepdossier.api.app:create_app",
        factory=True,
        host=host or settings.host,
        port=port or settings.port,
    )


@app.command()
def demo(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """离线演示：Stub LLM + Stub 搜索，无需任何 API key。"""
    _setup_logging(verbose)
    settings = load_settings(llm_provider="stub", retriever="stub", fetch_pages=False)
    store = RunStore(settings.db_path)
    question = "多智能体系统在企业的落地路径与风险？"
    run_id = store.new_run_id()
    store.create_run(run_id, question)
    result = asyncio.run(
        _run(settings, store, question, run_id, None, [console_sink()])
    )
    console.rule("[bold cyan]离线演示报告[/]")
    console.print(Markdown(result["report_md"]))


if __name__ == "__main__":
    app()
