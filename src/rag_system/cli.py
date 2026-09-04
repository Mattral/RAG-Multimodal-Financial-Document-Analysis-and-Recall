"""Enterprise CLI for the RAG Financial Multimodal system.

Commands:
  ingest   — Parse, embed, and index financial PDFs
  query    — Retrieve and answer from indexed documents
  evaluate — Run golden-dataset quality evals
  serve    — Start the FastAPI server
  health   — Check system/component health
  version  — Show version info
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, List, Optional

import structlog
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.syntax import Syntax
from rich.table import Table

from src.rag_system.utils.logger import setup_logging

app = typer.Typer(
    name="rag-financial",
    help="🏦 Enterprise Multimodal RAG for Financial Documents",
    pretty_exceptions_show_locals=False,
    rich_markup_mode="rich",
)
console = Console()
logger = structlog.get_logger(__name__)


def _version_callback(value: bool):
    if value:
        console.print("[bold cyan]RAG Financial Multimodal[/bold cyan] [green]v2.0.0[/green]")
        console.print(
            "Enterprise-grade multimodal RAG for financial document analysis"
        )
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", "-v", callback=_version_callback, is_eager=True
    ),
):
    pass


# ────────────────────────────────────────────────────────────────────────────────
# INGEST
# ────────────────────────────────────────────────────────────────────────────────


@app.command()
def ingest(
    files: Annotated[List[str], typer.Argument(help="PDF file paths to ingest")],
    tenant: Annotated[str, typer.Option("--tenant", "-t", help="Tenant ID")] = "default",
    no_vision: Annotated[
        bool, typer.Option("--no-vision", help="Skip vision/chart processing")
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", help="Verbose logging")] = False,
):
    """📄 Ingest and index financial PDF documents."""
    setup_logging(
        level="DEBUG" if verbose else "INFO",
        format_type="text" if verbose else "json",
    )

    missing = [f for f in files if not Path(f).exists()]
    if missing:
        console.print(f"[red]✗ Files not found:[/red] {missing}")
        raise typer.Exit(1)

    async def _run():
        from src.rag_system.pipeline import create_pipeline

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as prog:
            t = prog.add_task(f"[cyan]Ingesting {len(files)} file(s)…", total=None)
            pipeline = await create_pipeline()
            result = await pipeline.ingest(
                file_paths=files, tenant_id=tenant, process_vision=not no_vision
            )
            prog.update(t, completed=True)
        return result

    result = asyncio.run(_run())
    table = Table(title="✅ Ingest Summary", show_header=True, header_style="bold green")
    table.add_column("Metric")
    table.add_column("Value", style="cyan")
    table.add_row("Status", result.get("status", "?"))
    table.add_row("Tenant", result.get("tenant_id", "?"))
    table.add_row("Files", str(result.get("num_files", 0)))
    table.add_row("Chunks indexed", str(result.get("num_chunks", 0)))
    table.add_row("Latency", f"{result.get('latency_s', 0):.2f}s")
    console.print(table)


# ────────────────────────────────────────────────────────────────────────────────
# QUERY
# ────────────────────────────────────────────────────────────────────────────────


@app.command()
def query(
    question: Annotated[str, typer.Argument(help="Question to ask")],
    tenant: Annotated[str, typer.Option("--tenant", "-t")] = "default",
    top_k: Annotated[int, typer.Option("--top-k", "-k")] = 5,
    show_sources: Annotated[bool, typer.Option("--show-sources", "-s")] = False,
    json_out: Annotated[bool, typer.Option("--json", help="Output raw JSON")] = False,
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
):
    """🔍 Query indexed financial documents and get a grounded answer."""
    setup_logging(level="DEBUG" if verbose else "WARNING", format_type="json")

    async def _run():
        from src.rag_system.pipeline import create_pipeline

        with Progress(
            SpinnerColumn(), TextColumn("[cyan]Querying…"), console=console
        ) as prog:
            prog.add_task("", total=None)
            pipeline = await create_pipeline()
            return await pipeline.query(
                query_text=question, tenant_id=tenant, top_k=top_k
            )

    result = asyncio.run(_run())

    if json_out:
        console.print(
            Syntax(json.dumps(result, indent=2, default=str), "json", theme="monokai")
        )
        return

    console.print(
        Panel(
            f"[bold]{result.get('answer', 'No answer generated')}[/bold]",
            title="💡 Answer",
            border_style="cyan",
        )
    )

    m = result.get("metrics", {})
    console.print(
        f"[dim]Latency: {m.get('total_latency_ms', 0):.0f}ms  "
        f"Cost: ${m.get('cost_usd', 0):.5f}  "
        f"Chunks: {m.get('num_chunks', 0)}[/dim]"
    )

    if show_sources and result.get("sources"):
        table = Table(title="📚 Sources", show_header=True, header_style="bold blue")
        table.add_column("#")
        table.add_column("Document")
        table.add_column("Page")
        table.add_column("Score", justify="right")
        table.add_column("Preview")
        for i, src in enumerate(result["sources"], 1):
            table.add_row(
                str(i),
                src["document"],
                str(src.get("page") or "?"),
                f"{src.get('score', 0):.3f}",
                src.get("text_preview", "")[:60] + "…",
            )
        console.print(table)

    guards = result.get("guardrails", {})
    if guards and not guards.get("overall_passed", True):
        console.print(
            Panel(
                f"[yellow]⚠ Guardrail warning:[/yellow] {guards}",
                title="Guardrails",
                border_style="yellow",
            )
        )


# ────────────────────────────────────────────────────────────────────────────────
# EVALUATE
# ────────────────────────────────────────────────────────────────────────────────


@app.command()
def evaluate(
    dataset: Annotated[str, typer.Option("--dataset", "-d")]
    = "evals/golden_datasets/financial_qa.jsonl",
    tenant: Annotated[str, typer.Option("--tenant", "-t")] = "eval",
    fail_on_regression: Annotated[bool, typer.Option("--fail-on-regression")] = True,
    output: Annotated[Optional[str], typer.Option("--output", "-o")] = None,
):
    """📊 Run quality evaluation against the golden dataset."""

    async def _run():
        from src.rag_system.components.evaluator import (
            GoldenDatasetRunner,
            RagasEvaluator,
        )
        from src.rag_system.pipeline import create_pipeline

        pipeline = await create_pipeline()
        evaluator = RagasEvaluator()
        runner = GoldenDatasetRunner(
            pipeline=pipeline,
            evaluator=evaluator,
            golden_dataset_path=dataset,
        )
        return await runner.run(tenant_id=tenant)

    with Progress(
        SpinnerColumn(), TextColumn("[cyan]Running evals…"), TimeElapsedColumn(),
        console=console,
    ) as prog:
        prog.add_task("", total=None)
        report = asyncio.run(_run())

    table = Table(
        title="📊 Eval Report", show_header=True, header_style="bold magenta"
    )
    table.add_column("Metric")
    table.add_column("Value", style="cyan")
    table.add_row("Run ID", report.run_id)
    table.add_row("Samples", str(report.num_samples))
    table.add_row("Pass Rate", f"{report.pass_rate:.1%}")
    table.add_row("Avg Faithfulness", f"{report.avg_faithfulness:.3f}")
    table.add_row("Avg Answer Relevancy", f"{report.avg_answer_relevancy:.3f}")
    table.add_row("Avg Numeric Accuracy", f"{report.avg_numeric_accuracy:.3f}")
    table.add_row("Avg Latency", f"{report.avg_latency_ms:.0f}ms")
    table.add_row("Total Cost", f"${report.total_cost_usd:.4f}")
    table.add_row(
        "Regression", "⚠ YES" if report.regression_detected else "✅ None"
    )
    console.print(table)

    if output:
        Path(output).write_text(
            json.dumps(
                {
                    "run_id": report.run_id,
                    "pass_rate": report.pass_rate,
                    "avg_faithfulness": report.avg_faithfulness,
                    "regression_detected": report.regression_detected,
                },
                indent=2,
            )
        )
        console.print(f"[dim]Report saved to {output}[/dim]")

    if fail_on_regression and report.regression_detected:
        console.print("[red]✗ Quality regression detected — CI gate FAILED[/red]")
        raise typer.Exit(1)


# ────────────────────────────────────────────────────────────────────────────────
# SERVE
# ────────────────────────────────────────────────────────────────────────────────


@app.command()
def serve(
    host: Annotated[str, typer.Option("--host")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port")] = 8000,
    workers: Annotated[int, typer.Option("--workers")] = 1,
    reload: Annotated[bool, typer.Option("--reload")] = False,
):
    """🚀 Start the FastAPI server."""
    try:
        import uvicorn

        console.print(f"[green]Starting RAG Financial API on {host}:{port}[/green]")
        uvicorn.run(
            "src.rag_system.api.app:create_app",
            host=host,
            port=port,
            workers=workers,
            reload=reload,
            factory=True,
        )
    except ImportError:
        console.print("[red]uvicorn not installed. Run: pip install uvicorn[/red]")
        raise typer.Exit(1) from None


# ────────────────────────────────────────────────────────────────────────────────
# HEALTH
# ────────────────────────────────────────────────────────────────────────────────


@app.command()
def health():
    """🏥 Check system and component health."""

    async def _run():
        from src.rag_system.pipeline import create_pipeline

        pipeline = await create_pipeline()
        return await pipeline.health_check()

    result = asyncio.run(_run())
    color = "green" if result["status"] == "healthy" else "yellow"
    console.print(
        Panel(
            json.dumps(result, indent=2),
            title=f"[{color}]System Health: {result['status'].upper()}[/{color}]",
            border_style=color,
        )
    )


def main_cli():
    app()


if __name__ == "__main__":
    main_cli()
