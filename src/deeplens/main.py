"""CLI entry point for DeepLens — run entity research from the terminal.

Usage:
    python -m deeplens "Research Baby Monster"
    python -m deeplens "Research Elon Musk" --verbose
    python -m deeplens "Research AI coding assistants" --model gpt-4o
"""

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

from deeplens.graph import build_initial_state, create_graph

app = typer.Typer(
    name="deeplens",
    help="DeepLens — Multi-agent entity research system",
    add_completion=False,
)
console = Console()


def _setup_logging(verbose: bool) -> None:
    """Configure logging with Rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )


@app.command()
def research(
    query: str = typer.Argument(help="Research query, e.g. 'Research Baby Monster'"),
    model: str = typer.Option(None, "--model", "-m", help="Override LLM model name"),
    max_iterations: int = typer.Option(
        None, "--max-iterations", "-i", help="Max supervisor iterations"
    ),
    output_dir: str = typer.Option(
        "output", "--output-dir", "-o", help="Output directory for reports"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Run a DeepLens entity research session."""
    _setup_logging(verbose)

    query = query.strip()
    if not query:
        console.print("[red]Error: query cannot be empty[/]")
        raise typer.Exit(code=1)
    if len(query) > 500:
        query = query[:500]
        console.print("[yellow]Warning: query truncated to 500 characters[/]")

    # Apply CLI overrides to settings
    if model or max_iterations or output_dir != "output":
        import os

        if model:
            os.environ["MODEL_NAME"] = model
        if max_iterations:
            os.environ["MAX_ITERATIONS"] = str(max_iterations)
        if output_dir != "output":
            os.environ["OUTPUT_DIR"] = output_dir

        # Clear cached settings so overrides take effect
        from deeplens.config import get_settings

        get_settings.cache_clear()

    console.print(Panel(f"[bold blue]DeepLens[/] — Researching: [bold]{query}[/]", expand=False))

    try:
        graph = create_graph()
        initial_state = build_initial_state(query)

        console.print("[dim]Starting agent graph...[/]\n")

        # Stream execution to show progress
        final_state = None
        for event in graph.stream(initial_state, stream_mode="values"):
            final_state = event
            next_agent = event.get("next_agent", "")
            iteration = event.get("iteration_count", 0)
            if next_agent:
                console.print(f"  [cyan]->[/] iter={iteration} next=[bold]{next_agent}[/]")

        if final_state is None:
            console.print("[red]Error: graph produced no output[/]")
            raise typer.Exit(code=1)

        # Display results
        report = final_state.get("report_markdown", "")
        charts = final_state.get("charts", [])
        errors = final_state.get("errors", [])
        iterations = final_state.get("iteration_count", 0)
        sources = final_state.get("sources", [])
        web_count = len(final_state.get("web_results", []))
        article_count = len(final_state.get("web_articles", []))
        video_count = len(final_state.get("videos", []))

        console.print()
        console.print(Panel("[bold green]Research Complete[/]", expand=False))
        console.print(f"  Iterations: {iterations}")
        console.print(
            f"  Sources: {len(sources)} "
            f"({web_count} web, {article_count} articles, {video_count} videos)"
        )
        console.print(f"  Charts: {len(charts)}")
        if charts:
            for c in charts:
                console.print(f"    [dim]{c}[/]")
        if errors:
            console.print(f"  [yellow]Warnings: {len(errors)}[/]")
            for e in errors:
                console.print(f"    [dim yellow]{e}[/]")

        # Print report
        console.print()
        console.print(Panel("[bold]Report[/]", expand=False))
        console.print(report)

        # Show output location
        report_path = Path(output_dir) / "report.md"
        if report_path.exists():
            console.print(f"\n[dim]Report saved to: {report_path}[/]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Research interrupted by user[/]")
        raise typer.Exit(code=130)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/]")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
