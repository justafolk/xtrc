from __future__ import annotations

import json
import time

import typer
import uvicorn

from xtrc.client import HttpAinavClient
from xtrc.config import Settings
from xtrc.core.errors import AinavError
from xtrc.logging import setup_logging

app = typer.Typer(help="AI-powered local code navigation")


def _base_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def _print_json(payload: dict[str, object]) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=False))


@app.command()
def serve(
    host: str = typer.Option(Settings.from_env().host, help="Bind host"),
    port: int = typer.Option(Settings.from_env().port, help="Bind port"),
    log_level: str = typer.Option("info", help="uvicorn log level"),
) -> None:
    """Start xtrc daemon server."""
    setup_logging()
    uvicorn.run("xtrc.server:app", host=host, port=port, log_level=log_level, reload=False)


@app.command()
def index(
    repo_path: str = typer.Argument(".", help="Repository root"),
    rebuild: bool = typer.Option(False, "--rebuild", help="Full rebuild index"),
    host: str = typer.Option(Settings.from_env().host, help="Server host"),
    port: int = typer.Option(Settings.from_env().port, help="Server port"),
    as_json: bool = typer.Option(False, "--json", help="Print raw JSON"),
) -> None:
    """Index repository content."""
    client = HttpAinavClient(_base_url(host, port), timeout_seconds=1800)
    try:
        payload = client.index(repo_path=repo_path, rebuild=rebuild)
    except AinavError as exc:
        typer.echo(f"error[{exc.code}] {exc.message}")
        raise typer.Exit(code=1) from exc

    if as_json:
        _print_json(payload)
        return

    typer.echo(f"repo: {payload['repo_path']}")
    typer.echo(f"files scanned: {payload['files_scanned']}")
    typer.echo(f"files indexed: {payload['files_indexed']}")
    typer.echo(f"files deleted: {payload['files_deleted']}")
    typer.echo(f"chunks indexed: {payload['chunks_indexed']}")
    typer.echo(f"duration: {payload['duration_ms']}ms")


@app.command()
def query(
    query: str = typer.Argument(..., help="Natural language query"),
    repo_path: str = typer.Option(".", "--repo", help="Repository root"),
    top_k: int = typer.Option(8, "--top-k", min=1, max=50, help="Max number of results"),
    host: str = typer.Option(Settings.from_env().host, help="Server host"),
    port: int = typer.Option(Settings.from_env().port, help="Server port"),
    as_json: bool = typer.Option(False, "--json", help="Print raw JSON"),
) -> None:
    """Search indexed chunks and return jump targets."""
    client = HttpAinavClient(_base_url(host, port), timeout_seconds=120)
    started = time.perf_counter()
    try:
        payload = client.query(repo_path=repo_path, query=query, top_k=top_k)
    except AinavError as exc:
        typer.echo(f"error[{exc.code}] {exc.message}")
        raise typer.Exit(code=1) from exc
    response_ms = int((time.perf_counter() - started) * 1000)

    if as_json:
        payload["response_time_ms"] = response_ms
        _print_json(payload)
        return

    typer.echo(f"repo: {payload['repo_path']}")
    typer.echo(f"query: {payload['query']}")
    typer.echo(f"duration: {payload['duration_ms']}ms")
    typer.echo(f"response time: {response_ms}ms")
    selection = payload.get("selection")
    if isinstance(selection, dict):
        source = payload.get("selection_source", "vector")
        typer.echo(f"selection[{source}]: {selection.get('file')}:{selection.get('line')}")
        typer.echo(f"reason: {selection.get('reason')}")
        if payload.get("used_gemini"):
            typer.echo(
                f"gemini: model={payload.get('gemini_model')} latency={payload.get('gemini_latency_ms')}ms"
            )
        rewritten = payload.get("rewritten_query")
        if isinstance(rewritten, str) and rewritten:
            typer.echo(f"rewritten query: {rewritten}")

    results = payload.get("results", [])
    if not results:
        typer.echo("no matches")
        return

    for idx, result in enumerate(results, start=1):
        typer.echo(
            f"{idx}. {result['file_path']}:{result['start_line']} "
            f"score={result['score']:.3f} symbol={result['symbol'] or '-'}"
        )
        typer.echo(f"   {result['description']}")
        intents = result.get("matched_intents") or []
        keywords = result.get("matched_keywords") or []
        if intents:
            typer.echo(f"   intents: {', '.join(intents)}")
        if keywords:
            typer.echo(f"   keywords: {', '.join(keywords)}")
        explanation = result.get("explanation")
        if isinstance(explanation, str) and explanation:
            typer.echo(f"   why: {explanation}")


@app.command()
def status(
    repo_path: str = typer.Argument(".", help="Repository root"),
    host: str = typer.Option(Settings.from_env().host, help="Server host"),
    port: int = typer.Option(Settings.from_env().port, help="Server port"),
    as_json: bool = typer.Option(False, "--json", help="Print raw JSON"),
) -> None:
    """Show index status for repository."""
    client = HttpAinavClient(_base_url(host, port), timeout_seconds=30)
    try:
        payload = client.status(repo_path=repo_path)
    except AinavError as exc:
        typer.echo(f"error[{exc.code}] {exc.message}")
        raise typer.Exit(code=1) from exc

    if as_json:
        _print_json(payload)
        return

    typer.echo(f"repo: {payload['repo_path']}")
    typer.echo(f"model: {payload['model']}")
    typer.echo(f"indexed files: {payload['indexed_files']}")
    typer.echo(f"indexed chunks: {payload['indexed_chunks']}")
    typer.echo(f"last indexed at: {payload.get('last_indexed_at')}")
    typer.echo(f"healthy: {payload['healthy']}")


if __name__ == "__main__":
    app()
