"""
models_cmd.py — `effector models` subcommand group.

Subcommands
-----------
  list      List locally-installed Ollama models
  known     Show the built-in tier-1 / tier-2 catalogues
  pull      Pull (download) a model from Ollama
  remove    Remove a locally-installed model
  search    Search ollama.com library (best-effort web scrape)
  info      Show metadata for a specific local model
"""
from __future__ import annotations

import subprocess
import sys
from typing import Optional

import requests
import typer
from rich import box
from rich.table import Table

from effector.cli.display import (
    console,
    err_console,
    make_spinner,
    print_error,
    print_ok,
    print_warn,
)
from effector.cli.settings import KNOWN_TIER1_MODELS, KNOWN_TIER2_MODELS, get_settings

app = typer.Typer(
    name="models",
    help="Manage Ollama models used by the debate engine.",
    no_args_is_help=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Ollama API helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ollama_host() -> str:
    return get_settings().get("ollama", "host") or "http://127.0.0.1:11434"


def _list_local() -> list[dict]:
    """Call Ollama /api/tags. Returns list of model dicts or empty list."""
    try:
        resp = requests.get(f"{_ollama_host()}/api/tags", timeout=5)
        resp.raise_for_status()
        return resp.json().get("models", [])
    except requests.exceptions.ConnectionError:
        print_error(f"Cannot reach Ollama at {_ollama_host()} — is it running?")
        return []
    except Exception as exc:
        print_error(f"Ollama API error: {exc}")
        return []


def _local_model_names() -> list[str]:
    return [m.get("name", "") for m in _list_local()]


# ─────────────────────────────────────────────────────────────────────────────
# list
# ─────────────────────────────────────────────────────────────────────────────

@app.command("list")
def models_list(
    tier1_only: bool = typer.Option(False, "--tier1", help="Only show tier-1 known models"),
    tier2_only: bool = typer.Option(False, "--tier2", help="Only show tier-2 known models"),
) -> None:
    """List locally-installed Ollama models, annotated against the known catalogues."""
    models = _list_local()
    if not models:
        console.print("[dim]No models found or Ollama unreachable.[/dim]")
        raise typer.Exit()

    known_t1 = set(KNOWN_TIER1_MODELS)
    known_t2 = set(KNOWN_TIER2_MODELS)

    t = Table(
        "Model", "Size", "Tier", "Modified",
        box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False,
    )

    for m in models:
        name     = m.get("name", "?")
        size_b   = m.get("size", 0)
        modified = (m.get("modified_at") or "")[:10]

        size_str = _fmt_size(size_b)
        tier = ""
        if name in known_t1:
            tier = "[cyan]tier-1[/]"
        elif name in known_t2:
            tier = "[magenta]tier-2[/]"

        if tier1_only and "tier-1" not in tier:
            continue
        if tier2_only and "tier-2" not in tier:
            continue

        t.add_row(name, size_str, tier, modified)

    console.print(t)
    console.print(f"[dim]{len(models)} model(s) installed at {_ollama_host()}[/dim]")


# ─────────────────────────────────────────────────────────────────────────────
# known
# ─────────────────────────────────────────────────────────────────────────────

@app.command("known")
def models_known() -> None:
    """Display the built-in tier-1 and tier-2 model catalogues."""
    local = set(_local_model_names())

    t = Table("#", "Model", "Tier", "Installed", box=box.SIMPLE_HEAD, show_edge=False)
    for i, m in enumerate(KNOWN_TIER1_MODELS, 1):
        badge = "[green]✓[/]" if m in local else "[dim]–[/]"
        t.add_row(str(i), m, "[cyan]tier-1[/]", badge)
    for i, m in enumerate(KNOWN_TIER2_MODELS, len(KNOWN_TIER1_MODELS) + 1):
        badge = "[green]✓[/]" if m in local else "[dim]–[/]"
        t.add_row(str(i), m, "[magenta]tier-2[/]", badge)

    console.print(t)
    console.print(
        "\n[dim]Pull a model with:[/dim]  [bold]effector models pull <name>[/bold]"
    )


# ─────────────────────────────────────────────────────────────────────────────
# pull
# ─────────────────────────────────────────────────────────────────────────────

@app.command("pull")
def models_pull(
    model: str = typer.Argument(..., help="Model name to pull, e.g. mistral:7b"),
) -> None:
    """Pull (download) a model from the Ollama registry."""
    console.print(f"Pulling [bold]{model}[/bold] …")
    try:
        result = subprocess.run(
            ["ollama", "pull", model],
            check=False,
            text=True,
        )
        if result.returncode == 0:
            print_ok(f"{model} pulled successfully.")
        else:
            print_error(f"ollama pull exited with code {result.returncode}")
    except FileNotFoundError:
        print_error(
            "The `ollama` binary was not found in PATH.\n"
            "  Install Ollama from https://ollama.com and ensure it is in your PATH."
        )
        raise typer.Exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# remove
# ─────────────────────────────────────────────────────────────────────────────

@app.command("remove")
def models_remove(
    model: str = typer.Argument(..., help="Model name to remove"),
    yes:   bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Remove a locally-installed model."""
    if not yes:
        confirmed = typer.confirm(f"Remove [bold]{model}[/bold]?", default=False)
        if not confirmed:
            console.print("[dim]Aborted.[/dim]")
            raise typer.Exit()
    try:
        result = subprocess.run(
            ["ollama", "rm", model],
            check=False,
            text=True,
        )
        if result.returncode == 0:
            print_ok(f"{model} removed.")
        else:
            print_error(f"ollama rm exited with code {result.returncode}")
    except FileNotFoundError:
        print_error("The `ollama` binary was not found in PATH.")
        raise typer.Exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# search  (best-effort; parses Ollama's /api/search if available)
# ─────────────────────────────────────────────────────────────────────────────

@app.command("search")
def models_search(
    query: str = typer.Argument(..., help="Search term, e.g. 'coding' or 'deepseek'"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max results to display"),
) -> None:
    """
    Search the Ollama model library.

    Attempts the undocumented Ollama search API at ollama.com.
    Falls back to filtering the built-in catalogue if the network call fails.
    """
    console.print(f"Searching ollama.com for [bold]{query!r}[/bold] …\n")

    try:
        resp = requests.get(
            "https://ollama.com/api/models",
            params={"q": query, "limit": limit},
            headers={"Accept": "application/json"},
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
        hits: list[dict] = data if isinstance(data, list) else data.get("models", [])

        if not hits:
            raise ValueError("empty result")

        t = Table("Model", "Description", "Downloads", box=box.SIMPLE_HEAD, show_edge=False)
        for item in hits[:limit]:
            name = item.get("name", "?")
            desc = (item.get("description") or "")[:60]
            pulls = item.get("pulls", item.get("downloads", "?"))
            t.add_row(name, desc, str(pulls))
        console.print(t)

    except Exception:
        # Fallback: filter built-in catalogue
        print_warn("Could not reach ollama.com — filtering built-in catalogue instead.")
        q = query.lower()
        matches = [m for m in [*KNOWN_TIER1_MODELS, *KNOWN_TIER2_MODELS] if q in m.lower()]
        if matches:
            local = set(_local_model_names())
            t = Table("#", "Model", "Installed", box=box.SIMPLE_HEAD, show_edge=False)
            for i, m in enumerate(matches, 1):
                t.add_row(str(i), m, "[green]✓[/]" if m in local else "[dim]–[/]")
            console.print(t)
        else:
            console.print(f"[dim]No built-in matches for {query!r}.[/dim]")

    console.print(
        "\n[dim]Pull a model with:[/dim]  [bold]effector models pull <name>[/bold]"
    )


# ─────────────────────────────────────────────────────────────────────────────
# info
# ─────────────────────────────────────────────────────────────────────────────

@app.command("info")
def models_info(
    model: str = typer.Argument(..., help="Model name, e.g. mistral:7b"),
) -> None:
    """Show metadata for a locally-installed model via Ollama /api/show."""
    try:
        resp = requests.post(
            f"{_ollama_host()}/api/show",
            json={"name": model},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.ConnectionError:
        print_error(f"Cannot reach Ollama at {_ollama_host()}")
        raise typer.Exit(1)
    except Exception as exc:
        print_error(str(exc))
        raise typer.Exit(1)

    from effector.cli.display import kv_table
    details: dict = {}
    details["Model file"] = (data.get("modelfile") or "")[:80]
    details["Parameters"] = data.get("parameters", "")
    details["Template"]   = (data.get("template") or "")[:80]

    model_info = data.get("model_info") or data.get("details") or {}
    for k, v in model_info.items():
        details[k] = v

    kv_table(details, title=model)


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_size(bytes_: int) -> str:
    if bytes_ == 0:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if bytes_ < 1024:
            return f"{bytes_:.1f} {unit}"
        bytes_ /= 1024
    return f"{bytes_:.1f} PB"
