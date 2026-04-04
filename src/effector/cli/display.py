"""
display.py — Rich-powered display helpers for the Effector CLI.

All terminal output goes through this module so that format switches
(pretty / json / minimal / silent) are handled in one place and the
run logic stays format-agnostic.
"""
from __future__ import annotations

import json
from typing import Any

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text
from rich import box

# ── Shared console ────────────────────────────────────────────────────────────
console = Console(highlight=False)
err_console = Console(stderr=True, style="bold red")


# ─────────────────────────────────────────────────────────────────────────────
# Colour / style constants
# ─────────────────────────────────────────────────────────────────────────────
COLOUR_TIER1   = "cyan"
COLOUR_TIER2   = "magenta"
COLOUR_GATE    = "yellow"
COLOUR_WARN    = "red"
COLOUR_OK      = "green"
COLOUR_DIM     = "dim white"
COLOUR_HEADER  = "bold white"


# ─────────────────────────────────────────────────────────────────────────────
# Session header
# ─────────────────────────────────────────────────────────────────────────────

def print_session_header(
    task: str,
    agent1: str,
    agent2: str,
    arbiter: str,
    tau: float,
    theta: float,
    epsilon: float,
    max_rounds: int,
    vectorized: bool,
    ollama_host: str,
) -> None:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", justify="right")
    grid.add_column()

    rows = [
        ("Task",          f"[bold]{task[:120]}[/bold]"),
        ("Tier-1",        f"[{COLOUR_TIER1}]{agent1}[/] · [{COLOUR_TIER1}]{agent2}[/]"),
        ("Arbiter",       f"[{COLOUR_TIER2}]{arbiter}[/]"),
        ("Gates",         f"τ={tau}  θ={theta}  ε={epsilon}"),
        ("Max rounds",    str(max_rounds)),
        ("IEP-A3 vector", "[green]on[/]" if vectorized else "[dim]off[/]"),
        ("Ollama host",   ollama_host),
    ]
    for label, value in rows:
        grid.add_row(label, value)

    console.print(Panel(grid, title="[bold]⚡ Effector Session[/bold]", border_style="blue"))


# ─────────────────────────────────────────────────────────────────────────────
# Round display
# ─────────────────────────────────────────────────────────────────────────────

def print_round_header(round_num: int, tier: str = "local") -> None:
    tier_badge = f"[{COLOUR_TIER2}]ARBITER[/]" if tier == "cloud_arbiter" else f"[{COLOUR_TIER1}]tier-1[/]"
    console.rule(f"[bold]Round {round_num}[/bold]  {tier_badge}")


def print_round_responses(
    responses: list[dict[str, Any]],
    verbosity: int,
) -> None:
    """Render each agent's response as a panel row."""
    if verbosity == 0:
        return

    table = Table(
        box=box.SIMPLE_HEAD,
        show_edge=False,
        pad_edge=False,
        expand=True,
    )
    table.add_column("Agent",       style=COLOUR_TIER1, no_wrap=True, width=22)
    table.add_column("H-ID",        style="dim",        no_wrap=True, width=12)
    table.add_column("Pol",         justify="center",   no_wrap=True, width=6)
    table.add_column("Conf",        justify="right",    no_wrap=True, width=6)
    table.add_column("Answer",      ratio=1)

    for resp in responses:
        sig    = resp.get("signal", {})
        pol    = sig.get("polarity", 0)
        conf   = sig.get("confidence", 0.0)
        answer = resp.get("answer", "")
        hid    = resp.get("hypothesis_id", "?")

        pol_str   = {1: "[green]▲[/]", 0: "[yellow]■[/]", -1: "[red]▼[/]"}.get(pol, "?")
        conf_str  = f"{conf:.2f}"

        if verbosity > 0:
            answer = answer[:verbosity] + ("…" if len(answer) > verbosity else "")

        table.add_row(
            resp.get("agent_id", "?"),
            hid[:12],
            pol_str,
            conf_str,
            answer,
        )

    console.print(table)


def print_signal_manifold(accumulators: dict[str, dict]) -> None:
    """Print a compact signal manifold table."""
    if not accumulators:
        return
    t = Table(box=box.MINIMAL, show_header=True, header_style="dim")
    t.add_column("Hypothesis",  style="dim", no_wrap=True)
    t.add_column("S_g",  justify="right", style="green", width=8)
    t.add_column("S_i",  justify="right", style="red",   width=8)
    t.add_column("S_net",justify="right", style="bold",  width=8)

    for hid, acc in accumulators.items():
        s_net = acc.get("S_net", acc.get("S_g", 0) - acc.get("S_i", 0))
        net_style = "green" if s_net >= 0 else "red"
        t.add_row(
            hid[:18],
            f"{acc.get('S_g', 0):.3f}",
            f"{acc.get('S_i', 0):.3f}",
            f"[{net_style}]{s_net:.3f}[/]",
        )
    console.print(t)


# ─────────────────────────────────────────────────────────────────────────────
# Gate / trigger events
# ─────────────────────────────────────────────────────────────────────────────

def print_trigger(event: str, data: dict[str, Any]) -> None:
    ICONS = {
        "escalation_triggered": "⚡",
        "consensus_cleared":    "✅",
        "trigger_fired":        "⚠️",
        "tier2_invoked":        "🔮",
    }
    icon = ICONS.get(event, "ℹ️")
    trigger = data.get("trigger", event)
    detail  = ""
    if "tier_from" in data:
        detail = f" → escalating to [{COLOUR_TIER2}]{data.get('tier_to', '?')}[/]"
    if "winning_hypothesis" in data:
        detail = f" winner=[bold]{data['winning_hypothesis']}[/]  score={data.get('consensus_score', 0):.3f}"

    console.print(f"  {icon} [{COLOUR_GATE}]{trigger}[/]{detail}")


# ─────────────────────────────────────────────────────────────────────────────
# Final result
# ─────────────────────────────────────────────────────────────────────────────

def print_result(result: dict[str, Any]) -> None:
    """Full pretty-print of the debate result."""
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", justify="right")
    grid.add_column()

    score  = result.get("consensus_score", 0.0)
    reason = result.get("terminated_reason", "?")
    rounds = result.get("rounds", 0)
    disag  = result.get("disagreement_score", 0.0)

    score_style = "green" if score >= 0.7 else ("yellow" if score >= 0.4 else "red")

    grid.add_row("Answer",       f"[bold]{result.get('final_answer', '?')}[/bold]")
    grid.add_row("Consensus",    f"[{score_style}]{score:.3f}[/]")
    grid.add_row("Terminated",   reason)
    grid.add_row("Rounds",       str(rounds))
    grid.add_row("Disagreement", f"{disag:.3f}")
    if result.get("tier2_injected"):
        grid.add_row("Escalated",  "[magenta]yes — tier-2 arbiter invoked[/]")

    border = "green" if score >= 0.7 else ("yellow" if score >= 0.4 else "red")
    console.print(Panel(grid, title="[bold]Debate Result[/bold]", border_style=border))


def print_result_json(result: dict[str, Any]) -> None:
    """Dump result as pretty-printed JSON."""
    console.print_json(json.dumps(result, default=str))


def print_result_minimal(result: dict[str, Any]) -> None:
    """One-line summary."""
    # --- NEW: Handle Reflex results ---
    if result.get("_path") == "reflex":
        status = result.get("status", "EXECUTED")
        rat_id = str(result.get("rat_id", "unknown"))[:8]
        action = result.get("matched_action", {})
        verb = action.get("verb", "WRITE")
        target = action.get("target", "world_state")
        console.print(f"[REFLEX] {status} | RAT:{rat_id} | {verb} {target}")
        return
    # ----------------------------------

    score  = result.get("consensus_score", 0.0)
    reason = result.get("terminated_reason", "?")
    
    # Strip newlines so the answer actually fits on one line
    raw_answer = str(result.get("final_answer", ""))
    answer = raw_answer.replace("\n", " ")[:80]
    
    console.print(f"[{score:.3f}] {reason} | {answer}")


# ─────────────────────────────────────────────────────────────────────────────
# Model selection
# ─────────────────────────────────────────────────────────────────────────────

def select_model_interactive(
    known_models: list[str],
    prompt_label: str = "Select model",
    local_models: list[str] | None = None,
) -> str:
    """
    Show a numbered list of models and return the user's choice.
    Prefers locally-installed models (marked with ✓).
    """
    installed = set(local_models or [])
    console.print(f"\n[bold]{prompt_label}[/bold]")

    all_models = list(dict.fromkeys([*known_models, *(local_models or [])]))
    for i, m in enumerate(all_models, 1):
        badge = " [green]✓[/]" if m in installed else ""
        console.print(f"  [dim]{i:>2}.[/]  {m}{badge}")

    while True:
        raw = console.input("\n  Enter number or model name: ").strip()
        if raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(all_models):
                return all_models[idx]
        elif raw:
            return raw
        console.print("  [red]Invalid selection — try again.[/]")


# ─────────────────────────────────────────────────────────────────────────────
# Progress spinner
# ─────────────────────────────────────────────────────────────────────────────

def make_spinner(label: str = "Working…") -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Generic helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_error(msg: str) -> None:
    err_console.print(f"✗ {msg}")


def print_ok(msg: str) -> None:
    console.print(f"[green]✓[/] {msg}")


def print_warn(msg: str) -> None:
    console.print(f"[yellow]⚠[/] {msg}")


def kv_table(data: dict[str, Any], title: str = "") -> None:
    """Generic key-value table."""
    t = Table(title=title, box=box.SIMPLE_HEAD, show_header=False, expand=False)
    t.add_column(style="dim", justify="right")
    t.add_column()
    for k, v in data.items():
        t.add_row(str(k), str(v))
    console.print(t)
