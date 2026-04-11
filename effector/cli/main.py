"""
main.py — Effector CLI entry point.

Install the CLI entry-point via pyproject.toml::

    [project.scripts]
    effector = "effector.cli.main:app"

Or run directly::

    python -m effector.cli.main run --task "analyze cpu pressure"

Commands
--------
  run       Run a DASP debate session               (primary command)
  models    Manage Ollama models
  config    View and edit persistent configuration
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

from effector.cli import __version__
from effector.cli.config_cmd import app as config_app
from effector.cli.display import console, err_console, print_error, select_model_interactive
from effector.cli.models_cmd import app as models_app
from effector.cli.run_cmd import execute_run
from effector.cli.settings import (
    KNOWN_TIER1_MODELS,
    KNOWN_TIER2_MODELS,
    get_settings,
)

# ─────────────────────────────────────────────────────────────────────────────
# Root app
# ─────────────────────────────────────────────────────────────────────────────

app = typer.Typer(
    name="effector",
    help=(
        "[bold]Effector[/bold] — Debate-as-a-Service CLI.\n\n"
        "Orchestrates multi-agent LLM debates over local OS telemetry using\n"
        "the DASP-1.0 protocol and the Intention Envelope Protocol (IEP).\n\n"
        "Run [bold]effector run --help[/bold] to see all debate parameters.\n"
        "Run [bold]effector models known[/bold] to list available models.\n"
        "Run [bold]effector config show[/bold] to inspect your saved defaults.\n"
    ),
    rich_markup_mode="rich",
    no_args_is_help=True,
    add_completion=True,
)

app.add_typer(models_app, name="models")
app.add_typer(config_app, name="config")


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"effector  v{__version__}")
        raise typer.Exit()


# ─────────────────────────────────────────────────────────────────────────────
# `run` command
# ─────────────────────────────────────────────────────────────────────────────

@app.command(
    "run",
    help=(
        "Run a DASP multi-agent debate session.\n\n"
        "At minimum provide [bold]--task[/bold].  All other flags fall back to\n"
        "values in [bold]~/.config/effector/config.toml[/bold] (set with\n"
        "[bold]effector config set[/bold]).\n\n"
        "Exit codes:\n"
        "  0  Success (consensus reached or max rounds)\n"
        "  1  Fatal error (Ollama unreachable, bad args, etc.)\n"
        "  2  Inhibition gate fired (hard veto, no consensus possible)\n"
        "  3  Low consensus score (< 0.4) — consider re-running or adjusting θ\n"
    ),
)
def run(
    # ── Core ─────────────────────────────────────────────────────────────────
    task: Annotated[str, typer.Option(
        "--task", "-t",
        help=(
            "[required] The directive or prompt sent to all agents.\n"
            'e.g. "Is the system under abnormal CPU pressure?"'
        ),
    )],

    # ── Agent model selection ─────────────────────────────────────────────────
    agent1: Annotated[Optional[str], typer.Option(
        "--agent1",
        help=(
            "Tier-1 agent 1 model name.  Omit to pick interactively.\n"
            "Run [bold]effector models known[/bold] for the full catalogue.\n"
            "Examples: mistral:7b  qwen2.5:32b  deepseek-r1:14b"
        ),
    )] = None,

    agent2: Annotated[Optional[str], typer.Option(
        "--agent2",
        help=(
            "Tier-1 agent 2 model name.  Omit to pick interactively.\n"
            "Examples: qwen2.5:14b  qwen3:32b  phi3:medium"
        ),
    )] = None,

    arbiter: Annotated[Optional[str], typer.Option(
        "--arbiter",
        help=(
            "Tier-2 arbiter model.  Invoked only on stall / inhibition gate.\n"
            "Examples: nemotron  minimax-m2.7:cloud"
        ),
    )] = None,

    # ── Gate thresholds ───────────────────────────────────────────────────────
    tau: Annotated[float, typer.Option(
        "--tau",
        help=(
            "Inhibition gate threshold τ ∈ [0, 1].\n"
            "Fires when S_i ≥ τ · S_g (circuit breaker — hard veto).\n"
            "Lower = more sensitive.  Default 0.5."
        ),
        min=0.0, max=1.0,
    )] = None,

    theta: Annotated[float, typer.Option(
        "--theta",
        help=(
            "Consensus gate threshold θ ∈ [0, 1].\n"
            "Fires when S_net ≥ θ (victory condition).\n"
            "Higher = stricter consensus required.  Default 0.7."
        ),
        min=0.0, max=1.0,
    )] = None,

    epsilon: Annotated[float, typer.Option(
        "--epsilon", "-e",
        help=(
            "Stall gate threshold ε ≥ 0.\n"
            "Fires when |ΔS_net| < ε (deadlock detection → triggers tier-2).\n"
            "Lower = detect stalls sooner.  Default 0.05."
        ),
        min=0.0,
    )] = None,

    # ── Agent temperatures ────────────────────────────────────────────────────
    agent1_temp: Annotated[float, typer.Option(
        "--agent1-temp",
        help="Agent 1 base temperature [0.0–1.0]. Controls creativity/diversity. Default 0.7.",
        min=0.0, max=1.0,
    )] = None,

    agent1_temp_pivot: Annotated[Optional[int], typer.Option(
        "--agent1-temp-pivot",
        help=(
            "Agent 1 round number (1-indexed) at which temperature changes.\n"
            "Before pivot: base temp (or range-hi).  After: base*0.6 (or range-lo).\n"
            "Useful for annealing: start creative, converge to focused."
        ),
        min=1,
    )] = None,

    agent1_temp_range: Annotated[Optional[str], typer.Option(
        "--agent1-temp-range",
        help=(
            "Agent 1 temperature range as 'lo,hi' (e.g. '0.3,0.9').\n"
            "Without --agent1-temp-pivot: linearly sweeps hi→lo across rounds.\n"
            "With --agent1-temp-pivot: step function at pivot (hi before, lo after)."
        ),
    )] = None,

    agent2_temp: Annotated[float, typer.Option(
        "--agent2-temp",
        help="Agent 2 base temperature [0.0–1.0]. Default 0.7.",
        min=0.0, max=1.0,
    )] = None,

    agent2_temp_pivot: Annotated[Optional[int], typer.Option(
        "--agent2-temp-pivot",
        help="Agent 2 round at which temperature changes (see --agent1-temp-pivot).",
        min=1,
    )] = None,

    agent2_temp_range: Annotated[Optional[str], typer.Option(
        "--agent2-temp-range",
        help="Agent 2 temperature range 'lo,hi' (see --agent1-temp-range).",
    )] = None,

    # ── Session control ───────────────────────────────────────────────────────
    max_rounds: Annotated[int, typer.Option(
        "--max-rounds", "-m",
        help=(
            "Maximum debate rounds before forced escalation to the tier-2 arbiter.\n"
            "If consensus is reached earlier, the session terminates naturally."
        ),
        min=1,
    )] = None,

    timeout_s: Annotated[float, typer.Option(
        "--timeout",
        help="Per-agent network call timeout in seconds.  Default 60.",
        min=1.0,
    )] = None,

    # ── Output / display ──────────────────────────────────────────────────────
    verbosity: Annotated[int, typer.Option(
        "--verbosity",
        help=(
            "Characters of each agent answer to print per round.\n"
            "  0   = show only gates / signal manifold (no answers)\n"
            "  200 = default — truncate at 200 chars\n"
            "  -1  = print full agent answers (can be very long)"
        ),
    )] = None,

    output_format: Annotated[str, typer.Option(
        "--format", "-f",
        help=(
            "Result output format:\n"
            "  pretty  = Rich-formatted panels and tables (default)\n"
            "  json    = Raw JSON dump of the full result dict\n"
            "  minimal = One-line summary (score, reason, answer)\n"
            "  silent  = No terminal output (use --output to save result)"
        ),
    )] = None,

    output_file: Annotated[Optional[Path], typer.Option(
        "--output", "-o",
        help="Save the full debate result as JSON to this file path.",
    )] = None,

    persist_path: Annotated[Optional[Path], typer.Option(
        "--persist",
        help=(
            "Path to a JSONL file for envelope queue persistence.\n"
            "Each validated IEP envelope is appended for audit/replay."
        ),
    )] = None,

    # ── Telemetry ─────────────────────────────────────────────────────────────
    poll_telemetry: Annotated[bool, typer.Option(
        "--telemetry/--no-telemetry",
        help=(
            "Enable/disable the background OS telemetry poller.\n"
            "When enabled, agents receive live CPU, RAM, disk, network, and\n"
            "active-window data as world-state context via the StateBus.\n"
            "Requires psutil.  Default: on."
        ),
    )] = None,

    poll_interval_s: Annotated[float, typer.Option(
        "--poll-interval",
        help="Telemetry poll interval in seconds.  Default 2.0.",
        min=0.1,
    )] = None,

    # ── IEP / verification ────────────────────────────────────────────────────
    run_iep: Annotated[bool, typer.Option(
        "--iep/--no-iep",
        help=(
            "Run the IEP (Intention Envelope Protocol) verification pipeline\n"
            "after the debate.  Builds and validates a typed envelope from the\n"
            "debate result against the current world state.\n"
            "Default: on."
        ),
    )] = True,

    vectorized_bus: Annotated[bool, typer.Option(
        "--vectorized-bus/--no-vectorized-bus",
        help=(
            "Enable IEP-A3 snapshot vectorization.  Before dispatching agents,\n"
            "serializes the StateBus and fetches a dense embedding from Ollama.\n"
            "Enables cosine-similarity world-state verification instead of\n"
            "exact SHA-256 matching.  Requires an embedding model (see\n"
            "--embedding-model) to be installed locally.  Default: off."
        ),
    )] = None,

    embedding_model: Annotated[str, typer.Option(
        "--embedding-model",
        help=(
            "Ollama model used for IEP-A3 snapshot embeddings.\n"
            "Must be installed locally (effector models pull nomic-embed-text).\n"
            "Default: nomic-embed-text."
        ),
    )] = None,

    rat_threshold: Annotated[float, typer.Option(
        "--rat-threshold",
        help=(
            "Cosine similarity threshold for IEP-A3 snapshot validation.\n"
            "Below this score the validator returns NACK and forces a replan.\n"
            "Only used when --vectorized-bus is active.  Default 0.97."
        ),
        min=0.0, max=1.0,
    )] = None,

    epsilon_continue: Annotated[float, typer.Option(
        "--epsilon-continue",
        help="IEP post-execution divergence: below this, execution proceeds.  Default 0.1.",
        min=0.0,
    )] = None,

    epsilon_replan: Annotated[float, typer.Option(
        "--epsilon-replan",
        help="IEP post-execution divergence: above this, a replan signal fires.  Default 0.3.",
        min=0.0,
    )] = None,

    epsilon_escalate: Annotated[float, typer.Option(
        "--epsilon-escalate",
        help="IEP post-execution divergence: above this, an escalation signal fires.  Default 0.6.",
        min=0.0,
    )] = None,

    # ── Infrastructure ────────────────────────────────────────────────────────
    ollama_host: Annotated[str, typer.Option(
        "--host",
        help=(
            "Ollama API base URL.  Override to point at a remote GPU machine.\n"
            "Default: http://127.0.0.1:11434"
        ),
    )] = None,

    # ── Misc ──────────────────────────────────────────────────────────────────
    dry_run: Annotated[bool, typer.Option(
        "--dry-run",
        help="Validate configuration and display session parameters, then exit without running agents.",
    )] = False,

    interactive: Annotated[bool, typer.Option(
        "--interactive", "-i",
        help="Force interactive model selection even when --agent1/--agent2 are supplied.",
    )] = False,

    # ── Version ───────────────────────────────────────────────────────────────
    version: Annotated[Optional[bool], typer.Option(
        "--version", "-V",
        is_eager=True,
        callback=_version_callback,
        help="Print version and exit.",
    )] = None,
) -> None:
    """
    Run a DASP multi-agent debate session.

    \b
    Quick start:

        effector run -t "Is CPU pressure abnormal?"
        effector run -t "..." --agent1 mistral:7b --agent2 qwen2.5:14b --max-rounds 5
        effector run -t "..." --agent1-temp 0.9 --agent1-temp-range "0.3,0.9" --max-rounds 5
        effector run -t "..." --format json --output result.json

    \b
    Gate tuning cheat-sheet:

        τ (tau)     inhibition  lower = more sensitive circuit-breaker
        θ (theta)   consensus   higher = stricter win condition
        ε (epsilon) stall       lower = escalate to tier-2 sooner

    \b
    Temperature scheduling:

        --agent1-temp 0.8                       constant 0.8 all rounds
        --agent1-temp-range 0.3,0.9             linear sweep 0.9→0.3
        --agent1-temp-pivot 3                   cool to 48% at round 3
        --agent1-temp-range 0.3,0.9 --agent1-temp-pivot 3   step at round 3
    """
    s = get_settings()

    # ── Resolve settings (CLI > config file > default) ────────────────────────
    def _r(cli_val, *cfg_path, default=None):
        """Return cli_val if provided, else config value, else default."""
        if cli_val is not None:
            return cli_val
        return s.get(*cfg_path, default=default)

    resolved_agent1       = _r(agent1,          "agents", "agent1",          default="mistral:7b")
    resolved_agent2       = _r(agent2,          "agents", "agent2",          default="qwen2.5:14b")
    resolved_arbiter      = _r(arbiter,         "agents", "arbiter",         default="nemotron")
    resolved_tau          = _r(tau,             "debate", "tau",             default=0.5)
    resolved_theta        = _r(theta,           "debate", "theta",           default=0.7)
    resolved_epsilon      = _r(epsilon,         "debate", "epsilon",         default=0.05)
    resolved_a1_temp      = _r(agent1_temp,     "agents", "agent1_temp",     default=0.7)
    resolved_a1_pivot     = _r(agent1_temp_pivot, "agents", "agent1_temp_pivot", default=None)
    resolved_a1_range     = _r(agent1_temp_range, "agents", "agent1_temp_range", default=None)
    resolved_a2_temp      = _r(agent2_temp,     "agents", "agent2_temp",     default=0.7)
    resolved_a2_pivot     = _r(agent2_temp_pivot, "agents", "agent2_temp_pivot", default=None)
    resolved_a2_range     = _r(agent2_temp_range, "agents", "agent2_temp_range", default=None)
    resolved_max_rounds   = _r(max_rounds,      "debate", "max_rounds",      default=3)
    resolved_timeout      = _r(timeout_s,       "agents", "timeout_s",       default=60.0)
    resolved_verbosity    = _r(verbosity,       "debate", "verbosity",       default=200)
    resolved_format       = _r(output_format,   "output", "format",          default="pretty")
    resolved_persist      = _r(persist_path,    "output", "persist_path",    default=None)
    resolved_poll         = _r(poll_telemetry,  "telemetry", "enabled",      default=True)
    resolved_interval     = _r(poll_interval_s, "telemetry", "poll_interval_s", default=2.0)
    resolved_vector       = _r(vectorized_bus,  "iep", "vectorized_bus",     default=False)
    resolved_emb_model    = _r(embedding_model, "iep", "embedding_model",    default="nomic-embed-text")
    resolved_rat          = _r(rat_threshold,   "iep", "rat_threshold",      default=0.97)
    resolved_host         = _r(ollama_host,     "ollama", "host",            default="http://127.0.0.1:11434")

    # ── Interactive model selection ────────────────────────────────────────────
    if interactive or resolved_agent1 is None:
        resolved_agent1 = select_model_interactive(
            KNOWN_TIER1_MODELS,
            prompt_label="Select tier-1 agent 1 model",
        )
    if interactive or resolved_agent2 is None:
        resolved_agent2 = select_model_interactive(
            KNOWN_TIER1_MODELS,
            prompt_label="Select tier-1 agent 2 model",
        )

    # ── Validate gate ordering ────────────────────────────────────────────────
    ec = _r(epsilon_continue, "iep", "epsilon_continue", default=0.1)
    er = _r(epsilon_replan,   "iep", "epsilon_replan",   default=0.3)
    ee = _r(epsilon_escalate, "iep", "epsilon_escalate", default=0.6)

    if not (ec < er < ee):
        print_error(
            f"IEP epsilon thresholds must satisfy:\n"
            f"  epsilon_continue ({ec}) < epsilon_replan ({er}) < epsilon_escalate ({ee})\n"
            "Adjust with --epsilon-continue / --epsilon-replan / --epsilon-escalate."
        )
        raise typer.Exit(1)

    # ── Delegate to run_cmd ───────────────────────────────────────────────────
    exit_code = execute_run(
        task=task,
        agent1_model=resolved_agent1,
        agent2_model=resolved_agent2,
        arbiter_model=resolved_arbiter,
        agent1_temp=resolved_a1_temp,
        agent1_temp_pivot=resolved_a1_pivot,
        agent1_temp_range=resolved_a1_range,
        agent2_temp=resolved_a2_temp,
        agent2_temp_pivot=resolved_a2_pivot,
        agent2_temp_range=resolved_a2_range,
        tau=resolved_tau,
        theta=resolved_theta,
        epsilon=resolved_epsilon,
        max_rounds=resolved_max_rounds,
        timeout_s=resolved_timeout,
        verbosity=resolved_verbosity,
        output_format=resolved_format,
        output_file=output_file,
        persist_path=Path(resolved_persist) if resolved_persist else None,
        ollama_host=resolved_host,
        poll_telemetry=resolved_poll,
        poll_interval_s=resolved_interval,
        vectorized_bus=resolved_vector,
        embedding_model=resolved_emb_model,
        rat_threshold=resolved_rat,
        dry_run=dry_run,
        run_iep=run_iep,
        iep_keys_affected=["debate_answer"],
    )

    raise typer.Exit(exit_code)


# ─────────────────────────────────────────────────────────────────────────────
# `session` group  (lightweight — replay / inspect saved results)
# ─────────────────────────────────────────────────────────────────────────────

session_app = typer.Typer(
    name="session",
    help="Inspect and replay persisted debate sessions.",
    no_args_is_help=True,
)
app.add_typer(session_app, name="session")


@session_app.command("replay")
def session_replay(
    path: Path = typer.Argument(..., help="Path to a JSONL envelope queue file"),
    ack_only: bool = typer.Option(False, "--ack-only", help="Show only ACK'd envelopes"),
) -> None:
    """Print all envelopes from a persisted JSONL queue file."""
    from effector.queue.iep_queue import EnvelopeQueue
    from effector.cli.display import kv_table

    if not path.exists():
        print_error(f"File not found: {path}")
        raise typer.Exit(1)

    q = EnvelopeQueue(persist_path=path)
    items = q.replay_from_disk()

    if not items:
        console.print("[dim]No envelopes found.[/dim]")
        return

    for item in items:
        val = item.get("validation", {})
        status = val.get("status", "?")
        if ack_only and status != "ACK":
            continue
        style = "green" if status == "ACK" else "red"
        eid = item.get("envelope", {}).get("envelope_id", "?")[:12]
        console.print(f"[{style}]{status}[/]  {eid}…  {val.get('failure_reason', '')}")


@session_app.command("inspect")
def session_inspect(
    path: Path = typer.Argument(..., help="Path to a JSON result file (from --output)"),
) -> None:
    """Pretty-print a saved debate result file."""
    import json as _json
    if not path.exists():
        print_error(f"File not found: {path}")
        raise typer.Exit(1)
    try:
        result = _json.loads(path.read_text())
    except Exception as exc:
        print_error(f"Could not parse result file: {exc}")
        raise typer.Exit(1)
    print_result(result)


# ─────────────────────────────────────────────────────────────────────────────
# `doctor` — environment self-check
# ─────────────────────────────────────────────────────────────────────────────

@app.command("doctor")
def doctor(
    host: Optional[str] = typer.Option(None, "--host", help="Ollama host to check"),
) -> None:
    """
    Verify that all runtime dependencies are reachable.

    Checks:
    - Python version
    - Required Python packages (typer, rich, requests, psutil)
    - Ollama connectivity + local model list
    - Config file status
    - tomli-w availability (needed for config writes)
    """
    import importlib
    import platform

    ok   = "[green]✓[/]"
    fail = "[red]✗[/]"
    warn = "[yellow]⚠[/]"

    console.print("\n[bold]Effector environment check[/bold]\n")

    # Python version
    pv = sys.version_info
    badge = ok if pv >= (3, 11) else warn
    console.print(f"  {badge}  Python {platform.python_version()}" +
                  ("" if pv >= (3, 11) else "  [dim](3.11+ recommended for built-in tomllib)[/dim]"))

    # Key packages
    for pkg in ("typer", "rich", "requests", "psutil"):
        try:
            importlib.import_module(pkg)
            console.print(f"  {ok}  {pkg}")
        except ImportError:
            console.print(f"  {fail}  {pkg}  [dim](pip install {pkg})[/dim]")

    # tomllib / tomli
    try:
        import tomllib  # noqa: F401
        console.print(f"  {ok}  tomllib (built-in)")
    except ImportError:
        try:
            import tomli  # noqa: F401
            console.print(f"  {ok}  tomli")
        except ImportError:
            console.print(f"  {warn}  tomllib/tomli missing  [dim](pip install tomli)[/dim]")

    # tomli-w
    try:
        import tomli_w  # noqa: F401
        console.print(f"  {ok}  tomli-w")
    except ImportError:
        console.print(
            f"  {warn}  tomli-w missing  "
            "[dim](pip install tomli-w — required for config writes)[/dim]"
        )

    # Ollama
    import requests
    ollama_url = host or get_settings().get("ollama", "host", default="http://127.0.0.1:11434")
    try:
        resp = requests.get(f"{ollama_url}/api/tags", timeout=3)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        console.print(f"  {ok}  Ollama at {ollama_url}  [{len(models)} model(s)]")
    except Exception as exc:
        console.print(f"  {fail}  Ollama at {ollama_url}  [dim]{exc}[/dim]")

    # Config
    cfg_p = get_settings().path
    if cfg_p.exists():
        console.print(f"  {ok}  Config {cfg_p}")
    else:
        console.print(f"  {warn}  No config file yet — defaults in use.  ({cfg_p})")

    console.print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
