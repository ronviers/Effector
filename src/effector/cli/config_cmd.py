"""
config_cmd.py — `effector config` subcommand group.

Subcommands
-----------
  show            Print the active configuration
  set             Set a config value (section.key value)
  reset           Reset all settings to factory defaults
  path            Print the config file path
  profile save    Save current config as a named profile
  profile load    Restore a named profile
  profile list    List saved profiles
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.table import Table
from rich.tree import Tree

from effector.cli.display import console, kv_table, print_error, print_ok, print_warn
from effector.cli.settings import DEFAULT_CONFIG, config_dir, get_settings

app = typer.Typer(
    name="config",
    help="View and edit persistent Effector CLI configuration.",
    no_args_is_help=True,
)

profiles_app = typer.Typer(help="Named configuration profiles.")
app.add_typer(profiles_app, name="profile")


# ─────────────────────────────────────────────────────────────────────────────
# show
# ─────────────────────────────────────────────────────────────────────────────

@app.command("show")
def config_show(
    section: Optional[str] = typer.Argument(
        None,
        help="Optional section to display, e.g. 'debate' or 'agents'",
    ),
    format: str = typer.Option(
        "pretty",
        "--format", "-f",
        help="Output format: pretty | json",
    ),
) -> None:
    """
    Display the current configuration.

    Examples::

        effector config show
        effector config show debate
        effector config show --format json
    """
    cfg = get_settings().as_dict()

    if section:
        if section not in cfg:
            print_error(f"Unknown section {section!r}. Valid: {list(cfg)}")
            raise typer.Exit(1)
        data = {section: cfg[section]}
    else:
        data = cfg

    if format == "json":
        console.print_json(json.dumps(data, default=str))
        return

    # Pretty tree
    tree = Tree("[bold]effector config[/bold]")
    for sec, vals in data.items():
        branch = tree.add(f"[cyan]{sec}[/cyan]")
        if isinstance(vals, dict):
            for k, v in vals.items():
                branch.add(f"[dim]{k}[/dim] = {_fmt_val(v)}")
        else:
            branch.add(str(vals))
    console.print(tree)


# ─────────────────────────────────────────────────────────────────────────────
# set
# ─────────────────────────────────────────────────────────────────────────────

@app.command("set")
def config_set(
    key:   str = typer.Argument(..., help="Dot-separated key path, e.g. debate.tau"),
    value: str = typer.Argument(..., help="Value to set (type-coerced automatically)"),
) -> None:
    """
    Set a configuration value and persist it.

    Key paths mirror the TOML structure::

        effector config set debate.tau 0.6
        effector config set agents.agent1 qwen2.5:32b
        effector config set ollama.host http://192.168.1.10:11434
    """
    parts = key.split(".")
    if len(parts) < 2:
        print_error("Key must be in 'section.field' format, e.g. 'debate.tau'")
        raise typer.Exit(1)

    coerced = _coerce(value)
    s = get_settings()
    try:
        s.set(*parts, coerced)
        print_ok(f"{key} = {coerced!r}")
    except RuntimeError as exc:
        print_error(str(exc))
        raise typer.Exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# reset
# ─────────────────────────────────────────────────────────────────────────────

@app.command("reset")
def config_reset(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Reset all settings to factory defaults."""
    if not yes:
        confirmed = typer.confirm(
            "This will overwrite all custom settings. Continue?",
            default=False,
        )
        if not confirmed:
            console.print("[dim]Aborted.[/dim]")
            raise typer.Exit()

    s = get_settings()
    try:
        s.reset()
        print_ok("Configuration reset to factory defaults.")
    except RuntimeError as exc:
        print_error(str(exc))
        raise typer.Exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# path
# ─────────────────────────────────────────────────────────────────────────────

@app.command("path")
def config_path_cmd() -> None:
    """Print the path to the active config file."""
    p = get_settings().path
    exists = "[green]exists[/]" if p.exists() else "[yellow]not yet created[/]"
    console.print(f"{p}  {exists}")


# ─────────────────────────────────────────────────────────────────────────────
# diff  (compare against defaults)
# ─────────────────────────────────────────────────────────────────────────────

@app.command("diff")
def config_diff() -> None:
    """Show settings that differ from factory defaults."""
    current = get_settings().as_dict()
    diffs: list[tuple[str, str, str]] = []
    _collect_diff(DEFAULT_CONFIG, current, [], diffs)

    if not diffs:
        console.print("[dim]No changes from defaults.[/dim]")
        return

    t = Table("Key", "Default", "Current", box=box.SIMPLE_HEAD)
    for key_path, default_val, current_val in diffs:
        t.add_row(key_path, str(default_val), f"[bold]{current_val}[/bold]")
    console.print(t)


# ─────────────────────────────────────────────────────────────────────────────
# Profiles
# ─────────────────────────────────────────────────────────────────────────────

def _profiles_dir() -> Path:
    return config_dir() / "profiles"


@profiles_app.command("save")
def profile_save(
    name: str = typer.Argument(..., help="Profile name, e.g. 'high-creativity'"),
) -> None:
    """Save the current configuration as a named profile."""
    import json as _json
    pdir = _profiles_dir()
    pdir.mkdir(parents=True, exist_ok=True)
    dest = pdir / f"{name}.json"
    data = get_settings().as_dict()
    dest.write_text(_json.dumps(data, indent=2, default=str))
    print_ok(f"Profile saved: {dest}")


@profiles_app.command("load")
def profile_load(
    name: str = typer.Argument(..., help="Profile name to load"),
) -> None:
    """Restore a named profile as the active configuration."""
    import json as _json
    dest = _profiles_dir() / f"{name}.json"
    if not dest.exists():
        print_error(f"Profile {name!r} not found at {dest}")
        raise typer.Exit(1)

    try:
        data = _json.loads(dest.read_text())
    except Exception as exc:
        print_error(f"Could not read profile: {exc}")
        raise typer.Exit(1)

    s = get_settings()
    from effector.cli.settings import _deep_merge, DEFAULT_CONFIG
    s._data = _deep_merge(DEFAULT_CONFIG, data)  # noqa: SLF001
    try:
        s.save()
        print_ok(f"Profile {name!r} loaded.")
    except RuntimeError as exc:
        print_error(str(exc))
        raise typer.Exit(1)


@profiles_app.command("list")
def profile_list() -> None:
    """List all saved profiles."""
    pdir = _profiles_dir()
    profiles = sorted(pdir.glob("*.json")) if pdir.exists() else []
    if not profiles:
        console.print("[dim]No saved profiles.[/dim]")
        return
    for p in profiles:
        console.print(f"  [cyan]{p.stem}[/cyan]  [dim]{p}[/dim]")


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _coerce(raw: str):
    """Auto-coerce a string CLI value to int, float, bool, or str."""
    if raw.lower() in ("true", "yes", "on"):
        return True
    if raw.lower() in ("false", "no", "off"):
        return False
    if raw.lower() in ("none", "null", ""):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _fmt_val(v) -> str:
    if v is None:
        return "[dim]None[/dim]"
    if isinstance(v, bool):
        return "[green]true[/]" if v else "[red]false[/]"
    return str(v)


def _collect_diff(
    default: dict,
    current: dict,
    path: list[str],
    out: list,
) -> None:
    for k, dv in default.items():
        cv = current.get(k)
        full_key = ".".join([*path, k])
        if isinstance(dv, dict) and isinstance(cv, dict):
            _collect_diff(dv, cv, [*path, k], out)
        elif cv != dv:
            out.append((full_key, dv, cv))
