"""
settings.py — Persistent TOML configuration for the Effector CLI.

Config path (platform-aware):
  Windows : %LOCALAPPDATA%\\effector\\config.toml
  macOS   : ~/Library/Application Support/effector/config.toml
  Linux   : $XDG_CONFIG_HOME/effector/config.toml
              (default ~/.config/effector/config.toml)

Read requires Python 3.11+ built-in tomllib (or `pip install tomli`).
Write requires `pip install tomli-w`.
Falls back gracefully to defaults if either dep is absent.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# ── TOML read ─────────────────────────────────────────────────────────────────
try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]

# ── TOML write ────────────────────────────────────────────────────────────────
try:
    import tomli_w  # pip install tomli-w
except ImportError:
    tomli_w = None  # type: ignore[assignment]


# ─────────────────────────────────────────────────────────────────────────────
# Factory defaults — every CLI option has a default here so nothing is hardcoded
# across multiple files.
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG: dict[str, Any] = {
    "debate": {
        "tau": 0.5,
        "theta": 0.7,
        "epsilon": 0.05,
        "max_rounds": 3,
        "verbosity": 200,       # chars per agent response to print; -1 = unlimited
    },
    "agents": {
        "agent1": "mistral:7b",
        "agent2": "qwen2.5:14b",
        "arbiter": "nemotron",
        "agent1_temp": 0.7,
        "agent1_temp_pivot": None,      # round number to switch temperature
        "agent1_temp_range": None,      # "min,max" e.g. "0.3,0.9"
        "agent2_temp": 0.7,
        "agent2_temp_pivot": None,
        "agent2_temp_range": None,
        "timeout_s": 60.0,
    },
    "iep": {
        "vectorized_bus": False,
        "embedding_model": "nomic-embed-text",
        "rat_threshold": 0.97,
        "epsilon_continue": 0.1,
        "epsilon_replan": 0.3,
        "epsilon_escalate": 0.6,
    },
    "telemetry": {
        "enabled": True,
        "poll_interval_s": 2.0,
    },
    "output": {
        "format": "pretty",     # pretty | json | minimal | silent
        "file": None,           # path to save JSON result
        "persist_path": None,   # path for envelope queue JSONL persistence
    },
    "ollama": {
        "host": "http://127.0.0.1:11434",
    },
}

# ── Built-in model catalogues (editable via `effector models known`) ──────────
KNOWN_TIER1_MODELS: list[str] = [
    "deepseek-r1:14b",
    "qwen3.5:35b",
    "lfm2:24b-a2b",
    "phi3:medium",
    "mistral:7b",
    "qwen2.5:14b",
    "qwen2.5:32b",
    "Agent-Qwen2.5-Coder:latest",
    "qwen2.5-coder:32b",
    "qwen3-agent:latest",
    "Agent-Qwen-32B:latest",
    "qwen3:32b",
]

KNOWN_TIER2_MODELS: list[str] = [
    "nemotron-3-super:cloud",
    "minimax-m2.7:cloud",
    "nemotron",
]


# ─────────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────────

def config_dir() -> Path:
    """Return the platform-appropriate config directory."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "effector"


def config_path() -> Path:
    return config_dir() / "config.toml"


# ─────────────────────────────────────────────────────────────────────────────
# Deep merge
# ─────────────────────────────────────────────────────────────────────────────

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*; override wins on leaf conflicts."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Settings class
# ─────────────────────────────────────────────────────────────────────────────

class Settings:
    """
    Lazy-loaded TOML settings store with deep-merge defaults.

    Usage::

        s = Settings()
        s.get("debate", "tau")          # -> 0.5
        s.set("debate", "tau", 0.6)     # persists to disk
        s.as_dict()                     # full config snapshot
    """

    def __init__(self) -> None:
        self._path = config_path()
        self._data: dict[str, Any] = {}
        self._load()

    # ── I/O ──────────────────────────────────────────────────────────────────

    def _load(self) -> None:
        self._data = _deep_merge(DEFAULT_CONFIG, {})
        if self._path.exists() and tomllib is not None:
            try:
                with open(self._path, "rb") as fh:
                    on_disk = tomllib.load(fh)
                self._data = _deep_merge(DEFAULT_CONFIG, on_disk)
            except Exception as exc:
                # Never crash on load — just warn and use defaults
                print(f"[settings] Could not load {self._path}: {exc} — using defaults")

    def save(self) -> None:
        """Persist current config to disk. Requires tomli-w."""
        if tomli_w is None:
            raise RuntimeError(
                "tomli-w is required to save config.\n"
                "  pip install tomli-w"
            )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        
        # TOML does not support 'None'. Strip them out before saving.
        def _strip_nones(d: dict) -> dict:
            return {k: (_strip_nones(v) if isinstance(v, dict) else v) 
                    for k, v in d.items() if v is not None}
            
        clean_data = _strip_nones(self._data)
        
        with open(self._path, "wb") as fh:
            tomli_w.dump(clean_data, fh)

    # ── Access ────────────────────────────────────────────────────────────────

    def get(self, *path: str, default: Any = None) -> Any:
        """Retrieve a nested value by key path, e.g. ``s.get("debate", "tau")``."""
        node = self._data
        for key in path:
            if not isinstance(node, dict):
                return default
            node = node.get(key, default)
        return node

    def set(self, *path_and_value: Any) -> None:
        """
        Set a nested value and persist to disk.
        Last argument is the value; all prior arguments are the key path.

            s.set("debate", "tau", 0.6)
        """
        *path, value = path_and_value
        if not path:
            raise ValueError("set() requires at least one key and a value")
        node = self._data
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value
        self.save()

    def reset(self) -> None:
        """Overwrite config with factory defaults and persist."""
        self._data = _deep_merge(DEFAULT_CONFIG, {})
        self.save()

    def as_dict(self) -> dict[str, Any]:
        """Return a shallow copy of the full config dict."""
        return dict(self._data)

    @property
    def path(self) -> Path:
        return self._path

    def __repr__(self) -> str:
        return f"Settings(path={self._path!s})"


# ── Module-level singleton ────────────────────────────────────────────────────
_instance: Settings | None = None


def get_settings() -> Settings:
    global _instance
    if _instance is None:
        _instance = Settings()
    return _instance
