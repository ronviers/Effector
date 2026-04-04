"""
intent_router.py — Step 0: Deterministic Intent Routing (The Pre-Parser)

Maps incoming prompts / telemetry trigger strings to a structured IntendedAction
dict using exact string matching and regex. Zero LLM calls — this runs before
the ReflexEngine and must be sub-millisecond.

If the router cannot confidently map the task, it returns None and the caller
MUST route the raw task straight to the DASPCoordinator for LLM deliberation
and RAT issuance.

Design contract
---------------
- No imports from external packages beyond stdlib.
- No Ollama calls, no network I/O.
- All patterns are case-insensitive unless noted.
- A pattern match is confident only when verb AND target are unambiguous.
- Adding new patterns: append to ROUTE_TABLE; no other changes required.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IntendedAction:
    """Minimal structured action ready for RAT lookup."""
    verb: str          # READ | WRITE | CALL | DELEGATE | TERMINATE
    target: str        # resource string matching RAT authorized_actions[].target
    parameters: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"verb": self.verb, "target": self.target, "parameters": dict(self.parameters)}


@dataclass
class RouteEntry:
    """One routing rule."""
    pattern: re.Pattern[str]
    verb: str
    target: str
    # Optional parameter extractor: callable(match) -> dict
    param_fn: Any = None


# ---------------------------------------------------------------------------
# Route table — order matters; first match wins
# ---------------------------------------------------------------------------
#
# Patterns cover:
#   • Desktop cozy actions (Glimmer spawn, file move, ambient sync)
#   • OS telemetry reads (the most common reflex trigger)
#   • Spotify / media hooks
#   • File-system organise actions
#   • Brightness / ambient control
#
# Targets mirror the IEP resource naming used in authorized_actions[].target.

def _build_table() -> list[RouteEntry]:
    def _r(pattern: str, verb: str, target: str, param_fn=None) -> RouteEntry:
        return RouteEntry(re.compile(pattern, re.IGNORECASE), verb, target, param_fn)

    return [
        # ── Telemetry / OS state reads ───────────────────────────────────────
        _r(r"\bpoll[_ ]telemetry\b",          "READ",  "state_bus.telemetry"),
        _r(r"\bread[_ ](?:os[_ ])?desktop\b", "READ",  "os.desktop"),
        _r(r"\bcheck[_ ](?:cpu|ram|memory)\b","READ",  "state_bus.telemetry"),
        _r(r"\bget[_ ]active[_ ]window\b",    "READ",  "os.desktop.active_window"),
        _r(r"\bsnapshot[_ ]state\b",          "READ",  "state_bus.snapshot"),
        _r(r"\bread[_ ]weather\b",            "READ",  "api.weather"),
        _r(r"\bfetch[_ ]weather\b",           "READ",  "api.weather"),

        # ── Glimmer / companion spawns ───────────────────────────────────────
        _r(r"\bspawn[_ ]glimmer\b",           "WRITE", "desktop.overlay.glimmer"),
        _r(r"\bdrop[_ ]glimmer\b",            "WRITE", "desktop.overlay.glimmer"),
        _r(r"\bplace[_ ](?:a[_ ])?glimmer\b", "WRITE", "desktop.overlay.glimmer"),
        _r(r"\bspawn[_ ]companion\b",         "WRITE", "desktop.overlay.glimmer"),
        _r(r"\bspawn[_ ]music[_ ]glimmer\b",  "WRITE", "desktop.overlay.glimmer.music"),
        _r(r"\bspawn[_ ]librarian\b",         "WRITE", "desktop.overlay.glimmer.librarian"),
        _r(r"\bspawn[_ ]campfire\b",          "WRITE", "desktop.overlay.ambient.campfire"),

        # ── Ambient / environmental writes ───────────────────────────────────
        _r(r"\bdim[_ ](?:monitor|brightness|screen)\b",  "WRITE", "os.display.brightness"),
        _r(r"\bset[_ ]brightness\b",                      "WRITE", "os.display.brightness"),
        _r(r"\badjust[_ ]brightness\b",                   "WRITE", "os.display.brightness"),
        _r(r"\bsync[_ ]ambient\b",                        "WRITE", "desktop.ambient_sync"),
        _r(r"\bactivate[_ ](?:the[_ ])?overlay\b",        "WRITE", "desktop.overlay"),

        # ── File / folder organisation ────────────────────────────────────────
        _r(r"\bmove[_ ](?:desktop[_ ])?icons?\b",         "WRITE", "os.filesystem.desktop_icons"),
        _r(r"\borgani[sz]e[_ ](?:desktop|icons|files)\b", "WRITE", "os.filesystem.desktop_icons"),
        _r(r"\bmove[_ ]files?\b",                          "WRITE", "os.filesystem.move"),
        _r(r"\barchive[_ ]files?\b",                       "WRITE", "os.filesystem.archive"),
        _r(r"\bcreate[_ ]zen[_ ]habitat\b",                "WRITE", "os.filesystem.zen_habitat"),
        _r(r"\bassign[_ ]librarian\b",                     "WRITE", "os.filesystem.zen_habitat"),

        # ── Media / Spotify hooks ─────────────────────────────────────────────
        _r(r"\bspotify[_ ](?:opened?|started?|launched?)\b", "WRITE", "desktop.overlay.glimmer.music"),
        _r(r"\bmedia[_ ]playing\b",               "WRITE", "desktop.overlay.glimmer.music"),
        _r(r"\bmusic[_ ](?:started?|playing)\b",  "WRITE", "desktop.overlay.glimmer.music"),

        # ── Reputation / RAT management ───────────────────────────────────────
        _r(r"\bupdate[_ ]reputation\b",           "WRITE", "state_bus.reputation"),
        _r(r"\binvalidate[_ ]rat\b",              "WRITE", "rat_store.invalidate"),
        _r(r"\bstore[_ ]rat\b",                   "WRITE", "rat_store.store"),

        # ── Session lifecycle ─────────────────────────────────────────────────
        _r(r"\bterminate[_ ]session\b",           "TERMINATE", "session"),
        _r(r"\bend[_ ]session\b",                 "TERMINATE", "session"),
        
        # ── Logic Puzzle Sweep ────────────────────────────────────────────────
        _r(r"\bsolve the following logic puzzle\b", "WRITE", "world_state"),
    ]


_ROUTE_TABLE: list[RouteEntry] | None = None


def _get_table() -> list[RouteEntry]:
    global _ROUTE_TABLE
    if _ROUTE_TABLE is None:
        _ROUTE_TABLE = _build_table()
    return _ROUTE_TABLE


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class IntentRouter:
    """
    Deterministically maps a task string to an IntendedAction.

    Usage
    -----
        router = IntentRouter()
        action = router.route("spawn glimmer on tax_returns.pdf")
        if action is None:
            # No confident match — send to DASP LLM debate
            ...
        else:
            # Fast path: look up RAT and call ReflexEngine
            result = reflex_engine.evaluate_reflex(action.as_dict(), vector, bus)
    """

    def __init__(self, extra_routes: list[RouteEntry] | None = None) -> None:
        self._table: list[RouteEntry] = _get_table() + (extra_routes or [])

    def route(self, task: str) -> IntendedAction | None:
        """
        Return an IntendedAction if the task matches a known pattern,
        else None (caller should fall back to DASP LLM debate).
        """
        task = task.strip()
        for entry in self._table:
            m = entry.pattern.search(task)
            if m:
                params: dict[str, Any] = {}
                if entry.param_fn is not None:
                    try:
                        params = entry.param_fn(m) or {}
                    except Exception:
                        params = {}
                return IntendedAction(
                    verb=entry.verb,
                    target=entry.target,
                    parameters=params,
                )
        return None

    def add_route(self, pattern: str, verb: str, target: str, param_fn=None) -> None:
        """Register an additional route at runtime (prepended — takes priority)."""
        entry = RouteEntry(
            pattern=re.compile(pattern, re.IGNORECASE),
            verb=verb,
            target=target,
            param_fn=param_fn,
        )
        self._table.insert(0, entry)

    def explain(self, task: str) -> str:
        """Debug helper: show which pattern matched (or why nothing did)."""
        for i, entry in enumerate(self._table):
            m = entry.pattern.search(task)
            if m:
                return (
                    f"Route #{i}: pattern={entry.pattern.pattern!r} "
                    f"→ {entry.verb} {entry.target}"
                )
        return "No match — DASP fallback"
