"""
gate.py — Event Control
========================
Prevents spam. Collapses noise. Guards the perceptual threshold.

Three mechanisms, each addressing a different failure mode:

  Cooldown         — same event type fires too quickly (the drumroll problem)
  Deduplication    — same exact event fires multiple times in a burst
                     (the bus-writes-fast problem)
  Burst suppression — too many events of one type in a short window
                     (the cascade problem)

Design contract
---------------
- Stateful: must track time (stateless logic is not enough)
- Thread-safe: player.py dispatches from a thread executor
- Applied BEFORE playback: gate.allow() is the gatekeeper
- No audio knowledge
- No asset knowledge

Per-event-type configuration overrides are loaded from config.py.
The defaults apply to any event type not explicitly overridden.
"""

from __future__ import annotations

import threading
import time
from collections import deque

import effector.foley.config as config
from effector.foley.events import FoleyEvent


class EventGate:
    """
    Thread-safe rate limiter and deduplicator for FoleyEvents.

    Maintains per-event-type cooldown and burst windows,
    plus a short-window deduplicator for exact-event repetition.

    Usage
    -----
        gate = EventGate(config.GATE_CONFIG)
        if gate.allow(event):
            # proceed to resolve and play
    """

    def __init__(self, gate_config: dict | None = None) -> None:
        cfg = gate_config or config.GATE_CONFIG
        self._cooldown_s:     float = cfg["cooldown_ms"] / 1000.0
        self._dedupe_s:       float = cfg["dedupe_window_ms"] / 1000.0
        self._burst_window_s: float = cfg["burst_window_ms"] / 1000.0
        self._burst_limit:    int   = cfg["burst_limit"]
        self._overrides:      dict  = cfg.get("overrides", {})

        # State (all protected by a single lock for simplicity)
        self._lock = threading.Lock()

        # last allowed time per event_type
        self._last_allowed: dict[str, float] = {}

        # sliding window of allowed times per event_type (for burst detection)
        self._burst_windows: dict[str, deque] = {}

        # deduplication: last allowed time per exact (type, entity, position) triple
        self._dedupe: dict[tuple, float] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def allow(self, event: FoleyEvent) -> bool:
        """
        Return True if this event should be allowed through.
        Updates internal state on allow.
        Does not mutate state on reject.
        """
        now = time.monotonic()
        etype = event.event_type
        dedupe_key = (etype, event.entity_sym, event.position)

        cfg = self._effective_config(etype)

        with self._lock:
            # ── 1. Deduplication (tightest window, checked first) ──────────
            last_exact = self._dedupe.get(dedupe_key, 0.0)
            if now - last_exact < cfg["dedupe_s"]:
                return False

            # ── 2. Cooldown (per event type) ───────────────────────────────
            last_type = self._last_allowed.get(etype, 0.0)
            if now - last_type < cfg["cooldown_s"]:
                return False

            # ── 3. Burst suppression (sliding window) ──────────────────────
            window = self._burst_windows.setdefault(etype, deque())
            # Prune expired entries
            cutoff = now - cfg["burst_window_s"]
            while window and window[0] < cutoff:
                window.popleft()
            if len(window) >= cfg["burst_limit"]:
                return False

            # ── Allow: update all state ────────────────────────────────────
            self._last_allowed[etype] = now
            self._dedupe[dedupe_key] = now
            window.append(now)
            return True

    def reset(self, event_type: str | None = None) -> None:
        """
        Clear gate state for one event type, or all types if None.
        Useful for testing and after significant state changes.
        """
        with self._lock:
            if event_type is None:
                self._last_allowed.clear()
                self._burst_windows.clear()
                self._dedupe.clear()
            else:
                self._last_allowed.pop(event_type, None)
                self._burst_windows.pop(event_type, None)
                # Clear dedupe entries for this type
                keys = [k for k in self._dedupe if k[0] == event_type]
                for k in keys:
                    del self._dedupe[k]

    def stats(self) -> dict:
        """Diagnostic snapshot. Does not affect gate state."""
        with self._lock:
            return {
                "tracked_types": list(self._last_allowed.keys()),
                "dedupe_keys": len(self._dedupe),
                "burst_window_depths": {
                    k: len(v) for k, v in self._burst_windows.items()
                },
            }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _effective_config(self, event_type: str) -> dict:
        """
        Merge event-type overrides with this instance's configuration.

        Falls back to instance values, NOT the module-level config.
        This ensures a custom gate_config passed to EventGate() is
        actually honoured for event types that have no explicit override.
        """
        override = self._overrides.get(event_type, {})
        return {
            "cooldown_s":     override.get("cooldown_ms",      self._cooldown_s     * 1000) / 1000.0,
            "dedupe_s":       override.get("dedupe_window_ms", self._dedupe_s       * 1000) / 1000.0,
            "burst_window_s": override.get("burst_window_ms",  self._burst_window_s * 1000) / 1000.0,
            "burst_limit":    override.get("burst_limit",       self._burst_limit),
        }
