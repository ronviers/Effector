"""
State Bus — Local key-value world state with SHA-256 snapshot hashing.
Append-only delta log. Thread-safe.
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Callable


class StateBus:
    """
    Canonical, shared world state representation.
    Agents read from it; no agent writes without an acknowledged IEP envelope.
    """

    def __init__(self, initial_state: dict[str, Any] | None = None) -> None:
        self._state: dict[str, Any] = dict(initial_state or {})
        self._delta_log: list[dict[str, Any]] = []      # append-only
        self._rat_store: dict[str, dict] = {}           # RAT storage by rat_id
        self._reputation_store: dict[str, dict] = {}   # agent reputation records
        self._lock = threading.RLock()
        self._listeners: list[Callable[[str, dict[str, Any]], None]] = []

    # ─── Read ──────────────────────────────────────────────────────────────

    def read(self, keys: list[str] | None = None) -> dict[str, Any]:
        """Read world state. If keys given, returns only those keys."""
        with self._lock:
            if keys is None:
                return dict(self._state)
            return {k: self._state.get(k) for k in keys}

    def snapshot(self, keys: list[str] | None = None) -> tuple[str, datetime, dict[str, Any]]:
        """
        Capture a snapshot of current world state.
        Returns (sha256_hash, timestamp, state_slice).
        The hash is computed over the canonical JSON of the state slice.
        """
        with self._lock:
            state_slice = self.read(keys)
            ts = datetime.now(timezone.utc)
            h = self._hash_state(state_slice)
            return h, ts, state_slice

    def verify_hash(self, expected_hash: str, keys: list[str] | None = None) -> bool:
        """Check if current world state matches a previously issued hash."""
        current_hash, _, _ = self.snapshot(keys)
        return current_hash == expected_hash

    # ─── Write (IEP-acknowledged only) ─────────────────────────────────────

    def apply_delta(
        self,
        envelope_id: str,
        delta: dict[str, Any],
        agent_id: str,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Apply a state delta. Only called after IEP ACK.
        Returns the actual delta applied (may differ from intended for conflict detection).
        """
        with self._lock:
            before = {k: self._state.get(k) for k in delta}
            self._state.update(delta)
            after = {k: self._state[k] for k in delta}

            log_entry = {
                "log_id": str(uuid.uuid4()),
                "envelope_id": envelope_id,
                "agent_id": agent_id,
                "session_id": session_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "before": before,
                "after": after,
                "delta": delta,
            }
            self._delta_log.append(log_entry)
            self._emit("delta_applied", log_entry)
            return delta

    # ─── RAT store ─────────────────────────────────────────────────────────

    def store_rat(self, rat: dict[str, Any]) -> None:
        with self._lock:
            self._rat_store[rat["rat_id"]] = rat

    def get_rat(self, rat_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._rat_store.get(rat_id)

    def invalidate_rat(self, rat_id: str) -> None:
        with self._lock:
            self._rat_store.pop(rat_id, None)

    # ─── Reputation ────────────────────────────────────────────────────────

    R_INITIAL = 0.5
    R_FLOOR = 0.15
    EMA_ALPHA = 0.2

    def get_reputation(self, agent_id: str) -> float:
        """Return R_eff(a) = max(R(a), R_FLOOR)."""
        with self._lock:
            record = self._reputation_store.get(agent_id)
            if record is None:
                return max(self.R_INITIAL, self.R_FLOOR)
            return max(record["R"], self.R_FLOOR)

    def update_reputation(
        self,
        agent_id: str,
        envelope_id: str,
        session_id: str,
        divergence: float,
        epsilon_escalate: float,
    ) -> float:
        """Update agent reputation based on IEP post-execution divergence."""
        with self._lock:
            accuracy = 1.0 - min(divergence / max(epsilon_escalate, 1e-9), 1.0)

            record = self._reputation_store.setdefault(agent_id, {
                "agent_id": agent_id,
                "R": self.R_INITIAL,
                "sample_count": 0,
                "last_updated": None,
                "history": [],
            })

            # Exponential moving average
            record["R"] = (
                self.EMA_ALPHA * accuracy + (1 - self.EMA_ALPHA) * record["R"]
            )
            record["R_eff"] = max(record["R"], self.R_FLOOR)
            record["sample_count"] += 1
            record["last_updated"] = datetime.now(timezone.utc).isoformat()
            record["history"].append({
                "envelope_id": envelope_id,
                "session_id": session_id,
                "accuracy": accuracy,
                "divergence": divergence,
                "timestamp": record["last_updated"],
            })
            return record["R_eff"]

    # ─── Listeners / event bus ──────────────────────────────────────────────

    def on(self, callback: Callable[[str, dict[str, Any]], None]) -> None:
        """Register a listener for state bus events."""
        self._listeners.append(callback)

    def _emit(self, event: str, data: dict[str, Any]) -> None:
        for cb in self._listeners:
            try:
                cb(event, data)
            except Exception:
                pass  # listeners must not crash the bus

    # ─── Utilities ─────────────────────────────────────────────────────────

    @staticmethod
    def _hash_state(state: dict[str, Any]) -> str:
        """Deterministic SHA-256 of a state dict (sorted keys, canonical JSON)."""
        canonical = json.dumps(state, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def delta_log(self) -> list[dict[str, Any]]:
        """Return a copy of the append-only delta log."""
        with self._lock:
            return list(self._delta_log)

    def __repr__(self) -> str:
        return f"StateBus(keys={list(self._state.keys())})"
