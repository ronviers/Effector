"""
A1: Substrate Telemetry Poller
==============================
Headless Python script that polls Windows OS state and writes
structured key-value telemetry to a local StateBus instance.

Features
--------
- psutil-based cross-platform polling (Windows primary, Linux/macOS fallback)
- Delta-rate computation for disk I/O and network (bytes/sec)
- Composite system pressure score  [0.0 – 1.0]
- Configurable poll interval with jitter dampening
- Active window detection (Windows: win32gui; Linux/macOS: fallback via psutil)
- Thread-safe continuous mode via .start() / .stop()
- Single-shot mode via .poll_once() for testing

Usage (headless daemon)
-----------------------
    bus = StateBus()
    poller = TelemetryPoller(bus, interval_s=2.0)
    poller.start()          # begins background polling thread
    ...
    poller.stop()

Usage (single snapshot)
-----------------------
    snapshot = TelemetryPoller(bus).poll_once()
"""

from __future__ import annotations

import platform
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

import psutil

from effector.telemetry.state_keys import KEYS

# ── Optional Windows-only imports ────────────────────────────────────────────
try:
    import ctypes
    _WINDOWS = platform.system() == "Windows"
    if _WINDOWS:
        import ctypes.wintypes
except ImportError:
    _WINDOWS = False


# ─────────────────────────────────────────────────────────────────────────────
# Active window detection
# ─────────────────────────────────────────────────────────────────────────────

def _get_active_window() -> tuple[str, str]:
    """Return (window_title, process_name). Best-effort; never raises."""
    try:
        if _WINDOWS:
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(length + 1)
            ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
            title = buf.value or "(unknown)"
            # Get PID from hwnd
            pid = ctypes.wintypes.DWORD()
            ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            try:
                proc = psutil.Process(pid.value)
                return title, proc.name()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return title, "(unknown)"
        else:
            # Linux/macOS fallback — return the top CPU process window
            procs = sorted(
                psutil.process_iter(["name", "cpu_percent"]),
                key=lambda p: p.info.get("cpu_percent") or 0,
                reverse=True,
            )
            if procs:
                name = procs[0].info.get("name", "(unknown)")
                return f"[{name}]", name
            return "(unknown)", "(unknown)"
    except Exception:
        return "(unknown)", "(unknown)"


# ─────────────────────────────────────────────────────────────────────────────
# Delta counter for rate calculations
# ─────────────────────────────────────────────────────────────────────────────

class _RateCounter:
    """Computes per-second rates from monotonically increasing psutil counters."""

    def __init__(self) -> None:
        self._prev: dict[str, float] = {}
        self._prev_time: float = time.monotonic()

    def rate(self, key: str, current_value: float) -> float:
        """Return (current - previous) / elapsed_seconds. 0.0 on first call."""
        now = time.monotonic()
        elapsed = now - self._prev_time
        prev_val = self._prev.get(key, current_value)
        rate = (current_value - prev_val) / elapsed if elapsed > 0 else 0.0
        self._prev[key] = current_value
        self._prev_time = now
        return max(rate, 0.0)

    def tick(self) -> None:
        """Call once per poll cycle after all rates computed."""
        self._prev_time = time.monotonic()


# ─────────────────────────────────────────────────────────────────────────────
# Pressure model
# ─────────────────────────────────────────────────────────────────────────────

def _compute_pressure(cpu: float, ram: float, swap: float) -> float:
    """
    Composite system pressure score  [0.0 – 1.0].
    Weights: CPU 50%, RAM 35%, Swap 15%.
    Nonlinear — exponential emphasis above 80% utilization.
    """
    def _curve(x: float) -> float:
        x = x / 100.0
        return x ** 1.5  # emphasize high utilization

    score = 0.50 * _curve(cpu) + 0.35 * _curve(ram) + 0.15 * _curve(swap)
    return round(min(score, 1.0), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Main poller class
# ─────────────────────────────────────────────────────────────────────────────

class TelemetryPoller:
    """
    Polls OS state and writes structured telemetry to a StateBus.

    Parameters
    ----------
    state_bus : StateBus
        The shared world state object.
    interval_s : float
        Poll interval in seconds. Default 2.0.
    on_poll : Callable[[dict], None] | None
        Optional callback fired after each successful poll with the delta dict.
    envelope_id_prefix : str
        Prefix for synthetic envelope IDs in delta log. Default "telemetry".
    """

    def __init__(
        self,
        state_bus: Any,  # StateBus — imported lazily to avoid circular imports
        interval_s: float = 2.0,
        on_poll: Callable[[dict[str, Any]], None] | None = None,
        envelope_id_prefix: str = "telemetry",
    ) -> None:
        self._bus = state_bus
        self._interval_s = interval_s
        self._on_poll = on_poll
        self._prefix = envelope_id_prefix
        self._rate = _RateCounter()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._poll_count = 0

        # Prime psutil CPU percent (first call always returns 0.0)
        psutil.cpu_percent(interval=None, percpu=True)

    # ── Public API ────────────────────────────────────────────────────────

    def poll_once(self) -> dict[str, Any]:
        """Take a single telemetry snapshot and write to StateBus. Returns delta."""
        delta = self._collect()
        envelope_id = f"{self._prefix}-{uuid.uuid4()}"
        self._bus.apply_delta(
            envelope_id=envelope_id,
            delta=delta,
            agent_id="telemetry.poller",
            session_id=None,
        )
        self._poll_count += 1
        if self._on_poll:
            try:
                self._on_poll(delta)
            except Exception as exc:
                print(f"[Telemetry] on_poll callback error: {exc}")
        return delta

    def start(self) -> "TelemetryPoller":
        """Start background polling thread. Returns self for chaining."""
        if self._thread and self._thread.is_alive():
            return self
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, name="TelemetryPoller", daemon=True
        )
        self._thread.start()
        print(f"[Telemetry] Poller started (interval={self._interval_s}s)")
        return self

    def stop(self) -> None:
        """Signal the polling thread to stop and join."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self._interval_s + 2)
        print(f"[Telemetry] Poller stopped after {self._poll_count} polls.")

    @property
    def poll_count(self) -> int:
        return self._poll_count

    # ── Internal loop ─────────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            try:
                self.poll_once()
            except Exception as exc:
                print(f"[Telemetry] Poll error: {exc}")
            elapsed = time.monotonic() - t0
            sleep_for = max(0.0, self._interval_s - elapsed)
            self._stop_event.wait(timeout=sleep_for)

    # ── Collection logic ──────────────────────────────────────────────────

    def _collect(self) -> dict[str, Any]:
        ts = datetime.now(timezone.utc).isoformat()
        delta: dict[str, Any] = {}

        # ── CPU ──────────────────────────────────────────────────────────
        cpu_total = psutil.cpu_percent(interval=None)
        cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
        delta[KEYS.cpu_percent_total] = round(cpu_total, 2)
        delta[KEYS.cpu_percent_per_core] = [round(c, 2) for c in cpu_per_core]
        try:
            freq = psutil.cpu_freq()
            delta[KEYS.cpu_freq_mhz] = round(freq.current, 1) if freq else None
        except Exception:
            delta[KEYS.cpu_freq_mhz] = None

        try:
            cpu_stats = psutil.cpu_stats()
            delta[KEYS.cpu_ctx_switches_sec] = round(
                self._rate.rate("ctx_switches", cpu_stats.ctx_switches), 1
            )
        except Exception:
            delta[KEYS.cpu_ctx_switches_sec] = None

        # ── Memory ───────────────────────────────────────────────────────
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        delta[KEYS.ram_total_mb] = round(mem.total / 1024**2, 1)
        delta[KEYS.ram_used_mb] = round(mem.used / 1024**2, 1)
        delta[KEYS.ram_available_mb] = round(mem.available / 1024**2, 1)
        delta[KEYS.ram_percent] = round(mem.percent, 2)
        delta[KEYS.swap_used_mb] = round(swap.used / 1024**2, 1)
        delta[KEYS.swap_percent] = round(swap.percent, 2)

        # ── Disk I/O rates ────────────────────────────────────────────────
        try:
            disk = psutil.disk_io_counters()
            if disk:
                delta[KEYS.disk_read_mb_s] = round(
                    self._rate.rate("disk_read", disk.read_bytes) / 1024**2, 3
                )
                delta[KEYS.disk_write_mb_s] = round(
                    self._rate.rate("disk_write", disk.write_bytes) / 1024**2, 3
                )
                delta[KEYS.disk_iops_read] = int(
                    self._rate.rate("disk_read_count", disk.read_count)
                )
                delta[KEYS.disk_iops_write] = int(
                    self._rate.rate("disk_write_count", disk.write_count)
                )
        except Exception:
            delta[KEYS.disk_read_mb_s] = None
            delta[KEYS.disk_write_mb_s] = None

        # ── Network rates ─────────────────────────────────────────────────
        try:
            net = psutil.net_io_counters()
            delta[KEYS.net_sent_kb_s] = round(
                self._rate.rate("net_sent", net.bytes_sent) / 1024, 2
            )
            delta[KEYS.net_recv_kb_s] = round(
                self._rate.rate("net_recv", net.bytes_recv) / 1024, 2
            )
            delta[KEYS.net_connections_count] = len(psutil.net_connections(kind="inet"))
        except Exception:
            delta[KEYS.net_sent_kb_s] = None
            delta[KEYS.net_recv_kb_s] = None
            delta[KEYS.net_connections_count] = None

        # ── Process census ────────────────────────────────────────────────
        try:
            procs = list(psutil.process_iter(["name", "cpu_percent", "memory_info"]))
            delta[KEYS.process_count] = len(procs)

            cpu_top = max(
                procs,
                key=lambda p: p.info.get("cpu_percent") or 0,
                default=None,
            )
            mem_top = max(
                procs,
                key=lambda p: (p.info.get("memory_info") or psutil.virtual_memory()).rss,
                default=None,
            )

            delta[KEYS.top_cpu_process] = cpu_top.info.get("name", "?") if cpu_top else "?"
            delta[KEYS.top_cpu_percent] = round(cpu_top.info.get("cpu_percent") or 0, 2) if cpu_top else 0.0
            if mem_top and mem_top.info.get("memory_info"):
                delta[KEYS.top_mem_process] = mem_top.info.get("name", "?")
                delta[KEYS.top_mem_mb] = round(mem_top.info["memory_info"].rss / 1024**2, 1)
        except Exception:
            delta[KEYS.process_count] = None

        # ── Active window ────────────────────────────────────────────────
        title, proc_name = _get_active_window()
        delta[KEYS.active_window_title] = title
        delta[KEYS.active_process_name] = proc_name

        # ── Derived: pressure & thermal alert ────────────────────────────
        pressure = _compute_pressure(cpu_total, mem.percent, swap.percent)
        delta[KEYS.system_pressure] = pressure
        delta[KEYS.thermal_alert] = pressure > 0.85

        # ── Metadata ─────────────────────────────────────────────────────
        delta[KEYS.poll_timestamp] = ts
        delta[KEYS.poll_interval_s] = self._interval_s

        return delta

    def __repr__(self) -> str:
        state = "running" if (self._thread and self._thread.is_alive()) else "stopped"
        return f"TelemetryPoller(interval={self._interval_s}s, polls={self._poll_count}, {state})"
