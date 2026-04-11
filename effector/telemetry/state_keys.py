"""
Telemetry State Keys — canonical names for all OS-polled fields.
All StateBus keys written by the poller live in this namespace.
Grouping mirrors the IEP expected_state_change schema: agents can declare
predicted_delta using only these keys, keeping the ontology stable.
"""

from __future__ import annotations
from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class _Keys:
    # ── CPU ────────────────────────────────────────────────────────────────
    cpu_percent_total: str = "cpu.percent.total"
    cpu_percent_per_core: str = "cpu.percent.per_core"
    cpu_freq_mhz: str = "cpu.freq.mhz"
    cpu_ctx_switches_sec: str = "cpu.ctx_switches_sec"

    # ── Memory ─────────────────────────────────────────────────────────────
    ram_total_mb: str = "ram.total_mb"
    ram_used_mb: str = "ram.used_mb"
    ram_available_mb: str = "ram.available_mb"
    ram_percent: str = "ram.percent"
    swap_used_mb: str = "swap.used_mb"
    swap_percent: str = "swap.percent"

    # ── Disk I/O ───────────────────────────────────────────────────────────
    disk_read_mb_s: str = "disk.read_mb_s"
    disk_write_mb_s: str = "disk.write_mb_s"
    disk_iops_read: str = "disk.iops_read"
    disk_iops_write: str = "disk.iops_write"

    # ── Network ────────────────────────────────────────────────────────────
    net_sent_kb_s: str = "net.sent_kb_s"
    net_recv_kb_s: str = "net.recv_kb_s"
    net_connections_count: str = "net.connections_count"

    # ── Process / Desktop ─────────────────────────────────────────────────
    active_window_title: str = "desktop.active_window"
    active_process_name: str = "desktop.active_process"
    process_count: str = "process.count"
    top_cpu_process: str = "process.top_cpu.name"
    top_mem_process: str = "process.top_mem.name"
    top_cpu_percent: str = "process.top_cpu.percent"
    top_mem_mb: str = "process.top_mem.mb"

    # ── Derived / Health ──────────────────────────────────────────────────
    system_pressure: str = "system.pressure"      # composite 0-1
    thermal_alert: str = "system.thermal_alert"   # bool
    poll_timestamp: str = "telemetry.timestamp"
    poll_interval_s: str = "telemetry.interval_s"

    def all_keys(self) -> list[str]:
        return [getattr(self, f.name) for f in fields(self)]

    def cpu_keys(self) -> list[str]:
        return [k for k in self.all_keys() if k.startswith("cpu.")]

    def ram_keys(self) -> list[str]:
        return [k for k in self.all_keys() if k.startswith("ram.") or k.startswith("swap.")]

    def process_keys(self) -> list[str]:
        return [k for k in self.all_keys() if k.startswith("process.") or k.startswith("desktop.")]

    def health_keys(self) -> list[str]:
        return [self.system_pressure, self.thermal_alert, self.poll_timestamp]


# Singleton — import this everywhere
KEYS = _Keys()
