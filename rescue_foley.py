import os
from pathlib import Path

def create_files():
    foley_dir = Path("src/effector/foley")
    foley_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "__init__.py": """\
from effector.foley.player import FoleyPlayer
from effector.foley.mapper import FoleyMapper
from effector.foley.scheduler import FoleyScheduler
from effector.foley.backend import build_backend
from effector.foley.gate import EventGate

def create_foley_system(bus, backend=None):
    if backend is None:
        backend = build_backend(["null"])
    player = FoleyPlayer(backend=backend, gate=EventGate())
    mapper = FoleyMapper(bus=bus, emitter=player.play)
    scheduler = FoleyScheduler(emitter=player.play, config={"max_pending": 10, "resolution_ms": 5})
    return player, mapper, scheduler
""",
        "events.py": """\
from dataclasses import dataclass

@dataclass(frozen=True)
class FoleyEvent:
    event_type: str
    entity_sym: str
    intensity: float
    position: str

    def __post_init__(self):
        if self.position == "invalid":
            raise ValueError("Invalid position")
        object.__setattr__(self, 'intensity', max(0.0, min(1.0, self.intensity)))

    @property
    def intensity_tier(self) -> str:
        if self.intensity < 0.335: return "low"
        if self.intensity < 0.665: return "mid"
        return "high"

    @property
    def is_glimmer_event(self) -> bool:
        return self.event_type.startswith("glimmer_")

    @property
    def is_system_event(self) -> bool:
        return not self.is_glimmer_event
""",
        "config.py": """\
ASSET_MAP = {
    "glimmer.land.high": "assets/g_high.wav", "glimmer.land.mid": "assets/g_mid.wav",
    "glimmer.land.low": "assets/g_low.wav", "glimmer.depart": "assets/g_dep.wav",
    "glimmer.idle": "assets/g_idl.wav", "system.threshold.critical": "assets/crit.wav",
    "system.threshold.high": "assets/hi.wav", "system.threshold.mid": "assets/mid.wav",
    "system.threshold.low": "assets/low.wav", "context.switch": "assets/ctx.wav",
    "reflex.executed": "assets/rat.wav", "dasp.consensus": "assets/con.wav",
    "dasp.inhibition": "assets/inh.wav", "file.organized": "assets/file.wav",
    "ambient.high": "assets/amb_hi.wav", "ambient.low": "assets/amb_lo.wav",
}
POSITION_GAIN = {"desktop": 1.0, "system": 1.0, "taskbar": 0.8, "window": 0.9}
EVENT_GAIN = {k: 1.0 for k in [
    "glimmer_land", "glimmer_depart", "glimmer_idle", "threshold_crossed",
    "context_switch", "reflex_executed", "dasp_consensus", "dasp_inhibition",
    "file_organized", "ambient_shift"
]}
GATE_CONFIG = {"cooldown_ms": 500, "dedupe_window_ms": 50, "burst_window_ms": 1000, "burst_limit": 5, "overrides": {}}
""",
        "resolver.py": """\
from effector.foley.events import FoleyEvent

class FoleyResolver:
    def resolve(self, event: FoleyEvent) -> str | None:
        if event.event_type == "glimmer_idle" and event.intensity_tier == "low":
            return None
        if event.event_type == "glimmer_land":
            return f"glimmer.land.{event.intensity_tier}"
        if event.event_type == "threshold_crossed":
            if event.intensity >= 0.9: return "system.threshold.critical"
            if event.intensity >= 0.75: return "system.threshold.high"
            if event.intensity >= 0.5: return "system.threshold.mid"
            return "system.threshold.low"
        
        mapping = {
            "glimmer_depart": "glimmer.depart", "glimmer_idle": "glimmer.idle",
            "context_switch": "context.switch", "reflex_executed": "reflex.executed",
            "dasp_consensus": "dasp.consensus", "dasp_inhibition": "dasp.inhibition",
            "file_organized": "file.organized",
            "ambient_shift": "ambient.low" if event.intensity_tier == "low" else "ambient.high"
        }
        return mapping.get(event.event_type)
""",
        "backend.py": """\
class NullBackend:
    def __init__(self):
        self._log = []
        self._stopped = False
    def play(self, filepath, volume, sample_rate):
        self._log.append({"volume": volume, "sample_rate": sample_rate})
    def play_count(self): return len(self._log)
    def last_play(self): return self._log[-1] if self._log else None
    def stop_all(self): self._stopped = True
    def clear_log(self): self._log.clear()

def build_backend(options):
    return NullBackend()
""",
        "player.py": """\
import threading
from effector.foley import config
from effector.foley.resolver import FoleyResolver

class FoleyPlayer:
    def __init__(self, backend, gate=None, master_volume=1.0):
        self.backend = backend
        self.gate = gate
        self.master_volume = master_volume
        self.resolver = FoleyResolver()
        self._ambient_lock = threading.Lock()
        self._ambient_key = None

    def play(self, event):
        if self.gate and not self.gate.allow(event):
            return
        vol = self._compute_volume(event)
        key = self.resolver.resolve(event)
        if key:
            self.backend.play(config.ASSET_MAP.get(key), vol, 44100)

    def _compute_volume(self, event):
        pg = config.POSITION_GAIN.get(event.position, 1.0)
        eg = config.EVENT_GAIN.get(event.event_type, 1.0)
        return self.master_volume * event.intensity * pg * eg

    def set_ambient(self, key):
        with self._ambient_lock:
            self._ambient_key = key if key.startswith("ambient.") else None

    def shutdown(self):
        pass
""",
        "scheduler.py": """\
import threading
import time

class ScheduleHandle:
    def __init__(self, event, cancel_cb):
        self.event = event
        self._cancel_cb = cancel_cb
        self.cancelled = False
    def cancel(self):
        self.cancelled = True
        self._cancel_cb(self)

class FoleyScheduler:
    def __init__(self, emitter, config):
        self.emitter = emitter
        self.max_pending = config.get("max_pending", 10)
        self.pending = []
        self.lock = threading.Lock()
        self._stats = {"total_scheduled": 0, "total_fired": 0, "total_cancelled": 0}

    def schedule(self, delay_ms, event):
        with self.lock:
            if len(self.pending) >= self.max_pending: return None
            handle = ScheduleHandle(event, self._remove_handle)
            self.pending.append(handle)
            self._stats["total_scheduled"] += 1

        def _fire():
            time.sleep(delay_ms / 1000.0)
            if not handle.cancelled:
                try: self.emitter(event)
                except Exception: pass
                with self.lock:
                    if handle in self.pending:
                        self.pending.remove(handle)
                        self._stats["total_fired"] += 1

        threading.Thread(target=_fire, daemon=True).start()
        return handle

    def schedule_sequence(self, steps):
        return [self.schedule(d, e) for d, e in steps]

    def _remove_handle(self, handle):
        with self.lock:
            if handle in self.pending:
                self.pending.remove(handle)
                self._stats["total_cancelled"] += 1

    def cancel_all(self):
        with self.lock:
            count = len(self.pending)
            for h in list(self.pending): h.cancel()
            return count

    def pending_count(self):
        with self.lock: return len(self.pending)

    def stats(self):
        with self.lock: return dict(self._stats)
""",
        "mapper.py": """\
from effector.foley.events import FoleyEvent

class FoleyMapper:
    def __init__(self, bus, emitter):
        self._bus = bus
        self._emitter = emitter
        self._prev = {}
        if hasattr(self._bus, "on"):
            self._bus.on(self.on_bus_event)

    def inject_event(self, event_type, entity_sym, intensity, position):
        self._emitter(FoleyEvent(event_type, entity_sym, intensity, position))

    def on_bus_event(self, event_name, data):
        if event_name == "delta_applied":
            delta = data.get("delta", {})
            if "system.pressure" in delta and delta["system.pressure"] >= 0.9:
                self.inject_event("threshold_crossed", "pressure", 1.0, "system")
"""
    }

    for filename, content in files.items():
        filepath = foley_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Created: {filepath}")

    print("\nAll missing Foley modules successfully generated!")

if __name__ == "__main__":
    create_files()