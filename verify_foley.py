"""
verify_foley.py — Standalone Foley Verification
=================================================
Validates all foley components without:
  - Ollama
  - Audio hardware
  - The full effector package installed
  - Any external dependencies beyond stdlib

Run from H:\\Effector:
    python verify_foley.py

Or for verbose output:
    python verify_foley.py --verbose

Exit code 0 = all passed. Non-zero = failures.

This script shims the import path so you can run it from the project
root before pip install, in CI, or on a fresh checkout.
"""

from __future__ import annotations

import sys
import time
import threading
import traceback
from pathlib import Path

# ── Path shimming ──────────────────────────────────────────────────────────────
# Make effector.foley importable without pip install

_PROJECT_ROOT = Path(__file__).parent
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ── Test registry ─────────────────────────────────────────────────────────────

_TESTS: list[tuple[str, callable]] = []
_VERBOSE = "--verbose" in sys.argv or "-v" in sys.argv


def test(name: str):
    def decorator(fn):
        _TESTS.append((name, fn))
        return fn
    return decorator


def assert_eq(actual, expected, msg=""):
    if actual != expected:
        raise AssertionError(
            f"{msg + ': ' if msg else ''}"
            f"Expected {expected!r}, got {actual!r}"
        )


def assert_in(value, collection, msg=""):
    if value not in collection:
        raise AssertionError(
            f"{msg + ': ' if msg else ''}"
            f"{value!r} not in {collection!r}"
        )


def assert_none(value, msg=""):
    if value is not None:
        raise AssertionError(f"{msg + ': ' if msg else ''}Expected None, got {value!r}")


def assert_not_none(value, msg=""):
    if value is None:
        raise AssertionError(f"{msg + ': ' if msg else ''}Expected non-None value")


def wait_until(condition, timeout=1.0, interval=0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(interval)
    return False


# ── FoleyEvent tests ───────────────────────────────────────────────────────────

@test("FoleyEvent: basic construction")
def _():
    from effector.foley.events import FoleyEvent
    e = FoleyEvent("glimmer_land", "Fc", 0.8, "desktop")
    assert_eq(e.event_type, "glimmer_land")
    assert_eq(e.entity_sym, "Fc")
    assert_eq(e.intensity, 0.8)
    assert_eq(e.position, "desktop")


@test("FoleyEvent: intensity clamped to [0, 1]")
def _():
    from effector.foley.events import FoleyEvent
    e_low  = FoleyEvent("glimmer_land", "Fc", -0.5, "desktop")
    e_high = FoleyEvent("glimmer_land", "Fc",  1.5, "desktop")
    assert_eq(e_low.intensity, 0.0)
    assert_eq(e_high.intensity, 1.0)


@test("FoleyEvent: invalid position raises ValueError")
def _():
    from effector.foley.events import FoleyEvent
    try:
        FoleyEvent("glimmer_land", "Fc", 0.5, "invalid")
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass


@test("FoleyEvent: intensity_tier boundaries")
def _():
    from effector.foley.events import FoleyEvent
    cases = [
        (0.00, "low"), (0.33, "low"), (0.34, "mid"),
        (0.66, "mid"), (0.67, "high"), (1.00, "high"),
    ]
    for intensity, expected_tier in cases:
        e = FoleyEvent("glimmer_land", "Fc", intensity, "system")
        actual = e.intensity_tier
        assert_eq(actual, expected_tier, f"intensity={intensity}")


@test("FoleyEvent: is_glimmer_event and is_system_event")
def _():
    from effector.foley.events import FoleyEvent
    e1 = FoleyEvent("glimmer_land", "Fc", 0.5, "desktop")
    e2 = FoleyEvent("dasp_consensus", "Sg", 0.9, "system")
    assert e1.is_glimmer_event
    assert not e1.is_system_event
    assert not e2.is_glimmer_event
    assert e2.is_system_event


@test("FoleyEvent: immutable (frozen dataclass)")
def _():
    from effector.foley.events import FoleyEvent
    e = FoleyEvent("glimmer_land", "Fc", 0.5, "desktop")
    try:
        e.intensity = 0.9  # type: ignore
        raise AssertionError("Should have raised FrozenInstanceError")
    except Exception as exc:
        if "cannot assign" in str(exc).lower() or "frozen" in str(exc).lower() or "FrozenInstance" in type(exc).__name__:
            pass  # Expected
        else:
            raise


# ── FoleyResolver tests ────────────────────────────────────────────────────────

@test("FoleyResolver: glimmer_land high/mid/low")
def _():
    from effector.foley.events import FoleyEvent
    from effector.foley.resolver import FoleyResolver
    r = FoleyResolver()
    assert_eq(r.resolve(FoleyEvent("glimmer_land", "Fc", 0.9, "desktop")), "glimmer.land.high")
    assert_eq(r.resolve(FoleyEvent("glimmer_land", "Fc", 0.5, "desktop")), "glimmer.land.mid")
    assert_eq(r.resolve(FoleyEvent("glimmer_land", "Fc", 0.1, "desktop")), "glimmer.land.low")


@test("FoleyResolver: glimmer_idle low intensity is None")
def _():
    from effector.foley.events import FoleyEvent
    from effector.foley.resolver import FoleyResolver
    r = FoleyResolver()
    assert_none(r.resolve(FoleyEvent("glimmer_idle", "Cf", 0.2, "taskbar")))


@test("FoleyResolver: threshold_crossed critical at intensity >= 0.9")
def _():
    from effector.foley.events import FoleyEvent
    from effector.foley.resolver import FoleyResolver
    r = FoleyResolver()
    assert_eq(
        r.resolve(FoleyEvent("threshold_crossed", "pressure", 1.0, "system")),
        "system.threshold.critical"
    )
    assert_eq(
        r.resolve(FoleyEvent("threshold_crossed", "pressure", 0.90, "system")),
        "system.threshold.critical"
    )
    assert_eq(
        r.resolve(FoleyEvent("threshold_crossed", "pressure", 0.75, "system")),
        "system.threshold.high"
    )


@test("FoleyResolver: unknown event type returns None")
def _():
    from effector.foley.events import FoleyEvent
    from effector.foley.resolver import FoleyResolver
    r = FoleyResolver()
    assert_none(r.resolve(FoleyEvent("completely_unknown", "x", 0.5, "system")))


@test("FoleyResolver: all resolved keys exist in ASSET_MAP")
def _():
    from effector.foley.events import FoleyEvent
    from effector.foley.resolver import FoleyResolver
    from effector.foley import config

    r = FoleyResolver()
    test_events = [
        FoleyEvent("glimmer_land",      "Fc", 0.9, "desktop"),
        FoleyEvent("glimmer_land",      "Fc", 0.5, "desktop"),
        FoleyEvent("glimmer_land",      "Fc", 0.1, "desktop"),
        FoleyEvent("glimmer_depart",    "Fc", 0.5, "desktop"),
        FoleyEvent("glimmer_idle",      "Pr", 0.6, "taskbar"),
        FoleyEvent("threshold_crossed", "pressure", 1.0, "system"),
        FoleyEvent("threshold_crossed", "pressure", 0.75, "system"),
        FoleyEvent("threshold_crossed", "pressure", 0.5, "system"),
        FoleyEvent("threshold_crossed", "pressure", 0.2, "system"),
        FoleyEvent("context_switch",    "ctx", 0.6, "window"),
        FoleyEvent("reflex_executed",   "RAT", 0.7, "system"),
        FoleyEvent("dasp_consensus",    "Sg",  1.0, "system"),
        FoleyEvent("dasp_inhibition",   "veto", 0.8, "system"),
        FoleyEvent("file_organized",    "arc", 0.5, "desktop"),
        FoleyEvent("ambient_shift",     "high", 0.8, "system"),
        FoleyEvent("ambient_shift",     "low",  0.5, "system"),
    ]
    missing = []
    for event in test_events:
        key = r.resolve(event)
        if key is not None and key not in config.ASSET_MAP:
            missing.append((event.event_type, key))

    if missing:
        raise AssertionError(f"Keys not in ASSET_MAP: {missing}")


# ── EventGate tests ────────────────────────────────────────────────────────────

_PERMISSIVE_GATE = {
    "cooldown_ms": 1,
    "dedupe_window_ms": 1,
    "burst_window_ms": 2000,
    "burst_limit": 100,
    "overrides": {},
}

_STRICT_GATE = {
    "cooldown_ms": 60_000,
    "dedupe_window_ms": 30_000,
    "burst_window_ms": 2000,
    "burst_limit": 1,
    "overrides": {},
}


@test("EventGate: first event always allowed")
def _():
    from effector.foley.events import FoleyEvent
    from effector.foley.gate import EventGate
    gate = EventGate(_PERMISSIVE_GATE)
    assert gate.allow(FoleyEvent("glimmer_land", "Fc", 0.5, "desktop"))


@test("EventGate: second event within cooldown rejected")
def _():
    from effector.foley.events import FoleyEvent
    from effector.foley.gate import EventGate
    gate = EventGate(_STRICT_GATE)
    gate.allow(FoleyEvent("glimmer_land", "Fc", 0.5, "desktop"))
    assert not gate.allow(FoleyEvent("glimmer_land", "Fc", 0.5, "desktop"))


@test("EventGate: different event types independent")
def _():
    from effector.foley.events import FoleyEvent
    from effector.foley.gate import EventGate
    gate = EventGate(_STRICT_GATE)
    gate.allow(FoleyEvent("glimmer_land", "Fc", 0.5, "desktop"))
    # Different type — should be independent
    assert gate.allow(FoleyEvent("dasp_consensus", "Sg", 0.9, "system"))


@test("EventGate: reset clears state for one type")
def _():
    from effector.foley.events import FoleyEvent
    from effector.foley.gate import EventGate
    gate = EventGate(_STRICT_GATE)
    gate.allow(FoleyEvent("glimmer_land", "Fc", 0.5, "desktop"))
    assert not gate.allow(FoleyEvent("glimmer_land", "Fc", 0.5, "desktop"))
    gate.reset("glimmer_land")
    assert gate.allow(FoleyEvent("glimmer_land", "Fc", 0.5, "desktop"))


@test("EventGate: burst suppression fires exactly at limit")
def _():
    from effector.foley.events import FoleyEvent
    from effector.foley.gate import EventGate
    cfg = {"cooldown_ms": 1, "dedupe_window_ms": 1,
           "burst_window_ms": 2000, "burst_limit": 3, "overrides": {}}
    gate = EventGate(cfg)
    results = []
    for _ in range(6):
        time.sleep(0.02)
        results.append(gate.allow(FoleyEvent("glimmer_land", "Fc", 0.5, "desktop")))
    assert sum(results) == 3, f"Expected 3 allowed, got {sum(results)}: {results}"


# ── NullBackend tests ──────────────────────────────────────────────────────────

@test("NullBackend: records play calls")
def _():
    from effector.foley.backend import NullBackend
    b = NullBackend()
    assert_eq(b.play_count(), 0)
    b.play(None, 0.5, 44100)
    assert_eq(b.play_count(), 1)
    call = b.last_play()
    assert_not_none(call)
    assert_eq(call["volume"], 0.5)
    assert_eq(call["sample_rate"], 44100)


@test("NullBackend: stop_all sets stopped flag")
def _():
    from effector.foley.backend import NullBackend
    b = NullBackend()
    b.stop_all()
    assert b._stopped


@test("NullBackend: clear_log resets count")
def _():
    from effector.foley.backend import NullBackend
    b = NullBackend()
    for _ in range(5):
        b.play(None, 0.5, 44100)
    b.clear_log()
    assert_eq(b.play_count(), 0)


@test("build_backend: returns NullBackend when null is only option")
def _():
    from effector.foley.backend import build_backend, NullBackend
    b = build_backend(["null"])
    assert isinstance(b, NullBackend)


# ── FoleyPlayer tests ──────────────────────────────────────────────────────────

@test("FoleyPlayer: play calls backend (with NullBackend)")
def _():
    from effector.foley.events import FoleyEvent
    from effector.foley.backend import NullBackend
    from effector.foley.player import FoleyPlayer
    from effector.foley.gate import EventGate

    backend = NullBackend()
    player = FoleyPlayer(backend=backend, gate=EventGate(_PERMISSIVE_GATE), master_volume=0.7)
    player.play(FoleyEvent("glimmer_land", "Fc", 0.8, "desktop"))

    assert wait_until(lambda: backend.play_count() >= 1, timeout=1.0), "Backend never called"
    player.shutdown()


@test("FoleyPlayer: volume formula correct")
def _():
    from effector.foley.events import FoleyEvent
    from effector.foley.backend import NullBackend
    from effector.foley.player import FoleyPlayer
    from effector.foley import config

    player = FoleyPlayer(backend=NullBackend(), master_volume=0.7)
    event = FoleyEvent("glimmer_land", "Fc", 0.8, "desktop")
    expected = (
        0.7
        * 0.8
        * config.POSITION_GAIN["desktop"]
        * config.EVENT_GAIN["glimmer_land"]
    )
    actual = player._compute_volume(event)
    if abs(actual - expected) > 1e-9:
        raise AssertionError(f"Volume {actual:.6f} ≠ expected {expected:.6f}")
    player.shutdown()


@test("FoleyPlayer: gate rejection prevents backend call")
def _():
    from effector.foley.events import FoleyEvent
    from effector.foley.backend import NullBackend
    from effector.foley.player import FoleyPlayer
    from effector.foley.gate import EventGate

    backend = NullBackend()
    player = FoleyPlayer(backend=backend, gate=EventGate(_STRICT_GATE), master_volume=0.7)
    player.play(FoleyEvent("glimmer_land", "Fc", 0.8, "desktop"))  # First: allowed
    wait_until(lambda: backend.play_count() >= 1, timeout=1.0)
    count_after_first = backend.play_count()

    player.play(FoleyEvent("glimmer_land", "Fc", 0.8, "desktop"))  # Second: blocked
    time.sleep(0.1)
    assert_eq(backend.play_count(), count_after_first, "Gate should have blocked second play")
    player.shutdown()


@test("FoleyPlayer: set_ambient records key")
def _():
    from effector.foley.backend import NullBackend
    from effector.foley.player import FoleyPlayer

    player = FoleyPlayer(backend=NullBackend())
    player.set_ambient("ambient.low")
    with player._ambient_lock:
        assert_eq(player._ambient_key, "ambient.low")
    player.set_ambient("ambient.high")
    with player._ambient_lock:
        assert_eq(player._ambient_key, "ambient.high")
    player.shutdown()


@test("FoleyPlayer: set_ambient invalid key ignored")
def _():
    from effector.foley.backend import NullBackend
    from effector.foley.player import FoleyPlayer

    player = FoleyPlayer(backend=NullBackend())
    player.set_ambient("bad.key")
    with player._ambient_lock:
        assert_none(player._ambient_key)
    player.shutdown()


@test("FoleyPlayer: play is non-blocking")
def _():
    from effector.foley.events import FoleyEvent
    from effector.foley.backend import NullBackend
    from effector.foley.player import FoleyPlayer
    from effector.foley.gate import EventGate

    player = FoleyPlayer(backend=NullBackend(), gate=EventGate(_PERMISSIVE_GATE))
    start = time.monotonic()
    for _ in range(20):
        player.play(FoleyEvent("glimmer_land", "Fc", 0.5, "desktop"))
    elapsed = time.monotonic() - start
    assert elapsed < 0.05, f"play() loop took {elapsed*1000:.1f}ms — not non-blocking"
    player.shutdown()


# ── FoleyScheduler tests ───────────────────────────────────────────────────────

@test("FoleyScheduler: event fires after delay")
def _():
    from effector.foley.events import FoleyEvent
    from effector.foley.scheduler import FoleyScheduler

    fired = threading.Event()
    fired_at = [0.0]
    start = time.monotonic()

    def emit(e):
        fired_at[0] = time.monotonic()
        fired.set()

    sched = FoleyScheduler(emit, {"max_pending": 10, "resolution_ms": 5})
    sched.schedule(60, FoleyEvent("glimmer_land", "Fc", 0.7, "desktop"))

    assert fired.wait(timeout=0.5), "Event never fired"
    elapsed_ms = (fired_at[0] - start) * 1000
    assert 30 <= elapsed_ms <= 250, f"Delay {elapsed_ms:.1f}ms out of range"


@test("FoleyScheduler: cancelled event does not fire")
def _():
    from effector.foley.events import FoleyEvent
    from effector.foley.scheduler import FoleyScheduler

    count = [0]

    def emit(e):
        count[0] += 1

    sched = FoleyScheduler(emit, {"max_pending": 10, "resolution_ms": 5})
    se = sched.schedule(100, FoleyEvent("glimmer_land", "Fc", 0.7, "desktop"))
    assert se is not None
    se.cancel()
    time.sleep(0.15)
    assert_eq(count[0], 0, "Cancelled event fired")


@test("FoleyScheduler: sequence fires all events")
def _():
    from effector.foley.events import FoleyEvent
    from effector.foley.scheduler import FoleyScheduler

    fired = []

    def emit(e):
        fired.append(e.event_type)

    sched = FoleyScheduler(emit, {"max_pending": 10, "resolution_ms": 5})
    sched.schedule_sequence([
        (10,  FoleyEvent("glimmer_land",   "Fc", 0.8, "desktop")),
        (50,  FoleyEvent("dasp_consensus", "Sg", 1.0, "system")),
    ])
    assert wait_until(lambda: len(fired) >= 2, timeout=0.5), f"Only {len(fired)} fired"
    assert_in("glimmer_land", fired)
    assert_in("dasp_consensus", fired)


# ── FoleyMapper tests ──────────────────────────────────────────────────────────

@test("FoleyMapper: subscribes to bus on construction")
def _():
    from effector.foley.mapper import FoleyMapper

    class StubBus:
        def __init__(self): self.listeners = []
        def on(self, cb): self.listeners.append(cb)

    bus = StubBus()
    FoleyMapper(bus, emitter=lambda e: None)
    assert_eq(len(bus.listeners), 1)


@test("FoleyMapper: pressure crossing emits threshold_crossed")
def _():
    from effector.foley.events import FoleyEvent
    from effector.foley.mapper import FoleyMapper

    class StubBus:
        def __init__(self): self._listeners = []
        def on(self, cb): self._listeners.append(cb)
        def push(self, event_name, data):
            for cb in self._listeners: cb(event_name, data)

    bus = StubBus()
    emitted = []
    mapper = FoleyMapper(bus, emitter=emitted.append)
    mapper._prev["system.pressure"] = 0.0

    bus.push("delta_applied", {"delta": {"system.pressure": 0.90}})

    assert len(emitted) >= 1
    assert emitted[0].event_type == "threshold_crossed"
    assert emitted[0].intensity == 1.0


@test("FoleyMapper: inject_event bypasses StateBus")
def _():
    from effector.foley.mapper import FoleyMapper

    class StubBus:
        def on(self, cb): pass

    emitted = []
    mapper = FoleyMapper(StubBus(), emitter=emitted.append)
    mapper.inject_event("dasp_consensus", "Sg", 0.9, "system")
    assert_eq(len(emitted), 1)
    assert_eq(emitted[0].event_type, "dasp_consensus")


# ── Integration connector tests ────────────────────────────────────────────────

@test("DASPFoleyConnector: consensus_cleared emits dasp_consensus")
def _():
    from effector.foley.mapper import FoleyMapper
    from effector.foley.integration import DASPFoleyConnector

    class StubBus:
        def on(self, cb): pass

    emitted = []
    mapper = FoleyMapper(StubBus(), emitter=emitted.append)
    connector = DASPFoleyConnector(mapper)

    connector.on_event("consensus_cleared", {"consensus_score": 0.88})
    assert_eq(len(emitted), 1)
    assert_eq(emitted[0].event_type, "dasp_consensus")
    assert abs(emitted[0].intensity - 0.88) < 0.001


@test("DASPFoleyConnector: inhibition trigger emits dasp_inhibition")
def _():
    from effector.foley.mapper import FoleyMapper
    from effector.foley.integration import DASPFoleyConnector

    class StubBus:
        def on(self, cb): pass

    emitted = []
    mapper = FoleyMapper(StubBus(), emitter=emitted.append)
    connector = DASPFoleyConnector(mapper)

    connector.on_event("escalation_triggered", {"trigger": "inhibition"})
    assert_eq(len(emitted), 1)
    assert_eq(emitted[0].event_type, "dasp_inhibition")
    assert emitted[0].intensity > 0.8


@test("DASPFoleyConnector: unknown events ignored safely")
def _():
    from effector.foley.mapper import FoleyMapper
    from effector.foley.integration import DASPFoleyConnector

    class StubBus:
        def on(self, cb): pass

    emitted = []
    mapper = FoleyMapper(StubBus(), emitter=emitted.append)
    connector = DASPFoleyConnector(mapper)
    connector.on_event("some_unknown_event", {"data": 42})
    assert_eq(len(emitted), 0)


@test("PressureFoleyConnector: high pressure shifts ambient")
def _():
    from effector.foley.backend import NullBackend
    from effector.foley.player import FoleyPlayer
    from effector.foley.integration import PressureFoleyConnector

    player = FoleyPlayer(backend=NullBackend())
    conn = PressureFoleyConnector(player, initial_key="ambient.low")

    # Push pressure above threshold
    conn.on_bus_event("delta_applied", {"delta": {"system.pressure": 0.80}})
    with player._ambient_lock:
        assert_eq(player._ambient_key, "ambient.high")
    player.shutdown()


@test("PressureFoleyConnector: pressure recovery shifts back to low")
def _():
    from effector.foley.backend import NullBackend
    from effector.foley.player import FoleyPlayer
    from effector.foley.integration import PressureFoleyConnector

    player = FoleyPlayer(backend=NullBackend())
    conn = PressureFoleyConnector(player, initial_key="ambient.high")
    conn._current = "ambient.high"

    conn.on_bus_event("delta_applied", {"delta": {"system.pressure": 0.30}})
    with player._ambient_lock:
        assert_eq(player._ambient_key, "ambient.low")
    player.shutdown()


@test("compose_event_handlers: both handlers called")
def _():
    from effector.foley.integration import compose_event_handlers
    log = []
    h1 = lambda e, d: log.append(("h1", e))
    h2 = lambda e, d: log.append(("h2", e))
    combined = compose_event_handlers(h1, h2)
    combined("test_event", {})
    assert_in(("h1", "test_event"), log)
    assert_in(("h2", "test_event"), log)


@test("compose_event_handlers: error in one handler doesn't stop others")
def _():
    from effector.foley.integration import compose_event_handlers
    log = []
    def bad_handler(e, d): raise RuntimeError("boom")
    def good_handler(e, d): log.append(e)
    combined = compose_event_handlers(bad_handler, good_handler)
    combined("test_event", {})
    assert_in("test_event", log)


# ── create_foley_system factory ────────────────────────────────────────────────

@test("create_foley_system: returns (player, mapper, scheduler)")
def _():
    from effector.foley.backend import NullBackend
    from effector.foley.player import FoleyPlayer
    from effector.foley.mapper import FoleyMapper
    from effector.foley.scheduler import FoleyScheduler
    from effector.foley import create_foley_system

    class StubBus:
        def __init__(self): self._listeners = []
        def on(self, cb): self._listeners.append(cb)

    bus = StubBus()
    player, mapper, scheduler = create_foley_system(bus, backend=NullBackend())
    assert isinstance(player, FoleyPlayer)
    assert isinstance(mapper, FoleyMapper)
    assert isinstance(scheduler, FoleyScheduler)
    # Mapper should have subscribed to bus
    assert len(bus._listeners) == 1
    player.shutdown()


# ── Runner ─────────────────────────────────────────────────────────────────────

def run_all() -> int:
    passed = 0
    failed = 0
    errors = []

    print(f"\n{'━'*60}")
    print(f"  Effector Foley — Verification Suite")
    print(f"{'━'*60}\n")

    for name, fn in _TESTS:
        try:
            fn()
            passed += 1
            if _VERBOSE:
                print(f"  ✓  {name}")
        except Exception as exc:
            failed += 1
            errors.append((name, exc))
            print(f"  ✗  {name}")
            if _VERBOSE:
                traceback.print_exc()

    print(f"\n{'━'*60}")
    print(f"  {passed}/{passed + failed} passed", end="")
    if failed:
        print(f"  ·  {failed} failed")
        print()
        for name, exc in errors:
            print(f"  ✗  {name}")
            print(f"     {type(exc).__name__}: {exc}")
            if _VERBOSE:
                print()
    else:
        print("  · All clear.\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all())
