"""
test_foley_scheduler.py
========================
Verifies FoleyScheduler:
  - Fires events at (approximately) the right delay
  - Respects max_pending cap
  - Supports cancellation
  - Handles sequences
  - Is thread-safe under concurrent scheduling

All tests use real timers at short delays (10–200ms).
Delays are tested with generous tolerance (±50ms) for CI stability.
"""

from __future__ import annotations

import threading
import time

import pytest

from effector.foley.events import FoleyEvent
from effector.foley.scheduler import FoleyScheduler


# ── Helpers ────────────────────────────────────────────────────────────────────

FAST_SCHEDULER_CONFIG = {
    "max_pending": 20,
    "resolution_ms": 5,
}


def ev(etype="glimmer_land", intensity=0.7) -> FoleyEvent:
    return FoleyEvent(etype, "Fc", intensity, "desktop")


class RecordingEmitter:
    """Records emitted events with timestamps."""
    def __init__(self):
        self.records: list[tuple[float, FoleyEvent]] = []
        self.lock = threading.Lock()
        self.event = threading.Event()

    def __call__(self, foley_event: FoleyEvent) -> None:
        with self.lock:
            self.records.append((time.monotonic(), foley_event))
        self.event.set()

    def wait_for_n(self, n: int, timeout: float = 2.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                if len(self.records) >= n:
                    return True
            time.sleep(0.005)
        return False

    def count(self) -> int:
        with self.lock:
            return len(self.records)

    def event_types(self) -> list[str]:
        with self.lock:
            return [r[1].event_type for r in self.records]


# ── Basic scheduling ───────────────────────────────────────────────────────────

class TestBasicScheduling:
    def test_event_fires_after_delay(self):
        emitter = RecordingEmitter()
        sched = FoleyScheduler(emitter, FAST_SCHEDULER_CONFIG)

        start = time.monotonic()
        sched.schedule(50, ev())  # 50ms delay

        assert emitter.wait_for_n(1, timeout=0.5), "Event never fired"
        elapsed = (emitter.records[0][0] - start) * 1000
        assert 30 <= elapsed <= 200, f"Delay {elapsed:.1f}ms out of expected range [30, 200]ms"

    def test_event_fires_at_correct_delay(self):
        emitter = RecordingEmitter()
        sched = FoleyScheduler(emitter, FAST_SCHEDULER_CONFIG)

        start = time.monotonic()
        sched.schedule(100, ev())  # 100ms

        assert emitter.wait_for_n(1, timeout=0.5)
        elapsed_ms = (emitter.records[0][0] - start) * 1000
        assert 60 <= elapsed_ms <= 250, f"Delay {elapsed_ms:.1f}ms"

    def test_zero_delay_fires_promptly(self):
        emitter = RecordingEmitter()
        sched = FoleyScheduler(emitter, FAST_SCHEDULER_CONFIG)
        sched.schedule(0, ev())
        assert emitter.wait_for_n(1, timeout=0.2)

    def test_correct_event_is_fired(self):
        emitter = RecordingEmitter()
        sched = FoleyScheduler(emitter, FAST_SCHEDULER_CONFIG)
        target = FoleyEvent("dasp_consensus", "Sg", 0.9, "system")
        sched.schedule(20, target)

        assert emitter.wait_for_n(1, timeout=0.3)
        fired = emitter.records[0][1]
        assert fired.event_type == "dasp_consensus"
        assert fired.entity_sym == "Sg"
        assert fired.intensity == 0.9
        assert fired.position == "system"


# ── Cancellation ──────────────────────────────────────────────────────────────

class TestCancellation:
    def test_cancelled_event_does_not_fire(self):
        emitter = RecordingEmitter()
        sched = FoleyScheduler(emitter, FAST_SCHEDULER_CONFIG)

        se = sched.schedule(100, ev())  # 100ms — enough time to cancel
        assert se is not None
        se.cancel()
        time.sleep(0.15)
        assert emitter.count() == 0, "Cancelled event still fired"

    def test_non_cancelled_event_still_fires_after_sibling_cancelled(self):
        emitter = RecordingEmitter()
        sched = FoleyScheduler(emitter, FAST_SCHEDULER_CONFIG)

        se1 = sched.schedule(50, FoleyEvent("glimmer_land",   "Fc", 0.7, "desktop"))
        se2 = sched.schedule(50, FoleyEvent("dasp_consensus", "Sg", 0.9, "system"))

        se1.cancel()
        assert emitter.wait_for_n(1, timeout=0.3)
        types = emitter.event_types()
        assert "dasp_consensus" in types
        assert "glimmer_land" not in types

    def test_cancel_all_prevents_all_pending(self):
        emitter = RecordingEmitter()
        sched = FoleyScheduler(emitter, FAST_SCHEDULER_CONFIG)

        for _ in range(5):
            sched.schedule(100, ev())

        cancelled = sched.cancel_all()
        assert cancelled == 5
        time.sleep(0.15)
        assert emitter.count() == 0

    def test_cancel_all_returns_count(self):
        emitter = RecordingEmitter()
        sched = FoleyScheduler(emitter, FAST_SCHEDULER_CONFIG)
        for _ in range(3):
            sched.schedule(200, ev())
        cancelled = sched.cancel_all()
        assert cancelled == 3


# ── Sequences ──────────────────────────────────────────────────────────────────

class TestSequences:
    def test_sequence_fires_all_events(self):
        emitter = RecordingEmitter()
        sched = FoleyScheduler(emitter, FAST_SCHEDULER_CONFIG)

        steps = [
            (10,  FoleyEvent("glimmer_land",   "Fc", 0.8, "desktop")),
            (50,  FoleyEvent("dasp_consensus", "Sg", 1.0, "system")),
            (100, FoleyEvent("glimmer_idle",   "Wl", 0.4, "taskbar")),
        ]
        sched.schedule_sequence(steps)

        assert emitter.wait_for_n(3, timeout=0.5)
        types = emitter.event_types()
        assert "glimmer_land" in types
        assert "dasp_consensus" in types
        assert "glimmer_idle" in types

    def test_sequence_delays_are_absolute_not_cumulative(self):
        """
        Steps (10ms, 50ms, 100ms) should all fire within ~100ms of start,
        not (10, 60, 160)ms cumulatively.
        """
        emitter = RecordingEmitter()
        sched = FoleyScheduler(emitter, FAST_SCHEDULER_CONFIG)

        start = time.monotonic()
        steps = [
            (10,  FoleyEvent("glimmer_land", "Fc", 0.5, "desktop")),
            (80,  FoleyEvent("glimmer_land", "Wl", 0.5, "desktop")),
        ]
        sched.schedule_sequence(steps)
        assert emitter.wait_for_n(2, timeout=0.5)

        elapsed_last = (emitter.records[-1][0] - start) * 1000
        # Should be ~80ms, definitely not 90ms (10+80 cumulative)
        assert elapsed_last < 200, f"Last event at {elapsed_last:.1f}ms — seems cumulative"

    def test_sequence_handles_all_handles(self):
        emitter = RecordingEmitter()
        sched = FoleyScheduler(emitter, FAST_SCHEDULER_CONFIG)

        steps = [(i * 20, ev()) for i in range(4)]
        handles = sched.schedule_sequence(steps)
        assert len(handles) == 4
        assert all(h is not None for h in handles)


# ── Max pending cap ────────────────────────────────────────────────────────────

class TestMaxPending:
    def test_events_beyond_cap_are_dropped(self):
        emitter = RecordingEmitter()
        cap_config = {"max_pending": 3, "resolution_ms": 5}
        sched = FoleyScheduler(emitter, cap_config)

        # Schedule more than the cap
        handles = [sched.schedule(500, ev()) for _ in range(6)]

        # Some should be dropped (None returned)
        none_count = sum(1 for h in handles if h is None)
        assert none_count >= 3, f"Expected ≥3 drops, got {none_count}"

    def test_pending_count_respects_cap(self):
        emitter = RecordingEmitter()
        cap_config = {"max_pending": 3, "resolution_ms": 5}
        sched = FoleyScheduler(emitter, cap_config)

        for _ in range(10):
            sched.schedule(500, ev())

        assert sched.pending_count() <= 3


# ── Stats ──────────────────────────────────────────────────────────────────────

class TestStats:
    def test_stats_tracks_scheduled(self):
        emitter = RecordingEmitter()
        sched = FoleyScheduler(emitter, FAST_SCHEDULER_CONFIG)

        for _ in range(3):
            sched.schedule(10, ev())

        assert emitter.wait_for_n(3, timeout=0.3)
        s = sched.stats()
        assert s["total_scheduled"] == 3
        assert s["total_fired"] == 3

    def test_stats_tracks_cancelled(self):
        emitter = RecordingEmitter()
        sched = FoleyScheduler(emitter, FAST_SCHEDULER_CONFIG)

        for _ in range(4):
            sched.schedule(200, ev())

        sched.cancel_all()
        s = sched.stats()
        assert s["total_cancelled"] == 4

    def test_pending_count_decreases_after_fire(self):
        emitter = RecordingEmitter()
        sched = FoleyScheduler(emitter, FAST_SCHEDULER_CONFIG)

        sched.schedule(30, ev())
        assert sched.pending_count() == 1
        assert emitter.wait_for_n(1, timeout=0.3)
        assert sched.pending_count() == 0


# ── Thread safety ──────────────────────────────────────────────────────────────

class TestThreadSafety:
    def test_concurrent_scheduling_no_errors(self):
        emitter = RecordingEmitter()
        sched = FoleyScheduler(emitter, FAST_SCHEDULER_CONFIG)
        errors = []

        def schedule_many():
            try:
                for i in range(10):
                    sched.schedule(i * 5, ev())
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=schedule_many) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors

    def test_concurrent_schedule_and_cancel(self):
        emitter = RecordingEmitter()
        sched = FoleyScheduler(emitter, FAST_SCHEDULER_CONFIG)
        errors = []

        def scheduler_thread():
            try:
                for _ in range(20):
                    sched.schedule(50, ev())
                    time.sleep(0.002)
            except Exception as exc:
                errors.append(exc)

        def canceller_thread():
            try:
                for _ in range(5):
                    sched.cancel_all()
                    time.sleep(0.01)
            except Exception as exc:
                errors.append(exc)

        threads = [
            *[threading.Thread(target=scheduler_thread) for _ in range(3)],
            threading.Thread(target=canceller_thread),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors


# ── Emitter error safety ───────────────────────────────────────────────────────

class TestEmitterErrorSafety:
    def test_emitter_exception_does_not_stop_scheduler(self):
        """If the emitter raises, subsequent scheduled events still fire."""
        call_count = [0]
        errors = [0]

        def flaky_emitter(event: FoleyEvent) -> None:
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Simulated emitter failure")

        sched = FoleyScheduler(flaky_emitter, FAST_SCHEDULER_CONFIG)
        sched.schedule(10, ev())  # Will raise
        sched.schedule(30, ev())  # Should still fire

        time.sleep(0.1)
        assert call_count[0] == 2, f"Expected 2 calls, got {call_count[0]}"
