"""
Effector Engine — Main Reasoning Loop
======================================
Integrates:
  A1 — Substrate Telemetry (TelemetryPoller → StateBus)
  A2 — Asymmetric DASP     (local agents → cloud escalation)
  A3 — IEP Validation      (envelope emission → queue)

Lifecycle
---------
  1. TelemetryPoller writes OS state to StateBus every `poll_interval_s` seconds.
  2. After each poll, the reasoning loop fires an AsymmetricDASP debate
     grounded on the current telemetry snapshot.
  3. The debate result is packaged into an IEP envelope and validated.
  4. The envelope (ACK or NACK) is pushed to the EnvelopeQueue.
  5. A queue consumer logs/acts on ACKed envelopes.

Run
---
    python main_loop.py [--poll-interval 5] [--debate-interval 30] [--max-cycles 0]

Options
-------
  --poll-interval N     Telemetry poll interval in seconds  (default: 5)
  --debate-interval N   Seconds between debate cycles       (default: 30)
  --max-cycles N        Stop after N debate cycles; 0=run forever (default: 0)
  --queue-file PATH     JSONL file for envelope persistence (default: iep_queue.jsonl)
  --no-color            Disable ANSI colour output
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Project imports ────────────────────────────────────────────────────────
# Inline minimal StateBus so this file runs standalone without package install
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from effector.telemetry.poller import TelemetryPoller
    from effector.telemetry.state_keys import KEYS
    from effector.adapters.asymmetric_dasp import (
        AsymmetricDASPCoordinator,
        DEFAULT_TIER1,
        DEFAULT_TIER2,
        TierConfig,
    )
    from effector.queue.iep_queue import (
        EnvelopeQueue,
        IEPBuilder,
        IEPValidator,
    )
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    print("Ensure you run from the project root: python main_loop.py")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Minimal embedded StateBus (avoids circular import for standalone run)
# ─────────────────────────────────────────────────────────────────────────────

import hashlib as _hl
import threading as _th
import uuid as _uuid
import json as _json
from datetime import datetime as _dt, timezone as _tz


class _StateBus:
    def __init__(self, initial: dict | None = None) -> None:
        self._state: dict = dict(initial or {})
        self._log: list = []
        self._lock = _th.RLock()

    def read(self, keys=None):
        with self._lock:
            if keys is None:
                return dict(self._state)
            return {k: self._state.get(k) for k in keys}

    def snapshot(self, keys=None):
        with self._lock:
            sl = self.read(keys)
            ts = _dt.now(_tz.utc)
            h = _hl.sha256(_json.dumps(sl, sort_keys=True, default=str).encode()).hexdigest()
            return h, ts, sl

    def apply_delta(self, envelope_id, delta, agent_id, session_id=None):
        with self._lock:
            self._state.update(delta)
            self._log.append({
                "envelope_id": envelope_id, "agent_id": agent_id,
                "delta": delta, "ts": _dt.now(_tz.utc).isoformat()
            })
        return delta

    def delta_log(self):
        with self._lock:
            return list(self._log)

    def get_reputation(self, agent_id):
        return 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Pretty printer
# ─────────────────────────────────────────────────────────────────────────────

ANSI_RESET  = "\033[0m"
ANSI_BOLD   = "\033[1m"
ANSI_GREEN  = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_RED    = "\033[31m"
ANSI_CYAN   = "\033[36m"
ANSI_DIM    = "\033[2m"

_use_color = True


def _c(text: str, *codes: str) -> str:
    if not _use_color:
        return text
    return "".join(codes) + text + ANSI_RESET


def _banner(title: str) -> None:
    width = 62
    print()
    print(_c("━" * width, ANSI_BOLD))
    print(_c(f"  {title}", ANSI_BOLD + ANSI_CYAN))
    print(_c("━" * width, ANSI_BOLD))


def _section(title: str) -> None:
    print(_c(f"\n  ── {title} ──", ANSI_DIM))


# ─────────────────────────────────────────────────────────────────────────────
# DASP event handler
# ─────────────────────────────────────────────────────────────────────────────

def _make_dasp_event_handler(cycle_num: int):
    def on_event(event: str, data: dict) -> None:
        if event == "round_started":
            tier = data.get("tier", "local")
            print(
                _c(f"    [R{data['round']}] {tier.upper()} round started", ANSI_DIM)
            )
        elif event == "round_complete":
            responses = data.get("responses", [])
            accs = data.get("accumulators", {})
            for r in responses:
                sig = r.get("signal", {})
                pol = sig.get("polarity", 0)
                sym = {1: "✅", 0: "➖", -1: "❌"}.get(pol, "?")
                conf = sig.get("confidence", 0)
                print(
                    f"      {sym} [{r.get('agent_id','?')}] "
                    f"conf={conf:.2f}  {r.get('answer','')[:80]}"
                )
            for hid, acc in accs.items():
                s_net = acc["S_net"]
                color = ANSI_GREEN if s_net > 0.5 else (ANSI_YELLOW if s_net > 0 else ANSI_RED)
                print(
                    _c(f"      Signal [{hid}]: "
                       f"S_g={acc['S_g']:.3f}  S_i={acc['S_i']:.3f}  "
                       f"S_net={s_net:.3f}", color)
                )
        elif event == "escalation_triggered":
            print(
                _c(
                    f"\n  ⚡ ESCALATION: {data.get('trigger','?')} gate fired → "
                    f"invoking {data.get('tier_to','?')}",
                    ANSI_YELLOW + ANSI_BOLD,
                )
            )
        elif event == "tier2_invoked":
            print(
                _c(
                    f"  ☁️  Tier-2 [{data.get('model','?')}] arbiter invoked "
                    f"(trigger={data.get('trigger','?')})",
                    ANSI_YELLOW,
                )
            )
        elif event == "consensus_cleared":
            print(
                _c(
                    f"\n  ✅ Consensus cleared: score={data.get('consensus_score',0):.3f}  "
                    f"H={data.get('winning_hypothesis','?')}",
                    ANSI_GREEN + ANSI_BOLD,
                )
            )
        elif event == "session_complete":
            reason = data.get("terminated_reason", "?")
            print(
                _c(f"\n  🏁 Session complete: {reason}", ANSI_CYAN)
            )
    return on_event


# ─────────────────────────────────────────────────────────────────────────────
# Queue consumer
# ─────────────────────────────────────────────────────────────────────────────

def _consume_queue(eq: EnvelopeQueue, stop: threading.Event) -> None:
    """Background consumer — logs each item dequeued from the envelope queue."""
    while not stop.is_set():
        try:
            item = eq.get(timeout=1.0)
        except Exception:
            continue

        eid = item.envelope.get("envelope_id", "?")[:12]
        status = item.validation.status
        verb = item.envelope.get("intended_action", {}).get("verb", "?")
        action_sym = "✅" if item.is_ack() else "❌"

        print(
            _c(
                f"\n  📨 QUEUE → {action_sym} [{status}] "
                f"envelope={eid}... verb={verb}",
                ANSI_GREEN if item.is_ack() else ANSI_RED,
            )
        )

        if not item.is_ack():
            print(
                _c(
                    f"     NACK reason: {item.validation.failure_reason}",
                    ANSI_RED + ANSI_DIM,
                )
            )
        else:
            checks = ", ".join(item.validation.checks_passed)
            print(_c(f"     Checks: {checks}", ANSI_DIM))

        esc = item.envelope.get("expected_state_change", {})
        print(
            _c(
                f"     ESC  keys={esc.get('keys_affected',[])} "
                f"conf={esc.get('confidence',0):.2f}",
                ANSI_DIM,
            )
        )
        pd = esc.get("predicted_delta", {})
        if pd:
            pd_str = json.dumps(pd, default=str)[:120]
            print(_c(f"     Δ    {pd_str}", ANSI_DIM))


# ─────────────────────────────────────────────────────────────────────────────
# Main reasoning loop
# ─────────────────────────────────────────────────────────────────────────────

def run_loop(
    poll_interval_s: float = 5.0,
    debate_interval_s: float = 30.0,
    max_cycles: int = 0,
    queue_file: str = "iep_queue.jsonl",
    use_color: bool = True,
) -> None:
    global _use_color
    _use_color = use_color

    _banner("EFFECTOR ENGINE — Headless Multi-Agent Reasoning Loop")
    print(f"  Poll interval : {poll_interval_s}s")
    print(f"  Debate interval: {debate_interval_s}s")
    print(f"  Max cycles    : {max_cycles if max_cycles else '∞'}")
    print(f"  Queue file    : {queue_file}")

    # ── A1: Telemetry ─────────────────────────────────────────────────────
    bus = _StateBus()
    poller = TelemetryPoller(bus, interval_s=poll_interval_s)
    poller.start()

    # Wait for the first poll to settle
    time.sleep(poll_interval_s + 0.5)

    # ── A2: DASP coordinator (tiered) ─────────────────────────────────────
    coordinator = AsymmetricDASPCoordinator(
        tier1_agents=DEFAULT_TIER1,
        tier2_agent=DEFAULT_TIER2,
        tau_suppression=0.5,
        theta_consensus=0.65,
        epsilon_stall=0.04,
    )

    # ── A3: Validator + Queue ─────────────────────────────────────────────
    validator = IEPValidator(state_bus=bus)
    eq = EnvelopeQueue(persist_path=queue_file)

    # Background queue consumer
    stop_consumer = threading.Event()
    consumer_thread = threading.Thread(
        target=_consume_queue, args=(eq, stop_consumer),
        name="QueueConsumer", daemon=True,
    )
    consumer_thread.start()

    cycle = 0
    try:
        while True:
            cycle += 1
            _banner(f"Cycle {cycle}  —  {datetime.now(timezone.utc).isoformat()}")

            # ── Read current telemetry from StateBus ─────────────────────
            _section("A1 — Telemetry Snapshot")
            state = bus.read()
            snap_hash, snap_ts, _ = bus.snapshot()

            cpu = state.get(KEYS.cpu_percent_total, 0.0)
            ram = state.get(KEYS.ram_percent, 0.0)
            pressure = state.get(KEYS.system_pressure, 0.0)
            window = state.get(KEYS.active_window_title, "(unknown)")
            proc = state.get(KEYS.active_process_name, "(unknown)")
            net_kb = state.get(KEYS.net_recv_kb_s, 0.0)

            pressure_color = (
                ANSI_RED if pressure > 0.7 else
                ANSI_YELLOW if pressure > 0.4 else ANSI_GREEN
            )
            print(f"  CPU   : {cpu:5.1f}%  RAM: {ram:5.1f}%  "
                  + _c(f"Pressure: {pressure:.3f}", pressure_color))
            print(f"  Window: {window}  [{proc}]")
            print(f"  Net↓  : {net_kb:.1f} KB/s   Hash: {snap_hash[:16]}...")

            # ── Build debate task from telemetry ─────────────────────────
            task = (
                f"Analyse current system health. "
                f"CPU={cpu:.1f}%, RAM={ram:.1f}%, "
                f"pressure={pressure:.3f}, "
                f"active_window='{window}', active_process='{proc}', "
                f"net_recv={net_kb:.1f}KB/s. "
                f"Should any corrective action be taken? "
                f"What is the risk level and recommended response?"
            )

            # ── A2: Run tiered DASP debate ────────────────────────────────
            _section("A2 — Asymmetric DASP Debate")
            coordinator._on_event = _make_dasp_event_handler(cycle)

            debate_result = coordinator.run(
                task=task,
                snapshot_hash=snap_hash,
                state_bus=bus,
            )

            print(
                f"\n  Final answer  : {debate_result['final_answer'][:100]}"
            )
            print(
                f"  Consensus     : {debate_result['consensus_score']:.3f}  "
                f"({debate_result['terminated_reason']})"
            )
            if debate_result.get("tier2_injected"):
                print(_c("  ⚡ Tier-2 arbiter was invoked this cycle", ANSI_YELLOW))
            for esc in debate_result.get("escalations", []):
                print(_c(
                    f"  Escalation R{esc['round']}: {esc['trigger']} → {esc['tier_to']}",
                    ANSI_YELLOW
                ))

            # ── A3: Build IEP envelope → validate → enqueue ───────────────
            _section("A3 — IEP Envelope Emission")

            # Re-snapshot at emit time (state may have changed during debate)
            emit_hash, emit_ts, _ = bus.snapshot()

            envelope = IEPBuilder.from_debate_result(
                debate_result=debate_result,
                state_bus_snapshot_hash=emit_hash,
                keys_affected=[
                    KEYS.system_pressure,
                    KEYS.cpu_percent_total,
                    KEYS.ram_percent,
                ],
                predicted_delta={
                    KEYS.system_pressure: debate_result.get("consensus_score", 0.0),
                },
            )

            verdict = validator.validate(envelope)

            print(
                f"  Envelope ID   : {envelope['envelope_id'][:20]}..."
            )
            print(
                f"  Checks passed : {', '.join(verdict.checks_passed) or '(none)'}"
            )
            if verdict.checks_failed:
                print(
                    _c(
                        f"  Checks FAILED : {', '.join(verdict.checks_failed)}",
                        ANSI_RED,
                    )
                )
                print(
                    _c(f"  Reason        : {verdict.failure_reason}", ANSI_RED)
                )

            # Emit formatted envelope JSON to stdout and queue
            envelope_json = json.dumps(envelope, indent=2, default=str)
            print(
                _c(
                    f"\n  ── Emitted IEP Envelope (truncated) ──",
                    ANSI_DIM,
                )
            )
            for line in envelope_json.splitlines()[:18]:
                print(_c(f"  {line}", ANSI_DIM))
            if len(envelope_json.splitlines()) > 18:
                print(_c(f"  ... (+{len(envelope_json.splitlines())-18} lines)", ANSI_DIM))

            eq.put(envelope, verdict)

            # ── Queue stats ───────────────────────────────────────────────
            stats = eq.stats
            print(
                f"\n  Queue         : depth={stats['current_depth']}  "
                f"total={stats['total_enqueued']}  "
                f"ACK={stats['total_acked']}  "
                f"NACK={stats['total_nacked']}"
            )

            # ── Cycle control ─────────────────────────────────────────────
            if max_cycles and cycle >= max_cycles:
                print(f"\n  Max cycles ({max_cycles}) reached — stopping.")
                break

            print(
                _c(
                    f"\n  Sleeping {debate_interval_s}s until next cycle...",
                    ANSI_DIM,
                )
            )
            time.sleep(debate_interval_s)

    except KeyboardInterrupt:
        print("\n\n  Interrupted by user.")
    finally:
        poller.stop()
        stop_consumer.set()
        consumer_thread.join(timeout=3)

        _banner("Session Summary")
        print(f"  Cycles completed : {cycle}")
        print(f"  Telemetry polls  : {poller.poll_count}")
        stats = eq.stats
        print(
            f"  Envelopes        : total={stats['total_enqueued']}  "
            f"ACK={stats['total_acked']}  "
            f"NACK={stats['total_nacked']}"
        )
        replay = eq.replay_from_disk()
        print(f"  Persisted entries: {len(replay)}")
        print(f"  Delta log entries: {len(bus.delta_log())}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Effector headless reasoning loop")
    parser.add_argument("--poll-interval",   type=float, default=5.0,
                        help="Telemetry poll interval (s)")
    parser.add_argument("--debate-interval", type=float, default=30.0,
                        help="Seconds between debate cycles")
    parser.add_argument("--max-cycles",      type=int,   default=0,
                        help="Stop after N cycles (0=run forever)")
    parser.add_argument("--queue-file",      type=str,   default="iep_queue.jsonl",
                        help="JSONL file for envelope persistence")
    parser.add_argument("--no-color",        action="store_true",
                        help="Disable ANSI colour output")
    args = parser.parse_args()

    run_loop(
        poll_interval_s=args.poll_interval,
        debate_interval_s=args.debate_interval,
        max_cycles=args.max_cycles,
        queue_file=args.queue_file,
        use_color=not args.no_color,
    )
