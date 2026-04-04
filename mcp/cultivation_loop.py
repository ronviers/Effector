#!/usr/bin/env python3
"""
cultivation_loop.py — Nightly Snug Concept Cultivation Loop
============================================================
Background agent loop that autonomously explores the Snug entity space,
runs mini DASP sessions via the local Reflex Orchestrator, and commits 
approved syntheses to registry_state.json.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import subprocess
from rich.console import Console

console = Console()

def verify_models(allow_auto_pull: bool = True):
    required = ["mistral:7b", "qwen2.5:14b", "qwen2.5-coder:32b", "nomic-embed-text"]
    try:
        resp = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
        installed = [m["name"] for m in resp.json().get("models", [])]
        
        for req in required:
            if req not in installed:
                if allow_auto_pull:
                    with console.status(f"[bold yellow]Pulling missing model: {req}..."):
                        subprocess.run(["ollama", "pull", req], check=True)
                else:
                    console.print(f"[bold red]Missing required model: {req}. Run 'effector models pull {req}'.")
                    sys.exit(1)
    except requests.exceptions.ConnectionError:
        console.print("[bold red]Ollama is not running. Please start Ollama at 127.0.0.1:11434")
        sys.exit(1)

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

try:
    from registry_mcp import synthesize, _STATE, _STATE_PATH
except ImportError as exc:
    print(f"ERROR: cannot import registry_mcp — {exc}", file=sys.stderr)
    sys.exit(1)

# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class CultivationConfig:
    model: str                    = "qwen2.5:14b"
    sessions_per_run: int         = 6
    sleep_between_sessions_s: float = 20.0
    sleep_between_runs_s: float   = 3600.0
    continuous: bool              = False
    night_start_hour: int         = 22
    night_end_hour: int           = 6
    dry_run: bool                 = False
    min_combo_size: int           = 2
    max_combo_size: int           = 6
    deficit_probability: float    = 0.12
    verbose: bool                 = False
    log_path: Path = field(
        default_factory=lambda: Path(__file__).parent / "cultivation_log.jsonl"
    )

# ─── Logging ──────────────────────────────────────────────────────────────────

class CultivationLog:
    def __init__(self, path: Path) -> None:
        self.path = path

    def record(self, session: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(session, default=str) + "\n")

    def load_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows: list[dict] = []
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return rows

    def stats(self) -> dict:
        sessions = self.load_all()
        if not sessions:
            return {"total_sessions": 0, "ack": 0, "nack": 0, "dry_run": 0, "ack_rate": 0.0, "top_category_pairs": {}}
        
        acked  = [s for s in sessions if s.get("verdict") == "ACK"  and not s.get("dry_run")]
        nacked = [s for s in sessions if s.get("verdict") == "NACK"]
        dryrun = [s for s in sessions if s.get("dry_run")]
        
        cat_pairs: dict[str, int] = {}
        for s in acked:
            cats = [_STATE.entities.get(sym, {}).get("cat", "?") for sym in s.get("symbols", [])]
            seen: set[frozenset] = set()
            for i, ca in enumerate(cats):
                for j, cb in enumerate(cats):
                    if j <= i: continue
                    key_fs = frozenset([ca, cb])
                    if key_fs in seen: continue
                    seen.add(key_fs)
                    key = ":".join(sorted([ca, cb]))
                    cat_pairs[key] = cat_pairs.get(key, 0) + 1
                    
        top = dict(sorted(cat_pairs.items(), key=lambda x: -x[1])[:8])
        total_live = len(acked) + len(nacked)
        return {
            "total_sessions": len(sessions), "ack": len(acked), "nack": len(nacked),
            "dry_run": len(dryrun), "ack_rate": round(len(acked) / max(1, total_live), 3),
            "top_category_pairs": top,
        }

# ─── Entity combo selector ────────────────────────────────────────────────────

class ComboSelector:
    _SIZE_WEIGHTS = [0.50, 0.35, 0.15]
    def __init__(self, config: CultivationConfig) -> None:
        self.config = config

    def pick(self) -> list[str]:
        entities  = _STATE.entities
        min_s, max_s = self.config.min_combo_size, self.config.max_combo_size
        size = random.choices([min_s, 3, max_s], weights=self._SIZE_WEIGHTS, k=1)[0]
        size = max(min_s, min(max_s, size))

        positive  = [(sym, e) for sym, e in entities.items() if e["cat"] != "deficit"]
        deficits  = [(sym, e) for sym, e in entities.items() if e["cat"] == "deficit"]

        chosen: list[str] = []
        if deficits and random.random() < self.config.deficit_probability:
            chosen.append(random.choice(deficits)[0])

        remaining = size - len(chosen)
        available = [(sym, e) for sym, e in positive if sym not in chosen]
        weights   = [max(0.01, e["theta"]) for _, e in available]

        sample = random.choices([sym for sym, _ in available], weights=weights, k=min(remaining * 6, len(available) * 2))
        seen: set[str] = set(chosen)
        for sym in sample:
            if sym not in seen:
                seen.add(sym)
                chosen.append(sym)
            if len(chosen) >= size: break
        return chosen[:size]

# ─── Cultivation session ──────────────────────────────────────────────────────

from effector.adapters.asymmetric_dasp import AsymmetricDASPCoordinator, TierConfig
from effector.bus import StateBus
from effector.main_loop_reflex import ReflexOrchestrator
from effector.queue.iep_queue import IEPBuilder, IEPValidator
from effector.rat_store import LocalRATStore
import hashlib

class CultivationSession:
    def __init__(self, config):
        self.config = config
        self.rat_store = LocalRATStore()
        
        tier1 = [
            TierConfig(name="WEAVE-3", model="qwen2.5:14b", characterizer_model="qwen2.5-coder:32b"),
            TierConfig(name="SPARK-2", model="mistral:7b", characterizer_model="qwen2.5-coder:32b")
        ]
        tier2 = TierConfig(name="ARBITER", model="qwen2.5:14b", characterizer_model="qwen2.5-coder:32b")
        
        self.coordinator = AsymmetricDASPCoordinator(
            tier1_agents=tier1, tier2_agent=tier2,
            tau_suppression=0.5, theta_consensus=0.75,
            vectorized_bus=True
        )

    def run(self, symbols: list[str]) -> dict:
        forecast = synthesize(symbols, commit_ack=False)
        
        bus = StateBus(initial_state={
            "synthesis.symbols": symbols,
            "synthesis.prophecy": forecast.snug_prophecy,
            "synthesis.deficits": forecast.deficit_count
        })
        snap_hash, _, _ = bus.snapshot()

        orchestrator = ReflexOrchestrator(
            state_bus=bus, rat_store=self.rat_store, 
            dasp_run_fn=self.coordinator.run
        )
        
        task_prompt = f"Synthesize: {symbols}. Forecast: {forecast.model_dump_json()}"
        result = orchestrator.handle(task=task_prompt, snapshot_hash=snap_hash)

        # Map Reflex bypassed/dasp output to verdict
        verdict = "ACK" if result.get("consensus_score", 0) >= 0.75 or result.get("status") == "EXECUTED" else "NACK"

        envelope = IEPBuilder.from_debate_result(
            debate_result=result, 
            state_bus_snapshot_hash=snap_hash,
            keys_affected=["registry.bonds"]
        )
        
        validator = IEPValidator(state_bus=bus)
        val_result = validator.validate(envelope)

        commit_result = {}
        if val_result.status == "ACK":
            committed = synthesize(symbols, commit_ack=True)
            commit_result = {
                "outcome": committed.outcome,
                "bond_mutations": [
                    {"bond_key": m.bond_key, "direction": m.direction, "new_strength": m.new_strength}
                    for m in committed.bond_mutations
                ]
            }

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbols": symbols,
            "forecast": {"snug_prophecy": forecast.snug_prophecy},
            "verdict": "ACK" if val_result.status == "ACK" else "NACK",
            "reasoning": val_result.failure_reason if val_result.status != "ACK" else result.get("final_answer", "Reflex Cache Hit"),
            "commit_result": commit_result,
            "dry_run": False
        }

class _DrySession:
    def run(self, symbols: list[str]) -> dict:
        forecast = synthesize(symbols)
        return {
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "symbols":    symbols,
            "forecast": {"snug_prophecy": forecast.snug_prophecy},
            "verdict":    "DRY",
            "reasoning":  "dry-run mode — no commit",
            "commit_result": {},
            "dry_run":    True,
        }

# ─── Cultivation loop ─────────────────────────────────────────────────────────

class CultivationLoop:
    def __init__(self, config: CultivationConfig) -> None:
        self.config  = config
        self.log     = CultivationLog(config.log_path)
        self.selector = ComboSelector(config)
        if config.dry_run:
            self.runner = _DrySession()
        else:
            self.runner = CultivationSession(config)

    def _in_active_window(self) -> bool:
        if self.config.continuous: return True
        hour  = datetime.now().hour
        start, end = self.config.night_start_hour, self.config.night_end_hour
        if start > end: return hour >= start or hour < end
        return start <= hour < end

    def _seconds_until_window(self) -> int:
        from datetime import timedelta
        now    = datetime.now()
        target = now.replace(hour=self.config.night_start_hour, minute=0, second=0, microsecond=0)
        if target <= now: target += timedelta(days=1)
        return max(1, int((target - now).total_seconds()))

    def _log(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)

    def run_batch(self) -> None:
        n = self.config.sessions_per_run
        self._log(f"Batch start — {n} sessions")
        for i in range(n):
            symbols = self.selector.pick()
            label   = " + ".join(symbols)
            self._log(f"  [{i+1}/{n}] {label} ...")
            try:
                session = self.runner.run(symbols)
            except Exception as exc:
                self._log(f"    ERROR: {exc}")
                continue
            self.log.record(session)
            
            verdict  = session["verdict"]
            prophecy = session["forecast"]["snug_prophecy"]
            reason   = session.get("reasoning", "")
            n_mut    = len(session.get("commit_result", {}).get("bond_mutations", []))

            if verdict == "ACK":
                outcome = session["commit_result"].get("outcome", "?")
                self._log(f"    ACK  prophecy={prophecy:.1f}  {outcome}  {n_mut} bonds updated")
            elif verdict == "NACK":
                self._log(f"    NACK  prophecy={prophecy:.1f}  \"{reason[:80]}...\"")
            else:
                self._log(f"    {verdict}  prophecy={prophecy:.1f}")

            if i < n - 1: time.sleep(self.config.sleep_between_sessions_s)
        self._log("Batch complete.")

    def start(self) -> None:
        dry     = " [DRY-RUN]" if self.config.dry_run else ""
        mode    = "continuous" if self.config.continuous else f"night-only  {self.config.night_start_hour:02d}:00 – {self.config.night_end_hour:02d}:00 local"
        print(f"\n{'─'*60}\n  Effector Engine — Cultivation Loop{dry}\n{'─'*60}")
        print(f"  Mode:        {mode}")
        print(f"  Sessions:    {self.config.sessions_per_run}/run\n{'─'*60}\n")
        
        while True:
            if self._in_active_window():
                self.run_batch()
                self._log(f"Sleeping {self.config.sleep_between_runs_s:.0f}s before next run.")
                time.sleep(self.config.sleep_between_runs_s)
            else:
                wait   = self._seconds_until_window()
                h, rem = divmod(wait, 3600)
                self._log(f"Outside active window. Next window opens in {h}h {rem // 60}m. Sleeping.")
                time.sleep(min(wait, 1800))

def _print_stats(config: CultivationConfig) -> None:
    s = CultivationLog(config.log_path).stats()
    print(f"\n{'─'*52}\n  Cultivation Stats\n{'─'*52}")
    print(f"  Total sessions  : {s['total_sessions']}\n  ACK             : {s['ack']}  (rate: {s['ack_rate']*100:.0f}%)\n  NACK            : {s['nack']}")
    print(f"{'─'*52}\n")

def main() -> None:
    # ─── ADDED THE CALL YOU MISSED ───
    verify_models(allow_auto_pull=True)

    p = argparse.ArgumentParser(prog="cultivation_loop")
    p.add_argument("--sessions", type=int, default=6)
    p.add_argument("--interval", type=float, default=3600.0)
    p.add_argument("--session-gap", type=float, default=20.0)
    p.add_argument("--continuous", action="store_true")
    p.add_argument("--night-start", type=int, default=22)
    p.add_argument("--night-end", type=int, default=6)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--model", default="qwen2.5:14b")
    p.add_argument("--deficit-prob", type=float, default=0.12)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--stats", action="store_true")
    args = p.parse_args()

    config = CultivationConfig(
        model=args.model, sessions_per_run=args.sessions, sleep_between_sessions_s=args.session_gap,
        sleep_between_runs_s=args.interval, continuous=args.continuous, night_start_hour=args.night_start,
        night_end_hour=args.night_end, dry_run=args.dry_run, deficit_probability=args.deficit_prob, verbose=args.verbose,
    )

    if args.stats:
        _print_stats(config)
        return

    try:
        CultivationLoop(config).start()
    except KeyboardInterrupt:
        print("\nCultivation loop interrupted.")

if __name__ == "__main__":
    main()