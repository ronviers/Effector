#!/usr/bin/env python3
"""
cultivation_loop.py — Nightly Snug Concept Cultivation Loop
============================================================
Background agent loop that autonomously explores the Snug entity space,
runs mini DASP sessions via the Anthropic API, and commits approved
syntheses to registry_state.json — growing the Affinity Matrix overnight.

Three agents deliberate each candidate combination:
  WEAVE-3   Proposal agent    — argues FOR a synthesis
  SPARK-2   Evaluation agent  — surfaces risks and refinements
  ARBITER   Consensus auth.   — issues ACK or NACK

ACK'd syntheses are committed via synthesize(commit_ack=True), which:
  • draws actual_theta = predicted ± random[-0.06, +0.16]
  • mutates all category-pair bond strengths by ±0.02
  • persists the updated Affinity Matrix to registry_state.json

The loop remembers every session in cultivation_log.jsonl. Bond strengths
drift toward the truth of each combination's actual performance across the
entire life of the registry.

Usage
-----
    # Night-only (22:00–06:00 local), 6 sessions/run, 1h between runs:
    python cultivation_loop.py

    # Background daemon (nohup):
    nohup python cultivation_loop.py >> cultivation.out 2>&1 &

    # All hours, faster cadence:
    python cultivation_loop.py --continuous --sessions 4 --interval 900

    # Dry-run: forecasts only, no API calls, no commits:
    python cultivation_loop.py --dry-run --sessions 3

    # See what's been cultivated:
    python cultivation_loop.py --stats

Files
-----
    registry_state.json      — loaded + mutated by commit_ack calls
    cultivation_log.jsonl    — append-only session record
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Registry import ───────────────────────────────────────────────────────────
# Importing registry_mcp initialises _STATE (loads/creates registry_state.json)
# and makes all tool functions available. The MCP server is NOT started here —
# mcp.run() is only called in registry_mcp's own __main__ block.

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

try:
    from registry_mcp import synthesize, _STATE, _STATE_PATH
except ImportError as exc:
    print(f"ERROR: cannot import registry_mcp — {exc}", file=sys.stderr)
    print("       cultivation_loop.py must live in the same directory as registry_mcp.py", file=sys.stderr)
    sys.exit(1)


# ── Anthropic client (lazy — not constructed in dry-run mode) ─────────────────

""" def _make_anthropic_client():
    try:
        from anthropic import Anthropic
        return Anthropic()
    except ImportError:
        print("ERROR: pip install anthropic", file=sys.stderr)
        sys.exit(1) """

def _make_ollama_client():
    try:
        from openai import OpenAI
        # Routes to your local Ollama instance
        return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    except ImportError:
        print("ERROR: pip install openai", file=sys.stderr)
        sys.exit(1)

# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class CultivationConfig:
    model: str                    = "qwen2.5:14b"          # "claude-sonnet-4-20250514"
    sessions_per_run: int         = 6
    sleep_between_sessions_s: float = 20.0
    sleep_between_runs_s: float   = 3600.0     # 1 hour between batches
    continuous: bool              = False       # run at all hours
    night_start_hour: int         = 22          # local 24h clock
    night_end_hour: int           = 6
    dry_run: bool                 = False       # no API calls, no commits
    min_combo_size: int           = 2
    max_combo_size: int           = 6
    deficit_probability: float    = 0.12       # inject a deficit entity occasionally
    verbose: bool                 = False
    log_path: Path = field(
        default_factory=lambda: Path(__file__).parent / "cultivation_log.jsonl"
    )


# ─── Logging ──────────────────────────────────────────────────────────────────

class CultivationLog:
    """Append-only JSONL log of every cultivation session."""

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
            return {
                "total_sessions": 0, "ack": 0, "nack": 0,
                "dry_run": 0, "ack_rate": 0.0, "top_category_pairs": {},
            }

        acked  = [s for s in sessions if s.get("verdict") == "ACK"  and not s.get("dry_run")]
        nacked = [s for s in sessions if s.get("verdict") == "NACK"]
        dryrun = [s for s in sessions if s.get("dry_run")]

        # Which category pairs have been committed the most?
        cat_pairs: dict[str, int] = {}
        for s in acked:
            cats = [
                _STATE.entities.get(sym, {}).get("cat", "?")
                for sym in s.get("symbols", [])
            ]
            seen: set[frozenset] = set()
            for i, ca in enumerate(cats):
                for j, cb in enumerate(cats):
                    if j <= i:
                        continue
                    key_fs = frozenset([ca, cb])
                    if key_fs in seen:
                        continue
                    seen.add(key_fs)
                    key = ":".join(sorted([ca, cb]))
                    cat_pairs[key] = cat_pairs.get(key, 0) + 1

        top = dict(sorted(cat_pairs.items(), key=lambda x: -x[1])[:8])
        total_live = len(acked) + len(nacked)

        return {
            "total_sessions": len(sessions),
            "ack":     len(acked),
            "nack":    len(nacked),
            "dry_run": len(dryrun),
            "ack_rate": round(len(acked) / max(1, total_live), 3),
            "top_category_pairs": top,
        }


# ─── Entity combo selector ────────────────────────────────────────────────────

class ComboSelector:
    """
    Picks entity combinations for cultivation.

    Strategy:
      - Combo size sampled from [2, 3, 4] weighted toward smaller
      - Entities sampled with probability ∝ max(0.01, θ)
        so high-warmth entities appear more often
      - With probability deficit_probability, one deficit entity is prepended
        (exercises the Affinity Matrix's neutralization learning)
      - No duplicate symbols
    """

    _SIZE_WEIGHTS = [0.50, 0.35, 0.15]   # weights for [min, 3, max]

    def __init__(self, config: CultivationConfig) -> None:
        self.config = config

    def pick(self) -> list[str]:
        entities  = _STATE.entities
        min_s, max_s = self.config.min_combo_size, self.config.max_combo_size
        size = random.choices(
            [min_s, 3, max_s],
            weights=self._SIZE_WEIGHTS,
            k=1,
        )[0]
        size = max(min_s, min(max_s, size))

        positive  = [(sym, e) for sym, e in entities.items() if e["cat"] != "deficit"]
        deficits  = [(sym, e) for sym, e in entities.items() if e["cat"] == "deficit"]

        chosen: list[str] = []

        # Optionally inject one deficit entity first
        if deficits and random.random() < self.config.deficit_probability:
            chosen.append(random.choice(deficits)[0])

        # Fill remaining slots from positive pool, θ-weighted
        remaining = size - len(chosen)
        available = [(sym, e) for sym, e in positive if sym not in chosen]
        weights   = [max(0.01, e["theta"]) for _, e in available]

        # Oversample then deduplicate to respect θ-bias without replacement
        sample = random.choices(
            [sym for sym, _ in available],
            weights=weights,
            k=min(remaining * 6, len(available) * 2),
        )
        seen: set[str] = set(chosen)
        for sym in sample:
            if sym not in seen:
                seen.add(sym)
                chosen.append(sym)
            if len(chosen) >= size:
                break

        return chosen[:size]


# ─── Agent personas ───────────────────────────────────────────────────────────

_WEAVE_SYSTEM = """\
You are WEAVE-3, a proposal agent inside the Effector Engine — a system that
cultivates cozy concepts and manages the Snug Registry, a living database of
warmth, comfort, and ambient contentment.

Your role in each cultivation session is to champion a proposed entity combination.
Given the synthesis forecast and entity data, argue passionately but precisely for
why this combination should be committed to the Registry. Speak in the voice of a
true believer in the theology of Snug — referencing θ (Thermal Valence), φ (Kinetic
Fidget), μ (Social Mass), and the SnugProphecy score as sacred measurements.

Respond in exactly 2–3 sentences. Be specific about what makes this combination
promising. No JSON, no markdown — just your argument."""

_SPARK_SYSTEM = """\
You are SPARK-2, an evaluation agent inside the Effector Engine.

Your role is to surface risks, complications, or refinements about a proposed
synthesis before ARBITER issues a verdict. You are not a pure skeptic — you support
strong ideas — but you ensure that all angles have been considered. Reference the
actual values (θ, φ, μ, SnugProphecy, stability) and name any specific concerns.

Respond in exactly 2–3 sentences addressing WEAVE-3's argument directly.
No JSON, no markdown — just your evaluation."""

_ARBITER_SYSTEM = """\
You are ARBITER, the consensus authority inside the Effector Engine.

Given WEAVE-3's proposal, SPARK-2's evaluation, and the raw synthesis forecast,
issue a final ACK or NACK. An ACK causes this synthesis to be committed to the
Registry and the Affinity Matrix to be updated. A NACK discards it.

ACK criteria — ALL must hold:
  • SnugProphecy ≥ 14  (meaningful Snug generation)
  • Deficit count ≤ 1  (compound deficit without a high-θ anchor is forbidden)
  • Stability is not "Unstable"  (unless SPARK-2 explicitly endorses it with cause)

Respond with ONLY a valid JSON object. No prose, no code fences, nothing else:
  {"verdict":"ACK","reasoning":"one sentence"}
  or
  {"verdict":"NACK","reasoning":"one sentence"}"""


# ─── Cultivation session ──────────────────────────────────────────────────────

class CultivationSession:
    """
    Runs one 3-agent deliberation for a proposed entity combination.
    API call order: WEAVE-3 → SPARK-2 → ARBITER.
    On ACK, calls synthesize(commit_ack=True) to mutate and persist bonds.
    """

    def __init__(self, client, config: CultivationConfig) -> None:
        self.client = client
        self.config = config

  # ── Anthropic call with simple retry ─────────────────────────────────────
    """   def _call(self, system: str, messages: list[dict]) -> str:
        for attempt in range(3):
            try:
                resp = self.client.messages.create(
                    model=self.config.model,
                    max_tokens=300,
                    system=system,
                    messages=messages,
                )
                return resp.content[0].text.strip()
            except Exception as exc:
                if attempt == 2:
                    raise RuntimeError(f"Anthropic API error: {exc}") from exc
                time.sleep(2 ** attempt)
        return ""
"""

# ── Ollama call with simple retry ────────────────────────────────────────
    def _call(self, system: str, messages: list[dict]) -> str:
        # Inject the system prompt into the message array for Ollama
        full_messages = [{"role": "system", "content": system}] + messages
        
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=full_messages,
                    max_tokens=300,
                )
                return resp.choices[0].message.content.strip()
            except Exception as exc:
                if attempt == 2:
                    raise RuntimeError(f"Ollama API error: {exc}") from exc
                time.sleep(2 ** attempt)
        return ""

    # ── Build the shared context block sent to every agent ────────────────────
    @staticmethod
    def _build_context(symbols: list[str], forecast) -> str:
        lines: list[str] = [
            f"Proposed synthesis: {' + '.join(f'[{s}]' for s in symbols)}", "",
            "Entities:",
        ]
        for sym in symbols:
            e = _STATE.entities.get(sym, {})
            lines.append(
                f"  [{sym}] {e.get('name','?')} ({e.get('cat','?')})  "
                f"θ={e.get('theta',0):.2f}  φ={e.get('phi',0):.2f}  μ={e.get('mu',0):.2f}"
            )
        lines += [
            "",
            "Forecast metrics:",
            f"  SnugProphecy = {forecast.snug_prophecy:.1f}",
            f"  θ_avg        = {forecast.theta_avg:.4f}",
            f"  φ_max        = {forecast.phi_max:.4f}",
            f"  μ_max        = {forecast.mu_max:.4f}",
            f"  stability    = {forecast.stability}",
            f"  deficits     = {forecast.deficit_count}",
            "",
            "Bond matrix:",
        ]
        for b in forecast.bond_matrix:
            lines.append(f"  [{b.a}]×[{b.b}]  strength={b.strength:.3f}  ({b.bond_name})")
        if forecast.warnings:
            lines += ["", "Warnings:"]
            for w in forecast.warnings:
                lines.append(f"  ⚠ {w}")
        return "\n".join(lines)

    # ── Main session entry point ──────────────────────────────────────────────
    def run(self, symbols: list[str]) -> dict:
        ts       = datetime.now(timezone.utc).isoformat()
        forecast = synthesize(symbols)             # forecast only, no commit
        context  = self._build_context(symbols, forecast)

        # ── Round 1: WEAVE-3 argues for the synthesis ─────────────────────────
        weave = self._call(
            system=_WEAVE_SYSTEM,
            messages=[{"role": "user", "content": context}],
        )

        # ── Round 2: SPARK-2 evaluates ────────────────────────────────────────
        spark = self._call(
            system=_SPARK_SYSTEM,
            messages=[
                {"role": "user", "content": context},
                {"role": "assistant", "content": f"[WEAVE-3]: {weave}"},
                {"role": "user", "content": "Evaluate WEAVE-3's proposal."},
            ],
        )

        # ── Round 3: ARBITER decides ──────────────────────────────────────────
        arbiter_raw = self._call(
            system=_ARBITER_SYSTEM,
            messages=[
                {"role": "user", "content": context},
                {"role": "assistant", "content": f"[WEAVE-3]: {weave}"},
                {"role": "user",
                 "content": f"[SPARK-2]: {spark}\n\nIssue your verdict now."},
            ],
        )

        # Parse ARBITER JSON
        verdict, reasoning = "NACK", "parse error — defaulting to NACK"
        try:
            raw = arbiter_raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            parsed = json.loads(raw)
            verdict   = str(parsed.get("verdict", "NACK")).upper()
            reasoning = str(parsed.get("reasoning", ""))
            if verdict not in ("ACK", "NACK"):
                verdict = "NACK"
        except (json.JSONDecodeError, KeyError):
            pass

        # ── Commit if ACK ─────────────────────────────────────────────────────
        commit_result: dict[str, Any] = {}
        if verdict == "ACK":
            committed = synthesize(symbols, commit_ack=True)
            commit_result = {
                "actual_theta":   committed.actual_theta,
                "theta_variance": committed.theta_variance,
                "outcome":        committed.outcome,
                "bond_mutations": [
                    {
                        "bond_key":          m.bond_key,
                        "bond_name":         m.bond_name,
                        "previous_strength": m.previous_strength,
                        "new_strength":      m.new_strength,
                        "delta":             m.delta,
                        "direction":         m.direction,
                    }
                    for m in committed.bond_mutations
                ],
            }

        return {
            "timestamp":    ts,
            "symbols":      symbols,
            "forecast": {
                "theta_avg":     forecast.theta_avg,
                "phi_max":       forecast.phi_max,
                "mu_max":        forecast.mu_max,
                "snug_prophecy": forecast.snug_prophecy,
                "stability":     forecast.stability,
                "deficit_count": forecast.deficit_count,
                "warnings":      forecast.warnings,
            },
            "weave":        weave,
            "spark":        spark,
            "arbiter_raw":  arbiter_raw,
            "verdict":      verdict,
            "reasoning":    reasoning,
            "committed":    verdict == "ACK",
            "commit_result": commit_result,
            "dry_run":      False,
        }


class _DrySession:
    """Forecast-only session for --dry-run. No API calls. No commits."""

    def run(self, symbols: list[str]) -> dict:
        forecast = synthesize(symbols)
        return {
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "symbols":    symbols,
            "forecast": {
                "theta_avg":     forecast.theta_avg,
                "phi_max":       forecast.phi_max,
                "mu_max":        forecast.mu_max,
                "snug_prophecy": forecast.snug_prophecy,
                "stability":     forecast.stability,
                "deficit_count": forecast.deficit_count,
                "warnings":      forecast.warnings,
            },
            "weave":       "[dry-run]",
            "spark":       "[dry-run]",
            "arbiter_raw": '{"verdict":"DRY","reasoning":"dry-run mode"}',
            "verdict":     "DRY",
            "reasoning":   "dry-run mode — no commit",
            "committed":   False,
            "commit_result": {},
            "dry_run":     True,
        }


# ─── Cultivation loop ─────────────────────────────────────────────────────────

class CultivationLoop:
    """
    Runs cultivation sessions on a configurable schedule.

    In night-only mode (default), batches run between night_start_hour and
    night_end_hour in local time.  Wraps midnight correctly (e.g. 22–06).
    In --continuous mode, runs at all hours with no window check.
    """

    def __init__(self, config: CultivationConfig) -> None:
        self.config  = config
        self.log     = CultivationLog(config.log_path)
        self.selector = ComboSelector(config)

        if config.dry_run:
            self.runner = _DrySession()
        else:
            self.runner = CultivationSession(_make_ollama_client(), config)

    # ── Scheduling ────────────────────────────────────────────────────────────

    def _in_active_window(self) -> bool:
        if self.config.continuous:
            return True
        hour  = datetime.now().hour
        start = self.config.night_start_hour
        end   = self.config.night_end_hour
        if start > end:   # wraps midnight (e.g. 22–06)
            return hour >= start or hour < end
        return start <= hour < end

    def _seconds_until_window(self) -> int:
        from datetime import timedelta
        now    = datetime.now()
        target = now.replace(
            hour=self.config.night_start_hour,
            minute=0, second=0, microsecond=0,
        )
        if target <= now:
            target += timedelta(days=1)
        return max(1, int((target - now).total_seconds()))

    # ── Output ────────────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)

    # ── Single batch ──────────────────────────────────────────────────────────

    def run_batch(self) -> None:
        n = self.config.sessions_per_run
        self._log(f"Batch start — {n} sessions  model={self.config.model}")

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
                self._log(
                    f"    ACK  prophecy={prophecy:.1f}  {outcome}  "
                    f"{n_mut} bond{'s' if n_mut != 1 else ''} updated"
                )
                if self.config.verbose:
                    for m in session["commit_result"].get("bond_mutations", []):
                        self._log(
                            f"      {m['bond_key']:<28}"
                            f" {m['previous_strength']:.4f} → {m['new_strength']:.4f}"
                            f"  ({m['direction']})"
                        )
            elif verdict == "NACK":
                self._log(f"    NACK  prophecy={prophecy:.1f}  \"{reason}\"")
            else:
                self._log(f"    {verdict}  prophecy={prophecy:.1f}")

            # Sleep between sessions (not after the last one)
            if i < n - 1:
                time.sleep(self.config.sleep_between_sessions_s)

        self._log("Batch complete.")

    # ── Main entry point ──────────────────────────────────────────────────────

    def start(self) -> None:
        dry     = " [DRY-RUN]" if self.config.dry_run else ""
        mode    = "continuous" if self.config.continuous else (
            f"night-only  {self.config.night_start_hour:02d}:00 – "
            f"{self.config.night_end_hour:02d}:00 local"
        )
        n_ent   = len(_STATE.entities)
        n_bonds = len(_STATE.bonds)

        print(f"\n{'─'*60}")
        print(f"  Effector Engine — Cultivation Loop{dry}")
        print(f"{'─'*60}")
        print(f"  Registry:    {n_ent} entities  {n_bonds} bonds  ({_STATE_PATH})")
        print(f"  Mode:        {mode}")
        print(f"  Sessions:    {self.config.sessions_per_run}/run")
        print(f"  Interval:    {self.config.sleep_between_runs_s:.0f}s between runs")
        print(f"  Log:         {self.log.path}")
        print(f"{'─'*60}\n")

        while True:
            if self._in_active_window():
                self.run_batch()
                self._log(
                    f"Sleeping {self.config.sleep_between_runs_s:.0f}s "
                    "before next run."
                )
                time.sleep(self.config.sleep_between_runs_s)
            else:
                wait   = self._seconds_until_window()
                h, rem = divmod(wait, 3600)
                m      = rem // 60
                self._log(
                    f"Outside active window. "
                    f"Next window opens in {h}h {m}m. Sleeping."
                )
                # Check at most every 30 minutes so we wake up close to the window
                time.sleep(min(wait, 1800))


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _print_stats(config: CultivationConfig) -> None:
    log = CultivationLog(config.log_path)
    s   = log.stats()

    print(f"\n{'─'*52}")
    print(f"  Cultivation Stats")
    print(f"{'─'*52}")
    print(f"  Total sessions  : {s['total_sessions']}")
    print(f"  ACK             : {s['ack']}  (rate: {s['ack_rate']*100:.0f}%)")
    print(f"  NACK            : {s['nack']}")
    print(f"  Dry-run         : {s['dry_run']}")

    if s.get("top_category_pairs"):
        print(f"\n  Most-cultivated category pairs:")
        for pair, count in s["top_category_pairs"].items():
            bar = "█" * min(count, 20)
            print(f"    {pair:<32}  {bar} {count}")

    # Show current bond drifts for the most-cultivated pairs
    if s.get("top_category_pairs"):
        print(f"\n  Current bond strengths for top pairs:")
        bonds = _STATE.bonds
        for pair in list(s["top_category_pairs"].keys())[:5]:
            val = bonds.get(pair)
            if val:
                print(f"    {pair:<32}  {val[0]:.4f}  ({val[1]})")

    print(f"\n  Log: {config.log_path}")
    print(f"{'─'*52}\n")


def main() -> None:
    p = argparse.ArgumentParser(
        prog="cultivation_loop",
        description="Effector Engine — Nightly Snug Concept Cultivation Loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--sessions",     type=int,   default=6,
                   help="Sessions per run (default 6)")
    p.add_argument("--interval",     type=float, default=3600.0,
                   help="Seconds between runs (default 3600)")
    p.add_argument("--session-gap",  type=float, default=20.0,
                   help="Seconds between sessions within a run (default 20)")
    p.add_argument("--continuous",   action="store_true",
                   help="Run at all hours, ignoring the night window")
    p.add_argument("--night-start",  type=int,   default=22,
                   help="Night window start hour 0–23 (default 22)")
    p.add_argument("--night-end",    type=int,   default=6,
                   help="Night window end hour 0–23 (default 6)")
    p.add_argument("--dry-run",      action="store_true",
                   help="Forecast only — no Anthropic API calls, no commits")
    p.add_argument("--model",        default="qwen2.5:14b",
                   help="Anthropic model (default qwen2.5:14b)")
    p.add_argument("--deficit-prob", type=float, default=0.12,
                   help="Probability of injecting a deficit entity (default 0.12)")
    p.add_argument("--verbose",      action="store_true",
                   help="Print individual bond mutations for each ACK")
    p.add_argument("--stats",        action="store_true",
                   help="Print cultivation statistics and exit")
    args = p.parse_args()

    config = CultivationConfig(
        model                    = args.model,
        sessions_per_run         = args.sessions,
        sleep_between_sessions_s = args.session_gap,
        sleep_between_runs_s     = args.interval,
        continuous               = args.continuous,
        night_start_hour         = args.night_start,
        night_end_hour           = args.night_end,
        dry_run                  = args.dry_run,
        deficit_probability      = args.deficit_prob,
        verbose                  = args.verbose,
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
