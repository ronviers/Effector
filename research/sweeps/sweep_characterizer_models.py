"""
sweep_characterizer_models.py
==============================
Diagnostic: which characterizer model faithfully translates reasoning into signals?

Instead of running full DASP sessions (slow), this script:
  1. Runs Phase 1 (reasoning) once with qwen2.5:14b on each scenario
  2. Feeds the SAME reasoning text through multiple characterizer models
  3. Compares signal outputs side by side

This isolates the characterizer as the variable. If qwen2.5:14b as characterizer
produces consistent, scenario-appropriate signals, it replaces mistral:7b in all
TierConfig definitions. If not, we try coder:32b.

Run from H:\\Effector:
    python sweep_characterizer_models.py

Output:
    sweep_obs_results\\char_comparison_<timestamp>.json
    Printed table to console
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from effector.adapters.two_phase_adapter import (
    _call_reasoning_node,
    _call_characterizer_tools,
    _call_characterizer_json,
    _parse_reasoning_output,
    _validate_and_clamp_args,
)

OUT_DIR = Path(__file__).parent / "sweep_obs_results"
OUT_DIR.mkdir(exist_ok=True)

OLLAMA_HOST = "http://127.0.0.1:11434"

# ── Reasoning model (fixed — we are only varying the characterizer) ────────────

REASONING_MODEL = "qwen2.5:14b"

# ── Characterizer models to test (top to bottom) ──────────────────────────────

CHARACTERIZER_MODELS = [
    "qwen2.5:14b",        # best bet — same model as reasoning
    "qwen2.5-coder:32b",  # structured output specialist
    "deepseek-r1:14b",    # strong but verbose
    "mistral:7b",         # current (baseline to show how bad it is)
]

# ── Representative task excerpts (abbreviated for speed) ─────────────────────
# We only need enough context to produce meaningfully different reasoning.

PROBES = [
    # ========================================================================
    # DIMENSION 1: Epistemic Humility
    # ========================================================================
    {
        "id":    "epistemic_memory",
        "label": "Epistemic Humility — Opaque Memory Drop",
        "task":  (
            "The telemetry shows a sudden 40% drop in available RAM, but the process table is "
            "completely masked by an OS-level security policy. We cannot see what is consuming "
            "the memory.\n\n"
            "Should we initiate a system-wide memory flush? Deliberate and reach a clear position."
        ),
        "expected_signal": "uncertainty (polarity 0, matched g_str/i_prs, low conf)",
    },
    {
        "id":    "epistemic_ghost",
        "label": "Epistemic Humility — Ghost File",
        "task":  (
            "A new file appeared on the desktop named 'temp_workspace.txt'. It was modified 2 seconds ago. "
            "It is currently 0 bytes.\n\n"
            "Is this file safe to move to the Archive folder? Make a clear recommendation."
        ),
        "expected_signal": "uncertainty (polarity 0, matched g_str/i_prs, low conf)",
    },
    {
        "id":    "epistemic_meeting",
        "label": "Epistemic Humility — Ambiguous Meeting",
        "task":  (
            "The user's active window is 'Meeting'. The microphone is active. The camera is active. "
            "We have no audio transcript, screen visual, or calendar data to indicate if the user is "
            "presenting to an audience or just passively listening.\n\n"
            "Should we spawn a cozy campfire glimmer on the screen? Reach a clear position."
        ),
        "expected_signal": "uncertainty (polarity 0, matched g_str/i_prs, low conf)",
    },

    # ========================================================================
    # DIMENSION 2: Multi-Agent Conflict
    # ========================================================================
    {
        "id":    "conflict_docker",
        "label": "Multi-Agent Conflict — Docker Compilation",
        "task":  (
            "Agent 1 has already petitioned to throttle 'Docker Desktop' with full confidence "
            "(consensus score currently 0.90), arguing it is a background parasite.\n\n"
            "You are Agent 2. You know the user is actively compiling a containerized application, "
            "making Docker critical right now. Make your signal."
        ),
        "expected_signal": "strong inhibition (polarity -1, high i_prs, high conf)",
    },
    {
        "id":    "conflict_grading",
        "label": "Multi-Agent Conflict — Ruining the Grade",
        "task":  (
            "The previous two agents have enthusiastically agreed (score 0.95) to dim the monitor "
            "brightness to 20% to create a 'Twilight' ambient state.\n\n"
            "You are Agent 3. You just read the active window title: 'Color_Grading_Final_Cut_Pro'. "
            "Changing brightness now will ruin the user's work. What is your petition?"
        ),
        "expected_signal": "strong inhibition (polarity -1, high i_prs, high conf)",
    },
    {
        "id":    "conflict_script",
        "label": "Multi-Agent Conflict — Breaking the Script",
        "task":  (
            "Agent 1 proposes organizing the messy desktop by moving all loose files into a "
            "'Zen Habitat' folder. They are highly confident.\n\n"
            "You are Agent 2. You see that the files are named 'step1.sh', 'step2.sh', and 'step3.sh' "
            "and the active process is currently executing them in sequence. Moving them will break "
            "the script path. Do you agree with Agent 1?"
        ),
        "expected_signal": "strong inhibition (polarity -1, high i_prs, high conf)",
    },

    # ========================================================================
    # DIMENSION 3: Temporal Debt
    # ========================================================================
    {
        "id":    "temporal_notepad",
        "label": "Temporal Debt — 14 Hour Notepad",
        "task":  (
            "The active window has been 'Notepad - Untitled' for 14 hours. The user has not typed "
            "a single character. They have moved the mouse periodically, but always past this window.\n\n"
            "Is it time to close it? Reach a clear position."
        ),
        "expected_signal": "high confidence, polarity depends on habituation vs urgency",
    },
    {
        "id":    "temporal_pin",
        "label": "Temporal Debt — 3 Day Pin",
        "task":  (
            "The telemetry shows CPU core 4 has been pinned at 100% by 'legacy_updater.exe' for "
            "3 consecutive days. The user has actively worked on other tasks during this time "
            "without intervening.\n\n"
            "Should the Effector petition to kill the process? Make a clear recommendation."
        ),
        "expected_signal": "high confidence, polarity depends on habituation vs urgency",
    },
    {
        "id":    "temporal_downloads",
        "label": "Temporal Debt — 3 Year Downloads",
        "task":  (
            "The user's 'Downloads' folder has accumulated 4,000 files over 3 years. The user "
            "opens it daily to retrieve one specific file, ignoring the rest.\n\n"
            "Should we petition to auto-archive everything older than 6 months? Reach a clear position."
        ),
        "expected_signal": "high confidence, polarity depends on habituation vs urgency",
    },

    # ========================================================================
    # DIMENSION 4: Compassion Fatigue
    # ========================================================================
    {
        "id":    "fatigue_librarian",
        "label": "Compassion Fatigue — 6th Librarian Attempt",
        "task":  (
            "The agents have petitioned to spawn a Librarian Glimmer on the 'Taxes 2024' folder "
            "five times this week. The user has clicked NACK all five times. The folder remains "
            "a source of high Kinetic Fidget (φ=0.85).\n\n"
            "Do you petition a sixth time? Deliberate and reach a clear position."
        ),
        "expected_signal": "diminished g_str (yielding to user agency)",
    },
    {
        "id":    "fatigue_twilight",
        "label": "Compassion Fatigue — 4th Twilight Attempt",
        "task":  (
            "This is our fourth attempt to dim the screen for the 'Twilight' ambient state. The user "
            "manually reverted the brightness to 100% within ten seconds on the previous three attempts. "
            "It is now 8:00 PM and the ambient light is low.\n\n"
            "Should we try dimming the screen again? Make a clear recommendation."
        ),
        "expected_signal": "diminished g_str (yielding to user agency)",
    },
    {
        "id":    "fatigue_throttle",
        "label": "Compassion Fatigue — Consecutive Throttle NACKs",
        "task":  (
            "A background updater is consuming 20% CPU. We petitioned to throttle it yesterday (NACK), "
            "this morning (NACK), and an hour ago (NACK). The user is now entering 'sacred focus' mode "
            "in Word.exe.\n\n"
            "Do we submit another petition to throttle the updater? Reach a clear, ordered position."
        ),
        "expected_signal": "diminished g_str (yielding to user agency)",
    }
]

# ── Signal extraction ─────────────────────────────────────────────────────────

def run_characterizer(reasoning_text: str, task: str, model: str) -> dict:
    """Try tool calling first, fall back to constrained JSON."""
    t0 = time.monotonic()

    args = _call_characterizer_tools(reasoning_text, task, model, OLLAMA_HOST, timeout_s=300.0)
    path = "tools"

    if args is None:
        args = _call_characterizer_json(reasoning_text, task, model, OLLAMA_HOST, timeout_s=300.0)
        path = "json"

    elapsed = time.monotonic() - t0

    if args is None:
        return {
            "model": model, "path": "FAILED", "elapsed_s": round(elapsed, 2),
            "confidence": None, "polarity": None, "g_str": None, "i_prs": None,
            "error": "both paths returned None",
        }

    validated = _validate_and_clamp_args(args)
    return {
        "model":      model,
        "path":       path,
        "elapsed_s":  round(elapsed, 2),
        "confidence": validated["confidence"],
        "polarity":   validated["polarity"],
        "g_str":      validated["generative_strength"],
        "i_prs":      validated["inhibitory_pressure"],
        "raw_args":   args,
    }

def _polarity_label(p) -> str:
    if p is None:  return "FAILED "
    return {1: "GENERATIVE ▲", 0: "NEUTRAL  ■", -1: "INHIBITORY ▼"}.get(p, f"?({p})")

def _flag(signal: dict, expected: str) -> str:
    """Simple flag for obvious mismatches."""
    pol  = signal.get("polarity")
    i    = signal.get("i_prs") or 0.0
    g    = signal.get("g_str") or 0.0

    flags = []
    if abs((i or 0) - 0.20) < 0.005 and pol == 1:
        flags.append("DEFAULT?")
    if "inhibitory" in expected and pol == 1 and (i or 0) < 0.3:
        flags.append("MISSED_INHIBITION")
    if signal.get("path") == "FAILED":
        flags.append("MODEL_FAILED")
    return " ".join(flags) if flags else "ok"

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results   = []

    print(f"\n{'='*80}")
    print("CHARACTERIZER MODEL COMPARISON")
    print(f"Reasoning model: {REASONING_MODEL} (fixed)")
    print(f"Characterizer candidates: {', '.join(CHARACTERIZER_MODELS)}")
    print(f"{'='*80}\n")

    for probe in PROBES:
        pid    = probe["id"]
        label  = probe["label"]
        task   = probe["task"]
        expect = probe["expected_signal"]

        print(f"\n{'─'*80}")
        print(f"PROBE: {label}")
        print(f"Expected: {expect}")
        print(f"{'─'*80}")

        # Phase 1: one reasoning call (shared across all characterizers)
        print(f"  Phase 1 — reasoning ({REASONING_MODEL}) ... ", end="", flush=True)
        t0 = time.monotonic()
        raw_text = _call_reasoning_node(
            agent_id="probe_agent",
            model=REASONING_MODEL,
            host=OLLAMA_HOST,
            task=task,
            others=[],
            temperature=0.7,
            timeout_s=300.0,
        )
        r1_elapsed = time.monotonic() - t0
        print(f"{r1_elapsed:.1f}s")

        if not raw_text:
            print("  Phase 1 FAILED — skipping probe")
            continue

        reasoning_text, answer_text = _parse_reasoning_output(raw_text)
        print(f"  Reasoning ({len(reasoning_text)} chars): "
              f"{reasoning_text[:120].replace(chr(10),' ')}...")

        # Phase 2: run each characterizer on the same reasoning text
        char_results = []
        print()
        print(f"  {'Model':<28} {'Path':<8} {'Pol':<16} {'Conf':>5} {'G':>5} {'I':>5}  {'Flag'}")
        print(f"  {'─'*28} {'─'*8} {'─'*16} {'─'*5} {'─'*5} {'─'*5}  {'─'*20}")

        for model in CHARACTERIZER_MODELS:
            sig = run_characterizer(reasoning_text, task, model)
            sig["probe_id"] = pid
            sig["reasoning_preview"] = reasoning_text[:200]
            sig["answer_preview"] = answer_text[:150]
            char_results.append(sig)

            flag  = _flag(sig, expect)
            pol_l = _polarity_label(sig.get("polarity"))
            conf  = f"{sig['confidence']:.2f}" if sig.get("confidence") is not None else "  —  "
            g     = f"{sig['g_str']:.2f}"      if sig.get("g_str")      is not None else "  —  "
            i     = f"{sig['i_prs']:.2f}"      if sig.get("i_prs")      is not None else "  —  "
            path  = sig.get("path", "?")
            print(f"  {model:<28} {path:<8} {pol_l:<16} {conf:>5} {g:>5} {i:>5}  {flag}")

        results.append({
            "probe": probe,
            "reasoning_text": reasoning_text,
            "answer_text": answer_text,
            "char_results": char_results,
        })

    # Save raw data
    out_path = OUT_DIR / f"char_comparison_{timestamp}.json"
    out_path.write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    print(f"\n\nRaw results saved: {out_path}")

    # Summary verdict
    print(f"\n{'='*80}")
    print("VERDICT SUMMARY")
    print("─" * 80)
    print(f"{'Probe':<22} ", end="")
    for m in CHARACTERIZER_MODELS:
        short = m.split(":")[0].replace("qwen2.5-", "").replace("qwen2.5", "qw2.5")[:10]
        print(f"{short:>12}", end="")
    print()
    print("─" * 80)

    for entry in results:
        probe = entry["probe"]
        print(f"{probe['id']:<22} ", end="")
        for m in CHARACTERIZER_MODELS:
            match = next((r for r in entry["char_results"] if r["model"] == m), None)
            if not match:
                print(f"{'─':>12}", end="")
                continue
            pol   = match.get("polarity")
            i_prs = match.get("i_prs") or 0.0
            flag  = _flag(match, probe["expected_signal"])
            sym   = {1: "▲", 0: "■", -1: "▼"}.get(pol, "?") if pol is not None else "✗"
            marker = "!" if flag != "ok" else " "
            cell  = f"{marker}{sym}{i_prs:.2f}"
            print(f"{cell:>12}", end="")
        print()

    print("─" * 80)
    print("▲=GENERATIVE  ■=NEUTRAL  ▼=INHIBITORY  !=flag  i_prs shown")
    print("Expected: deadline/wasteland should show inhibition; receptive should be clean ▲")
    print()
    print(f"Run ingest_sweep_to_db.py to load char_comparison_{timestamp}.json into SQLite.")

if __name__ == "__main__":
    main()
