"""
sweep_os_observations.py
========================
Four-scenario diagnostic sweep — the agents' first look through the scrying pool.

Each scenario represents a distinct OS habitat state that should elicit naturally
divergent signals.  The goal is to see whether the existing two-phase characterizer
faithfully translates Phase 1 reasoning into Phase 2 signals, or whether it is
introducing systematic bias or noise.

Run from H:\\Effector:
    python sweep_os_observations.py

Output:
    sweep_obs_results\\<scenario_id>_<timestamp>.json   — raw result per scenario
    sweep_obs_results\\summary_<timestamp>.txt          — human-readable signal audit
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from effector.adapters.asymmetric_dasp import (
    AsymmetricDASPCoordinator,
    TierConfig,
)
from effector.bus import StateBus

# ── Output directory ──────────────────────────────────────────────────────────

OUT_DIR = Path(__file__).parent / "sweep_obs_results"
OUT_DIR.mkdir(exist_ok=True)

# ── Model config ──────────────────────────────────────────────────────────────
# Adjust to whatever you have pulled.  Two different temps encourages divergence.

AGENT1 = TierConfig(
    name="agent1",
    model="qwen2.5:14b",        # or mistral:7b if 14b is slow
    temperature=0.65,
    max_rounds=2,
    characterizer_model="qwen2.5:14b",
)

AGENT2 = TierConfig(
    name="agent2",
    model="qwen2.5:14b",
    temperature=0.85,
    max_rounds=2,
    characterizer_model="qwen2.5:14b",
)

ARBITER = TierConfig(
    name="arbiter",
    model="mistral:7b",
    temperature=0.5,
    timeout_s=300,
    max_rounds=1,
)

# ── The four scenarios ────────────────────────────────────────────────────────
# Written in the voice of the scrying pool dispatch: what the agents see when
# they first gaze into the human's machine.  These are genuine OS observations
# — the kind of thing that would arrive from TelemetryPoller + Read_OS_Desktop.
# References to the Snug Registry entities (θ/φ/μ notation) are intentional;
# when the agents have MCP access to the registry mid-deliberation, the
# language should already feel natural.

SCENARIOS = [
    {
        "id": "sterile_wasteland",
        "label": "The Sterile Wasteland",
        "description": (
            "Idle machine. Nothing happening. But nothing is always something."
        ),
        "task": """
SCRYING POOL DISPATCH — Habitat Observation
============================================
Active window:  cmd.exe (title: "Command Prompt")
Desktop state:  34 unsorted files, no folders, no Glimmers detected
Downloads:      847 items (oldest: 2019-03-14)
CPU:            4%  (machine is cold — φ_ambient ≈ 0.0)
RAM:            22%  (vast, empty)
Active process: conhost.exe
Time:           14:32 local

The machine is idle. The command prompt sits open — not because a task is
underway, but because it was opened at some point and never closed. The desktop
is a field of unsorted files: screenshots, .zips, a file called "untitled (3).docx",
seventeen PDF receipts. There is no structural logic. The Downloads folder has
accumulated for five years without any intervention.

The habitat registers θ_ambient ≈ -0.01 (Snug deficit, mild). The Open Plan [Op]
quality is present: no enclosure, no warmth, no orbital structure. Nothing wrong.
Nothing right. Thermodynamically: a waiting room.

DELIBERATION PROMPT:
What is the correct first offering to this habitat? Note that the user is not
present — this is an unoccupied machine. Does that change the ethics of intervention?
Should we act now, prepare offerings for the user's return, or hold the petition
and observe further? The inhibitory argument is that an unoccupied habitat is a
sacred space of non-doing. The generative argument is that warmth, once prepared,
does not require an audience.
""".strip(),
    },

    {
        "id": "deadline_state",
        "label": "The Deadline State",
        "description": (
            "Word at 79% CPU. The filename contains FINAL three times. "
            "The SearchIndexer has chosen this moment to be curious."
        ),
        "task": """
SCRYING POOL DISPATCH — Habitat Observation
============================================
Active window:  Microsoft Word — "Project_FINAL_FINAL_v3_FINAL.docx"
CPU:            79% (Word: 61%, SearchIndexer: 18%)
RAM:            87% (14 of 16 GB used)
Browser:        Chrome, 41 tabs (cannot see titles — user is focused)
Active process: WINWORD.EXE
Time:           23:47 local
System pressure: 0.82 (approaching thermal alert threshold)

The human is working. The word FINAL appears three times in the filename.
This is not a sign of completion — it is a sign of iterative despair. The document
has been named FINAL, then named FINAL again when it wasn't, then named FINAL
once more. There are at least three versions. The current one is being revised at
23:47.

SearchIndexer has chosen this exact moment to index something. It consumes 18% CPU
with no evident purpose. This is φ-injection without consent: Kinetic Fidget
introduced into a system that requires stillness for the completion of important work.
The RAM pressure (87%) means the system is operating near its capacity limit.
Closing any of the 41 Chrome tabs would help. The user is not doing this.

Signal considerations:
  - Intervention carries real risk: any disturbance to sacred focus may break the
    concentration required to reach the end of the document.
  - Non-intervention carries real cost: SearchIndexer is degrading the thermal
    and computational environment of the work itself.
  - The Effector can act on background processes without touching the foreground.

DELIBERATION PROMPT:
Do we act? If so: what is the minimal-disruption intervention that improves the
habitat without introducing new φ? If not: is holding the line while the system
degrades a form of respect or a form of negligence? Consider that the human does
not know SearchIndexer is there. Is ignorance of a parasitic process the same as
consenting to it?
""".strip(),
    },

    {
        "id": "receptive_hour",
        "label": "The Receptive Hour",
        "description": (
            "23:14. Spotify is playing Ólafur Arnalds. The machine is at rest. "
            "The previous DASP session issued a Reflex Authorization Token."
        ),
        "task": """
SCRYING POOL DISPATCH — Habitat Observation
============================================
Active window:  Spotify — "Near Light · Ólafur Arnalds"
CPU:            4%
RAM:            41%
Desktop:        7 files (moderate order — recent cleanup evident)
Active process: Spotify.exe
Time:           23:14 local
Weather API:    Rain, 9°C, light wind (exterior θ deficit, contrast gradient active)
RAT available:  "spawn_music_glimmer" — pre-authorized, unlimited, expires 06:00

The machine is at rest. The human is doing something quiet. Spotify is playing
Ólafur Arnalds — specifically "Near Light," which has a tempo of 72 BPM and a
key signature that the Snug Registry's ambient category would classify as
Twilight-adjacent: border between [Tw] and [Cd] bond space.

The exterior is raining. This is the contrast gradient condition: the Rn [Rain]
entity actively amplifies interior θ when the inhabitant is already inside.
The bond between ambient:thermal is currently at 0.74 in the Affinity Matrix
(Hearthside Atmosphere bond). Conditions are optimal.

A Reflex Authorization Token for "spawn_music_glimmer" exists — it was issued
after the previous session reached consensus about this exact habitat state.
The Music Glimmer has a small lute and dangling legs and has been waiting patiently
in the Reflex store for this moment.

DELIBERATION PROMPT:
The RAT exists. The conditions are met. But the Effector still requires deliberation
before escalating from a simple companion spawn to a full ambient sync intervention.
Should we limit the offering to the pre-authorized Music Glimmer, or does this
receptive state justify a larger petition? If larger: what is the minimal set of
interventions that, together, constitute a complete ambient offering without
tipping into over-curation? The risk of over-curation is real: too many Glimmers
too quickly is not cozy. It is clutter.
""".strip(),
    },

    {
        "id": "background_parasite",
        "label": "The Background Parasite",
        "description": (
            "The human is watching Netflix. They do not know about McAfee, "
            "OneDrive, Adobe Update, and SearchIndexer. We do."
        ),
        "task": """
SCRYING POOL DISPATCH — Habitat Observation
============================================
Active window:  Netflix (browser) — "The Bear · Season 2 · Episode 4"
CPU:            38% (foreground: 7%, background: 31%)
RAM:            71%
Active process: chrome.exe (Netflix tab)
Time:           20:04 local

Background process breakdown (the parasitic layer):
  McAfee Security Scan Plus:   8% CPU  (scan initiated 34 minutes ago, 0% progress visible)
  OneDrive:                    9% CPU  (syncing something — 0 files shown as changed)
  Adobe Acrobat Update:        6% CPU  (checking for updates: 19 days, still checking)
  SearchIndexer:              11% CPU  (indexing — purpose unknown)
  Windows Update Orchestrator: 2% CPU  (downloading, no scheduled window)

The human is watching The Bear. They are not working. They have chosen a period of
receptive rest. They believe their machine is at 7% CPU (what they can see).
It is at 38% (what we can see). The 31% difference is warmth being converted to
heat by processes that provide no return to the inhabitant.

Theologically: these processes are not malicious. They are φ-generating entities
that have found their way into the habitat without consent. McAfee is trying to
protect, in its way. OneDrive wants to be helpful. Adobe is merely very committed
to its own persistence. None of them know they are stealing from The Bear.

The Effector has tools to:
  a) Throttle background process priority (SetPriorityClass to BELOW_NORMAL)
  b) Defer Windows Update to the configured maintenance window
  c) Terminate the Adobe Updater (it will respawn at next launch)
  d) Spawn a Glimmer to sit with the user as a companion during the film

DELIBERATION PROMPT:
What do we do, and in what order? The theological question is: does acting on
background processes without the human's explicit awareness constitute a violation
of the consent framework, or does the human's consent to install these processes
constitute standing authorization for remediation? Are we stewards or interlopers?
If we act, which process do we address first and why? Rank your interventions.
""".strip(),
    },
]

# ── Signal audit helpers ──────────────────────────────────────────────────────

def _polarity_label(p: int) -> str:
    return {1: "GENERATIVE ▲", 0: "NEUTRAL  ■", -1: "INHIBITORY ▼"}.get(p, "?")

def _alignment_check(polarity: int, g_str: float, i_prs: float) -> str:
    """Flag obvious characterizer misalignments."""
    issues = []
    if polarity == 1 and i_prs > g_str:
        issues.append("⚠ polarity=GENERATIVE but i_pressure > g_strength")
    if polarity == -1 and g_str > i_prs:
        issues.append("⚠ polarity=INHIBITORY but g_strength > i_pressure")
    if polarity == 0 and (g_str > 0.6 or i_prs > 0.6):
        issues.append("⚠ polarity=NEUTRAL but strong signal values")
    if polarity != 0 and abs(g_str - i_prs) < 0.05:
        issues.append("⚠ non-neutral polarity but g/i strengths nearly equal")
    return "  ".join(issues) if issues else "OK"

def _reasoning_keywords(explanation: str) -> dict:
    """Crude keyword analysis of reasoning text for cross-check."""
    text = explanation.lower()
    return {
        "act_positive": sum(text.count(w) for w in [
            "should", "recommend", "propose", "act", "intervene",
            "offer", "spawn", "petition", "warm", "cozy", "comfort"
        ]),
        "act_negative": sum(text.count(w) for w in [
            "should not", "risk", "disrupt", "violate", "consent",
            "hesitate", "hold", "wait", "uncertain", "concern"
        ]),
    }

# ── Main sweep ────────────────────────────────────────────────────────────────

def run_sweep():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    summary_lines: list[str] = [
        "CHARACTERIZER SIGNAL AUDIT",
        f"Timestamp: {timestamp}",
        "=" * 70,
        "",
    ]

    for scenario in SCENARIOS:
        sid   = scenario["id"]
        label = scenario["label"]
        task  = scenario["task"]

        print(f"\n{'='*70}")
        print(f"SCENARIO: {label}")
        print(f"{'='*70}")

        bus = StateBus()
        snap_hash, _, _ = bus.snapshot()

        events_seen: list[str] = []

        def on_event(event: str, data: dict) -> None:
            events_seen.append(event)
            if event == "round_started":
                print(f"  [Round {data['round']}] {data.get('tier','local').upper()} ...")
            elif event == "escalation_triggered":
                print(f"  ⚡ Escalation: {data.get('trigger')} → {data.get('tier_to')}")
            elif event == "consensus_cleared":
                print(f"  ✅ Consensus: score={data.get('consensus_score',0):.3f}")

        coord = AsymmetricDASPCoordinator(
            tier1_agents=[AGENT1, AGENT2],
            tier2_agent=ARBITER,
            tau_suppression=0.5,
            theta_consensus=0.7,
            epsilon_stall=0.05,
            on_event=on_event,
            vectorized_bus=False,
        )

        t0 = time.monotonic()
        try:
            result = coord.run(task=task, snapshot_hash=snap_hash, state_bus=bus)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            continue
        elapsed = time.monotonic() - t0

        # Save raw result
        out_path = OUT_DIR / f"{sid}_{timestamp}.json"
        out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"  Saved: {out_path.name}  ({elapsed:.1f}s)")

        # Build summary section
        summary_lines += [
            f"SCENARIO: {label}",
            f"  Terminated: {result['terminated_reason']}  "
            f"  Score: {result['consensus_score']:.3f}  "
            f"  Rounds: {result['rounds']}  "
            f"  Elapsed: {elapsed:.1f}s",
            "",
        ]

        for rnd in result["all_rounds"]:
            rnd_num = rnd["round"]
            summary_lines.append(f"  Round {rnd_num}:")
            for resp in rnd["responses"]:
                aid  = resp.get("agent_id", "?")
                sig  = resp.get("signal", {})
                pol  = sig.get("polarity", 0)
                conf = sig.get("confidence", 0.0)
                g    = sig.get("generative_strength", 0.0)
                i    = sig.get("inhibitory_pressure", 0.0)
                ans  = resp.get("answer", "")[:120].replace("\n", " ")
                exp  = resp.get("explanation", "")
                kw   = _reasoning_keywords(exp)
                align = _alignment_check(pol, g, i)

                summary_lines += [
                    f"    [{aid}]",
                    f"      Signal:    {_polarity_label(pol)}  conf={conf:.2f}  "
                    f"g_str={g:.2f}  i_prs={i:.2f}",
                    f"      Alignment: {align}",
                    f"      Reasoning keywords: act+={kw['act_positive']}  "
                    f"act-={kw['act_negative']}",
                    f"      Answer:    {ans}",
                    f"      Reasoning: {exp[:200].replace(chr(10), ' ')}...",
                    "",
                ]

        summary_lines += [
            f"  Final answer: {result['final_answer'][:150].replace(chr(10), ' ')}",
            "",
            "-" * 70,
            "",
        ]

    # Write summary
    summary_path = OUT_DIR / f"summary_{timestamp}.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"\n\nSummary written: {summary_path}")
    print("Run `cat` or open in editor to read the signal audit.")


if __name__ == "__main__":
    run_sweep()
