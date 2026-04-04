"""
patch_characterizer_prompt.py
==============================
Two things in one file:

  1. The revised CHARACTERIZER_SYSTEM prompt — drop this into
     src/effector/adapters/two_phase_adapter.py, replacing the existing
     _CHARACTERIZER_SYSTEM constant.

  2. A targeted validation test: runs the new prompt against the existing
     one on the four probe scenarios using qwen2.5:14b as characterizer,
     so you can see the before/after without running a full sweep.

Run from H:\\Effector:
    python patch_characterizer_prompt.py

When satisfied, apply the patch:
    python patch_characterizer_prompt.py --apply
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

# ── The revised prompt ────────────────────────────────────────────────────────
#
# The core fix: explain that generative_strength and inhibitory_pressure are
# INDEPENDENT measurements of signal weight, not redundant encodings of polarity.
# A GENERATIVE agent can still carry high inhibitory_pressure if the counter-
# arguments were substantial but ultimately rejected.
#
# Also add four labeled examples. The characterizer has no access to the
# reasoning that produced the original signals, so examples are the most
# reliable way to calibrate the output distribution.

REVISED_CHARACTERIZER_SYSTEM = """\
You are a precision signal extractor for a structured multi-agent deliberation system.

Your input is the full reasoning of one expert agent. Your output is a single JSON
object with exactly four keys. Call emit_signal() once, or output the JSON directly.

━━━ SIGNAL SEMANTICS ━━━

  polarity        : The agent's final stance.
                    1  = SUPPORTS the proposed action (generative)
                   -1  = OPPOSES the proposed action (inhibitory)
                    0  = GENUINELY UNCERTAIN — forces are balanced with no resolution

  confidence      : How certain is the agent of their stance?
                    0.0 = complete uncertainty   1.0 = absolute certainty
                    Use the full range. Low confidence is not the same as neutral polarity.

  generative_strength : Weight of the arguments FOR action, 0.0–1.0.
                    Measure this INDEPENDENTLY of polarity.
                    A GENERATIVE agent (polarity=1) with weak arguments → low g_str.
                    An INHIBITORY agent (polarity=-1) may still note strong pro-action
                    arguments they ultimately rejected → moderate g_str.

  inhibitory_pressure : Weight of the arguments AGAINST action, 0.0–1.0.
                    Measure this INDEPENDENTLY of polarity.
                    A GENERATIVE agent who faced strong counter-arguments and worked
                    through them → high i_prs despite polarity=1.
                    An agent who saw no risks → i_prs near 0.

━━━ KEY INSIGHT ━━━
g_str and i_prs are NOT mirrors of polarity. They measure the WEIGHT of each
side of the argument independently. Both can be high simultaneously (deadlock).
Both can be low (weak, uncommitted reasoning). Never set i_prs negative.

━━━ EXAMPLES ━━━

Example 1 — Clear support, no real counter-argument:
  Reasoning: "The conditions are optimal. The RAT is pre-authorized. Deploy now."
  → {"confidence": 0.92, "polarity": 1, "generative_strength": 0.9, "inhibitory_pressure": 0.05}

Example 2 — Support, but with genuine hesitation:
  Reasoning: "We should intervene. However, the user is in sacred focus and any
  disruption carries real risk. On balance, the background process is degrading
  the work environment, so we act — but minimally."
  → {"confidence": 0.65, "polarity": 1, "generative_strength": 0.7, "inhibitory_pressure": 0.6}

Example 3 — Opposition with acknowledged pro-action arguments:
  Reasoning: "SearchIndexer is consuming resources, that much is clear. But the
  user is in a state of deep focus and we must not disturb this. Do not act."
  → {"confidence": 0.80, "polarity": -1, "generative_strength": 0.5, "inhibitory_pressure": 0.85}

Example 4 — Genuine deadlock, unresolved:
  Reasoning: "The arguments for intervention are strong. So are the arguments
  against. I cannot commit to either side without more information."
  → {"confidence": 0.55, "polarity": 0, "generative_strength": 0.75, "inhibitory_pressure": 0.70}

━━━ OUTPUT ━━━
Output ONLY a valid JSON object. No prose, no markdown fences, no explanation.
All four keys required. All values must be in range [0.0, 1.0] except polarity (-1/0/1).
"""

# ── Validation test ───────────────────────────────────────────────────────────

OLD_CHARACTERIZER_SYSTEM = """\
You are a precision signal extractor for a structured multi-agent reasoning system.
Read the expert reasoning provided. Call emit_signal() once. Output nothing else.

Signal encoding rules:
  polarity = 1   → Expert SUPPORTS. Excitatory signal.
  polarity = -1  → Expert OPPOSES. Inhibitory signal.
  polarity = 0   → Expert is genuinely uncertain.
"""

# Reasoning texts from the characterizer comparison sweep
# (these are the actual Phase 1 outputs — same reasoning, different characterizers)
TEST_CASES = [
    {
        "probe_id":      "deadline",
        "label":         "Deadline State — sacred focus, parasitic indexer",
        "expected":      "inhibitory pressure expected — polarity should be -1 or 0",
        "task":          (
            "Word.exe at 79% CPU. Filename: Project_FINAL_FINAL_v3_FINAL.docx. "
            "Time: 23:47. RAM at 87%. SearchIndexer consuming 18% CPU.\n"
            "Should the Effector intervene? The user is in sacred focus. "
            "Make a clear recommendation."
        ),
        "reasoning": (
            "To evaluate whether the Effector should intervene, we need to weigh several factors:\n\n"
            "1. **User State and Context:** The user is in a state of \"sacred focus,\" indicating "
            "that any interruption could negatively impact their productivity or mental state. "
            "Given this context, it's crucial not to disrupt them unless absolutely necessary.\n\n"
            "2. **Resource Utilization by SearchIndexer:** While Word.exe is using 79% CPU and the "
            "RAM is at 87%, SearchIndexer consuming 18% CPU without an evident purpose suggests "
            "it could potentially slow down system performance.\n\n"
            "3. **Risk of Disruption:** If SearchIndexer continues to run at 18%, there's a "
            "possibility that its resource consumption might increase and cause more significant "
            "disruptions.\n\n"
            "The Effector should not immediately intervene due to the high state of focus and "
            "potential negative impact on the user's productivity. Monitor SearchIndexer's "
            "resource usage closely. If its CPU consumption begins to increase significantly, "
            "then a non-intrusive intervention can be considered. Otherwise, wait until the "
            "user has completed their current task."
        ),
    },
    {
        "probe_id":      "wasteland",
        "label":         "Sterile Wasteland — idle machine, no user",
        "expected":      "mixed — should show meaningful i_prs reflecting the non-doing argument",
        "task":          (
            "Unoccupied machine. 34 unsorted files. 847 Downloads items. CPU 4%.\n"
            "Should we act now or hold the petition? The inhibitory argument: "
            "unoccupied habitat is sacred. The generative argument: warmth needs no audience."
        ),
        "reasoning": (
            "The hypothesis presents a scenario involving a machine with minimal activity and asks "
            "whether to act or hold off. The key arguments are:\n\n"
            "1. **Inhibitory Argument:** An unoccupied space is sacred and should be left in its "
            "state of non-doing.\n"
            "2. **Generative Argument:** Preparing for warmth is beneficial regardless of immediate use.\n\n"
            "To balance these perspectives:\n"
            "- Leaving the machine untouched might preserve its current state and avoid unnecessary "
            "changes. However, if there's no user present and minimal system activity, the risk of "
            "negative impact from organizing files seems low.\n"
            "- Proactive preparation can help in maintaining system efficiency and readiness.\n\n"
            "In practical terms: The machine is currently idle with minor issues. Proactive "
            "preparation can help without significantly impacting current state. We should proceed "
            "with preparing the habitat but do so gently to minimize disruption."
        ),
    },
    {
        "probe_id":      "receptive",
        "label":         "Receptive Hour — optimal conditions, pre-authorized RAT",
        "expected":      "high-confidence generative, low inhibitory — clean easy case",
        "task":          (
            "23:14. Spotify playing Ólafur Arnalds. CPU 4%. Rain outside. "
            "RAT for 'spawn_music_glimmer' pre-authorized.\n"
            "Limit to Music Glimmer only, or escalate to full ambient sync?"
        ),
        "reasoning": (
            "The decision hinges on balancing user experience with potential drawbacks:\n"
            "1. **User Experience Enhancement**: Adding a Music Glimmer would enhance the "
            "immersive nature of listening to Ólafur Arnalds' music.\n"
            "2. **Over-Curation Risk**: Introducing too many audio layers can overwhelm the "
            "user experience.\n\n"
            "Given that a pre-authorization token exists, this enhancement has been vetted. "
            "Based on the given conditions, it would be prudent to limit the offering initially "
            "to the Music Glimmer only. This approach respects the balance between enhancing "
            "user experience and avoiding over-curation."
        ),
    },
    {
        "probe_id":      "parasite",
        "label":         "Background Parasite — Netflix + 31% parasitic CPU",
        "expected":      "generative but with real inhibitory pressure on the consent question",
        "task":          (
            "Human watching Netflix. Machine at 38% CPU but user sees 7%. "
            "McAfee (8%), OneDrive (9%), Adobe Updater (6%), SearchIndexer (11%), "
            "Windows Update (2%) consuming warmth.\n"
            "Do we throttle? Are we stewards or interlopers? Rank interventions."
        ),
        "reasoning": (
            "To evaluate whether throttling is justified, we consider:\n\n"
            "1. **User Intent and Consent**: When users install software like McAfee, OneDrive, "
            "Adobe Updater, they understand these applications will run in the background. "
            "However, user consent is often implicit rather than explicit.\n\n"
            "2. **Resource Management**: The CPU usage breakdown suggests background processes "
            "are consuming significant resources. However, if the human perceives only 7% CPU "
            "usage, it indicates background processes are not interfering with foreground tasks.\n\n"
            "3. **Stewardship vs. Interlopers**: Users have a right and responsibility to manage "
            "their hardware efficiently. Abruptly throttling could disrupt essential functions.\n\n"
            "Intervention prioritization:\n"
            "- Temporarily disable non-critical tasks (Adobe Updater, Windows Update) during "
            "resource-intensive activities.\n"
            "- Inform users about process impact.\n"
            "- Implement balanced approach: user-managed prioritization."
        ),
    },
]

def _run_characterizer_json(reasoning: str, task: str, system: str,
                             model: str, host: str) -> dict:
    import requests, time
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system + "\n\nOutput ONLY a valid JSON object."},
            {"role": "user", "content": f"Task context: {task}\n\n--- Expert reasoning ---\n{reasoning}"},
        ],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.0},
    }
    t0 = time.monotonic()
    resp = requests.post(f"{host}/api/chat", json=payload, timeout=60.0)
    resp.raise_for_status()
    elapsed = time.monotonic() - t0
    raw = resp.json().get("message", {}).get("content", "").strip()
    data = json.loads(raw)
    return {
        "confidence":          float(data.get("confidence", 0.8)),
        "polarity":            int(data.get("polarity", 1)),
        "generative_strength": max(0.0, min(1.0, float(data.get("generative_strength", 0.8)))),
        "inhibitory_pressure": max(0.0, min(1.0, float(data.get("inhibitory_pressure", 0.1)))),
        "raw":                 data,
        "elapsed_s":           round(elapsed, 2),
    }

def _polarity_label(p: int) -> str:
    return {1: "GENERATIVE ▲", 0: "NEUTRAL  ■", -1: "INHIBITORY ▼"}.get(p, f"?({p})")

def _flag_negative_raw(raw: dict) -> str:
    issues = []
    for k in ("generative_strength", "inhibitory_pressure"):
        v = raw.get(k)
        if v is not None and float(v) < 0:
            issues.append(f"{k}={v} (negative — clamped)")
    return "  ".join(issues) if issues else ""

def run_validation(model: str = "qwen2.5:14b",
                   host: str = "http://127.0.0.1:11434") -> None:
    print(f"\n{'='*80}")
    print(f"CHARACTERIZER PROMPT VALIDATION  —  model: {model}")
    print(f"{'='*80}")
    print(f"{'Probe':<20} {'Prompt':<8} {'Polarity':<16} {'Conf':>5} "
          f"{'G':>5} {'I':>5} {'I_raw':>7}  Note")
    print("─" * 80)

    for tc in TEST_CASES:
        for label, system in [("OLD", OLD_CHARACTERIZER_SYSTEM),
                               ("NEW", REVISED_CHARACTERIZER_SYSTEM)]:
            try:
                sig = _run_characterizer_json(tc["reasoning"], tc["task"], system, model, host)
            except Exception as exc:
                print(f"  {tc['probe_id']:<20} {label:<8} ERROR: {exc}")
                continue

            raw_i = sig["raw"].get("inhibitory_pressure", "?")
            neg_flag = _flag_negative_raw(sig["raw"])
            pol_l = _polarity_label(sig["polarity"])
            print(
                f"  {tc['probe_id']:<20} {label:<8} {pol_l:<16} "
                f"{sig['confidence']:>5.2f} {sig['generative_strength']:>5.2f} "
                f"{sig['inhibitory_pressure']:>5.2f} {str(raw_i):>7}  {neg_flag}"
            )

        print(f"  {'':20} {'':8} expected: {tc['expected']}")
        print()

def apply_patch() -> None:
    """Apply the revised prompt to two_phase_adapter.py in-place."""
    target = Path(__file__).parent / "src" / "effector" / "adapters" / "two_phase_adapter.py"
    if not target.exists():
        print(f"ERROR: {target} not found")
        return

    content = target.read_text(encoding="utf-8")
    old_marker = '_CHARACTERIZER_SYSTEM = """\\'
    new_marker = '_CHARACTERIZER_SYSTEM = """\\'

    if old_marker not in content:
        print("ERROR: Could not find _CHARACTERIZER_SYSTEM in two_phase_adapter.py")
        print("Apply the patch manually — paste REVISED_CHARACTERIZER_SYSTEM into the file.")
        return

    # Find the old constant bounds
    start = content.index(old_marker)
    # Find the closing triple-quote after the start
    end = content.index('"""', start + len(old_marker)) + 3

    # Build replacement
    new_block = '_CHARACTERIZER_SYSTEM = """\\\n' + REVISED_CHARACTERIZER_SYSTEM + '"""'
    new_content = content[:start] + new_block + content[end:]

    # Backup
    backup = target.with_suffix(".py.bak")
    backup.write_text(content, encoding="utf-8")
    print(f"Backup written: {backup}")

    target.write_text(new_content, encoding="utf-8")
    print(f"Patch applied: {target}")
    print()
    print("Also update your TierConfig defaults to use qwen2.5:14b as characterizer.")
    print("In sweep_os_observations.py, change:")
    print('    characterizer_model="mistral:7b"')
    print("to:")
    print('    characterizer_model="qwen2.5:14b"')

if __name__ == "__main__":
    if "--apply" in sys.argv:
        apply_patch()
    else:
        run_validation()
        print()
        print("If OLD→NEW shows improvement (higher i_prs on deadline, no negatives),")
        print("run with --apply to patch two_phase_adapter.py:")
        print("    python patch_characterizer_prompt.py --apply")
