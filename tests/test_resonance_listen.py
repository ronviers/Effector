"""
tests/test_resonance_listen.py — Resonance Layer Listening Test
================================================================
Produces actual audio files for subjective human evaluation.

This is not a pass/fail test. It is a characterisation tool.
The files it creates are the test. You listen to them and report
what you hear. That report is the result.

What gets generated
-------------------
  VOICE (Kokoro-82M)
    Five short clips — one per agent archetype — each reading a
    snippet of lore-appropriate reasoning text at the voice and
    tone assigned to that archetype. Also one of each event
    announcement type (consensus, escalation, inhibition, etc.)

  WORLD SOUND (AudioLDM 2)
    One generated ambient clip per major IEP action category:
    file organisation, Glimmer arrival, ambient sync,
    brightness change, archive, and the default "miracle" tone.
    Generated at steps=20 (fast) so you can hear them quickly.

Output
------
  All files are written to:
    H:\\Effector\\resonance_test_audio\\

  A manifest (manifest.txt) lists every file with a one-line
  description of what it should sound like, so you know what
  you're listening for.

Usage
-----
  # Run the full listening test (Kokoro + AudioLDM 2):
  python tests/test_resonance_listen.py

  # Voice only (fast, ~10 seconds):
  python tests/test_resonance_listen.py --voice-only

  # World only:
  python tests/test_resonance_listen.py --world-only

  # Faster world generation (fewer steps, lower quality):
  python tests/test_resonance_listen.py --steps 10

  # After running, just open the folder:
  #   explorer resonance_test_audio
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

# ── Path fix ─────────────────────────────────────────────────────────────────
_HERE     = Path(__file__).parent
_ROOT     = _HERE.parent
_OUT_DIR  = _ROOT / "resonance_test_audio"



# ── Voice clips to generate ───────────────────────────────────────────────────
#
# Each entry: (filename_stem, agent_id, text, description_for_manifest)
#
# Texts are written in the agents' actual register — earnest, slightly archaic,
# theologically inflected. These are the kinds of things the engine actually says.

VOICE_CLIPS = [
    (
        "voice_weave_proposal",
        "WEAVE-3",
        (
            "The Downloads folder registers a Snug deficit of critical magnitude. "
            "I propose immediate deployment of the Librarian Glimmer "
            "to restore orbital order."
        ),
        "WEAVE agent (af_heart) — warm, earnest, proposing an action. "
        "Should sound like a small creature genuinely worried about your folder.",
    ),
    (
        "voice_spark_evaluation",
        "SPARK-2",
        (
            "The proposal has merit, but I observe the inhibitory pressure "
            "exceeds the generative signal by a margin of concern. "
            "We risk a stall gate before consensus is reached."
        ),
        "SPARK agent (am_michael) — analytical, slightly cautionary. "
        "Should sound like someone who has read the protocol very carefully.",
    ),
    (
        "voice_arbiter_verdict",
        "ARBITER",
        (
            "The coalition has deliberated. "
            "The Intention Envelope shall be formed. "
            "The Effector will judge its worth."
        ),
        "ARBITER (bf_emma) — authoritative, calm, finalising. "
        "Should sound like a very composed British official closing a meeting.",
    ),
    (
        "voice_nemotron_escalation",
        "nemotron",
        (
            "Local agents have reached an impasse. "
            "I am the cloud arbiter, and I have reviewed all prior arguments. "
            "The correct course is clear."
        ),
        "NEMOTRON cloud arbiter (bf_emma) — same voice as Arbiter, "
        "but the text signals higher authority. Should feel decisive.",
    ),
    (
        "voice_mistral_reasoning",
        "mistral",
        (
            "CPU pressure is low. The active window is sterile. "
            "A Glimmer deployment would increase ambient Snug "
            "without exceeding the inhibitory threshold."
        ),
        "MISTRAL tier-1 agent (af_heart) — matter-of-fact analysis. "
        "Should sound like a competent small creature reading a dashboard.",
    ),
]

# Event announcements — what plays between debate rounds
ANNOUNCEMENT_CLIPS = [
    (
        "announce_round_start",
        "round_started",
        2,
        "Round announcement (arbiter voice). Short, ceremonial.",
    ),
    (
        "announce_consensus",
        "consensus",
        0,
        "Consensus achieved. Should sound like relief and satisfaction.",
    ),
    (
        "announce_escalation",
        "escalation",
        0,
        "Escalation triggered. Should sound like something important is happening.",
    ),
    (
        "announce_inhibition",
        "inhibition",
        0,
        "Inhibition gate fired. A veto. Should sound like a firm no.",
    ),
    (
        "announce_offering_accepted",
        "offering_accepted",
        0,
        "ACK received. Miracle granted. Should feel warm and complete.",
    ),
    (
        "announce_offering_rejected",
        "offering_rejected",
        0,
        "NACK. The void. Should feel like a door closing quietly.",
    ),
]

# ── World sound clips to generate ─────────────────────────────────────────────
#
# Each entry: (filename_stem, iep_target, description_for_manifest)

WORLD_CLIPS = [
    (
        "world_glimmer_arrives",
        "desktop.overlay.glimmer",
        "A Glimmer landing on the desktop. Should be small, crystalline, "
        "like a tiny chime and a soft footstep. Something arrived.",
    ),
    (
        "world_files_organised",
        "os.filesystem.zen_habitat",
        "Files moving into the Zen Habitat. Should feel like "
        "satisfying organisation — books stacking, things clicking into place.",
    ),
    (
        "world_archive",
        "os.filesystem.archive",
        "Archiving files. Should feel like a thick book closing with finality. "
        "Heavy, settled, complete.",
    ),
    (
        "world_brightness_dims",
        "os.display.brightness",
        "Monitor brightness being reduced. Should feel like something "
        "softening — a dimmer switch turning, a room becoming quieter.",
    ),
    (
        "world_rain_sync",
        "desktop.ambient_sync",
        "Ambient sync when it is raining outside. Should feel like "
        "rain on a window — the contrast between outside and the warm inside.",
    ),
    (
        "world_campfire",
        "desktop.overlay.ambient.campfire",
        "A campfire being spawned at the bottom of the screen. "
        "Should feel like dry wood crackling, small pops, warmth.",
    ),
    (
        "world_default_miracle",
        "world_state",
        "The default miracle tone — when no specific sound is mapped. "
        "A resonant wooden bowl struck once. Clear, harmonic, fading.",
    ),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_output_dir() -> Path:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    return _OUT_DIR


def _save_numpy_as_wav(audio, sample_rate: int, path: Path) -> None:
    """Save float32 numpy array as 16-bit WAV."""
    import numpy as np
    import scipy.io.wavfile
    clipped   = np.clip(audio, -1.0, 1.0)
    as_int16  = (clipped * 32767).astype(np.int16)
    scipy.io.wavfile.write(str(path), sample_rate, as_int16)


def _open_folder(path: Path) -> None:
    """Open the output folder in the system file explorer."""
    import subprocess, sys
    try:
        if sys.platform == "win32":
            subprocess.Popen(["explorer", str(path)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        pass


# ── Voice generation ──────────────────────────────────────────────────────────

def run_voice_test(out_dir: Path, manifest_lines: list) -> dict:
    """
    Generate all voice clips and save them as WAV files.
    Returns a metrics dict.
    """
    from effector.resonance.resonance_voice import (
        ResonanceVoice,
        _resolve_voice,
        _EVENT_PHRASES,
    )
    import random

    metrics = {
        "clips_attempted": 0,
        "clips_succeeded": 0,
        "clips_failed":    [],
        "load_time_s":     0.0,
        "per_clip_ms":     {},
    }

    print("\n  Loading Kokoro-82M...")
    t0 = time.monotonic()
    voice = ResonanceVoice(enabled=True, speed=1.0)
    voice.start()
    loaded = voice.wait_ready(timeout=45.0)
    metrics["load_time_s"] = round(time.monotonic() - t0, 2)

    if not loaded:
        print("  ✗ Kokoro failed to load within 45s")
        manifest_lines.append("\n[VOICE — FAILED TO LOAD]\n")
        return metrics

    print(f"  ✓ Loaded in {metrics['load_time_s']:.1f}s")

    # We need the pipeline directly to capture audio bytes
    pipeline = voice._pipeline
    if pipeline is None:
        print("  ✗ Pipeline is None despite load_ok")
        return metrics

    manifest_lines.append("\n" + "─" * 56)
    manifest_lines.append("VOICE CLIPS  (Kokoro-82M TTS)")
    manifest_lines.append("─" * 56)

    # ── Agent reasoning clips ─────────────────────────────────────────────────
    for stem, agent_id, text, description in VOICE_CLIPS:
        metrics["clips_attempted"] += 1
        out_path = out_dir / f"{stem}.wav"
        used_voice = _resolve_voice(agent_id)

        print(f"  Synthesising: {stem}  [{used_voice}]...")
        t0 = time.monotonic()
        try:
            all_audio = []
            import numpy as np
            for _gs, _ps, audio_chunk in pipeline(text, voice=used_voice, speed=1.0):
                all_audio.append(audio_chunk)

            if all_audio:
                combined = np.concatenate(all_audio)
                _save_numpy_as_wav(combined, 24_000, out_path)
                elapsed_ms = (time.monotonic() - t0) * 1000
                metrics["per_clip_ms"][stem] = round(elapsed_ms)
                metrics["clips_succeeded"] += 1
                print(f"    ✓ {out_path.name}  ({elapsed_ms:.0f}ms)")
                manifest_lines.append(f"\n{out_path.name}")
                manifest_lines.append(f"  Agent    : {agent_id}  →  voice: {used_voice}")
                manifest_lines.append(f"  Text     : \"{text[:80]}{'…' if len(text)>80 else ''}\"")
                manifest_lines.append(f"  Expect   : {description}")
            else:
                raise ValueError("Pipeline returned no audio chunks")

        except Exception as exc:
            print(f"    ✗ Failed: {exc}")
            metrics["clips_failed"].append(stem)
            manifest_lines.append(f"\n{stem}.wav  — GENERATION FAILED: {exc}")

    # ── Event announcement clips ──────────────────────────────────────────────
    manifest_lines.append("\n" + "─" * 56)
    manifest_lines.append("EVENT ANNOUNCEMENTS")
    manifest_lines.append("─" * 56)

    for stem, event_key, round_num, description in ANNOUNCEMENT_CLIPS:
        metrics["clips_attempted"] += 1
        out_path = out_dir / f"{stem}.wav"
        phrases  = _EVENT_PHRASES.get(event_key, [])
        if not phrases:
            print(f"  Skipping {stem} — no phrase defined for event {event_key!r}")
            continue

        phrase = random.choice(phrases)
        if "{}" in phrase:
            phrase = phrase.format(round_num)

        print(f"  Synthesising: {stem}...")
        t0 = time.monotonic()
        try:
            import numpy as np
            all_audio = []
            for _gs, _ps, audio_chunk in pipeline(phrase, voice="bf_emma", speed=1.0):
                all_audio.append(audio_chunk)

            if all_audio:
                combined = np.concatenate(all_audio)
                _save_numpy_as_wav(combined, 24_000, out_path)
                elapsed_ms = (time.monotonic() - t0) * 1000
                metrics["per_clip_ms"][stem] = round(elapsed_ms)
                metrics["clips_succeeded"] += 1
                print(f"    ✓ {out_path.name}  ({elapsed_ms:.0f}ms)")
                manifest_lines.append(f"\n{out_path.name}")
                manifest_lines.append(f"  Text     : \"{phrase}\"")
                manifest_lines.append(f"  Expect   : {description}")
            else:
                raise ValueError("No audio chunks returned")
        except Exception as exc:
            print(f"    ✗ Failed: {exc}")
            metrics["clips_failed"].append(stem)
            manifest_lines.append(f"\n{stem}.wav  — GENERATION FAILED: {exc}")

    voice.stop()
    return metrics


# ── World sound generation ─────────────────────────────────────────────────────

def run_world_test(out_dir: Path, manifest_lines: list, num_steps: int = 20) -> dict:
    """
    Generate world sound clips and save them as WAV files.
    Returns a metrics dict.
    """
    from effector.resonance.resonance_world import ResonanceWorld, _resolve_prompt

    metrics = {
        "clips_attempted": 0,
        "clips_succeeded": 0,
        "clips_failed":    [],
        "load_time_s":     0.0,
        "per_clip_s":      {},
    }

    progress_lines = []

    def _on_progress(step: int, total: int, text: str) -> None:
        if step == 0 or step == total or step % 5 == 0:
            filled = int(20 * step / max(total, 1))
            bar = "█" * filled + "░" * (20 - filled)
            print(f"    [{bar}] {step}/{total}", end="\r")

    print(f"\n  Loading AudioLDM 2 (steps={num_steps})...")
    t0 = time.monotonic()
    world = ResonanceWorld(
        enabled=True,
        num_steps=num_steps,
        audio_length_s=4.0,
        on_progress=_on_progress,
    )
    world.start()
    loaded = world.wait_ready(timeout=90.0)
    metrics["load_time_s"] = round(time.monotonic() - t0, 2)

    if not loaded:
        print("  ✗ AudioLDM 2 failed to load within 90s")
        manifest_lines.append("\n[WORLD SOUND — FAILED TO LOAD]\n")
        return metrics

    print(f"  ✓ Loaded in {metrics['load_time_s']:.1f}s")

    manifest_lines.append("\n" + "─" * 56)
    manifest_lines.append(f"WORLD SOUND CLIPS  (AudioLDM 2, steps={num_steps})")
    manifest_lines.append("─" * 56)

    for stem, target, description in WORLD_CLIPS:
        metrics["clips_attempted"] += 1
        out_path = out_dir / f"{stem}.wav"
        prompt   = _resolve_prompt(target)

        print(f"\n  Generating: {stem}")
        print(f"    Target : {target}")
        print(f"    Prompt : {prompt[:70]}...")
        t0 = time.monotonic()
        try:
            import numpy as np
            import scipy.io.wavfile

            # Call the pipeline directly so we capture the output without
            # going through the temp-file path, and without triggering playback
            result = world._pipeline(
                prompt,
                num_inference_steps=num_steps,
                audio_length_in_s=4.0,
            )
            audio = result.audios[0]
            _save_numpy_as_wav(audio, 16_000, out_path)

            elapsed = time.monotonic() - t0
            metrics["per_clip_s"][stem] = round(elapsed, 1)
            metrics["clips_succeeded"] += 1
            print(f"    ✓ {out_path.name}  ({elapsed:.1f}s)")

            manifest_lines.append(f"\n{out_path.name}")
            manifest_lines.append(f"  Target   : {target}")
            manifest_lines.append(f"  Prompt   : \"{prompt[:100]}{'…' if len(prompt)>100 else ''}\"")
            manifest_lines.append(f"  Expect   : {description}")

        except Exception as exc:
            print(f"\n    ✗ Failed: {exc}")
            metrics["clips_failed"].append(stem)
            manifest_lines.append(f"\n{stem}.wav  — GENERATION FAILED: {exc}")
            if "'GPT2Model' object has no attribute '_update_model_kwargs_for_generation'" in str(exc):
                metrics["clips_failed"].append("__gpt2_cache_stale__")

    return metrics


# ── Manifest writer ───────────────────────────────────────────────────────────

def write_manifest(out_dir: Path, manifest_lines: list, metrics: dict) -> Path:
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M")
    path = out_dir / "manifest.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("═" * 56 + "\n")
        f.write("  RESONANCE LAYER — LISTENING TEST MANIFEST\n")
        f.write(f"  Generated: {ts}\n")
        f.write("═" * 56 + "\n")
        f.write("\nHOW TO USE THIS FILE\n")
        f.write("─" * 56 + "\n")
        f.write(
            "Each entry below names a WAV file and describes what\n"
            "it should sound like. Listen to the file, then report\n"
            "back whether the description matches what you hear.\n\n"
            "You can open all files in Windows Media Player or just\n"
            "double-click them in Explorer.\n\n"
            "WHAT TO REPORT:\n"
            "  - Does it sound like the description?\n"
            "  - Is there any speech / sound at all?\n"
            "  - How long is it? Does it feel right?\n"
            "  - Any distortion, clipping, or silence?\n"
            "  - Which clips feel most appropriate for the lore?\n"
        )
        for line in manifest_lines:
            f.write(line + "\n")

        # ── GPT2 cache warning if detected ────────────────────────────────────
        all_failed = metrics.get("world_failed", [])
        if "__gpt2_cache_stale__" in all_failed:
            f.write("\n" + "!" * 56 + "\n")
            f.write("  AUDIOLDM 2 CACHE IS STALE\n")
            f.write("!" * 56 + "\n")
            f.write(
                "\nThe cached model has a GPT2Model (encoder only)\n"
                "but AudioLDM 2 needs GPT2LMHeadModel (with generation head).\n"
                "This is a download-version mismatch. Fix:\n\n"
                "  1. Delete the stale cache:\n"
                "     Remove-Item -Recurse -Force "
                '"$env:USERPROFILE\\.cache\\huggingface\\hub\\models--cvssp--audioldm2"\n\n'
                "  2. Re-download:\n"
                "     python -c \"from diffusers import AudioLDM2Pipeline; "
                "AudioLDM2Pipeline.from_pretrained('cvssp/audioldm2')\"\n\n"
                "  3. Re-run this test.\n\n"
                "Voice clips (Kokoro) are unaffected and should work now.\n"
            )
    return path


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="test_resonance_listen",
        description="Generate audio files for subjective Resonance Layer evaluation",
    )
    parser.add_argument(
        "--voice-only", action="store_true",
        help="Only generate voice clips (Kokoro TTS)",
    )
    parser.add_argument(
        "--world-only", action="store_true",
        help="Only generate world sound clips (AudioLDM 2)",
    )
    parser.add_argument(
        "--steps", type=int, default=20,
        help="AudioLDM 2 diffusion steps (default 20 — fast but rough). "
             "Use 50 for production quality.",
    )
    args = parser.parse_args()

    out_dir = _ensure_output_dir()
    manifest_lines: list = []
    all_metrics: dict    = {"world_failed": []}

    print("\n" + "═" * 56)
    print("  RESONANCE LISTENING TEST")
    print(f"  Output → {out_dir}")
    print("═" * 56)

    if not args.world_only:
        vm = run_voice_test(out_dir, manifest_lines)
        all_metrics["voice"] = vm
        print(
            f"\n  Voice: {vm['clips_succeeded']}/{vm['clips_attempted']} clips"
            f" in {sum(vm['per_clip_ms'].values())/1000:.1f}s total"
        )

    if not args.voice_only:
        wm = run_world_test(out_dir, manifest_lines, num_steps=args.steps)
        all_metrics["world"] = wm
        all_metrics["world_failed"] = wm.get("clips_failed", [])
        print(
            f"\n  World: {wm['clips_succeeded']}/{wm['clips_attempted']} clips"
            f" in {sum(wm['per_clip_s'].values()):.1f}s total"
        )

    manifest_path = write_manifest(out_dir, manifest_lines, all_metrics)

    print("\n" + "═" * 56)
    print("  DONE")
    print("═" * 56)
    print(f"\n  Audio files : {out_dir}")
    print(f"  Manifest    : {manifest_path.name}")
    print("\n  Open folder and listen:")
    print(f"    explorer {out_dir}")
    print()

    # Print a quick listening guide to the terminal too
    voice_ok = all_metrics.get("voice", {}).get("clips_succeeded", 0)
    world_ok = all_metrics.get("world", {}).get("clips_succeeded", 0)

    if voice_ok == 0 and world_ok == 0:
        print("  ✗ No audio was generated. Check errors above.")
    else:
        print("  WHAT TO LISTEN FOR:")
        if voice_ok > 0:
            print("    Voice clips  — do the different agents sound distinct?")
            print("                   Is the ARBITER voice more authoritative")
            print("                   than the WEAVE voice?")
        if world_ok > 0:
            print("    World clips  — does each sound fit its action?")
            print("                   The Glimmer should feel small and crystalline.")
            print("                   The Archive should feel heavy and final.")
        print()

    # Open the folder automatically
    _open_folder(out_dir)

    # Return non-zero if world failed with the GPT2 issue specifically
    if "__gpt2_cache_stale__" in all_metrics.get("world_failed", []):
        print("  ⚠  AudioLDM 2 cache is stale. See manifest.txt for fix.")
        sys.exit(2)


if __name__ == "__main__":
    main()