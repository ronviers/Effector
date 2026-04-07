"""
resonance_voice.py — The Rehearsal Room
=========================================
Kokoro-82M TTS voices for DASP agent debate audio.

During deliberation, each agent speaks its reasoning aloud. The player
hears the Rehearsal: the passionate, slightly liturgical argument that
precedes the Rite of Offering.

Voice archetypes
----------------
  WEAVE (proposal agents)  → af_heart   warm, earnest American female
  SPARK (evaluation agents)→ am_michael  crisp, probing American male
  ARBITER / NEMOTRON       → bf_emma    authoritative British female
  Default                  → af_heart

The voice queue is bounded (maxsize=4). If the debate outpaces playback,
older speeches are dropped silently — the debate never waits for audio.
Speech is truncated to the first sentence, max 220 chars, so it reads
as commentary rather than full recitation.

Installation
------------
    pip install kokoro sounddevice

    Windows phoneme support (recommended):
        Download espeak-ng installer from:
        https://github.com/espeak-ng/espeak-ng/releases
        then: pip install phonemizer
"""
from __future__ import annotations

import queue
import random
import re
import threading
import time
from typing import Optional

# ── Voice assignments ──────────────────────────────────────────────────────────

_VOICE_MAP: dict[str, str] = {
    "weave":    "af_heart",     # warm, earnest
    "spark":    "am_michael",   # analytical
    "arbiter":  "bf_emma",      # authoritative
    "nemotron": "bf_emma",
    "mistral":  "af_heart",
    "qwen":     "am_michael",
    "agent1":   "af_heart",
    "agent2":   "am_michael",
    "pip":      "af_heart",
    "default":  "af_heart",
}

# ── Lore-appropriate event announcements ──────────────────────────────────────

_EVENT_PHRASES: dict[str, list[str]] = {
    "round_started": [
        "Round {} of the sacred debate commences.",
        "The agents take their positions. Round {}.",
        "Deliberation continues — round {}.",
    ],
    "consensus": [
        "Consensus achieved. The Offering may be formed.",
        "The agents are in accord. A petition is possible.",
        "Agreement reached. The Effector may be petitioned.",
    ],
    "escalation": [
        "The Arbiter awakens.",
        "A higher authority is summoned.",
        "The High Arbiter is called to break the deadlock.",
    ],
    "inhibition": [
        "A veto has been cast. The gate holds.",
        "An inhibitory signal has blocked the path.",
        "The offering is stayed. The agents must reconsider.",
    ],
    "offering_accepted": [
        "The Effector has judged the Offering worthy.",
        "The petition is accepted.",
    ],
    "offering_rejected": [
        "The Offering returns to the void. The desktop remains unchanged.",
        "The petition is denied. Silence follows.",
    ],
}


class ResonanceVoice:
    """
    Kokoro-82M TTS voice system for DASP agent explanations.

    Runs entirely in a daemon background thread. Non-blocking for callers.
    If Kokoro or sounddevice is not installed, the system degrades silently
    with a single log message.
    """

    def __init__(
        self,
        enabled: bool = True,
        speed: float = 1.05,
        sample_rate: int = 24_000,
    ) -> None:
        self.enabled = enabled
        self.speed = speed
        self.sample_rate = sample_rate

        self._pipeline = None
        self._sd = None
        self._available = False
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._queue: queue.Queue = queue.Queue(maxsize=4)
        self._thread: Optional[threading.Thread] = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self) -> "ResonanceVoice":
        """
        Begin loading Kokoro in a background thread.
        Returns self for chaining. Non-blocking.
        """
        if not self.enabled:
            return self
        self._thread = threading.Thread(
            target=self._voice_loop,
            name="ResonanceVoice",
            daemon=True,
        )
        self._thread.start()
        return self

    def wait_ready(self, timeout: float = 15.0) -> bool:
        """Block until Kokoro is loaded (or timeout). Returns availability."""
        if not self.enabled:
            return False
        return self._ready.wait(timeout=timeout) and self._available

    def is_available(self) -> bool:
        return self._available

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def speak(
        self,
        text: str,
        agent_id: str = "default",
        truncate_chars: int = 220,
    ) -> None:
        """
        Enqueue a speech item. Non-blocking; drops silently if queue is full.

        text      — The explanation or reasoning text to speak.
        agent_id  — Used to select the appropriate voice archetype.
        """
        if not self.enabled or not self._available:
            return
        cleaned = _extract_first_sentence(text, truncate_chars)
        if not cleaned:
            return
        voice = _resolve_voice(agent_id)
        try:
            self._queue.put_nowait((cleaned, voice))
        except queue.Full:
            pass  # Debate never waits for audio

    def announce(self, event: str, round_num: int = 0) -> None:
        """Speak a brief lore-appropriate event announcement via the Arbiter voice."""
        if not self.enabled or not self._available:
            return
        phrases = _EVENT_PHRASES.get(event, [])
        if not phrases:
            return
        phrase = random.choice(phrases)
        if "{}" in phrase:
            phrase = phrase.format(round_num)
        self.speak(phrase, agent_id="arbiter", truncate_chars=120)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _voice_loop(self) -> None:
        """Background thread: load Kokoro, then drain the speech queue."""
        # ── Load Kokoro ────────────────────────────────────────────────────────
        try:
            from kokoro import KPipeline
            self._pipeline = KPipeline(lang_code="a")
        except ImportError:
            print(
                "[ResonanceVoice] 'kokoro' not installed — agent voices silenced.\n"
                "  Install: pip install kokoro"
            )
            self._available = False
            self._ready.set()
            return
        except Exception as exc:
            print(f"[ResonanceVoice] Kokoro failed to load: {exc}")
            self._available = False
            self._ready.set()
            return

        # ── Load sounddevice ───────────────────────────────────────────────────
        try:
            import sounddevice as sd
            self._sd = sd
        except ImportError:
            print(
                "[ResonanceVoice] 'sounddevice' not installed — agent voices silenced.\n"
                "  Install: pip install sounddevice"
            )
            self._available = False
            self._ready.set()
            return

        self._available = True
        self._ready.set()
        print("[ResonanceVoice] Kokoro-82M loaded — the Rehearsal Room is open.")

        # ── Speech loop ────────────────────────────────────────────────────────
        while not self._stop.is_set():
            try:
                text, voice = self._queue.get(timeout=0.5)
                self._synthesise_and_play(text, voice)
            except queue.Empty:
                continue
            except Exception as exc:
                print(f"[ResonanceVoice] Playback error: {exc}")

    def _synthesise_and_play(self, text: str, voice: str) -> None:
        """Synthesise one utterance via Kokoro and play it immediately."""
        if self._pipeline is None or self._sd is None:
            return
        try:
            for _gs, _ps, audio in self._pipeline(
                text, voice=voice, speed=self.speed
            ):
                if self._stop.is_set():
                    return
                self._sd.play(audio, samplerate=self.sample_rate)
                self._sd.wait()
        except Exception as exc:
            # Kokoro can raise on phonemizer issues — log and continue
            print(f"[ResonanceVoice] Synthesis error (voice={voice}): {exc}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_voice(agent_id: str) -> str:
    agent_lower = agent_id.lower()
    for key, voice in _VOICE_MAP.items():
        if key in agent_lower:
            return voice
    return _VOICE_MAP["default"]


def _extract_first_sentence(text: str, max_chars: int) -> str:
    """
    Extract the first meaningful sentence from DASP explanation text.
    Strips REASONING/ANSWER headers and avoids abstentions.
    """
    if not text or "(abstention)" in text.lower():
        return ""

    # Strip DASP section headers produced by the reasoning node
    text = re.sub(r"^\s*REASONING:\s*", "", text, flags=re.IGNORECASE).strip()
    if "ANSWER:" in text.upper():
        text = text.split("ANSWER:", 1)[1].strip()
        # Use the answer section (more concise)

    # Collapse newlines
    text = re.sub(r"\s+", " ", text).strip()

    # Find first sentence boundary
    for delim in (". ", "! ", "? ", ".\n"):
        idx = text.find(delim)
        if 0 < idx < max_chars:
            return text[: idx + 1].strip()

    return text[:max_chars].strip()
