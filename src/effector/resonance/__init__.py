"""
effector.resonance — The Acoustic Layer of the Worldspillage
=============================================================

Two sub-systems, one theological purpose:

  ResonanceVoice  (The Rehearsal Room)
      Kokoro-82M TTS. Agents speak their explanations aloud during
      DASP deliberation. The player hears the earnest Acolytes and
      the stern Arbiter debating the thermodynamic merits of desktop
      organisation. Voice runs in a background thread — it never
      blocks the deliberation loop.

  ResonanceWorld  (The Sacred Interval + Manifestation)
      AudioLDM 2 text-to-audio generation. When an Intention Envelope
      receives ACK, the Effector generates a bespoke sound for the
      action *before* executing it. The diffusion process is exposed
      to the player as liturgy: the universe physically bending.
      NACK = silence. The speakers remain dead. The desktop stays messy.

Installation:
    pip install kokoro sounddevice soundfile scipy          # voices
    pip install diffusers transformers accelerate           # world sounds

    # Windows phoneme backend for Kokoro (optional but recommended):
    # Download espeak-ng from https://github.com/espeak-ng/espeak-ng/releases
    # then: pip install phonemizer
"""

from effector.resonance.resonance_voice import ResonanceVoice
from effector.resonance.resonance_world import ResonanceWorld

__all__ = ["ResonanceVoice", "ResonanceWorld"]


def create_resonance_system(
    voice_enabled: bool = True,
    world_enabled: bool = True,
    on_sacred_interval_progress=None,
) -> tuple["ResonanceVoice", "ResonanceWorld"]:
    """
    Factory: create both resonance subsystems.

    on_sacred_interval_progress: callable(step: int, total: int, text: str)
        Called during AudioLDM 2 diffusion steps. Use to update the UI.
    """
    voice = ResonanceVoice(enabled=voice_enabled)
    world = ResonanceWorld(
        enabled=world_enabled,
        on_progress=on_sacred_interval_progress,
    )
    return voice, world
