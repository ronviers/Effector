"""
resonance_world.py — The Sacred Interval & Manifestation
==========================================================

When The Effector grants an ACK, this module generates a unique,
bespoke audio signature for the specific action being taken. The
generation process is not hidden. It is the liturgy.

The player watches the diffusion steps count upward in the UI:
    ⚖️  THE EFFECTOR WEIGHS THE OFFERING
    [██████████░░░░░░░░░░] Step 25/50 — the universe bends

This is intentional. The 30-45 second wait is not a bug to apologise
for. It is the physics of miracle-work, made visible.

NACK: AudioLDM 2 is never invoked. The speakers remain dead silent.
      The desktop stays messy. The agents, on their next polling cycle,
      will notice the state hasn't changed and begin a confused, whispered
      new debate about why the universe ignored them.

Model: cvssp/audioldm2
  ~6.5 GB download on first use.
  GPU (RTX 4080 SUPER): ~15-25 seconds per generation.
  CPU: 45-90 seconds per generation (the full liturgical experience).
  enable_model_cpu_offload() is used on CUDA systems — weights live on
  CPU, compute runs on GPU, keeping VRAM available for the LLM agents.

Installation
------------
    pip install diffusers transformers accelerate sounddevice soundfile scipy
    pip install torch torchvision torchaudio  # if not already installed
"""
from __future__ import annotations

import os
import tempfile
import threading
import time
from typing import Callable, Optional

# ── Action → Prompt mapping ────────────────────────────────────────────────────
# Each entry maps an IEP action target (or prefix) to a TTA prompt.
# Prompts are written to evoke texture, weight, and warmth — not just sound.

_ACTION_PROMPTS: dict[str, str] = {
    "os.filesystem.zen_habitat": (
        "The soft satisfying thud of books being stacked neatly one by one, "
        "papers shuffling into perfect alignment, "
        "a gentle click of a binder closing, "
        "quiet warm ambient hum of an organized space settling into order"
    ),
    "os.filesystem.desktop_icons": (
        "Small smooth wooden tiles sliding across a polished surface with soft taps, "
        "delicate mechanical clicks of items snapping into place, "
        "the quiet sound of order being restored, "
        "a final gentle exhale of completion"
    ),
    "os.filesystem.move": (
        "A folder sliding gently across a desk, papers settling, "
        "soft ambient sound of careful organization, "
        "a gentle tap as something finds its home"
    ),
    "os.filesystem.archive": (
        "The heavy satisfying thud of a thick book being closed with care, "
        "leather binding creaking once, pages settling, "
        "quiet library ambience, dust motes in warm light"
    ),
    "desktop.overlay.glimmer": (
        "A single tiny crystalline bell chime, pure and clear, "
        "followed by the lightest imaginable footstep on aged wood, "
        "a soft intake of breath, the sound of something small arriving"
    ),
    "desktop.overlay.glimmer.music": (
        "A single warm piano note resonating in a small room, "
        "gentle harmonics unfolding, "
        "fading into comfortable reverberant silence"
    ),
    "desktop.overlay.glimmer.librarian": (
        "A quiet page turn, the soft thud of a book being placed precisely, "
        "a gentle approving hum, the scratch of a tiny pencil"
    ),
    "desktop.overlay.ambient.campfire": (
        "Dry wood crackling gently in a fireplace, "
        "warm soft pops and quiet hiss of burning kindling, "
        "the deep ambient warmth of a small contained fire"
    ),
    "desktop.overlay": (
        "A soft translucent sound, like a veil settling over still water, "
        "gentle ambient shimmer"
    ),
    "desktop.ambient_sync": (
        "Rain tapping softly and steadily on a window pane, "
        "the gentle creak of a warm house settling, "
        "muffled thunder in the far distance, cozy indoor silence"
    ),
    "os.display.brightness": (
        "A smooth analog dimmer switch being gently turned, "
        "electrical hum fading gradually, "
        "warm comfortable silence descending like a blanket"
    ),
    "rat_store.store": (
        "A wax seal being pressed into warm wax, a decisive soft thud, "
        "a quill briefly scratching parchment, "
        "official, ceremonial, final"
    ),
    "rat_store": (
        "The soft click of a lock engaging, quiet and precise"
    ),
    "state_bus.reputation": (
        "A gentle resonant tone, like a small glass bowl being struck once, "
        "harmonics fading cleanly"
    ),
    "world_state": (
        "A warm resonant wooden bowl struck once with a soft padded mallet, "
        "a single full harmonic tone expanding slowly outward and fading "
        "into complete comfortable stillness"
    ),
}

_DEFAULT_PROMPT = (
    "A warm soft resonant tone, like a small wooden bowl being struck gently, "
    "single clear harmonic, fading into comfortable silence"
)


class ResonanceWorld:
    """
    AudioLDM 2 text-to-audio generation for IEP manifestation events.

    Lifecycle
    ---------
    1. Call start() — begins loading AudioLDM 2 in a background thread.
    2. Call generate_and_play(target) from the IEP queue consumer thread.
       This BLOCKS until generation completes and audio has played.
       on_progress callbacks fire during diffusion for UI updates.
    3. The OS action executes AFTER this method returns.

    Graceful degradation
    --------------------
    If diffusers is not installed, or the model fails to load, is_available()
    returns False and generate_and_play() returns None immediately.
    The OS action still executes — the Sacred Interval is simply silent.
    """

    def __init__(
        self,
        enabled: bool = True,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
        num_steps: int = 50,
        audio_length_s: float = 4.0,
        model_id: str = "cvssp/audioldm2",
    ) -> None:
        self.enabled = enabled
        self.on_progress = on_progress   # callback(step, total, status_text)
        self.num_steps = num_steps
        self.audio_length_s = audio_length_s
        self.model_id = model_id

        self._pipeline = None
        self._available = False
        self._ready = threading.Event()
        self._load_lock = threading.Lock()

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self) -> "ResonanceWorld":
        """
        Begin loading AudioLDM 2 in a background thread. Non-blocking.
        Returns self for chaining.
        """
        if not self.enabled:
            self._ready.set()
            return self
        t = threading.Thread(
            target=self._load_pipeline,
            name="ResonanceWorldLoader",
            daemon=True,
        )
        t.start()
        return self

    def wait_ready(self, timeout: float = 120.0) -> bool:
        """Block until the model is loaded. Returns True if available."""
        return self._ready.wait(timeout=timeout) and self._available

    def is_available(self) -> bool:
        return self._available

    def generate_and_play(self, target: str) -> Optional[str]:
        """
        Generate bespoke audio for the given IEP action target and play it.

        BLOCKING — returns only after audio has played.
        Calls self.on_progress during diffusion steps.

        Returns the path to the generated WAV file, or None on failure.
        The caller is responsible for cleaning up temp files (or leaving them
        as receipts of past miracles).
        """
        if not self.enabled or not self._available or self._pipeline is None:
            return None

        prompt = _resolve_prompt(target)
        return self._generate(prompt, target)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _load_pipeline(self) -> None:
        with self._load_lock:
            if self._pipeline is not None:
                return
            try:
                import torch
                from diffusers import AudioLDM2Pipeline

                _emit(self.on_progress, 0, 1,
                      f"Loading {self.model_id} from HuggingFace cache...")

                has_cuda = torch.cuda.is_available()
                dtype = torch.float16 if has_cuda else torch.float32

                self._pipeline = AudioLDM2Pipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=dtype,
                )

                if has_cuda:
                    # CPU offload: model weights on RAM, compute on GPU.
                    # Per lore spec: "running via HuggingFace's CPU-offload"
                    self._pipeline.enable_model_cpu_offload()
                    device_label = f"CUDA (RTX offloaded)"
                else:
                    self._pipeline = self._pipeline.to("cpu")
                    device_label = "CPU (the full liturgical experience)"

                self._available = True
                _emit(self.on_progress, 1, 1,
                      f"AudioLDM 2 ready — {device_label}.")
                print(f"[ResonanceWorld] AudioLDM 2 loaded on {device_label}.")

            except ImportError:
                print(
                    "[ResonanceWorld] 'diffusers' not installed — world sounds silenced.\n"
                    "  Install: pip install diffusers transformers accelerate"
                )
                self._available = False
            except Exception as exc:
                print(f"[ResonanceWorld] Failed to load AudioLDM 2: {exc}")
                self._available = False
            finally:
                self._ready.set()

    def _generate(self, prompt: str, target: str) -> Optional[str]:
        """Run the diffusion pipeline and play the result."""
        try:
            import numpy as np
            import scipy.io.wavfile
        except ImportError:
            print("[ResonanceWorld] scipy/numpy not available — cannot save audio.")
            return None

        on_prog = self.on_progress
        total = self.num_steps
        step_tracker = [0]

        def _step_cb(pipe, step: int, timestep, callback_kwargs: dict) -> dict:
            step_tracker[0] = step + 1
            filled = int(20 * step_tracker[0] / total)
            bar = "█" * filled + "░" * (20 - filled)
            _emit(
                on_prog,
                step_tracker[0],
                total,
                f"[{bar}] Step {step_tracker[0]}/{total} — the universe bends",
            )
            return callback_kwargs

        _emit(on_prog, 0, total,
              f"⚖️  Petitioning The Effector for: {target}")
        time.sleep(0.1)
        _emit(on_prog, 0, total,
              f"⚖️  Diffusing reality... prompt received.")

        try:
            # Try modern diffusers callback API (>=0.25)
            result = self._pipeline(
                prompt,
                num_inference_steps=total,
                audio_length_in_s=self.audio_length_s,
                callback_on_step_end=_step_cb,
            )
        except TypeError:
            # Fallback: older diffusers without callback_on_step_end
            _emit(on_prog, 0, total,
                  "⚖️  Generating... (step tracking unavailable in this diffusers version)")
            try:
                result = self._pipeline(
                    prompt,
                    num_inference_steps=total,
                    audio_length_in_s=self.audio_length_s,
                )
            except Exception as exc:
                print(f"[ResonanceWorld] Generation failed: {exc}")
                _emit(on_prog, 0, 0, f"Generation failed: {exc}")
                return None
        except Exception as exc:
            print(f"[ResonanceWorld] Generation failed: {exc}")
            _emit(on_prog, 0, 0, f"Generation failed: {exc}")
            return None

        audio_data = result.audios[0]  # numpy float32, 16000 Hz

        # Save to a named temp file so we have a receipt of the miracle
        try:
            tmp = tempfile.NamedTemporaryFile(
                suffix=".wav",
                delete=False,
                prefix="effector_manifestation_",
                dir=tempfile.gettempdir(),
            )
            tmp_path = tmp.name
            tmp.close()

            # Convert float32 → int16 for WAV
            audio_int16 = (
                np.clip(audio_data, -1.0, 1.0) * 32767
            ).astype(np.int16)
            scipy.io.wavfile.write(tmp_path, 16_000, audio_int16)

        except Exception as exc:
            print(f"[ResonanceWorld] Failed to save audio: {exc}")
            return None

        _emit(on_prog, total, total,
              "✦ Manifestation rendered. Reality bends to receive it.")
        time.sleep(0.3)

        # Play the generated audio
        _play_wav(tmp_path)
        return tmp_path


# ── Module-level helpers ───────────────────────────────────────────────────────

def _resolve_prompt(target: str) -> str:
    """Find the best-matching TTA prompt for an IEP action target."""
    if target in _ACTION_PROMPTS:
        return _ACTION_PROMPTS[target]
    # Prefix-based fallback
    for key, prompt in _ACTION_PROMPTS.items():
        if target.startswith(key):
            return prompt
    # Category-based fallback (first dotted component)
    category = target.split(".")[0] if "." in target else target
    for key, prompt in _ACTION_PROMPTS.items():
        if key.startswith(category):
            return prompt
    return _DEFAULT_PROMPT


def _emit(
    callback: Optional[Callable[[int, int, str], None]],
    step: int,
    total: int,
    text: str,
) -> None:
    """Fire progress callback safely."""
    if callback is not None:
        try:
            callback(step, total, text)
        except Exception:
            pass


def _play_wav(path: str) -> None:
    """
    Play a WAV file. Tries sounddevice first (clean blocking playback),
    then falls back to the OS default media player.
    """
    try:
        import sounddevice as sd
        import scipy.io.wavfile
        sr, data = scipy.io.wavfile.read(path)
        if data.dtype != "float32":
            import numpy as np
            data = data.astype("float32") / 32768.0
        sd.play(data, samplerate=sr)
        sd.wait()
        return
    except ImportError:
        pass
    except Exception as exc:
        print(f"[ResonanceWorld] sounddevice playback failed: {exc}")

    # OS fallback (Windows: start; macOS: afplay; Linux: aplay)
    try:
        import subprocess, sys
        if sys.platform == "win32":
            os.startfile(path)
            time.sleep(6.0)
        elif sys.platform == "darwin":
            subprocess.run(["afplay", path], check=False)
        else:
            subprocess.run(["aplay", path], check=False)
    except Exception as exc:
        print(f"[ResonanceWorld] OS fallback playback failed: {exc}")
