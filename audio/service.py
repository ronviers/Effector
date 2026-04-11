"""
effector/audio/service.py
FastMCP server for the Resonance Layer (Kokoro TTS + AudioLDM 2).
"""
from mcp.server.fastmcp import FastMCP
from typing import Optional

# Assuming these were moved into effector/audio during the flattening.
# If they are still in effector/resonance, update the imports accordingly.
try:
    from effector.resonance.resonance_voice import ResonanceVoice
    from effector.resonance.resonance_world import ResonanceWorld
    _AUDIO_AVAILABLE = True
except ImportError:
    _AUDIO_AVAILABLE = False

mcp = FastMCP("effector-audio")

_voice: Optional[ResonanceVoice] = None
_world: Optional[ResonanceWorld] = None

@mcp.tool()
def audio_speak(text: str, agent_id: str = "default") -> None:
    """Enqueue text for Kokoro TTS synthesis and playback."""
    if _voice and _voice.is_available():
        _voice.speak(text, agent_id=agent_id)

@mcp.tool()
def audio_announce(kind: str) -> None:
    """Play a named event sound: consensus, inhibition, offering_accepted, etc."""
    if _voice and _voice.is_available():
        _voice.announce(kind)

@mcp.tool()
def audio_generate(target: str, prompt: Optional[str] = None) -> dict:
    """
    Generate a world sound for a given IEP action target using AudioLDM2.
    Blocks until generation is complete. Returns path to generated audio file.
    """
    if _world and _world.is_available():
        # The ResonanceWorld block handles generation and playback
        path = _world.generate_and_play(target)
        return {"status": "success", "path": path}
    return {"status": "unavailable", "path": None}

@mcp.tool()
def audio_status() -> dict:
    """Return current state: models loaded, availability."""
    return {
        "voice_available": _voice.is_available() if _voice else False,
        "world_available": _world.is_available() if _world else False,
        "audio_layer_installed": _AUDIO_AVAILABLE
    }

def _startup():
    """Initialize the heavy models when the service starts."""
    global _voice, _world
    if _AUDIO_AVAILABLE:
        print("Initializing ResonanceVoice...")
        _voice = ResonanceVoice(enabled=True)
        _voice.start()
        
        print("Initializing ResonanceWorld...")
        _world = ResonanceWorld(enabled=True)
        _world.start()

# Hook the startup into FastMCP's lifespan (FastMCP handles this differently depending on transport, 
# but for a basic script run, we can just call it before starting).