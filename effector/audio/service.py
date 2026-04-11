from mcp.server.fastmcp import FastMCP

mcp = FastMCP("effector-audio")

@mcp.tool()
def audio_speak(text: str, agent_id: str = "default") -> dict:
    """Enqueue text for Kokoro TTS synthesis and playback."""
    return {"status": "stub"}

@mcp.tool()
def audio_announce(kind: str) -> dict:
    """Play a named event sound: consensus, inhibition, offering_accepted, etc."""
    return {"status": "stub"}

@mcp.tool()
def audio_generate(target: str, prompt: str | None = None) -> dict:
    """Generate a world sound for a given IEP action target using AudioLDM2."""
    return {"status": "stub", "path": "stub.wav"}

@mcp.tool()
def audio_status() -> dict:
    """Return current state: models loaded, queue depth, VRAM usage."""
    return {"status": "stub"}
