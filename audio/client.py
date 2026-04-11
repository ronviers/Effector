"""
effector/audio/client.py
Thin client for the audio MCP service.
"""
import requests
from typing import Optional

class AudioClient:
    def __init__(self, host: str = "http://127.0.0.1:5558"):
        self.host = host
        # We will use the direct HTTP endpoints FastMCP creates
        self.api_base = f"{self.host}/tools"

    def _call_tool(self, tool_name: str, args: dict) -> dict:
        try:
            resp = requests.post(f"{self.api_base}/{tool_name}", json=args, timeout=45.0)
            if resp.status_code == 200:
                return resp.json()
        except requests.exceptions.RequestException:
            pass
        return {}

    def speak(self, text: str, agent_id: str = "default") -> None:
        self._call_tool("audio_speak", {"text": text, "agent_id": agent_id})

    def announce(self, kind: str) -> None:
        self._call_tool("audio_announce", {"kind": kind})

    def generate(self, target: str, prompt: Optional[str] = None) -> Optional[str]:
        result = self._call_tool("audio_generate", {"target": target, "prompt": prompt})
        return result.get("path")