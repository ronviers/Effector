from dataclasses import dataclass
from typing import Any

class AudioClient:
    def speak(self, text: str, agent_id: str = "default") -> None: pass
    def announce(self, kind: str) -> None: pass
    def generate(self, target: str, prompt: str | None = None) -> dict: 
        return {"status": "stub"}

class RegistryClient:
    def lookup_entity(self, sym: str) -> dict: 
        return {"status": "stub"}
    def search_entities(self, **kwargs) -> list: 
        return []

class EngineClient:
    def deliberate(self, task: str, context: dict | None = None, mode: str = "SUPERVISED") -> dict:
        return {"status": "stub", "consensus_score": 0.9, "envelope_id": "stub-123"}
    def ack(self, envelope_id: str) -> dict:
        return {"status": "stub"}

@dataclass
class Services:
    audio: AudioClient
    registry: RegistryClient
    engine: EngineClient
    telemetry: Any = None
    bus: Any = None
    ui: Any = None

def wire_services(cfg: dict) -> Services:
    return Services(
        audio=AudioClient(),
        registry=RegistryClient(),
        engine=EngineClient()
    )

def build_task(state: dict) -> str:
    return "Stub task"
