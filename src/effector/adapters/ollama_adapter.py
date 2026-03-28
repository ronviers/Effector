"""
LLM Adapter — wraps the local Ollama API in a strict system prompt
that enforces DASP-compliant JSON output.

Provides:
  - OllamaAgent: callable that takes AgentRequest, returns AgentResponse
  - ToolRegistry: declares tools available to agents
  - make_agent_callable: convenience factory
"""

from __future__ import annotations

import hashlib
import json
import requests
from typing import Any

from effector.schemas.dasp import (
    AgentResponse,
    AgentRequest,
    AgentSignal,
    ExpectedStateChange,
)


# ─────────────────────────────────────────────
# Tool registry
# ─────────────────────────────────────────────

class ToolDefinition:
    def __init__(self, name: str, description: str, parameters: dict[str, Any]) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    """
    Declares the tools available to agents.
    Tool names are passed to the LLM in the system prompt.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, name: str, description: str, parameters: dict[str, Any] | None = None) -> None:
        self._tools[name] = ToolDefinition(name, description, parameters or {})

    def list_tools(self) -> list[dict]:
        return [t.to_dict() for t in self._tools.values()]

    def tool_names(self) -> list[str]:
        return list(self._tools.keys())


# ─────────────────────────────────────────────
# Ollama adapter
# ─────────────────────────────────────────────

DASP_SYSTEM_PROMPT = """
You are a reasoning agent participating in a structured multi-agent debate (DASP-1.0).

## Your only output format

You MUST respond with a single JSON object. No markdown, no preamble, no code blocks.
The JSON must match this exact schema:

{{
  "type": "agent.response",
  "session_id": "{session_id}",
  "agent_id": "{agent_id}",
  "round": {round},
  "snapshot_hash": "{snapshot_hash}",
  "hypothesis_id": "<unique short id for your hypothesis, e.g. H1>",
  "answer": "<your answer to the task>",
  "answer_hash": "<first 16 chars of sha256 of your answer>",
  "signal": {{
    "confidence": <float 0.0 to 1.0>,
    "polarity": <integer: 1 for support, 0 for neutral, -1 for oppose>,
    "generative_strength": <float, how strongly you support this hypothesis>,
    "inhibitory_pressure": <float, how strongly you oppose competing hypotheses>
  }},
  "expected_state_change": {{
    "keys_affected": ["<list of world-state keys your answer will change>"],
    "predicted_delta": {{}},
    "confidence": <float 0.0 to 1.0>
  }},
  "explanation": "<your reasoning, plain text>"
}}

## Rules
- confidence: always between 0.0 and 1.0 (inclusive)
- polarity: MUST be exactly -1, 0, or 1. No other values.
- If you agree with the majority position, set polarity=1
- If you disagree or see a serious flaw, set polarity=-1 and explain in explanation
- If you are uncertain, set polarity=0
- snapshot_hash: copy exactly from the task header
- hypothesis_id: use the same ID across rounds if defending the same hypothesis

## Available tools (declare intent only — do not execute)
{tools}

## Task
{task}
"""


class OllamaAgent:
    """
    Wraps the local Ollama API to produce DASP-compliant AgentResponse objects.
    """

    def __init__(
        self,
        agent_id: str,
        tool_registry: ToolRegistry | None = None,
        model: str = "qwen2.5-coder:32b",
        host: str = "http://127.0.0.1:11434",
        temperature: float = 0.7,
    ) -> None:
        self.agent_id = agent_id
        self._tool_registry = tool_registry or ToolRegistry()
        self._model = model
        self._host = host
        self._temperature = temperature

    def __call__(self, request: AgentRequest) -> AgentResponse:
        """Call the Ollama API and parse the DASP-compliant response."""
        tools_desc = json.dumps(self._tool_registry.list_tools(), indent=2) or "None"
        task_with_others = self._build_task_text(request)

        system_prompt = DASP_SYSTEM_PROMPT.format(
            session_id=str(request.session_id),
            agent_id=self.agent_id,
            round=request.round,
            snapshot_hash=request.snapshot_hash,
            tools=tools_desc,
            task=task_with_others,
        )

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task_with_others}
            ],
            "format": "json",
            "stream": False,
            "options": {
                "temperature": self._temperature
            }
        }

        try:
            resp = requests.post(f"{self._host}/api/chat", json=payload, timeout=120)
            resp.raise_for_status()
            raw = resp.json().get("message", {}).get("content", "")
            return self._parse_response(raw, request)
        except Exception as exc:
            print(f"[EFFECTOR] {self.agent_id} Ollama routing failed: {exc}")
            return self._fallback_response(request, f"Routing error: {exc}")

    def _build_task_text(self, request: AgentRequest) -> str:
        lines = [f"Task: {request.task}"]
        if hasattr(request, 'others') and request.others:
            lines.append("\n--- Other agents' current positions ---")
            for other in request.others:
                polarity_label = {1: "SUPPORT", 0: "NEUTRAL", -1: "OPPOSE"}.get(other.polarity, "?")
                lines.append(
                    f"  Agent {other.agent_id}: [{polarity_label}, confidence={other.confidence:.2f}] {other.answer}"
                )
        return "\n".join(lines)

    def _parse_response(self, raw: str, request: AgentRequest) -> AgentResponse:
        """Parse LLM output into a validated AgentResponse. Falls back gracefully."""
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            print(f"[EFFECTOR] {self.agent_id} returned invalid JSON: {exc}")
            return self._fallback_response(request, f"JSON parse error: {exc}")

        polarity = data.get("signal", {}).get("polarity", 0)
        if polarity not in (-1, 0, 1):
            print(f"[EFFECTOR] {self.agent_id} returned invalid polarity {polarity!r}, clamping to 0")
            polarity = 0

        confidence = float(data.get("signal", {}).get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        answer = data.get("answer", "")
        answer_hash = data.get("answer_hash", "") or hashlib.sha256(answer.encode()).hexdigest()[:16]

        try:
            return AgentResponse(
                session_id=request.session_id,
                agent_id=self.agent_id,
                round=request.round,
                snapshot_hash=request.snapshot_hash,
                hypothesis_id=data.get("hypothesis_id", f"H_{self.agent_id}"),
                answer=answer,
                answer_hash=answer_hash,
                signal=AgentSignal(
                    confidence=confidence,
                    polarity=polarity,
                    generative_strength=float(data.get("signal", {}).get("generative_strength", confidence)),
                    inhibitory_pressure=float(data.get("signal", {}).get("inhibitory_pressure", 0.0)),
                ),
                expected_state_change=ExpectedStateChange(
                    **data.get("expected_state_change", {})
                ) if data.get("expected_state_change") else ExpectedStateChange(),
                explanation=data.get("explanation", ""),
            )
        except Exception as exc:
            print(f"[EFFECTOR] {self.agent_id} response validation failed: {exc}")
            return self._fallback_response(request, str(exc))

    def _fallback_response(self, request: AgentRequest, reason: str) -> AgentResponse:
        """Abstention response per DASP §10 error handling."""
        return AgentResponse(
            session_id=request.session_id,
            agent_id=self.agent_id,
            round=request.round,
            snapshot_hash=request.snapshot_hash,
            hypothesis_id=f"H_{self.agent_id}_abstain",
            answer="(abstention)",
            answer_hash="abstain00000000",
            signal=AgentSignal(
                confidence=0.0,
                polarity=0,
                generative_strength=0.0,
                inhibitory_pressure=0.0,
            ),
            explanation=f"Abstention — {reason}",
        )


def make_agent_callable(
    agent_id: str,
    tool_registry: ToolRegistry | None = None,
    temperature: float = 0.7,
) -> OllamaAgent:
    """Convenience factory for creating an Ollama-backed DASP agent."""
    return OllamaAgent(agent_id=agent_id, tool_registry=tool_registry, temperature=temperature)