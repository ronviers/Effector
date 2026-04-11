"""
ollama_adapter.py — Ollama-backed DASP Agent Adapter
=====================================================

OllamaAgent is the public-facing agent type for the DASPCoordinator's
agent_registry. It now delegates all LLM work to TwoPhaseOllamaAgent,
maintaining a fully backwards-compatible API surface.

Architecture change (DASP §4.3):
    The previous single-call approach combined reasoning and signal
    serialization in one LLM call using an instructional JSON prompt.
    That approach is prohibited by DASP §4.3 (the Prohibited Pattern:
    "instructional text such as 'Output only a valid JSON object'").

    OllamaAgent now orchestrates a two-phase pipeline:
      Phase 1 — Reasoning Node: pure cognitive prompt, free-form output.
      Phase 2 — Characterizer:  tool calling to extract mathematical signals.
    Python assembles the final AgentResponse from validated tool arguments
    and injected infrastructure fields. No LLM writes a hash or a round number.

ToolRegistry is retained unchanged for declaring MCP-layer debate tools
(capabilities an agent may request during reasoning context injection).
It is no longer passed into the LLM system prompt — that was the source
of the §4.3 violation. Tool declarations live here for the orchestration
layer to act on; the LLM never sees them.
"""
from __future__ import annotations

from typing import Any

from effector.schemas.dasp import AgentRequest, AgentResponse
from effector.adapters.two_phase_adapter import TwoPhaseOllamaAgent, make_two_phase_callable


# ─────────────────────────────────────────────────────────────────────────────
# Tool Registry
# ─────────────────────────────────────────────────────────────────────────────

class ToolDefinition:
    """Declares a capability the agent may request via MCP during deliberation."""

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
    Declares the MCP tools available to agents during the debate orchestration.

    NOTE: These tool declarations are consumed by the Orchestration Layer
    (goal-tree decomposition, context injection) before the Reasoning Node
    prompt is constructed. They are NOT injected into any LLM system prompt.
    If an agent needs to retrieve information, that retrieval happens at the
    orchestration layer; the result is passed as context to the Reasoning Node.

    If the debate's conclusion requires spawning a sub-task, the Coordinator
    wraps the final_answer in an IEP envelope with verb=DELEGATE. Agents do
    not issue DELEGATE themselves.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        self._tools[name] = ToolDefinition(name, description, parameters or {})

    def list_tools(self) -> list[dict]:
        return [t.to_dict() for t in self._tools.values()]

    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={self.tool_names()})"


# ─────────────────────────────────────────────────────────────────────────────
# OllamaAgent — backwards-compatible public API
# ─────────────────────────────────────────────────────────────────────────────

class OllamaAgent:
    """
    Backwards-compatible DASP agent that delegates to TwoPhaseOllamaAgent.

    Existing call sites continue to work unchanged:
        agent = OllamaAgent("analyst", tool_registry, model="qwen2.5:32b")
        response = agent(request)

    Parameters
    ----------
    agent_id : str
        Logical identifier. Used for persona selection and logging.
    tool_registry : ToolRegistry | None
        Declares MCP capabilities. Consumed by the orchestration layer;
        not injected into any LLM prompt.
    model : str
        Ollama model for the Reasoning Node (Phase 1).
    characterizer_model : str | None
        Ollama model for the Characterizer (Phase 2). Defaults to `model`.
        A smaller model (e.g. "mistral:7b") works well here.
    host : str
        Ollama API base URL.
    temperature : float
        Temperature for the Reasoning Node. Characterizer always runs at 0.0.
    timeout_s : float
        Per-phase network timeout in seconds.
    """

    def __init__(
        self,
        agent_id: str,
        tool_registry: ToolRegistry | None = None,
        model: str = "qwen2.5:32b",
        characterizer_model: str | None = None,
        host: str = "http://127.0.0.1:11434",
        temperature: float = 0.7,
        timeout_s: float = 90.0,
    ) -> None:
        self.agent_id = agent_id
        self._tool_registry = tool_registry or ToolRegistry()
        self._model = model
        self._host = host
        self._temperature = temperature

        self._delegate = TwoPhaseOllamaAgent(
            agent_id=agent_id,
            reasoning_model=model,
            characterizer_model=characterizer_model,
            host=host,
            reasoning_temperature=temperature,
            timeout_s=timeout_s,
        )

    def __call__(self, request: AgentRequest) -> AgentResponse:
        """Execute the two-phase pipeline and return a validated AgentResponse."""
        return self._delegate(request)

    @property
    def tool_registry(self) -> ToolRegistry:
        """Access the agent's declared MCP capabilities."""
        return self._tool_registry

    def __repr__(self) -> str:
        return (
            f"OllamaAgent(id={self.agent_id!r}, model={self._model!r}, "
            f"tools={self._tool_registry.tool_names()})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# make_agent_callable — convenience factory (backwards-compatible)
# ─────────────────────────────────────────────────────────────────────────────

def make_agent_callable(
    agent_id: str,
    tool_registry: ToolRegistry | None = None,
    model: str = "qwen2.5:32b",
    characterizer_model: str | None = None,
    temperature: float = 0.7,
    host: str = "http://127.0.0.1:11434",
    timeout_s: float = 90.0,
) -> OllamaAgent:
    """
    Convenience factory for creating a two-phase DASP-compliant agent.

    Usage::

        registry = {
            "analyst": make_agent_callable("analyst", model="qwen2.5:32b"),
            "skeptic": make_agent_callable("skeptic", model="mistral:7b"),
        }
    """
    return OllamaAgent(
        agent_id=agent_id,
        tool_registry=tool_registry,
        model=model,
        characterizer_model=characterizer_model,
        host=host,
        temperature=temperature,
        timeout_s=timeout_s,
    )
