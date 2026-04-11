"""
DASP-1.0 — Debate-as-a-Service Protocol Schemas
Strict Pydantic v2 models for all DASP message types.
"""
from __future__ import annotations
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, field_validator, model_validator


class ConsensusMode(str, Enum):
    unanimous = "unanimous"
    weighted = "weighted"
    quorum = "quorum"
    signal_superposition = "signal_superposition"

class TriggerType(str, Enum):
    inhibition_gate = "inhibition_gate"
    stall_gate = "stall_gate"
    hash_divergence = "hash_divergence"
    swap = "swap"
    copy = "copy"

class TriggerAction(str, Enum):
    abort_and_replan = "ABORT_AND_REPLAN"
    abort_and_escalate = "ABORT_AND_ESCALATE"
    optimize = "OPTIMIZE"
    diversity_injection = "DIVERSITY_INJECTION"
    re_snapshot = "RE_SNAPSHOT"

class HypothesisFinalState(str, Enum):
    consensus = "consensus"
    phase_canceled = "phase_canceled"
    stalled = "stalled"
    abandoned = "abandoned"

class GoalContext(BaseModel):
    root_goal_id: uuid.UUID
    parent_goal_id: uuid.UUID
    depth: int = Field(ge=0)
    branch_label: str = ""

class AgentInfo(BaseModel):
    id: str
    endpoint: str | None = None
    capabilities: list[str] = Field(default_factory=list)

class DebateRules(BaseModel):
    max_rounds: int = Field(ge=1)
    consensus_mode: ConsensusMode = ConsensusMode.signal_superposition
    theta_consensus: float = Field(default=0.7, ge=0.0, le=1.0)
    tau_suppression: float = Field(default=0.5, ge=0.0, le=1.0)
    epsilon_stall: float = Field(default=0.05, ge=0.0)
    epsilon_continue: float = Field(default=0.1, ge=0.0)
    epsilon_replan: float = Field(default=0.3, ge=0.0)
    epsilon_escalate: float = Field(default=0.6, ge=0.0)

    @model_validator(mode="after")
    def check_epsilon_ordering(self) -> "DebateRules":
        if not (self.epsilon_continue < self.epsilon_replan < self.epsilon_escalate):
            raise ValueError("epsilon thresholds must satisfy: epsilon_continue < epsilon_replan < epsilon_escalate")
        return self

class AgentSignal(BaseModel):
    confidence: float = Field(ge=0.0, le=1.0)
    polarity: int = Field(description="Strictly -1, 0, or 1")
    generative_strength: float = Field(default=0.0, ge=0.0)
    inhibitory_pressure: float = Field(default=0.0, ge=0.0)

    @field_validator("polarity")
    @classmethod
    def polarity_must_be_ternary(cls, v: int) -> int:
        if v not in (-1, 0, 1):
            raise ValueError(f"polarity must be -1, 0, or 1; got {v}")
        return v

    @model_validator(mode="after")
    def set_strength_from_polarity(self) -> "AgentSignal":
        if self.polarity >= 0 and self.generative_strength == 0.0:
            self.generative_strength = self.confidence
        if self.polarity <= 0 and self.inhibitory_pressure == 0.0:
            self.inhibitory_pressure = self.confidence
        return self

class ExpectedStateChange(BaseModel):
    keys_affected: list[str] = Field(default_factory=list)
    predicted_delta: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

class DebateStart(BaseModel):
    type: str = "debate.start"
    session_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    task: str
    snapshot_hash: str = Field(min_length=64, max_length=64)
    snapshot_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agents: list[AgentInfo]
    goal_context: GoalContext
    rules: DebateRules
    reputation_scope: str = "global"
    vectorized_bus: bool = False

    @field_validator("agents")
    @classmethod
    def at_least_one_agent(cls, v: list[AgentInfo]) -> list[AgentInfo]:
        if len(v) < 1:
            raise ValueError("A DASP session requires at least one agent")
        return v

class AgentSummary(BaseModel):
    agent_id: str
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    polarity: int

    @field_validator("polarity")
    @classmethod
    def polarity_ternary(cls, v: int) -> int:
        if v not in (-1, 0, 1):
            raise ValueError("polarity must be -1, 0, or 1")
        return v

class AgentRequest(BaseModel):
    type: str = "agent.request"
    session_id: uuid.UUID
    round: int = Field(ge=1)
    mode: str = Field(default="initial", pattern="^(initial|debate)$")
    task: str
    snapshot_hash: str = Field(min_length=64, max_length=64)
    self_state: AgentSummary | None = Field(default=None, alias="self")
    others: list[AgentSummary] = Field(default_factory=list)
    model_config = {"populate_by_name": True}

class AgentResponse(BaseModel):
    type: str = "agent.response"
    session_id: uuid.UUID
    agent_id: str
    round: int = Field(ge=1)
    snapshot_hash: str = Field(min_length=64, max_length=64)
    hypothesis_id: str
    answer: str
    answer_hash: str = Field(max_length=64)
    signal: AgentSignal
    expected_state_change: ExpectedStateChange = Field(default_factory=ExpectedStateChange)
    explanation: str = ""

    @field_validator("answer_hash")
    @classmethod
    def answer_hash_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("answer_hash must not be blank")
        return v

class TriggerFire(BaseModel):
    type: str = "trigger.fire"
    session_id: uuid.UUID
    round: int
    trigger: TriggerType
    action: TriggerAction
    hypothesis_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class DebateOptimize(BaseModel):
    type: str = "debate.optimize"
    session_id: uuid.UUID
    triggered_by: str
    input: dict[str, Any]
    output: dict[str, Any] = Field(default_factory=dict)

class HypothesisSignal(BaseModel):
    S_g: float
    S_i: float
    S_net: float
    final_state: HypothesisFinalState

class SignalTrace(BaseModel):
    per_hypothesis: dict[str, HypothesisSignal] = Field(default_factory=dict)

class DebateResult(BaseModel):
    session_id: uuid.UUID
    final_answer: str
    consensus_score: float = Field(ge=0.0, le=1.0)
    rounds: int = Field(ge=0)
    snapshot_hash: str = Field(min_length=64, max_length=64)
    snapshot_timestamp: datetime
    agents: list[str]
    winning_coalition: list[str]
    disagreement_score: float = Field(default=0.0, ge=0.0, le=1.0)
    signal_trace: SignalTrace = Field(default_factory=SignalTrace)
    trace: dict[str, Any] = Field(default_factory=dict)
    terminated_reason: str = "consensus"
