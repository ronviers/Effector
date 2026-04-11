"""IEP-1.0 — Intention Envelope Protocol Schemas"""
from __future__ import annotations
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, model_validator
from effector.schemas.dasp import GoalContext, ExpectedStateChange

class Verb(str, Enum):
    READ = "READ"; WRITE = "WRITE"; CALL = "CALL"; DELEGATE = "DELEGATE"; TERMINATE = "TERMINATE"

class AgentRole(str, Enum):
    planner = "planner"; executor = "executor"; verifier = "verifier"; observer = "observer"

class AbortAction(str, Enum):
    abort_and_replan = "ABORT_AND_REPLAN"
    abort_and_escalate = "ABORT_AND_ESCALATE"
    abort_and_terminate = "ABORT_AND_TERMINATE"

class VerificationStatus(str, Enum):
    ack = "ACK"
    nack_schema_error = "NACK_SCHEMA_ERROR"
    nack_ttl_expired = "NACK_TTL_EXPIRED"
    nack_stale_snapshot = "NACK_STALE_SNAPSHOT"
    nack_abort_condition = "NACK_ABORT_CONDITION"
    nack_authorization_error = "NACK_AUTHORIZATION_ERROR"
    pending = "PENDING"

class OperationalMode(str, Enum):
    deliberative = "DELIBERATIVE"; mixed = "MIXED"; reflex = "REFLEX"; supervised = "SUPERVISED"

class AgentIdentity(BaseModel):
    id: str
    role: AgentRole

class WorldModelSnapshot(BaseModel):
    snapshot_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    snapshot_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    relevant_keys: list[str] = Field(default_factory=list)
    hash: str = Field(min_length=64, max_length=64)

class IEPOrigin(BaseModel):
    dasp_session_id: uuid.UUID | None = None
    winning_coalition: list[str] = Field(default_factory=list)
    consensus_score: float = Field(default=0.0, ge=0.0, le=1.0)

class IntendedAction(BaseModel):
    verb: Verb
    target: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    estimated_duration_ms: int | None = None

class AbortCondition(BaseModel):
    condition: str
    action: AbortAction

    @model_validator(mode="after")
    def condition_is_simple(self) -> "AbortCondition":
        c = self.condition.strip()
        if not any(op in c for op in ("==", "!=", ">=", "<=", ">", "<")):
            raise ValueError(f"Abort condition must be a simple comparison; got: {c!r}")
        return self

class IntentionEnvelope(BaseModel):
    iep_version: str = "1.0"
    envelope_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    timestamp_issued: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent: AgentIdentity
    goal_context: GoalContext
    origin: IEPOrigin = Field(default_factory=IEPOrigin)
    world_model_snapshot: WorldModelSnapshot
    intended_action: IntendedAction
    expected_state_change: ExpectedStateChange | None = None
    abort_conditions: list[AbortCondition] = Field(default_factory=list)
    ttl_ms: int = Field(ge=1)
    requires_ack: bool = True

    @model_validator(mode="after")
    def esc_required_for_writes(self) -> "IntentionEnvelope":
        needs_esc = {Verb.WRITE, Verb.CALL, Verb.DELEGATE}
        if self.intended_action.verb in needs_esc and self.expected_state_change is None:
            raise ValueError(f"expected_state_change is required for verb={self.intended_action.verb.value}")
        return self

class ReflexEnvelope(BaseModel):
    iep_version: str = "1.0-A1"
    envelope_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    envelope_class: str = "REFLEX"
    timestamp_issued: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent: AgentIdentity
    rat_id: uuid.UUID
    goal_context: GoalContext
    world_model_snapshot: WorldModelSnapshot
    intended_action: IntendedAction
    expected_state_change: ExpectedStateChange | None = None
    ttl_ms: int = Field(ge=1)
    requires_ack: bool = True

    @model_validator(mode="after")
    def reflex_verb_restricted(self) -> "ReflexEnvelope":
        if self.intended_action.verb not in (Verb.READ, Verb.WRITE):
            raise ValueError(f"Reflex envelopes only permit READ or WRITE")
        return self

class VerificationResult(BaseModel):
    envelope_id: uuid.UUID
    status: VerificationStatus
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    failure_reason: str | None = None
    requires_human_ack: bool = False
    actual_delta: dict[str, Any] | None = None
    divergence_score: float | None = None
    replan_signal: bool = False
    escalation_signal: bool = False
