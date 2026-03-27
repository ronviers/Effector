"""
Effector Session API — start_session() and EffectorSession
"""
from __future__ import annotations
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from effector.adapters.anthropic_adapter import ToolRegistry
from effector.coordinator.coordinator import DASPCoordinator
from effector.schemas.dasp import AgentInfo, DebateResult, DebateRules, GoalContext, ExpectedStateChange
from effector.schemas.iep import (
    AbortCondition, AgentIdentity, AgentRole, IntendedAction,
    IntentionEnvelope, IEPOrigin, OperationalMode,
    VerificationResult, VerificationStatus, Verb, WorldModelSnapshot,
)
from effector.state_bus.bus import StateBus
from effector.state_bus.verifier import IEPVerifier


class IEPExecutor:
    def __init__(self, state_bus: StateBus, mode: OperationalMode = OperationalMode.deliberative,
                 on_envelope_received=None, on_ack=None, on_nack=None):
        self._bus = state_bus
        self._mode = mode
        self._verifier = IEPVerifier(state_bus)
        self._on_envelope_received = on_envelope_received
        self._on_ack = on_ack
        self._on_nack = on_nack

    def execute(self, envelope: IntentionEnvelope, actual_execute_fn=None,
                epsilon_continue=0.1, epsilon_replan=0.3, epsilon_escalate=0.6,
                require_human_ack=False) -> VerificationResult:
        if self._on_envelope_received:
            self._on_envelope_received(envelope)

        result = self._verifier.verify(envelope)
        if result.status != VerificationStatus.ack:
            print(f"\n[IEP] NACK — {result.status.value}: {result.failure_reason}")
            if self._on_nack:
                self._on_nack(result)
            return result

        if require_human_ack or self._mode == OperationalMode.supervised:
            result.requires_human_ack = True
            if not self._prompt_human_ack(envelope):
                result.status = VerificationStatus.nack_abort_condition
                result.failure_reason = "Human operator rejected"
                if self._on_nack:
                    self._on_nack(result)
                return result

        if actual_execute_fn is not None:
            actual_delta = actual_execute_fn(envelope)
        else:
            action = envelope.intended_action
            if action.verb == Verb.WRITE:
                actual_delta = action.parameters
                self._bus.apply_delta(
                    str(envelope.envelope_id), actual_delta, envelope.agent.id,
                    str(envelope.origin.dasp_session_id) if envelope.origin.dasp_session_id else None,
                )
            else:
                actual_delta = {}

        if envelope.expected_state_change and envelope.intended_action.verb != Verb.READ:
            post = self._verifier.post_execution_compare(
                envelope, actual_delta, epsilon_continue, epsilon_replan, epsilon_escalate)
            result.actual_delta = post.actual_delta
            result.divergence_score = post.divergence_score
            result.replan_signal = post.replan_signal
            result.escalation_signal = post.escalation_signal
            if post.replan_signal:
                print(f"[IEP] Replan signal: divergence={post.divergence_score:.3f}")
            if post.escalation_signal:
                print(f"[IEP] Escalation signal: divergence={post.divergence_score:.3f}")

        if self._on_ack:
            self._on_ack(result)
        return result

    def _prompt_human_ack(self, envelope: IntentionEnvelope) -> bool:
        action = envelope.intended_action
        print("\n" + "=" * 60)
        print("  IEP ENVELOPE — AWAITING HUMAN REVIEW")
        print("=" * 60)
        print(f"  Envelope ID : {envelope.envelope_id}")
        print(f"  Agent       : {envelope.agent.id} ({envelope.agent.role.value})")
        print(f"  Verb        : {action.verb.value}  Target: {action.target}")
        if envelope.expected_state_change:
            print(f"  Predicted Δ : {envelope.expected_state_change.predicted_delta}")
        print("-" * 60)
        while True:
            ans = input("  ACK (approve) or NACK (reject)? [a/n]: ").strip().lower()
            if ans in ("a", "ack", "y", "yes"):
                print("  ACK — proceeding"); print("=" * 60); return True
            elif ans in ("n", "nack", "no"):
                print("  NACK — rejected"); print("=" * 60); return False


class EffectorSession:
    def __init__(self, state_bus=None, rules=None, mode=OperationalMode.deliberative):
        self._bus = state_bus or StateBus()
        self._rules = rules or DebateRules(max_rounds=3)
        self._mode = mode
        self._coordinator = DASPCoordinator(self._bus)
        self._executor = IEPExecutor(self._bus, mode=mode)
        self._extra_hooks: dict[str, list] = {}

    def on(self, event: str, callback) -> "EffectorSession":
        dasp_events = {"session_started","round_started","round_complete",
                       "trigger_fired","optimize_started","optimize_complete","session_complete"}
        if event in dasp_events:
            self._coordinator.on(event, callback)
        else:
            self._extra_hooks.setdefault(event, []).append(callback)
        return self

    def run(self, goal: str, agents: list[AgentInfo], agent_registry: dict,
            tools=None, snapshot_keys=None, root_goal_id=None,
            require_human_ack=False, execute_fn=None) -> dict[str, Any]:
        rg_id = root_goal_id or uuid.uuid4()
        goal_context = GoalContext(root_goal_id=rg_id, parent_goal_id=rg_id, depth=0, branch_label=goal[:50])

        debate_result = self._coordinator.run(
            task=goal, agents=agents, agent_registry=agent_registry,
            goal_context=goal_context, rules=self._rules, snapshot_keys=snapshot_keys,
        )

        snapshot_hash, snap_ts, _ = self._bus.snapshot(snapshot_keys)
        envelope = self._build_envelope(debate_result, goal_context, snapshot_hash, snap_ts, snapshot_keys)

        for cb in self._extra_hooks.get("envelope_received", []):
            try: cb(envelope)
            except Exception: pass

        iep_result = self._executor.execute(
            envelope=envelope, actual_execute_fn=execute_fn,
            epsilon_continue=self._rules.epsilon_continue,
            epsilon_replan=self._rules.epsilon_replan,
            epsilon_escalate=self._rules.epsilon_escalate,
            require_human_ack=require_human_ack,
        )
        return {"debate_result": debate_result, "iep_result": iep_result, "envelope": envelope}

    def _build_envelope(self, result, goal_context, snapshot_hash, snap_ts, snapshot_keys):
        return IntentionEnvelope(
            agent=AgentIdentity(id="coordinator", role=AgentRole.executor),
            goal_context=goal_context,
            origin=IEPOrigin(dasp_session_id=result.session_id,
                             winning_coalition=result.winning_coalition,
                             consensus_score=result.consensus_score),
            world_model_snapshot=WorldModelSnapshot(
                snapshot_timestamp=snap_ts,
                relevant_keys=snapshot_keys or [],
                hash=snapshot_hash,
            ),
            intended_action=IntendedAction(verb=Verb.WRITE, target="world_state",
                                           parameters={"debate_answer": result.final_answer}),
            expected_state_change=ExpectedStateChange(
                keys_affected=["debate_answer"],
                predicted_delta={"debate_answer": result.final_answer},
                confidence=result.consensus_score,
            ),
            ttl_ms=30_000,
            requires_ack=True,
        )

    @property
    def state_bus(self) -> StateBus:
        return self._bus


def start_session(agents, goal, tools=None, agent_registry=None, state=None,
                  rules=None, mode=OperationalMode.deliberative, require_human_ack=False,
                  on_round_complete=None, on_envelope_received=None, on_session_complete=None):
    bus = StateBus(initial_state=state or {})
    session = EffectorSession(state_bus=bus, rules=rules, mode=mode)
    if on_round_complete:
        session.on("round_complete", on_round_complete)
    if on_envelope_received:
        session.on("envelope_received", on_envelope_received)
    if on_session_complete:
        session.on("session_complete", on_session_complete)
    return session.run(goal=goal, agents=agents, agent_registry=agent_registry or {},
                       tools=tools, require_human_ack=require_human_ack)
