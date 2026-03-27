"""
DASP Coordinator — Debate-as-a-Service Protocol §2
Manages the full session lifecycle:
  SNAPSHOT → INITIAL_ROUND → DEBATE_ROUNDS → CONSENSUS → RESULT
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from effector.coordinator.signal_engine import SignalEngine
from effector.schemas.dasp import (
    AgentInfo,
    AgentRequest,
    AgentResponse,
    AgentSummary,
    DebateOptimize,
    DebateResult,
    DebateRules,
    DebateStart,
    GoalContext,
    HypothesisFinalState,
    SignalTrace,
    TriggerFire,
    TriggerType,
    TriggerAction,
)
from effector.state_bus.bus import StateBus


# ─────────────────────────────────────────────
# Event types emitted to hooks
# ─────────────────────────────────────────────

EVENTS = {
    "session_started",
    "round_started",
    "round_complete",
    "trigger_fired",
    "optimize_started",
    "optimize_complete",
    "session_complete",
}


# ─────────────────────────────────────────────
# Agent callable protocol
# ─────────────────────────────────────────────

AgentCallable = Callable[[AgentRequest], AgentResponse]


class DASPCoordinator:
    """
    Runs a DASP-1.0 debate session with a provided set of agent callables.

    Usage:
        coordinator = DASPCoordinator(state_bus)
        coordinator.on("round_complete", my_hook)
        result = coordinator.run(session_params, agent_registry)
    """

    def __init__(self, state_bus: StateBus) -> None:
        self._bus = state_bus
        self._hooks: dict[str, list[Callable[[dict[str, Any]], None]]] = {
            e: [] for e in EVENTS
        }

    # ─── Event hooks ────────────────────────────────────────────────────────

    def on(self, event: str, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register a hook. Fires synchronously during session execution."""
        if event not in self._hooks:
            raise ValueError(f"Unknown event {event!r}. Valid events: {sorted(EVENTS)}")
        self._hooks[event].append(callback)

    def _emit(self, event: str, data: dict[str, Any]) -> None:
        for cb in self._hooks.get(event, []):
            try:
                cb(data)
            except Exception as exc:
                print(f"[EFFECTOR] Hook error on {event!r}: {exc}")

    # ─── Main entry point ────────────────────────────────────────────────────

    def run(
        self,
        task: str,
        agents: list[AgentInfo],
        agent_registry: dict[str, AgentCallable],
        goal_context: GoalContext,
        rules: DebateRules | None = None,
        snapshot_keys: list[str] | None = None,
        optimize_fn: Callable[[DebateOptimize], str] | None = None,
    ) -> DebateResult:
        """
        Execute a complete DASP session and return the result contract.

        Args:
            task:            The question or goal for debate.
            agents:          List of AgentInfo objects.
            agent_registry:  Dict mapping agent_id → callable(AgentRequest) → AgentResponse.
            goal_context:    Inherited from parent IEP DELEGATE envelope.
            rules:           DebateRules (defaults applied if None).
            snapshot_keys:   Keys to include in snapshot hash.
            optimize_fn:     Optional callable for OPTIMIZE phase.
        """
        if rules is None:
            rules = DebateRules(max_rounds=5)

        # ── Snapshot ──────────────────────────────────────────────────────
        snapshot_hash, snapshot_ts, _ = self._bus.snapshot(snapshot_keys)

        session_id = uuid.uuid4()
        session_start = DebateStart(
            session_id=session_id,
            task=task,
            snapshot_hash=snapshot_hash,
            snapshot_timestamp=snapshot_ts,
            agents=agents,
            goal_context=goal_context,
            rules=rules,
        )

        self._emit("session_started", {
            "session_id": str(session_id),
            "task": task,
            "agents": [a.id for a in agents],
            "snapshot_hash": snapshot_hash,
        })

        signal_engine = SignalEngine(
            tau_suppression=rules.tau_suppression,
            theta_consensus=rules.theta_consensus,
            epsilon_stall=rules.epsilon_stall,
            state_bus=self._bus,
        )

        all_events: list[dict] = []
        all_rounds: list[dict] = []
        prev_responses: list[AgentResponse] = []
        current_responses: list[AgentResponse] = []
        terminated_reason = "max_rounds"
        current_task = task

        # ── Round loop ────────────────────────────────────────────────────
        for round_num in range(1, rules.max_rounds + 1):
            mode = "initial" if round_num == 1 else "debate"

            self._emit("round_started", {
                "session_id": str(session_id),
                "round": round_num,
                "mode": mode,
            })

            # Build others list from previous round
            others = [
                AgentSummary(
                    agent_id=r.agent_id,
                    answer=r.answer,
                    confidence=r.signal.confidence,
                    polarity=r.signal.polarity,
                )
                for r in prev_responses
            ]

            # Dispatch requests to all agents
            current_responses = []
            hash_mismatches: list[str] = []

            for agent in agents:
                fn = agent_registry.get(agent.id)
                if fn is None:
                    print(f"[EFFECTOR] No callable registered for agent {agent.id!r} — abstaining")
                    continue

                request = AgentRequest(
                    session_id=session_id,
                    round=round_num,
                    mode=mode,
                    task=current_task,
                    snapshot_hash=snapshot_hash,
                    others=others,
                )

                try:
                    response = fn(request)
                except Exception as exc:
                    print(f"[EFFECTOR] Agent {agent.id!r} error: {exc} — abstaining")
                    continue

                # Validate snapshot hash echo
                if response.snapshot_hash != snapshot_hash:
                    hash_mismatches.append(agent.id)
                    print(f"[EFFECTOR] Hash mismatch from {agent.id!r} — treating as stale snapshot")
                    continue

                current_responses.append(response)

            if not current_responses:
                terminated_reason = "error"
                break

            # ── Hash divergence trigger ────────────────────────────────────
            if hash_mismatches:
                trigger = TriggerFire(
                    session_id=session_id,
                    round=round_num,
                    trigger=TriggerType.hash_divergence,
                    action=TriggerAction.re_snapshot,
                    details={"agents_with_mismatch": hash_mismatches},
                )
                self._emit("trigger_fired", trigger.model_dump(mode="json"))
                all_events.append(trigger.model_dump(mode="json"))
                # Re-snapshot and continue
                snapshot_hash, snapshot_ts, _ = self._bus.snapshot(snapshot_keys)

            # ── Ingest signals ─────────────────────────────────────────────
            signal_engine.ingest_responses(current_responses)
            gate_result = signal_engine.evaluate_gates()

            # ── Copy / Swap detection ─────────────────────────────────────
            if signal_engine.copy_detected(current_responses):
                trigger = TriggerFire(
                    session_id=session_id,
                    round=round_num,
                    trigger=TriggerType.copy,
                    action=TriggerAction.optimize,
                )
                self._emit("trigger_fired", trigger.model_dump(mode="json"))
                all_events.append(trigger.model_dump(mode="json"))
                gate_result.stall_fired = True  # treat copy as stall

            if signal_engine.swap_detected(prev_responses, current_responses):
                trigger = TriggerFire(
                    session_id=session_id,
                    round=round_num,
                    trigger=TriggerType.swap,
                    action=TriggerAction.diversity_injection,
                )
                self._emit("trigger_fired", trigger.model_dump(mode="json"))
                all_events.append(trigger.model_dump(mode="json"))

            # ── Round complete event ───────────────────────────────────────
            round_event = {
                "type": "round.complete",
                "session_id": str(session_id),
                "round": round_num,
                "responses": [r.model_dump(mode="json") for r in current_responses],
                "signal_manifold": {
                    hid: sig.model_dump()
                    for hid, sig in signal_engine.manifold_snapshot().items()
                },
                "gate_result": {
                    "inhibition_fired": gate_result.inhibition_fired,
                    "stall_fired": gate_result.stall_fired,
                    "consensus_cleared": gate_result.consensus_cleared,
                },
            }
            all_rounds.append(round_event)
            all_events.append(round_event)

            self._emit("round_complete", round_event)

            # ── P1: Inhibition gate ────────────────────────────────────────
            if gate_result.inhibition_fired:
                trigger = TriggerFire(
                    session_id=session_id,
                    round=round_num,
                    trigger=TriggerType.inhibition_gate,
                    action=TriggerAction.abort_and_replan,
                    details=gate_result.details,
                )
                self._emit("trigger_fired", trigger.model_dump(mode="json"))
                all_events.append(trigger.model_dump(mode="json"))
                terminated_reason = "inhibition"
                break

            # ── P2: Stall gate → Optimize ─────────────────────────────────
            if gate_result.stall_fired and optimize_fn is not None:
                opt_input = DebateOptimize(
                    session_id=session_id,
                    triggered_by="stall",
                    input={"original_task": task, "transcript": all_rounds},
                )
                self._emit("optimize_started", opt_input.model_dump(mode="json"))
                refined = optimize_fn(opt_input)
                current_task = refined
                opt_input.output = {"refined_task": refined}
                self._emit("optimize_complete", opt_input.model_dump(mode="json"))
                all_events.append(opt_input.model_dump(mode="json"))

            # ── P3: Consensus gate ────────────────────────────────────────
            if gate_result.consensus_cleared:
                terminated_reason = "consensus"
                break

            prev_responses = current_responses

        # ── Build result ──────────────────────────────────────────────────
        winning_hypothesis_id, best_score = signal_engine.best_hypothesis()
        if gate_result.consensus_cleared:
            winning_hypothesis_id = gate_result.winning_hypothesis
            best_score = gate_result.consensus_score

        # Identify winning coalition: agents who supported the winning hypothesis
        winning_coalition = []
        final_answer = current_task  # fallback
        if winning_hypothesis_id and current_responses:
            winners = [
                r for r in current_responses
                if r.hypothesis_id == winning_hypothesis_id and r.signal.polarity >= 0
            ]
            winning_coalition = [r.agent_id for r in winners]
            if winners:
                # Use answer from highest-confidence coalition member
                best_winner = max(winners, key=lambda r: r.signal.confidence)
                final_answer = best_winner.answer

        # Disagreement score: fraction of agents with polarity < 0 in final round
        if current_responses:
            dissenters = sum(1 for r in current_responses if r.signal.polarity < 0)
            disagreement = dissenters / len(current_responses)
        else:
            disagreement = 1.0

        result = DebateResult(
            session_id=session_id,
            final_answer=final_answer,
            consensus_score=min(best_score, 1.0),
            rounds=len(all_rounds),
            snapshot_hash=snapshot_hash,
            snapshot_timestamp=snapshot_ts,
            agents=[a.id for a in agents],
            winning_coalition=winning_coalition or [a.id for a in agents],
            disagreement_score=disagreement,
            signal_trace=SignalTrace(per_hypothesis=signal_engine.manifold_snapshot()),
            trace={"events": all_events, "rounds": all_rounds},
            terminated_reason=terminated_reason,
        )

        self._emit("session_complete", result.model_dump(mode="json"))
        return result
