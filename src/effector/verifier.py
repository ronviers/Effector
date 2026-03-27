"""
IEP Verifier — pre-flight checks for Intention Envelopes.
Implements the verification lifecycle from IEP-1.0 §5.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from effector.schemas.iep import (
    AbortAction,
    IntentionEnvelope,
    VerificationResult,
    VerificationStatus,
)
from effector.state_bus.bus import StateBus


class IEPVerifier:
    """
    Executes the five mandatory pre-flight checks in order.
    Performs post-execution divergence scoring.
    """

    def __init__(self, state_bus: StateBus, authorized_roles: dict[str, list[str]] | None = None) -> None:
        self._bus = state_bus
        # authorized_roles: { verb: [allowed agent roles] }
        self._authorized_roles = authorized_roles or {}

    # ─── Pre-flight verification ────────────────────────────────────────────

    def verify(self, envelope: IntentionEnvelope) -> VerificationResult:
        """
        Run all pre-flight checks in strict priority order.
        Returns ACK or the first NACK encountered.
        """
        result = self._check_ttl(envelope)
        if result.status != VerificationStatus.ack:
            return result

        result = self._check_snapshot_hash(envelope)
        if result.status != VerificationStatus.ack:
            return result

        result = self._check_abort_conditions(envelope)
        if result.status != VerificationStatus.ack:
            return result

        result = self._check_role_authorization(envelope)
        if result.status != VerificationStatus.ack:
            return result

        return VerificationResult(
            envelope_id=envelope.envelope_id,
            status=VerificationStatus.ack,
        )

    def _check_ttl(self, envelope: IntentionEnvelope) -> VerificationResult:
        issued_ms = envelope.timestamp_issued.timestamp() * 1000
        now_ms = time.time() * 1000
        if now_ms > issued_ms + envelope.ttl_ms:
            return VerificationResult(
                envelope_id=envelope.envelope_id,
                status=VerificationStatus.nack_ttl_expired,
                failure_reason=f"Envelope TTL expired: issued={envelope.timestamp_issued.isoformat()}, ttl={envelope.ttl_ms}ms",
            )
        return VerificationResult(envelope_id=envelope.envelope_id, status=VerificationStatus.ack)

    def _check_snapshot_hash(self, envelope: IntentionEnvelope) -> VerificationResult:
        snapshot = envelope.world_model_snapshot
        current_hash, _, _ = self._bus.snapshot(snapshot.relevant_keys or None)
        if current_hash != snapshot.hash:
            return VerificationResult(
                envelope_id=envelope.envelope_id,
                status=VerificationStatus.nack_stale_snapshot,
                failure_reason=(
                    f"Snapshot hash mismatch: envelope={snapshot.hash[:12]}... "
                    f"current={current_hash[:12]}..."
                ),
            )
        return VerificationResult(envelope_id=envelope.envelope_id, status=VerificationStatus.ack)

    def _check_abort_conditions(self, envelope: IntentionEnvelope) -> VerificationResult:
        """Substrate-level evaluation — simple key-value comparisons only."""
        current_state = self._bus.read()
        for cond in envelope.abort_conditions:
            if self._evaluate_condition(cond.condition, current_state):
                action_str = cond.action.value
                return VerificationResult(
                    envelope_id=envelope.envelope_id,
                    status=VerificationStatus.nack_abort_condition,
                    failure_reason=f"Abort condition triggered: {cond.condition!r} → {action_str}",
                )
        return VerificationResult(envelope_id=envelope.envelope_id, status=VerificationStatus.ack)

    def _check_role_authorization(self, envelope: IntentionEnvelope) -> VerificationResult:
        verb = envelope.intended_action.verb.value
        role = envelope.agent.role.value
        allowed = self._authorized_roles.get(verb, [])
        if allowed and role not in allowed:
            return VerificationResult(
                envelope_id=envelope.envelope_id,
                status=VerificationStatus.nack_authorization_error,
                failure_reason=f"Role {role!r} not authorized for verb {verb!r}",
            )
        return VerificationResult(envelope_id=envelope.envelope_id, status=VerificationStatus.ack)

    @staticmethod
    def _evaluate_condition(condition: str, state: dict[str, Any]) -> bool:
        """
        Parse and evaluate a simple substrate condition.
        Supported operators: ==, !=, >=, <=, >, <
        """
        for op in ("!=", "==", ">=", "<=", ">", "<"):
            if op in condition:
                parts = condition.split(op, 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    raw_val = parts[1].strip().strip('"').strip("'")
                    # Attempt numeric coercion
                    try:
                        expected: Any = float(raw_val) if "." in raw_val else int(raw_val)
                    except ValueError:
                        expected = raw_val
                    current = state.get(key)
                    if current is None:
                        return False
                    try:
                        if op == "==":  return current == expected
                        if op == "!=":  return current != expected
                        if op == ">=":  return current >= expected
                        if op == "<=":  return current <= expected
                        if op == ">":   return current > expected
                        if op == "<":   return current < expected
                    except TypeError:
                        return False
        return False

    # ─── Post-execution divergence ──────────────────────────────────────────

    def post_execution_compare(
        self,
        envelope: IntentionEnvelope,
        actual_delta: dict[str, Any],
        epsilon_continue: float,
        epsilon_replan: float,
        epsilon_escalate: float,
    ) -> VerificationResult:
        """
        Compare actual vs predicted state delta.
        Emits divergence score and replan/escalation signals per IEP §6.
        """
        predicted = {}
        if envelope.expected_state_change:
            predicted = envelope.expected_state_change.predicted_delta

        divergence = self._compute_divergence(actual_delta, predicted)

        replan = divergence >= epsilon_replan
        escalate = divergence >= epsilon_escalate

        return VerificationResult(
            envelope_id=envelope.envelope_id,
            status=VerificationStatus.ack,
            actual_delta=actual_delta,
            divergence_score=divergence,
            replan_signal=replan,
            escalation_signal=escalate,
        )

    @staticmethod
    def _compute_divergence(actual: dict, predicted: dict) -> float:
        """
        Simple divergence metric: fraction of affected keys where actual != predicted.
        Returns 0.0 (perfect match) to 1.0 (complete mismatch).
        """
        all_keys = set(actual) | set(predicted)
        if not all_keys:
            return 0.0
        mismatches = sum(1 for k in all_keys if actual.get(k) != predicted.get(k))
        return mismatches / len(all_keys)
