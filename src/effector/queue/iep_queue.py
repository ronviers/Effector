"""
A3: IEP Validation and Local Envelope Queue
=============================================
Routes DASP consensus results through the Intention Envelope Protocol:

  1. IEPBuilder       — constructs a typed IntentionEnvelope from a raw
                        debate result dict (no Pydantic dependency required).
  2. IEPValidator     — validates the envelope against the 5 pre-flight checks
                        from IEP §5, operating purely on Python dicts.
  3. EnvelopeQueue    — thread-safe in-memory queue with optional JSON-Lines
                        persistence to a local file. Each entry is a fully
                        serialised IEP envelope + validation verdict.

Usage
-----
    from effector.queue.iep_queue import EnvelopeQueue, IEPBuilder, IEPValidator

    queue = EnvelopeQueue(persist_path="iep_queue.jsonl")

    envelope = IEPBuilder.from_debate_result(
        debate_result=result_dict,
        state_bus_snapshot_hash=current_hash,
        keys_affected=["cache_enabled"],
    )

    verdict = IEPValidator(state_bus=bus).validate(envelope)
    queue.put(envelope, verdict)

Queue consumers
---------------
    item = queue.get()          # blocks until available
    item = queue.get_nowait()   # raises queue.Empty if empty
    all_items = queue.drain()   # returns list, non-blocking
"""

from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight envelope dict schema
# (mirrors IEP-1.0 §4 without hard Pydantic dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_world_model_snapshot(
    snapshot_hash: str,
    relevant_keys: list[str],
) -> dict:
    return {
        "snapshot_id": str(uuid.uuid4()),
        "snapshot_timestamp": _now_iso(),
        "relevant_keys": relevant_keys,
        "hash": snapshot_hash,
    }


def _make_expected_state_change(
    keys_affected: list[str],
    predicted_delta: dict[str, Any],
    confidence: float,
) -> dict:
    return {
        "keys_affected": keys_affected,
        "predicted_delta": predicted_delta,
        "confidence": round(max(0.0, min(1.0, confidence)), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# IEP Builder — constructs envelope from debate result
# ─────────────────────────────────────────────────────────────────────────────

class IEPBuilder:
    """
    Constructs a fully-populated IEP envelope dict from a raw DASP result dict.

    The builder extracts:
      - origin fields (session_id, coalition, consensus_score)
      - intended_action  (verb=WRITE, target=world_state, params from final_answer)
      - expected_state_change (aggregated from agent ESC declarations if present)

    Parameters
    ----------
    debate_result : dict
        Raw dict returned by AsymmetricDASPCoordinator.run() or any DASP result.
    state_bus_snapshot_hash : str
        SHA-256 of the world state at the time of envelope emission.
        MUST match what the winning coalition agents saw.
    keys_affected : list[str]
        World-state keys this action will touch.
    predicted_delta : dict | None
        Explicit predicted delta. If None, inferred from final_answer.
    ttl_ms : int
        Envelope TTL in milliseconds.
    agent_id : str
        Coordinator identity string.
    """

    @staticmethod
    def from_debate_result(
        debate_result: dict[str, Any],
        state_bus_snapshot_hash: str,
        keys_affected: list[str] | None = None,
        predicted_delta: dict[str, Any] | None = None,
        ttl_ms: int = 30_000,
        agent_id: str = "coordinator",
    ) -> dict[str, Any]:
        keys_affected = keys_affected or []
        final_answer = debate_result.get("final_answer", "(no answer)")
        consensus_score = float(debate_result.get("consensus_score", 0.0))
        session_id = debate_result.get("session_id", str(uuid.uuid4()))

        # Aggregate predicted deltas from round responses if available
        if predicted_delta is None:
            predicted_delta = IEPBuilder._aggregate_esc(debate_result, keys_affected)

        # If no structured delta could be inferred, represent the answer itself
        if not predicted_delta:
            predicted_delta = {"debate_answer": final_answer}
            if not keys_affected:
                keys_affected = ["debate_answer"]

        envelope: dict[str, Any] = {
            "iep_version": "1.0",
            "envelope_id": str(uuid.uuid4()),
            "timestamp_issued": _now_iso(),

            "agent": {
                "id": agent_id,
                "role": "executor",
            },

            "goal_context": {
                "root_goal_id": str(uuid.uuid4()),
                "parent_goal_id": session_id,
                "depth": 0,
                "branch_label": final_answer[:50],
            },

            "origin": {
                "dasp_session_id": session_id,
                "winning_coalition": debate_result.get("tier1_agents", []),
                "consensus_score": consensus_score,
            },

            "world_model_snapshot": _make_world_model_snapshot(
                snapshot_hash=state_bus_snapshot_hash,
                relevant_keys=keys_affected,
            ),

            "intended_action": {
                "verb": "WRITE",
                "target": "world_state",
                "parameters": {"debate_answer": final_answer},
                "estimated_duration_ms": 50,
            },

            "expected_state_change": _make_expected_state_change(
                keys_affected=keys_affected,
                predicted_delta=predicted_delta,
                confidence=consensus_score,
            ),

            "abort_conditions": [],
            "ttl_ms": ttl_ms,
            "requires_ack": True,
        }

        return envelope

    @staticmethod
    def _aggregate_esc(
        debate_result: dict,
        keys_affected: list[str],
    ) -> dict[str, Any]:
        """
        Walk the round transcript and collect the modal predicted_delta
        across all agent ESC declarations for the affected keys.
        """
        aggregated: dict[str, list] = {k: [] for k in keys_affected}

        rounds = debate_result.get("all_rounds", [])
        for rnd in rounds:
            for resp in rnd.get("responses", []):
                esc = resp.get("expected_state_change", {})
                pd = esc.get("predicted_delta", {})
                for k in keys_affected:
                    if k in pd:
                        aggregated[k].append(pd[k])

        # Modal value per key; skip keys with no predictions
        result: dict[str, Any] = {}
        for k, vals in aggregated.items():
            if not vals:
                continue
            # Most common value (simple mode)
            try:
                result[k] = max(set(vals), key=vals.count)
            except TypeError:
                result[k] = vals[-1]  # fallback: last seen
        return result


# ─────────────────────────────────────────────────────────────────────────────
# IEP Validator — five pre-flight checks per IEP §5
# ─────────────────────────────────────────────────────────────────────────────

class ValidationResult:
    __slots__ = (
        "envelope_id", "status", "failure_reason",
        "checks_passed", "checks_failed",
        "divergence_score", "actual_delta",
        "timestamp",
    )

    def __init__(
        self,
        envelope_id: str,
        status: str,
        failure_reason: str | None = None,
        checks_passed: list[str] | None = None,
        checks_failed: list[str] | None = None,
    ) -> None:
        self.envelope_id = envelope_id
        self.status = status
        self.failure_reason = failure_reason
        self.checks_passed: list[str] = checks_passed or []
        self.checks_failed: list[str] = checks_failed or []
        self.divergence_score: float | None = None
        self.actual_delta: dict | None = None
        self.timestamp = _now_iso()

    def to_dict(self) -> dict:
        return {
            "envelope_id": self.envelope_id,
            "status": self.status,
            "failure_reason": self.failure_reason,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "divergence_score": self.divergence_score,
            "actual_delta": self.actual_delta,
            "timestamp": self.timestamp,
        }

    def __repr__(self) -> str:
        return f"ValidationResult(status={self.status!r}, id={self.envelope_id[:8]}...)"


class IEPValidator:
    """
    Stateless five-check IEP pre-flight validator.
    Operates on raw envelope dicts — does not require Pydantic models.

    Checks (in priority order):
      1. Schema completeness — required fields present
      2. TTL freshness     — envelope not expired
      3. Snapshot hash     — world-state matches envelope hash
      4. Abort conditions  — substrate conditions not triggered
      5. Role authorization — agent role permitted for verb
    """

    REQUIRED_TOP_LEVEL = {
        "iep_version", "envelope_id", "timestamp_issued",
        "agent", "goal_context", "world_model_snapshot",
        "intended_action", "ttl_ms", "requires_ack",
    }
    WRITE_VERBS = {"WRITE", "CALL", "DELEGATE"}

    def __init__(
        self,
        state_bus: Any,  # duck-typed: needs .read() and .snapshot()
        authorized_roles: dict[str, list[str]] | None = None,
    ) -> None:
        self._bus = state_bus
        self._authorized_roles: dict[str, list[str]] = authorized_roles or {}

    def validate(self, envelope: dict[str, Any]) -> ValidationResult:
        eid = envelope.get("envelope_id", "unknown")
        result = ValidationResult(envelope_id=eid, status="ACK")

        checks = [
            ("schema_completeness", self._check_schema),
            ("ttl_freshness", self._check_ttl),
            ("snapshot_hash", self._check_snapshot),
            ("abort_conditions", self._check_abort),
            ("role_authorization", self._check_role),
        ]

        for check_name, check_fn in checks:
            ok, reason = check_fn(envelope)
            if ok:
                result.checks_passed.append(check_name)
            else:
                result.checks_failed.append(check_name)
                result.status = f"NACK_{check_name.upper()}"
                result.failure_reason = reason
                return result  # fail-fast per IEP §5

        return result

    def post_execution_compare(
        self,
        envelope: dict[str, Any],
        actual_delta: dict[str, Any],
        epsilon_continue: float = 0.1,
        epsilon_replan: float = 0.3,
        epsilon_escalate: float = 0.6,
    ) -> dict[str, Any]:
        """Compute forward-model divergence per IEP §6."""
        esc = envelope.get("expected_state_change", {})
        predicted = esc.get("predicted_delta", {})

        divergence = self._compute_divergence(actual_delta, predicted)
        replan = divergence >= epsilon_replan
        escalate = divergence >= epsilon_escalate

        return {
            "envelope_id": envelope.get("envelope_id"),
            "actual_delta": actual_delta,
            "predicted_delta": predicted,
            "divergence_score": round(divergence, 4),
            "replan_signal": replan,
            "escalation_signal": escalate,
            "threshold_used": {
                "epsilon_continue": epsilon_continue,
                "epsilon_replan": epsilon_replan,
                "epsilon_escalate": epsilon_escalate,
            },
        }

    # ── Check implementations ─────────────────────────────────────────────

    def _check_schema(self, env: dict) -> tuple[bool, str | None]:
        missing = self.REQUIRED_TOP_LEVEL - set(env.keys())
        if missing:
            return False, f"Missing required fields: {sorted(missing)}"
        verb = env.get("intended_action", {}).get("verb", "")
        if verb in self.WRITE_VERBS and "expected_state_change" not in env:
            return False, f"expected_state_change required for verb={verb}"
        return True, None

    def _check_ttl(self, env: dict) -> tuple[bool, str | None]:
        try:
            issued_str = env["timestamp_issued"]
            # Parse ISO timestamp
            issued_ts = datetime.fromisoformat(issued_str).timestamp()
            ttl_ms = int(env["ttl_ms"])
            deadline = issued_ts + ttl_ms / 1000.0
            if time.time() > deadline:
                return False, (
                    f"TTL expired: issued={issued_str}, ttl={ttl_ms}ms, "
                    f"expired {(time.time() - deadline)*1000:.0f}ms ago"
                )
            return True, None
        except Exception as exc:
            return False, f"TTL check error: {exc}"

    def _check_snapshot(self, env: dict) -> tuple[bool, str | None]:
        try:
            snap = env.get("world_model_snapshot", {})
            expected_hash = snap.get("hash", "")
            relevant_keys = snap.get("relevant_keys") or None
            current_hash, _, _ = self._bus.snapshot(relevant_keys)
            if current_hash != expected_hash:
                return False, (
                    f"Snapshot hash mismatch: "
                    f"envelope={expected_hash[:12]}... "
                    f"current={current_hash[:12]}..."
                )
            return True, None
        except Exception as exc:
            return False, f"Snapshot check error: {exc}"

    def _check_abort(self, env: dict) -> tuple[bool, str | None]:
        try:
            state = self._bus.read()
            for cond in env.get("abort_conditions", []):
                condition = cond.get("condition", "")
                action = cond.get("action", "ABORT_AND_REPLAN")
                if self._eval_condition(condition, state):
                    return False, f"Abort condition triggered: {condition!r} → {action}"
            return True, None
        except Exception as exc:
            return False, f"Abort condition check error: {exc}"

    def _check_role(self, env: dict) -> tuple[bool, str | None]:
        verb = env.get("intended_action", {}).get("verb", "")
        role = env.get("agent", {}).get("role", "")
        allowed = self._authorized_roles.get(verb, [])
        if allowed and role not in allowed:
            return False, f"Role {role!r} not authorized for verb {verb!r}"
        return True, None

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _eval_condition(condition: str, state: dict) -> bool:
        for op in ("!=", "==", ">=", "<=", ">", "<"):
            if op in condition:
                parts = condition.split(op, 1)
                if len(parts) != 2:
                    continue
                key = parts[0].strip()
                raw = parts[1].strip().strip('"').strip("'")
                try:
                    expected: Any = float(raw) if "." in raw else int(raw)
                except ValueError:
                    expected = raw
                current = state.get(key)
                if current is None:
                    return False
                try:
                    if op == "==": return current == expected
                    if op == "!=": return current != expected
                    if op == ">=": return current >= expected
                    if op == "<=": return current <= expected
                    if op == ">":  return current > expected
                    if op == "<":  return current < expected
                except TypeError:
                    return False
        return False

    @staticmethod
    def _compute_divergence(actual: dict, predicted: dict) -> float:
        all_keys = set(actual) | set(predicted)
        if not all_keys:
            return 0.0
        mismatches = sum(1 for k in all_keys if actual.get(k) != predicted.get(k))
        return mismatches / len(all_keys)


# ─────────────────────────────────────────────────────────────────────────────
# Envelope Queue — thread-safe in-memory + optional JSONL persistence
# ─────────────────────────────────────────────────────────────────────────────

class QueueItem:
    """A validated envelope ready for consumption."""
    __slots__ = ("envelope", "validation", "queued_at")

    def __init__(self, envelope: dict, validation: ValidationResult) -> None:
        self.envelope = envelope
        self.validation = validation
        self.queued_at = _now_iso()

    def to_dict(self) -> dict:
        return {
            "queued_at": self.queued_at,
            "envelope": self.envelope,
            "validation": self.validation.to_dict(),
        }

    def is_ack(self) -> bool:
        return self.validation.status == "ACK"

    def __repr__(self) -> str:
        eid = self.envelope.get("envelope_id", "?")[:8]
        return f"QueueItem(id={eid}..., status={self.validation.status!r})"


class EnvelopeQueue:
    """
    Thread-safe envelope queue with optional JSONL persistence.

    Items are written regardless of validation status — the consumer
    decides what to do with NACKs. This keeps the audit trail complete.

    Parameters
    ----------
    persist_path : str | Path | None
        If provided, each enqueued item is appended as a JSON line.
    maxsize : int
        Maximum in-memory queue depth (0 = unlimited).
    """

    def __init__(
        self,
        persist_path: str | Path | None = None,
        maxsize: int = 0,
    ) -> None:
        self._q: queue.Queue[QueueItem] = queue.Queue(maxsize=maxsize)
        self._persist_path = Path(persist_path) if persist_path else None
        self._lock = threading.Lock()
        self._total_enqueued = 0
        self._total_acked = 0
        self._total_nacked = 0

        if self._persist_path:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[EnvelopeQueue] Persisting to {self._persist_path}")

    def put(
        self,
        envelope: dict[str, Any],
        validation: ValidationResult,
        block: bool = True,
        timeout: float | None = None,
    ) -> QueueItem:
        """Enqueue an envelope + its validation verdict."""
        item = QueueItem(envelope=envelope, validation=validation)
        self._q.put(item, block=block, timeout=timeout)
        with self._lock:
            self._total_enqueued += 1
            if item.is_ack():
                self._total_acked += 1
            else:
                self._total_nacked += 1
        self._persist(item)
        return item

    def get(self, block: bool = True, timeout: float | None = None) -> QueueItem:
        """Dequeue the next item (blocks by default)."""
        return self._q.get(block=block, timeout=timeout)

    def get_nowait(self) -> QueueItem:
        """Dequeue without blocking. Raises queue.Empty if empty."""
        return self._q.get_nowait()

    def drain(self) -> list[QueueItem]:
        """Non-blocking drain. Returns all currently queued items."""
        items: list[QueueItem] = []
        while True:
            try:
                items.append(self._q.get_nowait())
            except queue.Empty:
                break
        return items

    def qsize(self) -> int:
        return self._q.qsize()

    def empty(self) -> bool:
        return self._q.empty()

    @property
    def stats(self) -> dict[str, int]:
        return {
            "total_enqueued": self._total_enqueued,
            "total_acked": self._total_acked,
            "total_nacked": self._total_nacked,
            "current_depth": self._q.qsize(),
        }

    def _persist(self, item: QueueItem) -> None:
        if not self._persist_path:
            return
        try:
            line = json.dumps(item.to_dict(), default=str) + "\n"
            with self._lock:
                with open(self._persist_path, "a", encoding="utf-8") as f:
                    f.write(line)
        except Exception as exc:
            print(f"[EnvelopeQueue] Persist error: {exc}")

    def replay_from_disk(self) -> list[dict]:
        """
        Load all persisted items from disk (for replay / audit).
        Returns list of raw dicts. Does NOT re-enqueue them.
        """
        if not self._persist_path or not self._persist_path.exists():
            return []
        items = []
        with open(self._persist_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return items

    def __repr__(self) -> str:
        return (
            f"EnvelopeQueue(depth={self.qsize()}, "
            f"enqueued={self._total_enqueued}, "
            f"acked={self._total_acked}, nacked={self._total_nacked})"
        )
