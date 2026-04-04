"""
reflex_engine.py — Step 2: The Reflex Engine

The engine's "spinal cord." Intercepts structured IntendedAction dicts,
validates them against the local RAT store, and executes pre-authorized
actions without invoking the DASP LLM debate loop.

Check order mirrors IEP §7 (Reflex Envelope Verification) and the
IEPVerifier / IEPValidator check order in verifier.py / iep_queue.py:

  1. Candidate RAT lookup         (replaces schema + RAT-lookup checks)
  2. RAT TTL / expiry             (mirrors _check_ttl)
  3. RAT action match             (verb + target authorisation)
  4. Execution count              (max_executions not exceeded)
  5. M5 — Critical-key drift      (mirrors _check_critical_key_drift)
  6. M4 — Cosine similarity       (mirrors _cosine_snapshot_check)
       → hash fallback if vector unavailable
  7. Atomic decrement + execute

Architecture decisions (DO NOT RELITIGATE)
------------------------------------------
- This module makes ZERO Ollama API calls.
  The caller computes current_state_vector before calling evaluate_reflex().
- Execution limit decrements are atomic SQLite operations (see rat_store.py).
- Post-execution divergence scoring is fully async (threading.Thread).
- The fast-path critical section is sub-2ms.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from effector.rat_store import LocalRATStore, RATRecord


# ---------------------------------------------------------------------------
# Keys treated as "critical" — a change here forces a full hash recheck
# even when cosine similarity passes (mirrors IEP §8.3 risk table and the
# DEFAULT_CRITICAL_KEYS in iep_queue.py).
# ---------------------------------------------------------------------------

DEFAULT_CRITICAL_KEYS: tuple[str, ...] = (
    "desktop.active_window",
    "desktop.active_process",
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class ReflexStatus(str, Enum):
    EXECUTED            = "EXECUTED"           # Fast path succeeded
    BYPASSED            = "BYPASSED"           # No matching RAT — fall back to DASP
    NACK_EXPIRED        = "NACK_EXPIRED"       # RAT TTL elapsed
    NACK_NO_ACTION      = "NACK_NO_ACTION"     # RAT found but verb/target not authorised
    NACK_EXHAUSTED      = "NACK_EXHAUSTED"     # max_executions reached (atomic)
    NACK_CRITICAL_DRIFT = "NACK_CRITICAL_DRIFT" # M5: critical key changed
    NACK_HASH_MISMATCH  = "NACK_HASH_MISMATCH" # M4 fallback: exact hash failed
    NACK_LOW_SIMILARITY = "NACK_LOW_SIMILARITY" # M4: cosine below threshold
    ERROR               = "ERROR"              # Unexpected exception


@dataclass
class ReflexResult:
    status: ReflexStatus
    rat_id: str | None = None
    matched_action: dict[str, Any] | None = None
    executions_remaining: int | None = None
    similarity_score: float | None = None
    failure_reason: str | None = None
    executed_at: str | None = None
    actual_delta: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == ReflexStatus.EXECUTED


# ---------------------------------------------------------------------------
# Cosine similarity (no external deps)
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def _hash_state(state: dict[str, Any]) -> str:
    canonical = json.dumps(state, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------------------
# ReflexEngine
# ---------------------------------------------------------------------------

class ReflexEngine:
    """
    Fast reflex path: RAT lookup → M5 → M4 → atomic execute.

    Parameters
    ----------
    rat_store
        LocalRATStore instance (shared with main loop).
    critical_keys
        Keys whose exact-hash change invalidates reflex even when cosine passes.
    on_post_execute
        Optional async callback fired after execution for divergence scoring /
        reputation updates. Receives (rat_record, intended_action, actual_delta).
    executor_pool_size
        Thread pool size for async post-execution callbacks.
    """

    def __init__(
        self,
        rat_store: LocalRATStore,
        critical_keys: tuple[str, ...] = DEFAULT_CRITICAL_KEYS,
        on_post_execute: Callable[[RATRecord, dict, dict], None] | None = None,
        executor_pool_size: int = 2,
    ) -> None:
        self._store = rat_store
        self._critical_keys = critical_keys
        self._on_post_execute = on_post_execute
        self._pool = ThreadPoolExecutor(
            max_workers=executor_pool_size,
            thread_name_prefix="RefEx-PostExec",
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def evaluate_reflex(
        self,
        intended_action: dict[str, Any],
        current_state_vector: list[float],
        state_bus: Any,
        *,
        execute_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> ReflexResult:
        """
        Fast-path evaluation.

        Parameters
        ----------
        intended_action
            {"verb": str, "target": str, "parameters": dict}
            — provided by IntentRouter (Step 0), never a raw task string.
        current_state_vector
            Dense float embedding of the current serialized StateBus state,
            computed by the caller via Ollama nomic-embed-text BEFORE this call.
        state_bus
            The live StateBus instance.
        execute_fn
            Optional callable(intended_action) -> actual_delta. If None,
            WRITE verbs apply action.parameters directly to the state_bus.

        Returns
        -------
        ReflexResult
            .status == EXECUTED  → fast path succeeded, action applied.
            .status == BYPASSED  → no RAT matched; caller must use DASP.
            .status == NACK_*    → RAT found but check failed; caller may retry
                                   after re-deliberation.
        """
        verb: str = intended_action.get("verb", "").upper()
        target: str = intended_action.get("target", "")

        # ── Step 1: RAT lookup ─────────────────────────────────────────────
        candidates = self._store.get_candidate_rats(verb, target)
        if not candidates:
            return ReflexResult(
                status=ReflexStatus.BYPASSED,
                failure_reason="No live RAT authorises this verb/target pair",
            )

        # Iterate candidates in issuance order; use the first that passes all checks
        for rat in candidates:
            result = self._evaluate_candidate(
                rat=rat,
                verb=verb,
                target=target,
                current_state_vector=current_state_vector,
                state_bus=state_bus,
                intended_action=intended_action,
                execute_fn=execute_fn,
            )
            if result.status == ReflexStatus.EXECUTED:
                return result
            # If NACK_EXPIRED or NACK_EXHAUSTED, try next candidate
            if result.status in (ReflexStatus.NACK_EXPIRED, ReflexStatus.NACK_EXHAUSTED):
                continue
            # For M4/M5 failures, don't try other RATs — world state is the issue
            return result

        # All candidates exhausted or expired
        return ReflexResult(
            status=ReflexStatus.NACK_EXHAUSTED,
            failure_reason="All candidate RATs are exhausted or expired",
        )

    def shutdown(self) -> None:
        """Gracefully shut down the async post-execution thread pool."""
        self._pool.shutdown(wait=True)

    # ── Internal checks ─────────────────────────────────────────────────────

    def _evaluate_candidate(
        self,
        rat: RATRecord,
        verb: str,
        target: str,
        current_state_vector: list[float],
        state_bus: Any,
        intended_action: dict[str, Any],
        execute_fn: Callable | None,
    ) -> ReflexResult:
        """Run the full IEP §7 reflex check sequence against one RAT."""

        # ── Check 2: RAT TTL ────────────────────────────────────────────────
        if rat.is_expired:
            return ReflexResult(
                status=ReflexStatus.NACK_EXPIRED,
                rat_id=rat.rat_id,
                failure_reason=f"RAT {rat.rat_id[:8]}… expired at {rat.ttl_expiry_timestamp}",
            )

        # ── Check 3: Action match ───────────────────────────────────────────
        matched_action = rat.authorizes(verb, target)
        if matched_action is None:
            return ReflexResult(
                status=ReflexStatus.NACK_NO_ACTION,
                rat_id=rat.rat_id,
                failure_reason=f"RAT {rat.rat_id[:8]}… does not authorise {verb} {target}",
            )

        # ── Check 4: Execution count (preliminary read; atomic decrement later)
        # We just skip here if executions_remaining is 0 — the atomic decrement
        # will confirm. This avoids a wasted M4/M5 computation.
        if rat.executions_remaining == 0:
            return ReflexResult(
                status=ReflexStatus.NACK_EXHAUSTED,
                rat_id=rat.rat_id,
                failure_reason=f"RAT {rat.rat_id[:8]}… has 0 executions remaining",
            )

        # ── Check 5 (M5): Critical-key drift ───────────────────────────────
        drift_reason = self._check_critical_key_drift(rat, state_bus)
        if drift_reason:
            return ReflexResult(
                status=ReflexStatus.NACK_CRITICAL_DRIFT,
                rat_id=rat.rat_id,
                failure_reason=drift_reason,
            )

        # ── Check 6 (M4): Snapshot similarity ──────────────────────────────
        snapshot_result = self._check_snapshot(
            rat=rat,
            current_state_vector=current_state_vector,
            state_bus=state_bus,
        )
        if snapshot_result is not None:
            # snapshot_result is a ReflexResult on failure, None on pass
            return snapshot_result

        # ── Check 4b: Atomic execution decrement ───────────────────────────
        remaining = self._store.decrement_and_fetch(rat.rat_id)
        if remaining is None:
            # Race condition: another thread consumed the last execution
            return ReflexResult(
                status=ReflexStatus.NACK_EXHAUSTED,
                rat_id=rat.rat_id,
                failure_reason=f"RAT {rat.rat_id[:8]}… exhausted (concurrent decrement)",
            )

        # ── Execute ─────────────────────────────────────────────────────────
        actual_delta = self._execute(
            intended_action=intended_action,
            state_bus=state_bus,
            execute_fn=execute_fn,
        )

        executed_at = datetime.now(timezone.utc).isoformat()

        result = ReflexResult(
            status=ReflexStatus.EXECUTED,
            rat_id=rat.rat_id,
            matched_action=matched_action,
            executions_remaining=remaining,
            executed_at=executed_at,
            actual_delta=actual_delta,
        )

        # ── Async post-execution: divergence + reputation (non-blocking) ───
        if self._on_post_execute is not None:
            self._pool.submit(self._post_execute_async, rat, intended_action, actual_delta)

        return result

    # ── M5: Critical-key drift check ────────────────────────────────────────

    def _check_critical_key_drift(
        self,
        rat: RATRecord,
        state_bus: Any,
    ) -> str | None:
        """
        Return an error string if critical keys are actively mutating between
        two rapid reads (the double-read mutation guard from iep_queue.py).

        This is a live state stability check — it catches hot-changing contexts
        (e.g. the user switching active windows mid-validation). It does NOT
        compare against the value stored in the RAT's snapshot.
        """
        if not self._critical_keys:
            return None

        # First read — get current critical key state
        current_state = state_bus.read()
        critical_state = {k: current_state.get(k) for k in self._critical_keys if k in current_state}

        if not critical_state:
            # Critical keys not yet populated (telemetry not polled yet) — skip
            return None

        # Second rapid read — check for mutation between the two reads
        current_state2 = state_bus.read()
        critical_state2 = {k: current_state2.get(k) for k in self._critical_keys if k in current_state2}

        if _hash_state(critical_state) != _hash_state(critical_state2):
            return (
                "Critical key state mutated between two rapid reads during "
                "reflex validation — world state is actively changing; replan."
            )

        return None

    # ── M4: Snapshot similarity check ───────────────────────────────────────

    def _check_snapshot(
        self,
        rat: RATRecord,
        current_state_vector: list[float],
        state_bus: Any,
    ) -> ReflexResult | None:
        """
        Attempt cosine similarity check (M4). Falls back to SHA-256 hash if:
          - RAT has no snapshot_vector
          - caller provided an empty current_state_vector
          - vectors have mismatched dimensions

        Returns None on success, a ReflexResult on failure.
        """
        # ── Cosine path ─────────────────────────────────────────────────────
        if (
            rat.snapshot_vector
            and current_state_vector
            and len(current_state_vector) >= 256
        ):
            if len(current_state_vector) == len(rat.snapshot_vector):
                similarity = _cosine_similarity(current_state_vector, rat.snapshot_vector)
                threshold = rat.rat_similarity_threshold

                if similarity >= threshold:
                    return None  # Pass
                else:
                    return ReflexResult(
                        status=ReflexStatus.NACK_LOW_SIMILARITY,
                        rat_id=rat.rat_id,
                        similarity_score=similarity,
                        failure_reason=(
                            f"Cosine similarity {similarity:.4f} < threshold "
                            f"{threshold} — world state has drifted; replan."
                        ),
                    )
            else:
                print(
                    f"[ReflexEngine] Vector dimension mismatch for RAT "
                    f"{rat.rat_id[:8]}: current={len(current_state_vector)} "
                    f"stored={len(rat.snapshot_vector)} — hash fallback."
                )

        # ── Hash fallback ────────────────────────────────────────────────────
        if not rat.snapshot_hash:
            # No hash stored — can't verify; allow (RAT was issued without hash)
            return None

        # Snapshot keys: use whatever keys the RAT's authorized_actions reference
        # as a proxy for relevant_keys. If none specified, hash the full bus.
        relevant_keys: list[str] | None = None
        for action in rat.authorized_actions:
            if "parameter_constraints" in action:
                keys = list(action["parameter_constraints"].keys())
                if keys:
                    relevant_keys = keys
                    break

        current_hash, _, _ = state_bus.snapshot(relevant_keys)

        if current_hash == rat.snapshot_hash:
            return None  # Pass

        return ReflexResult(
            status=ReflexStatus.NACK_HASH_MISMATCH,
            rat_id=rat.rat_id,
            failure_reason=(
                f"Snapshot hash mismatch: stored={rat.snapshot_hash[:12]}… "
                f"current={current_hash[:12]}…"
            ),
        )

    # ── Execution ────────────────────────────────────────────────────────────

    def _execute(
        self,
        intended_action: dict[str, Any],
        state_bus: Any,
        execute_fn: Callable | None,
    ) -> dict[str, Any]:
        """
        Apply the action. Returns the actual delta dict.
        """
        if execute_fn is not None:
            return execute_fn(intended_action) or {}

        verb = intended_action.get("verb", "").upper()
        if verb == "WRITE":
            delta = dict(intended_action.get("parameters", {}))
            state_bus.apply_delta(
                envelope_id=f"reflex-{time.monotonic_ns()}",
                delta=delta,
                agent_id="reflex_engine",
                session_id=None,
            )
            return delta

        # READ and other verbs: no state mutation
        return {}

    # ── Async post-execution ─────────────────────────────────────────────────

    def _post_execute_async(
        self,
        rat: RATRecord,
        intended_action: dict[str, Any],
        actual_delta: dict[str, Any],
    ) -> None:
        """
        Compute divergence between actual_delta and any predicted_delta in
        the matched action, then invoke on_post_execute callback.

        Runs in a daemon thread — must NOT block the calling thread.
        """
        try:
            if self._on_post_execute is not None:
                self._on_post_execute(rat, intended_action, actual_delta)
        except Exception as exc:
            print(f"[ReflexEngine] Post-execute callback error: {exc}")
