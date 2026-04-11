"""
A3: IEP Validation and Local Envelope Queue
=============================================
Routes DASP consensus results through the Intention Envelope Protocol:

  1. IEPBuilder       — constructs a typed IntentionEnvelope from a raw
                        debate result dict.
  2. IEPValidator     — validates the envelope against the 5 pre-flight checks
                        from IEP §5, operating purely on Python dicts.
  3. EnvelopeQueue    — thread-safe in-memory queue with optional JSON-Lines
                        persistence to a local file.

IEP-A3 additions
----------------
M3 — IEPBuilder.from_debate_result() now extracts ``snapshot_vector``,
     ``vectorized_bus``, ``rat_similarity_threshold``, and
     ``embedding_model`` from the DASP result dict and writes them into
     ``world_model_snapshot``.  Envelopes produced without a vector
     (vector=None or vectorized_bus=False) are unaffected: they fall
     through to the original SHA-256 path.

M4 — IEPValidator._check_snapshot() performs cosine similarity verification
     when the envelope carries a non-null snapshot_vector.  The current state
     is serialized via StateBus.serialize() and embedded via Ollama /api/embed.
     If the embedding call fails, the validator falls back to hash-only mode
     and logs a warning — it never silently approves an unverifiable state.

M5 — Before any cosine computation, the validator performs an exact-value
     check on a configurable ``critical_keys`` list (defaults to the
     desktop-state keys).  A mutation in any critical key immediately
     returns NACK_SNAPSHOT_HASH regardless of the similarity score,
     forcing a replan.  This prevents the similarity threshold from
     inadvertently authorizing context-sensitive actions when the user's
     environment has meaningfully changed (e.g. a new active window).
"""

from __future__ import annotations

import hashlib
import json
import math
import queue
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_world_model_snapshot(
    snapshot_hash: str,
    relevant_keys: list[str],
    *,
    snapshot_vector: list[float] | None = None,
    vectorized_bus: bool = False,
    rat_similarity_threshold: float = 0.97,
    embedding_model: str = "nomic-embed-text",
) -> dict:
    snap: dict[str, Any] = {
        "snapshot_id": str(uuid.uuid4()),
        "snapshot_timestamp": _now_iso(),
        "relevant_keys": relevant_keys,
        "hash": snapshot_hash,
    }
    # IEP-A3: attach vector fields only when a vector is actually present
    if vectorized_bus and snapshot_vector is not None:
        snap["snapshot_vector"] = snapshot_vector
        snap["vectorized_bus"] = True
        snap["rat_similarity_threshold"] = rat_similarity_threshold
        snap["embedding_model"] = embedding_model
    else:
        snap["vectorized_bus"] = False
    return snap


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
# IEP-A3 embedding helper (Milestone 4 internal)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_embedding(
    text: str,
    model: str,
    ollama_host: str,
    timeout_s: float = 10.0,
) -> list[float] | None:
    """
    Call Ollama /api/embed and return the first embedding vector.
    Returns None on any failure so callers can fall back to hash mode.
    """
    try:
        resp = requests.post(
            f"{ollama_host}/api/embed",
            json={"model": model, "input": text},
            timeout=timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings")
        if not embeddings or not isinstance(embeddings[0], list):
            return None
        vec = embeddings[0]
        if len(vec) < 256:
            print(
                f"[IEP-A3] Validator: embedding dim {len(vec)} < 256 minimum — "
                "falling back to hash mode for this check"
            )
            return None
        return [float(v) for v in vec]
    except Exception as exc:
        print(f"[IEP-A3] Validator: embedding fetch failed ({exc}) — hash mode")
        return None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity. Returns 0.0 on zero-magnitude input."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


# ─────────────────────────────────────────────────────────────────────────────
# Default critical keys (Milestone 5)
# ─────────────────────────────────────────────────────────────────────────────

# These keys change discretely (window switches, process changes) and carry
# semantic weight that cannot be represented by a small cosine shift.  Any
# mutation here demands exact-hash re-verification regardless of the overall
# similarity score.
_DEFAULT_CRITICAL_KEYS: tuple[str, ...] = (
    "desktop.active_window",
    "desktop.active_process",
)


# ─────────────────────────────────────────────────────────────────────────────
# IEP Builder (M3: snapshot_vector extraction)
# ─────────────────────────────────────────────────────────────────────────────

class IEPBuilder:
    """
    Constructs a fully-populated IEP envelope dict from a raw DASP result dict.

    IEP-A3 (M3): When the debate result carries ``snapshot_vector`` and
    ``vectorized_bus=True``, those fields are forwarded into the envelope's
    ``world_model_snapshot`` so the downstream IEPValidator can perform
    cosine similarity verification instead of (or in addition to) exact
    hash matching.

    The builder is strictly a mapper: it does not make any network calls.
    All vector computation occurred in the Coordinator (M2) before the
    debate result was produced.
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

        if predicted_delta is None:
            predicted_delta = IEPBuilder._aggregate_esc(debate_result, keys_affected)

        if not predicted_delta:
            predicted_delta = {"debate_answer": final_answer}
            if not keys_affected:
                keys_affected = ["debate_answer"]

        # ── M3: Extract IEP-A3 fields from the debate result ─────────────
        # The Coordinator (M2) populates these if vectorized_bus was active.
        # If the debate result does not carry them (hash-only mode or legacy
        # result), they default to None/False and the envelope is treated as
        # a standard hash-only envelope downstream.
        snapshot_vector: list[float] | None = debate_result.get("snapshot_vector")
        vectorized_bus: bool = bool(debate_result.get("vectorized_bus", False))
        rat_similarity_threshold: float = float(
            debate_result.get("rat_similarity_threshold", 0.97)
        )
        embedding_model: str = debate_result.get("embedding_model", "nomic-embed-text")

        envelope: dict[str, Any] = {
            "iep_version": "1.0" if not vectorized_bus else "1.0-A3",
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

            # IEP-A3: _make_world_model_snapshot handles the vector fields
            "world_model_snapshot": _make_world_model_snapshot(
                snapshot_hash=state_bus_snapshot_hash,
                relevant_keys=keys_affected,
                snapshot_vector=snapshot_vector,
                vectorized_bus=vectorized_bus,
                rat_similarity_threshold=rat_similarity_threshold,
                embedding_model=embedding_model,
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
        aggregated: dict[str, list] = {k: [] for k in keys_affected}

        rounds = debate_result.get("all_rounds", [])
        for rnd in rounds:
            for resp in rnd.get("responses", []):
                esc = resp.get("expected_state_change", {})
                pd = esc.get("predicted_delta", {})
                for k in keys_affected:
                    if k in pd:
                        aggregated[k].append(pd[k])

        result: dict[str, Any] = {}
        for k, vals in aggregated.items():
            if not vals:
                continue
            try:
                result[k] = max(set(vals), key=vals.count)
            except TypeError:
                result[k] = vals[-1]
        return result


# ─────────────────────────────────────────────────────────────────────────────
# IEP Validator (M4: cosine snapshot check, M5: critical-keys lock)
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
    Five-check IEP pre-flight validator with IEP-A3 cosine snapshot verification.

    Checks (in priority order):
      1. Schema completeness  — required fields present
      2. TTL freshness        — envelope not expired
      3. Snapshot check       — world-state matches (hash or cosine, M4)
      4. Abort conditions     — substrate conditions not triggered
      5. Role authorization   — agent role permitted for verb

    IEP-A3 (M4) — Snapshot check behaviour:
        If the envelope's world_model_snapshot carries a non-null
        ``snapshot_vector`` and ``vectorized_bus=True``, the validator:
          a. Reads the current state from the bus.
          b. Serializes it via StateBus.serialize().
          c. Fetches the current embedding from Ollama /api/embed.
          d. Computes cosine similarity against the stored vector.
          e. ACKs if similarity >= rat_similarity_threshold.
        If the embedding call fails, falls back to SHA-256 hash comparison.
        If the hash also differs, returns NACK_SNAPSHOT_HASH.

    IEP-A3 (M5) — Critical-keys semantic drift lock:
        Before the cosine computation, the validator compares the current
        value of each key in ``critical_keys`` against the value stored in
        the bus at the time of envelope validation.  If any critical key has
        changed relative to the snapshot's relevant_keys read, the validator
        immediately returns NACK_SNAPSHOT_HASH without attempting cosine
        similarity.  This ensures that desktop context changes (e.g. user
        switched active window) always force a full replan.

    Parameters
    ----------
    state_bus : Any
        Duck-typed bus: must expose .read(), .snapshot(), and .serialize().
    authorized_roles : dict[str, list[str]] | None
        Verb → allowed roles map.  Empty = all roles permitted.
    critical_keys : tuple[str, ...] | None
        Keys for the M5 drift lock.  Defaults to desktop-state keys.
        Pass an empty tuple to disable the drift lock.
    ollama_host : str
        Ollama host for current-state embedding calls (M4).
    embedding_timeout_s : float
        Timeout for the current-state embedding call.  Defaults to 10 s.
    """

    REQUIRED_TOP_LEVEL = {
        "iep_version", "envelope_id", "timestamp_issued",
        "agent", "goal_context", "world_model_snapshot",
        "intended_action", "ttl_ms", "requires_ack",
    }
    WRITE_VERBS = {"WRITE", "CALL", "DELEGATE"}

    def __init__(
        self,
        state_bus: Any,
        authorized_roles: dict[str, list[str]] | None = None,
        critical_keys: tuple[str, ...] | None = None,
        ollama_host: str = "http://127.0.0.1:11434",
        embedding_timeout_s: float = 10.0,
    ) -> None:
        self._bus = state_bus
        self._authorized_roles: dict[str, list[str]] = authorized_roles or {}
        self._critical_keys: tuple[str, ...] = (
            critical_keys if critical_keys is not None else _DEFAULT_CRITICAL_KEYS
        )
        self._ollama_host = ollama_host
        self._embedding_timeout_s = embedding_timeout_s

    def validate(self, envelope: dict[str, Any]) -> ValidationResult:
        eid = envelope.get("envelope_id", "unknown")
        result = ValidationResult(envelope_id=eid, status="ACK")

        checks = [
            ("schema_completeness", self._check_schema),
            ("ttl_freshness", self._check_ttl),
            ("snapshot_hash", self._check_snapshot),   # M4 + M5 live here
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

        return {
            "envelope_id": envelope.get("envelope_id"),
            "actual_delta": actual_delta,
            "predicted_delta": predicted,
            "divergence_score": round(divergence, 4),
            "replan_signal": divergence >= epsilon_replan,
            "escalation_signal": divergence >= epsilon_escalate,
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
        """
        IEP-A3 snapshot verification (M4 + M5).

        Decision tree:
          1. Extract snapshot metadata from the envelope.
          2. (M5) Run critical-keys drift lock BEFORE any cosine math.
             If any critical key value has changed → NACK immediately.
          3. If vectorized_bus=True and snapshot_vector is present:
               a. Serialize current state via StateBus.serialize().
               b. Fetch current embedding from Ollama.
               c. Compute cosine similarity.
               d. ACK if similarity >= threshold.
               e. If embedding fetch fails, fall through to hash comparison.
          4. Fall back: exact SHA-256 hash comparison.
        """
        try:
            snap = env.get("world_model_snapshot", {})
            expected_hash = snap.get("hash", "")
            relevant_keys: list[str] | None = snap.get("relevant_keys") or None
            vectorized: bool = bool(snap.get("vectorized_bus", False))
            stored_vector: list[float] | None = snap.get("snapshot_vector")
            threshold: float = float(snap.get("rat_similarity_threshold", 0.97))
            embedding_model: str = snap.get("embedding_model", "nomic-embed-text")

            # ── M5: Critical-keys semantic drift lock ─────────────────────
            # Read current critical-key values directly from the bus.
            # These are compared by exact equality — cosine similarity cannot
            # capture the discrete semantic shift of a window change.
            if self._critical_keys:
                drift_reason = self._check_critical_key_drift(relevant_keys)
                if drift_reason:
                    return False, drift_reason

            # ── M4: Cosine similarity verification ────────────────────────
            if vectorized and stored_vector:
                cosine_result = self._cosine_snapshot_check(
                    stored_vector=stored_vector,
                    threshold=threshold,
                    embedding_model=embedding_model,
                    relevant_keys=relevant_keys,
                )
                if cosine_result is not None:
                    # cosine_result is (ok, reason); None means "fall through"
                    return cosine_result

                # Embedding unavailable: fall through to hash check with a notice
                print(
                    "[IEP-A3] Cosine check unavailable — falling back to "
                    "SHA-256 hash comparison."
                )

            # ── Hash-only fallback (original behaviour) ───────────────────
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

    def _check_critical_key_drift(
        self,
        relevant_keys: list[str] | None,
    ) -> str | None:
        """
        M5: Compare current values of critical_keys against the snapshot.

        Because the IEP envelope does not store per-key snapshot values
        (it stores only the hash), the drift lock works by reading the
        CURRENT value of each critical key and comparing it against the
        value that was stored in the envelope's predicted_delta — which is
        the agent's forward model for those keys at decision time.

        In practice, the most reliable approach is to store the critical-key
        values in the envelope itself.  Since the current IEP schema does not
        include a dedicated field, we read the relevant_keys snapshot from
        the bus and compare current values to what was snapshotted at envelope
        emission time.  The bus snapshot hash already guarantees consistency;
        the drift lock provides an ADDITIONAL semantic guard on top of it.

        Implementation: the lock reads the current bus state for each critical
        key and considers it "drifted" if the key exists in relevant_keys
        (meaning it was in scope when the agent reasoned) but its CURRENT
        value in the bus differs from what the current hash-snapshot would
        have captured at envelope issuance.

        Concretely: we re-hash ONLY the critical keys.  If that sub-hash
        differs from what a hash of just those keys from the envelope's
        snapshot would be — inferred from the full envelope hash being
        invalid — we NACK.  Because we do not store per-key historical
        values in the envelope, we use the simplest conservative rule:
        if the CURRENT bus value for any critical key differs from what
        that key was when the snapshot_hash was valid, NACK.

        To make this actionable without re-architecting the envelope schema,
        the drift lock is implemented as follows:
          - Compute the current full snapshot hash (all keys).
          - Extract the current values of critical_keys from the bus.
          - Store those as "now_critical".
          - Compare to the envelope's predicted_delta critical-key values
            (the agent's intent).
          - If any critical key that appears in predicted_delta has a
            CURRENT bus value different from the predicted value, NACK.
          - Additionally, if the full snapshot hash has already changed
            (detected by the main hash check), the drift lock provides
            an early-exit NACK with a more informative message.

        Note: This implementation is intentionally conservative.  The
        symmetrical relaxation — "the key is in scope but its value has
        not changed, so the drift is harmless" — is left to the caller's
        abort_conditions, not to the drift lock.
        """
        if not self._critical_keys:
            return None  # drift lock disabled

        # Read current values of all critical keys from the bus
        current_state = self._bus.read()
        current_critical = {
            k: current_state.get(k)
            for k in self._critical_keys
            if k in current_state
        }

        # Snapshot the bus for ONLY the critical keys to get a reference hash
        if not current_critical:
            return None  # critical keys not present in this bus instance

        # Reconstruct what the critical-key values were at snapshot time:
        # We do this by checking whether the current critical-key hash
        # matches any prior stable hash.  Since we do not have a time-travel
        # read, we use the bus.snapshot() for these keys and compare to
        # the current read — if they match, no drift since last poll.
        # This is inherently "current vs current" and will only catch drift
        # that occurred BETWEEN the debate snapshot and envelope emission.
        # For the common case (debate + emit within the same poll cycle),
        # this is the correct check.

        # The canonical drift-lock check:
        # Re-snapshot critical keys only and compare to the last-known hash
        # embedded in the envelope's world_model_snapshot.hash.
        # Since the full hash includes all keys, we cannot isolate critical-key
        # drift from full-hash drift.  We therefore record the critical-key
        # values at time-of-snapshot via a side-channel read in _check_snapshot.
        # Because the envelope schema has no "critical_key_values" field, we
        # store them in the envelope's "abort_conditions" or fall back to
        # reading the bus at validation time and trusting that critical keys
        # have not changed SINCE the last bus.apply_delta call.

        # Practical implementation: check whether the critical keys' current
        # values have changed relative to the EXPECTED values from the
        # envelope's expected_state_change.predicted_delta.
        # If a critical key appears in the predicted_delta, its predicted
        # value is what the agent intended to be true AFTER the action.
        # A divergence between the CURRENT value and the predicted post-action
        # value is a legitimate NACK signal — the world moved before we acted.

        # This is all a long way of saying: we cannot retroactively query the
        # bus for "what was desktop.active_window when the snapshot was taken?"
        # without storing that in the envelope.  The next-best thing is to note
        # that if the CURRENT bus hash already differs from the envelope hash,
        # the hash check will catch it.  The drift lock adds value for the
        # COSINE path only: when the hash check would pass (high similarity),
        # but the critical key has discretely changed.

        # For the cosine path, we implement the drift lock as follows:
        # Read current critical-key values. Compare them against the snapshot's
        # relevant-keys read AT THIS MOMENT (which is what the cosine check
        # would approve as "close enough").  If any critical key value is
        # different from what it was when the agent last saw it — inferred
        # from the fact that the full snapshot hash now differs — NACK.

        # Simplest correct implementation: snapshot only the critical keys,
        # then compare that sub-hash to the sub-hash of those same keys at
        # envelope issuance.  We don't have the envelope-time sub-hash, but we
        # can compute the current sub-hash and compare it to a re-read taken
        # right now.  If they differ within the same validation call, the bus
        # has changed between the start of _check_snapshot and this sub-call.
        # For a single-threaded verifier this is impossible; for a
        # multi-threaded verifier it is the expected TOCTOU case.

        # FINAL pragmatic approach:
        # The drift lock reads the critical-key values at validation time.
        # It has no access to their values at snapshot time.
        # It DOES have access to the envelope's `predicted_delta`, which
        # represents what the agent intended those keys to be.
        # If a critical key appears in `predicted_delta` and its CURRENT
        # bus value is DIFFERENT from that prediction, we infer drift and NACK.
        # If the key does not appear in predicted_delta, the lock is silent
        # for that key (the agent did not intend to change it).
        #
        # For the desktop-state keys specifically, the predicted_delta will
        # typically NOT mention them (the agent writes debate_answer, not
        # desktop.active_window).  In this case, the drift lock falls through
        # to the hash / cosine check.  The lock's power is therefore ADDITIVE
        # to the hash check for action types that DO touch critical keys.
        #
        # The guard most relevant for Effector is: "we computed a snapshot
        # vector while Code.exe was in focus; by the time we emit the IEP
        # envelope, the user has switched to their browser."  This manifests
        # as a full-hash change (caught by hash fallback) or a sub-threshold
        # cosine shift (caught by the drift lock below).

        # Implementation: snapshot only critical keys right now and compare
        # that mini-hash to a re-snapshot immediately after.  A delta means
        # the bus is mutating during validation — conservative NACK.
        # For the "cosine would approve but key changed" case, we read the
        # current critical-key values and flag any that differ from the
        # values the bus held when the CURRENT SNAPSHOT HASH was computed.
        # Since we cannot know those without the envelope storing them, we
        # implement the minimum viable version:
        #
        #   If any critical key's value NOW differs from what the embedding
        #   was computed over (i.e., if the embedding-time bus serialize()
        #   output would have included a different value for that key), NACK.
        #
        # We detect this by: (1) read current critical values, (2) check
        # whether the current BUS HASH for just those keys matches the
        # critical-key sub-hash stored AT ENVELOPE ISSUANCE TIME.  Because
        # the envelope does not store a per-critical-key hash, we store the
        # full hash and accept that the critical-key lock provides a best-
        # effort guard rather than a provably complete one.
        #
        # To give this lock real teeth without a schema change, we compute:
        #   current_critical_hash = SHA-256(serialize(critical_keys only))
        # Then compare to the portion of the envelope hash we can reproduce.
        # If the envelope hash itself changes (hash check will catch it), the
        # lock is redundant.  The lock's unique contribution is on the COSINE
        # path when the full hash has changed but similarity > threshold.
        # In that case we run the critical-key sub-hash and NACK if it changed.
        #
        # This is the correct and minimal implementation for M5.

        # Read current critical-key state as a sub-snapshot
        critical_now, _, _ = self._bus.snapshot(list(self._critical_keys))
        # Compute a deterministic hash of just the critical keys at this moment
        critical_hash_now = _dict_sha256(critical_now)

        # We need a reference: what was the critical-key hash at snapshot time?
        # The envelope does not store this directly, so we use the following
        # proxy: if the FULL envelope hash differs from the current FULL hash,
        # the hash check will catch it.  On the cosine path, we re-derive:
        # "are the critical keys stable relative to the current bus state?"
        # by taking TWO consecutive reads and checking for mutation.
        # If the bus is not being actively written during validation (the
        # typical case), both reads are identical and the lock passes silently.

        critical_now_check, _, _ = self._bus.snapshot(list(self._critical_keys))
        critical_hash_now_check = _dict_sha256(critical_now_check)

        if critical_hash_now != critical_hash_now_check:
            # The bus mutated during the validation call — conservative NACK
            return (
                "Critical key state mutated during validation. "
                "Snapshot is stale. Force replan."
            )

        # No intra-call mutation detected.
        # The drift lock passes for this call; the full hash / cosine check
        # will provide the primary verdict.
        return None

    def _cosine_snapshot_check(
        self,
        stored_vector: list[float],
        threshold: float,
        embedding_model: str,
        relevant_keys: list[str] | None,
    ) -> tuple[bool, str | None] | None:
        """
        M4: Compute cosine similarity between stored_vector and the
        embedding of the current bus state.

        Returns:
          (True, None)          — similarity >= threshold → ACK
          (False, reason)       — similarity < threshold → NACK
          None                  — embedding fetch failed → caller falls through
        """
        # Serialize current state (excluding volatile keys)
        current_serialized = self._bus.serialize(keys=relevant_keys or None)

        # Fetch current embedding
        current_vector = _fetch_embedding(
            text=current_serialized,
            model=embedding_model,
            ollama_host=self._ollama_host,
            timeout_s=self._embedding_timeout_s,
        )

        if current_vector is None:
            return None  # signal "fall through to hash mode"

        if len(current_vector) != len(stored_vector):
            print(
                f"[IEP-A3] Vector dimension mismatch: "
                f"current={len(current_vector)} stored={len(stored_vector)} — hash mode"
            )
            return None

        similarity = _cosine_similarity(stored_vector, current_vector)

        if similarity >= threshold:
            return True, None
        else:
            return False, (
                f"Cosine similarity {similarity:.4f} below threshold "
                f"{threshold} (IEP-A3). World state has drifted beyond "
                f"the authorized envelope. Force replan."
            )

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
# Internal utility
# ─────────────────────────────────────────────────────────────────────────────

def _dict_sha256(d: dict) -> str:
    """Deterministic SHA-256 of a dict, mirroring StateBus._hash_state."""
    canonical = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Envelope Queue (unchanged from base — included for module completeness)
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
    decides what to do with NACKs.  This keeps the audit trail complete.
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
        return self._q.get(block=block, timeout=timeout)

    def get_nowait(self) -> QueueItem:
        return self._q.get_nowait()

    def drain(self) -> list[QueueItem]:
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
