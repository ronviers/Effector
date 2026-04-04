"""
main_loop_reflex.py — Step 3: Main Loop Integration

This module provides the orchestration-layer wiring for the Reflex Middleware.
It is designed to be imported and called from main_loop.py and run_cmd.py
rather than replacing them.

Orchestration cycle (per trigger/task):
  1. IntentRouter.route(task)  — deterministic, sub-ms
  2. If intent found → compute embedding → ReflexEngine.evaluate_reflex()
  3. If EXECUTED     → done (skip DASP entirely)
  4. If BYPASSED     → fall through to DASPCoordinator
  5. If NACK_*       → log + fall through to DASPCoordinator for re-deliberation
  6. DASPCoordinator.run() → on session_complete, store resulting RAT

The embedding call (step 2) is the only I/O on the fast path (~20–50ms).
The math inside ReflexEngine itself runs in <2ms.

Architecture decisions (DO NOT RELITIGATE)
------------------------------------------
- Embedding via local Ollama nomic-embed-text, called BEFORE ReflexEngine.
- ReflexEngine makes zero network calls.
- Post-execution divergence scoring is async (non-blocking).
- RAT issuance happens inside DASP session_complete hook.
"""

from __future__ import annotations

import time
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

import requests

from effector.intent_router import IntentRouter, IntendedAction
from effector.rat_store import LocalRATStore
from effector.reflex_engine import ReflexEngine, ReflexResult, ReflexStatus


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def fetch_embedding(
    text: str,
    model: str = "nomic-embed-text",
    host: str = "http://127.0.0.1:11434",
    timeout_s: float = 10.0,
) -> list[float] | None:
    """
    Fetch a dense embedding from the local Ollama instance.

    Returns None on failure — the ReflexEngine will fall back to hash mode.
    This is the ONLY network call on the reflex fast path.
    """
    try:
        resp = requests.post(
            f"{host}/api/embed",
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
                f"[ReflexPath] Embedding dim {len(vec)} < 256 — hash mode will be used."
            )
            return None
        return [float(v) for v in vec]
    except requests.exceptions.ConnectionError:
        print(f"[ReflexPath] Embedding model {model!r} unreachable — hash mode.")
        return None
    except Exception as exc:
        print(f"[ReflexPath] Embedding error: {exc} — hash mode.")
        return None


# ---------------------------------------------------------------------------
# RAT issuance helper (called from DASP session_complete hook)
# ---------------------------------------------------------------------------

def rat_from_debate_result(
    debate_result: dict[str, Any],
    state_bus: Any,
    intended_action: dict[str, Any] | None = None,
    rat_ttl_ms: int = 3_600_000,          # 1 hour default
    rat_min_confidence: float = 0.7,
    rat_similarity_threshold: float = 0.97,
    max_executions: int = -1,              # -1 = unlimited
    embedding_model: str = "nomic-embed-text",
    snapshot_vector: list[float] | None = None,
) -> dict[str, Any] | None:
    """
    Build a RAT from a completed DASP session result (IEP §6.2 schema).

    Only issues a RAT if:
      - consensus_score >= rat_min_confidence
      - A specific intended_action is provided (so we know what to pre-authorise)

    Returns the RAT dict (ready to pass to LocalRATStore.store_rat), or None
    if conditions are not met.
    """
    consensus_score = float(debate_result.get("consensus_score", 0.0))
    if consensus_score < rat_min_confidence:
        print(
            f"[ReflexPath] Consensus score {consensus_score:.3f} < "
            f"rat_min_confidence {rat_min_confidence} — no RAT issued."
        )
        return None

    if intended_action is None:
        return None

    snapshot_hash, _, _ = state_bus.snapshot()
    issued_at = datetime.now(timezone.utc).isoformat()
    session_id = debate_result.get("session_id", str(uuid.uuid4()))

    return {
        "rat_id": str(uuid.uuid4()),
        "issued_by_session": session_id,
        "issued_at": issued_at,
        "rat_ttl_ms": rat_ttl_ms,
        "rat_min_confidence": rat_min_confidence,
        "rat_similarity_threshold": rat_similarity_threshold,
        "authorized_actions": [
            {
                "verb": intended_action.get("verb", "WRITE"),
                "target": intended_action.get("target", ""),
                "max_executions": max_executions,
                "parameter_constraints": intended_action.get("parameters", {}),
            }
        ],
        "issuing_coalition": debate_result.get(
            "winning_coalition",
            debate_result.get("tier1_agents", []),
        ),
        "snapshot_hash": snapshot_hash,
        "snapshot_vector": snapshot_vector,
        "embedding_model": embedding_model,
    }


# ---------------------------------------------------------------------------
# ReflexOrchestrator — drop-in wrapper for the main loop
# ---------------------------------------------------------------------------

class ReflexOrchestrator:
    """
    Wraps the Reflex Middleware + IntentRouter into a single orchestration
    object that the main loop calls once per incoming task.

    Usage in main_loop.py / run_cmd.py
    ------------------------------------
        orchestrator = ReflexOrchestrator(
            state_bus=bus,
            rat_store=LocalRATStore(),
            dasp_run_fn=coordinator.run,          # existing DASP callable
            embedding_model="nomic-embed-text",
            ollama_host="http://127.0.0.1:11434",
        )

        # Per-task call:
        result = orchestrator.handle(task, snapshot_hash)
    """

    def __init__(
        self,
        state_bus: Any,
        rat_store: LocalRATStore,
        dasp_run_fn: Callable[..., dict[str, Any]],
        *,
        embedding_model: str = "nomic-embed-text",
        ollama_host: str = "http://127.0.0.1:11434",
        rat_ttl_ms: int = 3_600_000,
        rat_min_confidence: float = 0.7,
        rat_similarity_threshold: float = 0.97,
        on_reflex_executed: Callable[[ReflexResult], None] | None = None,
        on_dasp_complete: Callable[[dict[str, Any]], None] | None = None,
        on_post_execute: Callable[[Any, dict, dict], None] | None = None,
    ) -> None:
        self._bus = state_bus
        self._rat_store = rat_store
        self._dasp_run_fn = dasp_run_fn
        self._embedding_model = embedding_model
        self._ollama_host = ollama_host
        self._rat_ttl_ms = rat_ttl_ms
        self._rat_min_confidence = rat_min_confidence
        self._rat_similarity_threshold = rat_similarity_threshold
        self._on_reflex_executed = on_reflex_executed
        self._on_dasp_complete = on_dasp_complete

        self._router = IntentRouter()
        self._engine = ReflexEngine(
            rat_store=rat_store,
            on_post_execute=on_post_execute,
        )

    def handle(
        self,
        task: str,
        snapshot_hash: str,
        execute_fn: Callable | None = None,
        **dasp_kwargs: Any,
    ) -> dict[str, Any]:
        """
        Full sense-reflex-act cycle for a single task.

        Returns a unified result dict with `_path` key indicating
        "reflex" or "dasp" for downstream logging.
        """
        t0 = time.monotonic()

        # ── Step 1: Deterministic intent parsing ───────────────────────────
        intent: IntendedAction | None = self._router.route(task)

        if intent is not None:
            # ── Step 2: Compute embedding (only I/O on fast path) ──────────
            serialized_state = self._bus.serialize()
            current_vector = fetch_embedding(
                text=serialized_state,
                model=self._embedding_model,
                host=self._ollama_host,
            )

            # ── Step 3: Evaluate reflex ────────────────────────────────────
            reflex_result = self._engine.evaluate_reflex(
                intended_action=intent.as_dict(),
                current_state_vector=current_vector or [],
                state_bus=self._bus,
                execute_fn=execute_fn,
            )

            elapsed_ms = (time.monotonic() - t0) * 1000

            if reflex_result.status == ReflexStatus.EXECUTED:
                if self._on_reflex_executed:
                    try:
                        self._on_reflex_executed(reflex_result)
                    except Exception as exc:
                        print(f"[ReflexOrchestrator] on_reflex_executed error: {exc}")
                return {
                    "_path": "reflex",
                    "_elapsed_ms": round(elapsed_ms, 2),
                    "status": reflex_result.status.value,
                    "rat_id": reflex_result.rat_id,
                    "matched_action": reflex_result.matched_action,
                    "executions_remaining": reflex_result.executions_remaining,
                    "actual_delta": reflex_result.actual_delta,
                    "executed_at": reflex_result.executed_at,
                }

            # NACK or BYPASSED — log and fall through to DASP
            print(
                f"[ReflexOrchestrator] Reflex {reflex_result.status.value} "
                f"({elapsed_ms:.1f}ms) — {reflex_result.failure_reason!r} "
                f"— falling through to DASP."
            )

        # ── Step 4: DASP LLM debate fallback ──────────────────────────────
        debate_result = self._dasp_run_fn(
            task=task,
            snapshot_hash=snapshot_hash,
            state_bus=self._bus,
            **dasp_kwargs,
        )

        elapsed_ms = (time.monotonic() - t0) * 1000

        # ── Step 5: Issue RAT for next identical trigger ───────────────────
        if intent is not None:
            self._maybe_issue_rat(
                debate_result=debate_result,
                intended_action=intent.as_dict(),
            )

        if self._on_dasp_complete:
            try:
                self._on_dasp_complete(debate_result)
            except Exception as exc:
                print(f"[ReflexOrchestrator] on_dasp_complete error: {exc}")

        debate_result["_path"] = "dasp"
        debate_result["_elapsed_ms"] = round(elapsed_ms, 2)
        return debate_result

    def _maybe_issue_rat(
        self,
        debate_result: dict[str, Any],
        intended_action: dict[str, Any],
    ) -> None:
        """
        If the DASP session reached sufficient consensus, issue a RAT so
        the next identical trigger takes the reflex path.
        """
        # Compute embedding for the RAT snapshot asynchronously
        def _issue() -> None:
            serialized = self._bus.serialize()
            vector = fetch_embedding(
                text=serialized,
                model=self._embedding_model,
                host=self._ollama_host,
            )
            rat = rat_from_debate_result(
                debate_result=debate_result,
                state_bus=self._bus,
                intended_action=intended_action,
                rat_ttl_ms=self._rat_ttl_ms,
                rat_min_confidence=self._rat_min_confidence,
                rat_similarity_threshold=self._rat_similarity_threshold,
                snapshot_vector=vector,
                embedding_model=self._embedding_model,
            )
            if rat:
                self._rat_store.store_rat(rat)
                print(
                    f"[ReflexOrchestrator] RAT issued: {rat['rat_id'][:8]}… "
                    f"for {intended_action['verb']} {intended_action['target']}"
                )

        t = threading.Thread(target=_issue, name="RATIssuance", daemon=True)
        self._rat_thread = t
        t.start()

    def shutdown(self) -> None:
        """Graceful teardown."""
        if hasattr(self, '_rat_thread') and self._rat_thread and self._rat_thread.is_alive():
            self._rat_thread.join(timeout=5.0)
        self._engine.shutdown()
        self._rat_store.close()
