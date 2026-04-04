"""
test_reflex_middleware.py — Step 4: End-to-End Tests

Tests the full "Sense-Reflex-Act" loop using mocked Ollama embeddings.
Verifies: SQLite execution decrements, M4/M5 logic, TTL expiry, and
that the check order mirrors IEPVerifier exactly.

Run with:
    cd H:\\Effector
    pytest tests/test_reflex_middleware.py -v
"""

from __future__ import annotations

import json
import math
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Minimal StateBus stub (avoids importing the full effector package)
# ---------------------------------------------------------------------------

class _FakeStateBus:
    """Minimal StateBus compatible with ReflexEngine + LocalRATStore."""

    def __init__(self, initial: dict | None = None) -> None:
        self._state: dict[str, Any] = dict(initial or {})
        self._deltas: list[dict] = []

    def read(self, keys=None) -> dict:
        if keys is None:
            return dict(self._state)
        return {k: self._state.get(k) for k in keys}

    def snapshot(self, keys=None) -> tuple[str, Any, dict]:
        import hashlib, datetime as dt
        state = self.read(keys)
        canonical = json.dumps(state, sort_keys=True, default=str)
        h = hashlib.sha256(canonical.encode()).hexdigest()
        return h, dt.datetime.now(), state

    def serialize(self, keys=None, volatile_keys=None) -> str:
        state = self.read(keys)
        parts = [f"{k}: {v}" for k, v in sorted(state.items())]
        return " | ".join(parts)

    def apply_delta(self, envelope_id, delta, agent_id, session_id=None) -> dict:
        self._state.update(delta)
        self._deltas.append({"envelope_id": envelope_id, "delta": delta})
        return delta

    def set(self, key: str, value: Any) -> None:
        self._state[key] = value


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rat(
    *,
    rat_ttl_ms: int = 3_600_000,
    verb: str = "WRITE",
    target: str = "desktop.overlay.glimmer",
    max_executions: int = -1,
    snapshot_vector: list[float] | None = None,
    snapshot_hash: str = "",
    rat_similarity_threshold: float = 0.97,
    issued_at: str | None = None,
) -> dict[str, Any]:
    from datetime import datetime, timezone
    if issued_at is None:
        issued_at = datetime.now(timezone.utc).isoformat()
    return {
        "rat_id": str(uuid.uuid4()),
        "issued_by_session": str(uuid.uuid4()),
        "issued_at": issued_at,
        "rat_ttl_ms": rat_ttl_ms,
        "rat_min_confidence": 0.7,
        "rat_similarity_threshold": rat_similarity_threshold,
        "authorized_actions": [
            {
                "verb": verb,
                "target": target,
                "max_executions": max_executions,
                "parameter_constraints": {},
            }
        ],
        "issuing_coalition": ["agent1", "agent2"],
        "snapshot_hash": snapshot_hash,
        "snapshot_vector": snapshot_vector,
        "embedding_model": "nomic-embed-text",
    }


def _unit_vector(dim: int, value: float = 1.0) -> list[float]:
    """Return a vector of `dim` ones normalized to unit length."""
    mag = math.sqrt(dim) * value
    return [value / mag] * dim


@pytest.fixture
def db_path(tmp_path):
    """Fresh SQLite DB path for each test."""
    return tmp_path / "rats.db"


# ---------------------------------------------------------------------------
# LocalRATStore tests
# ---------------------------------------------------------------------------

class TestLocalRATStore:
    def test_store_and_retrieve(self, db_path):
        from effector.rat_store import LocalRATStore
        store = LocalRATStore(db_path=db_path)
        rat = _make_rat()
        store.store_rat(rat)

        record = store.get_rat(rat["rat_id"])
        assert record is not None
        assert record.rat_id == rat["rat_id"]
        assert record.executions_remaining == -1  # unlimited
        store.close()

    def test_candidate_matching(self, db_path):
        from effector.rat_store import LocalRATStore
        store = LocalRATStore(db_path=db_path)

        rat_write = _make_rat(verb="WRITE", target="desktop.overlay.glimmer")
        rat_read  = _make_rat(verb="READ",  target="os.desktop")
        store.store_rat(rat_write)
        store.store_rat(rat_read)

        write_candidates = store.get_candidate_rats("WRITE", "desktop.overlay.glimmer")
        assert len(write_candidates) == 1
        assert write_candidates[0].rat_id == rat_write["rat_id"]

        read_candidates = store.get_candidate_rats("READ", "os.desktop")
        assert len(read_candidates) == 1
        assert read_candidates[0].rat_id == rat_read["rat_id"]

        # No match
        assert store.get_candidate_rats("WRITE", "nonexistent.target") == []
        store.close()

    def test_prefix_target_matching(self, db_path):
        """Authorized 'desktop.overlay' should cover 'desktop.overlay.glimmer'."""
        from effector.rat_store import LocalRATStore
        store = LocalRATStore(db_path=db_path)
        rat = _make_rat(verb="WRITE", target="desktop.overlay")
        store.store_rat(rat)

        candidates = store.get_candidate_rats("WRITE", "desktop.overlay.glimmer")
        assert len(candidates) == 1
        store.close()

    def test_atomic_decrement(self, db_path):
        from effector.rat_store import LocalRATStore
        store = LocalRATStore(db_path=db_path)
        rat = _make_rat(max_executions=3)
        store.store_rat(rat)
        rid = rat["rat_id"]

        assert store.decrement_and_fetch(rid) == 2
        assert store.decrement_and_fetch(rid) == 1
        assert store.decrement_and_fetch(rid) == 0
        # Now exhausted — next decrement returns None
        assert store.decrement_and_fetch(rid) is None
        store.close()

    def test_unlimited_decrement(self, db_path):
        from effector.rat_store import LocalRATStore
        store = LocalRATStore(db_path=db_path)
        rat = _make_rat(max_executions=-1)
        store.store_rat(rat)
        rid = rat["rat_id"]

        for _ in range(100):
            assert store.decrement_and_fetch(rid) == -1
        store.close()

    def test_expired_rat_not_returned(self, db_path):
        from effector.rat_store import LocalRATStore
        from datetime import datetime, timezone

        # Issue a RAT that is already expired (ttl = 1ms in the past)
        issued_at = datetime.now(timezone.utc).isoformat()
        store = LocalRATStore(db_path=db_path)
        rat = _make_rat(rat_ttl_ms=1, issued_at=issued_at)
        store.store_rat(rat)
        time.sleep(0.01)  # let it expire

        assert store.get_rat(rat["rat_id"]) is None
        assert store.get_candidate_rats("WRITE", "desktop.overlay.glimmer") == []
        store.close()

    def test_invalidate(self, db_path):
        from effector.rat_store import LocalRATStore
        store = LocalRATStore(db_path=db_path)
        rat = _make_rat()
        store.store_rat(rat)
        assert store.get_rat(rat["rat_id"]) is not None

        removed = store.invalidate_rat(rat["rat_id"])
        assert removed is True
        assert store.get_rat(rat["rat_id"]) is None
        store.close()

    def test_purge_expired(self, db_path):
        from effector.rat_store import LocalRATStore
        from datetime import datetime, timezone

        store = LocalRATStore(db_path=db_path)
        # Store one expired + one live
        expired = _make_rat(rat_ttl_ms=1)
        live    = _make_rat(rat_ttl_ms=3_600_000)
        store.store_rat(expired)
        store.store_rat(live)
        time.sleep(0.01)

        removed = store.purge_expired()
        assert removed == 1
        assert store.get_rat(live["rat_id"]) is not None
        store.close()


# ---------------------------------------------------------------------------
# IntentRouter tests
# ---------------------------------------------------------------------------

class TestIntentRouter:
    def test_spawn_glimmer(self):
        from effector.intent_router import IntentRouter
        r = IntentRouter()
        intent = r.route("spawn glimmer on tax_returns.pdf")
        assert intent is not None
        assert intent.verb == "WRITE"
        assert "glimmer" in intent.target

    def test_poll_telemetry(self):
        from effector.intent_router import IntentRouter
        r = IntentRouter()
        intent = r.route("poll telemetry now")
        assert intent is not None
        assert intent.verb == "READ"

    def test_no_match_returns_none(self):
        from effector.intent_router import IntentRouter
        r = IntentRouter()
        intent = r.route("the quick brown fox does nothing in particular")
        assert intent is None

    def test_case_insensitive(self):
        from effector.intent_router import IntentRouter
        r = IntentRouter()
        assert r.route("SPAWN GLIMMER") is not None
        assert r.route("Spawn Glimmer") is not None

    def test_dim_brightness(self):
        from effector.intent_router import IntentRouter
        r = IntentRouter()
        intent = r.route("dim monitor because it is raining outside")
        assert intent is not None
        assert intent.verb == "WRITE"
        assert "brightness" in intent.target

    def test_spotify_hook(self):
        from effector.intent_router import IntentRouter
        r = IntentRouter()
        intent = r.route("spotify opened")
        assert intent is not None
        assert "music" in intent.target

    def test_extra_route_takes_priority(self):
        from effector.intent_router import IntentRouter
        r = IntentRouter()
        r.add_route(r"custom action", "CALL", "custom.target")
        intent = r.route("custom action now")
        assert intent is not None
        assert intent.verb == "CALL"
        assert intent.target == "custom.target"


# ---------------------------------------------------------------------------
# ReflexEngine tests
# ---------------------------------------------------------------------------

class TestReflexEngine:
    def _make_engine(self, db_path, **kwargs):
        from effector.rat_store import LocalRATStore
        from effector.reflex_engine import ReflexEngine
        store = LocalRATStore(db_path=db_path)
        engine = ReflexEngine(rat_store=store, **kwargs)
        return engine, store

    def test_bypassed_when_no_rat(self, db_path):
        from effector.reflex_engine import ReflexStatus
        engine, store = self._make_engine(db_path)
        bus = _FakeStateBus()

        result = engine.evaluate_reflex(
            {"verb": "WRITE", "target": "desktop.overlay.glimmer"},
            [],
            bus,
        )
        assert result.status == ReflexStatus.BYPASSED
        store.close()
        engine.shutdown()

    def test_executed_unlimited_rat(self, db_path):
        from effector.reflex_engine import ReflexStatus
        engine, store = self._make_engine(db_path)
        bus = _FakeStateBus()

        rat = _make_rat(verb="WRITE", target="desktop.overlay.glimmer")
        store.store_rat(rat)

        result = engine.evaluate_reflex(
            {"verb": "WRITE", "target": "desktop.overlay.glimmer", "parameters": {"x": 1}},
            [],  # no vector → hash fallback with empty snapshot_hash → passes
            bus,
        )
        assert result.status == ReflexStatus.EXECUTED
        assert result.rat_id == rat["rat_id"]
        store.close()
        engine.shutdown()

    def test_nack_expired(self, db_path):
        from effector.reflex_engine import ReflexStatus
        from datetime import datetime, timezone

        engine, store = self._make_engine(db_path)
        bus = _FakeStateBus()

        rat = _make_rat(rat_ttl_ms=1)
        store.store_rat(rat)
        time.sleep(0.01)  # let it expire in the DB
        # Force the in-memory record to also appear expired by using an old issued_at
        # The DB query filters by ttl_expiry_timestamp, so we just need to wait.

        result = engine.evaluate_reflex(
            {"verb": "WRITE", "target": "desktop.overlay.glimmer"},
            [],
            bus,
        )
        # No live RATs → BYPASSED (expired rows filtered by DB query)
        assert result.status == ReflexStatus.BYPASSED
        store.close()
        engine.shutdown()

    def test_nack_exhausted_after_decrement(self, db_path):
        from effector.reflex_engine import ReflexStatus
        engine, store = self._make_engine(db_path)
        bus = _FakeStateBus()

        rat = _make_rat(verb="WRITE", target="desktop.overlay.glimmer", max_executions=1)
        store.store_rat(rat)

        # First call: should succeed
        r1 = engine.evaluate_reflex(
            {"verb": "WRITE", "target": "desktop.overlay.glimmer"},
            [], bus,
        )
        assert r1.status == ReflexStatus.EXECUTED

        # Second call: should be exhausted
        r2 = engine.evaluate_reflex(
            {"verb": "WRITE", "target": "desktop.overlay.glimmer"},
            [], bus,
        )
        assert r2.status in (ReflexStatus.NACK_EXHAUSTED, ReflexStatus.BYPASSED)
        store.close()
        engine.shutdown()

    def test_m4_cosine_pass(self, db_path):
        from effector.reflex_engine import ReflexStatus
        engine, store = self._make_engine(db_path)
        bus = _FakeStateBus()

        dim = 512
        vec = _unit_vector(dim)
        rat = _make_rat(
            verb="WRITE",
            target="desktop.overlay.glimmer",
            snapshot_vector=vec,
            rat_similarity_threshold=0.95,
        )
        store.store_rat(rat)

        # Nearly identical vector → similarity ≈ 1.0
        similar_vec = [v + 0.0001 for v in vec]
        result = engine.evaluate_reflex(
            {"verb": "WRITE", "target": "desktop.overlay.glimmer"},
            similar_vec,
            bus,
        )
        assert result.status == ReflexStatus.EXECUTED
        store.close()
        engine.shutdown()

    def test_m4_cosine_fail(self, db_path):
        from effector.reflex_engine import ReflexStatus
        engine, store = self._make_engine(db_path)
        bus = _FakeStateBus()

        dim = 512
        vec_a = _unit_vector(dim, 1.0)
        vec_b = [-v for v in vec_a]      # exact negation → cosine ≈ -1.0

        rat = _make_rat(
            verb="WRITE",
            target="desktop.overlay.glimmer",
            snapshot_vector=vec_a,
            rat_similarity_threshold=0.97,
        )
        store.store_rat(rat)

        result = engine.evaluate_reflex(
            {"verb": "WRITE", "target": "desktop.overlay.glimmer"},
            vec_b,
            bus,
        )
        assert result.status == ReflexStatus.NACK_LOW_SIMILARITY
        assert result.similarity_score is not None
        store.close()
        engine.shutdown()

    def test_m5_critical_key_mutation_detected(self, db_path):
        """
        M5 check fires when critical keys mutate between two rapid reads.
        We test this by patching StateBus.read to return different values
        on successive calls.
        """
        from effector.reflex_engine import ReflexEngine, ReflexStatus
        from effector.rat_store import LocalRATStore

        store = LocalRATStore(db_path=db_path)
        rat = _make_rat(verb="WRITE", target="desktop.overlay.glimmer")
        store.store_rat(rat)

        call_count = [0]

        class _MutatingBus(_FakeStateBus):
            def read(self, keys=None):
                call_count[0] += 1
                # _check_critical_key_drift calls read() exactly twice:
                #   Call 1: "Firefox" → critical_state
                #   Call 2: "Chrome"  → critical_state2 → hashes differ → NACK
                window = "Firefox" if call_count[0] == 1 else "Chrome"
                return {"desktop.active_window": window}

        bus = _MutatingBus()
        engine = ReflexEngine(rat_store=store)

        result = engine.evaluate_reflex(
            {"verb": "WRITE", "target": "desktop.overlay.glimmer"},
            [],
            bus,
        )
        # M5 should detect the mutation and return NACK_CRITICAL_DRIFT
        assert result.status == ReflexStatus.NACK_CRITICAL_DRIFT
        store.close()
        engine.shutdown()

    def test_hash_fallback_when_no_vector(self, db_path):
        """When RAT has no snapshot_vector and no snapshot_hash, engine passes."""
        from effector.reflex_engine import ReflexStatus
        engine, store = self._make_engine(db_path)
        bus = _FakeStateBus()

        rat = _make_rat(snapshot_vector=None, snapshot_hash="")
        store.store_rat(rat)

        result = engine.evaluate_reflex(
            {"verb": "WRITE", "target": "desktop.overlay.glimmer"},
            [],
            bus,
        )
        assert result.status == ReflexStatus.EXECUTED
        store.close()
        engine.shutdown()

    def test_execute_fn_called(self, db_path):
        """Custom execute_fn receives the intended_action and its return is recorded."""
        from effector.reflex_engine import ReflexStatus
        engine, store = self._make_engine(db_path)
        bus = _FakeStateBus()

        rat = _make_rat()
        store.store_rat(rat)

        executed_with = []

        def custom_execute(action):
            executed_with.append(action)
            return {"custom_key": "custom_value"}

        result = engine.evaluate_reflex(
            {"verb": "WRITE", "target": "desktop.overlay.glimmer"},
            [],
            bus,
            execute_fn=custom_execute,
        )
        assert result.status == ReflexStatus.EXECUTED
        assert len(executed_with) == 1
        assert result.actual_delta == {"custom_key": "custom_value"}
        store.close()
        engine.shutdown()

    def test_post_execute_callback_fired_async(self, db_path):
        """on_post_execute is called asynchronously after EXECUTED."""
        import threading
        from effector.reflex_engine import ReflexStatus
        from effector.rat_store import LocalRATStore

        store = LocalRATStore(db_path=db_path)
        rat = _make_rat()
        store.store_rat(rat)

        callback_event = threading.Event()
        callback_args = []

        def on_post(rat_record, action, delta):
            callback_args.append((rat_record, action, delta))
            callback_event.set()

        from effector.reflex_engine import ReflexEngine
        engine = ReflexEngine(rat_store=store, on_post_execute=on_post)
        bus = _FakeStateBus()

        result = engine.evaluate_reflex(
            {"verb": "WRITE", "target": "desktop.overlay.glimmer", "parameters": {"k": 1}},
            [],
            bus,
        )
        assert result.status == ReflexStatus.EXECUTED
        callback_event.wait(timeout=2.0)
        assert len(callback_args) == 1
        store.close()
        engine.shutdown()


# ---------------------------------------------------------------------------
# ReflexOrchestrator integration tests
# ---------------------------------------------------------------------------

class TestReflexOrchestrator:
    def test_reflex_path_taken(self, db_path):
        """When a RAT exists and state matches, the reflex path is taken."""
        from effector.rat_store import LocalRATStore
        from effector.main_loop_reflex import ReflexOrchestrator

        store = LocalRATStore(db_path=db_path)
        rat = _make_rat(verb="WRITE", target="desktop.overlay.glimmer")
        store.store_rat(rat)

        bus = _FakeStateBus()
        dasp_called = []

        def fake_dasp(**kwargs):
            dasp_called.append(kwargs)
            return {"final_answer": "dasp_answer", "consensus_score": 0.9, "session_id": str(uuid.uuid4())}

        orchestrator = ReflexOrchestrator(
            state_bus=bus,
            rat_store=store,
            dasp_run_fn=fake_dasp,
            embedding_model="nomic-embed-text",
        )

        # Patch fetch_embedding to return empty (triggers hash fallback)
        import effector.main_loop_reflex as mlr
        original = mlr.fetch_embedding
        mlr.fetch_embedding = lambda **kw: []

        try:
            result = orchestrator.handle(
                task="spawn glimmer on tax_returns.pdf",
                snapshot_hash="a" * 64,
            )
        finally:
            mlr.fetch_embedding = original

        assert result["_path"] == "reflex"
        assert len(dasp_called) == 0
        store.close()
        orchestrator.shutdown()

    def test_dasp_fallback_when_no_rat(self, db_path):
        """When no RAT matches, DASP is called."""
        from effector.rat_store import LocalRATStore
        from effector.main_loop_reflex import ReflexOrchestrator

        store = LocalRATStore(db_path=db_path)
        bus = _FakeStateBus()
        dasp_called = []

        def fake_dasp(**kwargs):
            dasp_called.append(kwargs)
            return {
                "final_answer": "ok",
                "consensus_score": 0.85,
                "session_id": str(uuid.uuid4()),
                "tier1_agents": [],
                "winning_coalition": [],
            }

        orchestrator = ReflexOrchestrator(
            state_bus=bus,
            rat_store=store,
            dasp_run_fn=fake_dasp,
        )

        result = orchestrator.handle(
            task="spawn glimmer on desktop",
            snapshot_hash="a" * 64,
        )

        assert result["_path"] == "dasp"
        assert len(dasp_called) == 1
        store.close()
        orchestrator.shutdown()

    def test_dasp_fallback_on_unknown_task(self, db_path):
        """Unrecognised task always goes to DASP (no router match)."""
        from effector.rat_store import LocalRATStore
        from effector.main_loop_reflex import ReflexOrchestrator

        store = LocalRATStore(db_path=db_path)
        bus = _FakeStateBus()
        dasp_called = []

        def fake_dasp(**kwargs):
            dasp_called.append(True)
            return {
                "final_answer": "debated",
                "consensus_score": 0.8,
                "session_id": str(uuid.uuid4()),
                "tier1_agents": [],
                "winning_coalition": [],
            }

        orchestrator = ReflexOrchestrator(
            state_bus=bus,
            rat_store=store,
            dasp_run_fn=fake_dasp,
        )

        result = orchestrator.handle(
            task="something completely unrecognised and abstract",
            snapshot_hash="b" * 64,
        )

        assert result["_path"] == "dasp"
        assert len(dasp_called) == 1
        store.close()
        orchestrator.shutdown()


# ---------------------------------------------------------------------------
# rat_from_debate_result tests
# ---------------------------------------------------------------------------

class TestRatFromDebateResult:
    def test_issues_rat_on_high_consensus(self):
        from effector.main_loop_reflex import rat_from_debate_result
        bus = _FakeStateBus({"key": "value"})
        debate = {
            "final_answer": "spawn glimmer",
            "consensus_score": 0.9,
            "session_id": str(uuid.uuid4()),
            "tier1_agents": ["agent1"],
            "winning_coalition": ["agent1"],
        }
        action = {"verb": "WRITE", "target": "desktop.overlay.glimmer"}
        rat = rat_from_debate_result(debate, bus, intended_action=action)
        assert rat is not None
        assert rat["authorized_actions"][0]["verb"] == "WRITE"
        assert rat["authorized_actions"][0]["target"] == "desktop.overlay.glimmer"

    def test_no_rat_on_low_consensus(self):
        from effector.main_loop_reflex import rat_from_debate_result
        bus = _FakeStateBus()
        debate = {
            "final_answer": "uncertain",
            "consensus_score": 0.3,
            "session_id": str(uuid.uuid4()),
        }
        rat = rat_from_debate_result(
            debate, bus,
            intended_action={"verb": "WRITE", "target": "desktop.overlay"},
            rat_min_confidence=0.7,
        )
        assert rat is None

    def test_no_rat_without_intended_action(self):
        from effector.main_loop_reflex import rat_from_debate_result
        bus = _FakeStateBus()
        debate = {"final_answer": "ok", "consensus_score": 0.95, "session_id": str(uuid.uuid4())}
        rat = rat_from_debate_result(debate, bus, intended_action=None)
        assert rat is None
