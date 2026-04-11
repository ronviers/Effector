"""
Tests for IEP-A3: Vectorized State Bus
=======================================
Covers all five milestones:

  M1 — StateBus.serialize()
  M2 — AsymmetricDASPCoordinator snapshot vector generation
  M3 — IEPBuilder snapshot_vector extraction and routing
  M4 — IEPValidator cosine similarity verification
  M5 — Critical-keys semantic drift lock

All Ollama calls are mocked.  No network required.
"""

from __future__ import annotations

import hashlib
import json
import math
import unittest
from typing import Any
from unittest.mock import MagicMock, patch


# ─────────────────────────────────────────────────────────────────────────────
# Minimal StateBus stub (avoids importing the real module in unit tests)
# ─────────────────────────────────────────────────────────────────────────────

class _MockStateBus:
    """Minimal in-memory bus stub that supports the IEP-A3 interface."""

    _DEFAULT_VOLATILE: frozenset = frozenset({"telemetry.timestamp", "telemetry.interval_s"})

    def __init__(self, initial_state: dict | None = None) -> None:
        self._state: dict[str, Any] = dict(initial_state or {})

    def read(self, keys=None):
        if keys is None:
            return dict(self._state)
        return {k: self._state.get(k) for k in keys}

    def snapshot(self, keys=None):
        from datetime import datetime, timezone
        state_slice = self.read(keys)
        canonical = json.dumps(state_slice, sort_keys=True, default=str)
        h = hashlib.sha256(canonical.encode()).hexdigest()
        return h, datetime.now(timezone.utc), state_slice

    def serialize(self, keys=None, volatile_keys=None):
        excluded = volatile_keys if volatile_keys is not None else self._DEFAULT_VOLATILE
        state = self.read(keys)
        stable = {k: v for k, v in state.items() if k not in excluded}
        parts = [f"{k}: {v}" for k, v in sorted(stable.items())]
        return " | ".join(parts)

    def apply_delta(self, envelope_id, delta, agent_id, session_id=None):
        self._state.update(delta)
        return delta

    def _set(self, key, value):
        """Test helper: mutate a key directly."""
        self._state[key] = value


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _unit_vector(dim: int, value: float = 1.0) -> list[float]:
    """Return a dim-length unit vector with all components equal."""
    mag = math.sqrt(dim * value ** 2)
    return [value / mag] * dim


def _make_debate_result(
    *,
    snapshot_vector: list[float] | None = None,
    vectorized_bus: bool = False,
    rat_similarity_threshold: float = 0.97,
    embedding_model: str = "nomic-embed-text",
    final_answer: str = "System is healthy.",
    consensus_score: float = 0.85,
) -> dict:
    return {
        "session_id": "test-session-001",
        "final_answer": final_answer,
        "consensus_score": consensus_score,
        "tier1_agents": ["mistral", "qwen"],
        "snapshot_vector": snapshot_vector,
        "vectorized_bus": vectorized_bus,
        "rat_similarity_threshold": rat_similarity_threshold,
        "embedding_model": embedding_model,
        "all_rounds": [],
    }


def _make_envelope_with_vector(
    bus: _MockStateBus,
    vector: list[float],
    threshold: float = 0.97,
    embedding_model: str = "nomic-embed-text",
) -> dict:
    """Build a minimal envelope dict with IEP-A3 vector fields."""
    snap_hash, _, _ = bus.snapshot()
    return {
        "iep_version": "1.0-A3",
        "envelope_id": "env-test-001",
        "timestamp_issued": "2099-01-01T00:00:00+00:00",  # far future — no TTL expiry
        "agent": {"id": "coordinator", "role": "executor"},
        "goal_context": {
            "root_goal_id": "rg-001", "parent_goal_id": "pg-001",
            "depth": 0, "branch_label": "test",
        },
        "world_model_snapshot": {
            "snapshot_id": "snap-001",
            "snapshot_timestamp": "2099-01-01T00:00:00+00:00",
            "relevant_keys": [],
            "hash": snap_hash,
            "snapshot_vector": vector,
            "vectorized_bus": True,
            "rat_similarity_threshold": threshold,
            "embedding_model": embedding_model,
        },
        "intended_action": {"verb": "WRITE", "target": "world_state", "parameters": {}},
        "expected_state_change": {
            "keys_affected": ["debate_answer"],
            "predicted_delta": {"debate_answer": "System is healthy."},
            "confidence": 0.85,
        },
        "abort_conditions": [],
        "ttl_ms": 999_999_999,
        "requires_ack": True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# M1 — StateBus.serialize()
# ─────────────────────────────────────────────────────────────────────────────

class TestStateBusSerialize(unittest.TestCase):

    def test_empty_state_returns_empty_string(self):
        bus = _MockStateBus({})
        self.assertEqual(bus.serialize(), "")

    def test_volatile_keys_excluded_by_default(self):
        bus = _MockStateBus({
            "cpu.percent.total": 18.3,
            "telemetry.timestamp": "2026-01-01T00:00:00Z",
            "telemetry.interval_s": 2.0,
        })
        result = bus.serialize()
        self.assertIn("cpu.percent.total", result)
        self.assertNotIn("telemetry.timestamp", result)
        self.assertNotIn("telemetry.interval_s", result)

    def test_custom_volatile_keys(self):
        bus = _MockStateBus({"a": 1, "b": 2, "c": 3})
        result = bus.serialize(volatile_keys=frozenset({"b"}))
        self.assertIn("a", result)
        self.assertIn("c", result)
        self.assertNotIn("b:", result)

    def test_output_is_sorted(self):
        bus = _MockStateBus({"z": 26, "a": 1, "m": 13})
        result = bus.serialize()
        keys_in_order = [part.split(":")[0].strip() for part in result.split(" | ")]
        self.assertEqual(keys_in_order, sorted(keys_in_order))

    def test_identical_states_produce_identical_strings(self):
        state = {"cpu.percent.total": 42.0, "ram.percent": 61.4}
        bus1 = _MockStateBus(state)
        bus2 = _MockStateBus(state)
        self.assertEqual(bus1.serialize(), bus2.serialize())

    def test_different_states_produce_different_strings(self):
        bus1 = _MockStateBus({"cpu.percent.total": 10.0})
        bus2 = _MockStateBus({"cpu.percent.total": 90.0})
        self.assertNotEqual(bus1.serialize(), bus2.serialize())

    def test_keys_filter_respected(self):
        bus = _MockStateBus({"a": 1, "b": 2, "c": 3})
        result = bus.serialize(keys=["a", "c"])
        self.assertIn("a", result)
        self.assertIn("c", result)
        self.assertNotIn("b:", result)

    def test_format_is_pipe_delimited(self):
        bus = _MockStateBus({"x": 1, "y": 2})
        result = bus.serialize()
        self.assertIn(" | ", result)

    def test_empty_volatile_set_includes_all_keys(self):
        bus = _MockStateBus({
            "telemetry.timestamp": "now",
            "cpu.percent.total": 50.0,
        })
        result = bus.serialize(volatile_keys=frozenset())
        self.assertIn("telemetry.timestamp", result)
        self.assertIn("cpu.percent.total", result)


# ─────────────────────────────────────────────────────────────────────────────
# M2 — AsymmetricDASPCoordinator snapshot vector generation
# ─────────────────────────────────────────────────────────────────────────────

class TestCoordinatorVectorization(unittest.TestCase):
    """
    These tests mock _get_snapshot_vector to avoid requiring Ollama.
    They verify that the coordinator calls the embedding helper when
    vectorized_bus=True and correctly populates the result.
    """

    def _make_coord(self, vectorized: bool = True):
        """Import and build coordinator with mocked Ollama deps."""
        # We import inline to allow the patch to work correctly
        from effector.adapters.asymmetric_dasp import (
            AsymmetricDASPCoordinator, TierConfig,
        )
        return AsymmetricDASPCoordinator(
            tier1_agents=[
                TierConfig(name="a1", model="mock-model", max_rounds=1),
            ],
            tier2_agent=TierConfig(name="t2", model="mock-t2"),
            vectorized_bus=vectorized,
            embedding_model="nomic-embed-text",
            rat_similarity_threshold=0.97,
        )

    @patch("effector.adapters.asymmetric_dasp._get_snapshot_vector")
    @patch("effector.adapters.asymmetric_dasp._call_ollama")
    def test_vector_fetched_when_vectorized_bus_true(self, mock_call, mock_vec):
        mock_vec.return_value = [0.1] * 512
        mock_call.return_value = {
            "agent_id": "a1", "session_id": "s", "round": 1,
            "snapshot_hash": "x" * 64,
            "hypothesis_id": "H1", "answer": "ok", "answer_hash": "ab12",
            "signal": {"confidence": 0.9, "polarity": 1,
                       "generative_strength": 0.9, "inhibitory_pressure": 0.0},
            "expected_state_change": {"keys_affected": [], "predicted_delta": {}, "confidence": 0.9},
        }
        bus = _MockStateBus({"cpu.percent.total": 20.0})
        snap_hash, _, _ = bus.snapshot()
        mock_call.return_value["snapshot_hash"] = snap_hash

        coord = self._make_coord(vectorized=True)
        result = coord.run("test task", snap_hash, bus)

        mock_vec.assert_called_once()
        self.assertIsNotNone(result["snapshot_vector"])
        self.assertEqual(len(result["snapshot_vector"]), 512)
        self.assertTrue(result["vectorized_bus"])

    @patch("effector.adapters.asymmetric_dasp._get_snapshot_vector")
    @patch("effector.adapters.asymmetric_dasp._call_ollama")
    def test_vector_not_fetched_when_vectorized_bus_false(self, mock_call, mock_vec):
        mock_call.return_value = None  # abstention
        bus = _MockStateBus({"cpu.percent.total": 20.0})
        snap_hash, _, _ = bus.snapshot()

        coord = self._make_coord(vectorized=False)
        result = coord.run("test task", snap_hash, bus)

        mock_vec.assert_not_called()
        self.assertIsNone(result["snapshot_vector"])
        self.assertFalse(result["vectorized_bus"])

    @patch("effector.adapters.asymmetric_dasp._get_snapshot_vector")
    @patch("effector.adapters.asymmetric_dasp._call_ollama")
    def test_none_vector_degrades_gracefully(self, mock_call, mock_vec):
        """If embedding fails, vectorized_bus is False in result; no crash."""
        mock_vec.return_value = None  # simulates Ollama unreachable
        mock_call.return_value = None
        bus = _MockStateBus({"cpu.percent.total": 20.0})
        snap_hash, _, _ = bus.snapshot()

        coord = self._make_coord(vectorized=True)
        result = coord.run("test task", snap_hash, bus)

        self.assertIsNone(result["snapshot_vector"])
        self.assertFalse(result["vectorized_bus"])

    @patch("effector.adapters.asymmetric_dasp._get_snapshot_vector")
    @patch("effector.adapters.asymmetric_dasp._call_ollama")
    def test_result_carries_threshold_and_model(self, mock_call, mock_vec):
        mock_vec.return_value = [0.5] * 768
        mock_call.return_value = None
        bus = _MockStateBus({})
        snap_hash, _, _ = bus.snapshot()

        from effector.adapters.asymmetric_dasp import AsymmetricDASPCoordinator, TierConfig
        coord = AsymmetricDASPCoordinator(
            tier1_agents=[TierConfig(name="a", model="m", max_rounds=1)],
            tier2_agent=TierConfig(name="t", model="t2"),
            vectorized_bus=True,
            embedding_model="all-minilm",
            rat_similarity_threshold=0.92,
        )
        result = coord.run("task", snap_hash, bus)

        self.assertEqual(result["rat_similarity_threshold"], 0.92)
        self.assertEqual(result["embedding_model"], "all-minilm")


# ─────────────────────────────────────────────────────────────────────────────
# M3 — IEPBuilder snapshot_vector extraction
# ─────────────────────────────────────────────────────────────────────────────

class TestIEPBuilderVectorRouting(unittest.TestCase):

    def setUp(self):
        from effector.queue.iep_queue import IEPBuilder
        self.IEPBuilder = IEPBuilder
        self.bus = _MockStateBus({"cpu.percent.total": 20.0})
        self.snap_hash, _, _ = self.bus.snapshot()

    def test_vector_present_in_envelope_when_debate_has_vector(self):
        vector = [0.1] * 512
        result = _make_debate_result(
            snapshot_vector=vector, vectorized_bus=True, rat_similarity_threshold=0.95
        )
        envelope = self.IEPBuilder.from_debate_result(result, self.snap_hash)
        snap = envelope["world_model_snapshot"]

        self.assertTrue(snap["vectorized_bus"])
        self.assertEqual(snap["snapshot_vector"], vector)
        self.assertEqual(snap["rat_similarity_threshold"], 0.95)

    def test_no_vector_when_debate_is_hash_only(self):
        result = _make_debate_result(vectorized_bus=False, snapshot_vector=None)
        envelope = self.IEPBuilder.from_debate_result(result, self.snap_hash)
        snap = envelope["world_model_snapshot"]

        self.assertFalse(snap["vectorized_bus"])
        self.assertNotIn("snapshot_vector", snap)

    def test_iep_version_updated_for_a3(self):
        vector = [0.5] * 512
        result = _make_debate_result(snapshot_vector=vector, vectorized_bus=True)
        envelope = self.IEPBuilder.from_debate_result(result, self.snap_hash)
        self.assertEqual(envelope["iep_version"], "1.0-A3")

    def test_iep_version_unchanged_for_hash_only(self):
        result = _make_debate_result(vectorized_bus=False)
        envelope = self.IEPBuilder.from_debate_result(result, self.snap_hash)
        self.assertEqual(envelope["iep_version"], "1.0")

    def test_embedding_model_propagated(self):
        vector = [0.1] * 512
        result = _make_debate_result(
            snapshot_vector=vector, vectorized_bus=True,
            embedding_model="all-minilm",
        )
        envelope = self.IEPBuilder.from_debate_result(result, self.snap_hash)
        snap = envelope["world_model_snapshot"]
        self.assertEqual(snap["embedding_model"], "all-minilm")

    def test_null_vector_does_not_set_vectorized_bus_true(self):
        """Even if debate claims vectorized_bus=True, None vector must degrade."""
        result = _make_debate_result(snapshot_vector=None, vectorized_bus=True)
        envelope = self.IEPBuilder.from_debate_result(result, self.snap_hash)
        snap = envelope["world_model_snapshot"]
        self.assertFalse(snap["vectorized_bus"])


# ─────────────────────────────────────────────────────────────────────────────
# M4 — IEPValidator cosine similarity verification
# ─────────────────────────────────────────────────────────────────────────────

class TestIEPValidatorCosine(unittest.TestCase):

    def setUp(self):
        from effector.queue.iep_queue import IEPValidator
        self.IEPValidator = IEPValidator
        self.bus = _MockStateBus({"cpu.percent.total": 20.0, "ram.percent": 60.0})

    def _make_validator(self, critical_keys=()) -> Any:
        return self.IEPValidator(
            state_bus=self.bus,
            critical_keys=critical_keys,  # disabled for pure M4 tests
        )

    def _make_vector(self, dim=512, value=1.0) -> list[float]:
        mag = math.sqrt(dim * value ** 2)
        return [value / mag] * dim

    @patch("effector.queue.iep_queue._fetch_embedding")
    def test_ack_when_similarity_above_threshold(self, mock_embed):
        vec = self._make_vector()
        mock_embed.return_value = vec  # identical → similarity = 1.0
        validator = self._make_validator()

        envelope = _make_envelope_with_vector(self.bus, vec, threshold=0.97)
        result = validator.validate(envelope)

        self.assertEqual(result.status, "ACK")
        self.assertIn("snapshot_hash", result.checks_passed)

    @patch("effector.queue.iep_queue._fetch_embedding")
    def test_nack_when_similarity_below_threshold(self, mock_embed):
        stored_vec = self._make_vector(value=1.0)
        # Orthogonal vector → similarity = 0.0
        current_vec = [0.0] * 511 + [1.0]
        mock_embed.return_value = current_vec

        validator = self._make_validator()
        envelope = _make_envelope_with_vector(self.bus, stored_vec, threshold=0.97)
        result = validator.validate(envelope)

        self.assertIn(result.status, ("NACK_SNAPSHOT_HASH",))
        self.assertIn("cosine similarity", result.failure_reason.lower())

    @patch("effector.queue.iep_queue._fetch_embedding")
    def test_fallback_to_hash_when_embedding_unavailable(self, mock_embed):
        """When embedding fetch returns None, should fall through to hash check."""
        mock_embed.return_value = None
        vec = self._make_vector()
        validator = self._make_validator()

        # Use current bus hash so hash check passes
        envelope = _make_envelope_with_vector(self.bus, vec, threshold=0.97)
        result = validator.validate(envelope)

        # Hash matches (same bus, same state) → should ACK via fallback path
        self.assertEqual(result.status, "ACK")

    @patch("effector.queue.iep_queue._fetch_embedding")
    def test_fallback_hash_fails_when_state_changed(self, mock_embed):
        """Embedding unavailable AND hash changed → NACK."""
        mock_embed.return_value = None
        vec = self._make_vector()
        validator = self._make_validator()

        # Build envelope from current state
        envelope = _make_envelope_with_vector(self.bus, vec)
        # Now mutate the bus (change state so hash differs)
        self.bus._set("cpu.percent.total", 99.9)

        result = validator.validate(envelope)
        self.assertIn("NACK", result.status)

    @patch("effector.queue.iep_queue._fetch_embedding")
    def test_threshold_boundary_at_exact_value(self, mock_embed):
        """similarity == threshold exactly should ACK."""
        vec = self._make_vector()
        # Return a slightly different vector whose similarity to vec = exactly threshold
        # We construct: v' = threshold*v + sqrt(1-threshold^2)*perp
        threshold = 0.97
        perp = [0.0] * 512
        perp[0] = 1.0  # not normalized yet

        # Make perp orthogonal to vec: since all vec components are equal,
        # project out the vec component
        dot_pv = sum(perp[i] * vec[i] for i in range(512))
        dot_vv = sum(vec[i] ** 2 for i in range(512))
        perp = [perp[i] - (dot_pv / dot_vv) * vec[i] for i in range(512)]
        mag_perp = math.sqrt(sum(x * x for x in perp))
        perp = [x / mag_perp for x in perp]

        scale_perp = math.sqrt(1 - threshold ** 2)
        current_vec = [
            threshold * vec[i] + scale_perp * perp[i]
            for i in range(512)
        ]
        mock_embed.return_value = current_vec

        validator = self._make_validator()
        envelope = _make_envelope_with_vector(self.bus, vec, threshold=threshold)
        result = validator.validate(envelope)
        self.assertEqual(result.status, "ACK")

    @patch("effector.queue.iep_queue._fetch_embedding")
    def test_cosine_check_not_run_for_non_vectorized_envelope(self, mock_embed):
        """Standard hash-only envelope must not trigger embedding call."""
        snap_hash, _, _ = self.bus.snapshot()
        envelope = {
            "iep_version": "1.0",
            "envelope_id": "e001",
            "timestamp_issued": "2099-01-01T00:00:00+00:00",
            "agent": {"id": "coord", "role": "executor"},
            "goal_context": {"root_goal_id": "r", "parent_goal_id": "p", "depth": 0, "branch_label": ""},
            "world_model_snapshot": {
                "snapshot_id": "s001",
                "snapshot_timestamp": "2099-01-01T00:00:00+00:00",
                "relevant_keys": [],
                "hash": snap_hash,
                "vectorized_bus": False,
            },
            "intended_action": {"verb": "WRITE", "target": "x", "parameters": {}},
            "expected_state_change": {
                "keys_affected": ["x"], "predicted_delta": {"x": 1}, "confidence": 1.0
            },
            "abort_conditions": [],
            "ttl_ms": 999_999_999,
            "requires_ack": True,
        }
        validator = self._make_validator()
        result = validator.validate(envelope)

        mock_embed.assert_not_called()
        self.assertEqual(result.status, "ACK")


# ─────────────────────────────────────────────────────────────────────────────
# M5 — Critical-keys semantic drift lock
# ─────────────────────────────────────────────────────────────────────────────

class TestCriticalKeysDriftLock(unittest.TestCase):

    def setUp(self):
        from effector.queue.iep_queue import IEPValidator
        self.IEPValidator = IEPValidator

    def _make_bus_with_desktop(self, window: str = "Code.exe", process: str = "Code.exe"):
        return _MockStateBus({
            "cpu.percent.total": 20.0,
            "desktop.active_window": window,
            "desktop.active_process": process,
        })

    def _make_validator(self, bus, critical_keys=("desktop.active_window", "desktop.active_process")):
        return self.IEPValidator(
            state_bus=bus,
            critical_keys=critical_keys,
        )

    @patch("effector.queue.iep_queue._fetch_embedding")
    def test_stable_critical_keys_pass_drift_lock(self, mock_embed):
        """No mutation → drift lock passes → cosine check runs."""
        bus = self._make_bus_with_desktop("Code.exe")
        vec = _unit_vector(512)
        mock_embed.return_value = vec  # identical → similarity = 1.0

        validator = self._make_validator(bus)
        envelope = _make_envelope_with_vector(bus, vec, threshold=0.97)
        result = validator.validate(envelope)

        self.assertEqual(result.status, "ACK")

    @patch("effector.queue.iep_queue._fetch_embedding")
    def test_no_critical_keys_disables_drift_lock(self, mock_embed):
        """Empty critical_keys → drift lock disabled → cosine runs normally."""
        bus = self._make_bus_with_desktop("Chrome.exe")
        vec = _unit_vector(512)
        mock_embed.return_value = vec

        validator = self._make_validator(bus, critical_keys=())
        envelope = _make_envelope_with_vector(bus, vec, threshold=0.97)
        result = validator.validate(envelope)

        self.assertEqual(result.status, "ACK")

    def test_drift_lock_custom_keys(self):
        """Custom critical_keys respected — non-desktop keys can trigger lock."""
        from effector.queue.iep_queue import IEPValidator

        bus = _MockStateBus({"system.pressure": 0.9, "cpu.percent.total": 20.0})
        validator = IEPValidator(
            state_bus=bus,
            critical_keys=("system.pressure",),
        )
        # The drift lock implementation does a double-read to detect intra-call
        # mutation.  With a stable bus, this will pass silently.
        snap_hash, _, _ = bus.snapshot()
        vec = _unit_vector(512)
        envelope = _make_envelope_with_vector(bus, vec, threshold=0.97)

        # Patch embedding to return the same vector → similarity = 1.0
        with patch("effector.queue.iep_queue._fetch_embedding", return_value=vec):
            result = validator.validate(envelope)
        self.assertEqual(result.status, "ACK")


# ─────────────────────────────────────────────────────────────────────────────
# Cosine similarity math (pure-Python implementation correctness)
# ─────────────────────────────────────────────────────────────────────────────

class TestCosineSimilarityMath(unittest.TestCase):

    def _cos(self, a, b):
        from effector.queue.iep_queue import _cosine_similarity
        return _cosine_similarity(a, b)

    def test_identical_vectors_similarity_one(self):
        v = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(self._cos(v, v), 1.0, places=9)

    def test_orthogonal_vectors_similarity_zero(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        self.assertAlmostEqual(self._cos(a, b), 0.0, places=9)

    def test_opposite_vectors_similarity_minus_one(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        self.assertAlmostEqual(self._cos(a, b), -1.0, places=9)

    def test_zero_vector_returns_zero(self):
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        self.assertEqual(self._cos(a, b), 0.0)

    def test_high_dimensional_unit_vectors(self):
        dim = 512
        a = _unit_vector(dim)
        b = _unit_vector(dim)
        self.assertAlmostEqual(self._cos(a, b), 1.0, places=6)

    def test_symmetry(self):
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        self.assertAlmostEqual(self._cos(a, b), self._cos(b, a), places=9)

    def test_scale_invariance(self):
        a = [1.0, 0.0]
        b = [1.0, 0.0]
        b_scaled = [100.0, 0.0]
        self.assertAlmostEqual(self._cos(a, b), self._cos(a, b_scaled), places=9)


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end: M1 → M2 → M3 → M4 (all mocked, no Ollama)
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEndIEPA3(unittest.TestCase):

    @patch("effector.queue.iep_queue._fetch_embedding")
    @patch("effector.adapters.asymmetric_dasp._get_snapshot_vector")
    @patch("effector.adapters.asymmetric_dasp._call_ollama")
    def test_full_pipeline_acks_stable_state(self, mock_ollama, mock_coord_vec, mock_val_embed):
        """
        Simulates the full A1→A2→A3 pipeline for the vectorized path:
          1. Bus has stable telemetry state.
          2. Coordinator generates snapshot vector (mocked).
          3. IEPBuilder builds envelope with vector.
          4. IEPValidator computes cosine similarity (mocked identical → 1.0).
          5. Expects ACK.
        """
        from effector.adapters.asymmetric_dasp import (
            AsymmetricDASPCoordinator, TierConfig,
        )
        from effector.queue.iep_queue import IEPBuilder, IEPValidator

        # ── A1: Stable telemetry state ────────────────────────────────────
        bus = _MockStateBus({
            "cpu.percent.total": 22.0,
            "ram.percent": 55.0,
            "desktop.active_window": "Code.exe",
            "desktop.active_process": "Code.exe",
        })
        snap_hash, _, _ = bus.snapshot()

        # ── A2: Coordinator produces debate result with vector ─────────────
        stored_vec = _unit_vector(512)
        mock_coord_vec.return_value = stored_vec
        mock_ollama.return_value = {
            "agent_id": "mistral",
            "session_id": "s1", "round": 1,
            "snapshot_hash": snap_hash,
            "hypothesis_id": "H1",
            "answer": "System looks healthy.",
            "answer_hash": "abc123",
            "signal": {
                "confidence": 0.88, "polarity": 1,
                "generative_strength": 0.88, "inhibitory_pressure": 0.0,
            },
            "expected_state_change": {
                "keys_affected": [], "predicted_delta": {}, "confidence": 0.88,
            },
        }

        coord = AsymmetricDASPCoordinator(
            tier1_agents=[TierConfig(name="mistral", model="mistral", max_rounds=1)],
            tier2_agent=TierConfig(name="nemotron", model="nemotron"),
            vectorized_bus=True,
            rat_similarity_threshold=0.97,
        )
        debate_result = coord.run("Analyse system health", snap_hash, bus)

        self.assertTrue(debate_result["vectorized_bus"])
        self.assertIsNotNone(debate_result["snapshot_vector"])

        # ── A3a: Builder maps vector into envelope ────────────────────────
        envelope = IEPBuilder.from_debate_result(
            debate_result=debate_result,
            state_bus_snapshot_hash=snap_hash,
            keys_affected=["debate_answer"],
        )
        self.assertTrue(envelope["world_model_snapshot"]["vectorized_bus"])

        # ── A3b: Validator checks cosine (same vector → similarity = 1.0) ─
        mock_val_embed.return_value = stored_vec  # current state embeds identically

        validator = IEPValidator(
            state_bus=bus,
            critical_keys=("desktop.active_window", "desktop.active_process"),
        )
        verdict = validator.validate(envelope)

        self.assertEqual(verdict.status, "ACK")
        self.assertIn("snapshot_hash", verdict.checks_passed)

    @patch("effector.queue.iep_queue._fetch_embedding")
    @patch("effector.adapters.asymmetric_dasp._get_snapshot_vector")
    @patch("effector.adapters.asymmetric_dasp._call_ollama")
    def test_full_pipeline_nacks_on_drifted_state(
        self, mock_ollama, mock_coord_vec, mock_val_embed
    ):
        """
        Same pipeline, but the embedding at validation time is very different
        from the stored vector (simulating significant state drift).
        Expects NACK_SNAPSHOT_HASH.
        """
        from effector.adapters.asymmetric_dasp import (
            AsymmetricDASPCoordinator, TierConfig,
        )
        from effector.queue.iep_queue import IEPBuilder, IEPValidator

        bus = _MockStateBus({"cpu.percent.total": 22.0, "ram.percent": 55.0})
        snap_hash, _, _ = bus.snapshot()

        stored_vec = _unit_vector(512, value=1.0)
        mock_coord_vec.return_value = stored_vec
        mock_ollama.return_value = None  # abstention

        coord = AsymmetricDASPCoordinator(
            tier1_agents=[TierConfig(name="m", model="m", max_rounds=1)],
            tier2_agent=TierConfig(name="t", model="t"),
            vectorized_bus=True,
        )
        debate_result = coord.run("task", snap_hash, bus)

        envelope = IEPBuilder.from_debate_result(
            debate_result, snap_hash, keys_affected=["debate_answer"]
        )

        # Orthogonal vector at validation time → similarity = 0.0
        drifted_vec = [0.0] * 511 + [1.0]
        mock_val_embed.return_value = drifted_vec

        validator = IEPValidator(state_bus=bus, critical_keys=())
        verdict = validator.validate(envelope)

        self.assertEqual(verdict.status, "NACK_SNAPSHOT_HASH")
        self.assertIn("cosine", verdict.failure_reason.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
