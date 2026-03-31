"""
Unit Tests for DASP Signal Superposition Math
Phase 1: Deterministic Substrate Validation
"""

import sys
import os
import unittest
import uuid

# --- PATH FIX ---
# This forces Python to look in the 'src' directory next to 'tests'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Depending on your exact folder structure, dasp might be under schemas/
try:
    from effector.schemas.dasp import AgentResponse, AgentSignal, ExpectedStateChange, HypothesisFinalState
except ModuleNotFoundError:
    from effector.dasp import AgentResponse, AgentSignal, ExpectedStateChange, HypothesisFinalState

from effector.signal_engine import SignalEngine


def mock_response(
    agent_id: str, 
    round_num: int, 
    hyp_id: str, 
    polarity: int, 
    conf: float, 
    g_str: float = 0.0, 
    i_prs: float = 0.0
) -> AgentResponse:
    """Helper to quickly mint Pydantic-compliant AgentResponse objects."""
    
    signal = AgentSignal(
        confidence=conf,
        polarity=polarity,
        generative_strength=g_str,
        inhibitory_pressure=i_prs
    )
    
    return AgentResponse(
        session_id=uuid.uuid4(),
        agent_id=agent_id,
        round=round_num,
        snapshot_hash="0" * 64,  
        hypothesis_id=hyp_id,
        answer="Mock telemetry answer",
        answer_hash="mockhash12345678",
        signal=signal,
        expected_state_change=ExpectedStateChange()
    )


class TestSignalEngineGates(unittest.TestCase):
    def setUp(self):
        # Strict thresholds for our math tests
        self.engine = SignalEngine(
            tau_suppression=0.5,   # S_i >= 50% of S_g triggers inhibition
            theta_consensus=0.7,   # S_net >= 0.7 triggers consensus
            epsilon_stall=0.05,    # Delta S_net < 0.05 triggers stall
            use_reputation=False   # Ignore reputation weighting for baseline math
        )

    def test_inhibition_gate_fires(self):
        """P1: A strong veto should instantly cancel the hypothesis."""
        # mistral S_g = 0.8 * 0.8 = 0.64
        # qwen S_i = 0.6 * 0.6 = 0.36
        # Ratio = 0.36 / 0.64 = 0.5625 (>= 0.5 tau)
        responses = [
            mock_response("mistral", 1, "H1", polarity=1, conf=0.8),
            mock_response("qwen", 1, "H1", polarity=-1, conf=0.6) # Boosted from 0.5 to 0.6
        ]
        
        self.engine.ingest_responses(responses)
        result = self.engine.evaluate_gates()
        
        self.assertTrue(result.inhibition_fired, "Inhibition gate failed to fire.")
        self.assertFalse(result.consensus_cleared)
        self.assertEqual(
            self.engine._manifolds["H1"].final_state, 
            HypothesisFinalState.phase_canceled
        )

    def test_stall_gate_fires(self):
        """P2: Flatlining confidence between rounds should trigger a stall."""
        # Round 1: S_net reaches 0.4
        r1 = [mock_response("mistral", 1, "H1", polarity=1, conf=0.4)]
        self.engine.ingest_responses(r1)
        res1 = self.engine.evaluate_gates()
        self.assertFalse(res1.stall_fired, "Should not stall on round 1.")
        
        # Round 2: Agent gives the exact same signal. Delta = 0.0 (< 0.05 epsilon)
        r2 = [mock_response("mistral", 2, "H1", polarity=1, conf=0.4)]
        self.engine.ingest_responses(r2)
        res2 = self.engine.evaluate_gates()
        
        self.assertTrue(res2.stall_fired, "Stall gate failed to fire on flatline.")
        self.assertEqual(
            self.engine._manifolds["H1"].final_state, 
            HypothesisFinalState.stalled
        )

    def test_consensus_gate_clears(self):
        """P3: Overwhelming support should clear the consensus threshold."""
        # S_g = 0.9 * 0.9 = 0.81 (>= 0.7 theta). S_i = 0.0. 
        responses = [mock_response("mistral", 1, "H1", polarity=1, conf=0.9)] # Boosted from 0.8 to 0.9
        
        self.engine.ingest_responses(responses)
        result = self.engine.evaluate_gates()
        
        self.assertTrue(result.consensus_cleared, "Consensus gate failed to clear.")
        self.assertEqual(result.winning_hypothesis, "H1")
        self.assertEqual(result.consensus_score, 0.81) # Updated expected score
        self.assertEqual(
            self.engine._manifolds["H1"].final_state, 
            HypothesisFinalState.consensus
        )

    def test_swap_detection(self):
        """DASP Anti-pattern: Agents swapping polarities without net progress."""
        prev = [
            mock_response("mistral", 1, "H1", polarity=1, conf=0.5),
            mock_response("qwen", 1, "H1", polarity=-1, conf=0.5)
        ]
        curr = [
            mock_response("mistral", 2, "H1", polarity=-1, conf=0.5),
            mock_response("qwen", 2, "H1", polarity=1, conf=0.5)
        ]
        
        is_swap = self.engine.swap_detected(prev, curr)
        self.assertTrue(is_swap, "Engine failed to detect agent polarity swap.")

if __name__ == '__main__':
    unittest.main(verbosity=2)