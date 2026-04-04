import pytest
from pathlib import Path
from effector.intention.layer import (
    DecisionEngine, DeliveryAction, IntentionLayer, CommunicationEvent
)
from effector.intention.phi_model import PhiAssessment, PhiAssessor

class MockPhiAssessor:
    def __init__(self, phi_injection: float, urgency: float, reframeable: bool):
        self.phi_injection = phi_injection
        self.urgency = urgency
        self.reframeable = reframeable

    def assess(self, text: str, timeout_s: float = 10.0):
        return PhiAssessment(
            text=text,
            phi_injection=self.phi_injection,
            urgency=self.urgency,
            reframeable=self.reframeable,
            elapsed_ms=10.0
        )

class MockStateBus:
    def __init__(self, pressure: float, theta: float):
        self.pressure = pressure
        self.theta = theta
        
    def read(self, keys=None):
        return {"system.pressure": self.pressure, "habitat.theta": self.theta}

class MockReframer:
    def reframe(self, subject: str, preview: str = ""):
        return f"Reframed: {subject}", "Mock rationale"

@pytest.mark.parametrize("phi, urgency, reframe, pressure, expected_action", [
    # 1. Emergency Bypass
    (0.9, 0.9, False, 0.8, DeliveryAction.PRESENT_NOW),
    # 2. Ambient (No disruption)
    (0.2, 0.5, True, 0.8, DeliveryAction.PRESENT_NOW),
    # 3. Gentle & Cool (Reframeable) -> Soften
    (0.4, 0.5, True, 0.2, DeliveryAction.SOFTEN),
    # 4. Gentle & Cool (Not reframeable) -> Present Now
    (0.4, 0.5, False, 0.2, DeliveryAction.PRESENT_NOW),
    # 5. Gentle & Warm -> Hold
    (0.4, 0.5, True, 0.5, DeliveryAction.HOLD),
    # 6. Gentle & Hot (Reframeable) -> Reframe + Hold
    (0.4, 0.5, True, 0.8, DeliveryAction.REFRAME),
    # 7. Moderate & Cool (Reframeable) -> Soften
    (0.6, 0.5, True, 0.2, DeliveryAction.SOFTEN),
    # 8. Sharp & Hot (Reframeable) -> Reframe + Hold
    (0.9, 0.5, True, 0.8, DeliveryAction.REFRAME),
    # 9. Sharp & Hot (Not reframeable) -> Hold
    (0.9, 0.5, False, 0.8, DeliveryAction.HOLD),
])
def test_decision_engine_combinations(phi, urgency, reframe, pressure, expected_action):
    engine = DecisionEngine()
    action, delay, _ = engine.decide(phi, urgency, reframe, pressure, 0.5, "test-id")
    assert action == expected_action
    if action in (DeliveryAction.PRESENT_NOW, DeliveryAction.SOFTEN):
        assert delay == 0.0
    else:
        assert delay > 0.0

def test_phi_assessor_clamping(mocker):
    # Tests that the output bounds are strictly enforced (e.g. out of range predictions)
    mock_post = mocker.patch("requests.post")
    mock_post.return_value.json.return_value = {"embeddings": [[0.5] * 768]}
    
    mock_scaler = mocker.MagicMock()
    mock_scaler.transform.return_value = [[0.5] * 768]
    
    mock_ridge_phi = mocker.MagicMock()
    mock_ridge_phi.predict.return_value = [1.5]  # Overshoot > 1.0
    mock_ridge_urg = mocker.MagicMock()
    mock_ridge_urg.predict.return_value = [-0.5] # Undershoot < 0.0
    mock_clf = mocker.MagicMock()
    mock_clf.predict.return_value = [1]
    
    artefact = {
        "embedding_model": "test-model", "dim": 768, "scaler": mock_scaler,
        "ridge_phi": mock_ridge_phi, "ridge_urgency": mock_ridge_urg, "clf_reframe": mock_clf,
    }
    assessor = PhiAssessor(artefact)
    result = assessor.assess("test text")
    
    assert result.phi_injection == 1.0  # Clamped
    assert result.urgency == 0.0        # Clamped
    assert result.reframeable is True

def test_intention_layer_routing():
    bus = MockStateBus(pressure=0.8, theta=0.5)
    assessor = MockPhiAssessor(phi_injection=0.9, urgency=0.5, reframeable=True)
    reframer = MockReframer()
    
    layer = IntentionLayer(state_bus=bus, assessor=assessor, reframer=reframer)
    event = CommunicationEvent(source="email", subject="Test")
    
    decision = layer.process(event)
    assert decision.action == DeliveryAction.REFRAME
    assert decision.reframed_subject == "Reframed: Test"
    assert decision.delay_minutes > 0.0

def test_resolve_signal_head_path(monkeypatch):
    from effector.adapters.two_phase_adapter import _resolve_signal_head_path
    
    # Explicit path mismatch
    assert _resolve_signal_head_path("fake/path/signal_head.pkl") is None
    
    # Env var override mismatch
    monkeypatch.setenv("EFFECTOR_SIGNAL_HEAD", "fake/env/path.pkl")
    assert _resolve_signal_head_path() is None
    
    # None fallback (auto-discover)
    monkeypatch.delenv("EFFECTOR_SIGNAL_HEAD", raising=False)
    res = _resolve_signal_head_path()
    assert res is None or isinstance(res, Path)