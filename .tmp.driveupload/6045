"""
effector.intention — φ-Metabolizing Middleware
==============================================
The Intention Layer sits between the world's urgency and the human's attention.

Modules
-------
  phi_probe    Step 1: Validate the θ forward-model signal in the cultivation log
  signal_head  Step 2: Embedding-based drop-in for the LLM characterizer (Phase 2)
  phi_model    Step 3: φ injection + urgency model for incoming communications
  layer        Step 4: The Intention Layer decision engine

Quick start
-----------
  # Train all models
  python train_intention_layer.py

  # Demo
  python src/effector/intention/layer.py --demo
"""

from .layer import (
    CommunicationEvent,
    DeliveryAction,
    DeliveryDecision,
    IntentionLayer,
    Reframer,
    HabitatReader,
    DecisionEngine,
)
from .phi_model import PhiAssessor, PhiAssessment

__all__ = [
    "CommunicationEvent",
    "DeliveryAction",
    "DeliveryDecision",
    "IntentionLayer",
    "Reframer",
    "HabitatReader",
    "DecisionEngine",
    "PhiAssessor",
    "PhiAssessment",
]
