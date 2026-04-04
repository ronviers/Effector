"""
layer.py — Step 4: The Intention Layer
=======================================
φ-metabolizing middleware that sits between incoming communications and
the human's attention.

The layer does not filter. It metabolizes.

"Filter" implies a binary gate — allowed through or blocked. The Intention
Layer is continuous and context-sensitive: the same email that would be
surfaced immediately in a cold habitat (low pressure, low thermal state)
might be held for 12 minutes in a hot one (high pressure, active focus
state). The email hasn't changed. The habitat has.

Architecture
------------
                    ┌────────────────────────────────────┐
  Incoming comms    │         INTENTION LAYER            │
  ─────────────────►│                                    │
  (email, Slack,    │  PhiAssessor                       │
   system alert,    │    └─ embed(text) → φ + urgency    │
   calendar, etc.)  │                                    │
                    │  HabitatReader                     │
                    │    └─ StateBus → pressure + θ      │
                    │                                    │   DeliveryDecision
                    │  DecisionEngine                    ├──────────────────►
                    │    └─ decision function            │   PRESENT_NOW
                    │         ↓ IEP envelope if WRITE    │   SOFTEN (delay=0)
                    │                                    │   HOLD   (delay=N min)
                    └────────────────────────────────────┘   REFRAME (delay+text)

Decision function
-----------------
The function has four possible outputs, not two:

  PRESENT_NOW
    Surface immediately, without modification.
    Conditions: urgency > 0.85 (emergency bypass) OR φ < 0.25 (no disruption)

  SOFTEN
    Surface now, but with the subject/preview reframed to reduce φ payload.
    Conditions: φ in [0.25, 0.55] AND habitat.pressure < 0.4
             OR φ in [0.25, 0.75] AND habitat.pressure < 0.25

  HOLD
    Defer delivery for delay_minutes. Surface unmodified when delay elapses.
    Conditions: φ > 0.25 AND urgency < 0.85 AND habitat.pressure in [0.4, 0.7]

  REFRAME + HOLD
    Hold AND rewrite the subject/preview to reduce the φ payload on delivery.
    Conditions: φ > 0.55 AND urgency < 0.85 AND habitat.pressure > 0.5

Delay calculation
-----------------
  base_delay = φ_injection × (habitat.pressure - 0.3) × 20 minutes
  max delay = 45 minutes (never holds longer — urgency decay)
  urgency discount = delay × (1 - urgency)  (urgent items held less)

Reframing
---------
When reframeable=True and the decision is SOFTEN or REFRAME, the layer
submits a one-shot LLM call to rewrite the preview text. The reframe
prompt asks for a single sentence that:
  - Preserves the core information
  - Reduces the urgency signalling
  - Matches the Effector's calm, warm tone

The Glimmer that delivers a REFRAME'd notification carries a small envelope
and arrives with dangling legs.

IEP integration
---------------
Every delivery decision is wrapped in an Intention Envelope before execution.
PRESENT_NOW and SOFTEN decisions have TTL=30s (immediate).
HOLD and REFRAME decisions use the delay as TTL and are queued on the State Bus.

Usage
-----
  from effector.intention.layer import IntentionLayer, CommunicationEvent
  from effector.intention.phi_model import PhiAssessor
  from effector.bus import StateBus

  assessor = PhiAssessor.load(Path("data/phi_model.pkl"))
  layer = IntentionLayer(state_bus=bus, assessor=assessor)

  event = CommunicationEvent(
      source="email",
      sender="client@acme.com",
      subject="URGENT: need the report by noon",
      preview="Hi — any chance you can get me the Q3...",
  )
  decision = layer.process(event)
  print(decision)   # DeliveryDecision(action=HOLD, delay_minutes=12, ...)
"""

from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

# ── Communication event ───────────────────────────────────────────────────────

@dataclass
class CommunicationEvent:
    """
    An incoming communication that the Intention Layer must route.

    source:  "email" | "slack" | "system" | "calendar" | "social" | "other"
    subject: The notification subject line or message preview (≤ 200 chars).
    preview: Optional longer context (≤ 500 chars). Used for reframing.
    sender:  Optional — used for urgency context (not yet modeled).
    """
    source: str
    subject: str
    preview: str = ""
    sender: str  = ""
    received_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def assessment_text(self) -> str:
        """The text string that gets embedded for φ assessment."""
        if self.preview:
            return f"{self.subject} | {self.preview[:300]}"
        return self.subject

# ── Delivery decision ─────────────────────────────────────────────────────────

class DeliveryAction(str, Enum):
    PRESENT_NOW = "PRESENT_NOW"  # surface immediately, unmodified
    SOFTEN      = "SOFTEN"       # surface now, with reframed preview
    HOLD        = "HOLD"         # defer for delay_minutes, unmodified
    REFRAME     = "REFRAME"      # defer + reframe preview on delivery

@dataclass
class DeliveryDecision:
    """
    The Intention Layer's output for one CommunicationEvent.
    """
    event_id: str
    action: DeliveryAction
    delay_minutes: float          # 0.0 for PRESENT_NOW and SOFTEN
    phi_injection: float
    urgency: float
    reframeable: bool

    # Habitat state at decision time
    habitat_pressure: float
    habitat_theta: float

    # Reframed text (populated only if action ∈ {SOFTEN, REFRAME})
    reframed_subject: str | None = None
    reframe_rationale: str | None = None

    # IEP envelope (populated when wired into StateBus)
    envelope: dict[str, Any] | None = None

    # Reasoning trace
    decision_rationale: str = ""

    decision_time_ms: float = 0.0
    decided_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __str__(self) -> str:
        lines = [
            f"┌─ DeliveryDecision {'─'*40}",
            f"│  Action   : {self.action.value}",
        ]
        if self.delay_minutes > 0:
            lines.append(f"│  Delay    : {self.delay_minutes:.0f} minutes")
        lines += [
            f"│  φ        : {self.phi_injection:.3f}  urgency={self.urgency:.3f}",
            f"│  Habitat  : pressure={self.habitat_pressure:.3f}  θ={self.habitat_theta:.3f}",
        ]
        if self.reframed_subject:
            lines.append(f"│  Reframed : {self.reframed_subject[:60]}")
        lines.append(f"│  Reason   : {self.decision_rationale}")
        lines.append(f"└{'─'*50}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id":          self.event_id,
            "action":            self.action.value,
            "delay_minutes":     self.delay_minutes,
            "phi_injection":     self.phi_injection,
            "urgency":           self.urgency,
            "reframeable":       self.reframeable,
            "habitat_pressure":  self.habitat_pressure,
            "habitat_theta":     self.habitat_theta,
            "reframed_subject":  self.reframed_subject,
            "reframe_rationale": self.reframe_rationale,
            "decision_rationale":self.decision_rationale,
            "decision_time_ms":  self.decision_time_ms,
            "decided_at":        self.decided_at,
        }

# ── Habitat reader ────────────────────────────────────────────────────────────

class HabitatReader:
    """
    Reads thermal and pressure state from the StateBus.

    Returns (pressure, theta) where:
      pressure ∈ [0, 1]  — system.pressure from TelemetryPoller
      theta    ∈ [0, 1]  — thermal estimate from registry context
                           (defaults to 0.5 if not set by cultivation loop)
    """

    # Key for storing the cultivation loop's thermal estimate in the bus
    HABITAT_THETA_KEY    = "habitat.theta"
    HABITAT_PRESSURE_KEY = "system.pressure"

    def __init__(self, state_bus: Any) -> None:
        self._bus = state_bus

    def read(self) -> tuple[float, float]:
        state = self._bus.read()
        pressure = float(state.get(self.HABITAT_PRESSURE_KEY, 0.3))
        theta    = float(state.get(self.HABITAT_THETA_KEY, 0.5))
        return (
            max(0.0, min(1.0, pressure)),
            max(0.0, min(1.0, theta)),
        )

# ── Decision engine ───────────────────────────────────────────────────────────

class DecisionEngine:
    """
    Pure function: (phi, urgency, reframeable, pressure, theta) → DeliveryDecision.

    No LLM calls here. No IO. The reframe text is generated by IntentionLayer,
    not by this class.
    """

    # φ thresholds
    PHI_AMBIENT   = 0.25   # below → PRESENT_NOW always
    PHI_GENTLE    = 0.55   # below → SOFTEN or short HOLD
    PHI_MODERATE  = 0.75   # below → HOLD or REFRAME depending on pressure

    # Habitat thresholds
    PRESSURE_COOL   = 0.30  # habitat is spacious
    PRESSURE_WARM   = 0.55  # habitat is engaged
    PRESSURE_HOT    = 0.75  # habitat is under load

    # Urgency bypass
    URGENCY_BYPASS  = 0.85  # above → PRESENT_NOW regardless of φ

    def decide(
        self,
        phi: float,
        urgency: float,
        reframeable: bool,
        pressure: float,
        theta: float,
        event_id: str,
    ) -> tuple[DeliveryAction, float, str]:
        """
        Returns (action, delay_minutes, rationale).

        delay_minutes is 0 for PRESENT_NOW and SOFTEN.
        """
        # ── Emergency bypass ─────────────────────────────────────────────────
        if urgency >= self.URGENCY_BYPASS:
            return (
                DeliveryAction.PRESENT_NOW,
                0.0,
                f"urgency={urgency:.2f} ≥ {self.URGENCY_BYPASS} — emergency bypass",
            )

        # ── Ambient (no disruption) ───────────────────────────────────────────
        if phi <= self.PHI_AMBIENT:
            return (
                DeliveryAction.PRESENT_NOW,
                0.0,
                f"φ={phi:.2f} ≤ {self.PHI_AMBIENT} — ambient, no disruption",
            )

        # ── Delay calculation (used for HOLD and REFRAME) ─────────────────────
        # base_delay grows with φ and pressure; urgency discounts it
        pressure_above_floor = max(0.0, pressure - self.PRESSURE_COOL)
        base_delay = phi * pressure_above_floor * 30.0      # up to ~18 min
        urgency_discount = base_delay * urgency             # urgent items held less
        raw_delay = base_delay - urgency_discount
        delay = max(1.0, min(45.0, round(raw_delay, 1)))

        # ── Gentle φ range (0.25–0.55) ────────────────────────────────────────
        if phi <= self.PHI_GENTLE:
            if pressure <= self.PRESSURE_COOL:
                # Cool habitat can absorb gentle disruption
                if reframeable:
                    return (
                        DeliveryAction.SOFTEN,
                        0.0,
                        f"φ={phi:.2f} gentle, pressure={pressure:.2f} cool — soften and present",
                    )
                return (
                    DeliveryAction.PRESENT_NOW,
                    0.0,
                    f"φ={phi:.2f} gentle, pressure={pressure:.2f} cool, not reframeable — present",
                )
            elif pressure <= self.PRESSURE_WARM:
                # Warm habitat: short hold
                return (
                    DeliveryAction.HOLD,
                    max(3.0, delay),
                    f"φ={phi:.2f} gentle, pressure={pressure:.2f} warm — hold {delay:.0f}min",
                )
            else:
                # Hot habitat: SOFTEN if possible, else HOLD
                if reframeable:
                    return (
                        DeliveryAction.REFRAME,
                        delay,
                        f"φ={phi:.2f} gentle, pressure={pressure:.2f} hot — reframe+hold {delay:.0f}min",
                    )
                return (
                    DeliveryAction.HOLD,
                    delay,
                    f"φ={phi:.2f} gentle, pressure={pressure:.2f} hot — hold {delay:.0f}min",
                )

        # ── Moderate φ range (0.55–0.75) ─────────────────────────────────────
        if phi <= self.PHI_MODERATE:
            if pressure <= self.PRESSURE_COOL:
                if reframeable:
                    return (
                        DeliveryAction.SOFTEN,
                        0.0,
                        f"φ={phi:.2f} moderate, pressure={pressure:.2f} cool — soften and present",
                    )
                return (
                    DeliveryAction.PRESENT_NOW,
                    0.0,
                    f"φ={phi:.2f} moderate, pressure={pressure:.2f} cool — present (not reframeable)",
                )
            else:
                # Always REFRAME+HOLD if possible, else HOLD
                if reframeable:
                    return (
                        DeliveryAction.REFRAME,
                        delay,
                        f"φ={phi:.2f} moderate, pressure={pressure:.2f} — reframe+hold {delay:.0f}min",
                    )
                return (
                    DeliveryAction.HOLD,
                    delay,
                    f"φ={phi:.2f} moderate, pressure={pressure:.2f} — hold {delay:.0f}min",
                )

        # ── Sharp φ range (> 0.75) ────────────────────────────────────────────
        if pressure <= self.PRESSURE_COOL:
            # Warm habitat can absorb even sharp signals if reframeable
            if reframeable:
                return (
                    DeliveryAction.SOFTEN,
                    0.0,
                    f"φ={phi:.2f} sharp, but pressure={pressure:.2f} cool — soften",
                )
            return (
                DeliveryAction.PRESENT_NOW,
                0.0,
                f"φ={phi:.2f} sharp, pressure={pressure:.2f} cool, not reframeable — present",
            )

        # Hot habitat + sharp φ → always REFRAME+HOLD
        if reframeable:
            return (
                DeliveryAction.REFRAME,
                delay,
                f"φ={phi:.2f} sharp, pressure={pressure:.2f} hot — reframe+hold {delay:.0f}min",
            )
        return (
            DeliveryAction.HOLD,
            delay,
            f"φ={phi:.2f} sharp, pressure={pressure:.2f} hot — hold {delay:.0f}min",
        )

# ── Reframer ──────────────────────────────────────────────────────────────────

class Reframer:
    """
    One-shot LLM call to soften a notification subject/preview.

    Generates a single sentence that:
      - Preserves the core information
      - Reduces urgency signalling
      - Matches the Effector's calm, warm, slightly archaic tone

    Falls back to a simple prefix if Ollama is unavailable.
    """

    _SYSTEM = """\
You are the Effector's message reframing agent. Your task is to rewrite
an incoming notification into calmer language that preserves the core
information but reduces the φ payload (kinetic urgency) it carries.

Rules:
  - One sentence maximum.
  - Preserve the key fact (who, what, when) — do not omit critical information.
  - Remove exclamation marks, ALL CAPS, and urgency language.
  - Use gentle, informative phrasing. No alarm. No pressure.
  - Do not add "Note:" or any prefix. Output only the reframed sentence.

Examples:
  Input:  "URGENT: Production database unreachable — all services down"
  Output: "The production database appears to be unreachable at the moment."

  Input:  "Boss: We need to talk. Are you available now?"
  Output: "Your manager would like to connect with you when convenient."

  Input:  "PagerDuty: CRITICAL — checkout service is down"
  Output: "The checkout service is currently showing an issue."
"""

    def __init__(
        self,
        model: str = "mistral:7b",
        host: str = "http://127.0.0.1:11434",
        timeout_s: float = 15.0,
    ) -> None:
        self._model = model
        self._host = host
        self._timeout_s = timeout_s

    def reframe(self, subject: str, preview: str = "") -> tuple[str, str]:
        """
        Return (reframed_subject, rationale).
        Falls back to a generic softening if LLM is unavailable.
        """
        context = subject
        if preview:
            context = f"{subject}\n\nContext: {preview[:200]}"

        try:
            import requests
            payload = {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": self._SYSTEM},
                    {"role": "user", "content": f"Reframe: {context}"},
                ],
                "stream": False,
                "options": {"temperature": 0.3},
            }
            resp = requests.post(
                f"{self._host}/api/chat", json=payload, timeout=self._timeout_s
            )
            resp.raise_for_status()
            text = resp.json().get("message", {}).get("content", "").strip()
            if text:
                return text, f"LLM reframe via {self._model}"
        except Exception as exc:
            pass

        # Fallback: simple prefix
        cleaned = subject.lstrip("URGENT: ").lstrip("CRITICAL: ").lstrip("ALERT: ")
        cleaned = cleaned.strip(". !").rstrip(".")
        return f"A message arrived: {cleaned}.", "fallback reframe (Ollama unavailable)"

# ── Intention Layer ───────────────────────────────────────────────────────────

class IntentionLayer:
    """
    φ-metabolizing middleware.

    Instantiate once per application. Processes communication events
    on the hot path; reframing (LLM call) is only triggered for
    SOFTEN and REFRAME decisions.

    Parameters
    ----------
    state_bus:
        Live StateBus instance providing habitat telemetry.
    assessor:
        PhiAssessor loaded from data/phi_model.pkl.
    reframer:
        Reframer instance. Defaults to mistral:7b.
    phi_model_fallback:
        If True and assessor returns None, treat the event as φ=0.7, urgency=0.5
        (conservative — hold unless habitat is cold). If False, surface immediately.
    """

    def __init__(
        self,
        state_bus: Any,
        assessor: Any | None = None,
        reframer: Reframer | None = None,
        phi_model_fallback: bool = True,
    ) -> None:
        self._bus             = state_bus
        self._assessor        = assessor
        self._reframer        = reframer or Reframer()
        self._fallback_phi    = 0.7 if phi_model_fallback else 0.0
        self._habitat         = HabitatReader(state_bus)
        self._engine          = DecisionEngine()

    def process(self, event: CommunicationEvent) -> DeliveryDecision:
        """
        Route a communication event through the φ-metabolizing pipeline.

        Typical hot path latency:
          - Embedding call:       ~30ms
          - Regression inference: <1ms
          - Habitat read:         <1ms
          - Decision function:    <1ms
          - Total (no reframe):   ~35ms

        When action is SOFTEN or REFRAME, an additional LLM call fires
        asynchronously (or synchronously if the caller needs the reframed text
        before proceeding).
        """
        t0 = time.monotonic()

        # ── Assess φ ─────────────────────────────────────────────────────────
        assessment = None
        if self._assessor is not None:
            assessment = self._assessor.assess(event.assessment_text)

        if assessment is not None:
            phi         = assessment.phi_injection
            urgency     = assessment.urgency
            reframeable = assessment.reframeable
        else:
            # Model unavailable — use conservative defaults
            phi         = self._fallback_phi
            urgency     = 0.5
            reframeable = True

        # ── Read habitat ──────────────────────────────────────────────────────
        pressure, theta = self._habitat.read()

        # ── Decide ───────────────────────────────────────────────────────────
        action, delay, rationale = self._engine.decide(
            phi=phi,
            urgency=urgency,
            reframeable=reframeable,
            pressure=pressure,
            theta=theta,
            event_id=event.event_id,
        )

        elapsed = (time.monotonic() - t0) * 1000

        decision = DeliveryDecision(
            event_id=event.event_id,
            action=action,
            delay_minutes=delay,
            phi_injection=phi,
            urgency=urgency,
            reframeable=reframeable,
            habitat_pressure=pressure,
            habitat_theta=theta,
            decision_rationale=rationale,
            decision_time_ms=round(elapsed, 1),
        )

        # ── Reframe if needed ─────────────────────────────────────────────────
        if action in (DeliveryAction.SOFTEN, DeliveryAction.REFRAME) and reframeable:
            reframed, rationale_text = self._reframer.reframe(
                event.subject, event.preview
            )
            decision.reframed_subject  = reframed
            decision.reframe_rationale = rationale_text

        return decision

    def process_batch(
        self, events: list[CommunicationEvent]
    ) -> list[DeliveryDecision]:
        """Process multiple events; habitat state is read once, shared."""
        pressure, theta = self._habitat.read()
        results = []
        for event in events:
            # Override habitat read with pre-fetched values
            t0 = time.monotonic()
            assessment = None
            if self._assessor is not None:
                assessment = self._assessor.assess(event.assessment_text)
            phi         = assessment.phi_injection if assessment else self._fallback_phi
            urgency     = assessment.urgency       if assessment else 0.5
            reframeable = assessment.reframeable   if assessment else True
            action, delay, rationale = self._engine.decide(
                phi, urgency, reframeable, pressure, theta, event.event_id
            )
            elapsed = (time.monotonic() - t0) * 1000
            d = DeliveryDecision(
                event_id=event.event_id, action=action, delay_minutes=delay,
                phi_injection=phi, urgency=urgency, reframeable=reframeable,
                habitat_pressure=pressure, habitat_theta=theta,
                decision_rationale=rationale, decision_time_ms=round(elapsed, 1),
            )
            if action in (DeliveryAction.SOFTEN, DeliveryAction.REFRAME) and reframeable:
                r, rt = self._reframer.reframe(event.subject, event.preview)
                d.reframed_subject = r
                d.reframe_rationale = rt
            results.append(d)
        return results

    @classmethod
    def from_paths(
        cls,
        state_bus: Any,
        phi_model_path: Path | None = None,
        reframe_model: str = "mistral:7b",
        ollama_host: str = "http://127.0.0.1:11434",
    ) -> "IntentionLayer":
        """
        Convenience constructor that loads the φ model from disk.

        If phi_model_path is None or the file doesn't exist, the layer
        falls back to conservative defaults for all φ estimates.
        """
        from effector.intention.phi_model import PhiAssessor

        assessor = None
        if phi_model_path and phi_model_path.exists():
            assessor = PhiAssessor.load(phi_model_path, ollama_host)
        else:
            print(
                f"[IntentionLayer] φ model not found at {phi_model_path} — "
                "using conservative defaults. Train with phi_model.py."
            )

        reframer = Reframer(model=reframe_model, host=ollama_host)
        return cls(state_bus, assessor=assessor, reframer=reframer)

# ── Demo / CLI ────────────────────────────────────────────────────────────────

def demo(phi_model_path: Path, ollama_host: str) -> None:
    """
    Simulate the Intention Layer processing a mixed inbox in a hot habitat.

    Uses a mock StateBus with manually set habitat state to demonstrate
    how the same emails get routed differently across different pressures.
    """
    import sys

    class _MockBus:
        def __init__(self, pressure: float, theta: float):
            self._state = {"system.pressure": pressure, "habitat.theta": theta}
        def read(self, keys=None):
            return dict(self._state)

    test_events = [
        CommunicationEvent("email",    "Your weekly newsletter from The Browser"),
        CommunicationEvent("calendar", "Reminder: standup in 30 minutes"),
        CommunicationEvent("slack",    "@david mentioned you in #product: quick take?"),
        CommunicationEvent("email",    "URGENT: Client is on the phone asking for you"),
        CommunicationEvent("system",   "PagerDuty: CRITICAL — checkout service is down"),
        CommunicationEvent("email",    "Your order has shipped — arrives Thursday"),
        CommunicationEvent("slack",    "Boss: We need to talk. Are you available now?",
                           preview="Hey — I saw the Q3 numbers and want to discuss before the board call."),
        CommunicationEvent("system",   "Disk usage at 95% — /dev/sda1"),
    ]

    from effector.intention.phi_model import PhiAssessor

    assessor = None
    if phi_model_path.exists():
        assessor = PhiAssessor.load(phi_model_path, ollama_host)
    else:
        print(f"[demo] φ model not found at {phi_model_path} — using fallback φ estimates")

    for pressure, label in [(0.20, "COLD habitat  (pressure=0.20)"),
                             (0.55, "WARM habitat  (pressure=0.55)"),
                             (0.80, "HOT habitat   (pressure=0.80)")]:
        bus = _MockBus(pressure=pressure, theta=0.6)
        layer = IntentionLayer(
            state_bus=bus,
            assessor=assessor,
            reframer=Reframer(host=ollama_host),
        )

        print(f"\n{'═'*60}")
        print(f"  {label}")
        print(f"{'═'*60}")

        for event in test_events:
            decision = layer.process(event)
            action_label = {
                DeliveryAction.PRESENT_NOW: "▶ PRESENT NOW",
                DeliveryAction.SOFTEN:      "≈ SOFTEN     ",
                DeliveryAction.HOLD:        "⏸ HOLD       ",
                DeliveryAction.REFRAME:     "✎ REFRAME    ",
            }[decision.action]
            delay_str = f" +{decision.delay_minutes:.0f}min" if decision.delay_minutes else ""
            phi_str = f"φ={decision.phi_injection:.2f} urg={decision.urgency:.2f}"
            print(
                f"  {action_label}{delay_str:<7}  {phi_str}  "
                f'"{event.subject[:45]}{"…" if len(event.subject)>45 else ""}"'
            )

    print()

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(prog="intention_layer")
    parser.add_argument("--phi-model", type=Path, default=Path("data/phi_model.pkl"))
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument(
        "--demo", action="store_true",
        help="Run demo: process a mixed inbox across cold/warm/hot habitats"
    )
    parser.add_argument(
        "--text", type=str, default=None,
        help="Assess a single notification text"
    )
    parser.add_argument(
        "--pressure", type=float, default=0.5,
        help="Simulated habitat pressure for --text mode (default 0.5)"
    )
    parser.add_argument(
        "--theta", type=float, default=0.6,
        help="Simulated habitat theta for --text mode (default 0.6)"
    )
    args = parser.parse_args()

    if args.demo:
        demo(args.phi_model, args.host)
    elif args.text:
        class _M:
            def read(self, keys=None):
                return {"system.pressure": args.pressure, "habitat.theta": args.theta}
        layer = IntentionLayer.from_paths(
            _M(), phi_model_path=args.phi_model, ollama_host=args.host
        )
        event = CommunicationEvent("cli", args.text)
        decision = layer.process(event)
        print()
        print(decision)
    else:
        parser.print_help()
