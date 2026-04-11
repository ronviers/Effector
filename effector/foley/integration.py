"""
integration.py — Foley ↔ Effector Connectors
===============================================
Thin adapters that wire the foley pipeline into the existing
Effector event streams without modifying any existing code.

Every connector is optional and fail-safe: if foley crashes or
is absent, DASP, reflex, and IEP continue unaffected.

Usage
-----
    from effector.bus import StateBus
    from effector.foley import create_foley_system
    from effector.foley.integration import (
        DASPFoleyConnector,
        ReflexFoleyConnector,
        IEPFoleyConnector,
    )

    bus = StateBus()
    player, mapper, scheduler = create_foley_system(bus)

    # Wire DASP events (pass to AsymmetricDASPCoordinator as on_event)
    dasp_connector = DASPFoleyConnector(mapper)
    coordinator = AsymmetricDASPCoordinator(
        ...
        on_event=dasp_connector.on_event,
    )

    # Wire reflex events
    reflex_connector = ReflexFoleyConnector(mapper)
    orchestrator = ReflexOrchestrator(
        ...
        on_reflex_executed=reflex_connector.on_reflex_executed,
    )

    # Wire IEP queue observer
    iep_connector = IEPFoleyConnector(mapper)
    # Call iep_connector.on_validation(envelope, verdict) from your queue consumer

    # Wire ambient to system pressure
    pressure_connector = PressureFoleyConnector(player)
    bus.on(pressure_connector.on_bus_event)

Design principles
-----------------
- Each connector wraps exactly one foley component (mapper or player).
- Connectors do not hold state. They translate and forward.
- All methods are exception-safe: a foley error never propagates upstream.
- Connectors are composable: you can stack multiple on_event handlers
  via the compose() helper.
"""

from __future__ import annotations

from typing import Any, Callable

from effector.foley.mapper import FoleyMapper
from effector.foley.player import FoleyPlayer


# ── DASP Connector ────────────────────────────────────────────────────────────

class DASPFoleyConnector:
    """
    Translates AsymmetricDASPCoordinator events into foley events.

    Designed to be passed as the `on_event` callback to
    AsymmetricDASPCoordinator. Can be composed with other on_event
    handlers using the compose() helper below.

    Events mapped
    -------------
    consensus_cleared  → dasp_consensus   (intensity from consensus_score)
    escalation_triggered → dasp_inhibition (intensity 0.8 — notable but not critical)
    tier2_invoked      → dasp_inhibition  (softer — just noting the arbiter woke up)
    session_complete   → (no event — consensus_cleared already handled it)
    """

    def __init__(self, mapper: FoleyMapper) -> None:
        self._mapper = mapper

    def on_event(self, event_name: str, data: dict[str, Any]) -> None:
        try:
            if event_name == "consensus_cleared":
                score = float(data.get("consensus_score", 0.7))
                self._mapper.inject_event(
                    "dasp_consensus",
                    "Sg",           # Snug Prime — the unit of goal achievement
                    min(1.0, score),
                    "system",
                )

            elif event_name == "escalation_triggered":
                trigger = data.get("trigger", "")
                if trigger == "inhibition":
                    # Hard veto fired — notable
                    self._mapper.inject_event("dasp_inhibition", "veto", 0.85, "system")
                else:
                    # Stall → tier-2 escalation — gentler signal
                    self._mapper.inject_event("dasp_inhibition", "stall", 0.55, "system")

            elif event_name == "tier2_invoked":
                # Arbiter woke up — a quiet acknowledgement
                self._mapper.inject_event("dasp_inhibition", "arbiter", 0.40, "system")

        except Exception as exc:
            print(f"[DASPFoleyConnector] Error on {event_name!r}: {exc}")


# ── Reflex Connector ──────────────────────────────────────────────────────────

class ReflexFoleyConnector:
    """
    Translates ReflexOrchestrator outcomes into foley events.

    Pass on_reflex_executed to ReflexOrchestrator's on_reflex_executed
    parameter.

    Events mapped
    -------------
    EXECUTED  → reflex_executed   (crisp, satisfying — deliberation was skipped)
    BYPASSED  → (silent — nothing happened, no event warranted)
    NACK_*    → (silent — this triggers DASP fallback, which has its own events)
    """

    def __init__(self, mapper: FoleyMapper) -> None:
        self._mapper = mapper

    def on_reflex_executed(self, reflex_result: Any) -> None:
        """
        Called by ReflexOrchestrator when a reflex action executes.
        reflex_result is a ReflexResult dataclass from reflex_engine.py.
        """
        try:
            from effector.reflex_engine import ReflexStatus
            if reflex_result.status == ReflexStatus.EXECUTED:
                # Intensity from executions_remaining:
                #   -1 (unlimited) → 0.6 (background hum)
                #   diminishing count → gently increase significance
                remaining = getattr(reflex_result, "executions_remaining", -1)
                if remaining == -1:
                    intensity = 0.60
                elif remaining > 10:
                    intensity = 0.65
                elif remaining > 3:
                    intensity = 0.75
                else:
                    intensity = 0.85  # running low on authorization — worth noting

                self._mapper.inject_event(
                    "reflex_executed",
                    "RAT",
                    intensity,
                    "system",
                )
        except ImportError:
            # reflex_engine not available — skip gracefully
            pass
        except Exception as exc:
            print(f"[ReflexFoleyConnector] Error: {exc}")

    def on_dasp_complete(self, debate_result: dict[str, Any]) -> None:
        """
        Called by ReflexOrchestrator after a full DASP debate.
        Emits a consensus event if score is high enough to be worth noting.
        """
        try:
            score = float(debate_result.get("consensus_score", 0.0))
            if score >= 0.7:
                self._mapper.inject_event(
                    "dasp_consensus",
                    "Sg",
                    min(1.0, score),
                    "system",
                )
        except Exception as exc:
            print(f"[ReflexFoleyConnector] DASP complete error: {exc}")


# ── IEP Queue Connector ───────────────────────────────────────────────────────

class IEPFoleyConnector:
    """
    Translates IEP validation outcomes into foley events.

    Call on_validation() from your EnvelopeQueue consumer loop
    after each item is dequeued and validated.

    Events mapped
    -------------
    ACK   → (silent for routine actions — keeps the soundscape clean)
    NACK  → (silent — the DASP cycle will re-run and has its own events)

    The connector intentionally emits very few events from the IEP layer.
    The IEP is infrastructure; making it audible would create noise,
    not signal. File organization and Glimmer deployment events come
    from the mapper observing the StateBus delta log, not from IEP directly.
    """

    def __init__(self, mapper: FoleyMapper) -> None:
        self._mapper = mapper
        self._ack_count = 0
        self._nack_count = 0

    def on_validation(self, envelope: dict, verdict: Any) -> None:
        """
        Called after each envelope is validated.
        Only emits audio for semantically significant outcomes.
        """
        try:
            status = getattr(verdict, "status", None) or verdict.get("status", "")
            verb = envelope.get("intended_action", {}).get("verb", "")
            target = envelope.get("intended_action", {}).get("target", "")

            if status == "ACK":
                self._ack_count += 1
                # File organization is the one IEP action worth sounding
                if "filesystem" in target or "zen_habitat" in target:
                    self._mapper.inject_event(
                        "file_organized",
                        "archive",
                        0.5,
                        "desktop",
                    )
                elif "glimmer" in target:
                    # A Glimmer placement was approved — it will land
                    # The actual glimmer_land event fires from the StateBus
                    # overlay state change, not here. No duplicate.
                    pass

            elif status and status.startswith("NACK"):
                self._nack_count += 1
                # Snapshot staleness is worth a very quiet note
                if "SNAPSHOT" in status:
                    self._mapper.inject_event(
                        "threshold_crossed",
                        "stale",
                        0.25,
                        "system",
                    )

        except Exception as exc:
            print(f"[IEPFoleyConnector] Error: {exc}")


# ── Pressure ↔ Ambient Connector ──────────────────────────────────────────────

class PressureFoleyConnector:
    """
    Automatically shifts the ambient soundscape based on system pressure.

    Subscribes to StateBus events and calls player.set_ambient() when
    pressure crosses the low/high boundary. The foley gate will suppress
    rapid toggling.

    Usage
    -----
        connector = PressureFoleyConnector(player)
        bus.on(connector.on_bus_event)
    """

    LOW_TO_HIGH_THRESHOLD  = 0.65   # Pressure above this → ambient.high
    HIGH_TO_LOW_THRESHOLD  = 0.45   # Pressure below this → ambient.low
    HYSTERESIS = True               # Use different thresholds to prevent toggling

    def __init__(self, player: FoleyPlayer, initial_key: str = "ambient.low") -> None:
        self._player = player
        self._current = initial_key
        player.set_ambient(initial_key)

    def on_bus_event(self, event_name: str, data: dict[str, Any]) -> None:
        if event_name != "delta_applied":
            return
        pressure = data.get("delta", {}).get("system.pressure")
        if pressure is None:
            return
        try:
            self._maybe_shift(float(pressure))
        except Exception as exc:
            print(f"[PressureFoleyConnector] Error: {exc}")

    def _maybe_shift(self, pressure: float) -> None:
        if self._current == "ambient.low" and pressure >= self.LOW_TO_HIGH_THRESHOLD:
            self._current = "ambient.high"
            self._player.set_ambient("ambient.high")
        elif self._current == "ambient.high" and pressure <= self.HIGH_TO_LOW_THRESHOLD:
            self._current = "ambient.low"
            self._player.set_ambient("ambient.low")


# ── Composition helpers ───────────────────────────────────────────────────────

def compose_event_handlers(*handlers: Callable[[str, dict], None]) -> Callable[[str, dict], None]:
    """
    Compose multiple on_event handlers into one.

    Passes each event to all handlers in order. If any handler raises,
    the exception is caught and logged; remaining handlers still execute.

    Usage
    -----
        coordinator = AsymmetricDASPCoordinator(
            ...
            on_event=compose_event_handlers(
                your_existing_handler,
                dasp_connector.on_event,
            )
        )
    """
    def combined(event_name: str, data: dict) -> None:
        for handler in handlers:
            try:
                handler(event_name, data)
            except Exception as exc:
                print(f"[compose_event_handlers] Handler {handler.__name__!r} error: {exc}")
    combined.__name__ = "composed_handler"
    return combined


def wire_foley_to_main_loop(
    coordinator: Any,
    orchestrator: Any,
    bus: Any,
    mapper: FoleyMapper,
    player: FoleyPlayer,
) -> dict[str, Any]:
    """
    Convenience function: wire all connectors in one call.

    Returns a dict of the created connectors (for inspection / testing).

    This function MODIFIES the coordinator and orchestrator in-place
    by wrapping their event callbacks. It is idempotent: calling it
    twice will double-wrap, which is harmless but wasteful.

    Example
    -------
        bus = StateBus()
        player, mapper, scheduler = create_foley_system(bus)
        coordinator = AsymmetricDASPCoordinator(...)
        orchestrator = ReflexOrchestrator(...)

        wires = wire_foley_to_main_loop(
            coordinator, orchestrator, bus, mapper, player
        )
    """
    dasp_c    = DASPFoleyConnector(mapper)
    reflex_c  = ReflexFoleyConnector(mapper)
    iep_c     = IEPFoleyConnector(mapper)
    pressure_c = PressureFoleyConnector(player)

    # Wrap coordinator on_event
    existing_dasp = getattr(coordinator, "_on_event", lambda *a: None)
    coordinator._on_event = compose_event_handlers(existing_dasp, dasp_c.on_event)

    # Wire reflex orchestrator callbacks
    if orchestrator is not None:
        orchestrator._on_reflex_executed = reflex_c.on_reflex_executed
        orchestrator._on_dasp_complete = reflex_c.on_dasp_complete

    # Wire pressure → ambient
    bus.on(pressure_c.on_bus_event)

    return {
        "dasp": dasp_c,
        "reflex": reflex_c,
        "iep": iep_c,
        "pressure": pressure_c,
    }
