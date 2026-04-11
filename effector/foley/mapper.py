from effector.foley.events import FoleyEvent

class FoleyMapper:
    def __init__(self, bus, emitter):
        self._bus = bus
        self._emitter = emitter
        self._prev = {}
        if hasattr(self._bus, "on"):
            self._bus.on(self.on_bus_event)

    def inject_event(self, event_type, entity_sym, intensity, position):
        self._emitter(FoleyEvent(event_type, entity_sym, intensity, position))

    def on_bus_event(self, event_name, data):
        if event_name == "delta_applied":
            delta = data.get("delta", {})
            if "system.pressure" in delta and delta["system.pressure"] >= 0.9:
                self.inject_event("threshold_crossed", "pressure", 1.0, "system")
