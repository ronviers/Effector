from effector.foley.events import FoleyEvent

class FoleyResolver:
    def resolve(self, event: FoleyEvent) -> str | None:
        if event.event_type == "glimmer_idle" and event.intensity_tier == "low":
            return None
        if event.event_type == "glimmer_land":
            return f"glimmer.land.{event.intensity_tier}"
        if event.event_type == "threshold_crossed":
            if event.intensity >= 0.9: return "system.threshold.critical"
            if event.intensity >= 0.75: return "system.threshold.high"
            if event.intensity >= 0.5: return "system.threshold.mid"
            return "system.threshold.low"
        
        mapping = {
            "glimmer_depart": "glimmer.depart", "glimmer_idle": "glimmer.idle",
            "context_switch": "context.switch", "reflex_executed": "reflex.executed",
            "dasp_consensus": "dasp.consensus", "dasp_inhibition": "dasp.inhibition",
            "file_organized": "file.organized",
            "ambient_shift": "ambient.low" if event.intensity_tier == "low" else "ambient.high"
        }
        return mapping.get(event.event_type)
