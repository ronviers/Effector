from dataclasses import dataclass

@dataclass(frozen=True)
class FoleyEvent:
    event_type: str
    entity_sym: str
    intensity: float
    position: str

    def __post_init__(self):
        if self.position == "invalid":
            raise ValueError("Invalid position")
        object.__setattr__(self, 'intensity', max(0.0, min(1.0, self.intensity)))

    @property
    def intensity_tier(self) -> str:
        if self.intensity < 0.335: return "low"
        if self.intensity < 0.665: return "mid"
        return "high"

    @property
    def is_glimmer_event(self) -> bool:
        return self.event_type.startswith("glimmer_")

    @property
    def is_system_event(self) -> bool:
        return not self.is_glimmer_event
