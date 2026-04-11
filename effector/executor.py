"""
executor.py — The Physical Execution Layer
==========================================
Pops ACK'd Intention Envelopes from the queue and translates them into 
actual OS/Habitat changes. This is where Worldspillage becomes real.
"""

import os
import shutil
from pathlib import Path
from typing import Any

class LocalExecutor:
    def __init__(self, state_bus: Any, foley_mapper: Any = None):
        self.bus = state_bus
        self.foley_mapper = foley_mapper
        self.desktop_path = Path(os.path.expanduser("~/Desktop"))

    def execute(self, envelope: dict[str, Any]) -> dict[str, Any]:
        """
        Executes the authorized envelope. Returns the actual delta applied.
        """
        action = envelope.get("intended_action", {})
        verb = action.get("verb", "")
        target = action.get("target", "")
        params = action.get("parameters", {})
        
        actual_delta = {}

        if verb == "WRITE":
            if target == "os.filesystem.zen_habitat" or target == "os.filesystem.desktop_icons":
                actual_delta = self._organize_zen_habitat()
            elif target == "desktop.overlay.glimmer":
                actual_delta = self._spawn_glimmer(params)
            else:
                # Generic world state update
                actual_delta = params
                
        # Apply the actual changes back to the StateBus
        if actual_delta:
            self.bus.apply_delta(
                envelope_id=envelope.get("envelope_id"),
                delta=actual_delta,
                agent_id=envelope.get("agent", {}).get("id", "executor")
            )
            
        return actual_delta

    def _organize_zen_habitat(self) -> dict[str, Any]:
        """Moves loose desktop files into a cozy 'Zen Habitat' folder."""
        zen_dir = self.desktop_path / "Zen_Habitat"
        zen_dir.mkdir(exist_ok=True)
        
        moved_count = 0
        for item in self.desktop_path.iterdir():
            # Skip hidden files, shortcuts, and the zen dir itself
            if item.is_file() and not item.name.startswith(".") and not item.name.endswith(".lnk"):
                try:
                    shutil.move(str(item), str(zen_dir / item.name))
                    moved_count += 1
                except Exception:
                    pass
                    
        if moved_count > 0 and self.foley_mapper:
            # Trigger the satisfying "file_organized" foley sound!
            self.foley_mapper.inject_event("file_organized", "archive", 0.8, "desktop")
            
        return {"desktop.unsorted_files": 0, "desktop.zen_habitat_items": moved_count}

    def _spawn_glimmer(self, params: dict) -> dict[str, Any]:
        """Simulates spawning a digital companion."""
        if self.foley_mapper:
            self.foley_mapper.inject_event("glimmer_land", "Fc", 0.9, "desktop")
        return {"desktop.glimmers_detected": 1, "desktop.overlay_active": True}