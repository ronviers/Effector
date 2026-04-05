import threading
from effector.foley import config
from effector.foley.resolver import FoleyResolver

class FoleyPlayer:
    def __init__(self, backend, gate=None, master_volume=1.0):
        self.backend = backend
        self.gate = gate
        self.master_volume = master_volume
        self.resolver = FoleyResolver()
        self._ambient_lock = threading.Lock()
        self._ambient_key = None

    def play(self, event):
        if self.gate and not self.gate.allow(event):
            return
        vol = self._compute_volume(event)
        key = self.resolver.resolve(event)
        if key:
            self.backend.play(config.ASSET_MAP.get(key), vol, 44100)

    def _compute_volume(self, event):
        pg = config.POSITION_GAIN.get(event.position, 1.0)
        eg = config.EVENT_GAIN.get(event.event_type, 1.0)
        return self.master_volume * event.intensity * pg * eg

    def set_ambient(self, key):
        with self._ambient_lock:
            self._ambient_key = key if key.startswith("ambient.") else None

    def shutdown(self):
        pass
