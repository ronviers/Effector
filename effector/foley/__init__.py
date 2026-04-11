from effector.foley.player import FoleyPlayer
from effector.foley.mapper import FoleyMapper
from effector.foley.scheduler import FoleyScheduler
from effector.foley.backend import build_backend
from effector.foley.gate import EventGate

def create_foley_system(bus, backend=None):
    if backend is None:
        backend = build_backend(["null"])
    player = FoleyPlayer(backend=backend, gate=EventGate())
    mapper = FoleyMapper(bus=bus, emitter=player.play)
    scheduler = FoleyScheduler(emitter=player.play, config={"max_pending": 10, "resolution_ms": 5})
    return player, mapper, scheduler
