# Use raw strings to safely handle Windows backslashes
PURR = r"O:\Desktop\purring.wav"
TRAIN = r"O:\Desktop\trainAtmo.wav"
ISTANBUL = r"O:\Desktop\istanbulAtmo.wav"

ASSET_MAP = {
    # ── Glimmer / Interaction Sounds (Purring) ──
    "glimmer.land.high": PURR, 
    "glimmer.land.mid": PURR,
    "glimmer.land.low": PURR, 
    "glimmer.depart": PURR,
    "glimmer.idle": PURR, 
    "file.organized": PURR,         # Purr when files are moved to Zen_Habitat
    "reflex.executed": PURR,        # Purr when the Reflex fast-path hits
    "dasp.consensus": PURR,         # Purr when agents agree
    "dasp.inhibition": PURR,
    "context.switch": PURR,

    # ── Ambient Backgrounds & Thresholds ──
    "ambient.low": ISTANBUL,        # Low pressure = Relaxed Istanbul atmosphere
    "ambient.high": TRAIN,          # High pressure = Busy train station
    "system.threshold.low": ISTANBUL,
    "system.threshold.mid": ISTANBUL,
    "system.threshold.high": TRAIN,
    "system.threshold.critical": TRAIN,
}

POSITION_GAIN = {"desktop": 1.0, "system": 1.0, "taskbar": 0.8, "window": 0.9}

EVENT_GAIN = {k: 1.0 for k in [
    "glimmer_land", "glimmer_depart", "glimmer_idle", "threshold_crossed",
    "context_switch", "reflex_executed", "dasp_consensus", "dasp_inhibition",
    "file_organized", "ambient_shift"
]}

# Dropped the cooldown slightly so overlapping agent approvals can trigger overlapping purrs
GATE_CONFIG = {"cooldown_ms": 100, "dedupe_window_ms": 50, "burst_window_ms": 1000, "burst_limit": 5, "overrides": {}}