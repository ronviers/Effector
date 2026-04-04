"""
asymmetric_dasp_patch.py
========================
This file documents the three changes to asymmetric_dasp.py required
by the two-phase refactor. Apply them as targeted str_replace edits.

CHANGE 1  — Delete AGENT_SYSTEM_PROMPT entirely.
CHANGE 2  — Replace _call_ollama() with a thin wrapper around two_phase_call().
CHANGE 3  — Add `characterizer_model` field to TierConfig and thread it through
             _run_tier2_arbiter() and the tier-1 dispatch loop.

No other logic in AsymmetricDASPCoordinator changes.
The signal accumulation, gate evaluation, escalation, and result
assembly are all protocol mechanics — they stay exactly as written.
"""

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 1
# Remove the entire AGENT_SYSTEM_PROMPT constant.
#
# OLD (delete this block in full):
# ─────────────────────────────────────────────────────────────────────────────
"""
AGENT_SYSTEM_PROMPT = \"\"\"\
You are a reasoning agent in a structured multi-agent debate about system telemetry.

Your ONLY output is a single JSON object — no markdown, no preamble.

Schema:
{{
  "type": "agent.response",
  ...
}}
...
\"\"\"
"""
# ─────────────────────────────────────────────────────────────────────────────
# Nothing replaces it. The Reasoning Node and Characterizer each carry their
# own prompts internally inside two_phase_adapter.py.
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 2
# Replace _call_ollama() with a thin delegation to two_phase_call().
#
# OLD signature (and full implementation) is deleted.
# NEW implementation:
# ─────────────────────────────────────────────────────────────────────────────

# At the top of asymmetric_dasp.py, add this import alongside the others:
#
#   from effector.adapters.two_phase_adapter import two_phase_call

def _call_ollama(
    agent_id: str,
    model: str,
    host: str,
    session_id: str,
    round_num: int,
    task: str,
    snapshot_hash: str,
    others: list[dict],
    temperature: float = 0.7,
    timeout_s: float = 90.0,
    characterizer_model: str | None = None,   # NEW parameter
) -> dict | None:
    """
    Thin wrapper — delegates entirely to the two-phase adapter.

    The original implementation of this function combined reasoning and
    signal serialization in one LLM call using an instructional JSON prompt
    (AGENT_SYSTEM_PROMPT). That pattern is prohibited by DASP §4.3.

    This wrapper preserves the call signature used throughout
    AsymmetricDASPCoordinator so the rest of the coordinator is unchanged.
    """
    return two_phase_call(
        agent_id=agent_id,
        model=model,
        host=host,
        session_id=session_id,
        round_num=round_num,
        task=task,
        snapshot_hash=snapshot_hash,
        others=others,
        temperature=temperature,
        timeout_s=timeout_s,
        characterizer_model=characterizer_model,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 3
# Add `characterizer_model` to TierConfig and thread it through call sites.
# ─────────────────────────────────────────────────────────────────────────────

# In the TierConfig dataclass, add one field after `temperature_schedule`:
#
#   characterizer_model: str | None = None
#   # Ollama model for Phase 2 (Characterizer). Defaults to the reasoning model.
#   # Set to a small, fast model (e.g. "mistral:7b") to reduce Characterizer latency.

# In AsymmetricDASPCoordinator._run_tier2_arbiter(), the _call_ollama() call
# already passes kwargs through, so no change needed there.

# In the Tier-1 dispatch loop (inside run()), change:
#
#   OLD:
#       resp = _call_ollama(
#           agent_id=tier.name,
#           model=tier.model,
#           host=tier.host,
#           session_id=session_id,
#           round_num=round_num,
#           task=task,
#           snapshot_hash=snapshot_hash,
#           others=others_summary,
#           temperature=tier.get_temperature(local_round),
#           timeout_s=tier.timeout_s,
#       )
#
#   NEW (add characterizer_model):
#       resp = _call_ollama(
#           agent_id=tier.name,
#           model=tier.model,
#           host=tier.host,
#           session_id=session_id,
#           round_num=round_num,
#           task=task,
#           snapshot_hash=snapshot_hash,
#           others=others_summary,
#           temperature=tier.get_temperature(local_round),
#           timeout_s=tier.timeout_s,
#           characterizer_model=tier.characterizer_model,   # NEW
#       )

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT TIER PRESETS — updated to include characterizer_model
# ─────────────────────────────────────────────────────────────────────────────
#
# Replace the existing DEFAULT_TIER1 and DEFAULT_TIER2 with:

DEFAULT_TIER1_NEW = [
    # TierConfig(
    #     name="mistral",
    #     model="mistral",
    #     host="http://127.0.0.1:11434",
    #     temperature=0.65,
    #     max_rounds=3,
    #     characterizer_model="qwen2.5:14b",   # same model; fast enough for tool calls
    # ),
    # TierConfig(
    #     name="qwen",
    #     model="qwen2.5-coder:32b",
    #     host="http://127.0.0.1:11434",
    #     temperature=0.75,
    #     max_rounds=3,
    #     characterizer_model="qwen2.5:14b",   # delegate characterization to smaller model
    # ),
]

DEFAULT_TIER2_NEW = (
    # TierConfig(
    #     name="nemotron",
    #     model="nemotron",
    #     host="http://127.0.0.1:11434",
    #     temperature=0.5,
    #     timeout_s=120.0,
    #     max_rounds=1,
    #     characterizer_model="mistral:7b",
    # )
)
