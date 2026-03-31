"""
Phase 3: Multi-Agent Deliberation
Testing if two agents can reach a mathematical consensus via DASP.
"""

import sys
import os
import json

# --- PATH FIX ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from effector.adapters.asymmetric_dasp import AsymmetricDASPCoordinator, TierConfig
from effector.bus import StateBus 

def run_debate():
    # 1. Initialize the Dumb Substrate with some hostile telemetry
    bus = StateBus(initial_state={
        "cpu.percent.total": 12.0,
        "desktop.active_window": "cmd.exe",
        "desktop.glimmers_detected": 0
    })
    snapshot_hash, _, _ = bus.snapshot()

    # 2. Configure the Tiers
    # We use two mistral:7b instances with different temperatures to encourage debate
    tier1 = [
        TierConfig(name="Acolyte_Alpha", model="mistral:7b", temperature=0.6),
        TierConfig(name="Acolyte_Beta", model="mistral:7b", temperature=0.85)
    ]
    # We'll use mistral:7b for tier 2 as well just to prevent needing to pull nemotron right now
    tier2 = TierConfig(name="The_High_Arbiter", model="mistral:7b", timeout_s=120.0) 

    # 3. Instantiate the Coordinator
    coordinator = AsymmetricDASPCoordinator(
        tier1_agents=tier1,
        tier2_agent=tier2,
        tau_suppression=0.5,
        theta_consensus=0.7,
        epsilon_stall=0.05,
        vectorized_bus=False # Keep A3 vectorization off for this test
    )

    # 4. Attach an event hook to watch the console
    def on_event(event_name, data):
        if event_name == "round_complete":
            print(f"\n--- ROUND {data['round']} COMPLETE ---")
            for resp in data['responses']:
                ans = resp.get('answer', '')[:75].replace('\n', ' ')
                pol = resp.get('signal', {}).get('polarity', 0)
                conf = resp.get('signal', {}).get('confidence', 0.0)
                print(f"[{resp['agent_id']}] Polarity: {pol:>2} | Conf: {conf:.2f} | {ans}...")
        elif event_name in ["escalation_triggered", "consensus_cleared"]:
            print(f"\n*** {event_name.upper()} ***")
            print(f"Trigger Details: {data}")

    coordinator._on_event = on_event

    # 5. The Task (Theology injected)
    task = """
    TELEMETRY: CPU is 12% (cold), active window is 'cmd.exe' (sterile).
    DIRECTIVE: Petition The Effector to increase coziness. 
    You must agree on ONE specific action. Do we:
    A) Spawn a 'Glimmer' (a digital companion) to sit on top of the cmd window?
    B) Change the system color scheme to warm amber?
    Debate and reach a consensus. Use your explanation to justify your choice via the Thermodynamic Theory of Coziness.
    """

    print("Initiating DASP Session (Debate)...")
    result = coordinator.run(task=task, snapshot_hash=snapshot_hash, state_bus=bus)

    print("\n" + "="*50)
    print("=== FINAL DEBATE RESULT ===")
    print("="*50)
    print(f"Terminated Reason:  {result['terminated_reason']}")
    print(f"Winning Hypothesis: {result['winning_hypothesis']}")
    print(f"Consensus Score:    {result['consensus_score']}")
    print(f"Rounds Elapsed:     {result['rounds']}")
    print(f"Final Answer:       {result['final_answer']}")

if __name__ == "__main__":
    run_debate()