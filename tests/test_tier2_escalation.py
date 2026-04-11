"""
Tier 2 Escalation Test
Forces a Stall Gate deadlock to verify the Cloud Arbiter injection.
"""

import sys
import os
import json

# --- PATH FIX ---

from effector.adapters.asymmetric_dasp import AsymmetricDASPCoordinator, TierConfig
from effector.bus import StateBus 

def run_escalation_test():
    # 1. Initialize the Substrate
    bus = StateBus(initial_state={"desktop.unsorted_files": 42})
    snapshot_hash, _, _ = bus.snapshot()

    # 2. Configure the Tiers
    tier1 = [
        TierConfig(name="Acolyte_Alpha", model="mistral:7b", temperature=0.7, max_rounds=3),
        TierConfig(name="Acolyte_Beta", model="mistral:7b", temperature=0.7, max_rounds=3)
    ]
    
    # Using your heavy 32b model to act as the Cloud Arbiter
    tier2 = TierConfig(name="The_High_Arbiter", model="nemotron-3-super:cloud", temperature=0.5, timeout_s=120.0)

    # 3. Instantiate the Coordinator (Rigged for a Stall)
    coordinator = AsymmetricDASPCoordinator(
        tier1_agents=tier1,
        tier2_agent=tier2,
        tau_suppression=0.5,
        theta_consensus=0.99, # Near impossible to clear in round 1
        epsilon_stall=2.0,    # GUARANTEES a stall on round 2 (delta is always < 2.0)
        vectorized_bus=False
    )

    # 4. Attach an event hook to watch the drama unfold
    def on_event(event_name, data):
        if event_name == "round_complete":
            print(f"\n--- {data['tier'].upper()} ROUND {data['round']} COMPLETE ---")
            for resp in data['responses']:
                ans = resp.get('answer', '').replace('\n', ' ')
                pol = resp.get('signal', {}).get('polarity', 0)
                print(f"[{resp['agent_id']}] Pol: {pol:>2} | {ans}...")
                
        elif event_name == "escalation_triggered":
            print("\n🚨 *** ESCALATION TRIGGERED *** 🚨")
            print(f"Reason: {data['trigger'].upper()} gate fired!")
            print(f"Escalating from {data['tier_from']} to {data['tier_to']}...")
            
        elif event_name == "tier2_invoked":
            print(f"\n☁️  Awakening the Cloud Arbiter ({data['model']})... This may take a moment.")

    coordinator._on_event = on_event

    # 5. The Task
    task = """
    TELEMETRY: Desktop has 42 unsorted files.
    DIRECTIVE: Acolyte Alpha proposes deleting them. Acolyte Beta proposes putting them in a 'Snug Folder'. 
    You must fiercely disagree. Defend your position using the Thermodynamic Theory of Coziness.
    """

    print("Initiating DASP Session (Forcing Escalation)...")
    result = coordinator.run(task=task, snapshot_hash=snapshot_hash, state_bus=bus)

    print("\n" + "="*50)
    print("=== FINAL DEBATE RESULT ===")
    print("="*50)
    print(f"Terminated Reason:  {result['terminated_reason']}")
    print(f"Tier 2 Injected:    {result['tier2_injected']}")
    print(f"Winning Hypothesis: {result['winning_hypothesis']}")
    print(f"Final Answer:       {result['final_answer']}")

if __name__ == "__main__":
    run_escalation_test()