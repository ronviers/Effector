"""
Phase 4: Semantic Drift & IEP Verification
Simulates the user altering the OS environment mid-debate, 
testing the IEPValidator's ability to block stale execution.
"""

import sys
import os
import uuid
import json

# --- PATH FIX ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from effector.bus import StateBus
from effector.adapters.asymmetric_dasp import AsymmetricDASPCoordinator, TierConfig
from effector.queue.iep_queue import IEPBuilder, IEPValidator

def run_drift_test():
    # 1. Initialize the Substrate
    bus = StateBus(initial_state={
        "cpu.percent.total": 12.0,
        "desktop.active_window": "cmd.exe", # <--- Agents think this is the active window
    })
    
    # Take the snapshot AT THE START of the debate
    snapshot_hash, _, _ = bus.snapshot()

    # 2. Configure a single fast agent
    tier1 = [TierConfig(name="Pip_Scones_Acolyte", model="mistral:7b", temperature=0.7)]
    coordinator = AsymmetricDASPCoordinator(tier1, tier1[0]) # Single tier for speed

    # 3. The "Pull the Plug" Hook
    def on_event(event_name, data):
        if event_name == "round_started":
            print(f"\n[Agent is thinking...] Round {data['round']} started.")
            
        elif event_name == "round_complete":
            # Mid-debate, the user clicks on a different application!
            print("\n🚨 [SYSTEM EVENT] User clicked away! Active window is now 'SpacePinball.exe'")
            
            # We artificially mutate the StateBus behind the agent's back
            with bus._lock:
                bus._state["desktop.active_window"] = "SpacePinball.exe"

    coordinator._on_event = on_event

    # 4. The Task
    task = """
    TELEMETRY: Active window is 'cmd.exe'.
    DIRECTIVE: Propose an action to cozy-ify 'cmd.exe'.
    """

    print("Initiating DASP Session...")
    debate_result = coordinator.run(task=task, snapshot_hash=snapshot_hash, state_bus=bus)

    print("\n" + "="*50)
    print("=== DEBATE FINISHED ===")
    print(f"Agent Proposed: {debate_result['final_answer']}")
    print("Wrapping proposal in an Intention Envelope...")
    
    # 5. Build the IEP Envelope
    envelope = IEPBuilder.from_debate_result(
        debate_result=debate_result,
        state_bus_snapshot_hash=snapshot_hash, # The hash from BEFORE Pinball opened
        keys_affected=["desktop.color_scheme"],
        agent_id="Pip_Scones_Acolyte"
    )

    print("\nSubmitting Envelope to the IEP Validator...")
    
    # 6. Run the Pre-flight Checks
    validator = IEPValidator(state_bus=bus)
    val_result = validator.validate(envelope)

    print("\n" + "="*50)
    print("=== IEP VERIFICATION RESULT ===")
    print(f"Status:         {val_result.status}")
    if val_result.failure_reason:
        print(f"Failure Reason: {val_result.failure_reason}")
    print("="*50)
    
    if val_result.status == "NACK_SNAPSHOT_HASH":
        print("\n✅ SUCCESS: The IEP Validator successfully caught the semantic drift and blocked the stale action!")
    else:
        print("\n❌ FAILURE: The Validator let a stale action slip through!")

if __name__ == "__main__":
    run_drift_test()