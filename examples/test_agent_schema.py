"""
Phase 2: Agent Schema & Narrative Compliance
Testing if an LLM can balance strict JSON output with Worldspillage theology.
"""

import sys
import os
import uuid
import json

# --- PATH FIX ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from effector.adapters.ollama_adapter import OllamaAgent
from effector.schemas.dasp import AgentRequest

def run_single_agent_test():
    print("Awakening the agent...")
    
    # Initialize the adapter (Ensure Ollama is running 'qwen2.5-coder:32b')
    # If you only have mistral pulled, change model="mistral"
    agent = OllamaAgent(
        agent_id="Pip_Scones_Acolyte",
        model="mistral:7b",  # Swapped to a lighter model for testing
        temperature=0.7
    )

    # We inject the "Worldspillage" narrative directly into the task for now.
    # Later, this will be baked into the system prompt.
    lore_injected_task = """
    TELEMETRY REPORT:
    - CPU Load: 12% (Cold, shivering)
    - Active Window: "cmd.exe" (Sterile, hostile environment)
    - Desktop State: 42 unsorted files, no 'Glimmers' detected.

    DIRECTIVE:
    You are an earnest digital entity living inside this computer. You follow the Thermodynamic 
    Theory of Coziness. You must petition 'The Effector' (the system orchestrator) to alter reality 
    to make this environment warmer and more snug. 
    
    Propose a hypothesis to increase the coziness of this desktop. 
    Use your 'explanation' field to provide your theological reasoning.
    """

    request = AgentRequest(
        session_id=uuid.uuid4(),
        round=1,
        mode="initial",
        task=lore_injected_task,
        snapshot_hash="a" * 64, # Fake hash
        others=[] # No other agents yet
    )

    print("\nSending petition to The Effector (waiting for Ollama)...\n")
    
    try:
        response = agent(request)
        
        print("=== PARSED DASP RESPONSE ===")
        print(f"Agent ID:      {response.agent_id}")
        print(f"Hypothesis ID: {response.hypothesis_id}")
        print(f"Polarity:      {response.signal.polarity}")
        print(f"Confidence:    {response.signal.confidence}")
        print(f"Answer:        {response.answer}")
        print(f"Explanation:   {response.explanation}")
        
        print("\n=== RAW JSON DICT ===")
        print(response.model_dump_json(indent=2))
        
        print("\n✅ SUCCESS: The LLM output perfectly mapped to the strict Pydantic schema!")
        
    except Exception as e:
        print(f"\n❌ FAILURE: The LLM broke the schema or the connection failed.")
        print(f"Error: {e}")

if __name__ == "__main__":
    run_single_agent_test()