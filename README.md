# Effector

**DASP-1.0 + IEP-1.0 Multi-Agent Coordination Library**

Effector implements the full Cognitive Agent Architecture protocol suite: structured multi-agent deliberation (DASP) through to verified state execution (IEP).

---

## Install

```bash
pip install -e .
# or directly from source:
pip install pydantic anthropic
```

## Quick start

```python
from effector import start_session, AgentInfo, make_agent_callable

result = start_session(
    agents=[AgentInfo(id="alpha"), AgentInfo(id="beta")],
    goal="Should we enable caching for this API endpoint?",
    agent_registry={
        "alpha": make_agent_callable("alpha"),
        "beta":  make_agent_callable("beta"),
    },
    state={
        "cache_enabled": False,
        "api_calls_today": 847,
    },
)

print(result["debate_result"].final_answer)
print(result["iep_result"].status)
```

## Architecture

```
ORCHESTRATION
      │
  DELIBERATION  ← DASP-1.0 governs this layer
  (debate rounds, signal superposition, trigger detection)
      │  final_answer + coalition + snapshot_hash
  EXECUTION     ← IEP-1.0 governs this layer
  (envelope emission, pre-flight verification, state write)
      │
  STATE BUS
  (immutable world-state vector, append-only delta log)
```

## Package layout

```
effector/
├── __init__.py              # Public API
├── session.py               # start_session(), EffectorSession, IEPExecutor
├── schemas/
│   ├── dasp.py              # DebateStart, AgentRequest, AgentResponse, DebateResult
│   └── iep.py               # IntentionEnvelope, ReflexEnvelope, VerificationResult
├── coordinator/
│   ├── coordinator.py       # DASPCoordinator — main debate loop
│   └── signal_engine.py     # SignalEngine — S_g / S_i / gate evaluation
├── state_bus/
│   ├── bus.py               # StateBus — world state, SHA-256 snapshots, delta log
│   └── verifier.py          # IEPVerifier — pre-flight + post-execution checks
├── adapters/
│   └── anthropic_adapter.py # AnthropicAgent, ToolRegistry, make_agent_callable
└── tests/
    └── test_effector.py       # Full unit test suite (no LLM required)
```

## Event hooks

```python
session = EffectorSession(state_bus=bus, rules=rules)

session.on("round_complete",    lambda d: print(f"Round {d['round']} done"))
session.on("trigger_fired",     lambda d: print(f"Trigger: {d['trigger']}"))
session.on("envelope_received", lambda e: print(f"IEP: {e.envelope_id}"))
session.on("session_complete",  lambda d: print(f"Done: {d['terminated_reason']}"))
```

## Operational modes

| Mode | DASP | Description |
|------|------|-------------|
| `DELIBERATIVE` | Full | All actions require completed debate session |
| `MIXED` | + RAT issuance | Routine actions use pre-authorized Reflex Envelopes |
| `REFLEX` | RAT-only | Real-time loops; deliberation is offline/async |
| `SUPERVISED` | + human gate | Human must ACK before IEP envelope is emitted |

```python
from effector import OperationalMode

session = EffectorSession(mode=OperationalMode.supervised)
result  = session.run(..., require_human_ack=True)
# → prints ACK/NACK prompt to terminal before execution
```

## Tools

```python
from effector import ToolRegistry, make_agent_callable

tools = ToolRegistry()
tools.register("Read_Screen",    "Read the current screen contents", {})
tools.register("Create_Overlay", "Draw a text overlay on the screen", {
    "type": "object",
    "properties": {"text": {"type": "string"}, "x": {"type": "integer"}}
})
tools.register("Move_File",      "Move a file to a new location", {
    "type": "object",
    "properties": {"src": {"type": "string"}, "dst": {"type": "string"}}
})

agent = make_agent_callable("my_agent", tool_registry=tools)
```

## Running tests

```bash
python -m pytest effector/tests/ -v
# All tests run without an API key (mock agents only)
```

## Running the live example

```bash
export ANTHROPIC_API_KEY=sk-...
python example_debate.py
```

## Protocol compliance

| Feature | Spec ref | Status |
|---------|----------|--------|
| DASP signal superposition | DASP §6 | ✅ |
| Inhibition Gate (P1) | DASP §6 | ✅ |
| Stall Gate (P2) | DASP §6 | ✅ |
| Consensus Gate (P3) | DASP §6 | ✅ |
| Copy / Swap detection | DASP §5 | ✅ |
| Optimize phase | DASP §7 | ✅ (hook-based) |
| DASP result contract | DASP §8 | ✅ |
| IEP pre-flight (5 checks) | IEP §5 | ✅ |
| Post-execution divergence | IEP §6 | ✅ |
| DELEGATE verb semantics | IEP §7 | ✅ (schema) |
| Reflex Bypass (IEP-A1) | Addendum A | ✅ (schema + RAT store) |
| Operational modes (IEP-A2) | Addendum A | ✅ |
| Vectorized State Bus (IEP-A3) | Addendum A | schema ready |
| Agent reputation (DASP-A1) | Addendum A | ✅ |
| Snapshot hash binding | Integration §1 | ✅ |
| Epsilon parameter inheritance | Integration §3 | ✅ |
