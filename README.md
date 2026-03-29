# Effector Engine

**Headless multi-agent reasoning loop over live OS telemetry.**

Effector implements a complete DASP-1.0 + IEP-1.0 protocol stack and runs it continuously against real system state — CPU load, memory pressure, active windows, network throughput — using local LLMs as the reasoning layer, with automatic cloud escalation on deadlock.

```
  OS STATE (psutil)
       │
  ─────▼─────────────────────────────────────────────────
  A1 · TELEMETRY POLLER        TelemetryPoller → StateBus
  ─────┬─────────────────────────────────────────────────
       │ snapshot_hash
  ─────▼─────────────────────────────────────────────────
  A2 · ASYMMETRIC DASP         mistral + qwen (local)
       │                       ↓ on stall/inhibition gate
       │                       nemotron (cloud arbiter)
  ─────┬─────────────────────────────────────────────────
       │ debate_result + consensus_score
  ─────▼─────────────────────────────────────────────────
  A3 · IEP VALIDATION          IEPBuilder → IEPValidator
       │                       → EnvelopeQueue (JSONL log)
  ─────────────────────────────────────────────────────
```

---

## Background

This project is built on two formal protocols described in [*Cognitive Agent Architecture: Protocol Suite v1.0*](docs/protocol_suite.pdf):

**DASP-1.0 — Debate-as-a-Service Protocol** governs deliberation. A group of agents debate a hypothesis through structured rounds, emitting numerical signals (generative strength, inhibitory pressure, polarity) that are composed mathematically rather than resolved by majority vote. Three gates — Inhibition (P1), Stall (P2), Consensus (P3) — determine when and how the session terminates.

**IEP-1.0 — Intention Envelope Protocol** governs execution. Any agent that wants to write to shared state must first emit a typed envelope declaring its intent, a snapshot hash binding it to a specific world state, and a falsifiable forward model (`expected_state_change`). A verifier runs five pre-flight checks before the write is permitted. Divergence between the predicted and actual delta drives replanning.

The two protocols connect through the `snapshot_hash` field. Every envelope is cryptographically bound to the exact world state the deliberating agents reasoned from — creating an unbroken, auditable chain from observation to decision to action.

---

## Architecture

### A1 · Substrate Telemetry

`src/effector/telemetry/`

A daemon thread polls OS state via `psutil` every N seconds and writes structured key-value deltas to a `StateBus` dictionary. All canonical key names live in `state_keys.py` as a frozen `KEYS` singleton — agents reference these names in their `expected_state_change` declarations to keep the state ontology stable across sessions.

**Metrics collected**

| Group | Keys |
|---|---|
| CPU | `cpu.percent.total`, `cpu.percent.per_core`, `cpu.freq.mhz`, `cpu.ctx_switches_sec` |
| Memory | `ram.used_mb`, `ram.percent`, `swap.used_mb`, `swap.percent` |
| Disk I/O | `disk.read_mb_s`, `disk.write_mb_s`, `disk.iops_read`, `disk.iops_write` |
| Network | `net.sent_kb_s`, `net.recv_kb_s`, `net.connections_count` |
| Process | `process.count`, `process.top_cpu.name`, `process.top_mem.mb` |
| Desktop | `desktop.active_window`, `desktop.active_process` |
| Health | `system.pressure`, `system.thermal_alert` |

**System pressure score** is a composite metric computed as:

```
P = 0.50 · f(cpu) + 0.35 · f(ram) + 0.15 · f(swap)
f(x) = (x / 100)^1.5                                   # nonlinear — emphasises > 80%
```

The exponent curve means a system at 90% utilisation scores disproportionately higher than one at 45%, matching the nonlinear character of real performance degradation. `thermal_alert` fires when `P > 0.85`.

Disk I/O and network metrics are computed as per-second rates using a `_RateCounter` that tracks the delta between psutil's monotonically increasing byte counters across poll cycles.

Active window detection uses `GetForegroundWindow` + `GetWindowThreadProcessId` on Windows. On Linux/macOS it falls back to the highest-CPU process name from psutil.

```python
from effector.telemetry.poller import TelemetryPoller
from effector.state_bus.bus import StateBus

bus = StateBus()
poller = TelemetryPoller(bus, interval_s=2.0, on_poll=lambda d: print(d))
poller.start()
# ... runs in background daemon thread
poller.stop()

# or single-shot:
delta = TelemetryPoller(bus).poll_once()
```

---

### A2 · Asymmetric DASP

`src/effector/adapters/asymmetric_dasp.py`

A two-tier debate coordinator. **Tier-1 agents** (mistral + qwen, running locally via Ollama) debate for up to `max_rounds`. The **Tier-2 arbiter** (nemotron) is instantiated only when local agents deadlock — keeping cloud API costs and latency entirely off the happy path.

**Escalation triggers**

| Gate | Condition | Action |
|---|---|---|
| **Inhibition (P1)** | `S_i(H) ≥ τ · S_g(H)` | Tier-2 arbiter injected |
| **Stall (P2)** | `\|ΔS_net\| < ε_stall` | Tier-2 arbiter injected |
| **Consensus (P3)** | `S_net(H) ≥ θ` | Session terminates — no escalation |

Gates are evaluated in strict priority order. P1 (Inhibition) always preempts P3 (Consensus): a strong veto cannot be overridden by the size of the generative coalition, mirroring the role of striatal inhibitory interneurons in biological motor control.

When Tier-2 is invoked, the arbiter receives full session context including the deadlock reason ("stall" or "inhibition"), all prior agent positions, and confidence scores. Its response is injected as a single authoritative round, gates are re-evaluated, and the session terminates.

**Signal math** (gate functions are pure, standalone, independently testable):

```
S_g(H) = Σ confidence · generative_strength     for polarity ≥ 0
S_i(H) = Σ confidence · inhibitory_pressure     for polarity ≤ 0
S_net(H) = S_g(H) − S_i(H)
```

```python
from effector.adapters.asymmetric_dasp import AsymmetricDASPCoordinator, TierConfig

coord = AsymmetricDASPCoordinator(
    tier1_agents=[
        TierConfig(name="mistral", model="mistral", max_rounds=3),
        TierConfig(name="qwen", model="qwen2.5-coder:32b", max_rounds=3),
    ],
    tier2_agent=TierConfig(name="nemotron", model="nemotron", timeout_s=120.0),
    tau_suppression=0.5,
    theta_consensus=0.65,
    epsilon_stall=0.04,
    on_event=lambda event, data: print(event, data),
)

result = coord.run(task="Analyse system health...", snapshot_hash=snap_hash, state_bus=bus)
print(result["final_answer"])
print(result["tier2_injected"])   # True if cloud arbiter was invoked
print(result["escalations"])      # list of EscalationRecord dicts
```

**Result contract fields**

```
session_id          final_answer        consensus_score
rounds              snapshot_hash       terminated_reason
tier1_agents        tier2_agent         tier2_injected
disagreement_score  signal_manifold     escalations
all_rounds
```

---

### A3 · IEP Validation + Envelope Queue

`src/effector/queue/iep_queue.py`

Three cooperating classes that route the DASP result into a verified, auditable action record.

#### IEPBuilder

Constructs a typed IEP envelope dict from a raw debate result. Walks the round transcript to aggregate `predicted_delta` values — the modal value per affected key across all agent ESC declarations. Falls back to the `final_answer` string if agents emitted no structured predictions.

```python
envelope = IEPBuilder.from_debate_result(
    debate_result=result,
    state_bus_snapshot_hash=current_hash,
    keys_affected=["system.pressure", "cpu.percent.total"],
    predicted_delta={"system.pressure": 0.42},   # optional — inferred if omitted
    ttl_ms=30_000,
)
```

#### IEPValidator

Runs the five pre-flight checks from IEP §5 in strict fail-fast priority order:

| # | Check | NACK reason |
|---|---|---|
| 1 | Schema completeness | `NACK_SCHEMA_COMPLETENESS` |
| 2 | TTL freshness | `NACK_TTL_FRESHNESS` |
| 3 | Snapshot hash match | `NACK_SNAPSHOT_HASH` |
| 4 | Abort condition evaluation | `NACK_ABORT_CONDITIONS` |
| 5 | Role authorization | `NACK_ROLE_AUTHORIZATION` |

Check 3 is the binding joint between DASP and IEP: if the world state changed between the debate snapshot and envelope emission, the validator rejects the envelope. The agent must re-snapshot and re-deliberate.

The validator also exposes `post_execution_compare()`, which computes a divergence score between the `predicted_delta` and the `actual_delta` after the action completes, and emits `replan_signal` / `escalation_signal` at configurable `epsilon` thresholds.

```python
validator = IEPValidator(state_bus=bus, authorized_roles={"WRITE": ["executor"]})
verdict = validator.validate(envelope)
# verdict.status: "ACK" | "NACK_<check>"
# verdict.checks_passed / verdict.checks_failed
# verdict.failure_reason
```

#### EnvelopeQueue

Thread-safe `queue.Queue` wrapper. Every item — ACK or NACK — is appended as a JSON line to a local file. Complete audit trail is preserved regardless of outcome.

```python
queue = EnvelopeQueue(persist_path="iep_queue.jsonl")
queue.put(envelope, verdict)           # enqueue (blocks if maxsize set)

item = queue.get()                     # dequeue (blocking)
item = queue.get_nowait()              # dequeue (raises queue.Empty)
items = queue.drain()                  # drain all, non-blocking
history = queue.replay_from_disk()     # load full persisted log

queue.stats
# {"total_enqueued": N, "total_acked": N, "total_nacked": N, "current_depth": N}
```

---

## Running

### Continuous loop (requires Ollama)

```bash
# Default: poll every 5s, debate every 30s, run forever
python main_loop.py

# Options
python main_loop.py \
  --poll-interval 3 \
  --debate-interval 60 \
  --max-cycles 10 \
  --queue-file iep_queue.jsonl \
  --no-color
```

The loop prints a structured cycle report for each iteration:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Cycle 1  —  2026-03-29T14:22:01+00:00
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ── A1 — Telemetry Snapshot ──
  CPU   :  18.3%  RAM:  61.4%  Pressure: 0.187
  Window: Visual Studio Code  [Code.exe]
  Net↓  : 12.4 KB/s   Hash: 3fa8c21b9e004d1a...

  ── A2 — Asymmetric DASP Debate ──
    [R1] LOCAL round started
      ✅ [mistral] conf=0.82  System is healthy. No action required.
      ✅ [qwen]    conf=0.76  Pressure within normal bounds.
      Signal [H1]: S_g=1.214  S_i=0.000  S_net=1.214
  ✅ Consensus cleared: score=1.000  H=H1

  ── A3 — IEP Envelope Emission ──
  Envelope ID   : 7f3a1b2c9d4e...
  Checks passed : schema_completeness, ttl_freshness, snapshot_hash, abort_conditions, role_authorization

  📨 QUEUE → ✅ [ACK] envelope=7f3a1b2c... verb=WRITE
```

### Tests (no Ollama required)

```bash
pip install pytest psutil
python -m pytest tests/test_effector_loop.py -v
```

All 74 tests run with mocked Ollama calls. No API keys, no network.

```
74 passed in 1.45s
```

---

## Configuration

### Tier presets

Default tiers are defined in `asymmetric_dasp.py` as `DEFAULT_TIER1` and `DEFAULT_TIER2`. Override any field via `TierConfig`:

```python
TierConfig(
    name="mistral",
    model="mistral",              # Ollama model name
    host="http://127.0.0.1:11434",
    temperature=0.65,
    timeout_s=60.0,
    max_rounds=3,
)
```

To use a remote Ollama instance for Tier-2, set `host` to the remote address. To substitute a different provider, replace `_call_ollama` with any function matching its signature.

### Gate thresholds

| Parameter | Default | Meaning |
|---|---|---|
| `tau_suppression` | `0.5` | `S_i / S_g` ratio that fires the Inhibition gate |
| `theta_consensus` | `0.65` | Minimum `S_net` to clear the Consensus gate |
| `epsilon_stall` | `0.04` | Maximum `\|ΔS_net\|` before Stall gate fires |
| `epsilon_continue` | `0.1` | Divergence threshold for normal IEP continuation |
| `epsilon_replan` | `0.3` | Divergence threshold for replan signal |
| `epsilon_escalate` | `0.6` | Divergence threshold for escalation signal |

Lower `tau_suppression` makes the Inhibition gate more sensitive (fires on weaker vetoes). Lower `theta_consensus` makes consensus easier to reach. Lower `epsilon_stall` makes the Stall gate more aggressive.

---

## Project Structure

```
effector/
├── main_loop.py                         # A1 + A2 + A3 integration loop
├── iep_queue.jsonl                      # runtime — persisted envelope log
│
└── src/effector/
    ├── telemetry/
    │   ├── state_keys.py                # canonical KEYS singleton
    │   └── poller.py                    # TelemetryPoller (A1)
    │
    ├── adapters/
    │   ├── asymmetric_dasp.py           # AsymmetricDASPCoordinator (A2)
    │   └── ollama_adapter.py            # base Ollama adapter (upstream)
    │
    ├── queue/
    │   └── iep_queue.py                 # IEPBuilder, IEPValidator, EnvelopeQueue (A3)
    │
    ├── bus.py                           # StateBus (upstream)
    ├── coordinator.py                   # DASPCoordinator (upstream)
    ├── dasp.py                          # DASP-1.0 Pydantic schemas (upstream)
    ├── iep.py                           # IEP-1.0 Pydantic schemas (upstream)
    ├── session.py                       # EffectorSession API (upstream)
    ├── signal_engine.py                 # Signal superposition engine (upstream)
    └── verifier.py                      # IEPVerifier (upstream)

tests/
└── test_effector_loop.py                # 74 tests covering A1 / A2 / A3
```

---

## Test Coverage

| Class | Tests | What's covered |
|---|---|---|
| `TestTelemetryStateKeys` | 5 | Key uniqueness, namespace groupings |
| `TestPressureModel` | 6 | Parametric range checks, monotonicity, threshold |
| `TestTelemetryPoller` | 9 | StateBus writes, rate bounds, threading, callbacks |
| `TestSignalMath` | 11 | Gate math — generative, inhibitory, neutral ingestion; all three gate conditions; multi-hypothesis routing |
| `TestAsymmetricCoordinatorMocked` | 6 | Consensus without escalation; stall → tier-2; inhibition → tier-2; result contract; abstention on Ollama failure; event ordering |
| `TestIEPBuilder` | 6 | Envelope shape, ESC presence, confidence clamping, ESC aggregation from rounds |
| `TestIEPValidator` | 16 | All 5 NACK paths; all 8 comparison operators; fail-fast ordering; post-execution divergence and replan signal |
| `TestEnvelopeQueue` | 9 | Put/get, stats, drain, JSONL persistence and replay, thread safety (5 concurrent producers × 10 items), maxsize blocking |
| `TestIntegrationPipeline` | 2 | End-to-end A1→A2→A3 with mocked Ollama; snapshot hash chain preservation |

---

## Dependencies

```
psutil>=5.9        # OS telemetry
pydantic>=2.0      # upstream schema validation (dasp.py, iep.py)
requests           # Ollama HTTP calls
```

Ollama must be running locally with the target models pulled:

```bash
ollama pull mistral
ollama pull qwen2.5-coder:32b
ollama pull nemotron              # cloud arbiter — only pulled on escalation
```

---

## Protocol Compliance (this implementation)

| Feature | Spec ref | Status |
|---|---|---|
| Signal superposition (S_g, S_i, S_net) | DASP §6 | ✅ |
| Inhibition gate P1 | DASP §6 | ✅ |
| Stall gate P2 | DASP §6 | ✅ |
| Consensus gate P3 | DASP §6 | ✅ |
| Asymmetric tier escalation | Extension | ✅ |
| IEP pre-flight checks (all 5) | IEP §5 | ✅ |
| Snapshot hash binding (DASP→IEP joint) | Integration §1 | ✅ |
| Post-execution divergence + replan signal | IEP §6 | ✅ |
| ESC aggregation from agent declarations | IEP §3 | ✅ |
| JSONL audit log (append-only) | IEP §1 | ✅ |
| Agent reputation weighting | DASP-A1 | 🟡 stub — R=0.5 for all agents |
| Reflex bypass (RAT issuance) | IEP-A1 | ⬜ not implemented |
| Vectorised state bus (cosine similarity) | IEP-A3 | ⬜ not implemented |

---

## Status

Experimental. The A1/A2/A3 layer is new; the upstream protocol library (bus, coordinator, schemas, verifier) is from the base Effector project. Interfaces may change.