# Effector Engine

A headless multi-agent reasoning loop over live OS telemetry, implementing the complete **DASP-1.0 + IEP-1.0** protocol stack against real system state — CPU load, memory pressure, active windows, network throughput — using local LLMs as the reasoning layer, with automatic cloud escalation on deadlock.

```
  OS STATE (psutil)
       │
  ─────▼──────────────────────────────────────────────────
  A1 · TELEMETRY POLLER        TelemetryPoller → StateBus
  ─────┬──────────────────────────────────────────────────
       │ snapshot_hash  [+ snapshot_vector when A3 active]
  ─────▼──────────────────────────────────────────────────
  A2 · ASYMMETRIC DASP         mistral + qwen  (local)
       │                       ↓ on stall / inhibition gate
       │                       nemotron        (cloud arbiter)
  ─────┬──────────────────────────────────────────────────
       │ debate_result + consensus_score
  ─────▼──────────────────────────────────────────────────
  A3 · IEP VALIDATION          IEPBuilder → IEPValidator
       │                       → EnvelopeQueue  (JSONL log)
  ─────────────────────────────────────────────────────────
```

---

## Background

Effector is built on two formal protocols described in *Cognitive Agent Architecture: Protocol Suite v1.0*:

**DASP-1.0 — Debate-as-a-Service Protocol** governs deliberation. A group of agents debate a hypothesis through structured rounds, emitting numerical signals — generative strength, inhibitory pressure, polarity — that are composed mathematically rather than resolved by majority vote. Three gates evaluate termination in strict priority order:

| Gate | Condition | Action |
|------|-----------|--------|
| Inhibition (P1) | `S_i(H) ≥ τ · S_g(H)` | Abort / escalate |
| Stall (P2) | `\|ΔS_net\| < ε_stall` | Optimize phase |
| Consensus (P3) | `S_net(H) ≥ θ` | Terminate |

P1 always preempts P3. A strong inhibitory signal cannot be overridden by the size of the generative coalition — mirroring the role of striatal inhibitory interneurons in biological motor control.

**IEP-1.0 — Intention Envelope Protocol** governs execution. Any agent that wants to write to shared state must first emit a typed envelope declaring its intent, a snapshot hash binding it to a specific world state, and a falsifiable forward model (`expected_state_change`). A verifier runs five pre-flight checks before the write is permitted. Divergence between predicted and actual delta drives replanning.

The two protocols connect through the `snapshot_hash` field. Every envelope is cryptographically bound to the exact world state the deliberating agents reasoned from — creating an unbroken, auditable chain from observation to decision to action.

---

## Architecture

### A1 · Substrate Telemetry — `src/effector/telemetry/`

A daemon thread polls OS state via `psutil` every N seconds and writes structured key-value deltas to a `StateBus` dictionary. All canonical key names live in `state_keys.py` as a frozen `KEYS` singleton — agents reference these names in their `expected_state_change` declarations to keep the state ontology stable across sessions.

| Group | Keys |
|-------|------|
| CPU | `cpu.percent.total`, `cpu.percent.per_core`, `cpu.freq.mhz`, `cpu.ctx_switches_sec` |
| Memory | `ram.used_mb`, `ram.percent`, `swap.used_mb`, `swap.percent` |
| Disk I/O | `disk.read_mb_s`, `disk.write_mb_s`, `disk.iops_read`, `disk.iops_write` |
| Network | `net.sent_kb_s`, `net.recv_kb_s`, `net.connections_count` |
| Process | `process.count`, `process.top_cpu.name`, `process.top_mem.mb` |
| Desktop | `desktop.active_window`, `desktop.active_process` |
| Health | `system.pressure`, `system.thermal_alert` |

The composite system pressure score is nonlinear:

```
P = 0.50 · f(cpu) + 0.35 · f(ram) + 0.15 · f(swap)
f(x) = (x / 100)^1.5          # emphasises > 80% utilisation
```

`thermal_alert` fires when `P > 0.85`.

```python
from effector.telemetry.poller import TelemetryPoller
from effector.state_bus.bus import StateBus

bus = StateBus()
poller = TelemetryPoller(bus, interval_s=2.0, on_poll=lambda d: print(d))
poller.start()
# ...
poller.stop()

# or single-shot:
delta = TelemetryPoller(bus).poll_once()
```

### A2 · Asymmetric DASP — `src/effector/adapters/asymmetric_dasp.py`

A two-tier debate coordinator. Tier-1 agents (mistral + qwen, running locally via Ollama) debate for up to `max_rounds`. The Tier-2 arbiter (nemotron) is instantiated **only** when local agents deadlock — keeping cloud API costs and latency entirely off the happy path.

When Tier-2 is invoked, the arbiter receives full session context including the deadlock reason, all prior agent positions, and confidence scores. Its response is injected as a single authoritative round, gates are re-evaluated, and the session terminates.

```python
from effector.adapters.asymmetric_dasp import AsymmetricDASPCoordinator, TierConfig

coord = AsymmetricDASPCoordinator(
    tier1_agents=[
        TierConfig(name="mistral", model="mistral", max_rounds=3),
        TierConfig(name="qwen",    model="qwen2.5-coder:32b", max_rounds=3),
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

**Result contract fields:** `session_id`, `final_answer`, `consensus_score`, `rounds`, `snapshot_hash`, `terminated_reason`, `tier1_agents`, `tier2_agent`, `tier2_injected`, `disagreement_score`, `signal_manifold`, `escalations`, `all_rounds`

### A3 · IEP Validation + Envelope Queue — `src/effector/queue/iep_queue.py`

Three cooperating classes that route the DASP result into a verified, auditable action record.

**IEPBuilder** constructs a typed IEP envelope dict from a raw debate result. Walks the round transcript to aggregate `predicted_delta` values — the modal value per affected key across all agent ESC declarations. Falls back to the `final_answer` string if agents emitted no structured predictions.

**IEPValidator** runs the five pre-flight checks from IEP §5 in strict fail-fast priority order:

| # | Check | NACK reason |
|---|-------|-------------|
| 1 | Schema completeness | `NACK_SCHEMA_COMPLETENESS` |
| 2 | TTL freshness | `NACK_TTL_FRESHNESS` |
| 3 | Snapshot match | `NACK_SNAPSHOT_HASH` |
| 4 | Abort conditions | `NACK_ABORT_CONDITIONS` |
| 5 | Role authorization | `NACK_ROLE_AUTHORIZATION` |

Check 3 is the binding joint between DASP and IEP: if the world state changed between the debate snapshot and envelope emission, the validator rejects the envelope. The agent must re-snapshot and re-deliberate.

**EnvelopeQueue** is a thread-safe `queue.Queue` wrapper. Every item — ACK or NACK — is appended as a JSON line to a local file. The complete audit trail is preserved regardless of outcome.

```python
from effector.queue.iep_queue import IEPBuilder, IEPValidator, EnvelopeQueue

queue = EnvelopeQueue(persist_path="iep_queue.jsonl")

envelope = IEPBuilder.from_debate_result(
    debate_result=result,
    state_bus_snapshot_hash=current_hash,
    keys_affected=["system.pressure", "cpu.percent.total"],
)

validator = IEPValidator(state_bus=bus, authorized_roles={"WRITE": ["executor"]})
verdict   = validator.validate(envelope)

queue.put(envelope, verdict)
# verdict.status         → "ACK" | "NACK_<check>"
# verdict.checks_passed  → ["schema_completeness", "ttl_freshness", ...]
# verdict.failure_reason → str | None
```

---

## IEP-A3: Vectorized State Bus

The base protocol uses SHA-256 hashes for snapshot identity: exact-match or fail. IEP-A3 adds cosine similarity verification, allowing execution to proceed when the world is *close enough* to the state the agents reasoned from — without requiring byte-perfect equality.

This is implemented across four files in five milestones:

### M1 — Deterministic Serialization (`bus.py`)

`StateBus.serialize()` produces a sorted, pipe-delimited flat string of the world state suitable for embedding-model input. Volatile keys — `telemetry.timestamp`, `telemetry.interval_s` — are filtered out by default so that per-poll-cycle churn does not shift the embedding unnecessarily.

```python
bus.serialize()
# → "cpu.freq.mhz: 3600.0 | cpu.percent.total: 18.3 | desktop.active_window: Code.exe | ..."
```

The `StateBus` itself makes zero network calls. This preserves the **dumb substrate constraint** from IEP-A3.1: all vector computation happens in the Orchestration Layer.

### M2 — Coordinator-Side Vectorization (`asymmetric_dasp.py`)

When `vectorized_bus=True`, `AsymmetricDASPCoordinator.run()` calls Ollama's `/api/embed` endpoint immediately after capturing the SHA-256 snapshot hash and *before* dispatching any agent requests. The vector and associated metadata are propagated into the debate result.

If the embedding call fails or returns fewer than 256 dimensions (below the minimum recommended by A3.3), the session continues in hash-only mode. Failure is non-fatal and logged.

```python
coord = AsymmetricDASPCoordinator(
    ...,
    vectorized_bus=True,
    embedding_model="nomic-embed-text",    # any Ollama embed model
    rat_similarity_threshold=0.97,
)

result = coord.run(task, snap_hash, bus)
result["snapshot_vector"]            # list[float] | None
result["vectorized_bus"]             # True only if vector was successfully captured
result["rat_similarity_threshold"]   # 0.97
result["embedding_model"]            # "nomic-embed-text"
```

### M3 — Envelope Builder Augmentation (`iep_queue.py`)

`IEPBuilder.from_debate_result()` extracts the A3 fields from the debate result and writes them into `world_model_snapshot`. The `iep_version` is set to `"1.0-A3"` when a vector is present. If the debate result carries no vector (hash-only or upstream failure), the envelope is produced as a standard `"1.0"` envelope — no new fields, no behaviour change downstream.

### M4 — Cosine Snapshot Verification (`iep_queue.py`)

`IEPValidator._check_snapshot()` checks `vectorized_bus` and `snapshot_vector` in the envelope. When both are present, the verifier:

1. Reads the current bus state.
2. Serializes it via `StateBus.serialize()`.
3. Fetches the current embedding from Ollama `/api/embed`.
4. Computes cosine similarity against the stored vector.
5. ACKs if `similarity >= rat_similarity_threshold`.

If the embedding fetch fails for any reason, it falls through to SHA-256 hash comparison. The validator never silently approves an unverifiable state.

```
similarity(V_current, V_snapshot) = dot(V, Vs) / (‖V‖ · ‖Vs‖)
```

Recommended thresholds per A3.2:

| Mode | Threshold |
|------|-----------|
| High-confidence | 0.97 |
| Standard | 0.92 |
| Permissive | 0.85 |

### M5 — Semantic Drift Lock (`iep_queue.py`)

Before any cosine computation, the validator runs an exact-value check on a configurable `critical_keys` tuple (defaulting to `desktop.active_window` and `desktop.active_process`). These keys change discretely — a window switch is semantically significant regardless of how small a cosine shift it produces.

If any critical key has mutated between the snapshot and validation time, the validator immediately returns `NACK_SNAPSHOT_HASH` without attempting similarity math, forcing a full replan. The lock can be disabled by passing `critical_keys=()` or tuned to any set of keys.

```python
validator = IEPValidator(
    state_bus=bus,
    critical_keys=("desktop.active_window", "desktop.active_process"),
    ollama_host="http://127.0.0.1:11434",
)
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
  Checks passed : schema_completeness, ttl_freshness, snapshot_hash,
                  abort_conditions, role_authorization

  📨 QUEUE → ✅ [ACK] envelope=7f3a1b2c... verb=WRITE
```

### Tests (no Ollama required)

```bash
pip install pytest psutil
python -m pytest tests/ -v
```

All tests mock Ollama calls. No API keys, no network.

---

## Configuration

### Tier presets

```python
TierConfig(
    name="mistral",
    model="mistral",               # Ollama model name
    host="http://127.0.0.1:11434",
    temperature=0.65,
    timeout_s=60.0,
    max_rounds=3,
)
```

To use a remote Ollama instance for Tier-2, set `host` to the remote address.

### Gate thresholds

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `tau_suppression` | 0.5 | S_i / S_g ratio that fires the Inhibition gate |
| `theta_consensus` | 0.65 | Minimum S_net to clear the Consensus gate |
| `epsilon_stall` | 0.04 | Maximum \|ΔS_net\| before Stall gate fires |
| `epsilon_continue` | 0.1 | Divergence threshold for normal IEP continuation |
| `epsilon_replan` | 0.3 | Divergence threshold for replan signal |
| `epsilon_escalate` | 0.6 | Divergence threshold for escalation signal |

Lower `tau_suppression` makes the Inhibition gate more sensitive. Lower `theta_consensus` makes consensus easier to reach. Lower `epsilon_stall` makes the Stall gate more aggressive.

---

## Project Structure

```
effector/
├── main_loop.py                          # A1 + A2 + A3 integration loop
├── iep_queue.jsonl                       # runtime — persisted envelope log
│
└── src/effector/
    ├── telemetry/
    │   ├── state_keys.py                 # canonical KEYS singleton
    │   └── poller.py                     # TelemetryPoller (A1)
    │
    ├── adapters/
    │   ├── asymmetric_dasp.py            # AsymmetricDASPCoordinator (A2)
    │   └── ollama_adapter.py             # base Ollama adapter (upstream)
    │
    ├── queue/
    │   └── iep_queue.py                  # IEPBuilder, IEPValidator, EnvelopeQueue (A3)
    │
    ├── bus.py                            # StateBus (upstream)
    ├── coordinator.py                    # DASPCoordinator (upstream)
    ├── dasp.py                           # DASP-1.0 Pydantic schemas (upstream)
    ├── iep.py                            # IEP-1.0 Pydantic schemas (upstream)
    ├── session.py                        # EffectorSession API (upstream)
    ├── signal_engine.py                  # Signal superposition engine (upstream)
    └── verifier.py                       # IEPVerifier (upstream)

tests/
├── test_effector_loop.py                 # 74 tests — A1 / A2 / A3 base coverage
└── test_iep_a3.py                        # 37 tests — IEP-A3 vectorized bus
```

---

## Test Coverage

### Base suite (`test_effector_loop.py`) — 74 tests

| Class | Tests | What's covered |
|-------|-------|----------------|
| `TestTelemetryStateKeys` | 5 | Key uniqueness, namespace groupings |
| `TestPressureModel` | 6 | Parametric range checks, monotonicity, threshold |
| `TestTelemetryPoller` | 9 | StateBus writes, rate bounds, threading, callbacks |
| `TestSignalMath` | 11 | Gate math — all three gate conditions, multi-hypothesis routing |
| `TestAsymmetricCoordinatorMocked` | 6 | Consensus, stall→tier-2, inhibition→tier-2, abstention, event ordering |
| `TestIEPBuilder` | 6 | Envelope shape, ESC presence, confidence clamping, ESC aggregation |
| `TestIEPValidator` | 16 | All 5 NACK paths, all 8 comparison operators, fail-fast ordering, post-execution divergence |
| `TestEnvelopeQueue` | 9 | Put/get, stats, drain, JSONL persistence, thread safety (5×10 concurrent) |
| `TestIntegrationPipeline` | 2 | End-to-end A1→A2→A3 with mocked Ollama; snapshot hash chain |

### IEP-A3 suite (`test_iep_a3.py`) — 37 tests

| Class | Tests | What's covered |
|-------|-------|----------------|
| `TestStateBusSerialize` | 9 | Volatile key exclusion, determinism, sorting, pipe format, key filtering |
| `TestCoordinatorVectorization` | 4 | Vector fetched/skipped, None degradation, threshold/model propagation |
| `TestIEPBuilderVectorRouting` | 6 | Vector in envelope, hash-only path, version string, null-vector guard |
| `TestIEPValidatorCosine` | 6 | ACK above threshold, NACK below, hash fallback paths, non-vectorized bypass |
| `TestCriticalKeysDriftLock` | 3 | Stable keys pass, disabled lock, custom critical-key set |
| `TestCosineSimilarityMath` | 7 | Identity, orthogonal, opposite, zero-vector, high-dim, symmetry, scale invariance |
| `TestEndToEndIEPA3` | 2 | Full A1→A2→A3 ACK on stable state; NACK on drifted state |

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
ollama pull nomic-embed-text         # IEP-A3 vectorization
ollama pull nemotron                 # cloud arbiter — only pulled on escalation
```

---

## Protocol Compliance

| Feature | Spec ref | Status |
|---------|----------|--------|
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
| Deterministic state serialization | IEP-A3.1 | ✅ |
| Orchestrator-side vectorization | IEP-A3.1 | ✅ |
| Cosine similarity snapshot verification | IEP-A3 M4 | ✅ |
| Critical-key semantic drift lock | IEP-A3 M5 | ✅ |
| Hash fallback on embedding failure | IEP-A3.3 | ✅ |
| Agent reputation weighting | DASP-A1 | 🟡 stub — R=0.5 for all agents |
| Reflex bypass (RAT issuance) | IEP-A1 | ⬜ not implemented |

---

## Status

Experimental. The A1/A2/A3 layer is production-shaped but interfaces may change. The upstream protocol library (bus, coordinator, schemas, verifier) is from the base Effector project. IEP-A3 is complete at the orchestrator level; RAT-based reflex bypass (IEP-A1) is the natural next extension.