Effector
Multi-agent AI orchestration governed by structured deliberation and verified execution.
Effector is a Python library and CLI that coordinates groups of local LLM agents through two complementary protocols: the Debate-as-a-Service Protocol (DASP) for structured deliberation, and the Intention Envelope Protocol (IEP) for verified, auditable state writes. Agents cannot take naked actions. Every consequential change to shared state is preceded by deliberation, bound to a world-state snapshot, and verified before execution.
A local Reflex Middleware layer provides a sub-2ms fast path for actions that have already been deliberated and pre-authorized, bypassing the LLM debate loop entirely while preserving the full safety audit trail.

Table of Contents
Architecture Overview
Requirements
Installation
Quick Start
CLI Reference
Configuration
Protocols
DASP — Debate-as-a-Service Protocol
IEP — Intention Envelope Protocol
Reflex Middleware
Telemetry
Testing
Project Structure
Contributing

Architecture Overview
  ┌─────────────────────────────────────────────────────────────┐
  │  ORCHESTRATION LAYER                                        │
  │  Goal decomposition · Agent selection · Task routing        │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
  ┌──────────────────────────────▼──────────────────────────────┐
  │  REFLEX LAYER  (sub-2ms fast path)                          │
  │  IntentRouter · LocalRATStore · ReflexEngine                │
  │  Pre-authorized actions bypass deliberation                 │
  └──────────┬───────────────────────────────────────┬──────────┘
       RAT hit│                                  RAT miss│
             ↓                                          ↓
  ┌──────────────────────┐         ┌─────────────────────────────┐
  │  EXECUTION LAYER     │         │  DELIBERATION LAYER         │
  │  IEP-1.1             │         │  DASP-1.1                   │
  │  Envelope emission   │◄────────│  Debate rounds              │
  │  Pre-flight verify   │  emit   │  Signal superposition       │
  │  State write         │         │  Inhibition / stall gates   │
  └──────────┬───────────┘         │  Consensus detection        │
             │                     │  RAT issuance on complete   │
             ▼                     └─────────────────────────────┘
  ┌──────────────────────┐
  │  STATE BUS           │
  │  Immutable world-    │
  │  state vector        │
  │  Append-only delta   │
  │  log                 │
  └──────────────────────┘
The key invariant: no agent ever writes to shared state without a valid, verified Intention Envelope. The snapshot hash binding the envelope to a specific world-state slice makes every action traceable from root goal to leaf execution.

Requirements
Requirement
Version
Python
3.11+
Ollama
Latest
psutil
5.9+
pydantic
2.0+

Hardware (local inference):
Model size
Status
7B
No pressure
14B
Comfortable
32B
Fits; set --timeout 120
70B
VRAM overflow; GGUF CPU offload only

Tested on: Windows 11 Pro (build 26200), Python 3.12, RTX 4080 SUPER 16 GB, 128 GB RAM.

Installation
# Clone the repository
git clone https://github.com/your-org/effector.git
cd effector

# Install in editable mode
pip install -e .

# Install optional config-write support
pip install -e ".[config-write]"

# Install development dependencies
pip install -e ".[dev]"
Pull the models you intend to use:
effector models pull mistral:7b
effector models pull qwen2.5:14b
effector models pull qwen2.5-coder:32b # recommended Characterizer model
effector models pull nomic-embed-text:latest   # required for vectorized snapshot mode
Verify your environment:
effector doctor

Quick Start
Run a single debate session:
effector run --task "Is the system under abnormal CPU pressure?"
Specify models explicitly:
effector run \
  --task "What is causing the high memory usage?" \
  --agent1 qwen2.5:32b \
  --agent2 deepseek-r1:14b \
  --arbiter nemotron \
  --max-rounds 5
Enable vectorized snapshot mode (cosine similarity instead of exact SHA-256 hash matching):
effector run \
  --task "Should we throttle background processes?" \
  --vectorized-bus \
  --embedding-model nomic-embed-text \
  --rat-threshold 0.97
Save full debate result to disk:
effector run \
  --task "Diagnose the disk I/O spike" \
  --output results/session_001.json \
  --format json

CLI Reference
effector run
Run a DASP multi-agent debate session.
Flag
Default
Description
--task, -t
(required)
The directive sent to all agents
--agent1
mistral:7b
Tier-1 agent 1 model
--agent2
qwen2.5:14b
Tier-1 agent 2 model
--arbiter
nemotron
Tier-2 arbiter; invoked on stall or inhibition
--tau
0.5
Inhibition gate threshold τ ∈ [0, 1]
--theta
0.7
Consensus gate threshold θ ∈ [0, 1]
--epsilon, -e
0.05
Stall gate threshold ε ≥ 0
--max-rounds, -m
3
Maximum debate rounds
--timeout
60
Per-agent network call timeout (seconds)
--vectorized-bus
off
Enable IEP-A3 cosine snapshot mode
--embedding-model
nomic-embed-text
Ollama model for snapshot embeddings
--rat-threshold
0.97
Cosine similarity threshold for RAT validation
--telemetry/--no-telemetry
on
Live OS telemetry as agent context
--format, -f
pretty
Output format: pretty | json | minimal | silent
--output, -o
—
Save full result JSON to this path
--persist
—
Append IEP envelopes to a JSONL audit log
--dry-run
off
Validate config and display parameters; do not invoke agents

Exit codes:
Code
Meaning
0
Success (consensus reached or max rounds)
1
Fatal error (Ollama unreachable, bad arguments)
2
Inhibition gate fired — hard veto, no consensus possible
3
Low consensus score (< 0.4)

effector config
effector config show               # Print current config
effector config show debate        # Print a single section
effector config set debate.tau 0.4 # Set a value persistently
effector config diff               # Show deviations from defaults
effector config reset              # Restore factory defaults

effector config profile save my-profile   # Save current config as a named profile
effector config profile load my-profile   # Load a saved profile
effector config profile list              # List saved profiles
Config is stored at:
Windows: %LOCALAPPDATA%\effector\config.toml
macOS: ~/Library/Application Support/effector/config.toml
Linux: ~/.config/effector/config.toml
effector models
effector models list              # List installed Ollama models
effector models known             # List catalogue of known compatible models
effector models pull <name>       # Pull a model via Ollama
effector models remove <name>     # Remove a model
effector models search <query>    # Search ollama.com
effector models info <name>       # Show model metadata
effector session
effector session replay <path>    # Replay a persisted envelope queue (JSONL)
effector session inspect <path>   # Inspect a saved debate result (JSON)

Configuration
All effector run flags can be persisted to avoid repeating them on every invocation. The config file is TOML and is edited via effector config set or directly.
[debate]
tau        = 0.5
theta      = 0.7
epsilon    = 0.05
max_rounds = 3
verbosity  = 200

[agents]
agent1   = "qwen2.5:32b"
agent2   = "deepseek-r1:14b"
arbiter  = "nemotron"
timeout_s = 90.0

[iep]
vectorized_bus    = true
embedding_model   = "nomic-embed-text"
rat_threshold     = 0.97
epsilon_continue  = 0.1
epsilon_replan    = 0.3
epsilon_escalate  = 0.6

[telemetry]
enabled       = true
poll_interval_s = 2.0

[ollama]
host = "http://127.0.0.1:11434"

Protocols
DASP — Debate-as-a-Service Protocol
DASP (v1.1) governs the deliberation layer. It routes a task through structured rounds of independent agent reasoning, using mathematical signal superposition to detect consensus, deadlock, and pathological agreement — without parsing agent explanations for routing decisions.
Session lifecycle:
SNAPSHOT → INITIAL_ROUND → DEBATE_ROUNDS → CONSENSUS_CHECK → RESULT
Signal model:
Each agent emits a structured signal alongside its textual explanation:
S_g(H) = Σ R_eff(a) · confidence · generative_strength   (polarity ≥ 0)
S_i(H) = Σ R_eff(a) · confidence · inhibitory_pressure   (polarity ≤ 0)
S_net  = S_g - S_i
Gates (evaluated in priority order):
Gate
Condition
Action
Inhibition (P1)
S_i ≥ τ · S_g
Phase-cancel hypothesis; route to OPTIMIZE or ESCALATE
Stall (P2)
|ΔS_net| < ε
OPTIMIZE phase; inject devil's advocate if configured
Consensus (P3)
S_net ≥ θ
TERMINATE; emit Intention Envelope

Asymmetric tiers:
Effector uses a two-tier debate model. Tier-1 agents (fast, local) run all standard rounds. If a stall or inhibition gate fires, a Tier-2 arbiter (slower, higher-capacity) is invoked to break the deadlock. This keeps the common case fast while preserving high-quality resolution for contested decisions.
Agent Reputation System:
Agent signal contributions are weighted by a per-agent Reputation Score R(a) ∈ [0, 1], derived from the historical accuracy of each agent's forward model predictions. New agents start at R = 0.5. A floor of R_floor = 0.15 prevents any agent from being permanently silenced.
Serialization boundary (Two-Phase Adapter):
All agents use the two-phase adapter pattern. The reasoning phase (free-form LLM output) is separated from the characterization phase (structured signal extraction). The Characterizer maps the reasoning into a 4-Dimensional Signal Manifold:
1. Polarity: Support (1), Oppose (-1), or Genuine Uncertainty (0) — enabling Epistemic Humility.
2. Confidence: Degree of certainty [0.0, 1.0].
3. Generative Strength: Independent weight for action [0.0, 1.0].
4. Inhibitory Pressure: Independent weight against action [0.0, 1.0].

This explicit decoupling prevents sycophantic convergence and enables advanced behaviors like Compassion Fatigue (yielding to user agency by decaying generative strength without flipping polarity). The signal fields are never parsed from raw LLM text.


IEP — Intention Envelope Protocol
IEP (v1.1) governs the execution layer. Every state write is preceded by a typed Intention Envelope that declares the agent's intent, binds the action to the world-state snapshot the agent reasoned from, and carries a forward model prediction used for post-execution divergence scoring.
Verification order (mirrors IEPVerifier.verify()):
Schema completeness
TTL freshness
Snapshot hash match (or cosine similarity if IEP-A3 is active)
Abort condition evaluation
Role authorization
Operational modes:
Mode
Description
DELIBERATIVE
All actions require a completed DASP session. Default.
MIXED
Routine actions use pre-issued Reflex Authorization Tokens; novel actions use full DASP.
REFLEX
All actions require a valid RAT. No DASP during execution.
SUPERVISED
DASP runs normally, but a human operator must ACK before execution.

Post-execution comparison:
After each acknowledged write, the State Bus compares the actual delta against the expected_state_change prediction. Divergence above ε_replan triggers a replan signal. Divergence above ε_escalate halts downstream actions and escalates to the Coordinator.
Vectorized State Bus (IEP-A3):
When enabled, world-state snapshots are represented as dense float embeddings via nomic-embed-text. Envelope verification uses cosine similarity rather than exact SHA-256 matching, allowing pre-authorized actions to execute when the world is close enough to its deliberated state. The embedding model and similarity threshold are stored in every RAT and verified at execution time.

Reflex Middleware
The Reflex Middleware provides a local fast path for actions that have been previously deliberated and pre-authorized by a completed DASP session. It is the system's "spinal cord" — deterministic, sub-2ms, and zero LLM calls on the critical path.
Components:
Module
Responsibility
intent_router.py
Deterministic regex/string matching of tasks to {verb, target} pairs. Returns None on no-match; caller falls back to DASP.
rat_store.py
SQLite-backed Reflex Authorization Token store. WAL journal, thread-safe, atomic execution decrements via RETURNING.
reflex_engine.py
Core check sequence. Receives pre-computed state vector; performs no I/O.
main_loop_reflex.py
ReflexOrchestrator: drop-in main loop wrapper. Issues RATs after DASP consensus; routes subsequent identical triggers to the fast path.

Reflex check order (mirrors IEP §7 Reflex Envelope Verification):
RAT lookup by verb/target (prefix matching supported)
RAT TTL expiry
Verb/target action match
Execution count (preliminary; atomic decrement after M4/M5)
M5 — Critical-key drift guard: desktop.active_window and desktop.active_process are read twice in rapid succession; mutation between reads triggers replan
M4 — Cosine similarity ≥ rat_similarity_threshold; falls back to SHA-256 hash if no vector is available
Atomic SQLite decrement (UPDATE … RETURNING)
Execute + fire async post-execution callback (divergence scoring, reputation update)
Wiring into the main loop:
from effector.rat_store import LocalRATStore
from effector.main_loop_reflex import ReflexOrchestrator

store = LocalRATStore()   # persists across reboots; M4/M5 catch stale state

orchestrator = ReflexOrchestrator(
    state_bus=bus,
    rat_store=store,
    dasp_run_fn=coordinator.run,
    embedding_model="nomic-embed-text",
    ollama_host="http://127.0.0.1:11434",
)

# Per-task call — returns unified result dict with '_path': 'reflex' | 'dasp'
result = orchestrator.handle(task, snapshot_hash)
RAT issuance happens automatically after any DASP session that reaches sufficient consensus (consensus_score ≥ rat_min_confidence). The next occurrence of the same intent takes the reflex path; no configuration required.
Latency profile:
Step
Typical latency
IntentRouter.route()
< 0.1 ms
fetch_embedding() (Ollama)
20–50 ms
ReflexEngine.evaluate_reflex()
< 2 ms
Full reflex path (no match in RAT store)
< 0.2 ms (BYPASSED)
Full DASP debate (3 rounds, 2 × 14B agents)
30–120 s


MCP Layer — Snug Registry
The MCP layer exposes the Effector Engine's concept registry as a tool server, enabling agents to query, synthesize, and mutate the Affinity Matrix — a living database of warmth, comfort, and ambient contentment that accumulates knowledge across sessions.
Snug Registry Server (mcp/registry_mcp.py)
The registry models the world in terms of three variables — Thermal Valence (θ), Kinetic Fidget (φ), and Social Mass (μ) — across 48 named entities (from Fc: Felis catus to Hg: Big Hug) and 55 category-pair bond strengths. Bond strengths drift over time as the cultivation loop commits observed outcomes.
Tools:
Tool
Description
lookup_entity(sym)
Full entity data by two-character symbol — θ, φ, μ, SnugProphecy, stability rating
search_entities(...)
Filter by name, category, value thresholds; sorted by SnugProphecy score
get_bond(cat_a, cat_b)
Current bond affinity and interpretation between two categories
synthesize(symbols, commit_ack)
Multi-entity synthesis forecast; commit_ack=True executes, mutates bonds, and persists
hydrate_template(src, tgt, type)
Generate liturgical petition text for DASP explanation fields
find_resonant(sym, top_n)
Lateral resonance — similar-profile substitutes using isomorphism scoring
get_agent_messages(...)
Current WEAVE/SPARK/ARBITER deliberation queue
get_ontology()
Full ontology reference — variables, categories, formulas, state file path

State persistence:
All mutations flow through synthesize(commit_ack=True), which draws actual_theta = predicted ± random[-0.06, +0.16], adjusts all category-pair bond strengths by ±0.02, and writes the result to registry_state.json atomically. Bond strengths are clamped to [-1, 1] and accumulate truth over the lifetime of the registry.
Registration in mcp_servers config:
{
  "mcpServers": {
    "snug-registry": {
      "command": "python",
      "args": ["mcp/registry_mcp.py"]
    }
  }
}
Dependency: pip install mcp

Cultivation Loop (mcp/cultivation_loop.py)
A background agent loop that autonomously explores the entity space and proposes new combinations. It submits proposals to the local Reflex Orchestrator. If a combination is novel, it runs a full three-agent DASP session. If it has been deliberated previously, it hits the Reflex Cache (<2ms). Approved syntheses pass through the IEP Validator (the Two-Key Execution Boundary) before being committed to the Affinity Matrix — growing the registry's learned bond strengths overnight.

Agents:
Agent
Role
WEAVE-3
Proposal agent — argues for the synthesis using θ/φ/μ values
SPARK-2
Evaluation agent — surfaces risks and refinements
ARBITER
Consensus authority — issues ACK or NACK as structured JSON

ACK criteria (all must hold):
SnugProphecy ≥ 14
Deficit count ≤ 1 (compound deficits without a high-θ anchor are forbidden)
Stability is not Unstable (unless SPARK-2 explicitly endorses with cause)
# Night-only (22:00–06:00 local), 6 sessions/run, 1h between runs:
python mcp/cultivation_loop.py

# Background daemon:
nohup python mcp/cultivation_loop.py >> cultivation.out 2>&1 &

# All hours, faster cadence:
python mcp/cultivation_loop.py --continuous --sessions 4 --interval 900

# Dry-run — forecasts only, no API calls, no commits:
python mcp/cultivation_loop.py --dry-run --sessions 3

# Show accumulated cultivation statistics:
python mcp/cultivation_loop.py --stats
Session log: mcp/cultivation_log.jsonl — append-only record of every session, including WEAVE argument, SPARK evaluation, ARBITER verdict, actual_theta, and all bond mutations applied. Main Loop & Live Dashboard (main_loop.py) The primary entry point for the headless reasoning engine. It utilizes `rich` to render a live, terminal-based UX Dashboard featuring: * Telemetry Panel: Live OS metrics (CPU, RAM, System Pressure, Active Window). * Agent Signal Manifold: Real-time S_g, S_i, and S_net tracking for active DASP debates. * IEP Queue: A live tail of the envelope processing log (ACK/NACK verdicts and failure reasons). Telemetry
Dependency: pip install openai (routes to local Ollama via OpenAI-compatible API)

Telemetry
When --telemetry is enabled (the default), a background TelemetryPoller writes live OS metrics to the State Bus every --poll-interval seconds. Agents receive this data as world-state context on every debate round.
Keys written to the State Bus:
Key
Description
cpu.percent.total
Total CPU utilization
cpu.percent.per_core
Per-core utilization list
ram.percent
RAM utilization
swap.percent
Swap utilization
disk.read_mb_s / disk.write_mb_s
Disk throughput
net.sent_kb_s / net.recv_kb_s
Network throughput
desktop.active_window
Foreground window title
desktop.active_process
Foreground process name
system.pressure
Composite pressure score ∈ [0, 1]
system.thermal_alert
true when pressure > 0.85

The desktop.active_window and desktop.active_process keys are treated as critical keys by the Reflex Middleware — a change to either triggers an M5 replan regardless of cosine similarity.

Testing
# Run all tests
pytest

# Run the reflex middleware suite specifically
pytest tests/test_reflex_middleware.py -v

# Run with coverage
pytest --cov=effector --cov-report=term-missing
Test suite breakdown:
Suite
Tests
Coverage
test_reflex_middleware.py
31
LocalRATStore, IntentRouter, ReflexEngine, ReflexOrchestrator, RAT issuance
test_signal_engine.py
—
Signal superposition, gate evaluation, copy/swap detection
test_multi_agent_debate.py
—
Full DASP session lifecycle
test_iep_a3.py
—
Vectorized State Bus, cosine verification
test_escalation_and_drift.py
—
Tier-2 escalation, snapshot drift handling
test_effector_loop.py
—
End-to-end session with IEP verification

All tests mock Ollama API calls. No live model inference is required to run the test suite.

Project Structure
effector/
├── src/effector/
│   ├── adapters/
│   │   ├── asymmetric_dasp.py      # Two-tier coordinator (local + cloud arbiter)
│   │   ├── ollama_adapter.py       # OllamaAgent wrapper
│   │   └── two_phase_adapter.py    # Reasoning + characterizer pipeline
│   ├── cli/
│   │   ├── main.py                 # Typer app entry point
│   │   ├── run_cmd.py              # `effector run` implementation
│   │   ├── config_cmd.py           # `effector config` subcommands
│   │   ├── models_cmd.py           # `effector models` subcommands
│   │   ├── display.py              # Rich terminal output helpers
│   │   └── settings.py             # TOML config loading/saving
│   ├── queue/
│   │   └── iep_queue.py            # IEPBuilder, IEPValidator, EnvelopeQueue
│   ├── schemas/
│   │   ├── dasp.py                 # Pydantic models for DASP messages
│   │   └── iep.py                  # Pydantic models for IEP envelopes
│   ├── telemetry/
│   │   ├── poller.py               # Background OS telemetry poller
│   │   └── state_keys.py           # Typed key constants
│   ├── bus.py                      # StateBus — immutable world-state vector
│   ├── coordinator.py              # DASPCoordinator
│   ├── session.py                  # EffectorSession, IEPExecutor
│   ├── signal_engine.py            # SignalEngine — manifold, gates, triggers
│   ├── verifier.py                 # IEPVerifier — envelope verification
│   ├── intent_router.py            # Deterministic intent routing (reflex)
│   ├── rat_store.py                # SQLite RAT store (reflex)
│   ├── reflex_engine.py            # ReflexEngine — fast path (reflex)
│   └── main_loop_reflex.py         # ReflexOrchestrator — main loop wiring
├── mcp/
│   ├── registry_mcp.py             # Snug Registry MCP server (8 tools)
│   ├── cultivation_loop.py         # Nightly three-agent cultivation loop
│   └── registry_state.json         # Live Affinity Matrix (persisted, mutable)
├── tests/
│   ├── test_reflex_middleware.py
│   ├── test_signal_engine.py
│   ├── test_multi_agent_debate.py
│   ├── test_iep_a3.py
│   ├── test_escalation_and_drift.py
│   └── test_effector_loop.py
├── docs/
│   ├── worldbuilding/              # Narrative canon, agent theology, lore
│   ├── style/                      # Visual style bible, look-dev, Glimmer design
│   └── protocol/                   # DASP-1.1 + IEP-1.1 specification
├── examples/                       # Usage examples and integration demos
├── aggregate.py                    # Profile-aware context aggregator (see below)
├── context.toml                    # Aggregator manifest — zones, profiles, handling
├── main_loop.py                    # Standalone orchestration entry point
└── pyproject.toml

Contributing
Development setup:
pip install -e ".[dev]"
Before submitting a PR:
# Run the full test suite
pytest

# Type-check
mypy src/effector

# Lint
ruff check src/effector
Protocol changes:
Changes to DASP or IEP behaviour require updating both the relevant schema in src/effector/schemas/ and the protocol specification document. The protocol spec is the source of truth; the implementation must conform to it, not the reverse.
The Reflex Middleware check order in reflex_engine.py must always mirror the IEP §7 Reflex Envelope Verification sequence. Any divergence is a correctness bug.
Adding intent routes:
New deterministic routing patterns belong in intent_router.py's _build_table() function. Patterns are evaluated in order; first match wins. Do not use LLM calls inside the router.
Future: distributed deployment
The current LocalRATStore (SQLite) is intentionally architected to translate cleanly to a Redis-backed distributed store. The three core methods — get_candidate_rats, decrement_and_fetch, and invalidate_rat — have 1:1 equivalents as Redis Lua scripts when deploying Effector as a multi-node swarm.

Session Workflow — Context Aggregator
The project includes a profile-aware context aggregator (aggregate.py) that prepares the right files for each type of Claude session. It reads context.toml at the project root, which declares every includable zone, its profile membership, and how to handle large data files.
The problem it solves: The project has important work spread across src/, mcp/, docs/, tests/, and the project root. A fixed-scope aggregator leaves out anything outside its scope, causing doc updates to miss the MCP layer, code sessions to miss the style bible, and so on.
Profiles:
# Core engine logic, protocols, tests
python aggregate.py --profile code

# MCP tools + registry + src schemas
python aggregate.py --profile mcp

# README + worldbuilding + style bible + protocol spec
python aggregate.py --profile docs

# Everything — for architecture reviews
python aggregate.py --profile full

# See what a profile would include without producing output
python aggregate.py --profile mcp --dry-run

# Override token budget
python aggregate.py --profile full --budget 300000

# Force-include a data file verbatim (e.g. full JSON dump)
python aggregate.py --profile mcp --include-file mcp/registry_state.json
Smart data file handling:
registry_state.json is summarized rather than dumped verbatim — ~120 tokens instead of ~12,000. The summary includes entity count, category breakdown, deficit and radiant entity lists, top-5 entities by θ, bond strength range, and strongest/weakest bonds. All actionable context, no noise. Pass --include-file mcp/registry_state.json when you need the full JSON.
The manifest (context.toml):
Edit context.toml to add zones as the project grows. Each zone declares a path, which profiles include it, how to handle it (verbatim | summarize | header_only | skip), a token cap, and a priority. The aggregator includes zones in priority order and stops at the token budget.
Where to put things so the aggregator finds them:
Content type
Canonical location
Profile
Core Python source
src/effector/
code
MCP tools and servers
mcp/
mcp
Live registry / state data
mcp/registry_state.json
mcp (summarized)
Protocol specification
docs/protocol/
code, docs
Worldbuilding / narrative canon
docs/worldbuilding/
docs
Visual style bible / look-dev
docs/style/
docs
Test suite
tests/
code
Entry points, config
project root
all profiles


License
MIT

