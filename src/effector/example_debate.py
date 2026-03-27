"""
example_debate.py — End-to-end test using real Anthropic API
============================================================
Run: python example_debate.py

Requires ANTHROPIC_API_KEY in environment.
Runs a 2-agent debate with dummy tools, verifies DASP JSON output,
and walks the full IEP lifecycle automatically (no human ACK prompt).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from effector import (
    AgentInfo,
    EffectorSession,
    DebateRules,
    OperationalMode,
    StateBus,
    ToolRegistry,
    make_agent_callable,
)


def main():
    print("=" * 60)
    print("  EFFECTOR — Live Debate Example")
    print("=" * 60)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n⚠️  ANTHROPIC_API_KEY not set — cannot run live example.")
        print("    Set it and re-run, or run the test suite instead:")
        print("    python -m pytest effector/tests/ -v\n")
        return

    # ── World state ─────────────────────────────────────────────────────────
    initial_state = {
        "cache_enabled": False,
        "api_calls_today": 847,
        "avg_response_ms": 340,
        "cache_hit_rate": 0.0,
        "cost_per_1k_calls_usd": 2.40,
    }
    bus = StateBus(initial_state)
    print(f"\n📦 World state: {initial_state}")

    # ── Tools ────────────────────────────────────────────────────────────────
    tools = ToolRegistry()
    tools.register(
        "Read_State",
        "Read the current world state key-value store",
        {"type": "object", "properties": {"key": {"type": "string"}}}
    )
    tools.register(
        "Write_State",
        "Write a value to the world state (requires IEP envelope)",
        {"type": "object", "properties": {"key": {"type": "string"}, "value": {}}}
    )
    tools.register(
        "Estimate_Cache_Savings",
        "Given current API call volume, estimate monthly cost savings if caching is enabled",
        {}
    )

    # ── Agents ───────────────────────────────────────────────────────────────
    agents = [
        AgentInfo(id="optimizer", capabilities=["analysis", "cost_modeling"]),
        AgentInfo(id="skeptic",   capabilities=["risk_analysis", "edge_cases"]),
    ]
    registry = {
        "optimizer": make_agent_callable("optimizer", tools, temperature=0.6),
        "skeptic":   make_agent_callable("skeptic",   tools, temperature=0.8),
    }

    # ── Event hooks ──────────────────────────────────────────────────────────
    def on_round_complete(data):
        round_num = data.get("round")
        responses = data.get("responses", [])
        print(f"\n  ── Round {round_num} complete ({len(responses)} responses) ──")
        for r in responses:
            sig = r.get("signal", {})
            polarity_sym = {1: "✅", 0: "➖", -1: "❌"}.get(sig.get("polarity", 0), "?")
            print(f"    {polarity_sym} [{r['agent_id']}] conf={sig.get('confidence', 0):.2f} | {r['answer'][:80]}")
        manifold = data.get("signal_manifold", {})
        for hid, sig in manifold.items():
            print(f"    Signal [{hid}]: S_g={sig['S_g']:.3f}  S_i={sig['S_i']:.3f}  S_net={sig['S_net']:.3f}")

    def on_envelope_received(envelope):
        print(f"\n📨 IEP Envelope received: {envelope.envelope_id}")
        print(f"   Verb: {envelope.intended_action.verb.value}")
        print(f"   Coalition: {envelope.origin.winning_coalition}")
        print(f"   Consensus score: {envelope.origin.consensus_score:.3f}")

    def on_trigger_fired(data):
        print(f"\n⚡ Trigger fired: {data['trigger']} → {data['action']}")

    # ── Session ───────────────────────────────────────────────────────────────
    rules = DebateRules(
        max_rounds=3,
        theta_consensus=0.6,
        tau_suppression=0.6,
        epsilon_continue=0.1,
        epsilon_replan=0.35,
        epsilon_escalate=0.7,
    )

    session = EffectorSession(state_bus=bus, rules=rules, mode=OperationalMode.deliberative)
    session.on("round_complete", on_round_complete)
    session.on("envelope_received", on_envelope_received)
    session.on("trigger_fired", on_trigger_fired)
    session.on("session_complete", lambda d: print(f"\n🏁 Session complete: {d['terminated_reason']}"))

    print(f"\n🗣️  Goal: Should we enable response caching?")
    print(f"   Agents: {[a.id for a in agents]}")
    print(f"   Max rounds: {rules.max_rounds}")

    result = session.run(
        goal=(
            "Should we enable response caching for this API? "
            "Consider: we make 847 calls/day at $2.40/1k, avg latency 340ms. "
            "Caching would cut latency ~90% for repeated queries. "
            "Downside: stale data risk for time-sensitive queries."
        ),
        agents=agents,
        agent_registry=registry,
        tools=tools,
        snapshot_keys=list(initial_state.keys()),
        require_human_ack=False,  # set True for SUPERVISED mode terminal prompt
    )

    debate = result["debate_result"]
    iep = result["iep_result"]

    print("\n" + "=" * 60)
    print("  RESULT")
    print("=" * 60)
    print(f"  Final answer  : {debate.final_answer}")
    print(f"  Consensus     : {debate.consensus_score:.3f}")
    print(f"  Rounds run    : {debate.rounds}")
    print(f"  Terminated    : {debate.terminated_reason}")
    print(f"  Coalition     : {debate.winning_coalition}")
    print(f"  IEP status    : {iep.status.value}")
    if iep.divergence_score is not None:
        print(f"  Divergence    : {iep.divergence_score:.3f}")
    if iep.replan_signal:
        print("  ⚠️  Replan signal emitted")
    print("=" * 60)

    print(f"\n📊 Final world state: {bus.read()}")
    print(f"📜 Delta log entries: {len(bus.delta_log())}")


if __name__ == "__main__":
    main()
