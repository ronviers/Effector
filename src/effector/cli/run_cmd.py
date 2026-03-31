"""
run_cmd.py — Debate session orchestration for the CLI.

Responsibilities
----------------
- Build per-round temperature schedules from CLI flags
- Construct TierConfig objects
- Wire StateBus + TelemetryPoller
- Create and call AsymmetricDASPCoordinator
- Route coordinator events to the Rich display layer
- Optionally run IEPBuilder → IEPValidator pipeline
- Persist / print the result

This module is deliberately separate from main.py so that it is
independently testable and importable by other entry points (e.g. a
future REST gateway).

Temperature Scheduling
----------------------
Three flags control per-round temperature for each agent:

  --agentN-temp T           Base temperature (all rounds, unless overridden)
  --agentN-temp-range lo,hi Linearly sweep from hi → lo across all rounds
  --agentN-temp-pivot R     At round R, abruptly switch temperatures

Interaction matrix:

  range only         → linear sweep hi→lo across all rounds
  pivot only         → base temp for rounds < R; base*0.6 for rounds >= R
  range + pivot      → hi for rounds < R; lo for rounds >= R (step function)
  neither            → constant base temp

The schedule is pre-computed into a list[float] of length max_rounds and
injected into TierConfig.temperature_schedule (see patch note below).

NOTE: TierConfig.temperature_schedule is an extension field.  Apply the
two-line patch to src/effector/adapters/asymmetric_dasp.py:

    # In the TierConfig dataclass — add:
    temperature_schedule: list[float] | None = None

    # Add a helper method:
    def get_temperature(self, round_num: int) -> float:
        if self.temperature_schedule:
            idx = min(round_num - 1, len(self.temperature_schedule) - 1)
            return self.temperature_schedule[idx]
        return self.temperature

    # In AsymmetricDASPCoordinator._tier1_loop() — replace:
    #   temperature=tier.temperature,
    # with:
    #   temperature=tier.get_temperature(local_round),
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

from effector.cli.display import (
    console,
    print_error,
    print_result,
    print_result_json,
    print_result_minimal,
    print_round_header,
    print_round_responses,
    print_session_header,
    print_signal_manifold,
    print_trigger,
    print_ok,
    print_warn,
)
from effector.cli.settings import get_settings


# ─────────────────────────────────────────────────────────────────────────────
# Temperature scheduling
# ─────────────────────────────────────────────────────────────────────────────

def build_temp_schedule(
    base: float,
    pivot: Optional[int],
    temp_range_str: Optional[str],
    max_rounds: int,
) -> list[float]:
    """
    Pre-compute a per-round temperature list of length *max_rounds*.

    Args:
        base:           Flat temperature when no schedule is active.
        pivot:          Round number (1-indexed) at which temperature changes.
        temp_range_str: "lo,hi" string.  If given, drives the sweep/step.
        max_rounds:     Total rounds in the debate.

    Returns:
        List of floats, one per round (index 0 = round 1).
    """
    if temp_range_str:
        parts = temp_range_str.split(",")
        if len(parts) != 2:
            raise ValueError(
                f"--temp-range must be 'lo,hi', e.g. '0.3,0.9'.  Got: {temp_range_str!r}"
            )
        lo, hi = float(parts[0].strip()), float(parts[1].strip())

        if pivot is not None:
            # Step function: hi before pivot, lo from pivot onwards
            return [
                hi if r < pivot else lo
                for r in range(1, max_rounds + 1)
            ]
        else:
            # Linear anneal from hi → lo across all rounds
            schedule: list[float] = []
            for r in range(1, max_rounds + 1):
                if max_rounds > 1:
                    t = (r - 1) / (max_rounds - 1)
                    temp = hi + (lo - hi) * t
                else:
                    temp = (hi + lo) / 2.0
                schedule.append(round(temp, 4))
            return schedule

    if pivot is not None:
        # No range specified: cool to 60% of base after pivot
        cooled = round(base * 0.6, 4)
        return [
            base if r < pivot else cooled
            for r in range(1, max_rounds + 1)
        ]

    # No scheduling at all — constant
    return [base] * max_rounds


# ─────────────────────────────────────────────────────────────────────────────
# TierConfig builder
# ─────────────────────────────────────────────────────────────────────────────

def build_tier_configs(
    agent1_model: str,
    agent2_model: str,
    agent1_temp: float,
    agent1_temp_pivot: Optional[int],
    agent1_temp_range: Optional[str],
    agent2_temp: float,
    agent2_temp_pivot: Optional[int],
    agent2_temp_range: Optional[str],
    max_rounds: int,
    timeout_s: float,
    ollama_host: str,
):
    """
    Return a (tier1_list, None) pair for the coordinator.
    Imports TierConfig lazily so the module is importable even without
    the effector package installed (e.g. during unit-testing the CLI itself).
    """
    from effector.adapters.asymmetric_dasp import TierConfig

    def _tc(name: str, model: str, temp: float, schedule: list[float]) -> Any:
        tc = TierConfig(
            name=name,
            model=model,
            host=ollama_host,
            temperature=temp,
            timeout_s=timeout_s,
            max_rounds=max_rounds,
        )
        # Attach schedule if the patched field exists
        if hasattr(tc, "temperature_schedule"):
            tc.temperature_schedule = schedule  # type: ignore[attr-defined]
        return tc

    a1_schedule = build_temp_schedule(agent1_temp, agent1_temp_pivot, agent1_temp_range, max_rounds)
    a2_schedule = build_temp_schedule(agent2_temp, agent2_temp_pivot, agent2_temp_range, max_rounds)

    tier1 = [
        _tc("agent1", agent1_model, agent1_temp, a1_schedule),
        _tc("agent2", agent2_model, agent2_temp, a2_schedule),
    ]
    return tier1


def build_tier2_config(
    arbiter_model: str,
    timeout_s: float,
    ollama_host: str,
):
    from effector.adapters.asymmetric_dasp import TierConfig
    return TierConfig(
        name="arbiter",
        model=arbiter_model,
        host=ollama_host,
        temperature=0.5,     # arbiter is always deliberate
        timeout_s=timeout_s,
        max_rounds=1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Event → display routing
# ─────────────────────────────────────────────────────────────────────────────

def make_event_handler(
    verbosity: int,
    output_format: str,
) -> Callable[[str, dict], None]:
    """
    Return an on_event(event, data) callback wired to the display layer.

    verbosity:
        -1  = show full agent answers
         0  = show only gates / signals (no per-response text)
        >0  = truncate answers to this many characters
    """
    silent = output_format == "silent"
    minimal = output_format == "minimal"

    def handler(event: str, data: dict) -> None:
        if silent:
            return

        if event == "session_started" and not minimal:
            console.print(
                f"[dim]Session {data.get('session_id', '')[:8]}…  "
                f"task: {data.get('task', '')[:80]}[/dim]"
            )

        elif event == "round_started" and not minimal:
            print_round_header(data.get("round", 0), data.get("tier", "local"))

        elif event == "round_complete" and not minimal:
            responses = data.get("responses", [])
            if verbosity != 0:
                print_round_responses(responses, verbosity)
            accumulators = data.get("accumulators", {})
            if accumulators:
                print_signal_manifold(accumulators)

        elif event in (
            "trigger_fired", "escalation_triggered",
            "consensus_cleared", "tier2_invoked",
        ):
            print_trigger(event, data)

    return handler


# ─────────────────────────────────────────────────────────────────────────────
# IEP pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_iep_pipeline(
    debate_result: dict[str, Any],
    state_bus: Any,
    snapshot_hash: str,
    keys_affected: list[str],
    vectorized_bus: bool,
    rat_threshold: float,
    embedding_model: str,
    ollama_host: str,
    output_format: str,
) -> dict[str, Any]:
    """
    Build → validate → display an IEP envelope from the debate result.
    Returns the validation result dict.
    """
    from effector.queue.iep_queue import IEPBuilder, IEPValidator

    envelope = IEPBuilder.from_debate_result(
        debate_result=debate_result,
        state_bus_snapshot_hash=snapshot_hash,
        keys_affected=keys_affected,
    )

    validator = IEPValidator(
        state_bus=state_bus,
        ollama_host=ollama_host,
    )
    validation = validator.validate(envelope)

    if output_format not in ("silent", "minimal"):
        status_style = "green" if validation.status == "ACK" else "red"
        console.print(
            f"\n  IEP [{status_style}]{validation.status}[/]"
            + (f"  — {validation.failure_reason}" if validation.failure_reason else "")
            + f"  (checks passed: {', '.join(validation.checks_passed)})"
        )

    return validation.to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# Main run orchestration
# ─────────────────────────────────────────────────────────────────────────────

def execute_run(
    *,
    task: str,
    agent1_model: str,
    agent2_model: str,
    arbiter_model: str,
    agent1_temp: float,
    agent1_temp_pivot: Optional[int],
    agent1_temp_range: Optional[str],
    agent2_temp: float,
    agent2_temp_pivot: Optional[int],
    agent2_temp_range: Optional[str],
    tau: float,
    theta: float,
    epsilon: float,
    max_rounds: int,
    verbosity: int,
    output_format: str,
    output_file: Optional[Path],
    persist_path: Optional[Path],
    ollama_host: str,
    timeout_s: float,
    poll_telemetry: bool,
    poll_interval_s: float,
    vectorized_bus: bool,
    embedding_model: str,
    rat_threshold: float,
    dry_run: bool,
    run_iep: bool,
    iep_keys_affected: list[str],
) -> int:
    """
    Full run pipeline.  Returns exit code (0 = success, 1 = error).
    """
    # ── Pre-flight display ────────────────────────────────────────────────────
    if output_format == "pretty":
        print_session_header(
            task=task,
            agent1=agent1_model,
            agent2=agent2_model,
            arbiter=arbiter_model,
            tau=tau,
            theta=theta,
            epsilon=epsilon,
            max_rounds=max_rounds,
            vectorized=vectorized_bus,
            ollama_host=ollama_host,
        )

    if dry_run:
        console.print("\n[yellow]Dry-run mode — no agents invoked.[/yellow]")
        return 0

    # ── StateBus + Telemetry ──────────────────────────────────────────────────
    from effector.state_bus.bus import StateBus
    bus = StateBus()
    poller = None

    if poll_telemetry:
        try:
            from effector.telemetry.poller import TelemetryPoller
            poller = TelemetryPoller(bus, interval_s=poll_interval_s)
            poller.start()
            # Give the first poll cycle time to populate the bus
            time.sleep(min(poll_interval_s, 1.5))
        except ImportError:
            print_warn("TelemetryPoller not available — psutil may be missing.")
        except Exception as exc:
            print_warn(f"Telemetry poller failed to start: {exc}")

    snapshot_hash, snap_ts, _ = bus.snapshot()

    # ── Build coordinator ─────────────────────────────────────────────────────
    try:
        tier1 = build_tier_configs(
            agent1_model=agent1_model,
            agent2_model=agent2_model,
            agent1_temp=agent1_temp,
            agent1_temp_pivot=agent1_temp_pivot,
            agent1_temp_range=agent1_temp_range,
            agent2_temp=agent2_temp,
            agent2_temp_pivot=agent2_temp_pivot,
            agent2_temp_range=agent2_temp_range,
            max_rounds=max_rounds,
            timeout_s=timeout_s,
            ollama_host=ollama_host,
        )
        tier2 = build_tier2_config(arbiter_model, timeout_s, ollama_host)
    except Exception as exc:
        print_error(f"Failed to build tier configs: {exc}")
        return 1

    from effector.adapters.asymmetric_dasp import AsymmetricDASPCoordinator
    coordinator = AsymmetricDASPCoordinator(
        tier1_agents=tier1,
        tier2_agent=tier2,
        tau_suppression=tau,
        theta_consensus=theta,
        epsilon_stall=epsilon,
        on_event=make_event_handler(verbosity, output_format),
        vectorized_bus=vectorized_bus,
        embedding_model=embedding_model,
        rat_similarity_threshold=rat_threshold,
        embedding_host=ollama_host,
    )

    # ── Run debate ────────────────────────────────────────────────────────────
    t0 = time.monotonic()
    try:
        result = coordinator.run(task=task, snapshot_hash=snapshot_hash, state_bus=bus)
    except Exception as exc:
        print_error(f"Coordinator error: {exc}")
        if poller:
            poller.stop()
        return 1
    finally:
        if poller:
            poller.stop()

    elapsed = time.monotonic() - t0
    result["_elapsed_s"] = round(elapsed, 2)

    # ── IEP pipeline (optional) ───────────────────────────────────────────────
    if run_iep:
        try:
            iep_result = run_iep_pipeline(
                debate_result=result,
                state_bus=bus,
                snapshot_hash=snapshot_hash,
                keys_affected=iep_keys_affected,
                vectorized_bus=vectorized_bus,
                rat_threshold=rat_threshold,
                embedding_model=embedding_model,
                ollama_host=ollama_host,
                output_format=output_format,
            )
            result["_iep_validation"] = iep_result
        except Exception as exc:
            print_warn(f"IEP pipeline error: {exc}")

    # ── Display result ────────────────────────────────────────────────────────
    if output_format == "pretty":
        console.print(f"\n  [dim]elapsed: {elapsed:.1f}s[/dim]")
        print_result(result)
    elif output_format == "json":
        print_result_json(result)
    elif output_format == "minimal":
        print_result_minimal(result)
    # silent: no output

    # ── Persist result ────────────────────────────────────────────────────────
    if output_file:
        try:
            output_file.write_text(json.dumps(result, indent=2, default=str))
            print_ok(f"Result saved → {output_file}")
        except Exception as exc:
            print_error(f"Could not write output file: {exc}")

    score = result.get("consensus_score", 0.0)
    reason = result.get("terminated_reason", "")
    if reason == "inhibition":
        return 2      # special exit code: inhibition gate
    if score < 0.4:
        return 3      # low-consensus warning
    return 0
