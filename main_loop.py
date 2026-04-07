"""
Effector Engine — Main Reasoning Loop
======================================
Integrates Substrate Telemetry, Asymmetric DASP, Reflex Caching, IEP,
Foley Ambient Audio, and the Resonance Layer.

Resonance Layer additions
--------------------------
Phase 1 — The Rehearsal Room (ResonanceVoice / Kokoro-82M)
    Agents speak their reasoning aloud during DASP deliberation.
    The player hears the Acolytes passionately debating desktop thermodynamics.

Phase 2 — The Rite of Offering (existing IEP Queue UI)
    The debate concludes. The Rich UI displays the Intention Envelope.
    The agents have fallen silent. The petition is on the altar.

Phase 3 — The Sacred Interval (ResonanceWorld / AudioLDM 2)
    The player clicks ACK (or the Reflex cache approves). The UI locks into
    the Sacred Interval display, showing diffusion step progress. The OS
    action is HELD until the audio generation completes. The 30-second wait
    is the physics of miracle-work, made visible.

Phase 4 — Manifestation or The Void
    ACK: The generated audio plays. At that exact millisecond, the OS
         action executes. Reality changes.
    NACK: Nothing. The speakers stay dead. The desktop stays messy. The
          agents will notice in the next polling cycle.
"""

from __future__ import annotations
import argparse
import sys
import time
import threading
import queue as _queue
from datetime import datetime, timezone
from pathlib import Path
from collections import deque
from typing import Any, Optional

import requests
import subprocess

# ── Rich imports ──────────────────────────────────────────────────────────────
from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

console = Console()

# ─── Thread-safe UI state ─────────────────────────────────────────────────────
ui_lock = threading.Lock()
ui_telemetry_text: str = "Waiting for telemetry..."
ui_dasp_status:    str = "IDLE"
ui_manifold_text:  str = "No active debate."
ui_logs: deque = deque(maxlen=18)

# Sacred Interval state (set by ResonanceWorld on_progress callback)
ui_sacred_active:  bool = False
ui_sacred_text:    str = ""
ui_sacred_prompt:  str = ""
ui_sacred_step:    int = 0
ui_sacred_total:   int = 50


def push_log(msg: str, style: str = "dim") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    with ui_lock:
        ui_logs.append(f"[{style}][{ts}] {msg}[/]")


def update_ui_layout(layout: Layout) -> None:
    """Repaint all three panels from global state. Thread-safe via ui_lock."""
    with ui_lock:
        _update_telemetry(layout)
        _update_dasp_or_sacred(layout)
        _update_queue(layout)


def _update_telemetry(layout: Layout) -> None:
    layout["telemetry"].update(
        Panel(
            ui_telemetry_text,
            title="[cyan]A1: Telemetry Snapshot[/cyan]",
            border_style="cyan",
        )
    )


def _update_dasp_or_sacred(layout: Layout) -> None:
    if ui_sacred_active:
        # The Sacred Interval takes over the DASP panel
        filled  = int(20 * ui_sacred_step / max(ui_sacred_total, 1))
        bar     = "█" * filled + "░" * (20 - filled)
        content = (
            f"[bold gold1]⚖️  THE EFFECTOR WEIGHS THE OFFERING[/bold gold1]\n\n"
            f"{ui_sacred_text}\n\n"
            f"[dim]{ui_sacred_prompt[:80]}[/dim]"
        )
        layout["dasp"].update(
            Panel(
                content,
                title="[gold1]✦ The Sacred Interval[/gold1]",
                border_style="gold1",
            )
        )
    else:
        content = f"[bold yellow]Status:[/bold yellow] {ui_dasp_status}\n\n{ui_manifold_text}"
        layout["dasp"].update(
            Panel(
                content,
                title="[magenta]A2: DASP Signal Manifold[/magenta]",
                border_style="magenta",
            )
        )


def _update_queue(layout: Layout) -> None:
    log_content = "\n".join(ui_logs)
    layout["queue"].update(
        Panel(
            log_content,
            title="[green]A3: IEP Queue & Logs[/green]",
            border_style="green",
        )
    )


def create_layout() -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
    )
    layout["main"].split_row(
        Layout(name="telemetry", ratio=1),
        Layout(name="dasp",      ratio=2),
        Layout(name="queue",     ratio=2),
    )
    layout["header"].update(
        Panel(
            "[bold cyan]EFFECTOR ENGINE[/bold cyan]"
            " — Headless Reasoning Loop  [dim]| The Agents Are Watching[/dim]",
            style="bold white on dark_blue",
        )
    )
    return layout


# ─── Model bootstrapper ───────────────────────────────────────────────────────

def verify_models(allow_auto_pull: bool = True) -> None:
    required = [
        "mistral:7b",
        "qwen2.5:14b",
        "qwen2.5-coder:32b",
        "nomic-embed-text",
    ]
    try:
        resp = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
        installed = [m["name"] for m in resp.json().get("models", [])]
        for req in required:
            if req not in installed:
                if allow_auto_pull:
                    with console.status(f"[bold yellow]Pulling missing model: {req}..."):
                        subprocess.run(["ollama", "pull", req], check=True)
                else:
                    console.print(
                        f"[bold red]Missing required model: {req}. "
                        f"Run 'effector models pull {req}'."
                    )
                    sys.exit(1)
    except requests.exceptions.ConnectionError:
        console.print(
            "[bold red]Ollama is not running. "
            "Please start Ollama at 127.0.0.1:11434"
        )
        sys.exit(1)


sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from effector.telemetry.poller    import TelemetryPoller
    from effector.telemetry.state_keys import KEYS
    from effector.adapters.asymmetric_dasp import (
        AsymmetricDASPCoordinator,
        TierConfig,
    )
    from effector.queue.iep_queue     import EnvelopeQueue, IEPBuilder, IEPValidator
    from effector.bus                 import StateBus
    from effector.rat_store           import LocalRATStore
    from effector.main_loop_reflex    import ReflexOrchestrator
    from effector.executor            import LocalExecutor
    from effector.foley               import create_foley_system
    from effector.foley.integration   import wire_foley_to_main_loop
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

# Resonance Layer — optional; degrade gracefully if not installed
try:
    from effector.resonance import ResonanceVoice, ResonanceWorld
    _RESONANCE_AVAILABLE = True
except ImportError:
    _RESONANCE_AVAILABLE = False
    print(
        "[ResonanceLayer] resonance module not found. "
        "Place src/effector/resonance/ in your project to enable voices and world sounds."
    )


# ─── DASP event handler (with optional voice) ─────────────────────────────────

def _make_dasp_event_handler(
    voice: Optional["ResonanceVoice"] = None,
) -> Any:
    """
    Returns an on_event(event, data) callback wired to the display layer
    and optionally to the ResonanceVoice for agent speech.
    """
    def on_event(event: str, data: dict) -> None:
        global ui_dasp_status, ui_manifold_text

        with ui_lock:
            if event == "round_started":
                ui_dasp_status = (
                    f"Round {data['round']} ({data.get('tier', 'local')})"
                )
                push_log(
                    f"DASP: Round {data['round']} started.", "cyan"
                )
                # Announce round number via TTS
                if voice and voice.is_available():
                    voice.announce("round_started", data.get("round", 0))

            elif event == "round_complete":
                # ── Update signal manifold display ─────────────────────────────
                lines = []
                for hid, acc in data.get("accumulators", {}).items():
                    net   = acc["S_net"]
                    color = "green" if net > 0.5 else ("yellow" if net > 0 else "red")
                    lines.append(
                        f"[{color}][{hid[:10]}] "
                        f"S_g:{acc['S_g']:.2f} | "
                        f"S_i:{acc['S_i']:.2f} | "
                        f"Net:{net:.2f}[/]"
                    )
                ui_manifold_text = "\n".join(lines)

                # ── Speak agent explanations ───────────────────────────────────
                if voice and voice.is_available():
                    for resp in data.get("responses", []):
                        explanation = resp.get("explanation", "")
                        agent_id    = resp.get("agent_id", "default")
                        if explanation and "(abstention)" not in explanation.lower():
                            voice.speak(explanation, agent_id=agent_id)

            elif event == "escalation_triggered":
                trigger = data.get("trigger", "")
                ui_dasp_status = (
                    f"[bold red]ESCALATION: {trigger} gate fired[/]"
                )
                push_log("DASP: Escalating to Tier-2...", "yellow")
                if voice and voice.is_available():
                    if trigger == "inhibition":
                        voice.announce("inhibition")
                    else:
                        voice.announce("escalation")

            elif event == "consensus_cleared":
                score = data.get("consensus_score", 0)
                ui_dasp_status = (
                    f"[bold green]CONSENSUS REACHED (Score: {score:.2f})[/]"
                )
                if voice and voice.is_available():
                    voice.announce("consensus")

    return on_event


# ─── Sacred Interval progress callback ────────────────────────────────────────

def _make_sacred_interval_progress(layout: Layout) -> Any:
    """
    Returns a callback(step, total, text) that updates the Sacred Interval
    UI in real time during AudioLDM 2 diffusion.
    """
    def on_progress(step: int, total: int, text: str) -> None:
        global ui_sacred_active, ui_sacred_text, ui_sacred_step, ui_sacred_total

        with ui_lock:
            ui_sacred_active = step < total or (step == 0 and total > 0)
            ui_sacred_text   = text
            ui_sacred_step   = step
            ui_sacred_total  = total if total > 0 else 50

        # Update the layout immediately so the Live widget picks it up
        try:
            _update_dasp_or_sacred(layout)
            _update_queue(layout)
        except Exception:
            pass

    return on_progress


# ─── IEP Queue consumer with Sacred Interval ──────────────────────────────────

def _consume_queue(
    eq: EnvelopeQueue,
    stop: threading.Event,
    executor: Any,
    world: Optional["ResonanceWorld"] = None,
    voice: Optional["ResonanceVoice"] = None,
    layout: Optional[Layout] = None,
) -> None:
    """
    IEP queue consumer.

    ACK path  — Sacred Interval (generate audio) → play → execute OS action.
    NACK path — Silence. The void. The agents notice next cycle.
    """
    global ui_sacred_active, ui_sacred_text

    while not stop.is_set():
        try:
            item = eq.get(timeout=1.0)
        except _queue.Empty:
            continue
        except Exception:
            continue

        eid    = item.envelope.get("envelope_id", "?")[:8]
        verb   = item.envelope.get("intended_action", {}).get("verb",   "?")
        target = item.envelope.get("intended_action", {}).get("target", "world_state")

        if item.is_ack():
            # ── Phase 2 → 3: Rite of Offering accepted ────────────────────────
            push_log(
                f"IEP: [green]ACK[/] {eid} ({verb}) — "
                f"[italic]The Offering is accepted[/italic]",
                "green",
            )

            # ── Phase 3: The Sacred Interval ───────────────────────────────────
            if world and world.is_available():
                push_log(
                    f"⚖️  [gold1]The Sacred Interval begins — "
                    f"The Effector generates: {target}[/]",
                    "gold1",
                )

                with ui_lock:
                    ui_sacred_active = True
                    ui_sacred_text   = f"⚖️  Preparing the manifestation for: {target}"
                    ui_sacred_step   = 0

                if layout is not None:
                    _update_dasp_or_sacred(layout)

                # BLOCKING: generates audio (30-45s on CPU, 15-25s on GPU)
                audio_path = world.generate_and_play(target)

                with ui_lock:
                    ui_sacred_active = False
                    ui_sacred_text   = ""

                if layout is not None:
                    try:
                        _update_dasp_or_sacred(layout)
                    except Exception:
                        pass

                if audio_path:
                    push_log(
                        f"✦ [bold gold1]Manifestation complete.[/] "
                        f"The miracle has a sound.",
                        "bold",
                    )
                else:
                    push_log(
                        "✦ [dim](World sound unavailable — "
                        "manifesting in silence)[/dim]",
                        "dim",
                    )
            else:
                push_log(
                    f"✦ [dim]World sound not loaded — "
                    f"manifesting in silence.[/dim]",
                    "dim",
                )

            # ── Phase 4: Manifestation — execute the OS action ─────────────────
            push_log(f"Executing {eid} → {target}...", "yellow")
            try:
                executor.execute(item.envelope)
                push_log(
                    f"✦ [green bold]Reality altered.[/green bold] "
                    f"The agents will see the change.",
                    "bold green",
                )
                if voice and voice.is_available():
                    voice.announce("offering_accepted")
            except Exception as exc:
                push_log(f"[red]Execution error: {exc}[/red]", "red")

        else:
            # ── Phase 4: The Void — NACK ────────────────────────────────────────
            # No audio. No execution. The speakers stay dead.
            reason = item.validation.failure_reason or "unknown"
            push_log(
                f"IEP: [red]NACK[/] {eid} — "
                f"[italic]The Offering dissolves into the void[/italic]",
                "red",
            )
            push_log(
                f"[dim]Reason: {reason}. "
                f"The desktop remains unchanged. "
                f"The agents will notice.[/dim]",
                "dim",
            )
            if voice and voice.is_available():
                voice.announce("offering_rejected")


# ─── Main Loop ────────────────────────────────────────────────────────────────

def run_loop(
    poll_interval_s: float = 5.0,
    debate_interval_s: float = 30.0,
    max_cycles: int = 0,
    queue_file: str = "iep_queue.jsonl",
    voice_enabled: bool = True,
    world_enabled: bool = True,
) -> None:

    verify_models()

    # ── Substrate ─────────────────────────────────────────────────────────────
    bus    = StateBus()
    poller = TelemetryPoller(bus, interval_s=poll_interval_s)
    poller.start()
    time.sleep(poll_interval_s + 0.5)

    # ── DASP Coordinator ──────────────────────────────────────────────────────
    tier1 = [
        TierConfig(
            name="mistral",
            model="mistral:7b",
            characterizer_model="qwen2.5-coder:32b",
            max_rounds=2,
        ),
        TierConfig(
            name="qwen",
            model="qwen2.5:14b",
            characterizer_model="qwen2.5-coder:32b",
            max_rounds=2,
        ),
    ]
    tier2 = TierConfig(
        name="nemotron",
        model="qwen2.5:14b",
        characterizer_model="qwen2.5-coder:32b",
        max_rounds=1,
    )
    coordinator = AsymmetricDASPCoordinator(
        tier1_agents=tier1,
        tier2_agent=tier2,
        tau_suppression=0.5,
        theta_consensus=0.65,
        epsilon_stall=0.04,
        vectorized_bus=True,
    )

    # ── Reflex middleware ─────────────────────────────────────────────────────
    rat_store    = LocalRATStore()
    orchestrator = ReflexOrchestrator(
        state_bus=bus,
        rat_store=rat_store,
        dasp_run_fn=coordinator.run,
    )

    # ── Foley ambient audio ───────────────────────────────────────────────────
    push_log("Initialising Foley ambient audio...")
    player, mapper, scheduler = create_foley_system(bus)
    wire_foley_to_main_loop(coordinator, orchestrator, bus, mapper, player)

    # ── IEP pipeline ──────────────────────────────────────────────────────────
    validator = IEPValidator(state_bus=bus)
    eq        = EnvelopeQueue(persist_path=queue_file)
    executor  = LocalExecutor(state_bus=bus, foley_mapper=mapper)

    # ── Resonance Layer ───────────────────────────────────────────────────────
    voice: Optional[ResonanceVoice] = None
    world: Optional[ResonanceWorld] = None

    if _RESONANCE_AVAILABLE:
        layout_ref: list[Optional[Layout]] = [None]  # forward ref for closure

        def _on_sacred_progress(step: int, total: int, text: str) -> None:
            global ui_sacred_active, ui_sacred_text, ui_sacred_step, ui_sacred_total
            with ui_lock:
                ui_sacred_active = step < total or (step == 0 and total > 0)
                ui_sacred_text   = text
                ui_sacred_step   = step
                ui_sacred_total  = total if total > 0 else 50
            
        voice = ResonanceVoice(enabled=voice_enabled)
        world = ResonanceWorld(
            enabled=world_enabled,
            on_progress=_on_sacred_progress,
        )

        push_log("Starting Resonance Layer (Kokoro + AudioLDM 2)...")
        voice.start()
        world.start()

        # Non-blocking — the loop starts while models load in background.
        # Voices become available once Kokoro loads (~3-10s).
        # World sounds become available once AudioLDM 2 loads (~30-120s
        # or download time on first run).
    else:
        push_log(
            "[dim]Resonance Layer not available — "
            "no voices, no world sounds.[/dim]",
            "dim",
        )
        layout_ref = [None]

    # ── Wire voice into DASP coordinator ─────────────────────────────────────
    coordinator._on_event = _make_dasp_event_handler(voice=voice)

    # ── Layout ────────────────────────────────────────────────────────────────
    layout = create_layout()
    if _RESONANCE_AVAILABLE and layout_ref is not None:
        layout_ref[0] = layout  # inject into sacred interval callback closure

    # ── Queue consumer thread ─────────────────────────────────────────────────
    stop_consumer = threading.Event()
    consumer_thread = threading.Thread(
        target=_consume_queue,
        args=(eq, stop_consumer, executor),
        kwargs={"world": world, "voice": voice, "layout": layout},
        name="IEPQueueConsumer",
        daemon=True,
    )
    consumer_thread.start()

    # ── Main loop ─────────────────────────────────────────────────────────────
    cycle = 0
    with Live(layout, refresh_per_second=1, screen=True):
        try:
            while True:
                cycle += 1
                state = bus.read()
                snap_hash, snap_ts, _ = bus.snapshot()

                cpu      = state.get(KEYS.cpu_percent_total, 0.0)
                ram      = state.get(KEYS.ram_percent, 0.0)
                pressure = state.get(KEYS.system_pressure, 0.0)
                window   = state.get(KEYS.active_window_title, "(unknown)")
                proc     = state.get(KEYS.active_process_name, "(unknown)")

                # ── Update telemetry panel ─────────────────────────────────────
                global ui_dasp_status
                with ui_lock:
                    global ui_telemetry_text
                    ui_telemetry_text = (
                        f"Cycle: {cycle}\n\n"
                        f"CPU: {cpu:.1f}%\n"
                        f"RAM: {ram:.1f}%\n"
                        f"Pressure: {pressure:.3f}\n\n"
                        f"Window: {window}\n"
                        f"Process: {proc}"
                    )
                    ui_dasp_status = "Evaluating telemetry..."

                update_ui_layout(layout)

                # ── Build task for this cycle ──────────────────────────────────
                task = (
                    f"Analyse system health and habitat cozy-ness. "
                    f"CPU={cpu:.1f}%, RAM={ram:.1f}%, pressure={pressure:.3f}, "
                    f"window='{window}'. "
                    f"If the desktop is messy or the user looks stressed, "
                    f"propose organizing the desktop files into a Zen Habitat."
                )

                push_log("Checking Reflex Cache...")
                result = orchestrator.handle(task=task, snapshot_hash=snap_hash)

                if result.get("_path") == "reflex":
                    push_log(
                        f"⚡ Reflex hit — {result.get('status')} "
                        f"(RAT: {str(result.get('rat_id', ''))[:8]})",
                        "yellow",
                    )
                    with ui_lock:
                        ui_dasp_status = (
                            f"[yellow]Reflex Executed[/] "
                            f"(RAT: {str(result.get('rat_id', ''))[:8]})"
                        )
                        ui_manifold_text = "Bypassed LLM deliberation."
                else:
                    push_log("DASP debate complete.", "white")

                # ── Build and validate IEP envelope ───────────────────────────
                emit_hash, _, _ = bus.snapshot()
                envelope = IEPBuilder.from_debate_result(
                    debate_result=result,
                    state_bus_snapshot_hash=emit_hash,
                    keys_affected=[KEYS.system_pressure],
                )
                verdict = validator.validate(envelope)
                eq.put(envelope, verdict)

                with ui_lock:
                    ui_dasp_status = "Sleeping..."
                update_ui_layout(layout)

                if max_cycles and cycle >= max_cycles:
                    break

                time.sleep(debate_interval_s)

        except KeyboardInterrupt:
            pass
        finally:
            poller.stop()
            stop_consumer.set()
            consumer_thread.join(timeout=5.0)
            rat_store.close()
            if voice:
                voice.stop()
            # AudioLDM 2 pipeline has no explicit close; GC handles it
            if "player" in dir():
                player.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Effector Engine — Headless Reasoning Loop"
    )
    parser.add_argument("--poll-interval",    type=float, default=5.0)
    parser.add_argument("--debate-interval",  type=float, default=30.0)
    parser.add_argument("--max-cycles",       type=int,   default=0)
    parser.add_argument("--queue-file",       type=str,   default="iep_queue.jsonl")
    parser.add_argument(
        "--no-voice",
        action="store_true",
        help="Disable Kokoro TTS agent voices",
    )
    parser.add_argument(
        "--no-world",
        action="store_true",
        help="Disable AudioLDM 2 world sound generation (Sacred Interval)",
    )
    args = parser.parse_args()

    run_loop(
        poll_interval_s=args.poll_interval,
        debate_interval_s=args.debate_interval,
        max_cycles=args.max_cycles,
        queue_file=args.queue_file,
        voice_enabled=not args.no_voice,
        world_enabled=not args.no_world,
    )
