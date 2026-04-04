"""
Effector Engine — Main Reasoning Loop
======================================
Integrates Substrate Telemetry, Asymmetric DASP, Reflex Caching, and IEP.
Now featuring a Rich Live Dashboard.
"""

from __future__ import annotations
import argparse
import sys
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from collections import deque

import requests
import subprocess

# ── Rich Imports for Live UI ──
from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text

console = Console()

# ─── Thread-Safe UI State ───────────────────────────────────────────────────
ui_lock = threading.Lock()
ui_telemetry_text = "Waiting for telemetry..."
ui_dasp_status = "IDLE"
ui_manifold_text = "No active debate."
ui_logs = deque(maxlen=15)

def push_log(msg: str, style: str = "dim"):
    ts = datetime.now().strftime("%H:%M:%S")
    with ui_lock:
        ui_logs.append(f"[{style}][{ts}] {msg}[/]")

def update_ui_layout(layout: Layout):
    with ui_lock:
        layout["telemetry"].update(Panel(ui_telemetry_text, title="[cyan]A1: Telemetry Snapshot", border_style="cyan"))
        
        main_dasp_content = f"[bold yellow]Status:[/bold yellow] {ui_dasp_status}\n\n{ui_manifold_text}"
        layout["dasp"].update(Panel(main_dasp_content, title="[magenta]A2: DASP Signal Manifold", border_style="magenta"))
        
        log_content = "\n".join(ui_logs)
        layout["queue"].update(Panel(log_content, title="[green]A3: IEP Queue & Logs", border_style="green"))

def create_layout() -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1)
    )
    layout["main"].split_row(
        Layout(name="telemetry", ratio=1),
        Layout(name="dasp", ratio=2),
        Layout(name="queue", ratio=2)
    )
    layout["header"].update(Panel("[bold cyan]EFFECTOR ENGINE[/bold cyan] — Headless Reasoning Loop", style="bold white on dark_blue"))
    return layout

# ─── Bootstrapper ───────────────────────────────────────────────────────────
def verify_models(allow_auto_pull: bool = True):
    required = ["mistral:7b", "qwen2.5:14b", "qwen2.5-coder:32b", "nomic-embed-text"]
    try:
        resp = requests.get("http://127.0.0.1:11434/api/tags", timeout=3)
        installed = [m["name"] for m in resp.json().get("models", [])]
        for req in required:
            if req not in installed:
                if allow_auto_pull:
                    with console.status(f"[bold yellow]Pulling missing model: {req}..."):
                        subprocess.run(["ollama", "pull", req], check=True)
                else:
                    console.print(f"[bold red]Missing required model: {req}. Run 'effector models pull {req}'.")
                    sys.exit(1)
    except requests.exceptions.ConnectionError:
        console.print("[bold red]Ollama is not running. Please start Ollama at 127.0.0.1:11434")
        sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent / "src"))
try:
    from effector.telemetry.poller import TelemetryPoller
    from effector.telemetry.state_keys import KEYS
    from effector.adapters.asymmetric_dasp import AsymmetricDASPCoordinator, TierConfig
    from effector.queue.iep_queue import EnvelopeQueue, IEPBuilder, IEPValidator
    from effector.bus import StateBus
    from effector.rat_store import LocalRATStore
    from effector.main_loop_reflex import ReflexOrchestrator
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

# ─── DASP UI Hook ───────────────────────────────────────────────────────────
def _make_dasp_event_handler():
    def on_event(event: str, data: dict) -> None:
        global ui_dasp_status, ui_manifold_text
        with ui_lock:
            if event == "round_started":
                ui_dasp_status = f"Round {data['round']} ({data.get('tier', 'local')})"
                push_log(f"DASP: Round {data['round']} started.", "cyan")
            elif event == "round_complete":
                lines = []
                for hid, acc in data.get("accumulators", {}).items():
                    net = acc['S_net']
                    color = "green" if net > 0.5 else ("yellow" if net > 0 else "red")
                    lines.append(f"[{color}][{hid[:10]}] S_g:{acc['S_g']:.2f} | S_i:{acc['S_i']:.2f} | Net:{net:.2f}[/]")
                ui_manifold_text = "\n".join(lines)
            elif event == "escalation_triggered":
                ui_dasp_status = f"[bold red]ESCALATION: {data.get('trigger')} gate fired[/]"
                push_log("DASP Escalating to Tier-2...", "yellow")
            elif event == "consensus_cleared":
                ui_dasp_status = f"[bold green]CONSENSUS REACHED (Score: {data.get('consensus_score', 0):.2f})[/]"
    return on_event

# ─── Queue Consumer ─────────────────────────────────────────────────────────
def _consume_queue(eq: EnvelopeQueue, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            item = eq.get(timeout=1.0)
            status = item.validation.status
            eid = item.envelope.get("envelope_id", "?")[:8]
            verb = item.envelope.get("intended_action", {}).get("verb", "?")
            if item.is_ack():
                push_log(f"IEP: [green]ACK[/] {eid} ({verb})", "green")
            else:
                push_log(f"IEP: [red]NACK[/] {eid} - {item.validation.failure_reason}", "red")
        except:
            continue

# ─── Main Loop ──────────────────────────────────────────────────────────────
def run_loop(poll_interval_s: float, debate_interval_s: float, max_cycles: int, queue_file: str):
    verify_models()
    
    bus = StateBus()
    poller = TelemetryPoller(bus, interval_s=poll_interval_s)
    poller.start()
    time.sleep(poll_interval_s + 0.5)

    tier1 = [
        TierConfig(name="mistral", model="mistral:7b", characterizer_model="qwen2.5-coder:32b", max_rounds=2),
        TierConfig(name="qwen", model="qwen2.5:14b", characterizer_model="qwen2.5-coder:32b", max_rounds=2)
    ]
    tier2 = TierConfig(name="nemotron", model="qwen2.5:14b", characterizer_model="qwen2.5-coder:32b", max_rounds=1)

    coordinator = AsymmetricDASPCoordinator(
        tier1_agents=tier1, tier2_agent=tier2,
        tau_suppression=0.5, theta_consensus=0.65, epsilon_stall=0.04,
        vectorized_bus=True
    )
    coordinator._on_event = _make_dasp_event_handler()

    rat_store = LocalRATStore()
    orchestrator = ReflexOrchestrator(state_bus=bus, rat_store=rat_store, dasp_run_fn=coordinator.run)

    validator = IEPValidator(state_bus=bus)
    eq = EnvelopeQueue(persist_path=queue_file)

    stop_consumer = threading.Event()
    threading.Thread(target=_consume_queue, args=(eq, stop_consumer), daemon=True).start()

    layout = create_layout()
    cycle = 0

    with Live(layout, refresh_per_second=4, screen=True):
        try:
            while True:
                cycle += 1
                state = bus.read()
                snap_hash, snap_ts, _ = bus.snapshot()

                cpu = state.get(KEYS.cpu_percent_total, 0.0)
                ram = state.get(KEYS.ram_percent, 0.0)
                pressure = state.get(KEYS.system_pressure, 0.0)
                window = state.get(KEYS.active_window_title, "(unknown)")
                proc = state.get(KEYS.active_process_name, "(unknown)")

                global ui_telemetry_text, ui_dasp_status, ui_manifold_text
                with ui_lock:
                    ui_telemetry_text = f"Cycle: {cycle}\n\nCPU: {cpu:.1f}%\nRAM: {ram:.1f}%\nPressure: {pressure:.3f}\n\nWindow: {window}\nProcess: {proc}"
                    ui_dasp_status = "Evaluating telemetry..."
                update_ui_layout(layout)

                task = (
                    f"Analyse system health. CPU={cpu:.1f}%, RAM={ram:.1f}%, pressure={pressure:.3f}, "
                    f"window='{window}'. Should any corrective action be taken?"
                )

                push_log("Orchestrator checking Reflex Cache...")
                result = orchestrator.handle(task=task, snapshot_hash=snap_hash)

                if result.get("_path") == "reflex":
                    push_log(f"⚡ Reflex Cache Hit! Status: {result.get('status')}", "yellow")
                    with ui_lock:
                        ui_dasp_status = f"[yellow]Reflex Executed[/] (RAT: {result.get('rat_id', '')[:8]})"
                        ui_manifold_text = "Bypassed LLM Debate."
                else:
                    push_log("DASP Debate Complete.", "white")

                emit_hash, _, _ = bus.snapshot()
                envelope = IEPBuilder.from_debate_result(
                    debate_result=result, 
                    state_bus_snapshot_hash=emit_hash,
                    keys_affected=[KEYS.system_pressure]
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
            rat_store.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--debate-interval", type=float, default=30.0)
    parser.add_argument("--max-cycles", type=int, default=0)
    parser.add_argument("--queue-file", type=str, default="iep_queue.jsonl")
    args = parser.parse_args()

    run_loop(args.poll_interval, args.debate_interval, args.max_cycles, args.queue_file)