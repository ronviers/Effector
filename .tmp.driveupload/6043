"""
train_intention_layer.py — Orchestrates all four Intention Layer training steps
===============================================================================
Run this once from the project root after pulling Ollama models.

Steps executed
--------------
  Step 1  phi_probe       — Cultivation log → θ forward model R² check
  Step 2  signal_head     — Sweep data → embedding-based characterizer
  Step 3  phi_model       — Labeled dataset → φ injection model
  (Step 4 layer.py needs no training — it's a decision function)

All models are saved to data/ and can be loaded independently.

Requirements
------------
  pip install scikit-learn
  ollama pull nomic-embed-text:latest    # default embedding model
  # OR:
  ollama pull qwen3:embedding     # higher-quality, slower

Usage
-----
  # Full pipeline (default embedding model)
  python train_intention_layer.py

  # Use a different embedding model
  python train_intention_layer.py --model qwen3:embedding

  # Skip cultivation probe (if log is missing or small)
  python train_intention_layer.py --skip-probe

  # Skip signal head training (if no sweep data)
  python train_intention_layer.py --skip-head

  # Only train the φ model (Step 3 — just needs Ollama)
  python train_intention_layer.py --phi-only

  # Dry run — check data availability, don't embed
  python train_intention_layer.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT / "src"))

# ── Default paths ─────────────────────────────────────────────────────────────
CULTIVATION_LOG = _ROOT / "mcp" / "cultivation_log.jsonl"
SWEEP_DIRS = [
    _ROOT / "src" / "effector" / "sweep_results_models",
    _ROOT / "src" / "effector" / "sweep_results_logic",
]
DATA_DIR         = _ROOT / "data"
PHI_PROBE_PATH   = DATA_DIR / "phi_probe.pkl"
SIGNAL_HEAD_PATH = DATA_DIR / "signal_head.pkl"
PHI_MODEL_PATH   = DATA_DIR / "phi_model.pkl"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _header(title: str) -> None:
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")

def _check_ollama(host: str) -> bool:
    try:
        import requests
        resp = requests.get(f"{host}/api/tags", timeout=3)
        models = [m["name"] for m in resp.json().get("models", [])]
        return True, models
    except Exception as exc:
        return False, []

def _check_sklearn() -> bool:
    try:
        import sklearn
        return True
    except ImportError:
        return False

def _count_cultivation_samples(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    import json
    count = 0
    with open(log_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if not row.get("dry_run") and row.get("verdict") == "ACK":
                    reasoning = row.get("reasoning", "")
                    if reasoning and "(abstention)" not in reasoning:
                        count += 1
            except Exception:
                pass
    return count

def _count_sweep_samples(sweep_dirs: list[Path]) -> int:
    import json
    count = 0
    for d in sweep_dirs:
        if not d.exists():
            continue
        for path in d.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                for rnd in data.get("all_rounds", []):
                    for resp in rnd.get("responses", []):
                        exp = resp.get("explanation", "")
                        sig = resp.get("signal", {})
                        if exp and "(abstention)" not in exp and float(sig.get("confidence", 0)) > 0:
                            count += 1
            except Exception:
                pass
    return count

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="train_intention_layer",
        description="Train all Intention Layer models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", default="nomic-embed-text:latest",
        help="Ollama embedding model (default: nomic-embed-text:latest)"
    )
    parser.add_argument(
        "--host", default="http://127.0.0.1:11434",
        help="Ollama API host"
    )
    parser.add_argument("--skip-probe", action="store_true",
                        help="Skip Step 1 (cultivation log probe)")
    parser.add_argument("--skip-head", action="store_true",
                        help="Skip Step 2 (signal head training)")
    parser.add_argument("--phi-only", action="store_true",
                        help="Only train the φ model (Step 3)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check data availability only, no training")
    parser.add_argument("--test-frac", type=float, default=0.20,
                        help="Test fraction for evaluation (default 0.20)")
    args = parser.parse_args()

    print()
    print("  Effector Engine — Intention Layer Training")
    print("  =" * 22)

    # ── Pre-flight checks ─────────────────────────────────────────────────────
    _header("Pre-flight checks")

    ollama_ok, installed_models = _check_ollama(args.host)
    if not ollama_ok:
        print(f"  ✗ Ollama unreachable at {args.host}")
        print("    Start Ollama and try again.")
        sys.exit(1)
    print(f"  ✓ Ollama reachable  ({len(installed_models)} models installed)")

    if args.model not in installed_models:
        print(f"  ✗ Embedding model {args.model!r} not installed")
        print(f"    Run: ollama pull {args.model}")
        sys.exit(1)
    print(f"  ✓ Embedding model   {args.model!r}")

    if not _check_sklearn():
        print("  ✗ scikit-learn not installed")
        print("    Run: pip install scikit-learn")
        sys.exit(1)
    print("  ✓ scikit-learn")

    n_cult = _count_cultivation_samples(CULTIVATION_LOG)
    n_sweep = _count_sweep_samples(SWEEP_DIRS)
    print(f"  ─")
    print(f"  Cultivation log     : {n_cult} ACK entries  ({'OK' if n_cult >= 20 else 'LOW — run more cultivation sessions'})")
    print(f"  Sweep results       : {n_sweep} agent samples  ({'OK' if n_sweep >= 20 else 'LOW — run more sweeps'})")
    print(f"  φ dataset           : 52 labeled examples  (built-in)")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Data dir            : {DATA_DIR}")

    if args.dry_run:
        print("\n  Dry run — no training performed.")
        return

    t_total = time.monotonic()

    # ── Step 1: Cultivation log probe ─────────────────────────────────────────
    if not args.skip_probe and not args.phi_only:
        _header("Step 1 — Cultivation Log Probe (θ forward model)")
        if n_cult < 10:
            print(f"  Skipping: only {n_cult} usable ACK entries (need ≥10)")
            print("  Run: python mcp/cultivation_loop.py --sessions 12")
        else:
            sys.path.insert(0, str(_ROOT / "src" / "effector" / "intention"))
            from effector.intention.phi_probe import run_probe
            result = run_probe(
                log_path=CULTIVATION_LOG,
                embedding_model=args.model,
                ollama_host=args.host,
                test_frac=args.test_frac,
                save_path=PHI_PROBE_PATH,
                verbose=True,
            )
            print()
            print(result.report())
            if result.prophecy_r2 is not None and result.prophecy_r2 > 0.30:
                print("  → R² > 0.30: embedding geometry has usable θ signal.")
                print("    The signal head (Step 2) should show reduced characterizer error.")
            else:
                print("  → R² ≤ 0.30: geometry signal is weak. More cultivation data needed.")
    else:
        if not args.phi_only:
            print("\n  Step 1: skipped")

    # ── Step 2: Signal head ───────────────────────────────────────────────────
    if not args.skip_head and not args.phi_only:
        _header("Step 2 — Embedding-Based Signal Head (characterizer replacement)")
        if n_sweep < 20:
            print(f"  Skipping: only {n_sweep} agent samples (need ≥20)")
            print("  Run more effector sweeps to build training data.")
        else:
            from effector.intention.signal_head import load_sweep_data, train_head
            samples = load_sweep_data(SWEEP_DIRS)
            result = train_head(
                samples=samples,
                embedding_model=args.model,
                ollama_host=args.host,
                test_frac=args.test_frac,
                save_path=SIGNAL_HEAD_PATH,
                verbose=True,
            )
            print()
            print(result.report())
            if result.confidence_mae is not None and result.confidence_mae < 0.08:
                print(
                    "  → MAE < 0.08: signal head is ready for use as Phase 2 replacement.\n"
                    "    Set EFFECTOR_SIGNAL_HEAD=data/signal_head.pkl to activate."
                )
            else:
                print(
                    "  → MAE ≥ 0.08: signal head below quality target.\n"
                    "    More diverse sweep data will improve performance."
                )
    else:
        if not args.phi_only:
            print("\n  Step 2: skipped")

    # ── Step 3: φ injection model ─────────────────────────────────────────────
    _header("Step 3 — φ Injection Model (Intention Layer core)")
    from effector.intention.phi_model import train_phi_model
    result = train_phi_model(
        embedding_model=args.model,
        ollama_host=args.host,
        save_path=PHI_MODEL_PATH,
        verbose=True,
    )
    print()
    print(result.report())

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.monotonic() - t_total
    _header(f"Training complete  ({elapsed:.0f}s)")
    for label, path, target in [
        ("φ probe model",  PHI_PROBE_PATH,   "Step 1"),
        ("Signal head",    SIGNAL_HEAD_PATH,  "Step 2"),
        ("φ injection",    PHI_MODEL_PATH,    "Step 3"),
    ]:
        status = "✓" if path.exists() else "✗ not created"
        print(f"  {status}  {label:<22} {path.relative_to(_ROOT)}")

    print()
    print("  Next steps:")
    print("  ─")
    print("  # Run the Intention Layer demo (cold/warm/hot habitat)")
    print("  python src/effector/intention/layer.py --demo")
    print()
    print("  # Assess a single notification")
    print('  python src/effector/intention/layer.py --text "Meeting in 10 minutes" --pressure 0.7')
    print()
    print("  # Activate the signal head (drop-in for Phase 2 characterizer)")
    print("  # In your TierConfig or effector config:")
    print("  #   EFFECTOR_SIGNAL_HEAD = data/signal_head.pkl")
    print()

if __name__ == "__main__":
    main()
