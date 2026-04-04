"""
signal_head.py — Step 2: Embedding-Based DASP Signal Head
===========================================================
Replaces the second LLM call in TwoPhaseOllamaAgent's characterizer with
a set of four sklearn regressors trained on sweep result data.

The Characterizer (Phase 2) currently: embed reasoning_text → LLM call → signals
This module:                            embed reasoning_text → dot product → signals

Latency improvement: ~2700ms (LLM) → ~35ms (embedding + regression).
Quality target: within 0.08 MAE of the LLM characterizer on held-out sweep data.

Training data source
--------------------
All JSON files in src/effector/sweep_results_models/ and
src/effector/sweep_results_logic/ contain agent responses with:
  - explanation: Phase 1 reasoning text  (X)
  - signal.*:    four float values        (y)

Non-abstention responses only (confidence > 0 required).

Model architecture
------------------
  Four independent estimators, all operating on the same embedding:
    1. Ridge(alpha=1.0)              → confidence     [0, 1]
    2. LogisticRegression(C=0.5)    → polarity       {-1, 0, 1}  (multi-class)
    3. Ridge(alpha=1.0)              → generative_strength  [0, 1]
    4. Ridge(alpha=1.0)              → inhibitory_pressure  [0, 1]

  A shared StandardScaler is fit on the training embeddings and applied to all.

  Output is clamped to valid ranges before returning.

Usage
-----
  # Train
  python src/effector/intention/signal_head.py train \\
      --sweep-dir src/effector/sweep_results_models \\
      --save data/signal_head.pkl

  # Evaluate
  python src/effector/intention/signal_head.py eval \\
      --sweep-dir src/effector/sweep_results_models \\
      --load data/signal_head.pkl

  # Live inference (replaces Phase 2 in TwoPhaseOllamaAgent)
  python src/effector/intention/signal_head.py infer \\
      --load data/signal_head.pkl \\
      --text "The CPU is spiking due to SearchIndexer. We should wait."

Drop-in integration
-------------------
  from effector.intention.signal_head import EmbeddingCharacterizer
  
  # In TwoPhaseOllamaAgent.__init__:
  if EmbeddingCharacterizer.available(head_path):
      self._characterizer = EmbeddingCharacterizer.load(head_path)
  
  # In TwoPhaseOllamaAgent._characterize(reasoning_text):
  if self._characterizer:
      return self._characterizer.predict(reasoning_text)   # ~35ms, no LLM
  return None  # falls back to LLM path
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

# ── Data loading ──────────────────────────────────────────────────────────────

@dataclass
class AgentSample:
    """One (reasoning_text, signals) training example from a sweep result file."""
    source_file: str
    agent_id: str
    round_num: int
    explanation: str        # Phase 1 reasoning text (X)
    confidence: float       # y[0]
    polarity: int           # y[1] in {-1, 0, 1}
    generative_strength: float  # y[2]
    inhibitory_pressure: float  # y[3]

def _extract_samples_from_result(path: Path) -> list[AgentSample]:
    """Parse one sweep result JSON file and return all non-abstention samples."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    samples: list[AgentSample] = []
    for rnd in data.get("all_rounds", []):
        for resp in rnd.get("responses", []):
            explanation = resp.get("explanation", "").strip()
            # Filter abstentions
            if not explanation or "(abstention)" in explanation.lower():
                continue

            sig = resp.get("signal", {})
            confidence = float(sig.get("confidence", 0.0))
            if confidence <= 0.0:
                continue  # zero-confidence = abstention signal

            polarity = int(sig.get("polarity", 0))
            if polarity not in (-1, 0, 1):
                polarity = 0

            g_str = float(sig.get("generative_strength", 0.0))
            i_prs = float(sig.get("inhibitory_pressure", 0.0))

            samples.append(AgentSample(
                source_file=path.name,
                agent_id=resp.get("agent_id", "?"),
                round_num=resp.get("round", 0),
                explanation=explanation,
                confidence=max(0.0, min(1.0, confidence)),
                polarity=polarity,
                generative_strength=max(0.0, min(1.0, g_str)),
                inhibitory_pressure=max(0.0, min(1.0, i_prs)),
            ))
    return samples

def load_sweep_data(sweep_dirs: list[Path]) -> list[AgentSample]:
    """
    Load all agent response samples from sweep result directories.
    Accepts both sweep_results_models/ and sweep_results_logic/.
    """
    all_samples: list[AgentSample] = []
    for sweep_dir in sweep_dirs:
        if not sweep_dir.exists():
            print(f"  [WARN] Sweep dir not found: {sweep_dir}")
            continue
        json_files = sorted(sweep_dir.glob("*.json"))
        print(f"  Loading {len(json_files)} files from {sweep_dir.name} …")
        for path in json_files:
            all_samples.extend(_extract_samples_from_result(path))

    print(f"  Total samples loaded: {len(all_samples)}")
    return all_samples

# ── Embedding (shared with phi_probe) ────────────────────────────────────────

def embed_texts(
    texts: list[str],
    model: str = "nomic-embed-text",
    host: str = "http://127.0.0.1:11434",
    batch_size: int = 8,
    timeout_s: float = 30.0,
    verbose: bool = True,
) -> list[list[float]]:
    all_vectors: list[list[float]] = []
    n = len(texts)
    for i in range(0, n, batch_size):
        batch = texts[i : i + batch_size]
        if verbose and i % (batch_size * 4) == 0:
            print(f"  Embedding {i+1}–{min(i+batch_size, n)}/{n} …", end="\r", flush=True)
        resp = requests.post(
            f"{host}/api/embed",
            json={"model": model, "input": batch},
            timeout=timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        all_vectors.extend([list(map(float, v)) for v in data.get("embeddings", [])])
    if verbose:
        print(f"  Embedded {n} texts.                              ")
    return all_vectors

# ── Training ──────────────────────────────────────────────────────────────────

@dataclass
class HeadEvalResult:
    n_train: int
    n_test: int
    dim: int
    embedding_model: str

    confidence_mae: float | None = None
    gstr_mae: float | None = None
    iprs_mae: float | None = None
    polarity_accuracy: float | None = None
    polarity_majority: float | None = None

    messages: list[str] = field(default_factory=list)
    model_path: Path | None = None

    def report(self) -> str:
        lines = [
            "╔══════════════════════════════════════════════════════╗",
            "║     Signal Head: Embedding → DASP Signals (Step 2)   ║",
            "╚══════════════════════════════════════════════════════╝",
            f"  Train / Test   : {self.n_train} / {self.n_test}",
            f"  Embedding      : {self.embedding_model}  ({self.dim} dims)",
            "",
        ]
        target = "✓ TARGET" if self.confidence_mae is not None and self.confidence_mae < 0.08 else "✗"
        if self.confidence_mae is not None:
            lines += [
                "  ── Regression targets (MAE) ──────────────────────────",
                f"  confidence      : {self.confidence_mae:.4f}  [{target}]",
                f"  gen_strength    : {self.gstr_mae:.4f}",
                f"  inh_pressure    : {self.iprs_mae:.4f}",
                "",
            ]
        if self.polarity_accuracy is not None:
            gain = self.polarity_accuracy - self.polarity_majority
            signal = "✓ USEFUL" if gain > 0.05 else "✗ MARGINAL"
            lines += [
                "  ── Polarity classifier (accuracy) ────────────────────",
                f"  Accuracy        : {self.polarity_accuracy:.3f}  [{signal}]",
                f"  Majority base   : {self.polarity_majority:.3f}",
                f"  Gain            : {gain:+.3f}",
                "",
            ]
        if self.model_path:
            lines.append(f"  Saved           : {self.model_path}")
        for msg in self.messages:
            lines.append(f"  NOTE: {msg}")
        lines.append("")
        return "\n".join(lines)

def train_head(
    samples: list[AgentSample],
    embedding_model: str,
    ollama_host: str,
    test_frac: float = 0.2,
    save_path: Path | None = None,
    verbose: bool = True,
) -> HeadEvalResult:
    """Train the four signal head estimators and optionally save them."""
    if len(samples) < 20:
        result = HeadEvalResult(
            n_train=0, n_test=0, dim=0, embedding_model=embedding_model,
            messages=[f"Only {len(samples)} samples — need ≥20. Run more sweeps."],
        )
        return result

    if verbose:
        print(f"Embedding {len(samples)} reasoning texts …")
    t0 = time.monotonic()
    vectors = embed_texts(
        [s.explanation for s in samples],
        model=embedding_model,
        host=ollama_host,
        verbose=verbose,
    )
    dim = len(vectors[0])
    if verbose:
        print(f"  Embedding done in {time.monotonic()-t0:.1f}s")

    try:
        from sklearn.linear_model import Ridge, LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return HeadEvalResult(
            n_train=0, n_test=0, dim=dim, embedding_model=embedding_model,
            messages=["sklearn not installed. pip install scikit-learn"],
        )

    X = vectors
    y_conf  = [s.confidence           for s in samples]
    y_pol   = [s.polarity             for s in samples]
    y_gstr  = [s.generative_strength  for s in samples]
    y_iprs  = [s.inhibitory_pressure  for s in samples]

    X_tr, X_te, \
    yc_tr, yc_te, \
    yp_tr, yp_te, \
    yg_tr, yg_te, \
    yi_tr, yi_te = train_test_split(
        X, y_conf, y_pol, y_gstr, y_iprs,
        test_size=test_frac, random_state=42,
    )

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # Confidence
    ridge_conf = Ridge(alpha=1.0)
    ridge_conf.fit(X_tr_s, yc_tr)
    conf_pred = ridge_conf.predict(X_te_s)
    conf_mae = sum(abs(a - b) for a, b in zip(yc_te, conf_pred)) / len(yc_te)

    # Generative strength
    ridge_gstr = Ridge(alpha=1.0)
    ridge_gstr.fit(X_tr_s, yg_tr)
    gstr_pred = ridge_gstr.predict(X_te_s)
    gstr_mae = sum(abs(a - b) for a, b in zip(yg_te, gstr_pred)) / len(yg_te)

    # Inhibitory pressure
    ridge_iprs = Ridge(alpha=1.0)
    ridge_iprs.fit(X_tr_s, yi_tr)
    iprs_pred = ridge_iprs.predict(X_te_s)
    iprs_mae = sum(abs(a - b) for a, b in zip(yi_te, iprs_pred)) / len(yi_te)

    # Polarity (multi-class)
    clf_pol = LogisticRegression(C=0.5, max_iter=2000, random_state=42)
    clf_pol.fit(X_tr_s, yp_tr)
    pol_pred = clf_pol.predict(X_te_s)
    pol_acc = sum(a == b for a, b in zip(yp_te, pol_pred)) / len(yp_te)
    from collections import Counter
    pol_majority = max(Counter(yp_te).values()) / len(yp_te)

    eval_result = HeadEvalResult(
        n_train=len(X_tr),
        n_test=len(X_te),
        dim=dim,
        embedding_model=embedding_model,
        confidence_mae=round(conf_mae, 4),
        gstr_mae=round(gstr_mae, 4),
        iprs_mae=round(iprs_mae, 4),
        polarity_accuracy=round(pol_acc, 4),
        polarity_majority=round(pol_majority, 4),
    )

    # Save — re-train on all data
    if save_path is not None:
        X_all_s = scaler.fit_transform(X)
        for model, y in [(ridge_conf, y_conf), (ridge_gstr, y_gstr), (ridge_iprs, y_iprs)]:
            model.fit(X_all_s, y)
        clf_pol.fit(X_all_s, y_pol)

        artefact = {
            "version": "1.0",
            "embedding_model": embedding_model,
            "dim": dim,
            "scaler": scaler,
            "ridge_confidence": ridge_conf,
            "ridge_gstr": ridge_gstr,
            "ridge_iprs": ridge_iprs,
            "clf_polarity": clf_pol,
            "eval": {
                "confidence_mae": eval_result.confidence_mae,
                "gstr_mae": eval_result.gstr_mae,
                "iprs_mae": eval_result.iprs_mae,
                "polarity_accuracy": eval_result.polarity_accuracy,
            },
        }
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as fh:
            pickle.dump(artefact, fh)
        eval_result.model_path = save_path

    return eval_result

# ── EmbeddingCharacterizer — drop-in for TwoPhaseOllamaAgent Phase 2 ─────────

class EmbeddingCharacterizer:
    """
    Drop-in replacement for the LLM-based Characterizer (DASP §4.3 Pattern 1).

    The embedding is computed via Ollama (same endpoint as IEP-A3 snapshots).
    The four signal values are produced by sklearn regressors in <1ms after
    the embedding returns.

    Latency: ~35ms (embedding only) vs ~2700ms (full LLM call).

    DASP §4.3 compliance: the serialization boundary is preserved — the
    Reasoning Node output (free-form text) flows into an embedding, and
    the sklearn models produce the schema-structured signals deterministically.
    No LLM writes a hash, UUID, or round number.
    """

    def __init__(self, artefact: dict, ollama_host: str = "http://127.0.0.1:11434") -> None:
        self._model   = artefact["embedding_model"]
        self._dim     = artefact["dim"]
        self._scaler  = artefact["scaler"]
        self._ridge_c = artefact["ridge_confidence"]
        self._ridge_g = artefact["ridge_gstr"]
        self._ridge_i = artefact["ridge_iprs"]
        self._clf_pol = artefact["clf_polarity"]
        self._host    = ollama_host

    @classmethod
    def load(cls, path: Path, ollama_host: str = "http://127.0.0.1:11434") -> "EmbeddingCharacterizer":
        with open(path, "rb") as fh:
            artefact = pickle.load(fh)
        return cls(artefact, ollama_host)

    @classmethod
    def available(cls, path: Path | None) -> bool:
        return path is not None and path.exists()

    def predict(
        self,
        reasoning_text: str,
        timeout_s: float = 10.0,
    ) -> dict[str, float | int] | None:
        """
        Embed reasoning_text and regress to four signal values.

        Returns a dict matching the emit_signal tool schema:
          confidence, polarity, generative_strength, inhibitory_pressure

        Returns None on embedding failure (caller should fall back to LLM path).
        """
        try:
            resp = requests.post(
                f"{self._host}/api/embed",
                json={"model": self._model, "input": reasoning_text},
                timeout=timeout_s,
            )
            resp.raise_for_status()
            embeddings = resp.json().get("embeddings", [])
            if not embeddings:
                return None
            vec = embeddings[0]
        except Exception as exc:
            print(f"[SignalHead] Embedding failed: {exc} — falling back to LLM")
            return None

        if len(vec) != self._dim:
            print(
                f"[SignalHead] Dim mismatch: expected {self._dim}, got {len(vec)} — "
                "falling back to LLM"
            )
            return None

        X = self._scaler.transform([vec])

        confidence = float(self._ridge_c.predict(X)[0])
        g_str      = float(self._ridge_g.predict(X)[0])
        i_prs      = float(self._ridge_i.predict(X)[0])
        polarity   = int(self._clf_pol.predict(X)[0])

        # Clamp to valid ranges
        confidence = max(0.0, min(1.0, confidence))
        g_str      = max(0.0, min(1.0, g_str))
        i_prs      = max(0.0, min(1.0, i_prs))
        if polarity not in (-1, 0, 1):
            polarity = 0

        return {
            "confidence":          round(confidence, 4),
            "polarity":            polarity,
            "generative_strength": round(g_str, 4),
            "inhibitory_pressure": round(i_prs, 4),
        }

# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(prog="signal_head")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # train
    p_train = sub.add_parser("train", help="Train signal head from sweep data")
    p_train.add_argument("--sweep-dir", type=Path, action="append", dest="sweep_dirs",
                         default=None, help="Sweep result directory (may repeat)")
    p_train.add_argument("--save", type=Path, default=Path("data/signal_head.pkl"))
    p_train.add_argument("--model", default="nomic-embed-text")
    p_train.add_argument("--host", default="http://127.0.0.1:11434")
    p_train.add_argument("--test-frac", type=float, default=0.2)

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate a saved signal head")
    p_eval.add_argument("--sweep-dir", type=Path, action="append", dest="sweep_dirs", default=None)
    p_eval.add_argument("--load", type=Path, required=True)
    p_eval.add_argument("--model", default="nomic-embed-text")
    p_eval.add_argument("--host", default="http://127.0.0.1:11434")

    # infer
    p_infer = sub.add_parser("infer", help="Live inference on one text")
    p_infer.add_argument("--load", type=Path, required=True)
    p_infer.add_argument("--text", type=str, required=True)
    p_infer.add_argument("--host", default="http://127.0.0.1:11434")

    args = parser.parse_args()

    if args.cmd in ("train", "eval"):
        sweep_dirs = args.sweep_dirs or [
            Path("src/effector/sweep_results_models"),
            Path("src/effector/sweep_results_logic"),
        ]
        samples = load_sweep_data(sweep_dirs)
        if not samples:
            print("No samples found. Check sweep directories.")
            return

    if args.cmd == "train":
        result = train_head(
            samples, args.model, args.host, args.test_frac, args.save, verbose=True
        )
        print(result.report())

    elif args.cmd == "eval":
        with open(args.load, "rb") as fh:
            artefact = pickle.load(fh)
        result = train_head(
            samples,
            artefact["embedding_model"],
            args.host,
            test_frac=0.5,  # larger test set for evaluation
            save_path=None,
            verbose=True,
        )
        print(result.report())

    elif args.cmd == "infer":
        if not args.load.exists():
            print(f"Model file not found: {args.load}")
            return
        char = EmbeddingCharacterizer.load(args.load, args.host)
        t0 = time.monotonic()
        signals = char.predict(args.text)
        elapsed = (time.monotonic() - t0) * 1000
        if signals:
            print(f"\nSignals ({elapsed:.0f}ms):")
            for k, v in signals.items():
                print(f"  {k:26s}: {v}")
        else:
            print("Inference failed — check Ollama connectivity.")

if __name__ == "__main__":
    main()
