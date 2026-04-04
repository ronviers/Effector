"""
phi_probe.py — Step 1: Cultivation Log Forward-Model Probe
===========================================================
Answers the question: does the geometry of a WEAVE synthesis proposal already
predict whether it will over-deliver on θ?

Data source: mcp/cultivation_log.jsonl
  64 ACK sessions, each with:
    reasoning        — the DASP final_answer (WEAVE argument for the synthesis)
    forecast         — {"snug_prophecy": float}  →  theta_avg = (prophecy-8)/28
    commit_result    — {"outcome": "exceeded_forecast"|"fell_short", ...}

Two regression tasks:
  1. snug_prophecy regression   (Ridge)   R² target > 0.25
  2. outcome classification     (Logistic) accuracy target > 60%

If R² > 0.30, the embedding geometry contains a usable forward model signal —
the same geometry can be used in the signal head (Step 2) to replace the
second LLM characterizer call entirely.

Usage
-----
  python src/effector/intention/phi_probe.py --log mcp/cultivation_log.jsonl
  python src/effector/intention/phi_probe.py --log mcp/cultivation_log.jsonl --save
  python src/effector/intention/phi_probe.py --log mcp/cultivation_log.jsonl --plot
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

# ── Embedding helper ──────────────────────────────────────────────────────────

def embed_texts(
    texts: list[str],
    model: str = "nomic-embed-text",
    host: str = "http://127.0.0.1:11434",
    batch_size: int = 8,
    timeout_s: float = 30.0,
    verbose: bool = True,
) -> list[list[float]]:
    """
    Embed a list of texts via Ollama /api/embed.
    Returns a list of float vectors; raises on unrecoverable failure.

    Batches to avoid hitting Ollama's input size limits.
    """
    all_vectors: list[list[float]] = []
    n = len(texts)
    for i in range(0, n, batch_size):
        batch = texts[i : i + batch_size]
        if verbose:
            print(f"  Embedding {i+1}–{min(i+batch_size, n)}/{n} ...", end="\r", flush=True)
        try:
            resp = requests.post(
                f"{host}/api/embed",
                json={"model": model, "input": batch},
                timeout=timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings = data.get("embeddings", [])
            if len(embeddings) != len(batch):
                raise ValueError(
                    f"Expected {len(batch)} embeddings, got {len(embeddings)}"
                )
            all_vectors.extend([list(map(float, v)) for v in embeddings])
        except Exception as exc:
            raise RuntimeError(
                f"Embedding batch {i}–{i+batch_size} failed: {exc}\n"
                f"  Model: {model!r}  Host: {host}\n"
                f"  Pull with: ollama pull {model}"
            ) from exc
    if verbose:
        print(f"  Embedded {n} texts ({len(all_vectors[0])} dims each)      ")
    return all_vectors

# ── Data loading ──────────────────────────────────────────────────────────────

@dataclass
class CultivationEntry:
    """One row from cultivation_log.jsonl."""
    timestamp: str
    symbols: list[str]
    reasoning: str          # WEAVE proposal text (embed this)
    snug_prophecy: float    # forecast: max(0, theta_avg)*28+8
    theta_avg: float        # derived: (prophecy-8)/28
    outcome: str | None     # "exceeded_forecast" | "fell_short" | None (NACK)
    verdict: str            # "ACK" | "NACK" | "DRY"

def load_cultivation_log(path: Path) -> list[CultivationEntry]:
    """Load and parse cultivation_log.jsonl. Returns all non-dry-run entries."""
    entries: list[CultivationEntry] = []
    if not path.exists():
        raise FileNotFoundError(
            f"Cultivation log not found: {path}\n"
            "  Run the cultivation loop first: python mcp/cultivation_loop.py --sessions 6"
        )
    with open(path, encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"  [WARN] Line {line_num}: JSON parse error — {exc}")
                continue

            if row.get("dry_run"):
                continue

            reasoning = row.get("reasoning", "").strip()
            # Skip abstentions and cache hits — no useful proposal text
            if not reasoning or reasoning in ("(abstention)", "Reflex Cache Hit"):
                continue

            prophecy = float(row.get("forecast", {}).get("snug_prophecy", 0.0))
            theta_avg = (prophecy - 8.0) / 28.0 if prophecy > 8.0 else 0.0

            outcome: str | None = None
            commit = row.get("commit_result", {})
            if isinstance(commit, dict):
                outcome = commit.get("outcome")

            entries.append(CultivationEntry(
                timestamp=row.get("timestamp", ""),
                symbols=row.get("symbols", []),
                reasoning=reasoning,
                snug_prophecy=prophecy,
                theta_avg=theta_avg,
                outcome=outcome,
                verdict=row.get("verdict", "NACK"),
            ))

    return entries

# ── Regression / classification ───────────────────────────────────────────────

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    ma = math.sqrt(sum(x * x for x in a))
    mb = math.sqrt(sum(x * x for x in b))
    return dot / (ma * mb) if ma and mb else 0.0

@dataclass
class ProbeResult:
    """Results from one probe run."""
    n_total: int
    n_ack: int
    n_with_outcome: int
    dim: int
    embedding_model: str

    # Task 1: snug_prophecy regression
    prophecy_r2: float | None = None
    prophecy_rmse: float | None = None
    prophecy_baseline_rmse: float | None = None  # predict-mean baseline

    # Task 2: outcome classification
    outcome_accuracy: float | None = None
    outcome_majority_accuracy: float | None = None  # majority-class baseline
    outcome_n_exceeded: int = 0
    outcome_n_fell_short: int = 0

    # Geometry sanity check
    intra_class_similarity: float | None = None  # mean cosine(exceeded vectors)
    inter_class_similarity: float | None = None  # mean cosine(exceeded vs fell_short)

    # Saved artefact path
    model_path: Path | None = None

    messages: list[str] = field(default_factory=list)

    def report(self) -> str:
        lines = [
            "╔══════════════════════════════════════════════════════╗",
            "║         φ-Probe: Cultivation Log Forward Model       ║",
            "╚══════════════════════════════════════════════════════╝",
            f"  Entries loaded     : {self.n_total}  (ACK: {self.n_ack})",
            f"  With outcome label : {self.n_with_outcome}",
            f"  Embedding model    : {self.embedding_model}  ({self.dim} dims)",
            "",
        ]
        if self.prophecy_r2 is not None:
            signal = "✓ SIGNAL" if self.prophecy_r2 > 0.25 else "✗ WEAK"
            lines += [
                "  ── Task 1: snug_prophecy regression (Ridge) ───────────",
                f"  R²              : {self.prophecy_r2:+.4f}  [{signal}]",
                f"  RMSE            : {self.prophecy_rmse:.4f}  "
                f"(baseline: {self.prophecy_baseline_rmse:.4f})",
                "",
            ]
        if self.outcome_accuracy is not None:
            signal = "✓ SIGNAL" if self.outcome_accuracy > 0.60 else "✗ WEAK"
            lines += [
                "  ── Task 2: outcome classification (Logistic) ──────────",
                f"  Accuracy        : {self.outcome_accuracy:.3f}  [{signal}]",
                f"  Majority class  : {self.outcome_majority_accuracy:.3f}",
                f"  Exceeded        : {self.outcome_n_exceeded}",
                f"  Fell short      : {self.outcome_n_fell_short}",
                "",
            ]
        if self.intra_class_similarity is not None:
            sep = self.intra_class_similarity - self.inter_class_similarity
            lines += [
                "  ── Geometry ─────────────────────────────────────────────",
                f"  Intra-class cos : {self.intra_class_similarity:.4f}  (exceeded vectors)",
                f"  Inter-class cos : {self.inter_class_similarity:.4f}  (exceeded vs fell_short)",
                f"  Separation      : {sep:+.4f}"
                + ("  [good]" if sep > 0.02 else "  [marginal]"),
                "",
            ]
        for msg in self.messages:
            lines.append(f"  NOTE: {msg}")
        if self.model_path:
            lines.append(f"  Saved           : {self.model_path}")
        lines.append("")
        return "\n".join(lines)

def run_probe(
    log_path: Path,
    embedding_model: str = "nomic-embed-text",
    ollama_host: str = "http://127.0.0.1:11434",
    test_frac: float = 0.2,
    save_path: Path | None = None,
    verbose: bool = True,
) -> ProbeResult:
    """
    Full probe pipeline.  Returns a ProbeResult with R² and accuracy scores.

    Parameters
    ----------
    log_path:        Path to mcp/cultivation_log.jsonl
    embedding_model: Ollama model name for embeddings
    ollama_host:     Ollama host
    test_frac:       Fraction of data to hold out for evaluation
    save_path:       If given, save the trained models as a pickle
    verbose:         Print progress
    """
    # ── Load ──────────────────────────────────────────────────────────────────
    if verbose:
        print(f"Loading cultivation log from {log_path} …")
    entries = load_cultivation_log(log_path)

    result = ProbeResult(
        n_total=len(entries),
        n_ack=sum(1 for e in entries if e.verdict == "ACK"),
        n_with_outcome=sum(1 for e in entries if e.outcome is not None),
        dim=0,
        embedding_model=embedding_model,
    )

    if len(entries) < 10:
        result.messages.append(
            f"Only {len(entries)} usable entries — need ≥10 for meaningful probe. "
            "Run more cultivation sessions."
        )
        return result

    # ── Embed ─────────────────────────────────────────────────────────────────
    if verbose:
        print(f"Embedding {len(entries)} reasoning texts via {embedding_model} …")
    t0 = time.monotonic()
    vectors = embed_texts(
        [e.reasoning for e in entries],
        model=embedding_model,
        host=ollama_host,
        verbose=verbose,
    )
    elapsed = time.monotonic() - t0
    if verbose:
        print(f"  Done in {elapsed:.1f}s")

    result.dim = len(vectors[0])

    # ── Task 1: snug_prophecy regression ─────────────────────────────────────
    try:
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        X = vectors
        y_prophecy = [e.snug_prophecy for e in entries]

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y_prophecy, test_size=test_frac, random_state=42
        )

        scaler_p = StandardScaler()
        X_tr_s = scaler_p.fit_transform(X_tr)
        X_te_s = scaler_p.transform(X_te)

        ridge_p = Ridge(alpha=10.0)
        ridge_p.fit(X_tr_s, y_tr)

        y_pred = ridge_p.predict(X_te_s)
        ss_res = sum((a - b) ** 2 for a, b in zip(y_te, y_pred))
        y_mean = sum(y_te) / len(y_te)
        ss_tot = sum((v - y_mean) ** 2 for v in y_te)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rmse = math.sqrt(ss_res / len(y_te))
        baseline_rmse = math.sqrt(ss_tot / len(y_te))

        result.prophecy_r2 = round(r2, 4)
        result.prophecy_rmse = round(rmse, 4)
        result.prophecy_baseline_rmse = round(baseline_rmse, 4)

    except ImportError:
        result.messages.append("sklearn not installed — skipping regression. pip install scikit-learn")

    # ── Task 2: outcome classification ────────────────────────────────────────
    labeled = [(v, e) for v, e in zip(vectors, entries) if e.outcome is not None]
    if len(labeled) >= 10:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler

            X_lab = [pair[0] for pair in labeled]
            y_lab = [1 if pair[1].outcome == "exceeded_forecast" else 0 for pair in labeled]

            n_exc = sum(y_lab)
            n_fell = len(y_lab) - n_exc
            result.outcome_n_exceeded = n_exc
            result.outcome_n_fell_short = n_fell

            X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
                X_lab, y_lab, test_size=test_frac, random_state=42, stratify=y_lab
            )

            scaler_o = StandardScaler()
            X_tr2_s = scaler_o.fit_transform(X_tr2)
            X_te2_s = scaler_o.transform(X_te2)

            clf = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
            clf.fit(X_tr2_s, y_tr2)

            y_pred2 = clf.predict(X_te2_s)
            acc = sum(a == b for a, b in zip(y_te2, y_pred2)) / len(y_te2)
            majority = max(sum(y_te2), len(y_te2) - sum(y_te2)) / len(y_te2)

            result.outcome_accuracy = round(acc, 4)
            result.outcome_majority_accuracy = round(majority, 4)

        except ImportError:
            pass

        # ── Geometry check ────────────────────────────────────────────────────
        exceeded_vecs = [pair[0] for pair in labeled if pair[1].outcome == "exceeded_forecast"]
        fell_vecs     = [pair[0] for pair in labeled if pair[1].outcome == "fell_short"]

        if len(exceeded_vecs) >= 2 and len(fell_vecs) >= 1:
            # Intra: mean cosine between exceeded vectors (sample up to 50 pairs)
            intra_scores = []
            pairs_tried = 0
            for i in range(len(exceeded_vecs)):
                for j in range(i + 1, len(exceeded_vecs)):
                    if pairs_tried >= 50:
                        break
                    intra_scores.append(_cosine_similarity(exceeded_vecs[i], exceeded_vecs[j]))
                    pairs_tried += 1

            # Inter: mean cosine between exceeded and fell_short
            inter_scores = []
            for a in exceeded_vecs[:10]:
                for b in fell_vecs[:10]:
                    inter_scores.append(_cosine_similarity(a, b))

            result.intra_class_similarity = round(sum(intra_scores) / len(intra_scores), 4)
            result.inter_class_similarity = round(sum(inter_scores) / len(inter_scores), 4)

    else:
        result.messages.append(
            f"Only {len(labeled)} outcome-labeled entries — need ≥10 for outcome probe."
        )

    # ── Save ──────────────────────────────────────────────────────────────────
    if save_path is not None:
        try:
            from sklearn.linear_model import Ridge, LogisticRegression
            from sklearn.preprocessing import StandardScaler

            artefact = {
                "embedding_model": embedding_model,
                "dim": result.dim,
                "prophecy_r2": result.prophecy_r2,
                "outcome_accuracy": result.outcome_accuracy,
            }

            # Re-train on ALL data for the saved model
            X_all = vectors
            y_all_p = [e.snug_prophecy for e in entries]
            sc_all = StandardScaler()
            X_all_s = sc_all.fit_transform(X_all)
            r_all = Ridge(alpha=10.0)
            r_all.fit(X_all_s, y_all_p)
            artefact["prophecy_scaler"] = sc_all
            artefact["prophecy_ridge"] = r_all

            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "wb") as fh:
                pickle.dump(artefact, fh)
            result.model_path = save_path

        except ImportError:
            result.messages.append("sklearn not available — model not saved.")

    return result

# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import sys
    parser = argparse.ArgumentParser(
        prog="phi_probe",
        description="Step 1: Probe cultivation log for θ forward-model signal",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("mcp/cultivation_log.jsonl"),
        help="Path to cultivation_log.jsonl",
    )
    parser.add_argument(
        "--model",
        default="nomic-embed-text",
        help="Ollama embedding model (nomic-embed-text, qwen3:embedding, etc.)",
    )
    parser.add_argument(
        "--host",
        default="http://127.0.0.1:11434",
        help="Ollama host",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        metavar="PATH",
        help="Save trained probe models to PATH (e.g. data/phi_probe.pkl)",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.2,
        help="Fraction held out for evaluation (default 0.2)",
    )
    args = parser.parse_args()

    result = run_probe(
        log_path=args.log,
        embedding_model=args.model,
        ollama_host=args.host,
        test_frac=args.test_frac,
        save_path=args.save,
        verbose=True,
    )
    print()
    print(result.report())

    # Exit 0 if meaningful signal found in either task, else 1
    has_signal = (
        (result.prophecy_r2 is not None and result.prophecy_r2 > 0.25)
        or (result.outcome_accuracy is not None and result.outcome_accuracy > 0.60)
    )
    sys.exit(0 if has_signal else 1)

if __name__ == "__main__":
    main()
