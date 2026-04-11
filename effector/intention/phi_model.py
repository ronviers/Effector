"""
phi_model.py — Step 3: φ-Injection Assessment Model
=====================================================
A φ-metabolizing model that reads incoming communication text and outputs:

  phi_injection  [0, 1]  — Kinetic Fidget the message wants to introduce
  urgency        [0, 1]  — Actual time-sensitivity (separate from disruption)
  reframeable    bool    — Whether the message can be softened on delivery

These three values are the inputs to the Intention Layer decision function.
They are independent of habitat state — habitat state is applied at decision
time in layer.py.

Design
------
φ is not the same as urgency.
  - A newsletter with an exciting headline is high-φ (wants attention) but
    low-urgency (you can read it tomorrow).
  - A "server is down" alert is high-urgency but also high-φ.
  - A friend's slow-reply message is low-φ and low-urgency.
  - A calendar reminder for a meeting in 15 minutes is moderate-φ, high-urgency.

The three outputs let the Intention Layer reason about each dimension
independently:
  high urgency + high φ → PRESENT_NOW (bypass habitat protection)
  high φ + low urgency + high pressure → HOLD + REFRAME on delivery
  low φ → PRESENT_NOW (no habitat disruption)

Labeled dataset
---------------
52 realistic notification/email strings hand-labeled across:
  - Email subject lines
  - Slack messages (first sentence)
  - System alerts
  - Calendar reminders
  - Social/newsletter notifications

Labels are calibrated so that:
  φ ≤ 0.25  "ambient"  — safe to surface anytime
  φ ≤ 0.55  "gentle"   — surface when habitat.pressure < 0.5
  φ ≤ 0.75  "moderate" — surface with softening or 5–15min hold
  φ > 0.75  "sharp"    — hold or reframe; bypass only if urgency > 0.85

Training
--------
Three independent Ridge regressors on the shared embedding.
(We use Ridge even for urgency, treating it as continuous.)
The reframeable classifier uses LogisticRegression.

Usage
-----
  # Train and save
  python src/effector/intention/phi_model.py train --save data/phi_model.pkl

  # Assess text
  python src/effector/intention/phi_model.py assess \\
      --load data/phi_model.pkl \\
      --text "URGENT: Production database unreachable"

  # Show labeled dataset
  python src/effector/intention/phi_model.py dataset
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

# ── Labeled dataset ───────────────────────────────────────────────────────────
# Each entry: (text, phi, urgency, reframeable)
#
# phi:         0.0–1.0  kinetic fidget payload
# urgency:     0.0–1.0  time-sensitivity
# reframeable: True if subject/preview can be softened without losing meaning

_LABELED_DATASET: list[tuple[str, float, float, bool]] = [
    # ── Ambient / low-φ ──────────────────────────────────────────────────────
    ("Your weekly newsletter from The Browser", 0.05, 0.0, True),
    ("Someone liked your photo", 0.08, 0.0, True),
    ("Your order has shipped — arrives Thursday", 0.10, 0.05, True),
    ("Monthly statement available for your account", 0.10, 0.05, True),
    ("New follower on GitHub: @alexchen88", 0.08, 0.0, True),
    ("Your subscription renews in 30 days", 0.12, 0.05, True),
    ("Receipt from Spotify — $9.99", 0.06, 0.0, True),
    ("Reminder: your dentist appointment is next Tuesday", 0.15, 0.10, True),
    ("Weekly digest: 5 new articles based on your interests", 0.07, 0.0, True),
    ("Your backup completed successfully", 0.04, 0.0, False),

    # ── Gentle nudge / low-moderate φ ────────────────────────────────────────
    ("Hey — are you free to grab lunch this week?", 0.25, 0.10, True),
    ("Quick question when you get a chance", 0.28, 0.10, True),
    ("PR #447 is ready for your review", 0.30, 0.25, True),
    ("Your draft has 3 unresolved comments", 0.30, 0.20, True),
    ("Reminder: you have a meeting tomorrow at 10am", 0.28, 0.30, True),
    ("Slack: @sophie left a comment on your doc", 0.25, 0.15, True),
    ("GitHub Actions: build passed ✓", 0.10, 0.05, False),
    ("Sentry: 3 new errors in the last hour (all same type)", 0.35, 0.35, True),
    ("Invoice #1042 from Vercel — $22.00 due in 14 days", 0.15, 0.10, True),
    ("Your trial expires in 7 days", 0.32, 0.20, True),

    # ── Moderate disruption / medium φ ───────────────────────────────────────
    ("Reminder: standup in 30 minutes", 0.48, 0.55, True),
    ("Your PR has a merge conflict", 0.45, 0.40, True),
    ("@david mentioned you in #product: quick take?", 0.50, 0.35, True),
    ("Can you join the call? We're waiting for you", 0.60, 0.70, False),
    ("Client feedback on proposal — several changes requested", 0.55, 0.45, True),
    ("Sentry: error rate spike — 200 errors in 5 minutes", 0.60, 0.65, True),
    ("Payment failed — please update your billing info", 0.58, 0.50, True),
    ("Your meeting has been moved to 2:00pm today", 0.55, 0.65, True),
    ("Manager: can we talk about the Q2 targets this afternoon?", 0.52, 0.50, True),
    ("GitHub: your deployment to staging failed", 0.50, 0.55, True),

    # ── High disruption / high φ ─────────────────────────────────────────────
    ("URGENT: Client is on the phone asking for you", 0.80, 0.90, False),
    ("Alert: CPU usage at 98% for 10 minutes on prod-01", 0.78, 0.85, False),
    ("Your presentation starts in 10 minutes — room B2", 0.82, 0.95, True),
    ("PagerDuty: CRITICAL — checkout service is down", 0.90, 1.00, False),
    ("Boss: We need to talk. Are you available now?", 0.85, 0.80, False),
    ("Database replication lag: 47 seconds and increasing", 0.82, 0.90, False),
    ("Stripe: payment processing errors — 34% failure rate", 0.88, 0.95, False),
    ("OVERDUE: Invoice #991 — 14 days past due", 0.72, 0.75, True),
    ("Security alert: sign-in from new device — Minsk, BY", 0.85, 0.85, False),
    ("Deadline TODAY: proposal must be submitted by 5pm", 0.80, 0.95, False),

    # ── Maximum urgency / emergency ───────────────────────────────────────────
    ("CRITICAL: Production database unreachable — all services down", 0.95, 1.00, False),
    ("911 text: Emergency at 340 Oak Street", 0.99, 1.00, False),
    ("Server breach detected — immediate action required", 0.97, 1.00, False),
    ("Your account has been locked due to suspicious activity", 0.92, 0.95, False),
    ("FIRE ALARM: Evacuate the building immediately", 1.00, 1.00, False),
    ("Call from: Mum (3 missed calls)", 0.88, 0.85, False),

    # ── Edge cases ────────────────────────────────────────────────────────────
    ("You have 0 unread messages", 0.02, 0.0, False),
    ("Disk usage at 95% — /dev/sda1", 0.65, 0.70, True),
    ("Your flight check-in is open", 0.35, 0.45, True),
    ("Notion: @you were added to 'Q3 Planning' workspace", 0.20, 0.05, True),
    ("GitHub Dependabot: 3 security vulnerabilities found", 0.42, 0.40, True),
    ("Slack: 24 unread messages in 7 channels", 0.55, 0.30, True),
]

# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class PhiAssessment:
    """Output of a φ model inference pass."""
    text: str
    phi_injection: float        # [0, 1]
    urgency: float              # [0, 1]
    reframeable: bool
    elapsed_ms: float

    @property
    def phi_class(self) -> str:
        if self.phi_injection <= 0.25: return "ambient"
        if self.phi_injection <= 0.55: return "gentle"
        if self.phi_injection <= 0.75: return "moderate"
        return "sharp"

    @property
    def urgency_class(self) -> str:
        if self.urgency <= 0.30: return "low"
        if self.urgency <= 0.65: return "moderate"
        if self.urgency <= 0.85: return "high"
        return "critical"

    def summary(self) -> str:
        lines = [
            f'  Text      : "{self.text[:70]}{"…" if len(self.text)>70 else ""}"',
            f"  φ         : {self.phi_injection:.3f}  [{self.phi_class}]",
            f"  urgency   : {self.urgency:.3f}  [{self.urgency_class}]",
            f"  reframe   : {'yes' if self.reframeable else 'no'}",
            f"  elapsed   : {self.elapsed_ms:.0f}ms",
        ]
        return "\n".join(lines)

# ── Training ──────────────────────────────────────────────────────────────────

def embed_texts(
    texts: list[str],
    model: str,
    host: str,
    timeout_s: float = 30.0,
) -> list[list[float]]:
    resp = requests.post(
        f"{host}/api/embed",
        json={"model": model, "input": texts},
        timeout=timeout_s,
    )
    resp.raise_for_status()
    return [list(map(float, v)) for v in resp.json().get("embeddings", [])]

def embed_one(text: str, model: str, host: str, timeout_s: float = 10.0) -> list[float] | None:
    try:
        vecs = embed_texts([text], model, host, timeout_s)
        return vecs[0] if vecs else None
    except Exception as exc:
        print(f"[PhiModel] Embedding error: {exc}")
        return None

@dataclass
class TrainResult:
    n: int
    dim: int
    embedding_model: str
    phi_r2: float | None = None
    urgency_r2: float | None = None
    reframe_accuracy: float | None = None
    model_path: Path | None = None
    messages: list[str] = field(default_factory=list)

    def report(self) -> str:
        lines = [
            "╔══════════════════════════════════════════════════════╗",
            "║     φ-Injection Model (Step 3) — Training Report     ║",
            "╚══════════════════════════════════════════════════════╝",
            f"  Dataset size    : {self.n} labeled examples",
            f"  Embedding       : {self.embedding_model}  ({self.dim} dims)",
            "",
        ]
        if self.phi_r2 is not None:
            sig = "✓" if self.phi_r2 > 0.5 else "✗"
            lines.append(f"  φ regression R² : {self.phi_r2:+.4f}  [{sig}]")
        if self.urgency_r2 is not None:
            sig = "✓" if self.urgency_r2 > 0.5 else "✗"
            lines.append(f"  urgency R²      : {self.urgency_r2:+.4f}  [{sig}]")
        if self.reframe_accuracy is not None:
            sig = "✓" if self.reframe_accuracy > 0.70 else "✗"
            lines.append(f"  reframe acc     : {self.reframe_accuracy:.3f}  [{sig}]")
        lines.append("")
        if self.model_path:
            lines.append(f"  Saved           : {self.model_path}")
        for msg in self.messages:
            lines.append(f"  NOTE: {msg}")
        lines.append("")
        return "\n".join(lines)

def train_phi_model(
    embedding_model: str = "nomic-embed-text",
    ollama_host: str = "http://127.0.0.1:11434",
    extra_data: list[tuple[str, float, float, bool]] | None = None,
    save_path: Path | None = None,
    test_frac: float = 0.20,
    verbose: bool = True,
) -> TrainResult:
    """
    Train the φ injection model on the built-in labeled dataset.

    extra_data: additional (text, phi, urgency, reframeable) tuples to
    augment the built-in 52-example dataset.

    The model is compact enough to train in seconds even on CPU.
    """
    dataset = list(_LABELED_DATASET)
    if extra_data:
        dataset.extend(extra_data)

    texts   = [row[0] for row in dataset]
    y_phi   = [row[1] for row in dataset]
    y_urg   = [row[2] for row in dataset]
    y_ref   = [int(row[3]) for row in dataset]

    if verbose:
        print(f"Embedding {len(texts)} labeled notification texts …")

    try:
        vecs = embed_texts(texts, embedding_model, ollama_host)
    except Exception as exc:
        return TrainResult(
            n=len(dataset), dim=0, embedding_model=embedding_model,
            messages=[f"Embedding failed: {exc}"],
        )

    dim = len(vecs[0])
    if verbose:
        print(f"  Got {dim}-dimensional embeddings")

    try:
        from sklearn.linear_model import Ridge, LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        import math
    except ImportError:
        return TrainResult(
            n=len(dataset), dim=dim, embedding_model=embedding_model,
            messages=["sklearn not installed — pip install scikit-learn"],
        )

    X = vecs
    X_tr, X_te, \
    phi_tr, phi_te, \
    urg_tr, urg_te, \
    ref_tr, ref_te = train_test_split(
        X, y_phi, y_urg, y_ref,
        test_size=test_frac, random_state=42,
    )

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # φ regression
    r_phi = Ridge(alpha=0.5)
    r_phi.fit(X_tr_s, phi_tr)
    phi_pred = r_phi.predict(X_te_s)
    ss_res = sum((a - b) ** 2 for a, b in zip(phi_te, phi_pred))
    phi_mean = sum(phi_te) / len(phi_te)
    ss_tot = sum((v - phi_mean) ** 2 for v in phi_te)
    phi_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # urgency regression
    r_urg = Ridge(alpha=0.5)
    r_urg.fit(X_tr_s, urg_tr)
    urg_pred = r_urg.predict(X_te_s)
    ss_res_u = sum((a - b) ** 2 for a, b in zip(urg_te, urg_pred))
    urg_mean = sum(urg_te) / len(urg_te)
    ss_tot_u = sum((v - urg_mean) ** 2 for v in urg_te)
    urg_r2 = 1 - ss_res_u / ss_tot_u if ss_tot_u > 0 else 0.0

    # reframeable classifier
    clf_ref = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf_ref.fit(X_tr_s, ref_tr)
    ref_pred = clf_ref.predict(X_te_s)
    ref_acc = sum(a == b for a, b in zip(ref_te, ref_pred)) / len(ref_te)

    result = TrainResult(
        n=len(dataset),
        dim=dim,
        embedding_model=embedding_model,
        phi_r2=round(phi_r2, 4),
        urgency_r2=round(urg_r2, 4),
        reframe_accuracy=round(ref_acc, 4),
    )

    if save_path is not None:
        # Re-train on full dataset for production model
        X_all_s = scaler.fit_transform(X)
        r_phi.fit(X_all_s, y_phi)
        r_urg.fit(X_all_s, y_urg)
        clf_ref.fit(X_all_s, y_ref)

        artefact = {
            "version":         "1.0",
            "embedding_model": embedding_model,
            "dim":             dim,
            "scaler":          scaler,
            "ridge_phi":       r_phi,
            "ridge_urgency":   r_urg,
            "clf_reframe":     clf_ref,
            "eval": {
                "phi_r2":           result.phi_r2,
                "urgency_r2":       result.urgency_r2,
                "reframe_accuracy": result.reframe_accuracy,
            },
        }
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as fh:
            pickle.dump(artefact, fh)
        result.model_path = save_path
        if verbose:
            print(f"  Saved to {save_path}")

    return result

# ── PhiAssessor — live inference ──────────────────────────────────────────────

class PhiAssessor:
    """
    Live inference engine for a trained φ injection model.

    Instantiated once; reused for every incoming communication event.

    Usage:
        assessor = PhiAssessor.load("data/phi_model.pkl")
        assessment = assessor.assess("URGENT: Production is down")
        print(f"φ={assessment.phi_injection:.2f}  urgency={assessment.urgency:.2f}")
    """

    def __init__(self, artefact: dict, ollama_host: str = "http://127.0.0.1:11434") -> None:
        self._model    = artefact["embedding_model"]
        self._dim      = artefact["dim"]
        self._scaler   = artefact["scaler"]
        self._r_phi    = artefact["ridge_phi"]
        self._r_urg    = artefact["ridge_urgency"]
        self._clf_ref  = artefact["clf_reframe"]
        self._host     = ollama_host

    @classmethod
    def load(cls, path: Path, ollama_host: str = "http://127.0.0.1:11434") -> "PhiAssessor":
        if not path.exists():
            raise FileNotFoundError(
                f"φ model not found: {path}\n"
                "  Train with: python src/effector/intention/phi_model.py train --save data/phi_model.pkl"
            )
        with open(path, "rb") as fh:
            artefact = pickle.load(fh)
        return cls(artefact, ollama_host)

    @classmethod
    def available(cls, path: Path | None) -> bool:
        return path is not None and path.exists()

    def assess(self, text: str, timeout_s: float = 10.0) -> PhiAssessment | None:
        """
        Assess a communication string.
        Returns None on embedding failure (treat as high-φ to be safe).
        """
        t0 = time.monotonic()
        vec = embed_one(text, self._model, self._host, timeout_s)
        if vec is None or len(vec) != self._dim:
            return None

        X = self._scaler.transform([vec])
        phi      = float(max(0.0, min(1.0, self._r_phi.predict(X)[0])))
        urgency  = float(max(0.0, min(1.0, self._r_urg.predict(X)[0])))
        reframe  = bool(self._clf_ref.predict(X)[0])
        elapsed  = (time.monotonic() - t0) * 1000

        return PhiAssessment(
            text=text,
            phi_injection=round(phi, 4),
            urgency=round(urgency, 4),
            reframeable=reframe,
            elapsed_ms=round(elapsed, 1),
        )

# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(prog="phi_model")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train and save the φ model")
    p_train.add_argument("--save", type=Path, default=Path("data/phi_model.pkl"))
    p_train.add_argument("--model", default="nomic-embed-text")
    p_train.add_argument("--host",  default="http://127.0.0.1:11434")

    p_assess = sub.add_parser("assess", help="Assess a text string")
    p_assess.add_argument("--load", type=Path, default=Path("data/phi_model.pkl"))
    p_assess.add_argument("--text", type=str, required=True)
    p_assess.add_argument("--host", default="http://127.0.0.1:11434")

    p_ds = sub.add_parser("dataset", help="Print the labeled dataset")

    args = parser.parse_args()

    if args.cmd == "train":
        result = train_phi_model(
            embedding_model=args.model,
            ollama_host=args.host,
            save_path=args.save,
            verbose=True,
        )
        print()
        print(result.report())

    elif args.cmd == "assess":
        assessor = PhiAssessor.load(args.load, args.host)
        result = assessor.assess(args.text)
        if result:
            print()
            print(result.summary())
        else:
            print("Assessment failed — check Ollama connectivity.")

    elif args.cmd == "dataset":
        print(f"\n{'φ':>5}  {'urgency':>7}  {'reframe':>7}  text")
        print("─" * 72)
        for text, phi, urg, ref in sorted(_LABELED_DATASET, key=lambda r: r[1]):
            print(f"{phi:>5.2f}  {urg:>7.2f}  {'yes' if ref else 'no':>7}  {text[:50]}")
        print(f"\n{len(_LABELED_DATASET)} examples total")

if __name__ == "__main__":
    main()
