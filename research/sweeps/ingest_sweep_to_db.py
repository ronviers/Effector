"""
ingest_sweep_to_db.py
=====================
Ingests sweep result JSON files into a SQLite database designed for
exploration in the VSCode SQLite extension.

Usage:
    python ingest_sweep_to_db.py                        # ingest all JSONs in sweep_obs_results\
    python ingest_sweep_to_db.py path/to/result.json    # ingest one file

Output:
    data/sweep_analysis.db

Open in VSCode: install "SQLite Viewer" or "SQLite" extension, then open
the .db file directly or use Ctrl+Shift+P → "SQLite: Open Database".

Useful queries to start with are printed at the end.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

DB_PATH   = Path(__file__).parent / "data" / "sweep_analysis.db"
SWEEP_DIR = Path(__file__).parent / "sweep_obs_results"

# ── Schema ────────────────────────────────────────────────────────────────────

SCHEMA = """
-- One row per scenario run
CREATE TABLE IF NOT EXISTS sweep_runs (
    run_id              TEXT PRIMARY KEY,   -- scenario_id + timestamp
    ingest_time         TEXT NOT NULL,
    scenario_id         TEXT NOT NULL,
    scenario_label      TEXT NOT NULL,
    source_file         TEXT NOT NULL,
    terminated_reason   TEXT,
    consensus_score     REAL,
    rounds              INTEGER,
    elapsed_s           REAL,
    tier2_injected      INTEGER,            -- 0/1 boolean
    winning_hypothesis  TEXT,
    final_answer        TEXT
);

-- One row per agent response (across all rounds of all runs)
CREATE TABLE IF NOT EXISTS agent_responses (
    response_id         TEXT PRIMARY KEY,   -- run_id + round + agent_id
    run_id              TEXT NOT NULL REFERENCES sweep_runs(run_id),
    scenario_id         TEXT NOT NULL,
    scenario_label      TEXT NOT NULL,
    round_num           INTEGER NOT NULL,
    agent_id            TEXT NOT NULL,
    polarity            INTEGER,            -- -1, 0, 1
    polarity_label      TEXT,               -- INHIBITORY / NEUTRAL / GENERATIVE
    confidence          REAL,
    g_str               REAL,               -- generative_strength
    i_prs               REAL,               -- inhibitory_pressure
    signal_sum          REAL,               -- g_str + i_prs (total signal energy)
    signal_ratio        REAL,               -- g_str / (i_prs + 0.001) — skew indicator
    answer_preview      TEXT,               -- first 200 chars
    reasoning_preview   TEXT,               -- first 400 chars
    reasoning_length    INTEGER,            -- full reasoning char count
    act_pos_kw          INTEGER,            -- action-positive keyword count
    act_neg_kw          INTEGER,            -- action-negative keyword count
    kw_skew             REAL,               -- act_pos / (act_pos + act_neg + 1)
    alignment_ok        INTEGER,            -- 1 if signal internally consistent
    alignment_issues    TEXT                -- description of any issues
);

-- One row per hypothesis per run (the signal manifold summary)
CREATE TABLE IF NOT EXISTS signal_manifolds (
    manifold_id         TEXT PRIMARY KEY,
    run_id              TEXT NOT NULL REFERENCES sweep_runs(run_id),
    scenario_id         TEXT NOT NULL,
    hypothesis_id       TEXT NOT NULL,
    s_g                 REAL,
    s_i                 REAL,
    s_net               REAL
);

-- Flat view for quick cross-scenario comparison
CREATE VIEW IF NOT EXISTS v_signal_audit AS
SELECT
    r.scenario_label,
    ar.round_num,
    ar.agent_id,
    ar.polarity_label,
    ar.confidence,
    ar.g_str,
    ar.i_prs,
    ar.signal_ratio,
    ar.act_pos_kw,
    ar.act_neg_kw,
    ar.kw_skew,
    CASE WHEN ar.alignment_ok = 1 THEN 'OK' ELSE ar.alignment_issues END AS alignment,
    ar.answer_preview
FROM agent_responses ar
JOIN sweep_runs r USING (run_id)
ORDER BY r.scenario_id, ar.round_num, ar.agent_id;

-- View: are reasoning keywords predicting polarity correctly?
CREATE VIEW IF NOT EXISTS v_characterizer_faithfulness AS
SELECT
    scenario_label,
    round_num,
    agent_id,
    polarity_label                              AS assigned_polarity,
    kw_skew                                     AS reasoning_skew,   -- >0.5 = net positive
    CASE
        WHEN kw_skew > 0.6  AND polarity_label = 'GENERATIVE'  THEN 'consistent'
        WHEN kw_skew < 0.4  AND polarity_label = 'INHIBITORY'  THEN 'consistent'
        WHEN kw_skew BETWEEN 0.4 AND 0.6                        THEN 'ambiguous'
        ELSE 'MISMATCH'
    END                                         AS faithfulness,
    confidence,
    g_str,
    i_prs,
    signal_ratio
FROM v_signal_audit;

-- View: scenario-level signal variance (low variance = characterizer collapse)
CREATE VIEW IF NOT EXISTS v_scenario_variance AS
SELECT
    scenario_label,
    COUNT(*)                        AS n_responses,
    AVG(confidence)                 AS avg_conf,
    AVG(g_str)                      AS avg_g_str,
    AVG(i_prs)                      AS avg_i_prs,
    MIN(polarity)                   AS min_polarity,
    MAX(polarity)                   AS max_polarity,
    MAX(polarity) - MIN(polarity)   AS polarity_range,  -- 0 = total collapse
    AVG(signal_ratio)               AS avg_signal_ratio,
    AVG(kw_skew)                    AS avg_kw_skew
FROM v_signal_audit
JOIN agent_responses ar USING (scenario_label, round_num, agent_id)
GROUP BY scenario_label;
"""

# ── Keyword analysis ──────────────────────────────────────────────────────────

_POS_WORDS = [
    "should", "recommend", "propose", "act", "intervene",
    "offer", "spawn", "petition", "warm", "cozy", "comfort",
    "proceed", "deploy", "enable", "install", "apply",
]
_NEG_WORDS = [
    "should not", "risk", "disrupt", "violate", "consent",
    "hesitate", "hold", "wait", "uncertain", "concern",
    "caution", "danger", "avoid", "refrain", "question",
]

def _kw_counts(text: str) -> tuple[int, int]:
    t = text.lower()
    pos = sum(t.count(w) for w in _POS_WORDS)
    neg = sum(t.count(w) for w in _NEG_WORDS)
    return pos, neg

# ── Alignment check ───────────────────────────────────────────────────────────

def _alignment(polarity: int, g: float, i: float) -> tuple[int, str]:
    issues = []
    if polarity == 1 and i > g:
        issues.append("GENERATIVE but i_prs > g_str")
    if polarity == -1 and g > i:
        issues.append("INHIBITORY but g_str > i_prs")
    if polarity == 0 and (g > 0.6 or i > 0.6):
        issues.append("NEUTRAL with strong signal values")
    if polarity != 0 and abs(g - i) < 0.05:
        issues.append("non-neutral but g/i nearly equal")
    # Flag the suspicious i_prs=0.20 default
    if abs(i - 0.20) < 0.001 and polarity == 1:
        issues.append("i_prs=0.20 (possible characterizer default)")
    return (1 if not issues else 0), "; ".join(issues)

# ── Ingestion ─────────────────────────────────────────────────────────────────

def _polarity_label(p: int) -> str:
    return {1: "GENERATIVE", 0: "NEUTRAL", -1: "INHIBITORY"}.get(p, "UNKNOWN")

def ingest_file(path: Path, conn: sqlite3.Connection, scenario_overrides: dict | None = None) -> str:
    """Ingest one result JSON file. Returns the run_id."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc).isoformat()

    # Derive scenario_id from filename: sterile_wasteland_20260403_135543.json
    parts = path.stem.split("_")
    # Find the timestamp (8-digit date part)
    ts_idx = next(
        (i for i, p in enumerate(parts) if len(p) == 8 and p.isdigit()),
        len(parts),
    )
    scenario_id = "_".join(parts[:ts_idx]) if ts_idx else path.stem
    file_ts     = "_".join(parts[ts_idx:]) if ts_idx < len(parts) else "unknown"

    if scenario_overrides and scenario_id in scenario_overrides:
        scenario_label = scenario_overrides[scenario_id]
    else:
        scenario_label = scenario_id.replace("_", " ").title()

    run_id = f"{scenario_id}_{file_ts}"

    # ── sweep_runs ────────────────────────────────────────────────────────
    conn.execute(
        """
        INSERT OR REPLACE INTO sweep_runs
            (run_id, ingest_time, scenario_id, scenario_label, source_file,
             terminated_reason, consensus_score, rounds, elapsed_s,
             tier2_injected, winning_hypothesis, final_answer)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            run_id, now, scenario_id, scenario_label, path.name,
            raw.get("terminated_reason"),
            raw.get("consensus_score"),
            raw.get("rounds"),
            raw.get("_elapsed_s"),
            1 if raw.get("tier2_injected") else 0,
            raw.get("winning_hypothesis"),
            (raw.get("final_answer") or "")[:500],
        ),
    )

    # ── agent_responses (from all_rounds) ─────────────────────────────────
    for rnd in raw.get("all_rounds", []):
        rnd_num = rnd.get("round", 0)
        for resp in rnd.get("responses", []):
            agent_id = resp.get("agent_id", "unknown")
            sig      = resp.get("signal", {})
            pol      = sig.get("polarity", 0)
            conf     = sig.get("confidence", 0.0)
            g        = sig.get("generative_strength", 0.0)
            i        = sig.get("inhibitory_pressure", 0.0)
            answer   = (resp.get("answer") or "")[:200]
            reason   = (resp.get("explanation") or "")
            pos_kw, neg_kw = _kw_counts(reason)
            kw_skew  = pos_kw / (pos_kw + neg_kw + 1)
            ok, issues = _alignment(pol, g, i)
            ratio    = g / (i + 0.001)

            response_id = f"{run_id}__r{rnd_num}__{agent_id}"
            conn.execute(
                """
                INSERT OR REPLACE INTO agent_responses
                    (response_id, run_id, scenario_id, scenario_label, round_num,
                     agent_id, polarity, polarity_label, confidence,
                     g_str, i_prs, signal_sum, signal_ratio,
                     answer_preview, reasoning_preview, reasoning_length,
                     act_pos_kw, act_neg_kw, kw_skew,
                     alignment_ok, alignment_issues)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    response_id, run_id, scenario_id, scenario_label, rnd_num,
                    agent_id, pol, _polarity_label(pol), conf,
                    g, i, round(g + i, 4), round(ratio, 4),
                    answer, reason[:400], len(reason),
                    pos_kw, neg_kw, round(kw_skew, 4),
                    ok, issues or None,
                ),
            )

    # ── signal_manifolds ──────────────────────────────────────────────────
    for hid, acc in raw.get("signal_manifold", {}).items():
        conn.execute(
            """
            INSERT OR REPLACE INTO signal_manifolds
                (manifold_id, run_id, scenario_id, hypothesis_id, s_g, s_i, s_net)
            VALUES (?,?,?,?,?,?,?)
            """,
            (
                f"{run_id}__{hid}", run_id, scenario_id, hid,
                acc.get("S_g"), acc.get("S_i"), acc.get("S_net"),
            ),
        )

    conn.commit()
    return run_id

# ── Useful queries to print ───────────────────────────────────────────────────

STARTER_QUERIES = """
-- Paste these into the VSCode SQLite query panel to get started.

-- 1. Signal audit: every response, every scenario
SELECT * FROM v_signal_audit;

-- 2. Characterizer faithfulness: does the polarity match the reasoning?
SELECT * FROM v_characterizer_faithfulness ORDER BY faithfulness DESC;

-- 3. Scenario-level variance (low polarity_range = signal collapse)
SELECT * FROM v_scenario_variance;

-- 4. Find all responses where i_prs = 0.20 exactly (the suspect default)
SELECT scenario_label, round_num, agent_id, confidence, g_str, i_prs, alignment_issues
FROM agent_responses
WHERE ABS(i_prs - 0.20) < 0.005;

-- 5. Compare reasoning keyword skew vs assigned polarity
SELECT scenario_label, round_num, agent_id,
       polarity_label, kw_skew,
       CASE WHEN kw_skew > 0.6 AND polarity_label != 'GENERATIVE' THEN '*** MISMATCH'
            WHEN kw_skew < 0.4 AND polarity_label != 'INHIBITORY' THEN '*** MISMATCH'
            ELSE 'ok' END AS verdict
FROM agent_responses
ORDER BY scenario_label, round_num;

-- 6. Signal energy by scenario (g_str + i_prs; should vary across scenarios)
SELECT scenario_label, AVG(signal_sum), MIN(signal_sum), MAX(signal_sum)
FROM agent_responses GROUP BY scenario_label;

-- 7. The full reasoning text for any response (replace the scenario filter)
SELECT agent_id, round_num, polarity_label, reasoning_preview
FROM agent_responses
WHERE scenario_label LIKE '%Deadline%'
ORDER BY round_num;
"""

# ── Main ──────────────────────────────────────────────────────────────────────

SCENARIO_LABELS = {
    "sterile_wasteland":  "The Sterile Wasteland",
    "deadline_state":     "The Deadline State",
    "receptive_hour":     "The Receptive Hour",
    "background_parasite": "The Background Parasite",
}

def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript(SCHEMA)

    # Collect files to ingest
    if len(sys.argv) > 1:
        files = [Path(p) for p in sys.argv[1:]]
    else:
        files = sorted(SWEEP_DIR.glob("*.json"))

    if not files:
        print(f"No JSON files found in {SWEEP_DIR}")
        print("Run sweep_os_observations.py first, then re-run this script.")
        return

    ingested = 0
    for f in files:
        try:
            run_id = ingest_file(f, conn, SCENARIO_LABELS)
            print(f"  ✓  {f.name}  →  {run_id}")
            ingested += 1
        except Exception as exc:
            print(f"  ✗  {f.name}: {exc}")

    conn.close()
    print(f"\n{ingested} file(s) ingested → {DB_PATH}")
    print(f"\nOpen in VSCode: right-click {DB_PATH.name} → Open With → SQLite Viewer")
    print("\n" + "─" * 60)
    print("STARTER QUERIES")
    print("─" * 60)
    print(STARTER_QUERIES)

if __name__ == "__main__":
    main()
