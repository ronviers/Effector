"""
rat_store.py — Step 1: SQLite RAT Store

Implements LocalRATStore: a sqlite3-backed store for Reflex Authorization
Tokens (RATs) conforming to IEP §6.2.

Design decisions (from architecture doc, DO NOT RELITIGATE)
-----------------------------------------------------------
- Native sqlite3 only — no external deps, no Docker.
- Atomic execution decrement via a single RETURNING query (no read-check-write).
- RATs persist across host reboots; M4/M5 checks catch stale state naturally.
- Background cleanup thread removes expired rows every `cleanup_interval_s`.
- snapshot_vector stored as JSON text (SQLite has no native float-array type).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Default DB path
# ---------------------------------------------------------------------------

def _default_db_path() -> Path:
    """Place the DB next to this file's package root, inside the project."""
    here = Path(__file__).resolve().parent
    # Walk up to find the project root (contains pyproject.toml) or stay local
    for parent in [here, here.parent, here.parent.parent]:
        if (parent / "pyproject.toml").exists():
            return parent / "data" / "rats.db"
    return here / "rats.db"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CREATE_RATS_TABLE = """
CREATE TABLE IF NOT EXISTS rats (
    rat_id                  TEXT    PRIMARY KEY,
    issued_by_session       TEXT    NOT NULL,
    issued_at               TEXT    NOT NULL,          -- ISO-8601
    rat_ttl_ms              INTEGER NOT NULL,
    ttl_expiry_timestamp    REAL    NOT NULL,          -- UNIX epoch seconds
    rat_min_confidence      REAL    NOT NULL DEFAULT 0.7,
    rat_similarity_threshold REAL   NOT NULL DEFAULT 0.97,
    authorized_actions      TEXT    NOT NULL,          -- JSON list
    issuing_coalition       TEXT    NOT NULL DEFAULT '[]', -- JSON list of agent IDs
    snapshot_hash           TEXT    NOT NULL DEFAULT '',
    snapshot_vector         TEXT    DEFAULT NULL,      -- JSON list[float] or NULL
    embedding_model         TEXT    DEFAULT 'nomic-embed-text',
    executions_remaining    INTEGER NOT NULL DEFAULT -1  -- -1 = unlimited
)
"""

_CREATE_EXPIRY_INDEX = """
CREATE INDEX IF NOT EXISTS idx_rats_expiry
ON rats (ttl_expiry_timestamp)
"""

_CREATE_ACTION_INDEX = """
CREATE INDEX IF NOT EXISTS idx_rats_verb_target
ON rats (rat_id)  -- candidates fetched by rat_id after verb/target match in Python
"""


# ---------------------------------------------------------------------------
# Data class for a loaded RAT
# ---------------------------------------------------------------------------

@dataclass
class RATRecord:
    rat_id: str
    issued_by_session: str
    issued_at: str
    rat_ttl_ms: int
    ttl_expiry_timestamp: float
    rat_min_confidence: float
    rat_similarity_threshold: float
    authorized_actions: list[dict[str, Any]]
    issuing_coalition: list[str]
    snapshot_hash: str
    snapshot_vector: list[float] | None
    embedding_model: str
    executions_remaining: int

    @property
    def is_expired(self) -> bool:
        return time.time() > self.ttl_expiry_timestamp

    def authorizes(self, verb: str, target: str) -> dict[str, Any] | None:
        """
        Return the first authorized_action entry matching verb + target,
        or None if none match.

        Target matching: exact string equality first, then prefix match
        (e.g. authorized target 'desktop.overlay' covers 'desktop.overlay.glimmer').
        """
        verb_upper = verb.upper()
        for action in self.authorized_actions:
            if action.get("verb", "").upper() != verb_upper:
                continue
            auth_target: str = action.get("target", "")
            if auth_target == target:
                return action
            # Prefix match: authorized 'desktop.overlay' covers 'desktop.overlay.glimmer'
            if target.startswith(auth_target + ".") or auth_target == "*":
                return action
        return None


# ---------------------------------------------------------------------------
# LocalRATStore
# ---------------------------------------------------------------------------

class LocalRATStore:
    """
    SQLite-backed store for Reflex Authorization Tokens.

    Thread-safe: SQLite connections are per-thread (check_same_thread=False
    is set; the WAL journal mode serialises writers at the DB level).
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        cleanup_interval_s: float = 60.0,
    ) -> None:
        self._db_path = Path(db_path or _default_db_path())
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._cleanup_interval_s = cleanup_interval_s
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._cleanup_thread: threading.Thread | None = None
        self._stop_cleanup = threading.Event()
        self._initialise()
        self._start_cleanup_thread()

    # ── Private helpers ────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        """Return (and cache) a single connection for the current thread."""
        # Use a module-level thread-local connection for read paths;
        # the write lock ensures serial writes.
        conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            isolation_level=None,  # autocommit; we manage transactions manually
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = self._connect()
        return self._conn

    def _initialise(self) -> None:
        conn = self._connect()
        conn.execute(_CREATE_RATS_TABLE)
        conn.execute(_CREATE_EXPIRY_INDEX)
        conn.execute(_CREATE_ACTION_INDEX)
        conn.commit()
        conn.close()

    def _row_to_record(self, row: sqlite3.Row) -> RATRecord:
        return RATRecord(
            rat_id=row["rat_id"],
            issued_by_session=row["issued_by_session"],
            issued_at=row["issued_at"],
            rat_ttl_ms=row["rat_ttl_ms"],
            ttl_expiry_timestamp=row["ttl_expiry_timestamp"],
            rat_min_confidence=row["rat_min_confidence"],
            rat_similarity_threshold=row["rat_similarity_threshold"],
            authorized_actions=json.loads(row["authorized_actions"] or "[]"),
            issuing_coalition=json.loads(row["issuing_coalition"] or "[]"),
            snapshot_hash=row["snapshot_hash"] or "",
            snapshot_vector=(
                json.loads(row["snapshot_vector"])
                if row["snapshot_vector"]
                else None
            ),
            embedding_model=row["embedding_model"] or "nomic-embed-text",
            executions_remaining=row["executions_remaining"],
        )

    # ── Cleanup ────────────────────────────────────────────────────────────

    def _start_cleanup_thread(self) -> None:
        self._stop_cleanup.clear()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            name="RATStore-Cleanup",
            daemon=True,
        )
        self._cleanup_thread.start()

    def _cleanup_loop(self) -> None:
        while not self._stop_cleanup.wait(timeout=self._cleanup_interval_s):
            try:
                self.purge_expired()
            except Exception as exc:
                print(f"[RATStore] Cleanup error: {exc}")

    def purge_expired(self) -> int:
        """Delete expired RATs. Returns number of rows deleted."""
        now = time.time()
        with self._lock:
            conn = self._get_conn()
            cur = conn.execute(
                "DELETE FROM rats WHERE ttl_expiry_timestamp <= ?", (now,)
            )
            conn.commit()
            return cur.rowcount

    def close(self) -> None:
        """Shut down the cleanup thread and close the DB connection."""
        self._stop_cleanup.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5.0)
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None

    # ── Public CRUD ────────────────────────────────────────────────────────

    def store_rat(self, rat: dict[str, Any]) -> None:
        """
        Persist a RAT conforming to IEP §6.2 schema.

        The caller (typically DASPCoordinator at session end) passes a raw
        RAT dict. We compute `ttl_expiry_timestamp` here from
        `issued_at + rat_ttl_ms`.

        If a RAT with the same `rat_id` already exists it is replaced.
        """
        issued_at_str: str = rat.get("issued_at", datetime.now(timezone.utc).isoformat())
        rat_ttl_ms: int = int(rat.get("rat_ttl_ms", 3_600_000))  # default 1 hour

        try:
            issued_ts = datetime.fromisoformat(issued_at_str).timestamp()
        except Exception:
            issued_ts = time.time()

        ttl_expiry = issued_ts + rat_ttl_ms / 1000.0

        # Determine initial executions_remaining.
        # We use the minimum max_executions across authorized_actions,
        # ignoring -1 (unlimited). If all are -1, store -1.
        actions: list[dict] = rat.get("authorized_actions", [])
        limited = [a["max_executions"] for a in actions if a.get("max_executions", -1) != -1]
        executions_remaining = min(limited) if limited else -1

        vector = rat.get("snapshot_vector")
        vector_json = json.dumps(vector) if vector else None

        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """
                INSERT OR REPLACE INTO rats (
                    rat_id, issued_by_session, issued_at, rat_ttl_ms,
                    ttl_expiry_timestamp, rat_min_confidence,
                    rat_similarity_threshold, authorized_actions,
                    issuing_coalition, snapshot_hash, snapshot_vector,
                    embedding_model, executions_remaining
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rat["rat_id"],
                    rat.get("issued_by_session", ""),
                    issued_at_str,
                    rat_ttl_ms,
                    ttl_expiry,
                    float(rat.get("rat_min_confidence", 0.7)),
                    float(rat.get("rat_similarity_threshold", 0.97)),
                    json.dumps(actions),
                    json.dumps(rat.get("issuing_coalition", [])),
                    rat.get("snapshot_hash", ""),
                    vector_json,
                    rat.get("embedding_model", "nomic-embed-text"),
                    executions_remaining,
                ),
            )
            conn.commit()

    def get_candidate_rats(self, verb: str, target: str) -> list[RATRecord]:
        """
        Fetch all non-expired RATs that *might* authorize (verb, target).

        We load all live RATs and filter in Python because SQLite can't query
        inside the JSON `authorized_actions` blob efficiently. The result set
        is small in practice (typically 0–10 RATs).
        """
        now = time.time()
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                """
                SELECT * FROM rats
                WHERE ttl_expiry_timestamp > ?
                  AND (executions_remaining > 0 OR executions_remaining = -1)
                """,
                (now,),
            ).fetchall()

        candidates: list[RATRecord] = []
        for row in rows:
            record = self._row_to_record(row)
            if record.authorizes(verb, target) is not None:
                candidates.append(record)
        return candidates

    def decrement_and_fetch(self, rat_id: str) -> int | None:
        """
        Atomically decrement executions_remaining for a RAT and return the
        new value. Returns None if the RAT is exhausted or not found.

        Uses a single SQLite query with RETURNING to avoid read-check-write.
        For unlimited RATs (executions_remaining = -1) this is a no-op that
        returns -1 immediately.
        """
        with self._lock:
            conn = self._get_conn()

            # Check if unlimited first (avoid decrementing -1)
            row = conn.execute(
                "SELECT executions_remaining FROM rats WHERE rat_id = ?",
                (rat_id,),
            ).fetchone()

            if row is None:
                return None

            if row["executions_remaining"] == -1:
                return -1  # unlimited — always authorised

            # Atomic decrement: only succeeds if executions > 0
            cur = conn.execute(
                """
                UPDATE rats
                   SET executions_remaining = executions_remaining - 1
                 WHERE rat_id = ?
                   AND executions_remaining > 0
                RETURNING executions_remaining
                """,
                (rat_id,),
            )
            result_row = cur.fetchone()
            conn.commit()

            if result_row is None:
                return None  # exhausted (race condition handled atomically)
            return result_row[0]

    def invalidate_rat(self, rat_id: str) -> bool:
        """
        Delete a RAT by ID (e.g. after divergence above epsilon_escalate).
        Returns True if a row was deleted.
        """
        with self._lock:
            conn = self._get_conn()
            cur = conn.execute("DELETE FROM rats WHERE rat_id = ?", (rat_id,))
            conn.commit()
            return cur.rowcount > 0

    def get_rat(self, rat_id: str) -> RATRecord | None:
        """Fetch a single RAT by ID, or None if not found / expired."""
        now = time.time()
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT * FROM rats WHERE rat_id = ? AND ttl_expiry_timestamp > ?",
                (rat_id, now),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def list_active_rats(self) -> list[RATRecord]:
        """Return all currently valid RATs (for debugging / inspection)."""
        now = time.time()
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT * FROM rats WHERE ttl_expiry_timestamp > ? ORDER BY issued_at",
                (now,),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def __repr__(self) -> str:
        try:
            count = self._get_conn().execute(
                "SELECT COUNT(*) FROM rats WHERE ttl_expiry_timestamp > ?",
                (time.time(),),
            ).fetchone()[0]
        except Exception:
            count = "?"
        return f"LocalRATStore(db={self._db_path}, active_rats={count})"
