"""Versioned decision state with idempotent observations; no real-world execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

from .decision import Problem


class DecisionSession:
    def __init__(self, path: Path, initial: Problem | None = None):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK(id=1), version INTEGER, problem TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, digest TEXT, version INTEGER, payload TEXT)")
        if initial is not None:
            self.db.execute("INSERT OR IGNORE INTO state VALUES (1,0,?)", (json.dumps(initial.to_dict()),))
            self.db.commit()
        if self.db.execute("SELECT 1 FROM state").fetchone() is None:
            raise ValueError("A new session requires an initial problem")

    def close(self):
        self.db.close()

    def snapshot(self) -> tuple[int, Problem]:
        version, raw = self.db.execute("SELECT version,problem FROM state WHERE id=1").fetchone()
        return version, Problem.from_dict(json.loads(raw))

    def update(self, event_id: str, expected_version: int, payload: dict) -> int:
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(raw.encode()).hexdigest()
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            existing = self.db.execute("SELECT digest,version FROM events WHERE id=?", (event_id,)).fetchone()
            if existing:
                if existing[0] != digest:
                    raise ValueError("An event ID cannot be reused for a different payload")
                return existing[1]
            version, problem = self.snapshot()
            if version != expected_version:
                raise ValueError("Stale decision state version")
            if payload["kind"] == "observation":
                if not problem.feasible("observe"):
                    raise ValueError("Observation is unavailable")
                probability, updated = problem.posterior(payload["signal"])
                if probability == 0:
                    raise ValueError("Observation contradicts the supplied observation model")
            elif payload["kind"] == "revision":
                updated = Problem.from_dict(payload["problem"])
                if not payload.get("reason"):
                    raise ValueError("A revised objective or model requires a reason")
            else:
                raise ValueError("Unsupported event type")
            version += 1
            self.db.execute("UPDATE state SET version=?,problem=? WHERE id=1", (version, json.dumps(updated.to_dict())))
            self.db.execute("INSERT INTO events VALUES (?,?,?,?)", (event_id, digest, version, raw))
        return version

    def situation_packet(self) -> dict:
        version, problem = self.snapshot()
        return {"protocol_version": "treyrin-decision-state-v1", "state_version": version,
                "problem": problem.to_dict(), "executed_real_world_actions": False,
                "events": [{"id": row[0], "digest": row[1], "version": row[2], "payload": json.loads(row[3])}
                           for row in self.db.execute("SELECT id,digest,version,payload FROM events ORDER BY version")]}
