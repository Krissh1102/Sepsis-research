"""Audit Trail: append-only, hash-chained event log.

Each record stores the hash of the previous record, so editing or deleting an
earlier record breaks verification. This makes tampering *evident*; it is not a
substitute for access control or a write-once store in production.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

GENESIS = "0" * 64


def _digest(prev_hash: str, body: dict) -> str:
    blob = json.dumps(body, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256((prev_hash + blob).encode("utf-8")).hexdigest()


class AuditTrail:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.records: list[dict] = []
        if self.path and self.path.exists():
            self.records = [json.loads(l) for l in self.path.read_text().splitlines() if l.strip()]

    def append(self, event: str, payload: dict, actor: str = "system") -> dict:
        prev = self.records[-1]["hash"] if self.records else GENESIS
        body = dict(seq=len(self.records), ts=datetime.now(timezone.utc).isoformat(),
                    actor=actor, event=event, payload=json.loads(json.dumps(payload, default=str)))
        rec = dict(body, prev_hash=prev, hash=_digest(prev, body))
        self.records.append(rec)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
        return rec

    def verify(self) -> tuple[bool, int | None]:
        """Return ``(ok, first_bad_seq)``."""
        prev = GENESIS
        for i, rec in enumerate(self.records):
            body = {k: rec[k] for k in ("seq", "ts", "actor", "event", "payload")}
            if rec.get("prev_hash") != prev or rec.get("hash") != _digest(prev, body) or rec["seq"] != i:
                return False, i
            prev = rec["hash"]
        return True, None

    def for_case(self, case_id: str) -> list[dict]:
        return [r for r in self.records if r["payload"].get("case_id") == case_id]
