"""
AI Verification pipeline — append-only, hash-chained "sealed ledger".

Every case that flows through the pipeline emits one event: source hash,
AI-record hash, detector verdicts, quality score, and the human/clinical
decision. Events are chained with SHA-256 (curr_hash = H(prev_hash || payload)),
so any post-hoc modification, deletion, or reordering breaks the chain and is
localised by verify_integrity(). This mirrors tamper-evident audit-trail designs
(llm-audit-trail, AuditWeave, aiAuthZ) used for AI accountability.

PHI note: only content *hashes* and non-identifying metadata are stored, never
raw clinical text — the "store digests, keep PII off the ledger" pattern.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict, field
from typing import Any

GENESIS = "GENESIS"


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def content_hash(obj: Any) -> str:
    return hashlib.sha256(_canonical({"c": obj}).encode()).hexdigest()


@dataclass
class LedgerEvent:
    seq: int
    case_id: str
    source_hash: str
    ai_record_hash: str
    quality_score: float
    n_flagged: int
    decision: str          # human/clinical decision (Stage 6)
    prev_hash: str
    curr_hash: str = ""

    def payload(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("curr_hash")
        return d

    def compute_hash(self) -> str:
        return hashlib.sha256(
            (self.prev_hash + _canonical(self.payload())).encode()
        ).hexdigest()


class SealedLedger:
    def __init__(self):
        self._events: list[LedgerEvent] = []

    @property
    def head(self) -> str:
        return self._events[-1].curr_hash if self._events else GENESIS

    def append(self, case_id: str, source_hash: str, ai_record_hash: str,
               quality_score: float, n_flagged: int, decision: str) -> LedgerEvent:
        ev = LedgerEvent(
            seq=len(self._events),
            case_id=case_id,
            source_hash=source_hash,
            ai_record_hash=ai_record_hash,
            quality_score=round(quality_score, 4),
            n_flagged=n_flagged,
            decision=decision,
            prev_hash=self.head,
        )
        ev.curr_hash = ev.compute_hash()
        self._events.append(ev)
        return ev

    def __len__(self) -> int:
        return len(self._events)

    def verify_integrity(self) -> tuple[bool, int | None, str]:
        """Recompute the whole chain. Returns (ok, first_bad_seq, message)."""
        prev = GENESIS
        for ev in self._events:
            if ev.prev_hash != prev:
                return False, ev.seq, f"predecessor mismatch at seq {ev.seq}"
            if ev.compute_hash() != ev.curr_hash:
                return False, ev.seq, f"content hash mismatch at seq {ev.seq}"
            prev = ev.curr_hash
        return True, None, "chain intact"

    def to_records(self) -> list[dict]:
        return [asdict(e) for e in self._events]
