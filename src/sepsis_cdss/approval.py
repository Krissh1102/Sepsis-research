"""Human Approval Gate.

Nothing the system produces is "released" until a named clinician has reviewed
it. The AI output is a *proposal*; the clinician approves, overrides (with a
required rationale and their own action) or defers. Every transition is written
to the audit trail.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from .audit import AuditTrail


class Status(str, Enum):
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    OVERRIDDEN = "overridden"
    DEFERRED = "deferred"


@dataclass
class Review:
    case_id: str
    proposal: dict
    status: Status = Status.PENDING_REVIEW
    clinician_id: str | None = None
    decision_ts: str | None = None
    rationale: str | None = None
    final_action: str | None = None
    history: list[dict] = field(default_factory=list)


class ApprovalGate:
    def __init__(self, audit: AuditTrail):
        self.audit = audit
        self._reviews: dict[str, Review] = {}

    def submit(self, case_id: str, proposal: dict) -> Review:
        if case_id in self._reviews:
            raise ValueError(f"case {case_id} already submitted")
        rev = Review(case_id=case_id, proposal=proposal)
        self._reviews[case_id] = rev
        self.audit.append("proposal_submitted", dict(case_id=case_id, proposal=proposal))
        return rev

    def get(self, case_id: str) -> Review:
        return self._reviews[case_id]

    def review(self, case_id: str, clinician_id: str, decision: str,
               rationale: str | None = None, final_action: str | None = None) -> Review:
        if not clinician_id or not clinician_id.strip():
            raise ValueError("a clinician_id is required")
        rev = self._reviews[case_id]
        decision = decision.lower()
        if decision not in {"approve", "override", "defer"}:
            raise ValueError("decision must be approve, override or defer")
        if decision == "override" and not (rationale and rationale.strip() and final_action and final_action.strip()):
            raise ValueError("override requires both a rationale and the clinician's final_action")
        rev.status = {"approve": Status.APPROVED, "override": Status.OVERRIDDEN,
                      "defer": Status.DEFERRED}[decision]
        rev.clinician_id = clinician_id
        rev.rationale = rationale
        rev.decision_ts = datetime.now(timezone.utc).isoformat()
        rev.final_action = (final_action if decision == "override"
                            else rev.proposal.get("recommended_action") if decision == "approve" else None)
        rev.history.append(dict(decision=decision, clinician_id=clinician_id, ts=rev.decision_ts))
        self.audit.append("clinician_decision",
                          dict(case_id=case_id, decision=decision, rationale=rationale,
                               final_action=rev.final_action), actor=clinician_id)
        return rev

    def release(self, case_id: str) -> dict:
        """Return the final, clinician-authorised action or raise."""
        rev = self._reviews[case_id]
        if rev.status not in (Status.APPROVED, Status.OVERRIDDEN):
            raise PermissionError(f"case {case_id} has no clinician approval (status: {rev.status.value})")
        self.audit.append("released", dict(case_id=case_id, final_action=rev.final_action,
                                           status=rev.status.value), actor=rev.clinician_id)
        return dict(case_id=case_id, final_action=rev.final_action, status=rev.status.value,
                    clinician_id=rev.clinician_id, rationale=rev.rationale)

    def pending(self) -> list[str]:
        return [c for c, r in self._reviews.items() if r.status == Status.PENDING_REVIEW]
