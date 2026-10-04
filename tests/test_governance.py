import json

import pytest

from sepsis_cdss.approval import ApprovalGate, Status
from sepsis_cdss.audit import AuditTrail
from sepsis_cdss.evidence import EvidenceRetriever


def test_audit_chain_verifies_and_detects_tampering(tmp_path):
    path = tmp_path / "audit.jsonl"
    a = AuditTrail(path)
    for i in range(5):
        a.append("event", dict(case_id="c1", i=i))
    assert a.verify() == (True, None)
    assert AuditTrail(path).verify() == (True, None)  # reload from disk

    lines = path.read_text().splitlines()
    rec = json.loads(lines[2])
    rec["payload"]["i"] = 999
    lines[2] = json.dumps(rec, sort_keys=True)
    path.write_text("\n".join(lines) + "\n")
    assert AuditTrail(path).verify() == (False, 2)


def test_audit_detects_deleted_record(tmp_path):
    path = tmp_path / "audit.jsonl"
    a = AuditTrail(path)
    for i in range(4):
        a.append("event", dict(i=i))
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:1] + lines[2:]) + "\n")
    assert AuditTrail(path).verify()[0] is False


def test_release_blocked_until_clinician_decides():
    gate = ApprovalGate(AuditTrail())
    gate.submit("c1", dict(recommended_action="Review"))
    with pytest.raises(PermissionError):
        gate.release("c1")
    gate.review("c1", "dr.x", "defer")
    with pytest.raises(PermissionError):
        gate.release("c1")
    assert gate.get("c1").status == Status.DEFERRED


def test_approve_and_override_paths():
    gate = ApprovalGate(AuditTrail())
    gate.submit("a", dict(recommended_action="Review"))
    gate.review("a", "dr.x", "approve")
    assert gate.release("a")["final_action"] == "Review"

    gate.submit("b", dict(recommended_action="Review"))
    with pytest.raises(ValueError):
        gate.review("b", "dr.x", "override")  # needs rationale + action
    gate.review("b", "dr.x", "override", rationale="Reason", final_action="Other")
    assert gate.release("b")["final_action"] == "Other"


def test_clinician_id_required_and_duplicate_submit_rejected():
    gate = ApprovalGate(AuditTrail())
    gate.submit("a", {})
    with pytest.raises(ValueError):
        gate.review("a", " ", "approve")
    with pytest.raises(ValueError):
        gate.submit("a", {})


def test_every_gate_transition_is_audited():
    audit = AuditTrail()
    gate = ApprovalGate(audit)
    gate.submit("a", dict(recommended_action="Review"))
    gate.review("a", "dr.x", "approve")
    gate.release("a")
    assert [r["event"] for r in audit.for_case("a")] == ["proposal_submitted", "clinician_decision", "released"]


def test_retriever_returns_relevant_builtin_entry():
    hits = EvidenceRetriever().search("mortality SOFA score marital status", k=2)
    assert hits and hits[0]["id"] == "jin2026"
    assert EvidenceRetriever().search("   ") == []


def test_retriever_loads_directory(tmp_path):
    (tmp_path / "bundle.md").write_text("antibiotic timing guideline excerpt for septic shock")
    r = EvidenceRetriever.from_directory(tmp_path)
    assert r.search("antibiotic timing", k=1)[0]["id"] == "bundle"
