"""Command line: ``sepsis-cdss demo`` runs the whole flow on a synthetic cohort."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import LABEL, generate_cohort, split_patients
from .features import build_features
from .pipeline import SepsisCDSS


def run_demo(n_patients: int, seed: int, out: Path, n_cases: int = 3) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    hourly, patients = generate_cohort(n_patients, seed)
    train_ids, val_ids, test_ids = split_patients(patients, seed)

    audit_path = out / "audit.jsonl"
    if audit_path.exists():
        audit_path.unlink()
    system = SepsisCDSS(audit_path=audit_path)
    info = system.fit(hourly[hourly["patient_id"].isin(train_ids + val_ids)],
                      patients[patients["patient_id"].isin(train_ids + val_ids)], val_ids)

    test = hourly[hourly["patient_id"].isin(test_ids)]
    Xt, mt = build_features(test, system.medians)
    metrics = dict(sepsis_prediction=system.prediction.evaluate(Xt, mt[LABEL].to_numpy()), **info)

    # Walk a few held-out septic patients through the full human-in-the-loop flow.
    septic_test = patients[patients["patient_id"].isin(test_ids) & (patients["sepsis"] == 1)]
    reports = []
    for n, (_, row) in enumerate(septic_test.head(n_cases).iterrows()):
        stay = test[test["patient_id"] == row["patient_id"]]
        onset_hour = int(row["onset"])
        stay = stay[stay["ICULOS"] <= onset_hour + 1]  # data available around onset
        case_id = f"case-{n + 1:03d}"
        report = system.assess(stay, case_id)
        (out / f"{case_id}.md").write_text(report.to_markdown(), encoding="utf-8")
        if n == 0:
            try:
                system.release(case_id)
            except PermissionError as e:
                print(f"[gate] release blocked before review: {e}")
            system.review(case_id, "dr.demo", "approve")
        elif n == 1:
            system.review(case_id, "dr.demo", "override",
                          rationale="Demo override: clinician judgement differs.",
                          final_action="Clinician-directed alternative (demo)")
        reports.append(dict(case_id=case_id, risk=report.sepsis_risk, alert=report.alert_level,
                            status=system.gate.get(case_id).status.value))
        print(f"{case_id}: risk={report.sepsis_risk:.3f} alert={report.alert_level} "
              f"status={system.gate.get(case_id).status.value}")

    ok, bad = system.audit.verify()
    metrics.update(cases=reports, audit_ok=ok, audit_records=len(system.audit.records))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("sepsis_prediction",)}, indent=2))
    print(f"audit chain valid: {ok} ({len(system.audit.records)} records)")
    return metrics


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sepsis-cdss")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="train on a synthetic cohort and walk cases through the approval gate")
    d.add_argument("--n-patients", type=int, default=1200)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--out", type=Path, default=Path("outputs"))
    d.add_argument("--cases", type=int, default=3)
    v = sub.add_parser("verify-audit", help="verify an audit.jsonl hash chain")
    v.add_argument("path", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "demo":
        run_demo(a.n_patients, a.seed, a.out, a.cases)
    else:
        from .audit import AuditTrail
        if not a.path.exists():
            raise SystemExit(f"audit file not found: {a.path}")
        ok, bad = AuditTrail(a.path).verify()
        print("valid" if ok else f"TAMPERED at record {bad}")
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
