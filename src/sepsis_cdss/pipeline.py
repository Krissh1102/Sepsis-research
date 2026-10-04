"""Orchestrates the agents into one human-supervised decision-support flow."""
from __future__ import annotations

import pandas as pd

from .approval import ApprovalGate
from .audit import AuditTrail
from .data import LABEL
from .evidence import EvidenceRetriever
from .features import build_features, fit_medians, patient_features
from .mortality import DEKD
from .prediction import SepsisPredictionAgent
from .reasoning import CaseReport, build_reasoning, evidence_query, top_drivers
from .treatment import NullTreatmentAgent
from .verification import deletion_faithfulness

SYNTHETIC_CAVEAT = ("Research prototype. Trained on whatever data it was given (the bundled demo uses "
                    "synthetic data), not validated prospectively and not a medical device.")


class SepsisCDSS:
    def __init__(self, audit_path=None, retriever: EvidenceRetriever | None = None,
                 prediction: SepsisPredictionAgent | None = None, mortality: DEKD | None = None,
                 treatment=None, trained_on_synthetic: bool = True):
        self.audit = AuditTrail(audit_path)
        self.gate = ApprovalGate(self.audit)
        self.retriever = retriever or EvidenceRetriever()
        self.prediction = prediction or SepsisPredictionAgent()
        self.mortality = mortality or DEKD()
        self.treatment = treatment or NullTreatmentAgent()
        self.trained_on_synthetic = trained_on_synthetic

    # -- training -------------------------------------------------------------
    def fit(self, hourly: pd.DataFrame, patients: pd.DataFrame, val_ids: list[str]) -> dict:
        """Train on every patient not in ``val_ids``; use ``val_ids`` for the alert threshold."""
        self.medians = fit_medians(hourly[~hourly["patient_id"].isin(val_ids)])
        X, meta = build_features(hourly, self.medians)
        is_val = meta["patient_id"].isin(val_ids).to_numpy()
        y = meta[LABEL].to_numpy()
        self.prediction.fit(X[~is_val], y[~is_val])
        self.prediction.calibrate_threshold(X[is_val], y[is_val])
        self.baseline = X[~is_val].median()

        pf = patient_features(X, meta).join(patients.set_index("patient_id")[["sepsis", "died"]])
        train_pf = pf[(~pf.index.isin(val_ids)) & (pf["sepsis"] == 1)]
        self.mortality_columns = [c for c in pf.columns if c not in ("sepsis", "died")]
        self.mortality.fit(train_pf[self.mortality_columns], train_pf["died"].to_numpy())
        self.audit.append("models_trained", dict(
            n_train_patients=int((~patients["patient_id"].isin(val_ids)).sum()),
            threshold=self.prediction.threshold, phenotype_sizes=self.mortality.cluster_sizes_))
        return dict(threshold=self.prediction.threshold, phenotype_sizes=self.mortality.cluster_sizes_)

    # -- inference ------------------------------------------------------------
    def assess(self, patient_hourly: pd.DataFrame, case_id: str) -> CaseReport:
        """Score one patient's stay so far and submit the proposal for review."""
        X, meta = build_features(patient_hourly, self.medians)
        last = X.iloc[[-1]]
        risk = float(self.prediction.predict_proba(last)[0])
        flagged = risk >= self.prediction.threshold
        phi, _ = self.prediction.contributions(last)
        drivers = top_drivers(last.iloc[0].to_numpy(), phi[0], list(last.columns))

        pf = patient_features(X, meta)[self.mortality_columns]
        mort = float(self.mortality.predict_proba(pf)[0])
        pheno = int(self.mortality.phenotype(pf)[0])
        mphi = self.mortality.contributions(pf)
        mdrivers = top_drivers(pf.iloc[0].to_numpy(), mphi[0], list(pf.columns), k=3)

        evidence = self.retriever.search(evidence_query(drivers, mdrivers), k=3)
        verification = deletion_faithfulness(self.prediction.predict_proba, last, phi,
                                             self.baseline, k=5)
        action = ("Clinician review for possible sepsis" if flagged
                  else "Continue routine monitoring")
        caveats = [SYNTHETIC_CAVEAT if self.trained_on_synthetic else
                   "Research prototype; not validated prospectively and not a medical device.",
                   "The mortality model is trained on septic patients only and is not meaningful "
                   "for patients without sepsis.",
                   getattr(self.treatment, "note", "A treatment agent is configured; its output is a proposal only.")]
        miss = 1 - float(patient_hourly.tail(1)[["Lactate", "WBC", "Creatinine"]].notna().mean(axis=1).iloc[0])
        if miss > 0:
            caveats.append("Some key labs were not measured this hour; their values are carried forward "
                           "or imputed.")
        treat = self.treatment.propose(patient_hourly)
        report = CaseReport(
            case_id=case_id, sepsis_risk=risk, threshold=self.prediction.threshold,
            alert_level="HIGH" if flagged else "LOW", recommended_action=action,
            mortality_risk=mort, phenotype=pheno, drivers=drivers, mortality_drivers=mdrivers,
            evidence=evidence, verification=verification, caveats=caveats,
            reasoning=build_reasoning(risk, self.prediction.threshold, flagged, drivers, mort, pheno,
                                      mdrivers, evidence, verification))
        proposal = report.to_dict()
        if treat is not None:
            proposal["treatment_proposal"] = treat
        self.gate.submit(case_id, proposal)
        return report

    def review(self, case_id, clinician_id, decision, rationale=None, final_action=None):
        return self.gate.review(case_id, clinician_id, decision, rationale, final_action)

    def release(self, case_id):
        return self.gate.release(case_id)
