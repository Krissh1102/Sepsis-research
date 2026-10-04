import numpy as np
import pytest

from sepsis_cdss.data import LABEL, generate_cohort, split_patients
from sepsis_cdss.features import build_features
from sepsis_cdss.pipeline import SepsisCDSS


@pytest.fixture(scope="module")
def trained():
    hourly, patients = generate_cohort(500, seed=1)
    tr, va, te = split_patients(patients, seed=1)
    system = SepsisCDSS()
    system.fit(hourly[hourly["patient_id"].isin(tr + va)],
               patients[patients["patient_id"].isin(tr + va)], va)
    return system, hourly, patients, te


def test_prediction_beats_chance_on_held_out_patients(trained):
    system, hourly, _, te = trained
    X, meta = build_features(hourly[hourly["patient_id"].isin(te)], system.medians)
    m = system.prediction.evaluate(X, meta[LABEL].to_numpy())
    assert m["auroc"] > 0.75
    assert m["sensitivity"] > 0.5


def test_mortality_student_outputs_probabilities(trained):
    system, hourly, patients, te = trained
    X, meta = build_features(hourly[hourly["patient_id"].isin(te)], system.medians)
    from sepsis_cdss.features import patient_features
    pf = patient_features(X, meta)[system.mortality_columns]
    p = system.mortality.predict_proba(pf)
    assert p.shape == (len(pf),) and np.all((p >= 0) & (p <= 1))
    assert set(system.mortality.phenotype(pf)) <= set(range(system.mortality.k))


def test_end_to_end_case_requires_clinician_before_release(trained):
    system, hourly, patients, te = trained
    pid = patients[patients["patient_id"].isin(te) & (patients["sepsis"] == 1)]["patient_id"].iloc[0]
    report = system.assess(hourly[hourly["patient_id"] == pid], "e2e-1")
    assert 0 <= report.sepsis_risk <= 1 and report.alert_level in {"HIGH", "LOW"}
    assert report.drivers and report.reasoning and report.caveats
    assert 0 <= report.verification["fraction_faithful"] <= 1
    assert "e2e-1" in report.to_markdown()
    with pytest.raises(PermissionError):
        system.release("e2e-1")
    system.review("e2e-1", "dr.test", "approve")
    assert system.release("e2e-1")["status"] == "approved"
    assert system.audit.verify()[0]


def test_shap_explanation_is_faithful_on_average(trained):
    system, hourly, patients, te = trained
    ids = patients[patients["patient_id"].isin(te) & (patients["sepsis"] == 1)]["patient_id"].head(5)
    ratios = []
    for i, pid in enumerate(ids):
        r = system.assess(hourly[hourly["patient_id"] == pid], f"faith-{i}")
        ratios.append(r.verification["ratio"])
    assert np.mean(ratios) > 1.0
