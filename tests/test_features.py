import numpy as np

from sepsis_cdss.data import CLINICAL, generate_cohort
from sepsis_cdss.features import build_features, fit_medians, patient_features


def _cohort():
    return generate_cohort(40, seed=3)


def test_shapes_and_no_nans():
    hourly, _ = _cohort()
    X, meta = build_features(hourly, fit_medians(hourly))
    assert len(X) == len(meta) == len(hourly)
    assert not X.isna().any().any()
    assert all(c in X.columns for c in ("Lactate_obs", "HR_delta", "HR_d3", "HR_m6", "ShockIndex"))


def test_mask_marks_observed_values():
    hourly, _ = _cohort()
    hourly = hourly.sort_values(["patient_id", "ICULOS"]).reset_index(drop=True)
    X, _ = build_features(hourly, fit_medians(hourly))
    assert np.array_equal(X["Lactate_obs"].to_numpy(), hourly["Lactate"].notna().to_numpy().astype(float))


def test_no_lookahead():
    """Changing future hours must not change features at earlier hours."""
    hourly, _ = _cohort()
    med = fit_medians(hourly)
    pid = hourly["patient_id"].iloc[0]
    one = hourly[hourly["patient_id"] == pid].copy()
    X1, _ = build_features(one, med)
    altered = one.copy()
    altered.loc[altered["ICULOS"] > 12, CLINICAL] = 999.0
    X2, _ = build_features(altered, med)
    early = (one.sort_values("ICULOS")["ICULOS"] <= 12).to_numpy()
    assert np.allclose(X1[early].to_numpy(), X2[early].to_numpy())


def test_patient_features_one_row_per_patient():
    hourly, patients = _cohort()
    X, meta = build_features(hourly, fit_medians(hourly))
    pf = patient_features(X, meta)
    assert len(pf) == len(patients) and not pf.isna().any().any()
