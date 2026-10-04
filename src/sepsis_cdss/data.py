"""Data schema, synthetic cohort generator and a PhysioNet-2019 loader.

The column names follow the PhysioNet/CinC 2019 challenge schema (40 clinical
variables + ``SepsisLabel``), which is the dataset used by Choudhury et al.,
Liu et al. and Wang & Yao in the literature review.

The synthetic generator exists so the whole pipeline can run end-to-end without
credentialed data. It is NOT clinically realistic: it only encodes a few
qualitative patterns (vital-sign drift before onset, sparse labs, two
phenotypes with different mortality drivers) so the methods can be exercised
and tested.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen",
    "Platelets",
]
DEMO = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]
CLINICAL = VITALS + LABS
ALL_VARS = CLINICAL + DEMO
LABEL = "SepsisLabel"

# (mean, sd) of a "healthy-ish" ICU patient. Rough, for synthetic data only.
_BASE = {
    "HR": (80, 10), "O2Sat": (97, 2), "Temp": (36.9, 0.4), "SBP": (120, 14),
    "MAP": (85, 10), "DBP": (65, 9), "Resp": (16, 3), "EtCO2": (34, 4),
    "BaseExcess": (0, 2), "HCO3": (24, 2.5), "FiO2": (0.5, 0.1), "pH": (7.4, 0.04),
    "PaCO2": (40, 5), "SaO2": (96, 2), "AST": (30, 12), "BUN": (18, 6),
    "Alkalinephos": (90, 25), "Calcium": (8.8, 0.5), "Chloride": (104, 3),
    "Creatinine": (1.0, 0.3), "Bilirubin_direct": (0.3, 0.1), "Glucose": (110, 20),
    "Lactate": (1.5, 0.5), "Magnesium": (2.0, 0.25), "Phosphate": (3.5, 0.6),
    "Potassium": (4.1, 0.4), "Bilirubin_total": (0.8, 0.3), "TroponinI": (0.1, 0.05),
    "Hct": (37, 4), "Hgb": (12.5, 1.5), "PTT": (30, 5), "WBC": (9, 2.5),
    "Fibrinogen": (300, 60), "Platelets": (230, 50),
}
# Shift (per unit severity) reached around sepsis onset.
_SHIFT_COMMON = {
    "HR": 22, "Resp": 7, "SBP": -8, "MAP": -5, "Lactate": 0.8, "WBC": 2,
    "Creatinine": 0.3, "Platelets": -25, "Bilirubin_total": 0.6, "BUN": 5,
    "HCO3": -2, "pH": -0.03,
}
_SHIFT_A = {"Temp": 1.3, "WBC": 6, "HR": 6, "Resp": 2}  # inflammatory phenotype
_SHIFT_B = {  # hypoperfusion phenotype
    "Lactate": 2.4, "SBP": -20, "MAP": -14, "Creatinine": 0.8,
    "Platelets": -40, "pH": -0.06, "BaseExcess": -4,
}
_OBS_PROB = {v: 0.92 for v in VITALS}
_OBS_PROB.update({"EtCO2": 0.05, "O2Sat": 0.95})
_OBS_PROB.update({v: 0.10 for v in LABS})
_OBS_PROB.update({"FiO2": 0.30, "SaO2": 0.15})


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate_cohort(n_patients: int = 1200, seed: int = 0,
                    sepsis_prevalence: float = 0.30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(hourly, patients)``.

    ``hourly``: one row per patient-hour with the PhysioNet columns,
    ``patient_id`` and ``SepsisLabel`` (1 from 6 h before onset, as in the
    challenge). ``patients``: one row per patient with ``sepsis``, ``onset``,
    ``phenotype`` (A/B/-), ``severity``, ``age`` and ``died``.
    """
    rng = np.random.default_rng(seed)
    frames, rows = [], []
    for i in range(n_patients):
        pid = f"p{i:05d}"
        n = int(rng.integers(24, 61))
        t = np.arange(n)
        septic = rng.random() < sepsis_prevalence
        phenotype = str(rng.choice(["A", "B"])) if septic else "-"
        severity = float(rng.uniform(0.6, 1.6)) if septic else 0.0
        onset = int(rng.integers(10, n)) if septic else -1
        ramp = _sigmoid((t - (onset - 8)) / 3.0) if septic else np.zeros(n)

        age = float(np.clip(rng.normal(62, 15), 20, 90))
        data = {}
        for var in CLINICAL:
            mu, sd = _BASE[var]
            offset = rng.normal(0, 0.5 * sd)
            x = mu + offset + rng.normal(0, sd, n)
            if septic:
                shift = _SHIFT_COMMON.get(var, 0.0)
                shift += (_SHIFT_A if phenotype == "A" else _SHIFT_B).get(var, 0.0)
                x = x + ramp * severity * shift
            data[var] = x
        data["O2Sat"] = np.clip(data["O2Sat"], 60, 100)
        for var in LABS + ["EtCO2"]:
            data[var] = np.maximum(data[var], 0.0) if var not in ("BaseExcess",) else data[var]
        df = pd.DataFrame(data)
        for var, p in _OBS_PROB.items():
            df.loc[rng.random(n) > p, var] = np.nan
        unit1 = int(rng.random() < 0.5)
        df["Age"] = age
        df["Gender"] = int(rng.random() < 0.55)
        df["Unit1"] = unit1
        df["Unit2"] = 1 - unit1
        df["HospAdmTime"] = -float(rng.uniform(0, 100))
        df["ICULOS"] = t + 1
        df[LABEL] = ((t >= onset - 6) & septic).astype(int)
        df.insert(0, "patient_id", pid)
        frames.append(df)

        if septic and phenotype == "A":
            logit = -1.8 + 0.07 * (age - 62)
        elif septic:
            logit = -1.8 + 2.2 * (severity - 1.0) + 0.01 * (age - 62)
        else:
            logit = -3.2 + 0.03 * (age - 62)
        died = int(rng.random() < _sigmoid(logit))
        rows.append(dict(patient_id=pid, sepsis=int(septic), onset=onset,
                         phenotype=phenotype, severity=severity, age=age, died=died))
    return pd.concat(frames, ignore_index=True), pd.DataFrame(rows)


def load_physionet_psv(directory: str | Path) -> pd.DataFrame:
    """Load PhysioNet/CinC 2019 ``.psv`` files (pipe separated) into one frame."""
    files = sorted(Path(directory).glob("*.psv"))
    if not files:
        raise FileNotFoundError(f"no .psv files found in {directory}")
    frames = []
    for f in files:
        d = pd.read_csv(f, sep="|")
        d.insert(0, "patient_id", f.stem)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def split_patients(patients: pd.DataFrame, seed: int = 0,
                   fractions=(0.6, 0.2, 0.2)) -> tuple[list[str], list[str], list[str]]:
    """Patient-level (never row-level) train/val/test split."""
    ids = patients["patient_id"].to_numpy().copy()
    np.random.default_rng(seed).shuffle(ids)
    n_tr = int(fractions[0] * len(ids))
    n_va = int(fractions[1] * len(ids))
    return list(ids[:n_tr]), list(ids[n_tr:n_tr + n_va]), list(ids[n_tr + n_va:])
