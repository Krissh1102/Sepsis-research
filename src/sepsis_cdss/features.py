"""Feature engineering.

Follows the ideas in the reviewed sepsis-prediction papers:
  * missing-value mask + forward fill + deltas (Choudhury et al., Wang & Yao)
  * time-dependent features: lagged differences and rolling means
    (Liu et al.)
  * shock index (HR / SBP), a hand-crafted feature mentioned by Choudhury et al.
All features at hour t use only data up to hour t (no look-ahead).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import CLINICAL, DEMO, LABEL, VITALS


def fit_medians(df: pd.DataFrame) -> dict[str, float]:
    """Training-set medians used to impute variables never observed so far."""
    return df[CLINICAL + [d for d in DEMO if d != "ICULOS"]].median().fillna(0.0).to_dict()


def build_features(df: pd.DataFrame, medians: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(X, meta)`` for an hourly frame.

    ``meta`` carries ``patient_id``, ``ICULOS`` and (if present) ``SepsisLabel``
    in the same row order as ``X``.
    """
    df = df.sort_values(["patient_id", "ICULOS"]).reset_index(drop=True)
    pid = df["patient_id"]
    med = pd.Series(medians)

    raw = df[CLINICAL]
    obs = raw.notna().astype(np.int8).add_suffix("_obs")
    ff = raw.groupby(pid).ffill()
    ff = ff.fillna(med[CLINICAL])
    delta = ff.groupby(pid).diff().fillna(0.0).add_suffix("_delta")

    vit = ff[VITALS]
    vit_g = vit.groupby(pid)
    d3 = (vit - vit_g.shift(3)).fillna(0.0).add_suffix("_d3")
    m6 = (vit_g.rolling(6, min_periods=1).mean()
          .reset_index(level=0, drop=True).sort_index().add_suffix("_m6"))
    shock = (ff["HR"] / ff["SBP"].clip(lower=1.0)).rename("ShockIndex")

    demo = df[DEMO].copy()
    for c in DEMO:
        if c != "ICULOS":
            demo[c] = demo[c].fillna(med[c])
    demo["ICULOS"] = demo["ICULOS"].fillna(0)

    X = pd.concat([ff, obs, delta, d3, m6, shock, demo], axis=1).astype(float)
    meta_cols = ["patient_id", "ICULOS"] + ([LABEL] if LABEL in df.columns else [])
    return X, df[meta_cols].copy()


def patient_features(X: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Patient-level aggregates (mean / max / min over the stay so far)."""
    cols = CLINICAL
    g = X[cols].groupby(meta["patient_id"].to_numpy())
    parts = []
    for name, fn in (("mean", g.mean), ("max", g.max), ("min", g.min)):
        parts.append(fn().add_suffix(f"_{name}"))
    demo = X[["Age", "Gender", "Unit1"]].groupby(meta["patient_id"].to_numpy()).first()
    n_hours = meta.groupby("patient_id")["ICULOS"].max().rename("n_hours")
    return pd.concat(parts + [demo, n_hours], axis=1)


_DESCRIPTIONS = {
    "HR": "heart rate", "O2Sat": "oxygen saturation", "Temp": "temperature",
    "SBP": "systolic blood pressure", "MAP": "mean arterial pressure",
    "DBP": "diastolic blood pressure", "Resp": "respiratory rate",
    "EtCO2": "end-tidal CO2", "Lactate": "lactate", "WBC": "white blood cell count",
    "Creatinine": "creatinine", "Platelets": "platelet count", "BUN": "blood urea nitrogen",
    "FiO2": "inspired oxygen fraction", "pH": "blood pH", "ICULOS": "ICU length of stay",
    "HospAdmTime": "time from hospital to ICU admission", "Age": "age",
    "ShockIndex": "shock index (HR/SBP)", "Bilirubin_total": "total bilirubin",
    "BaseExcess": "base excess", "HCO3": "bicarbonate",
}
_SUFFIX = {"_obs": " (measured this hour)", "_delta": " (change since last hour)",
           "_d3": " (3-hour change)", "_m6": " (6-hour mean)",
           "_mean": " (stay mean)", "_max": " (stay max)", "_min": " (stay min)"}


def describe_feature(name: str) -> str:
    """Plain-language label for a feature column name."""
    for suf, text in _SUFFIX.items():
        if name.endswith(suf):
            base = name[: -len(suf)]
            return _DESCRIPTIONS.get(base, base) + text
    return _DESCRIPTIONS.get(name, name)
