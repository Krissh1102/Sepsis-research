"""Sepsis Prediction Agent: XGBoost + focal weighted BCE + TreeSHAP.

Based on Choudhury et al. (loss), Liu et al. (time-dependent features) and
Wang & Yao (missing-value mask), see ``features.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from .losses import make_xgb_objective


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class SepsisPredictionAgent:
    def __init__(self, alpha=0.5, beta=0.025, gamma=0.5, rounds=150,
                 max_depth=4, learning_rate=0.1, seed=0):
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.rounds = rounds
        self.params = dict(max_depth=max_depth, eta=learning_rate, subsample=0.8,
                           colsample_bytree=0.8, min_child_weight=1.0,
                           tree_method="hist", seed=seed, verbosity=0)
        self.booster: xgb.Booster | None = None
        self.init_margin = 0.0
        self.threshold = 0.5
        self.feature_names: list[str] = []

    # -- helpers -----------------------------------------------------------
    def _dmatrix(self, X: pd.DataFrame, y=None) -> xgb.DMatrix:
        dm = xgb.DMatrix(X, label=y)
        dm.set_base_margin(np.full(len(X), self.init_margin))
        return dm

    # -- training ----------------------------------------------------------
    def fit(self, X: pd.DataFrame, y) -> "SepsisPredictionAgent":
        y = np.asarray(y).astype(float)
        prev = float(np.clip(y.mean(), 1e-4, 1 - 1e-4))
        self.init_margin = float(np.log(prev / (1 - prev)))
        self.feature_names = list(X.columns)
        self.booster = xgb.train(self.params, self._dmatrix(X, y), self.rounds,
                                 obj=make_xgb_objective(self.alpha, self.beta, self.gamma))
        return self

    def calibrate_threshold(self, X_val: pd.DataFrame, y_val) -> float:
        """Pick the threshold maximising Youden's J on a validation set."""
        p = self.predict_proba(X_val)
        fpr, tpr, thr = roc_curve(y_val, p)
        self.threshold = float(thr[np.argmax(tpr - fpr)])
        return self.threshold

    # -- inference ---------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        margin = self.booster.predict(self._dmatrix(X[self.feature_names]), output_margin=True)
        return _sigmoid(margin)

    def contributions(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """TreeSHAP contributions in margin space: ``(phi[n, f], bias[n])``."""
        c = self.booster.predict(self._dmatrix(X[self.feature_names]), pred_contribs=True)
        return c[:, :-1], c[:, -1]

    def evaluate(self, X: pd.DataFrame, y) -> dict:
        p = self.predict_proba(X)
        y = np.asarray(y)
        pred = p >= self.threshold
        tp = int(((pred == 1) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        return dict(
            auroc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
            threshold=self.threshold, sensitivity=tp / max(tp + fn, 1),
            specificity=tn / max(tn + fp, 1), prevalence=float(y.mean()), n=int(len(y)),
        )
