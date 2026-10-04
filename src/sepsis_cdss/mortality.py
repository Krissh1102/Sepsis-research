"""Mortality + Phenotyping Agent: "Dynamic Ensemble, then Knowledge Distillation".

Re-implementation of the idea in He, Liu & Guo (IEEE JBHI 2025):

 Stage 1  Fit a global XGBoost, compute TreeSHAP values, cluster patients in
          SHAP space (K-means, so patients with similar feature->risk mappings
          share a cluster = a data-driven phenotype), keep the eta fraction
          nearest each centre, train one base model per cluster, and fuse base
          predictions with distance-based weights in SHAP space.
 Stage 2  Distil the fused ensemble into one student XGBoost trained on a
          convex mix of true labels and softened ensemble probabilities, so the
          final model is a single tree model that SHAP can explain.

Deviations from the paper (documented, not hidden): fusion distances are
normalised by their per-sample mean for numerical stability, and trees are small
because the demo cohort is small.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class DEKD:
    def __init__(self, k=3, eta=0.8, fusion_alpha=2.0, kd_temperature=2.0,
                 kd_lambda=0.5, rounds=80, max_depth=3, seed=0, min_cluster=25):
        self.k, self.eta = k, eta
        self.fusion_alpha = fusion_alpha
        self.kd_temperature, self.kd_lambda = kd_temperature, kd_lambda
        self.rounds, self.seed, self.min_cluster = rounds, seed, min_cluster
        self.params = dict(objective="binary:logistic", max_depth=max_depth, eta=0.1,
                           subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                           seed=seed, verbosity=0)
        self.columns: list[str] = []

    # -- small wrappers ------------------------------------------------------
    def _train(self, X, y) -> xgb.Booster:
        return xgb.train(self.params, xgb.DMatrix(X, label=y), self.rounds)

    @staticmethod
    def _predict(booster, X) -> np.ndarray:
        return booster.predict(xgb.DMatrix(X))

    @staticmethod
    def _shap(booster, X) -> np.ndarray:
        return booster.predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]

    # -- training ------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y) -> "DEKD":
        y = np.asarray(y).astype(float)
        self.columns = list(X.columns)
        self.global_ = self._train(X, y)
        R = self._shap(self.global_, X)

        km = KMeans(n_clusters=self.k, n_init=10, random_state=self.seed).fit(R)
        self.centers_ = km.cluster_centers_
        self.cluster_sizes_ = np.bincount(km.labels_, minlength=self.k).tolist()

        self.base_: list[xgb.Booster] = []
        for c in range(self.k):
            idx = np.where(km.labels_ == c)[0]
            d = np.linalg.norm(R[idx] - self.centers_[c], axis=1)
            idx = idx[np.argsort(d)[: max(int(self.eta * len(idx)), 1)]]
            if len(idx) >= self.min_cluster and len(np.unique(y[idx])) == 2:
                self.base_.append(self._train(X.iloc[idx], y[idx]))
            else:  # too small / single class: fall back to the global model
                self.base_.append(self.global_)

        H = self._fuse(X, R)
        soft = _sigmoid(_logit(H) / self.kd_temperature)
        target = self.kd_lambda * y + (1 - self.kd_lambda) * soft
        self.student_ = self._train(X, target)
        return self

    def _fuse(self, X, R) -> np.ndarray:
        preds = np.column_stack([self._predict(b, X) for b in self.base_])
        d = cdist(R, self.centers_)
        d = d / (d.mean(axis=1, keepdims=True) + 1e-9)
        w = np.exp(-self.fusion_alpha * d)
        w /= w.sum(axis=1, keepdims=True)
        return (w * preds).sum(axis=1)

    # -- inference -----------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._predict(self.student_, X[self.columns])

    def predict_teacher(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.columns]
        return self._fuse(X, self._shap(self.global_, X))

    def phenotype(self, X: pd.DataFrame) -> np.ndarray:
        """Cluster id (data-driven phenotype) = nearest centre in SHAP space."""
        R = self._shap(self.global_, X[self.columns])
        return cdist(R, self.centers_).argmin(axis=1)

    def contributions(self, X: pd.DataFrame) -> np.ndarray:
        return self._shap(self.student_, X[self.columns])
