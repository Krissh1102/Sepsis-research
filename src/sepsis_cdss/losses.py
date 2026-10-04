"""Focal weighted binary cross-entropy for gradient boosting.

Per-sample loss (Choudhury et al., ICETCI 2025, Eq. 4), with p = sigmoid(z):

    y = 1:  -alpha * (1 - p)^gamma * log(p)
    y = 0:  -(1 - alpha) * beta * p^gamma * log(1 - p)

alpha: linear weight between the two classes, beta: extra penalty on false
positives (beta < 1 favours sensitivity), gamma: focal focusing parameter.

The gradient is derived here with respect to the raw margin z (what XGBoost
needs) and checked against finite differences in the tests. The Hessian is a
central difference of the gradient, clipped to stay positive, because the
focal loss is non-convex.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-7


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def focal_weighted_bce(z, y, alpha=0.5, beta=0.025, gamma=0.5):
    """Mean-free per-sample loss as a function of the margin ``z``."""
    p = np.clip(_sigmoid(np.asarray(z, float)), _EPS, 1 - _EPS)
    y = np.asarray(y)
    pos = -alpha * (1 - p) ** gamma * np.log(p)
    neg = -(1 - alpha) * beta * p ** gamma * np.log(1 - p)
    return np.where(y == 1, pos, neg)


def grad_z(z, y, alpha=0.5, beta=0.025, gamma=0.5):
    """d loss / d z."""
    p = np.clip(_sigmoid(np.asarray(z, float)), _EPS, 1 - _EPS)
    y = np.asarray(y)
    d_pos = -alpha * (-gamma * (1 - p) ** (gamma - 1) * np.log(p) + (1 - p) ** gamma / p)
    d_neg = -(1 - alpha) * beta * (gamma * p ** (gamma - 1) * np.log(1 - p) - p ** gamma / (1 - p))
    return np.where(y == 1, d_pos, d_neg) * p * (1 - p)


def grad_hess(z, y, alpha=0.5, beta=0.025, gamma=0.5, h=1e-3, min_hess=1e-3):
    g = grad_z(z, y, alpha, beta, gamma)
    z = np.asarray(z, float)
    hess = (grad_z(z + h, y, alpha, beta, gamma) - grad_z(z - h, y, alpha, beta, gamma)) / (2 * h)
    return g, np.clip(hess, min_hess, 1e3)


def make_xgb_objective(alpha=0.5, beta=0.025, gamma=0.5):
    """Objective for ``xgboost.train(..., obj=...)`` (receives raw margins)."""
    def _obj(preds, dtrain):
        return grad_hess(preds, dtrain.get_label(), alpha, beta, gamma)
    return _obj
