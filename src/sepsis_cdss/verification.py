"""Explanation Verification Agent: is the explanation faithful?

The literature review (Gap 4) notes that explanations are rarely checked for
faithfulness. A deletion test is used here: neutralise (set to the training
median) the features SHAP says matter most and see whether the prediction moves
more than when the same number of random features are neutralised. If the top
features really drive the model, deleting them must change the output more.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def deletion_faithfulness(predict_fn, X: pd.DataFrame, contribs: np.ndarray,
                          baseline: pd.Series, k: int = 5, n_random: int = 30,
                          seed: int = 0) -> dict:
    """Deletion test for each row of ``X``.

    ``contribs`` has shape ``(len(X), n_features)`` aligned with ``X.columns``.
    Returns per-batch summary: mean |change| when removing the top-k features vs
    random-k, their ratio, and the fraction of rows where top-k > random-k.
    """
    rng = np.random.default_rng(seed)
    cols = list(X.columns)
    base_vals = baseline.reindex(cols).to_numpy(dtype=float)
    p0 = np.asarray(predict_fn(X), float)

    top_change = np.zeros(len(X))
    rand_change = np.zeros(len(X))
    for i in range(len(X)):
        row = X.iloc[[i]].to_numpy(dtype=float)
        top_idx = np.argsort(-np.abs(contribs[i]))[:k]
        t = row.copy()
        t[0, top_idx] = base_vals[top_idx]
        top_change[i] = abs(p0[i] - float(predict_fn(pd.DataFrame(t, columns=cols))[0]))

        r = np.repeat(row, n_random, axis=0)
        for j in range(n_random):
            idx = rng.choice(len(cols), size=k, replace=False)
            r[j, idx] = base_vals[idx]
        rand_change[i] = float(np.abs(p0[i] - predict_fn(pd.DataFrame(r, columns=cols))).mean())

    return dict(
        k=k,
        mean_top_k_change=float(top_change.mean()),
        mean_random_k_change=float(rand_change.mean()),
        ratio=float(top_change.mean() / max(rand_change.mean(), 1e-9)),
        fraction_faithful=float((top_change > rand_change).mean()),
        faithful=bool(top_change.mean() > rand_change.mean()),
    )
