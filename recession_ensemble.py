"""Ensemble combiners for probability predictions. No leakage — weights fit on validation split only."""
import numpy as np
from itertools import combinations
from scipy.optimize import minimize
from sklearn.metrics import log_loss


def simple_avg(prob_cols):
    return np.mean(np.column_stack(prob_cols), axis=1)


def weighted_avg(prob_cols, weights):
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return np.column_stack(prob_cols) @ w


def optimize_weights(prob_cols_val, y_val):
    """Constrained: w_i >= 0, sum(w_i) = 1. Minimize log loss on validation."""
    n = len(prob_cols_val)

    def obj(w):
        p = np.clip(weighted_avg(prob_cols_val, w), 1e-6, 1 - 1e-6)
        return log_loss(y_val, p)

    w0   = np.ones(n) / n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bnds = [(0.0, 1.0)] * n
    res  = minimize(obj, w0, method="SLSQP", bounds=bnds, constraints=cons,
                    options={"maxiter": 200})
    return res.x / res.x.sum()


def generate_combos(top_models, sizes=(2, 3)):
    combos = []
    for k in sizes:
        if k <= len(top_models):
            combos.extend(combinations(top_models, k))
    return combos


def combo_label(combo, kind):
    return f"{kind}({'+'.join(combo)})"
