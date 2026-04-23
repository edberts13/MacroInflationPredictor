"""
Benchmark: fit a logistic regression on the same signals the heuristic uses,
to get a calibrated probability baseline for comparison with the ML winner.
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

from recession_data import build_features, _fetch_usrec, build_target

HEURISTIC_COLS = ["YC_10_3M", "SAHM", "HY_SPREAD", "NFCI", "SP500_RET6", "VIX"]


def main():
    if os.path.exists("macro_enhanced.csv"):
        raw = pd.read_csv("macro_enhanced.csv", index_col="date", parse_dates=True)
    else:
        raw = pd.read_csv("macro_raw.csv", index_col="date", parse_dates=True)

    feats = build_features(raw)
    cols  = [c for c in HEURISTIC_COLS if c in feats.columns
             and feats[c].notna().mean() >= 0.5]
    print(f"[heuristic] cols used: {cols}")
    X = feats[cols].dropna()

    usrec = _fetch_usrec(start=str(feats.index.min().date()))
    y = build_target(usrec, horizon_months=12).reindex(X.index).dropna()
    X = X.loc[y.index]

    cv  = TimeSeriesSplit(n_splits=5)
    oof = np.full(len(X), np.nan)
    for tr, va in cv.split(X):
        if y.iloc[tr].nunique() < 2:
            print(f"  [skip fold] train has only class {y.iloc[tr].unique()}")
            continue
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("clf",    LogisticRegression(max_iter=5000))])
        pipe.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = pipe.predict_proba(X.iloc[va])[:, 1]
    mask = ~np.isnan(oof)
    ll  = log_loss(y[mask], np.clip(oof[mask], 1e-6, 1 - 1e-6))
    auc = roc_auc_score(y[mask], oof[mask]) if y[mask].nunique() > 1 else float("nan")
    br  = brier_score_loss(y[mask], oof[mask])
    print(f"[heuristic-calibrated] cols={cols}")
    print(f"[heuristic-calibrated] logloss={ll:.4f}  AUC={auc:.3f}  brier={br:.4f}")

    os.makedirs("output", exist_ok=True)
    pd.DataFrame({"date": X.index, "heuristic_prob": oof,
                  "y_true": y.values}).to_csv(
        "output/recession_heuristic_probs.csv", index=False)


if __name__ == "__main__":
    main()
