"""Expanding-window, rolling-origin time-series backtest."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


def rolling_backtest(features: pd.DataFrame, models: dict,
                     initial_train_end="2015-12-31",
                     target_col="inflation_future"):
    # Drop ALL horizon target columns to avoid leakage; keep only the one we want
    all_target_cols = [c for c in features.columns if c.startswith("inflation_future")]
    drop_cols = [c for c in all_target_cols if c != target_col]
    X = features.drop(columns=drop_cols + ([target_col] if target_col in features.columns else []))
    y = features[target_col] if target_col in features.columns else features["inflation_future"]

    dates = features.index
    test_mask = dates > pd.Timestamp(initial_train_end)
    test_dates = dates[test_mask]

    # Store per-model predictions
    preds = {name: pd.Series(index=test_dates, dtype=float) for name in models}
    actuals = y.loc[test_dates].copy()

    # Rolling: retrain yearly (speed), predict month by month
    last_train_year = None
    fitted = {}
    for d in test_dates:
        train_cutoff = d - pd.DateOffset(months=1)
        year = d.year
        if year != last_train_year:
            Xtr = X.loc[:train_cutoff]
            ytr = y.loc[:train_cutoff]
            fitted = {}
            for name, m in models.items():
                try:
                    mm = _clone(m)
                    mm.fit(Xtr.values, ytr.values)
                    fitted[name] = mm
                except Exception as e:
                    print(f"  [backtest] {name} fit failed @ {d}: {e}")
            last_train_year = year

        xrow = X.loc[[d]].values
        for name, mm in fitted.items():
            try:
                preds[name].loc[d] = float(mm.predict(xrow)[0])
            except Exception:
                pass

    pred_df = pd.DataFrame(preds)
    return pred_df, actuals


def _clone(est):
    from sklearn.base import clone
    return clone(est)


def score(actuals: pd.Series, pred_df: pd.DataFrame):
    rows = []
    a = actuals.dropna()
    for col in pred_df.columns:
        p = pred_df[col].dropna()
        idx = a.index.intersection(p.index)
        if len(idx) < 5:
            continue
        ai, pi = a.loc[idx], p.loc[idx]
        rmse = float(np.sqrt(mean_squared_error(ai, pi)))
        mae  = float(mean_absolute_error(ai, pi))
        # direction: sign of change vs previous actual
        ai_shift = ai.shift(1)
        da = np.sign(ai - ai_shift)
        dp = np.sign(pi - ai_shift)
        mask = ~da.isna()
        dir_acc = float((da[mask] == dp[mask]).mean()) if mask.any() else np.nan
        rows.append({"model": col, "RMSE": rmse, "MAE": mae, "DirAcc": dir_acc, "N": len(idx)})
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
