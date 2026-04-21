"""Ensembling: simple avg, inverse-RMSE weighted avg, stacking meta-model."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def simple_average(pred_df: pd.DataFrame) -> pd.Series:
    return pred_df.mean(axis=1).rename("Ensemble_Avg")


def weighted_average(pred_df: pd.DataFrame, scores_df: pd.DataFrame) -> pd.Series:
    w = 1.0 / scores_df.set_index("model")["RMSE"]
    w = w / w.sum()
    cols = [c for c in pred_df.columns if c in w.index]
    out = (pred_df[cols] * w.loc[cols].values).sum(axis=1)
    return out.rename("Ensemble_Weighted")


def stacking(pred_df: pd.DataFrame, actuals: pd.Series, split_frac=0.5) -> pd.Series:
    """Fit a meta Ridge on first split_frac of OOS preds, predict the rest."""
    df = pred_df.dropna().join(actuals.rename("y"), how="inner").dropna()
    n = len(df)
    cut = int(n * split_frac)
    if cut < 10 or n - cut < 5:
        return pd.Series(dtype=float, name="Ensemble_Stack")
    Xtr, ytr = df.iloc[:cut, :-1].values, df.iloc[:cut, -1].values
    Xte = df.iloc[cut:, :-1].values
    meta = Ridge(alpha=1.0).fit(Xtr, ytr)
    out = pd.Series(meta.predict(Xte), index=df.index[cut:], name="Ensemble_Stack")
    return out
