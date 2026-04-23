"""
Baseline forecasters — establish a floor for 'did our model add value?'

All are sklearn-compatible (fit / predict) so they drop into rolling_backtest.
Baselines operate on CPI_YOY directly when present; otherwise fall back to
the mean of y_train. They do NOT use the full feature matrix.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class NaiveForecast(BaseEstimator, RegressorMixin):
    """Predict = last observed target (persistence)."""
    def fit(self, X, y):
        y = np.asarray(y).ravel()
        self.last_ = float(y[-1]) if len(y) else 0.0
        return self

    def predict(self, X):
        X = np.asarray(X)
        return np.full(len(X), self.last_, dtype=float)


class RollingMean(BaseEstimator, RegressorMixin):
    """Predict = rolling mean of last `window` target observations."""
    def __init__(self, window: int = 6):
        self.window = window

    def fit(self, X, y):
        y = np.asarray(y).ravel()
        w = min(self.window, len(y)) if len(y) else 1
        self.mean_ = float(np.mean(y[-w:])) if len(y) else 0.0
        return self

    def predict(self, X):
        X = np.asarray(X)
        return np.full(len(X), self.mean_, dtype=float)


class AR1(BaseEstimator, RegressorMixin):
    """
    AR(1)-style baseline: linear regression of y_t on the most-correlated
    column of X. If X is empty, falls back to the mean of y.
    """
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] == 0 or len(y) < 3:
            self.intercept_ = float(np.mean(y)) if len(y) else 0.0
            self.slope_ = 0.0
            self.col_ = 0
            return self
        # pick the column with largest |corr| with y
        corrs = []
        for j in range(X.shape[1]):
            col = X[:, j]
            if np.std(col) < 1e-9:
                corrs.append(0.0)
            else:
                corrs.append(abs(np.corrcoef(col, y)[0, 1]))
        self.col_ = int(np.nanargmax(corrs))
        x = X[:, self.col_]
        # simple OLS
        x_mean = x.mean(); y_mean = y.mean()
        denom = ((x - x_mean) ** 2).sum()
        self.slope_ = float(((x - x_mean) * (y - y_mean)).sum() / denom) if denom > 0 else 0.0
        self.intercept_ = float(y_mean - self.slope_ * x_mean)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] == 0:
            return np.full(len(X), self.intercept_, dtype=float)
        col = X[:, self.col_] if self.col_ < X.shape[1] else X[:, 0]
        return self.intercept_ + self.slope_ * col


def get_baselines():
    """Dict of baseline name -> estimator. Prefixed 'base_' to sort together."""
    return {
        "base_Naive":       NaiveForecast(),
        "base_RollMean6":   RollingMean(window=6),
        "base_AR1":         AR1(),
    }
