"""Model zoo."""
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


def get_models(random_state=42):
    models = {
        "Linear": Pipeline([("sc", StandardScaler()), ("m", LinearRegression())]),
        "Ridge":  Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]),
        "Lasso":  Pipeline([("sc", StandardScaler()), ("m", Lasso(alpha=0.05, max_iter=20000))]),
        "RandomForest":     RandomForestRegressor(n_estimators=300, max_depth=6, random_state=random_state, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=random_state),
        "MLP": Pipeline([("sc", StandardScaler()),
                         ("m", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000,
                                            random_state=random_state, early_stopping=True))]),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                                         random_state=random_state, n_jobs=-1, verbosity=0)
    if HAS_LGB:
        models["LightGBM"] = LGBMRegressor(n_estimators=400, max_depth=-1, num_leaves=31,
                                           learning_rate=0.05, random_state=random_state,
                                           n_jobs=-1, verbose=-1)
    return models
