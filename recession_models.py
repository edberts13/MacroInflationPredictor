"""Model zoo + hyperparameter grids for recession classification."""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble     import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline     import Pipeline
from sklearn.preprocessing import StandardScaler


def make_pipeline(estimator):
    """Scaler in pipeline = no leakage during CV."""
    return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])


def get_model_space(mode: str = "fast") -> dict:
    """
    Returns {name: (pipeline, param_grid)} where param_grid keys are
    prefixed 'clf__' so GridSearchCV sees them inside the pipeline.
    """
    if mode == "fast":
        space = {
            "Logit_L2": (make_pipeline(LogisticRegression(max_iter=5000, solver="liblinear")),
                         {"clf__C": [0.1, 1.0, 10.0], "clf__penalty": ["l2"]}),
            "Logit_L1": (make_pipeline(LogisticRegression(max_iter=5000, solver="liblinear")),
                         {"clf__C": [0.1, 1.0, 10.0], "clf__penalty": ["l1"]}),
            "RandomForest": (make_pipeline(RandomForestClassifier(random_state=42, n_jobs=-1)),
                             {"clf__n_estimators": [200, 400],
                              "clf__max_depth":   [3, 5, None],
                              "clf__min_samples_leaf": [2, 5]}),
            "GBM": (make_pipeline(GradientBoostingClassifier(random_state=42)),
                    {"clf__n_estimators": [150, 300],
                     "clf__learning_rate": [0.05, 0.1],
                     "clf__max_depth":     [2, 3]}),
        }
    else:  # full
        space = {
            "Logit_L2": (make_pipeline(LogisticRegression(max_iter=10000, solver="liblinear")),
                         {"clf__C": [0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0],
                          "clf__penalty": ["l2"]}),
            "Logit_L1": (make_pipeline(LogisticRegression(max_iter=10000, solver="liblinear")),
                         {"clf__C": [0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0],
                          "clf__penalty": ["l1"]}),
            "ElasticNet": (make_pipeline(LogisticRegression(max_iter=10000, solver="saga",
                                                            penalty="elasticnet")),
                           {"clf__C": [0.1, 1.0, 10.0],
                            "clf__l1_ratio": [0.2, 0.5, 0.8]}),
            "RandomForest": (make_pipeline(RandomForestClassifier(random_state=42, n_jobs=-1)),
                             {"clf__n_estimators": [200, 400, 800],
                              "clf__max_depth":   [3, 5, 7, None],
                              "clf__min_samples_leaf": [2, 5, 10],
                              "clf__max_features": ["sqrt", 0.5]}),
            "GBM": (make_pipeline(GradientBoostingClassifier(random_state=42)),
                    {"clf__n_estimators":  [150, 300, 500],
                     "clf__learning_rate": [0.03, 0.05, 0.1],
                     "clf__max_depth":     [2, 3, 4],
                     "clf__subsample":     [0.8, 1.0]}),
        }
        try:
            from xgboost import XGBClassifier
            space["XGBoost"] = (make_pipeline(XGBClassifier(eval_metric="logloss",
                                                            random_state=42,
                                                            verbosity=0,
                                                            n_jobs=-1)),
                                {"clf__n_estimators":  [300, 600],
                                 "clf__learning_rate": [0.03, 0.05, 0.1],
                                 "clf__max_depth":     [3, 5],
                                 "clf__subsample":     [0.8, 1.0]})
        except ImportError:
            pass
        try:
            from lightgbm import LGBMClassifier
            space["LightGBM"] = (make_pipeline(LGBMClassifier(random_state=42, verbosity=-1, n_jobs=-1)),
                                 {"clf__n_estimators":  [300, 600],
                                  "clf__learning_rate": [0.03, 0.05, 0.1],
                                  "clf__num_leaves":    [15, 31],
                                  "clf__subsample":     [0.8, 1.0]})
        except ImportError:
            pass
    return space
