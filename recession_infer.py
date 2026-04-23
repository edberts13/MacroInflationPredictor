"""Load the winner and produce a current recession probability."""
import json, os
import pandas as pd

from recession_data   import build_Xy, build_features, subset_cols
from recession_models import get_model_space

OUT_DIR = "output"


def load_winner():
    p = os.path.join(OUT_DIR, "recession_winner.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def fit_and_predict_current(raw: pd.DataFrame, winner: dict) -> dict:
    """Refit the winner on ALL data, produce probability for latest available month."""
    if winner["kind"] != "single":
        raise NotImplementedError("Ensemble refit not wired; extend if needed.")
    X, y = build_Xy(raw, horizon=winner["horizon_months"],
                    subset=winner["subset"],
                    include_m1=winner["include_m1"],
                    include_m2=winner["include_m2"])
    pipe, _ = get_model_space("fast")[winner["name"]]
    params = json.loads(winner["params"])
    pipe.set_params(**{f"clf__{k}": v for k, v in params.items()})
    pipe.fit(X, y)

    feats_latest = build_features(raw)
    cols = subset_cols(feats_latest, winner["subset"],
                       winner["include_m1"], winner["include_m2"])
    x_now = feats_latest[cols].dropna().iloc[[-1]]
    prob = float(pipe.predict_proba(x_now)[0, 1])
    return {"date": x_now.index[-1], "prob": prob, "model": winner["name"]}
