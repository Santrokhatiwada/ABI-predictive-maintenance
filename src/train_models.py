"""Phase 7: train models on the TRAIN period only. Run: python -m src.train_models"""
import json
import platform
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (FEATURES, HORIZON_HOURS, MODELS, REPORTS, SEED, TEST_START, VAL_START, VERSION)

CAT_COLS = ["model"]


def make_preprocessor(num_cols, cat_cols, scale):
    steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [("num", Pipeline(steps), num_cols),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)],
        sparse_threshold=0)


def model_zoo():
    return {
        "dummy": (DummyClassifier(strategy="prior"), False),
        "logreg": (LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=SEED), True),
        "random_forest": (RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=50, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=SEED), False),
        "hist_gb": (HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=31, class_weight="balanced",
            early_stopping=False, random_state=SEED), False),
    }


def main():
    df = pd.read_parquet(FEATURES / "feature_table.parquet")
    with open(FEATURES / "feature_meta.json") as f:
        meta = json.load(f)
    feats = meta["feature_cols"]
    num_cols = [c for c in feats if c not in CAT_COLS]

    df = df[df["split"] != "purged"]
    train, val, test = (df[df["split"] == s] for s in ("train", "val", "test"))
    assert train["datetime"].max() < val["datetime"].min()
    assert val["datetime"].max() < test["datetime"].min()
    print(f"train {len(train):,} rows ({int(train['y'].sum()):,} positive) | "
          f"val {len(val):,} ({int(val['y'].sum()):,}) | test {len(test):,} ({int(test['y'].sum()):,})")

    outdir = MODELS / VERSION
    outdir.mkdir(parents=True, exist_ok=True)
    eval_df = pd.concat([val, test]).reset_index(drop=True)
    preds = eval_df[["machine_id", "datetime", "split", "y"]].copy()
    info = {}

    for name, (est, scale) in model_zoo().items():
        pipe = Pipeline([("pre", make_preprocessor(num_cols, CAT_COLS, scale)), ("model", est)])
        t0 = time.time()
        pipe.fit(train[feats], train["y"])
        fit_s = time.time() - t0
        t0 = time.time()
        p = pipe.predict_proba(eval_df[feats])[:, 1]
        pred_s = time.time() - t0
        preds[f"proba_{name}"] = p
        joblib.dump(pipe, outdir / f"{name}.joblib")

        # reload check: saved model must reproduce the same scores
        again = joblib.load(outdir / f"{name}.joblib").predict_proba(eval_df[feats].head(2000))[:, 1]
        assert np.allclose(again, p[:2000]), f"{name}: reloaded model differs"

        info[name] = {"fit_seconds": round(fit_s, 1),
                      "predict_ms_per_10k_rows": round(1000 * pred_s / len(eval_df) * 10000, 1),
                      "params": {k: str(v) for k, v in est.get_params().items()}}
        print(f"{name:14} fit {fit_s:6.1f}s | predict {info[name]['predict_ms_per_10k_rows']} ms/10k rows | reload OK")

    preds.to_parquet(REPORTS / f"predictions_{VERSION}.parquet", index=False)
    metadata = {
        "version": VERSION, "trained_at": datetime.now().isoformat(timespec="seconds"),
        "horizon_hours": HORIZON_HOURS, "seed": SEED, "features": feats, "categorical": CAT_COLS,
        "train_period": [str(train["datetime"].min()), str(train["datetime"].max())],
        "val_period": [str(val["datetime"].min()), str(val["datetime"].max())],
        "test_period": [str(test["datetime"].min()), str(test["datetime"].max())],
        "split_config": {"VAL_START": str(VAL_START), "TEST_START": str(TEST_START)},
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
        "positives": {"train": int(train["y"].sum()), "val": int(val["y"].sum()), "test": int(test["y"].sum())},
        "library_versions": {"sklearn": sklearn.__version__, "pandas": pd.__version__,
                             "python": platform.python_version()},
        "models": info,
    }
    with open(outdir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nsaved models + metadata to {outdir}/ and scores to reports/predictions_{VERSION}.parquet")


if __name__ == "__main__":
    main()