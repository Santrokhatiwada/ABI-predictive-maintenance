"""Phase 8: evaluate on validation, select, then report once on test. Run: python -m src.evaluate_models"""
import json

import sys
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)


PROJECT = Path(
    "/Users/santrokhatiwada/Documents/MAT/SEM 1/Data Analytics/Assignment 2/abi_pdm"
)

if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.config import (
    CLEAN,
    FEATURES,
    FIG,
    HORIZON_HOURS,
    MODELS,
    REPORTS,
    SEED,
    TEST_START,
    VAL_START,
    VERSION,
)

plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3})
FIG.mkdir(parents=True, exist_ok=True)
NAMES = ["dummy", "logreg", "random_forest", "hist_gb"]

preds = pd.read_parquet(REPORTS / f"predictions_{VERSION}.parquet")
fails = pd.read_parquet(CLEAN / "failures_clean.parquet")[["machine_id", "datetime"]].drop_duplicates()
val, test = preds[preds["split"] == "val"], preds[preds["split"] == "test"]


def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.nanargmax(f1))])


def event_recall(d, flag, start):
    """Failure events with >=1 alert in the 24h before them (alert at t covers [t, t+24h))."""
    h = pd.Timedelta(hours=HORIZON_HOURS)
    f = fails[fails["datetime"] >= start + h]
    by_m = {m: np.sort(g["datetime"].values) for m, g in d.loc[flag, ["machine_id", "datetime"]].groupby("machine_id")}
    hit = 0
    for r in f.itertuples():
        arr = by_m.get(r.machine_id)
        if arr is None:
            continue
        i = np.searchsorted(arr, (r.datetime - h).to_datetime64(), side="left")
        j = np.searchsorted(arr, r.datetime.to_datetime64(), side="left")
        hit += int(j > i)
    return hit, len(f)


rows = []
for name in NAMES:
    thr = best_f1_threshold(val["y"], val[f"proba_{name}"])
    row = {"model": name, "threshold_from_val": round(thr, 4)}
    for sname, d, start in (("val", val, VAL_START), ("test", test, TEST_START)):
        y, p = d["y"].values, d[f"proba_{name}"].values
        flag = p >= thr
        tn, fp, fn, tp = confusion_matrix(y, flag, labels=[0, 1]).ravel()
        hit, n_ev = event_recall(d, flag, start)
        row.update({
            f"{sname}_pr_auc": average_precision_score(y, p), f"{sname}_roc_auc": roc_auc_score(y, p),
            f"{sname}_brier": brier_score_loss(y, p), f"{sname}_accuracy": accuracy_score(y, flag),
            f"{sname}_precision": precision_score(y, flag, zero_division=0),
            f"{sname}_recall": recall_score(y, flag, zero_division=0), f"{sname}_f1": f1_score(y, flag, zero_division=0),
            f"{sname}_TP": int(tp), f"{sname}_FP": int(fp), f"{sname}_FN": int(fn), f"{sname}_TN": int(tn),
            f"{sname}_alert_rate_%": 100 * flag.mean(), f"{sname}_event_recall": f"{hit}/{n_ev}"})
    rows.append(row)
res = pd.DataFrame(rows)
res.round(4).to_csv(REPORTS / "model_comparison.csv", index=False)
pd.set_option("display.width", 250)
print("=== VALIDATION (used for selection; thresholds were tuned here, so these are optimistic) ===")
print(res.filter(regex="^(model|val_)").round(4).T.to_string(header=False))
print("\n=== TEST (reported once, thresholds fixed from validation) ===")
print(res.filter(regex="^(model|test_)").round(4).T.to_string(header=False))
print(f"\nPositive rate: val {100*val['y'].mean():.2f}% | test {100*test['y'].mean():.2f}%  "
      f"(a no-skill PR-AUC equals this rate)")

# ---- selection (validation PR-AUC, excluding the dummy baseline) ----
cand = res[res["model"] != "dummy"].sort_values("val_pr_auc", ascending=False)
best = cand.iloc[0]["model"]
best_thr = float(cand.iloc[0]["threshold_from_val"])
print(f"\nSELECTED (highest validation PR-AUC): {best}  threshold={best_thr:.4f}")
with open(MODELS / VERSION / "selected_model.json", "w") as f:
    json.dump({"version": VERSION, "model": best, "threshold": best_thr,
               "selection_rule": "highest validation PR-AUC; threshold maximises validation F1"}, f, indent=2)

# ---- Plots ----
def plot_curves(kind):
    fig, ax = plt.subplots(figsize=(6, 5))
    for n in NAMES:
        y, p = test["y"], test[f"proba_{n}"]
        if kind == "pr":
            pr, rc, _ = precision_recall_curve(y, p)
            ax.plot(rc, pr, label=f"{n} (AP={average_precision_score(y, p):.3f})")
        else:
            fpr, tpr, _ = roc_curve(y, p)
            ax.plot(fpr, tpr, label=f"{n} (AUC={roc_auc_score(y, p):.3f})")
    if kind == "pr":
        ax.axhline(test["y"].mean(), color="k", ls=":", label="no-skill")
        ax.set(xlabel="recall", ylabel="precision", title="Precision-recall curves (test period)")
    else:
        ax.plot([0, 1], [0, 1], "k:"); ax.set(xlabel="false positive rate", ylabel="true positive rate",
                                              title="ROC curves (test period)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / f"eval_{kind}_curves.png", dpi=300); plt.close(fig)

plot_curves("pr"); plot_curves("roc")

# confusion matrix for the selected model (test)
r = res[res["model"] == best].iloc[0]
cm = np.array([[r["test_TN"], r["test_FP"]], [r["test_FN"], r["test_TP"]]])
fig, ax = plt.subplots(figsize=(4.5, 4))
ax.imshow(cm, cmap="Blues"); ax.grid(False)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=12)
ax.set_xticks([0, 1]); ax.set_xticklabels(["no alert", "alert"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["no failure", "failure"])
ax.set(xlabel="predicted", ylabel="actual", title=f"{best}: test confusion matrix")
fig.tight_layout(); fig.savefig(FIG / "eval_confusion_matrix.png", dpi=300); plt.close(fig)

# threshold trade-off on VALIDATION for the selected model
pv = val[f"proba_{best}"]
prec, rec, thr = precision_recall_curve(val["y"], pv)
alerts_per_day = [(pv >= t).mean() * 100 * 24 for t in thr]       # alert machine-hours per day across 100 machines
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(thr, prec[:-1], label="precision"); ax.plot(thr, rec[:-1], label="recall")
ax.axvline(best_thr, color="k", ls="--", label="chosen threshold")
ax.set(xlabel="score threshold", ylabel="precision / recall", title=f"{best}: threshold trade-off (validation)")
ax.legend(); fig.tight_layout(); fig.savefig(FIG / "eval_threshold_tradeoff.png", dpi=300); plt.close(fig)

# calibration (test)
frac, mean_p = calibration_curve(test["y"], test[f"proba_{best}"], n_bins=10, strategy="quantile")
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(mean_p, frac, marker="o"); ax.plot([0, 1], [0, 1], "k:")
ax.set(xlabel="mean predicted score", ylabel="observed failure rate", title=f"{best}: calibration (test)")
fig.tight_layout(); fig.savefig(FIG / "eval_calibration.png", dpi=300); plt.close(fig)

# permutation importance on a validation sample (association, not causation)
pipe = joblib.load(MODELS / VERSION / f"{best}.joblib")
df = pd.read_parquet(FEATURES / "feature_table.parquet")
with open(FEATURES / "feature_meta.json") as f:
    feats = json.load(f)["feature_cols"]
samp = df[df["split"] == "val"].sample(40000, random_state=SEED)
pi = permutation_importance(pipe, samp[feats], samp["y"], scoring="average_precision",
                            n_repeats=3, random_state=SEED, n_jobs=1)
imp = pd.Series(pi.importances_mean, index=feats).sort_values(ascending=False)
imp.to_csv(REPORTS / "permutation_importance.csv", header=["drop_in_PR_AUC"])
print("\nTop 10 features by permutation importance (drop in validation PR-AUC):\n", imp.head(10).round(4).to_string())
top = imp.head(15)[::-1]
fig, ax = plt.subplots(figsize=(7, 5))
ax.barh(top.index, top.values); ax.set(xlabel="drop in PR-AUC when shuffled", title=f"{best}: top features")
fig.tight_layout(); fig.savefig(FIG / "eval_feature_importance.png", dpi=300); plt.close(fig)
print("\nSaved plots to reports/figures/eval_*.png and table to reports/model_comparison.csv")