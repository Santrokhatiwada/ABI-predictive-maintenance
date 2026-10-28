
from pathlib import Path
import sys
import json
import pandas as pd

PROJECT = Path(
    "/Users/santrokhatiwada/Documents/MAT/SEM 1/Data Analytics/Assignment 2/abi_pdm"
)

if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.config import FEATURES, HORIZON_HOURS, assign_split
from src.features import add_labels, build_features, load_clean

FEATURES.mkdir(parents=True, exist_ok=True)
t = load_clean()
error_types = sorted(t["errors"]["error_id"].unique())
components = sorted(t["maint"]["component"].unique())

feats = build_features(t["telemetry"], t["errors"], t["maint"], t["machines"], error_types, components)
print("rows after feature build (warm-up removed):", len(feats))
df = add_labels(feats, t["failures"], HORIZON_HOURS, last_ts=t["telemetry"]["datetime"].max())

n_unobserved = int((~df["fully_observed"]).sum())
n_fail_now = int(((df["failure_at_t"] == 1) & df["fully_observed"]).sum())
model_df = df[df["fully_observed"] & (df["failure_at_t"] == 0)].copy()
model_df["split"] = assign_split(model_df["datetime"])
print(f"dropped: {n_unobserved} rows without 24h of future data; {n_fail_now} rows at the failure moment")
print("final rows:", len(model_df))

non_feature = {"machine_id", "datetime", "y", "failure_at_t", "fully_observed", "split"}
feature_cols = [c for c in model_df.columns if c not in non_feature and not c.startswith("audit_")]

# ---------------- LEAKAGE AUDIT ----------------
banned = ("next", "fail", "audit", "hours_to")
assert not [c for c in feature_cols if any(b in c.lower() for b in banned)], "suspicious feature name"
assert "machine_id" not in feature_cols and "y" not in feature_cols
corr = model_df[feature_cols].select_dtypes("number").corrwith(model_df["y"]).abs().sort_values(ascending=False)
print("\nTop 5 |correlation| with label (investigate anything > 0.5):\n", corr.head(5).round(3).to_string())
assert corr.max() < 0.9, "a feature is almost identical to the label: probable leakage"

# ---------------- SUMMARY BY SPLIT ----------------
summ = model_df.groupby("split").agg(rows=("y", "size"), positives=("y", "sum"),
                                     first=("datetime", "min"), last=("datetime", "max"))
summ["positive_rate_%"] = (100 * summ["positives"] / summ["rows"]).round(3)
summ["failure_events"] = (model_df[model_df["y"] == 1]
                          .drop_duplicates(["machine_id", "audit_next_failure_time"])
                          .groupby("split").size())
print("\n", summ.to_string())
print("\nmissing values in feature columns (hours_since_* may legitimately be NaN):")
print(model_df[feature_cols].isna().sum()[lambda s: s > 0].to_string() or "none")

model_df.to_parquet(FEATURES / "feature_table.parquet", index=False)
with open(FEATURES / "feature_meta.json", "w") as f:
    json.dump({"feature_cols": feature_cols, "error_types": error_types, "components": components,
               "horizon_hours": HORIZON_HOURS}, f, indent=2)
print(f"\nsaved feature_table.parquet ({len(feature_cols)} feature columns) and feature_meta.json")