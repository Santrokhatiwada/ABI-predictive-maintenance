"""
Phase 3B: Investigate sensor outliers WITHOUT changing the data.

Reads  : data/clean/telemetry_clean.parquet, data/clean/failures_clean.parquet
Writes : reports/sensor_descriptive_stats.csv
         reports/outlier_summary.csv
         reports/outlier_vs_prefailure.csv
         reports/outliers_by_machine.csv
         reports/figures/sensor_boxplots.png   (300 dpi)

NOTE: 'pre_failure_24h' below is used ONLY to study the data here. It is an
exploratory label and is never used as a model feature.

Run:  python src/investigate_outliers.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CLEAN = Path("/Users/santrokhatiwada/Documents/MAT/SEM 1/Data Analytics/Assignment 2/abi_pdm/data/clean")
REPORTS = Path("/Users/santrokhatiwada/Documents/MAT/SEM 1/Data Analytics/Assignment 2/abi_pdm/reports")
FIG = REPORTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)
SENSORS = ["volt", "rotate", "pressure", "vibration"]

tel = pd.read_parquet(CLEAN / "telemetry_clean.parquet")
fail = pd.read_parquet(CLEAN / "failures_clean.parquet")

# Descriptive statistics, including extreme percentiles 
desc = tel[SENSORS].describe(percentiles=[.001, .01, .05, .25, .5, .75, .95, .99, .999]).T
desc["skew"] = tel[SENSORS].skew()
desc["n_negative"] = (tel[SENSORS] < 0).sum()
desc["n_zero"] = (tel[SENSORS] == 0).sum()
desc.to_csv(REPORTS / "sensor_descriptive_stats.csv")
print("=== DESCRIPTIVE STATISTICS ===")
print(desc.round(3).to_string())

#  Outlier counts: IQR rule (1.5x and 3x) and robust z-score 
bounds, rows = {}, []
for c in SENSORS:
    s = tel[c].dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    lo3, hi3 = q1 - 3.0 * iqr, q3 + 3.0 * iqr
    med = s.median()
    mad = (s - med).abs().median()            # median absolute deviation
    if mad > 0:
        rz = 0.6745 * (s - med) / mad         # "robust z-score"
        n_rz = int((rz.abs() > 3.5).sum())
    else:
        n_rz = np.nan
    n15 = int(((s < lo) | (s > hi)).sum())
    n3 = int(((s < lo3) | (s > hi3)).sum())
    rows.append({"sensor": c, "q1": q1, "q3": q3, "lower_fence_1.5": lo,
                 "upper_fence_1.5": hi, "n_beyond_1.5_IQR": n15,
                 "pct_beyond_1.5_IQR": round(100 * n15 / len(s), 3),
                 "n_beyond_3_IQR": n3, "n_robust_z_gt_3.5": n_rz,
                 "min": s.min(), "max": s.max()})
    bounds[c] = (lo, hi)
outlier_summary = pd.DataFrame(rows)
outlier_summary.to_csv(REPORTS / "outlier_summary.csv", index=False)
print("\n=== OUTLIER SUMMARY ===")
print(outlier_summary.round(3).to_string(index=False))

#  Do outliers cluster just before failures? 
t = tel[["machine_id", "datetime"] + SENSORS].sort_values("datetime")
f = (fail[["machine_id", "datetime"]].drop_duplicates()
     .rename(columns={"datetime": "next_failure"}).sort_values("next_failure"))
m = pd.merge_asof(t, f, left_on="datetime", right_on="next_failure",
                  by="machine_id", direction="forward")
assert len(m) == len(t), "merge_asof changed the number of telemetry rows"
m["hours_to_failure"] = (m["next_failure"] - m["datetime"]) / pd.Timedelta(hours=1)
m["pre_failure_24h"] = m["hours_to_failure"].between(0, 24)
print(f"\nReadings in the 24h before a failure: {int(m['pre_failure_24h'].sum())} "
      f"of {len(m)} ({100 * m['pre_failure_24h'].mean():.2f}%)")

cmp_rows = []
for c in SENSORS:
    lo, hi = bounds[c]
    m[f"{c}_outlier"] = (m[c] < lo) | (m[c] > hi)
    rate = m.groupby("pre_failure_24h")[f"{c}_outlier"].mean()
    med = m.groupby("pre_failure_24h")[c].median()
    r_norm, r_pre = rate.get(False, np.nan), rate.get(True, np.nan)
    cmp_rows.append({
        "sensor": c,
        "outlier_rate_normal_%": round(100 * r_norm, 3),
        "outlier_rate_prefailure_%": round(100 * r_pre, 3),
        "ratio_pre_vs_normal": round(r_pre / r_norm, 2) if r_norm else np.nan,
        "median_normal": round(med.get(False, np.nan), 3),
        "median_prefailure": round(med.get(True, np.nan), 3),
    })
cmp = pd.DataFrame(cmp_rows)
cmp.to_csv(REPORTS / "outlier_vs_prefailure.csv", index=False)
print("\n=== OUTLIER RATE: NORMAL vs 24h BEFORE FAILURE ===")
print(cmp.to_string(index=False))

#  Are outliers concentrated in a few machines? -
by_machine = m.groupby("machine_id")[[f"{c}_outlier" for c in SENSORS]].sum()
by_machine["total"] = by_machine.sum(axis=1)
by_machine = by_machine.sort_values("total", ascending=False)
by_machine.to_csv(REPORTS / "outliers_by_machine.csv")
print("\n=== TOP 5 MACHINES BY OUTLIER COUNT ===")
print(by_machine.head(5).to_string())

# Boxplots (report-quality) 
fig, axes = plt.subplots(1, 4, figsize=(12, 4))
for ax, c in zip(axes, SENSORS):
    ax.boxplot(tel[c].dropna(), whis=1.5, flierprops={"markersize": 1, "alpha": 0.3})
    ax.set_title(c)
    ax.set_xticks([])
    ax.set_ylabel("reading (units not documented)")
fig.suptitle("Sensor distributions: all machines, all hours (whiskers = 1.5 x IQR)")
fig.tight_layout()
fig.savefig(FIG / "sensor_boxplots.png", dpi=300, bbox_inches="tight")
print(f"\nSaved {FIG / 'sensor_boxplots.png'}")