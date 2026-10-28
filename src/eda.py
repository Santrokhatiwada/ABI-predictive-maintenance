"""Phase 4: Exploratory data analysis. Run: python -m src.eda"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import CLEAN, FIG, SENSORS

plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3})
FIG.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", FIG / name)


tel = pd.read_parquet(CLEAN / "telemetry_clean.parquet")
err = pd.read_parquet(CLEAN / "errors_clean.parquet")
mnt = pd.read_parquet(CLEAN / "maint_clean.parquet")
fail = pd.read_parquet(CLEAN / "failures_clean.parquet")
mac = pd.read_parquet(CLEAN / "machines_clean.parquet")

# ---------------- 0. Numeric facts (these go in your report) ----------------
print("=== ROW COUNTS ===")
for n, d in [("telemetry", tel), ("errors", err), ("maint", mnt), ("failures", fail), ("machines", mac)]:
    print(f"{n:10} {len(d):>8}")
print("\nmachine age:\n", mac["age"].describe().round(2).to_string())
print("\nmachines per model:\n", mac["model"].value_counts().sort_index().to_string())
print("\nerrors by type:\n", err["error_id"].value_counts().sort_index().to_string())
print("\nfailures by component:\n", fail["failed_component"].value_counts().sort_index().to_string())
print("\nmaintenance by component:\n", mnt["component"].value_counts().sort_index().to_string())
print("\nfailures by month:\n", fail.groupby(fail["datetime"].dt.to_period("M")).size().to_string())
fpm = fail.groupby("machine_id").size().reindex(mac["machine_id"], fill_value=0)
print("\nfailures per machine:\n", fpm.describe().round(2).to_string())
print("machines with zero failures:", int((fpm == 0).sum()))
print("rows with a failure sharing machine+timestamp with another failure row:",
      int(fail.duplicated(["machine_id", "datetime"], keep=False).sum()))
print("maintenance events before telemetry start:", int((mnt["datetime"] < tel["datetime"].min()).sum()))
print("\nMISSING VALUES per table (all tables):")
for n, d in [("telemetry", tel), ("errors", err), ("maint", mnt), ("failures", fail), ("machines", mac)]:
    print(f"  {n}: {int(d.isna().sum().sum())} missing cells")

# ---------------- Fig 1: sensor histograms (telemetry) ----------------
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, c in zip(axes.ravel(), SENSORS):
    ax.hist(tel[c], bins=60, color="#4C72B0")
    ax.set_title(c); ax.set_xlabel("reading (units not documented)"); ax.set_ylabel("machine-hours")
fig.suptitle("Distribution of sensor readings (all machines, all hours)")
save(fig, "eda01_sensor_histograms.png")

# ---------------- Fig 2: sensor trend for the machine with most failures ----------------
m_id = int(fail["machine_id"].value_counts().idxmax())
comps = sorted(fail["failed_component"].unique())
colors = dict(zip(comps, plt.cm.tab10.colors))
daily = tel[tel["machine_id"] == m_id].set_index("datetime")[SENSORS].resample("D").mean()
fm = fail[fail["machine_id"] == m_id]
fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
for ax, c in zip(axes, SENSORS):
    ax.plot(daily.index, daily[c], lw=1)
    ax.set_ylabel(c)
    for r in fm.itertuples():
        ax.axvline(r.datetime, color=colors[r.failed_component], alpha=0.7, lw=1)
axes[0].legend([Line2D([0], [0], color=colors[c]) for c in comps], comps, ncol=4, loc="upper right")
axes[0].set_title(f"Daily mean sensor readings, machine {m_id} (vertical lines = failures)")
save(fig, "eda02_sensor_trend_machine.png")

# ---------------- Fig 3: machine age and model ----------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(mac["age"], bins=range(int(mac["age"].min()), int(mac["age"].max()) + 2), edgecolor="white")
axes[0].set_xlabel("age (units not stated in files)"); axes[0].set_ylabel("machines"); axes[0].set_title("Machine age")
mac["model"].value_counts().sort_index().plot.bar(ax=axes[1], rot=0)
axes[1].set_xlabel("model"); axes[1].set_ylabel("machines"); axes[1].set_title("Machines per model")
save(fig, "eda03_machine_age_model.png")

# ---------------- Fig 4: error events ----------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
err["error_id"].value_counts().sort_index().plot.bar(ax=axes[0], rot=0)
axes[0].set_ylabel("events"); axes[0].set_title("Error events by type")
epm = err.groupby("machine_id").size().reindex(mac["machine_id"], fill_value=0)
axes[1].hist(epm, bins=20, edgecolor="white")
axes[1].set_xlabel("error events in the year"); axes[1].set_ylabel("machines"); axes[1].set_title("Errors per machine")
save(fig, "eda04_error_frequency.png")

# ---------------- Fig 5: failures by component and by month ----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fail["failed_component"].value_counts().sort_index().plot.bar(ax=axes[0], rot=0)
axes[0].set_ylabel("failures"); axes[0].set_title("Failures by component")
fm_tab = (fail.assign(month=fail["datetime"].dt.to_period("M").astype(str))
          .groupby(["month", "failed_component"]).size().unstack(fill_value=0))
fm_tab.plot.bar(stacked=True, ax=axes[1], rot=45)
axes[1].set_ylabel("failures"); axes[1].set_title("Failures per month")
save(fig, "eda05_failure_frequency.png")

# ---------------- Fig 6: failures per machine by model ----------------
per_machine = fpm.rename("n_failures").reset_index().merge(mac, on="machine_id")
fig, ax = plt.subplots(figsize=(6, 4))
per_machine.groupby("model")["n_failures"].mean().plot.bar(ax=ax, rot=0)
ax.set_ylabel("mean failures per machine"); ax.set_title("Failures per machine, by model")
save(fig, "eda06_failures_by_model.png")
print("\nfailures per machine by model:\n", per_machine.groupby("model")["n_failures"].agg(["mean", "std", "count"]).round(2).to_string())
print("\ncorr(age, failures per machine):", round(per_machine["age"].corr(per_machine["n_failures"]), 3))

# ---------------- Fig 7: maintenance events ----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
mnt["component"].value_counts().sort_index().plot.bar(ax=axes[0], rot=0)
axes[0].set_ylabel("events"); axes[0].set_title("Maintenance events by component")
mnt.groupby(mnt["datetime"].dt.to_period("M")).size().plot(ax=axes[1], marker="o")
axes[1].set_ylabel("events"); axes[1].set_title("Maintenance events per month")
save(fig, "eda07_maintenance_frequency.png")

# ---------------- Fig 8: sensor profile in the 72h before a failure ----------------
t = tel.sort_values("datetime")
f = (fail[["machine_id", "datetime"]].drop_duplicates()
     .rename(columns={"datetime": "next_failure"}).sort_values("next_failure"))
m = pd.merge_asof(t, f, left_on="datetime", right_on="next_failure", by="machine_id", direction="forward")
assert len(m) == len(t)
m["h"] = (m["next_failure"] - m["datetime"]) / pd.Timedelta(hours=1)
z = (m[SENSORS] - m[SENSORS].mean()) / m[SENSORS].std()      # standardise so sensors share one axis
sel = m["h"].between(0, 72)
prof = z[sel].groupby(np.floor(m.loc[sel, "h"])).mean()
fig, ax = plt.subplots(figsize=(9, 4.5))
for c in SENSORS:
    ax.plot(prof.index, prof[c], label=c)
ax.axhline(0, color="k", lw=0.8)
ax.axvline(24, color="red", ls="--", lw=1)
ax.invert_xaxis()
ax.set_xlabel("hours before failure (time flows to the right)")
ax.set_ylabel("mean standardised reading (0 = overall mean)")
ax.set_title("Average sensor behaviour before failures"); ax.legend()
save(fig, "eda08_prefailure_profile.png")
print("\nstandardised mean by hours-before-failure (selected):\n",
      prof.loc[[0, 6, 12, 24, 48, 72]].round(3).to_string())

# ---------------- Fig 9: Spearman correlation ----------------
cdat = m[SENSORS].assign(prefailure_24h=m["h"].between(0, 24).astype(int))
corr = cdat.corr(method="spearman")
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns); ax.grid(False)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(im); ax.set_title("Spearman correlation")
save(fig, "eda09_correlation.png")
print("\nSpearman correlation:\n", corr.round(3).to_string())

# ---------------- Interactive Plotly chart ----------------
d = tel[tel["machine_id"] == m_id]
pf = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=SENSORS)
for i, c in enumerate(SENSORS, 1):
    pf.add_trace(go.Scatter(x=d["datetime"], y=d[c], mode="lines", line=dict(width=1), name=c), row=i, col=1)
for r in fm.itertuples():
    pf.add_shape(type="line", xref="x", yref="paper", x0=r.datetime, x1=r.datetime,
                 y0=0, y1=1, line=dict(color="red", dash="dot", width=1))
pf.update_layout(height=800, title=f"Machine {m_id}: hourly sensors (red = failures)", showlegend=False)
pf.write_html(FIG / "interactive_machine.html", include_plotlyjs=True)
print("saved", FIG / "interactive_machine.html")