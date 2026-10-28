
from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

PROJECT = Path(
    "/Users/santrokhatiwada/Documents/MAT/SEM 1/Data Analytics/Assignment 2/abi_pdm"
)
# Make src importable
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

# Import configuration
from src.config import (
    CLEAN,
    ERROR_WINDOWS,
    HORIZON_HOURS,
    ROLL_SENSOR_WINDOWS,
    SENSORS,
)



def load_clean() -> dict:
    return {n: pd.read_parquet(CLEAN / f"{n}_clean.parquet")
            for n in ["telemetry", "errors", "maint", "failures", "machines"]}


def _hours_since_last(base: pd.DataFrame, ev: pd.DataFrame) -> pd.Series:
    """Hours since the latest event at or before each row's time (same machine). NaN if none."""
    if len(ev) == 0:
        return pd.Series(float("nan"), index=base.index)
    left = base[["machine_id", "datetime"]].reset_index().sort_values("datetime", kind="stable")
    right = (ev[["machine_id", "datetime"]].drop_duplicates()
             .rename(columns={"datetime": "event_time"}).assign(datetime=lambda d: d["event_time"])
             .sort_values("datetime"))
    merged = pd.merge_asof(left, right, on="datetime", by="machine_id", direction="backward")
    merged = merged.set_index("index").sort_index()
    return (merged["datetime"] - merged["event_time"]).dt.total_seconds() / 3600.0


def build_features(tel, errors, maint, machines, error_types=None, components=None) -> pd.DataFrame:
    """One row per machine-hour. Uses ONLY data at or before each row's timestamp."""
    base = (tel[["machine_id", "datetime"] + SENSORS]
            .sort_values(["machine_id", "datetime"]).reset_index(drop=True))
    n0 = len(base)
    error_types = list(error_types) if error_types is not None else sorted(errors["error_id"].unique())
    components = list(components) if components is not None else sorted(maint["component"].unique())
    new = {}

    # (a) rolling sensor statistics over PAST rows (window includes the current hour)
    g = base.groupby("machine_id")
    for c in SENSORS:
        for w in ROLL_SENSOR_WINDOWS:
            r = g[c].rolling(w, min_periods=w)
            new[f"{c}_mean_{w}h"] = r.mean().reset_index(level=0, drop=True)
            new[f"{c}_std_{w}h"] = r.std().reset_index(level=0, drop=True)

    # (b) error counts in trailing windows
    keys = base[["machine_id", "datetime"]]
    if len(errors) == 0:
        tmp = keys.copy()
        tmp[error_types] = 0
    else:
        e = errors.copy()
        e["datetime"] = e["datetime"].dt.ceil("h")      # first telemetry hour at/after the event
        hourly = (e.groupby(["machine_id", "datetime", "error_id"]).size()
                  .unstack("error_id", fill_value=0).reindex(columns=error_types, fill_value=0)
                  .reset_index())
        hourly.columns.name = None
        tmp = keys.merge(hourly, on=["machine_id", "datetime"], how="left", validate="one_to_one")
        tmp[error_types] = tmp[error_types].fillna(0)
    assert len(tmp) == n0, "error join changed the row count"
    gt = tmp.groupby("machine_id")
    for et in error_types:
        for w in ERROR_WINDOWS:
            new[f"{et}_count_{w}h"] = gt[et].rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)

    # (c) hours since last maintenance, per component (events at or before t only)
    for comp in components:
        ev = maint.loc[maint["component"] == comp, ["machine_id", "datetime"]]
        new[f"hours_since_{comp}_maint"] = _hours_since_last(base, ev)

    out = pd.concat([base, pd.DataFrame(new)], axis=1)
    out = out.merge(machines[["machine_id", "model", "age"]], on="machine_id", how="left", validate="many_to_one")
    assert len(out) == n0, "machine join changed the row count"
    assert not out.duplicated(["machine_id", "datetime"]).any(), "duplicate machine-hours"

    warm = [f"{c}_mean_{max(ROLL_SENSOR_WINDOWS)}h" for c in SENSORS]
    return out.dropna(subset=warm).reset_index(drop=True)      # drop warm-up rows


def add_labels(df: pd.DataFrame, failures: pd.DataFrame, horizon_hours: int = HORIZON_HOURS,
               last_ts: pd.Timestamp | None = None) -> pd.DataFrame:
    """Adds y plus audit columns. Audit columns use the FUTURE and must never be features."""
    df = df.reset_index(drop=True)
    last_ts = last_ts if last_ts is not None else df["datetime"].max()
    f = (failures[["machine_id", "datetime"]].drop_duplicates()
         .rename(columns={"datetime": "next_failure_time"}).sort_values("next_failure_time"))

    left = df[["machine_id", "datetime"]].reset_index().sort_values("datetime", kind="stable")
    m = pd.merge_asof(left, f, left_on="datetime", right_on="next_failure_time",
                      by="machine_id", direction="forward", allow_exact_matches=False)  # strictly after t
    m = m.set_index("index").sort_index()

    out = df.copy()
    out["audit_next_failure_time"] = m["next_failure_time"]
    out["audit_hours_to_next_failure"] = (m["next_failure_time"] - out["datetime"]).dt.total_seconds() / 3600.0
    out["y"] = (out["audit_hours_to_next_failure"] <= horizon_hours).astype(int)

    ft = f.rename(columns={"next_failure_time": "datetime"}).assign(failure_at_t=1)
    n0 = len(out)
    out = out.merge(ft, on=["machine_id", "datetime"], how="left", validate="many_to_one")
    assert len(out) == n0
    out["failure_at_t"] = out["failure_at_t"].fillna(0).astype(int)
    out["fully_observed"] = (out["datetime"] + pd.Timedelta(hours=horizon_hours)) <= last_ts
    return out