"""
Phase 3A: Clean the five raw PdM CSV files.

Reads  : data/raw/*.csv                (never modified)
Writes : data/clean/*_clean.parquet    (cleaned tables)
         data/clean/quarantine_*.csv   (rows removed for a problem, kept for audit)
         reports/cleaning_log.csv      (before/after row counts per step)
         reports/data_quality_report.json (facts measured on YOUR data)

Run from the project root:  python src/clean_data.ipynb
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("/Users/santrokhatiwada/Documents/MAT/SEM 1/Data Analytics/Assignment 2/abi_pdm/data/raw")
CLEAN = Path("/Users/santrokhatiwada/Documents/MAT/SEM 1/Data Analytics/Assignment 2/abi_pdm/data/clean")
REPORTS = Path("/Users/santrokhatiwada/Documents/MAT/SEM 1/Data Analytics/Assignment 2/abi_pdm/reports")

RAW_FILES = {
    "telemetry": "PdM_telemetry.csv",
    "errors": "PdM_errors.csv",
    "maint": "PdM_maint.csv",
    "failures": "PdM_failures.csv",
    "machines": "PdM_machines.csv",
}
RENAME = {
    "telemetry": {"machineID": "machine_id"},
    "errors": {"machineID": "machine_id", "errorID": "error_id"},
    "maint": {"machineID": "machine_id", "comp": "component"},
    "failures": {"machineID": "machine_id", "failure": "failed_component"},
    "machines": {"machineID": "machine_id"},
}
EXPECTED_COLS = {
    "telemetry": ["datetime", "machine_id", "volt", "rotate", "pressure", "vibration"],
    "errors": ["datetime", "machine_id", "error_id"],
    "maint": ["datetime", "machine_id", "component"],
    "failures": ["datetime", "machine_id", "failed_component"],
    "machines": ["machine_id", "model", "age"],
}
SENSORS = ["volt", "rotate", "pressure", "vibration"]
ISO_FMT = "%Y-%m-%d %H:%M:%S"


# Logging helper: every cleaning decision is recorded here
class CleaningLog:
    """Collects step-by-step row counts, measured facts and quarantined rows."""

    def __init__(self):
        self.steps = []        # list of dicts -> reports/cleaning_log.csv
        self.facts = {}        # measured facts -> reports/data_quality_report.json
        self.quarantined = []  # list of (label, DataFrame)

    def record(self, file, step, before, after, decision):
        self.steps.append({
            "file": file, "step": step,
            "rows_before": int(before), "rows_after": int(after),
            "rows_removed": int(before - after), "decision": decision,
        })

    def fact(self, file, key, value):
        self.facts[f"{file}.{key}"] = value

    def quarantine(self, file, label, df):
        if len(df) > 0:
            self.quarantined.append((f"{file}_{label}", df.copy()))

# Small reusable helpers

def standardise_columns(df, file, log):
    """Strip header whitespace, rename to snake_case, check required columns exist."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=RENAME[file])
    missing = [c for c in EXPECTED_COLS[file] if c not in df.columns]
    if missing:
        raise ValueError(
            f"{file}: missing expected columns {missing}. Found {list(df.columns)}"
        )
    extra = [c for c in df.columns if c not in EXPECTED_COLS[file]]
    if extra:
        log.fact(file, "extra_columns_ignored", extra)
    return df[EXPECTED_COLS[file]].copy()


def parse_datetime(raw, file, log):
    """Parse text to timestamps. Try the ISO format first; fall back per-value."""
    parsed = pd.to_datetime(raw, format=ISO_FMT, errors="coerce")
    retry = parsed.isna() & raw.notna()
    n_retry = int(retry.sum())
    if n_retry:
        parsed.loc[retry] = pd.to_datetime(raw[retry], format="mixed", errors="coerce")
    log.fact(file, "datetime_needed_fallback_parser", n_retry)
    log.fact(file, "datetime_missing_or_unparseable", int(parsed.isna().sum()))
    return parsed


def to_numeric_checked(series, file, col, log):
    """Convert text to numbers. Count values that were present but not numeric."""
    num = pd.to_numeric(series, errors="coerce")
    log.fact(file, f"{col}_missing_in_raw", int(series.isna().sum()))
    log.fact(file, f"{col}_non_numeric_text", int((series.notna() & num.isna()).sum()))
    return num


def normalise_text(series, file, col, log):
    """Strip spaces and lower-case a categorical column; count how many values changed."""
    cleaned = series.str.strip().str.lower()
    changed = int((series.notna() & (series != cleaned)).sum())
    log.fact(file, f"{col}_values_changed_by_text_normalisation", changed)
    return cleaned


def drop_rows(df, mask, file, step, reason, log):
    """Drop rows where mask is True and log the before/after counts."""
    before = len(df)
    out = df.loc[~mask].copy()
    log.record(file, step, before, len(out), reason)
    return out


# Per-file cleaning functions

def clean_machines(raw_df, log):
    file = "machines"
    df = standardise_columns(raw_df, file, log)
    log.record(file, "loaded", len(df), len(df), "raw rows read as text")

    df["machine_id"] = to_numeric_checked(df["machine_id"], file, "machine_id", log)
    df["age"] = to_numeric_checked(df["age"], file, "age", log)
    df["model"] = normalise_text(df["model"], file, "model", log)

    # machine_id must be a whole number. (NaN % 1 is NaN, and NaN == 0 is False, so
    # this also catches missing IDs.)
    bad_id = ~((df["machine_id"] % 1) == 0)
    log.quarantine(file, "bad_machine_id", df[bad_id])
    df = drop_rows(df, bad_id, file, "drop_bad_machine_id",
                   "machine_id missing or not a whole number: row unusable", log)
    df["machine_id"] = df["machine_id"].astype("int64")

    df = drop_rows(df, df.duplicated(), file, "drop_exact_duplicates",
                   "identical rows carry no extra information", log)

    dup_id = df.duplicated("machine_id", keep="first")
    log.quarantine(file, "conflicting_machine_id", df[df.duplicated("machine_id", keep=False)])
    df = drop_rows(df, dup_id, file, "drop_duplicate_machine_id",
                   "machine_id must be unique; first row kept, all copies quarantined", log)

    # Flag-only checks (no deletion): impossible or missing values are reported
    log.fact(file, "age_missing", int(df["age"].isna().sum()))
    log.fact(file, "age_negative", int((df["age"] < 0).sum()))
    log.fact(file, "age_min", float(df["age"].min()))
    log.fact(file, "age_max", float(df["age"].max()))
    log.fact(file, "model_missing", int(df["model"].isna().sum()))
    log.fact(file, "model_value_counts", df["model"].value_counts().sort_index().to_dict())

    if df["age"].notna().all() and ((df["age"] % 1) == 0).all():
        df["age"] = df["age"].astype("int64")

    return df.sort_values("machine_id").reset_index(drop=True)


def clean_telemetry(raw_df, valid_ids, log):
    file = "telemetry"
    df = standardise_columns(raw_df, file, log)
    log.record(file, "loaded", len(df), len(df), "raw rows read as text")

    df["machine_id"] = to_numeric_checked(df["machine_id"], file, "machine_id", log)
    df["datetime"] = parse_datetime(df["datetime"], file, log)
    for c in SENSORS:
        df[c] = to_numeric_checked(df[c], file, c, log)
        is_inf = np.isinf(df[c])
        log.fact(file, f"{c}_infinite_values_set_to_nan", int(is_inf.sum()))
        df.loc[is_inf, c] = np.nan   # infinity is not a measurement

    bad_key = ~((df["machine_id"] % 1) == 0) | df["datetime"].isna()
    log.quarantine(file, "bad_key", df[bad_key])
    df = drop_rows(df, bad_key, file, "drop_unusable_key",
                   "machine_id or datetime missing/invalid: cannot place reading in time", log)
    df["machine_id"] = df["machine_id"].astype("int64")

    unknown = ~df["machine_id"].isin(valid_ids)
    log.quarantine(file, "unknown_machine", df[unknown])
    df = drop_rows(df, unknown, file, "drop_unknown_machine",
                   "machine_id not in machines table: cannot join metadata", log)

    df = drop_rows(df, df.duplicated(), file, "drop_exact_duplicates",
                   "identical rows carry no extra information", log)

    # Business key: one reading per machine per hour.
    # Rows that share a key but differ in values are CONFLICTS: quarantine all, keep first.
    key = ["machine_id", "datetime"]
    log.quarantine(file, "conflicting_key", df[df.duplicated(key, keep=False)])
    df = drop_rows(df, df.duplicated(key, keep="first"), file, "drop_conflicting_key",
                   "same machine+hour, different values: first kept, all copies quarantined", log)

    # Report-only checks: nothing is changed here
    for c in SENSORS:
        log.fact(file, f"{c}_missing_after_cleaning", int(df[c].isna().sum()))
        log.fact(file, f"{c}_negative_values", int((df[c] < 0).sum()))
    log.fact(file, "rows_with_all_sensors_missing", int(df[SENSORS].isna().all(axis=1).sum()))
    off_grid = (df["datetime"].dt.minute != 0) | (df["datetime"].dt.second != 0)
    log.fact(file, "timestamps_not_on_the_hour", int(off_grid.sum()))

    # Continuity: does each machine have an unbroken hourly series?
    g = df.groupby("machine_id")["datetime"].agg(["min", "max", "count"])
    expected = ((g["max"] - g["min"]) / pd.Timedelta(hours=1)).round().astype(int) + 1
    missing_hours = expected - g["count"]
    log.fact(file, "machines_with_telemetry", int(len(g)))
    log.fact(file, "machines_with_hour_gaps", int((missing_hours > 0).sum()))
    log.fact(file, "total_missing_hours", int(missing_hours.sum()))
    log.fact(file, "rows_per_machine_min", int(g["count"].min()))
    log.fact(file, "rows_per_machine_max", int(g["count"].max()))
    log.fact(file, "first_timestamp", str(df["datetime"].min()))
    log.fact(file, "last_timestamp", str(df["datetime"].max()))

    # Sensor outliers are deliberately NOT touched here (see 3B).
    return df.sort_values(key).reset_index(drop=True)


def clean_events(raw_df, file, cat_col, cat_prefix, valid_ids, log):
    """Cleans errors / maint / failures (all share: datetime, machine_id, one category)."""
    df = standardise_columns(raw_df, file, log)
    log.record(file, "loaded", len(df), len(df), "raw rows read as text")

    df["machine_id"] = to_numeric_checked(df["machine_id"], file, "machine_id", log)
    df["datetime"] = parse_datetime(df["datetime"], file, log)
    df[cat_col] = normalise_text(df[cat_col], file, cat_col, log)

    unusable = (~((df["machine_id"] % 1) == 0)) | df["datetime"].isna() | df[cat_col].isna()
    log.quarantine(file, "missing_key_fields", df[unusable])
    df = drop_rows(df, unusable, file, "drop_missing_key_fields",
                   "an event without machine, time or type cannot be used", log)
    df["machine_id"] = df["machine_id"].astype("int64")

    unknown = ~df["machine_id"].isin(valid_ids)
    log.quarantine(file, "unknown_machine", df[unknown])
    df = drop_rows(df, unknown, file, "drop_unknown_machine",
                   "machine_id not in machines table", log)

    # For these files the business key is ALL columns (machine, time, type), so
    # exact duplicates and key duplicates are the same thing.
    df = drop_rows(df, df.duplicated(), file, "drop_duplicate_events",
                   "same machine, time and type recorded twice", log)

    # Report-only checks
    pattern = rf"^{cat_prefix}\d+$"
    bad_cat = ~df[cat_col].str.match(pattern)
    log.fact(file, f"{cat_col}_not_matching_{cat_prefix}N", int(bad_cat.sum()))
    log.fact(file, f"{cat_col}_value_counts", df[cat_col].value_counts().sort_index().to_dict())
    shared = df.duplicated(["machine_id", "datetime"], keep=False)
    log.fact(file, "rows_sharing_machine_and_timestamp_with_another_row", int(shared.sum()))
    log.fact(file, "first_timestamp", str(df["datetime"].min()))
    log.fact(file, "last_timestamp", str(df["datetime"].max()))

    return df.sort_values(["machine_id", "datetime", cat_col]).reset_index(drop=True)

# Cross-file consistency checks (report-only)

def cross_file_checks(tables, log):
    tel = tables["telemetry"]
    t_min, t_max = tel["datetime"].min(), tel["datetime"].max()
    tel_keys = tel[["machine_id", "datetime"]].assign(_in_telemetry=True)

    for name in ["errors", "maint", "failures"]:
        ev = tables[name]
        log.fact("cross", f"{name}_rows", int(len(ev)))
        log.fact("cross", f"{name}_before_telemetry_start", int((ev["datetime"] < t_min).sum()))
        log.fact("cross", f"{name}_after_telemetry_end", int((ev["datetime"] > t_max).sum()))

        merged = ev.merge(tel_keys, on=["machine_id", "datetime"], how="left")
        # telemetry keys are unique, so this merge must not multiply rows:
        assert len(merged) == len(ev), f"{name}: merge multiplied rows"
        log.fact("cross", f"{name}_without_telemetry_row_at_same_machine_hour",
                 int(merged["_in_telemetry"].isna().sum()))

    machine_ids = set(tables["machines"]["machine_id"])
    log.fact("cross", "machines_without_telemetry",
             int(len(machine_ids - set(tel["machine_id"]))))
    for name in ["errors", "maint", "failures"]:
        log.fact("cross", f"machines_with_no_{name}",
                 int(len(machine_ids - set(tables[name]["machine_id"]))))


def _json_default(o):
    return o.item() if hasattr(o, "item") else str(o)


# Main
def main():
    CLEAN.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    for old in CLEAN.glob("quarantine_*.csv"):   # avoid stale files from earlier runs
        old.unlink()

    log = CleaningLog()

    raw = {}
    for name, fname in RAW_FILES.items():
        path = RAW / fname
        if not path.exists():
            raise SystemExit(f"Missing raw file: {path}")
        raw[name] = pd.read_csv(path, dtype=str)   # everything as text on purpose
        log.fact(name, "raw_rows", int(len(raw[name])))

    machines = clean_machines(raw["machines"], log)
    valid_ids = set(machines["machine_id"])

    tables = {
        "machines": machines,
        "telemetry": clean_telemetry(raw["telemetry"], valid_ids, log),
        "errors": clean_events(raw["errors"], "errors", "error_id", "error", valid_ids, log),
        "maint": clean_events(raw["maint"], "maint", "component", "comp", valid_ids, log),
        "failures": clean_events(raw["failures"], "failures", "failed_component",
                                 "comp", valid_ids, log),
    }
    cross_file_checks(tables, log)

    # Save cleaned tables (parquet keeps data types)
    for name, df in tables.items():
        df.to_parquet(CLEAN / f"{name}_clean.parquet", index=False)
        log.record(name, "saved", len(df), len(df), f"data/clean/{name}_clean.parquet")
    for label, qdf in log.quarantined:
        qdf.to_csv(CLEAN / f"quarantine_{label}.csv", index=False)

    pd.DataFrame(log.steps).to_csv(REPORTS / "cleaning_log.csv", index=False)
    with open(REPORTS / "data_quality_report.json", "w") as f:
        json.dump(log.facts, f, indent=2, default=_json_default)

    # Reload check: can every cleaned file be read back identically?
    for name, df in tables.items():
        back = pd.read_parquet(CLEAN / f"{name}_clean.parquet")
        assert back.shape == df.shape, f"{name}: shape changed on reload"
        assert list(back.columns) == list(df.columns), f"{name}: columns changed"
        assert back.isna().sum().to_dict() == df.isna().sum().to_dict(), f"{name}: nulls changed"
    print("RELOAD CHECK PASSED for all 5 cleaned files\n")

    pd.set_option("display.width", 200, "display.max_colwidth", 70)
    print("=== CLEANING LOG (rows before/after each step) ===")
    print(pd.DataFrame(log.steps).to_string(index=False))
    print("\n=== MEASURED FACTS ===")
    for k, v in sorted(log.facts.items()):
        print(f"{k}: {v}")
    print(f"\nQuarantine files written: {len(log.quarantined)}")
    for label, qdf in log.quarantined:
        print(f"  quarantine_{label}.csv  ({len(qdf)} rows)")


if __name__ == "__main__":
    main()