from pathlib import Path
import sys

PROJECT = (
    Path.home()
    / "Documents"
    / "MAT"
    / "SEM 1"
    / "Data Analytics"
    / "Assignment 2"
    / "abi_pdm"
)
import subprocess
import sys
from pathlib import Path

PROJECT = Path(PROJECT)

notebook_path = PROJECT / "src" / "clean_data.ipynb"

result = subprocess.run(
    [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        str(notebook_path),
    ],
    cwd=PROJECT,
    capture_output=True,
    text=True,
)

print(result.stdout)

if result.returncode != 0:
    print(result.stderr)
    raise RuntimeError(
        "Data cleaning failed. Check the notebook execution error."
    )

print("Data cleaning completed successfully.")

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.clean_data import CleaningLog, clean_events, clean_machines, clean_telemetry

VALID = {1, 2, 3}
CLEAN = Path("/Users/santrokhatiwada/Documents/MAT/SEM 1/Data Analytics/Assignment 2/abi_pdm/data/clean")


def raw_errors(rows):
    return pd.DataFrame(rows, columns=["datetime", "machineID", "errorID"])


def raw_tel(rows):
    return pd.DataFrame(rows, columns=["datetime", "machineID", "volt", "rotate",
                                       "pressure", "vibration"])


# ----------------- unit tests on tiny synthetic data ------------------------
def test_invalid_datetime_row_removed():
    df = raw_errors([["2015-01-01 06:00:00", "1", "error1"], ["not a date", "1", "error2"]])
    out = clean_events(df, "errors", "error_id", "error", VALID, CleaningLog())
    assert len(out) == 1 and out.loc[0, "error_id"] == "error1"


def test_exact_duplicates_removed():
    row = ["2015-01-01 06:00:00", "1", "error1"]
    out = clean_events(raw_errors([row, row]), "errors", "error_id", "error", VALID, CleaningLog())
    assert len(out) == 1


def test_unknown_machine_is_quarantined_not_lost():
    log = CleaningLog()
    df = raw_errors([["2015-01-01 06:00:00", "99", "error1"]])
    out = clean_events(df, "errors", "error_id", "error", VALID, log)
    assert len(out) == 0 and len(log.quarantined) == 1


def test_text_is_normalised():
    df = raw_errors([["2015-01-01 06:00:00", "1", "  ERROR1 "]])
    out = clean_events(df, "errors", "error_id", "error", VALID, CleaningLog())
    assert out.loc[0, "error_id"] == "error1"


def test_missing_required_column_raises():
    df = pd.DataFrame({"datetime": ["2015-01-01 06:00:00"], "machineID": ["1"]})
    with pytest.raises(ValueError):
        clean_events(df, "errors", "error_id", "error", VALID, CleaningLog())


def test_outlier_kept_and_missing_not_imputed():
    df = raw_tel([["2015-01-01 06:00:00", "1", "170.0", "440.0", "100.0", "40.0"],
                  ["2015-01-01 07:00:00", "1", "9999.0", "440.0", np.nan, "40.0"]])
    out = clean_telemetry(df, VALID, CleaningLog())
    assert len(out) == 2
    assert out["volt"].max() == 9999.0          # extreme value retained
    assert out["pressure"].isna().sum() == 1    # missing value left as NaN


def test_telemetry_key_conflict_keeps_first_and_quarantines():
    log = CleaningLog()
    df = raw_tel([["2015-01-01 06:00:00", "1", "170.0", "440.0", "100.0", "40.0"],
                  ["2015-01-01 06:00:00", "1", "171.0", "441.0", "101.0", "41.0"]])
    out = clean_telemetry(df, VALID, log)
    assert len(out) == 1 and out.loc[0, "volt"] == 170.0
    assert any("conflicting_key" in label for label, _ in log.quarantined)


def test_duplicate_machine_id_in_machines():
    df = pd.DataFrame({"machineID": ["1", "1", "2"], "model": ["model3", "model4", "model3"],
                       "age": ["18", "7", "8"]})
    out = clean_machines(df, CleaningLog())
    assert out["machine_id"].tolist() == [1, 2]


# ----------------- checks on YOUR real cleaned files ------------------------
needs_data = pytest.mark.skipif(not (CLEAN / "telemetry_clean.parquet").exists(),
                                reason="run src/clean_data.ipynb first")


@needs_data
def test_real_telemetry_sorted_unique_typed():
    tel = pd.read_parquet(CLEAN / "telemetry_clean.parquet")
    keys = tel[["machine_id", "datetime"]]
    assert not keys.duplicated().any()
    assert keys.equals(keys.sort_values(["machine_id", "datetime"]).reset_index(drop=True))
    assert str(tel["datetime"].dtype).startswith("datetime64")


@needs_data
def test_real_all_machine_ids_exist_in_machines():
    ids = set(pd.read_parquet(CLEAN / "machines_clean.parquet")["machine_id"])
    for name in ["telemetry", "errors", "maint", "failures"]:
        got = set(pd.read_parquet(CLEAN / f"{name}_clean.parquet")["machine_id"])
        assert got <= ids, f"{name} has machine_ids not in machines"


@needs_data
def test_real_no_missing_keys_in_event_files():
    for name in ["errors", "maint", "failures"]:
        df = pd.read_parquet(CLEAN / f"{name}_clean.parquet")
        assert not df[["machine_id", "datetime"]].isna().any().any()