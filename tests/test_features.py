from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT = Path(
    "/Users/santrokhatiwada/Documents/MAT/SEM 1/Data Analytics/Assignment 2/abi_pdm"
)

# Make src available to Python
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.config import SENSORS
from src.features import add_labels, build_features




def synth(n_hours=300, machines=(1, 2, 3), seed=0):
    rng = np.random.default_rng(seed)
    times = pd.date_range("2015-01-01 06:00", periods=n_hours, freq="h")
    tel = pd.concat([pd.DataFrame({"datetime": times, "machine_id": m,
                                   **{s: rng.normal(100, 10, n_hours) for s in SENSORS}})
                     for m in machines], ignore_index=True)
    errors = pd.DataFrame({"datetime": rng.choice(times, 40), "machine_id": rng.choice(machines, 40),
                           "error_id": rng.choice(["error1", "error2"], 40)}).drop_duplicates()
    early = pd.date_range("2014-12-01", periods=n_hours + 720, freq="h")      # includes pre-telemetry history
    maint = pd.DataFrame({"datetime": rng.choice(early, 20), "machine_id": rng.choice(machines, 20),
                          "component": rng.choice(["comp1", "comp2"], 20)})
    mac = pd.DataFrame({"machine_id": list(machines), "model": ["model1", "model2", "model1"], "age": [5, 10, 15]})
    return tel, errors, maint, mac


ET, CP = ["error1", "error2"], ["comp1", "comp2"]


def test_label_window_is_exact():
    times = pd.date_range("2015-01-01 00:00", periods=100, freq="h")
    df = pd.DataFrame({"machine_id": 1, "datetime": times})
    fail = pd.DataFrame({"machine_id": [1], "datetime": [times[50]]})
    out = add_labels(df, fail, 24)
    pos = out.loc[out["y"] == 1, "datetime"]
    assert pos.min() == times[26] and pos.max() == times[49]          # (t, t+24h] window
    assert out.loc[out["datetime"] == times[50], "failure_at_t"].iloc[0] == 1
    assert out.loc[out["datetime"] == times[75], "fully_observed"].iloc[0]
    assert not out.loc[out["datetime"] == times[76], "fully_observed"].iloc[0]   # censored tail


def test_features_unchanged_when_future_is_deleted():
    tel, err, mnt, mac = synth()
    cutoff = tel["datetime"].iloc[200]
    full = build_features(tel, err, mnt, mac, ET, CP)
    cut = build_features(tel[tel["datetime"] <= cutoff], err[err["datetime"] <= cutoff],
                         mnt[mnt["datetime"] <= cutoff], mac, ET, CP)
    a = full[full["datetime"] <= cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, cut.reset_index(drop=True), check_dtype=False)


def test_features_unchanged_when_future_sensors_are_corrupted():
    tel, err, mnt, mac = synth()
    cutoff = tel["datetime"].iloc[200]
    bad = tel.copy()
    bad.loc[bad["datetime"] > cutoff, SENSORS] = 1e6
    a = build_features(tel, err, mnt, mac, ET, CP)
    b = build_features(bad, err, mnt, mac, ET, CP)
    pd.testing.assert_frame_equal(a[a["datetime"] <= cutoff].reset_index(drop=True),
                                  b[b["datetime"] <= cutoff].reset_index(drop=True), check_dtype=False)


def test_no_row_multiplication_and_unique_keys():
    tel, err, mnt, mac = synth()
    out = build_features(tel, err, mnt, mac, ET, CP)
    assert not out.duplicated(["machine_id", "datetime"]).any()
    assert len(out) == len(tel) - 3 * 23            # only the 23 warm-up hours per machine removed


def test_time_since_maintenance_never_negative():
    tel, err, mnt, mac = synth()
    out = build_features(tel, err, mnt, mac, ET, CP)
    for c in ["hours_since_comp1_maint", "hours_since_comp2_maint"]:
        assert (out[c].dropna() >= 0).all()