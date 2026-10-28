"""Phase 10 setup: cut the held-back data (after LIVE_CUTOFF) into weekly 'new data' files.
Files use the ORIGINAL raw column names, so they can be uploaded in the dashboard exactly like a user's file.
Run:  python -m src.make_stream            (add --no-drift for a stationary stream)

SYNTHETIC DRIFT: from batch DRIFT_START_BATCH the `volt` readings get a constant offset added (a simulated
sensor recalibration). It is injected on purpose to test monitoring, and is recorded in manifest.json."""
from __future__ import annotations

import argparse
import json

import pandas as pd

from src import abi_config as C
from src.config import CLEAN

FMT = "%Y-%m-%d %H:%M:%S"


def _write(df: pd.DataFrame, path) -> None:
    out = df.copy()
    out["datetime"] = out["datetime"].dt.strftime(FMT)
    out.to_csv(path, index=False)


def make_variants(t: dict, manifest: list) -> None:
    """Two demo variants of batch 2 (same week, altered on purpose; both are SYNTHETIC):
       * batch_02_telemetry_SPIKE.csv : one machine's last 24 hours get an abnormal sensor pattern
       * batch_02_errors_BURST.csv    : another machine gets a burst of error events in its last 6 hours
    Use them INSTEAD of the normal batch_02 telemetry / errors file."""
    b2 = manifest[1]
    start, end = pd.Timestamp(b2["start"]), pd.Timestamp(b2["end"])
    f = t["failures"]
    near = set(f[(f["datetime"] >= start - pd.Timedelta(days=3)) & (f["datetime"] <= end + pd.Timedelta(days=3))]["machine_id"])
    quiet = [m for m in sorted(t["telemetry"]["machine_id"].unique()) if m not in near]
    spike_m, burst_m = int(quiet[0]), int(quiet[1])
    tel = t["telemetry"][(t["telemetry"]["datetime"] >= start) & (t["telemetry"]["datetime"] <= end)].copy()
    last24 = (tel["machine_id"] == spike_m) & (tel["datetime"] > end - pd.Timedelta(hours=24))
    tel.loc[last24, "volt"] += 50
    tel.loc[last24, "rotate"] -= 120
    tel.loc[last24, "pressure"] += 25
    tel.loc[last24, "vibration"] += 12
    _write(tel.rename(columns={"machine_id": "machineID"}), C.STREAM_DIR / "batch_02_telemetry_SPIKE.csv")
    err = t["errors"][(t["errors"]["datetime"] >= start) & (t["errors"]["datetime"] <= end)].copy()
    burst = pd.DataFrame({"datetime": [end - pd.Timedelta(hours=h) for h in (5, 4, 3, 2, 1)],
                          "machine_id": burst_m, "error_id": ["error1", "error2", "error3", "error4", "error5"]})
    err = pd.concat([err, burst], ignore_index=True).sort_values(["machine_id", "datetime"])
    _write(err.rename(columns={"machine_id": "machineID", "error_id": "errorID"}), C.STREAM_DIR / "batch_02_errors_BURST.csv")
    manifest[1]["variants"] = {"spike_machine": spike_m, "error_burst_machine": burst_m}


def main(drift: bool = True) -> None:
    C.STREAM_DIR.mkdir(parents=True, exist_ok=True)
    for old in list(C.STREAM_DIR.glob("*.csv")) + list(C.STREAM_DIR.glob("*.json")):
        old.unlink()
    t = {n: pd.read_parquet(CLEAN / f"{n}_clean.parquet") for n in ["telemetry", "errors", "maint", "failures"]}
    end_all = t["telemetry"]["datetime"].max()
    step = pd.Timedelta(hours=C.BATCH_HOURS)
    manifest, k, start = [], 1, C.LIVE_CUTOFF
    while start <= end_all:
        end = start + step                                   # half-open window [start, end)
        sl = lambda d: d[(d["datetime"] >= start) & (d["datetime"] < end)].copy()
        tel, err, mnt, fail = sl(t["telemetry"]), sl(t["errors"]), sl(t["maint"]), sl(t["failures"])
        drifted = bool(drift and k >= C.DRIFT_START_BATCH)
        if drifted:
            tel[C.DRIFT_SENSOR] = tel[C.DRIFT_SENSOR] + C.DRIFT_OFFSET
        tag = f"batch_{k:02d}"
        _write(tel.rename(columns={"machine_id": "machineID"}), C.STREAM_DIR / f"{tag}_telemetry.csv")
        _write(err.rename(columns={"machine_id": "machineID", "error_id": "errorID"}), C.STREAM_DIR / f"{tag}_errors.csv")
        _write(mnt.rename(columns={"machine_id": "machineID", "component": "comp"}), C.STREAM_DIR / f"{tag}_maint.csv")
        _write(fail.rename(columns={"machine_id": "machineID", "failed_component": "failure"}), C.STREAM_DIR / f"{tag}_outcomes.csv")
        manifest.append({"batch": k, "tag": tag, "start": str(tel["datetime"].min()), "end": str(tel["datetime"].max()),
                         "rows": len(tel), "errors": len(err), "maint": len(mnt), "failures": len(fail),
                         "synthetic_drift_injected": drifted})
        k += 1
        start = end
    make_variants(t, manifest)
    with open(C.STREAM_DIR / "manifest.json", "w") as f:
        json.dump({"drift_sensor": C.DRIFT_SENSOR, "drift_offset": C.DRIFT_OFFSET if drift else 0,
                   "drift_start_batch": C.DRIFT_START_BATCH if drift else None, "batches": manifest}, f, indent=2)
    print(f"wrote {len(manifest)} batches to {C.STREAM_DIR}")
    print(pd.DataFrame(manifest).to_string(index=False))


def load_manifest() -> dict:
    with open(C.STREAM_DIR / "manifest.json") as f:
        return json.load(f)


def batch_files(tag: str) -> dict:
    """Read one batch's four CSVs as raw DataFrames."""
    return {k: pd.read_csv(C.STREAM_DIR / f"{tag}_{k}.csv") for k in ("telemetry", "errors", "maint", "outcomes")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-drift", action="store_true")
    main(drift=not ap.parse_args().no_drift)
