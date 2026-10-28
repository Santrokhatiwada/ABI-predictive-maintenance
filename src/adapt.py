"""Phase 10: the Adaptive BI mechanism.

Capabilities (each one is a function you can point at in the demo):
  A. validate_upload / process_batch     new data in -> validate -> store -> features -> score -> log
  B. risk_band / metrics_at              documented, tunable decision rules
  C. record_outcomes / mature_predictions  feedback: actual outcomes attached once the 24h window has elapsed
  D. compute_drift / performance_check   data-drift (needs no labels) vs performance (needs labels)
  E. train_challenger / promote / rollback  gated retraining with versioning and rollback
Nothing here "learns" on its own: the model changes ONLY when train_challenger() is called AND the
predefined criteria are met.
"""
from __future__ import annotations

import json
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import Pipeline

from src import abi_config as C
from src.config import FEATURES, MODELS, PROJECT, REPORTS, SEED
from src.features import add_labels, build_features
from src.warehouse import insert_facts

RAW_COLS = {
    "telemetry": ["datetime", "machineID", "volt", "rotate", "pressure", "vibration"],
    "errors": ["datetime", "machineID", "errorID"],
    "maint": ["datetime", "machineID", "comp"],
    "outcomes": ["datetime", "machineID", "failure"],
}
RENAME = {"machineID": "machine_id", "errorID": "error_id", "comp": "component", "failure": "failed_component"}
SENSORS = ["volt", "rotate", "pressure", "vibration"]
_PIPE_CACHE: dict = {}


# =============================================================================
# small helpers
# =============================================================================
def feature_meta() -> dict:
    with open(FEATURES / "feature_meta.json") as f:
        return json.load(f)


def get_state(con, key, default=None):
    r = con.execute("SELECT value FROM system_state WHERE key = ?", [key]).fetchall()
    return r[0][0] if r else default


def set_state(con, key, value):
    con.execute("DELETE FROM system_state WHERE key = ?", [key])
    con.execute("INSERT INTO system_state VALUES (?, ?)", [key, str(value)])


def log_event(con, version, action, detail=""):
    con.execute("INSERT INTO model_events VALUES (?, ?, ?, ?)", [pd.Timestamp.now(), version, action, detail])


def load_pipeline(rel_path: str) -> Pipeline:
    if rel_path not in _PIPE_CACHE:
        _PIPE_CACHE[rel_path] = joblib.load(PROJECT / rel_path)
    return _PIPE_CACHE[rel_path]


def get_champion(con) -> dict | None:
    df = con.execute("SELECT * FROM model_registry WHERE status = 'champion'").df()
    return None if df.empty else df.iloc[0].to_dict()


def best_f1_threshold(y, p) -> tuple[float, float]:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    i = int(np.nanargmax(f1))
    return float(thr[i]), float(f1[i])


def medium_threshold(y, p, high) -> float:
    """'Medium' = scores above the MEDIUM_NEG_QUANTILE of validation scores on NON-failure hours
    (unusually high compared with normal operation) but still below the 'high' alert threshold."""
    neg = np.asarray(p)[np.asarray(y) == 0]
    return float(min(np.quantile(neg, C.MEDIUM_NEG_QUANTILE), high))


def risk_band(score, high, medium):
    s = np.asarray(score)
    return np.where(s >= high, "high", np.where(s >= medium, "medium", "low"))


def metrics_at(y, score, thr) -> dict:
    y = np.asarray(y).astype(int)
    flag = np.asarray(score) >= thr
    tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
    fn, tn = int((~flag & (y == 1)).sum()), int((~flag & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * prec * rec / (prec + rec) if tp > 0 else (0.0 if tp + fn > 0 else np.nan)
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "precision": prec, "recall": rec, "f1": f1,
            "alert_rate": float(flag.mean()) if len(flag) else np.nan}


# =============================================================================
# A. VALIDATION of uploaded files
# =============================================================================
def validate_upload(kind: str, raw: pd.DataFrame, con):
    """Returns (clean_df | None, errors, warnings). Any error means the file is rejected."""
    errors, warnings = [], []
    raw = raw.copy()
    raw.columns = [str(c).strip() for c in raw.columns]
    missing = [c for c in RAW_COLS[kind] if c not in raw.columns]
    if missing:
        return None, [f"{kind}: missing required column(s) {missing}. Found {list(raw.columns)}"], warnings
    df = raw[RAW_COLS[kind]].rename(columns=RENAME).copy()
    if len(df) == 0:
        return None, [f"{kind}: file has no rows"], warnings

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    n_bad_dt = int(df["datetime"].isna().sum())
    if n_bad_dt:
        errors.append(f"{kind}: {n_bad_dt} row(s) have an unreadable datetime")
    mid = pd.to_numeric(df["machine_id"], errors="coerce")
    bad_id = mid.isna() | ((mid % 1) != 0)
    if bad_id.any():
        errors.append(f"{kind}: {int(bad_id.sum())} row(s) have a missing or non-integer machineID")
    known = {r[0] for r in con.execute("SELECT machine_id FROM dim_machine").fetchall()}
    unknown = sorted(set(mid[~bad_id].astype(int)) - known)
    if unknown:
        errors.append(f"{kind}: unknown machineID(s) {unknown[:10]}{'...' if len(unknown) > 10 else ''}")

    if kind == "telemetry":
        for c in SENSORS:
            num = pd.to_numeric(df[c], errors="coerce")
            bad = num.isna() | np.isinf(num)
            if bad.any():
                errors.append(f"telemetry: column '{c}' has {int(bad.sum())} missing / non-numeric / infinite value(s)")
            df[c] = num
        ok_dt = df["datetime"].dropna()
        if ((ok_dt.dt.minute != 0) | (ok_dt.dt.second != 0)).any():
            errors.append("telemetry: timestamps must be exactly on the hour (e.g. 2015-09-01 06:00:00)")
    elif kind == "errors":
        df["error_id"] = df["error_id"].astype(str).str.strip().str.lower()
        valid = {r[0] for r in con.execute("SELECT error_type_id FROM dim_error_type").fetchall()}
        bad = sorted(set(df["error_id"]) - valid)
        if bad:
            errors.append(f"errors: unknown errorID value(s) {bad}. Valid: {sorted(valid)}")
    else:
        col = "component" if kind == "maint" else "failed_component"
        df[col] = df[col].astype(str).str.strip().str.lower()
        valid = {r[0] for r in con.execute("SELECT component_id FROM dim_component").fetchall()}
        bad = sorted(set(df[col]) - valid)
        if bad:
            errors.append(f"{kind}: unknown component value(s) {bad}. Valid: {sorted(valid)}")
    if errors:
        return None, errors, warnings

    df["machine_id"] = mid.astype("int64")
    key = {"telemetry": ["machine_id", "datetime"], "errors": ["machine_id", "datetime", "error_id"],
           "maint": ["machine_id", "datetime", "component"], "outcomes": ["machine_id", "datetime", "failed_component"]}[kind]
    ndup = int(df.duplicated(key).sum())
    table = {"telemetry": ("fact_telemetry", "reading_time", None), "errors": ("fact_error", "event_time", "error_type_id"),
             "maint": ("fact_maintenance", "event_time", "component_id"), "outcomes": ("fact_failure", "event_time", "component_id")}[kind]
    con.register("chk_df", df)
    if kind == "telemetry":
        if ndup:
            return None, [f"telemetry: {ndup} duplicate (machineID, datetime) row(s) inside the file"], warnings
        n_exist = con.execute("SELECT COUNT(*) FROM chk_df c JOIN fact_telemetry f ON f.machine_id=c.machine_id AND f.reading_time=c.datetime").fetchall()[0][0]
        con.unregister("chk_df")
        if n_exist:
            return None, [f"telemetry: {n_exist} row(s) already exist in the warehouse (this batch looks already uploaded)"], warnings
        g = df.groupby("machine_id")["datetime"].agg(["min", "max", "count"])
        gaps = ((g["max"] - g["min"]) / pd.Timedelta(hours=1)).round().astype(int) + 1 - g["count"]
        if (gaps > 0).any():
            warnings.append(f"telemetry: {int((gaps > 0).sum())} machine(s) have missing hours inside the batch; "
                            "rolling-window features for them are approximate")
        if g.shape[0] < len(known):
            warnings.append(f"telemetry: only {g.shape[0]} of {len(known)} machines are present")
        last = con.execute("SELECT machine_id, MAX(reading_time) FROM fact_telemetry GROUP BY 1").df().set_index("machine_id").iloc[:, 0]
        j = g.join(last.rename("last_loaded"), how="inner")
        non_contig = ((j["min"] - j["last_loaded"]) != pd.Timedelta(hours=1)).sum()
        if non_contig:
            warnings.append(f"telemetry: for {int(non_contig)} machine(s) the batch does not start exactly 1h after the "
                            "latest stored reading (gap or overlap in time)")
    else:
        if ndup:
            warnings.append(f"{kind}: {ndup} duplicate event row(s) inside the file were ignored")
            df = df.drop_duplicates(key)
            con.unregister("chk_df"); con.register("chk_df", df)
        tbl, tcol, ccol = table
        join_c = f"AND f.{ccol} = c.{'error_id' if kind == 'errors' else 'component' if kind == 'maint' else 'failed_component'}"
        exists = con.execute(f"SELECT c.machine_id, c.datetime FROM chk_df c JOIN {tbl} f ON f.machine_id=c.machine_id AND f.{tcol}=c.datetime {join_c}").df()
        con.unregister("chk_df")
        if len(exists):
            warnings.append(f"{kind}: {len(exists)} event(s) were already stored and were skipped")
            ex = set(zip(exists["machine_id"], exists["datetime"]))
            df = df[[(m, d) not in ex for m, d in zip(df["machine_id"], df["datetime"])]]
    return df.reset_index(drop=True), errors, warnings


# =============================================================================
# features for scoring (identical code path to training: src.features.build_features)
# =============================================================================
def live_features(con, start, end) -> pd.DataFrame:
    meta = feature_meta()
    h = pd.Timedelta(hours=C.HISTORY_HOURS)
    p = [start - h, end]
    tel = con.execute("""SELECT machine_id, reading_time AS datetime, volt, rotate, pressure, vibration
                         FROM fact_telemetry WHERE reading_time >= ? AND reading_time <= ?
                         ORDER BY machine_id, reading_time""", p).df()
    err = con.execute("""SELECT machine_id, event_time AS datetime, error_type_id AS error_id FROM fact_error
                        WHERE event_time >= ? AND event_time <= ?""", p).df()
    mnt = con.execute("""SELECT machine_id, event_time AS datetime, component_id AS component FROM fact_maintenance
                        WHERE event_time <= ?""", [end]).df()
    mac = con.execute("SELECT machine_id, model, age FROM dim_machine").df()
    f = build_features(tel, err, mnt, mac, meta["error_types"], meta["components"])
    return f[(f["datetime"] >= start) & (f["datetime"] <= end)].reset_index(drop=True)


# =============================================================================
# D. DRIFT: Population Stability Index against the champion's training data
# =============================================================================
def monitored_cols(feats) -> list[str]:
    return [c for c in feats if c.endswith("_mean_24h") or c.endswith("_count_24h")]


def make_baseline(df: pd.DataFrame, cols, bins: int = 10) -> dict:
    out = {}
    for c in cols:
        x = df[c].dropna().to_numpy(float)
        q = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
        interior = q[1:-1]
        counts = np.bincount(np.searchsorted(interior, x, side="right"), minlength=len(interior) + 1)
        out[c] = {"interior_edges": interior.tolist(), "expected": (counts / counts.sum()).tolist()}
    return {"features": out, "n_rows": int(len(df))}


def psi_value(entry: dict, x) -> float:
    interior = np.asarray(entry["interior_edges"], float)
    exp = np.clip(np.asarray(entry["expected"], float), 1e-4, None)
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0 or len(interior) == 0:
        return 0.0
    cnt = np.bincount(np.searchsorted(interior, x, side="right"), minlength=len(exp))
    act = np.clip(cnt / cnt.sum(), 1e-4, None)
    return float(np.sum((act - exp) * np.log(act / exp)))


def drift_status(psi: float) -> str:
    return "alert" if psi >= C.PSI_ALERT else "watch" if psi >= C.PSI_WATCH else "stable"


def compute_drift(feats: pd.DataFrame, baseline: dict) -> pd.DataFrame:
    rows = []
    for c, entry in baseline["features"].items():
        v = psi_value(entry, feats[c])
        rows.append({"feature": c, "psi": v, "status": drift_status(v)})
    return pd.DataFrame(rows)


def baseline_path(version: str):
    return MODELS / version / "baseline_stats.json"


# =============================================================================
# registry: register v1 as the first champion
# =============================================================================
def init_champion(con) -> dict:
    """Register models/v1 as champion. Thresholds come from VALIDATION data only (Phase 8 outputs)."""
    sel = json.load(open(MODELS / "v1" / "selected_model.json"))
    meta_m = json.load(open(MODELS / "v1" / "metadata.json"))
    preds = pd.read_parquet(REPORTS / "predictions_v1.parquet")
    val = preds[preds["split"] == "val"]
    p, y = val[f"proba_{sel['model']}"].to_numpy(), val["y"].to_numpy()
    high = float(sel["threshold"])
    med = medium_threshold(y, p, high)
    m = metrics_at(y, p, high)
    pr = float(average_precision_score(y, p))
    feats = feature_meta()["feature_cols"]
    ft = pd.read_parquet(FEATURES / "feature_table.parquet", columns=["datetime", "split"] + monitored_cols(feats))
    base = make_baseline(ft[ft["split"] == "train"], monitored_cols(feats))
    with open(baseline_path("v1"), "w") as f:
        json.dump(base, f)
    con.execute("DELETE FROM model_registry")
    con.execute("INSERT INTO model_registry VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                ["v1", sel["model"], "champion", f"models/v1/{sel['model']}.joblib", pd.Timestamp(meta_m["trained_at"]),
                 pd.Timestamp(meta_m["train_period"][0]), pd.Timestamp(meta_m["train_period"][1]),
                 high, med, pr, m["f1"], json.dumps({"source": "Phase 8 validation", "rule": sel["selection_rule"]})])
    set_state(con, "previous_champion", "")
    log_event(con, "v1", "registered_champion", f"thr_high={high:.4f} thr_medium={med:.4f}")
    return {"version": "v1", "threshold_high": high, "threshold_medium": med, "val_pr_auc": pr, "val_f1": m["f1"]}


# =============================================================================
# A+B+D. ingest a batch: validate -> store -> features -> score -> drift -> log (one transaction)
# =============================================================================
def process_batch(con, raw: dict, source: str = "upload") -> dict:
    res = {"ok": False, "errors": [], "warnings": []}
    if raw.get("telemetry") is None:
        res["errors"].append("A telemetry file is required.")
        return res
    champ = get_champion(con)
    if champ is None:
        res["errors"].append("No champion model is registered. Run: python -m src.reset_demo")
        return res
    clean = {}
    for kind in ("telemetry", "errors", "maint"):
        if raw.get(kind) is not None:
            df, e, w = validate_upload(kind, raw[kind], con)
            res["errors"] += e; res["warnings"] += w
            clean[kind] = df
    if res["errors"]:
        return res
    tel = clean["telemetry"]
    start, end = tel["datetime"].min(), tel["datetime"].max()
    for kind in ("errors", "maint"):
        if kind in clean and len(clean[kind]) and clean[kind]["datetime"].max() > end:
            res["errors"].append(f"{kind}: contains events after the last telemetry hour ({end}); upload them with the next batch")
    if res["errors"]:
        return res

    batch_id = f"{start:%Y%m%d%H}_{end:%Y%m%d%H}"
    if con.execute("SELECT COUNT(*) FROM batch_log WHERE batch_id = ?", [batch_id]).fetchall()[0][0]:
        batch_id += f"_{datetime.now():%H%M%S}"
    t0 = time.time()
    con.execute("BEGIN TRANSACTION")
    try:
        n_t = insert_facts(con, "telemetry", tel)
        n_e = insert_facts(con, "errors", clean.get("errors", pd.DataFrame(columns=["machine_id", "datetime", "error_id"])))
        n_m = insert_facts(con, "maint", clean.get("maint", pd.DataFrame(columns=["machine_id", "datetime", "component"])))
        feats = live_features(con, start, end)
        if feats.empty:
            raise ValueError("no scorable rows (not enough history before the batch)")
        pipe = load_pipeline(champ["model_path"])
        cols = feature_meta()["feature_cols"]
        score = pipe.predict_proba(feats[cols])[:, 1]
        band = risk_band(score, champ["threshold_high"], champ["threshold_medium"])
        now = pd.Timestamp.now()
        pred = pd.DataFrame({
            "batch_id": batch_id, "machine_id": feats["machine_id"].to_numpy(), "prediction_time": feats["datetime"].to_numpy(),
            "horizon_hours": C.HORIZON_HOURS, "score": score, "risk_band": band,
            "threshold_high": champ["threshold_high"], "threshold_medium": champ["threshold_medium"],
            "model_version": champ["version"], "scored_at": now, "actual_outcome": np.nan,
            "outcome_status": "pending", "outcome_recorded_at": pd.NaT})
        pred["actual_outcome"] = pred["actual_outcome"].astype("Int64")
        con.register("pred_df", pred)
        con.execute("INSERT INTO predictions SELECT * FROM pred_df")
        con.unregister("pred_df")

        base = json.load(open(baseline_path(champ["version"])))
        drift = compute_drift(feats, base)
        con.register("drift_df", drift.assign(batch_id=batch_id, model_version=champ["version"], checked_at=now)
                     [["batch_id", "model_version", "feature", "psi", "status", "checked_at"]])
        con.execute("INSERT INTO drift_log SELECT * FROM drift_df")
        con.unregister("drift_df")
        d_status = "alert" if (drift["status"] == "alert").any() else "watch" if (drift["status"] == "watch").any() else "stable"
        secs = time.time() - t0
        con.execute("INSERT INTO batch_log VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    [batch_id, now, source, start, end, n_t, n_e, n_m, champ["version"], len(pred), d_status,
                     float(drift["psi"].max()), f"processed in {secs:.1f}s"])
        con.execute("COMMIT")
    except Exception as ex:
        con.execute("ROLLBACK")
        res["errors"].append(f"Processing failed and was rolled back (nothing was stored): {ex}")
        return res
    latest = pred[pred["prediction_time"] == pred["prediction_time"].max()]
    res.update(ok=True, batch_id=batch_id, start=start, end=end, n_telemetry=n_t, n_errors=n_e, n_maint=n_m,
               n_scored=len(pred), model_version=champ["version"], drift_status=d_status,
               max_psi=float(drift["psi"].max()), drift=drift, seconds=round(secs, 1),
               high_risk_now=int((latest["risk_band"] == "high").sum()),
               medium_risk_now=int((latest["risk_band"] == "medium").sum()))
    return res


# =============================================================================
# C. FEEDBACK: record actual failures, then attach outcomes to matured predictions
# =============================================================================
def record_outcomes(con, raw_outcomes: pd.DataFrame, known_until) -> dict:
    """`known_until`: failures are declared COMPLETE up to this time (absence of a row = no failure)."""
    res = {"ok": False, "errors": [], "warnings": []}
    df, e, w = validate_upload("outcomes", raw_outcomes, con)
    res["errors"], res["warnings"] = e, w
    if df is None:
        return res
    known_until = pd.Timestamp(known_until)
    if len(df) and df["datetime"].max() > known_until:
        res["errors"].append("outcomes: contains failures later than the 'known until' time you declared")
        return res
    con.execute("BEGIN TRANSACTION")
    try:
        n = insert_facts(con, "failures", df)
        prev = pd.Timestamp(get_state(con, "outcomes_known_until"))
        set_state(con, "outcomes_known_until", max(prev, known_until))
        con.execute("COMMIT")
    except Exception as ex:
        con.execute("ROLLBACK")
        res["errors"].append(f"Failed and rolled back: {ex}")
        return res
    res.update(ok=True, failures_added=n, matured=mature_predictions(con))
    return res


def mature_predictions(con) -> int:
    """Attach outcomes ONLY to predictions whose whole horizon lies before outcomes_known_until."""
    known = pd.Timestamp(get_state(con, "outcomes_known_until"))
    pend = con.execute("""SELECT machine_id, prediction_time, model_version FROM predictions
                          WHERE outcome_status = 'pending'
                            AND prediction_time + horizon_hours * INTERVAL 1 HOUR <= ?""", [known]).df()
    if pend.empty:
        return 0
    fails = con.execute("SELECT machine_id, event_time AS datetime FROM fact_failure WHERE event_time <= ?", [known]).df()
    lab = add_labels(pend.rename(columns={"prediction_time": "datetime"}), fails, C.HORIZON_HOURS, last_ts=known)
    lab["status"] = np.where(lab["failure_at_t"] == 1, "excluded", "matured")
    lab["y_out"] = np.where(lab["failure_at_t"] == 1, np.nan, lab["y"]).astype(float)
    u = lab.rename(columns={"datetime": "prediction_time"})[["machine_id", "prediction_time", "model_version", "y_out", "status"]]
    con.register("u_df", u)
    con.execute("""UPDATE predictions SET actual_outcome = CAST(u.y_out AS INTEGER), outcome_status = u.status,
                          outcome_recorded_at = now()
                   FROM u_df u WHERE predictions.machine_id = u.machine_id
                     AND predictions.prediction_time = u.prediction_time
                     AND predictions.model_version = u.model_version""")
    con.unregister("u_df")
    return len(u)


# =============================================================================
# D. performance monitoring (needs matured labels)
# =============================================================================
def matured_predictions(con, version: str | None = None) -> pd.DataFrame:
    q = "SELECT * FROM predictions WHERE outcome_status = 'matured'"
    if version:
        return con.execute(q + " AND model_version = ?", [version]).df()
    return con.execute(q).df()


def performance_by_batch(con) -> pd.DataFrame:
    d = matured_predictions(con)
    rows = []
    for (b, v), g in d.groupby(["batch_id", "model_version"]):
        m = metrics_at(g["actual_outcome"], g["score"], g["threshold_high"].iloc[0])
        m.update(batch_id=b, model_version=v, rows=len(g), positives=int(g["actual_outcome"].sum()),
                 week_start=g["prediction_time"].min())
        rows.append(m)
    return pd.DataFrame(rows).sort_values("week_start").reset_index(drop=True) if rows else pd.DataFrame()


def performance_check(con) -> dict:
    champ = get_champion(con)
    d = matured_predictions(con, champ["version"])
    out = {"status": "insufficient_labels", "version": champ["version"], "baseline_f1": champ["val_f1"]}
    if d.empty:
        return out
    d = d[d["prediction_time"] >= d["prediction_time"].max() - pd.Timedelta(days=C.PERF_WINDOW_DAYS)]
    pos = int(d["actual_outcome"].sum())
    out.update(positives=pos, rows=len(d), window_start=d["prediction_time"].min())
    if pos < C.PERF_MIN_POSITIVES:
        return out
    m = metrics_at(d["actual_outcome"], d["score"], champ["threshold_high"])
    out.update(m)
    out["status"] = "degraded" if m["f1"] < champ["val_f1"] - C.PERF_F1_DROP else "ok"
    return out


# =============================================================================
# E. controlled retraining: champion / challenger
# =============================================================================
def _training_table(con, known_end: pd.Timestamp) -> pd.DataFrame:
    meta = feature_meta()
    since = known_end - pd.Timedelta(days=C.TRAIN_WINDOW_DAYS + C.VAL_DAYS + 10)
    tel = con.execute("""SELECT machine_id, reading_time AS datetime, volt, rotate, pressure, vibration FROM fact_telemetry
                         WHERE reading_time >= ? ORDER BY machine_id, reading_time""", [since]).df()
    err = con.execute("SELECT machine_id, event_time AS datetime, error_type_id AS error_id FROM fact_error WHERE event_time >= ?", [since]).df()
    mnt = con.execute("SELECT machine_id, event_time AS datetime, component_id AS component FROM fact_maintenance").df()
    fail = con.execute("SELECT machine_id, event_time AS datetime, component_id AS failed_component FROM fact_failure WHERE event_time <= ?", [known_end]).df()
    mac = con.execute("SELECT machine_id, model, age FROM dim_machine").df()
    f = build_features(tel, err, mnt, mac, meta["error_types"], meta["components"])
    df = add_labels(f, fail, C.HORIZON_HOURS, last_ts=known_end)
    return df[df["fully_observed"] & (df["failure_at_t"] == 0)].reset_index(drop=True)


def next_version(con) -> str:
    vs = [int(v[1:]) for (v,) in con.execute("SELECT version FROM model_registry").fetchall()]
    return f"v{max(vs) + 1}"


def train_challenger(con, n_estimators: int = 200) -> dict:
    """Train on recent labelled data, compare with the champion on the newest labelled block."""
    from src.train_models import CAT_COLS, make_preprocessor, model_zoo   # same recipe as v1
    champ = get_champion(con)
    known = pd.Timestamp(get_state(con, "outcomes_known_until"))
    df = _training_table(con, known)
    res = {"ok": False, "champion": champ["version"]}
    if df.empty:
        res["reason"] = "no labelled data available"
        return res
    label_end = df["datetime"].max()
    val_start = label_end - pd.Timedelta(days=C.VAL_DAYS)
    gap = pd.Timedelta(hours=C.HORIZON_HOURS)
    train = df[(df["datetime"] >= val_start - pd.Timedelta(days=C.TRAIN_WINDOW_DAYS)) & (df["datetime"] < val_start - gap)]
    val = df[df["datetime"] >= val_start]
    new_events = int(con.execute("SELECT COUNT(DISTINCT (machine_id, event_time)) FROM fact_failure WHERE event_time > ? AND event_time <= ?",
                                 [champ["train_end"], known]).fetchall()[0][0])
    res.update(train_rows=len(train), val_rows=len(val), train_positives=int(train["y"].sum()),
               val_positives=int(val["y"].sum()), new_events_since_champion=new_events,
               train_period=(str(train["datetime"].min()), str(train["datetime"].max())) if len(train) else None,
               val_period=(str(val["datetime"].min()), str(val["datetime"].max())))
    if new_events < C.MIN_NEW_EVENTS or train["y"].sum() < 50 or val["y"].sum() < 50:
        res["reason"] = (f"not enough new labelled data (new events {new_events} < {C.MIN_NEW_EVENTS}, "
                         f"train positives {int(train['y'].sum())}, val positives {int(val['y'].sum())})")
        return res

    feats = feature_meta()["feature_cols"]
    est, _ = model_zoo()["random_forest"]
    est.set_params(n_estimators=n_estimators)
    pipe = Pipeline([("pre", make_preprocessor([c for c in feats if c not in CAT_COLS], CAT_COLS, False)), ("model", est)])
    t0 = time.time()
    pipe.fit(train[feats], train["y"])
    res["fit_seconds"] = round(time.time() - t0, 1)

    y = val["y"].to_numpy()
    p_ch = load_pipeline(champ["model_path"]).predict_proba(val[feats])[:, 1]
    p_new = pipe.predict_proba(val[feats])[:, 1]
    pr_ch, pr_new = average_precision_score(y, p_ch), average_precision_score(y, p_new)
    thr_ch, f1_ch = best_f1_threshold(y, p_ch)
    thr_new, f1_new = best_f1_threshold(y, p_new)
    med_new = medium_threshold(y, p_new, thr_new)
    checks = {"enough_new_events": new_events >= C.MIN_NEW_EVENTS,
              "pr_auc_beats_champion_by_margin": bool(pr_new >= pr_ch + C.PROMOTE_MARGIN),
              "f1_not_worse": bool(f1_new >= f1_ch)}
    passed = all(checks.values())
    version = next_version(con)
    out = MODELS / version
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out / "model.joblib")
    with open(baseline_path(version), "w") as f:
        json.dump(make_baseline(val, monitored_cols(feats)), f)   # baseline = the NEW normal (newest labelled block)
    evaluation = {"champion_pr_auc": pr_ch, "challenger_pr_auc": pr_new, "champion_f1": f1_ch, "challenger_f1": f1_new,
                  "criteria": checks, "passed": passed, "margin": C.PROMOTE_MARGIN, "n_estimators": n_estimators}
    with open(out / "metadata.json", "w") as f:
        json.dump({"version": version, "trained_at": datetime.now().isoformat(timespec="seconds"), "features": feats,
                   "train_period": res["train_period"], "val_period": res["val_period"], "seed": SEED,
                   "evaluation": evaluation}, f, indent=2)
    con.execute("INSERT INTO model_registry VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                [version, "random_forest", "challenger", f"models/{version}/model.joblib", pd.Timestamp.now(),
                 train["datetime"].min(), train["datetime"].max(), thr_new, med_new, pr_new, f1_new, json.dumps(evaluation)])
    log_event(con, version, "challenger_trained", f"PR-AUC {pr_new:.4f} vs champion {pr_ch:.4f}; passed={passed}")
    res.update(ok=True, version=version, passed=passed, evaluation=evaluation)
    return res


def promote(con, version: str, force: bool = False) -> dict:
    row = con.execute("SELECT * FROM model_registry WHERE version = ?", [version]).df()
    if row.empty or row.iloc[0]["status"] != "challenger":
        return {"ok": False, "reason": f"{version} is not a challenger"}
    ev = json.loads(row.iloc[0]["evaluation_json"])
    if not ev.get("passed") and not force:
        return {"ok": False, "reason": "challenger did not meet the predefined promotion criteria"}
    old = get_champion(con)["version"]
    con.execute("UPDATE model_registry SET status = 'retired' WHERE version = ?", [old])
    con.execute("UPDATE model_registry SET status = 'champion' WHERE version = ?", [version])
    set_state(con, "previous_champion", old)
    log_event(con, version, "promoted" + ("_FORCED" if force and not ev.get("passed") else ""), f"replaced {old}")
    return {"ok": True, "promoted": version, "replaced": old}


def reject(con, version: str) -> None:
    con.execute("UPDATE model_registry SET status = 'rejected' WHERE version = ? AND status = 'challenger'", [version])
    log_event(con, version, "rejected", "challenger not promoted")


def rollback(con) -> dict:
    prev = get_state(con, "previous_champion", "")
    if not prev:
        return {"ok": False, "reason": "no previous champion to roll back to"}
    cur = get_champion(con)["version"]
    con.execute("UPDATE model_registry SET status = 'rolled_back' WHERE version = ?", [cur])
    con.execute("UPDATE model_registry SET status = 'champion' WHERE version = ?", [prev])
    set_state(con, "previous_champion", "")
    log_event(con, prev, "rollback", f"{cur} withdrawn, {prev} restored")
    return {"ok": True, "restored": prev, "withdrawn": cur}


# =============================================================================
# dashboard helpers
# =============================================================================
def latest_machine_risk(con) -> pd.DataFrame:
    """One row per machine: latest score, 24h max, band and priority rank (champion's decision rule)."""
    return con.execute("""
        WITH last AS (SELECT MAX(prediction_time) AS t FROM predictions),
        cur AS (SELECT p.* FROM predictions p, last WHERE p.prediction_time = last.t),
        mx AS (SELECT p.machine_id, MAX(p.score) AS max_score_24h FROM predictions p, last
               WHERE p.prediction_time > last.t - INTERVAL 24 HOUR GROUP BY 1)
        SELECT c.machine_id, m.model, m.age, c.prediction_time, c.score, mx.max_score_24h, c.risk_band,
               c.model_version, c.threshold_high, c.threshold_medium
        FROM cur c JOIN mx USING (machine_id) JOIN dim_machine m USING (machine_id)
        ORDER BY c.score DESC, mx.max_score_24h DESC""").df()
