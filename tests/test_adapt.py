"""Phase 10 tests (unit + one end-to-end). Run: pytest tests/test_adapt.py -v"""
import duckdb
import numpy as np
import pandas as pd
import pytest

from src import abi_config as C
from src import adapt
from src import warehouse as W
from src.config import CLEAN, MODELS, REPORTS


@pytest.fixture()
def con():
    c = duckdb.connect(":memory:")
    W.create_schema(c)
    W._load_dim_date(c, "2015-01-01", "2015-03-01")
    c.execute("INSERT INTO dim_machine VALUES (1,'model1',5),(2,'model2',9)")
    c.execute("INSERT INTO dim_component VALUES ('comp1'),('comp2')")
    c.execute("INSERT INTO dim_error_type VALUES ('error1'),('error2')")
    yield c
    c.close()


def good_tel():
    return pd.DataFrame({"datetime": ["2015-01-01 06:00:00", "2015-01-01 07:00:00"], "machineID": [1, 1],
                         "volt": [170.0, 171.0], "rotate": [440.0, 441.0], "pressure": [100.0, 101.0], "vibration": [40.0, 41.0]})


# ---------------- validation ----------------
def test_valid_file_passes(con):
    df, e, w = adapt.validate_upload("telemetry", good_tel(), con)
    assert e == [] and len(df) == 2


@pytest.mark.parametrize("mutate,fragment", [
    (lambda d: d.drop(columns=["volt"]), "missing required"),
    (lambda d: d.assign(machineID=[1, 99]), "unknown machineID"),
    (lambda d: d.assign(datetime=["2015-01-01 06:00:00", "not a date"]), "unreadable datetime"),
    (lambda d: d.assign(volt=[170.0, np.nan]), "missing / non-numeric"),
    (lambda d: d.assign(volt=["abc", "171"]), "missing / non-numeric"),
    (lambda d: d.assign(datetime=["2015-01-01 06:30:00", "2015-01-01 07:00:00"]), "on the hour"),
    (lambda d: pd.concat([d, d.iloc[[0]]]), "duplicate"),
])
def test_bad_telemetry_is_rejected_with_clear_message(con, mutate, fragment):
    df, e, _ = adapt.validate_upload("telemetry", mutate(good_tel()), con)
    assert df is None and any(fragment in m for m in e)


def test_rows_already_in_warehouse_are_rejected(con):
    W.insert_facts(con, "telemetry", adapt.validate_upload("telemetry", good_tel(), con)[0])
    df, e, _ = adapt.validate_upload("telemetry", good_tel(), con)
    assert df is None and any("already exist" in m for m in e)


def test_unknown_error_type_rejected_and_event_dupes_skipped(con):
    bad = pd.DataFrame({"datetime": ["2015-01-01 06:00:00"], "machineID": [1], "errorID": ["error9"]})
    assert adapt.validate_upload("errors", bad, con)[0] is None
    ok = pd.DataFrame({"datetime": ["2015-01-01 06:00:00"] * 2, "machineID": [1, 1], "errorID": [" ERROR1", "error1"]})
    df, e, w = adapt.validate_upload("errors", ok, con)
    assert e == [] and len(df) == 1 and any("duplicate" in m for m in w)


# ---------------- decision rules ----------------
def test_risk_bands_follow_thresholds():
    b = adapt.risk_band([0.01, 0.2, 0.9], high=0.5, medium=0.1)
    assert list(b) == ["low", "medium", "high"]


def test_metrics_at_known_confusion_matrix():
    m = adapt.metrics_at([1, 1, 0, 0, 0], [0.9, 0.2, 0.8, 0.1, 0.1], 0.5)
    assert (m["TP"], m["FP"], m["FN"], m["TN"]) == (1, 1, 1, 2)
    assert m["precision"] == 0.5 and m["recall"] == 0.5


# ---------------- drift ----------------
def test_psi_is_small_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    base = adapt.make_baseline(pd.DataFrame({"x": rng.normal(0, 1, 20000)}), ["x"])
    same = adapt.psi_value(base["features"]["x"], rng.normal(0, 1, 5000))
    shifted = adapt.psi_value(base["features"]["x"], rng.normal(1.5, 1, 5000))
    assert same < C.PSI_WATCH and shifted > C.PSI_ALERT


def test_psi_handles_constant_feature():
    base = adapt.make_baseline(pd.DataFrame({"x": np.zeros(100)}), ["x"])
    assert adapt.psi_value(base["features"]["x"], np.zeros(10)) == 0.0


# ---------------- delayed labels ----------------
def _pred(con, times, version="v1"):
    df = pd.DataFrame({"batch_id": "b", "machine_id": 1, "prediction_time": pd.to_datetime(times), "horizon_hours": 24,
                       "score": 0.5, "risk_band": "high", "threshold_high": 0.4, "threshold_medium": 0.1,
                       "model_version": version, "scored_at": pd.Timestamp.now(), "actual_outcome": pd.array([pd.NA] * len(times), dtype="Int64"),
                       "outcome_status": "pending", "outcome_recorded_at": pd.NaT})
    con.register("p", df)
    con.execute("INSERT INTO predictions SELECT * FROM p")
    con.unregister("p")


def test_outcomes_attach_only_after_the_horizon_has_elapsed(con):
    _pred(con, ["2015-01-01 00:00", "2015-01-02 00:00", "2015-01-03 00:00"])
    W.insert_facts(con, "failures", pd.DataFrame({"machine_id": [1], "datetime": pd.to_datetime(["2015-01-01 10:00"]), "failed_component": ["comp1"]}))
    adapt.set_state(con, "outcomes_known_until", "2015-01-03 12:00:00")
    assert adapt.mature_predictions(con) == 2            # 01-03 00:00 + 24h is beyond the known time (01-03 12:00)
    r = con.execute("SELECT prediction_time, actual_outcome, outcome_status FROM predictions ORDER BY 1").df()
    assert list(r["outcome_status"]) == ["matured", "matured", "pending"]
    assert r.loc[0, "actual_outcome"] == 1 and r.loc[1, "actual_outcome"] == 0


def test_failure_at_prediction_time_is_excluded(con):
    _pred(con, ["2015-01-01 10:00"])
    W.insert_facts(con, "failures", pd.DataFrame({"machine_id": [1], "datetime": pd.to_datetime(["2015-01-01 10:00"]), "failed_component": ["comp1"]}))
    adapt.set_state(con, "outcomes_known_until", "2015-01-05 00:00:00")
    adapt.mature_predictions(con)
    assert con.execute("SELECT outcome_status FROM predictions").fetchall()[0][0] == "excluded"


# ---------------- registry: promote / rollback ----------------
def _reg(con, v, status, passed=True):
    import json
    con.execute("INSERT INTO model_registry VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                [v, "rf", status, f"models/{v}/m.joblib", pd.Timestamp.now(), pd.Timestamp("2015-01-01"), pd.Timestamp("2015-02-01"),
                 0.5, 0.1, 0.9, 0.9, json.dumps({"passed": passed})])


def test_promote_requires_passed_criteria_and_rollback_restores(con):
    _reg(con, "v1", "champion"); _reg(con, "v2", "challenger", passed=False); _reg(con, "v3", "challenger", passed=True)
    assert adapt.promote(con, "v2")["ok"] is False                    # failed criteria -> refused
    assert adapt.get_champion(con)["version"] == "v1"
    assert adapt.promote(con, "v3")["ok"] is True
    assert adapt.get_champion(con)["version"] == "v3"
    assert con.execute("SELECT status FROM model_registry WHERE version='v1'").fetchall()[0][0] == "retired"
    assert adapt.rollback(con)["restored"] == "v1"
    assert adapt.get_champion(con)["version"] == "v1"
    assert adapt.rollback(con)["ok"] is False                         # nothing further to roll back to


# ---------------- end-to-end on the real data (skipped if Phases 3-8 outputs are missing) ----------------
needs = pytest.mark.skipif(not ((CLEAN / "telemetry_clean.parquet").exists() and (MODELS / "v1" / "selected_model.json").exists()
                                and (REPORTS / "predictions_v1.parquet").exists()),
                           reason="needs cleaned data and the v1 model")


@needs
def test_end_to_end_batch_scoring_and_feedback(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "DB_PATH", tmp_path / "test.duckdb")
    W.build_warehouse(verbose=False)
    con = W.connect()
    adapt.init_champion(con)
    end = C.LIVE_CUTOFF + pd.Timedelta(hours=47)
    def raw(name, col_map):
        d = pd.read_parquet(CLEAN / f"{name}_clean.parquet")
        d = d[(d["datetime"] >= C.LIVE_CUTOFF) & (d["datetime"] <= end)].rename(columns=col_map)
        d["datetime"] = d["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
        return d
    tel = raw("telemetry", {"machine_id": "machineID"})
    n_before = con.execute("SELECT COUNT(*) FROM fact_telemetry").fetchall()[0][0]
    res = adapt.process_batch(con, {"telemetry": tel, "errors": raw("errors", {"machine_id": "machineID", "error_id": "errorID"}),
                                    "maint": raw("maint", {"machine_id": "machineID", "component": "comp"})})
    assert res["ok"], res["errors"]
    assert res["n_scored"] == len(tel)                                   # one prediction per machine-hour
    assert con.execute("SELECT COUNT(*) FROM fact_telemetry").fetchall()[0][0] == n_before + len(tel)
    assert con.execute("SELECT COUNT(*) FROM predictions").fetchall()[0][0] == len(tel)
    assert adapt.process_batch(con, {"telemetry": tel})["ok"] is False   # re-upload rejected
    assert con.execute("SELECT COUNT(*) FROM fact_telemetry").fetchall()[0][0] == n_before + len(tel)   # nothing double-counted
    out = raw("failures", {"machine_id": "machineID", "failed_component": "failure"})
    fb = adapt.record_outcomes(con, out, end)
    assert fb["ok"] and fb["matured"] == len(tel) - 2400                 # last 24h of every machine still pending
    assert con.execute("SELECT COUNT(*) FROM predictions WHERE outcome_status='pending'").fetchall()[0][0] == 2400
    con.close()


@needs
def test_batch_with_bad_schema_changes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "DB_PATH", tmp_path / "test2.duckdb")
    W.build_warehouse(verbose=False)
    con = W.connect(); adapt.init_champion(con)
    bad = pd.DataFrame({"datetime": ["2015-09-01 00:00:00"], "machineID": [1], "volt": [1.0]})
    res = adapt.process_batch(con, {"telemetry": bad})
    assert not res["ok"] and con.execute("SELECT COUNT(*) FROM predictions").fetchall()[0][0] == 0
    con.close()
