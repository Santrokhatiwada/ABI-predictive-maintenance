"""Phase 9 tests. Run: pytest tests/test_warehouse.py -v"""
import duckdb
import pandas as pd
import pytest

from src import warehouse as W


@pytest.fixture()
def con():
    c = duckdb.connect(":memory:")
    W.create_schema(c)
    W._load_dim_date(c, "2015-01-01", "2015-03-01")
    c.execute("INSERT INTO dim_machine VALUES (1,'model1',5),(2,'model2',9)")
    c.execute("INSERT INTO dim_component VALUES ('comp1'),('comp2')")
    c.execute("INSERT INTO dim_error_type VALUES ('error1')")
    yield c
    c.close()


def tel(machine, times):
    return pd.DataFrame({"machine_id": machine, "datetime": pd.to_datetime(times),
                         "volt": 1.0, "rotate": 2.0, "pressure": 3.0, "vibration": 4.0})


def test_all_expected_tables_exist(con):
    names = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert set(W.TABLES) <= names


def test_insert_and_zero_orphans(con):
    W.insert_facts(con, "telemetry", tel(1, ["2015-01-01 06:00", "2015-01-01 07:00"]))
    W.insert_facts(con, "failures", pd.DataFrame({"machine_id": [1], "datetime": pd.to_datetime(["2015-01-01 07:00"]),
                                                  "failed_component": ["comp1"]}))
    rep = W.integrity_report(con)
    assert (rep[rep["meaning"].str.contains("orphan")]["value"] == 0).all()


def test_unknown_machine_is_rejected_by_foreign_key(con):
    with pytest.raises(duckdb.Error):
        W.insert_facts(con, "telemetry", tel(99, ["2015-01-01 06:00"]))


def test_duplicate_grain_is_rejected_by_primary_key(con):
    W.insert_facts(con, "telemetry", tel(1, ["2015-01-01 06:00"]))
    with pytest.raises(duckdb.Error):
        W.insert_facts(con, "telemetry", tel(1, ["2015-01-01 06:00"]))


def test_date_outside_dim_date_is_rejected(con):
    with pytest.raises(duckdb.Error):
        W.insert_facts(con, "telemetry", tel(1, ["2019-01-01 06:00"]))


def test_bi_queries_run(con):
    W.insert_facts(con, "failures", pd.DataFrame({"machine_id": [1, 2], "datetime": pd.to_datetime(["2015-01-05 06:00", "2015-02-05 06:00"]),
                                                  "failed_component": ["comp1", "comp2"]}))
    assert len(W.q_failures_by_component(con)) == 2
    assert W.q_failures_by_model(con)["failures"].sum() == 2
    assert len(W.q_failures_by_month(con)) == 2
