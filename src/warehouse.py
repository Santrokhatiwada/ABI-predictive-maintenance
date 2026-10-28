"""Phase 9: DuckDB star schema + BI queries.
Run:  python -m src.warehouse            (rebuilds the warehouse from data/clean/*.parquet)"""
from __future__ import annotations

import duckdb
import pandas as pd

from src import abi_config as C
from src.config import CLEAN

DDL = """
-- ---------- dimensions ----------
CREATE TABLE dim_machine (
    machine_id INTEGER PRIMARY KEY,
    model      VARCHAR NOT NULL,
    age        INTEGER
);
CREATE TABLE dim_date (
    date_key INTEGER PRIMARY KEY,          -- yyyymmdd
    full_date DATE NOT NULL,
    year INTEGER, quarter INTEGER, month INTEGER, month_name VARCHAR,
    week_of_year INTEGER, day_of_month INTEGER, day_of_week INTEGER,
    day_name VARCHAR, is_weekend BOOLEAN
);
CREATE TABLE dim_component  (component_id  VARCHAR PRIMARY KEY);
CREATE TABLE dim_error_type (error_type_id VARCHAR PRIMARY KEY);

-- ---------- facts (insert-only history) ----------
CREATE TABLE fact_telemetry (               -- grain: one machine, one hour
    machine_id   INTEGER   NOT NULL REFERENCES dim_machine(machine_id),
    reading_time TIMESTAMP NOT NULL,
    date_key     INTEGER   NOT NULL REFERENCES dim_date(date_key),
    volt DOUBLE, rotate DOUBLE, pressure DOUBLE, vibration DOUBLE,
    PRIMARY KEY (machine_id, reading_time)
);
CREATE TABLE fact_error (                   -- grain: one error event
    machine_id    INTEGER   NOT NULL REFERENCES dim_machine(machine_id),
    event_time    TIMESTAMP NOT NULL,
    date_key      INTEGER   NOT NULL REFERENCES dim_date(date_key),
    error_type_id VARCHAR   NOT NULL REFERENCES dim_error_type(error_type_id),
    PRIMARY KEY (machine_id, event_time, error_type_id)
);
CREATE TABLE fact_maintenance (             -- grain: one component replacement
    machine_id   INTEGER   NOT NULL REFERENCES dim_machine(machine_id),
    event_time   TIMESTAMP NOT NULL,
    date_key     INTEGER   NOT NULL REFERENCES dim_date(date_key),
    component_id VARCHAR   NOT NULL REFERENCES dim_component(component_id),
    PRIMARY KEY (machine_id, event_time, component_id)
);
CREATE TABLE fact_failure (                 -- grain: one component failure
    machine_id   INTEGER   NOT NULL REFERENCES dim_machine(machine_id),
    event_time   TIMESTAMP NOT NULL,
    date_key     INTEGER   NOT NULL REFERENCES dim_date(date_key),
    component_id VARCHAR   NOT NULL REFERENCES dim_component(component_id),
    PRIMARY KEY (machine_id, event_time, component_id)
);

-- ---------- operational tables written by the adaptive mechanism (Phase 10) ----------
CREATE TABLE model_registry (
    version VARCHAR PRIMARY KEY, algorithm VARCHAR, status VARCHAR, model_path VARCHAR,
    created_at TIMESTAMP, train_start TIMESTAMP, train_end TIMESTAMP,
    threshold_high DOUBLE, threshold_medium DOUBLE, val_pr_auc DOUBLE, val_f1 DOUBLE,
    evaluation_json VARCHAR
);
CREATE TABLE model_events (event_time TIMESTAMP, version VARCHAR, action VARCHAR, detail VARCHAR);
CREATE TABLE batch_log (
    batch_id VARCHAR PRIMARY KEY, ingested_at TIMESTAMP, source VARCHAR,
    data_start TIMESTAMP, data_end TIMESTAMP, n_telemetry INTEGER, n_errors INTEGER, n_maint INTEGER,
    model_version VARCHAR, n_scored INTEGER, drift_status VARCHAR, max_psi DOUBLE, message VARCHAR
);
CREATE TABLE predictions (
    batch_id VARCHAR, machine_id INTEGER, prediction_time TIMESTAMP, horizon_hours INTEGER,
    score DOUBLE, risk_band VARCHAR, threshold_high DOUBLE, threshold_medium DOUBLE,
    model_version VARCHAR, scored_at TIMESTAMP,
    actual_outcome INTEGER, outcome_status VARCHAR, outcome_recorded_at TIMESTAMP
);
CREATE TABLE drift_log (batch_id VARCHAR, model_version VARCHAR, feature VARCHAR, psi DOUBLE,
                        status VARCHAR, checked_at TIMESTAMP);
CREATE TABLE system_state (key VARCHAR PRIMARY KEY, value VARCHAR);
"""

TABLES = ["dim_machine", "dim_date", "dim_component", "dim_error_type", "fact_telemetry",
          "fact_error", "fact_maintenance", "fact_failure", "model_registry", "model_events",
          "batch_log", "predictions", "drift_log", "system_state"]


def connect(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Short-lived connections avoid file-lock clashes between the dashboard and scripts."""
    C.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(C.DB_PATH), read_only=read_only)


def db_exists() -> bool:
    return C.DB_PATH.exists()


def create_schema(con) -> None:
    for t in reversed(TABLES):
        con.execute(f"DROP TABLE IF EXISTS {t}")
    con.execute(DDL)


def date_key(ts: pd.Series) -> pd.Series:
    return ts.dt.strftime("%Y%m%d").astype("int64")


def _load_dim_date(con, start="2014-05-01", end="2016-02-01") -> None:
    d = pd.DataFrame({"full_date": pd.date_range(start, end, freq="D")})
    d["date_key"] = d["full_date"].dt.strftime("%Y%m%d").astype(int)
    d["year"], d["quarter"], d["month"] = d["full_date"].dt.year, d["full_date"].dt.quarter, d["full_date"].dt.month
    d["month_name"] = d["full_date"].dt.month_name()
    d["week_of_year"] = d["full_date"].dt.isocalendar().week.astype(int)
    d["day_of_month"], d["day_of_week"] = d["full_date"].dt.day, d["full_date"].dt.dayofweek
    d["day_name"] = d["full_date"].dt.day_name()
    d["is_weekend"] = d["day_of_week"] >= 5
    cols = ["date_key", "full_date", "year", "quarter", "month", "month_name", "week_of_year",
            "day_of_month", "day_of_week", "day_name", "is_weekend"]
    con.register("d_df", d[cols])
    con.execute("INSERT INTO dim_date SELECT * FROM d_df")
    con.unregister("d_df")


def insert_facts(con, kind: str, df: pd.DataFrame) -> int:
    """kind in telemetry/errors/maint/failures. df uses the CLEAN column names. Returns rows inserted."""
    if len(df) == 0:
        return 0
    df = df.copy()
    df["date_key"] = date_key(df["datetime"])
    con.register("ins_df", df)
    if kind == "telemetry":
        con.execute("""INSERT INTO fact_telemetry
            SELECT machine_id, datetime, date_key, volt, rotate, pressure, vibration FROM ins_df""")
    elif kind == "errors":
        con.execute("INSERT INTO fact_error SELECT machine_id, datetime, date_key, error_id FROM ins_df")
    elif kind == "maint":
        con.execute("INSERT INTO fact_maintenance SELECT machine_id, datetime, date_key, component FROM ins_df")
    elif kind == "failures":
        con.execute("INSERT INTO fact_failure SELECT machine_id, datetime, date_key, failed_component FROM ins_df")
    else:
        raise ValueError(kind)
    con.unregister("ins_df")
    return len(df)


def build_warehouse(cutoff: pd.Timestamp | None = C.LIVE_CUTOFF, verbose: bool = True) -> dict:
    """(Re)create the schema and load cleaned history strictly BEFORE `cutoff` (None = load everything)."""
    t = {n: pd.read_parquet(CLEAN / f"{n}_clean.parquet") for n in
         ["machines", "telemetry", "errors", "maint", "failures"]}
    con = connect()
    try:
        create_schema(con)
        _load_dim_date(con)
        con.register("m_df", t["machines"][["machine_id", "model", "age"]])
        con.execute("INSERT INTO dim_machine SELECT * FROM m_df")
        con.unregister("m_df")
        for comp in sorted(set(t["maint"]["component"]) | set(t["failures"]["failed_component"])):
            con.execute("INSERT INTO dim_component VALUES (?)", [comp])
        for et in sorted(t["errors"]["error_id"].unique()):
            con.execute("INSERT INTO dim_error_type VALUES (?)", [et])
        loaded = {}
        for kind in ["telemetry", "errors", "maint", "failures"]:
            d = t[kind] if cutoff is None else t[kind][t[kind]["datetime"] < cutoff]
            loaded[kind] = insert_facts(con, kind, d)
        con.execute("INSERT INTO system_state VALUES ('outcomes_known_until', ?)",
                    [str((cutoff or t["telemetry"]["datetime"].max()) - pd.Timedelta(hours=1))])
        con.execute("INSERT INTO system_state VALUES ('warehouse_built_at', ?)", [pd.Timestamp.now().isoformat()])
    finally:
        con.close()
    if verbose:
        print("warehouse loaded:", loaded)
    return loaded


# ---------------------------------------------------------------------------
# Integrity checks (used by tests and shown in the report)
# ---------------------------------------------------------------------------
def integrity_report(con) -> pd.DataFrame:
    rows = [(t, con.execute(f"SELECT COUNT(*) FROM {t}").fetchall()[0][0], "row count") for t in TABLES]
    orphan = {
        "telemetry -> dim_machine": "SELECT COUNT(*) FROM fact_telemetry f LEFT JOIN dim_machine d USING(machine_id) WHERE d.machine_id IS NULL",
        "telemetry -> dim_date": "SELECT COUNT(*) FROM fact_telemetry f LEFT JOIN dim_date d USING(date_key) WHERE d.date_key IS NULL",
        "error -> dim_error_type": "SELECT COUNT(*) FROM fact_error f LEFT JOIN dim_error_type d USING(error_type_id) WHERE d.error_type_id IS NULL",
        "maintenance -> dim_component": "SELECT COUNT(*) FROM fact_maintenance f LEFT JOIN dim_component d USING(component_id) WHERE d.component_id IS NULL",
        "failure -> dim_component": "SELECT COUNT(*) FROM fact_failure f LEFT JOIN dim_component d USING(component_id) WHERE d.component_id IS NULL",
        "failure -> dim_machine": "SELECT COUNT(*) FROM fact_failure f LEFT JOIN dim_machine d USING(machine_id) WHERE d.machine_id IS NULL",
    }
    for name, sql in orphan.items():
        rows.append((name, con.execute(sql).fetchall()[0][0], "orphan rows (must be 0)"))
    return pd.DataFrame(rows, columns=["check", "value", "meaning"])


# ---------------------------------------------------------------------------
# BI queries (the dashboard calls these; each is plain SQL on the star schema)
# ---------------------------------------------------------------------------
_LO, _HI = "1900-01-01", "2100-01-01"


def q_failures_by_component(con, d0=_LO, d1=_HI):
    return con.execute("""SELECT component_id AS component, COUNT(*) AS failures FROM fact_failure
                          WHERE event_time >= ? AND event_time < ? GROUP BY 1 ORDER BY 1""", [d0, d1]).df()


def q_failures_by_month(con, d0=_LO, d1=_HI):
    return con.execute("""SELECT d.year, d.month, d.month_name, f.component_id AS component, COUNT(*) AS failures
                          FROM fact_failure f JOIN dim_date d USING(date_key)
                          WHERE f.event_time >= ? AND f.event_time < ?
                          GROUP BY 1,2,3,4 ORDER BY 1,2,4""", [d0, d1]).df()


def q_failures_by_model(con, d0=_LO, d1=_HI):
    return con.execute("""SELECT m.model, COUNT(f.machine_id) AS failures,
                                 (SELECT COUNT(*) FROM dim_machine x WHERE x.model = m.model) AS machines_in_model
                          FROM dim_machine m LEFT JOIN fact_failure f
                               ON f.machine_id = m.machine_id AND f.event_time >= ? AND f.event_time < ?
                          GROUP BY m.model ORDER BY m.model""", [d0, d1]).df()


def q_errors_by_type(con, d0=_LO, d1=_HI):
    return con.execute("""SELECT error_type_id AS error_type, COUNT(*) AS events FROM fact_error
                          WHERE event_time >= ? AND event_time < ? GROUP BY 1 ORDER BY 1""", [d0, d1]).df()


def q_mtbf(con, d0=_LO, d1=_HI):
    """Observed machine-hours divided by failure events (a simple fleet MTBF)."""
    return con.execute("""SELECT (SELECT COUNT(*) FROM fact_telemetry WHERE reading_time >= ? AND reading_time < ?) AS machine_hours,
                                 (SELECT COUNT(*) FROM fact_failure WHERE event_time >= ? AND event_time < ?) AS failures""",
                       [d0, d1, d0, d1]).df().iloc[0]


def q_machine_failure_history(con, machine_id: int):
    return con.execute("""SELECT event_time, component_id AS component FROM fact_failure
                          WHERE machine_id = ? ORDER BY event_time""", [machine_id]).df()


def q_machine_maintenance(con, machine_id: int):
    return con.execute("""SELECT event_time, component_id AS component FROM fact_maintenance
                          WHERE machine_id = ? ORDER BY event_time DESC LIMIT 50""", [machine_id]).df()


def q_machine_errors(con, machine_id: int):
    return con.execute("""SELECT event_time, error_type_id AS error_type FROM fact_error
                          WHERE machine_id = ? ORDER BY event_time DESC LIMIT 50""", [machine_id]).df()


def q_machine_sensors(con, machine_id: int, hours: int):
    return con.execute("""SELECT reading_time, volt, rotate, pressure, vibration FROM fact_telemetry
                          WHERE machine_id = ? AND reading_time > (SELECT MAX(reading_time) FROM fact_telemetry) - INTERVAL (?) HOUR
                          ORDER BY reading_time""", [machine_id, hours]).df()


def q_data_extent(con):
    r = con.execute("SELECT MIN(reading_time), MAX(reading_time), COUNT(*) FROM fact_telemetry").fetchall()[0]
    return {"first": r[0], "last": r[1], "rows": r[2]}


if __name__ == "__main__":
    build_warehouse()
    con = connect(read_only=True)
    pd.set_option("display.width", 160)
    print(integrity_report(con).to_string(index=False))
    print("\nfailures by component:\n", q_failures_by_component(con).to_string(index=False))
    print("\nfailures by model:\n", q_failures_by_model(con).to_string(index=False))
    print("\nMTBF inputs:\n", q_mtbf(con).to_string())
    con.close()
