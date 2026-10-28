"""Phase 10/13: reproducible adaptation experiment on a SIMULATED stream (historical data replayed in weekly
batches, with a synthetic sensor offset injected from batch DRIFT_START_BATCH). This is NOT a live factory.

For every batch: validate -> store -> score with the current champion -> drift check -> upload outcomes ->
mature labels -> performance check. After the batches in SIM_RETRAIN_AFTER a challenger is trained and
promoted ONLY if the predefined criteria pass. Finally the adaptive system is compared with a STATIC one
(v1 never retrained, same fixed rule) on exactly the same labelled hours.

Run:  python -m src.simulate            (about 3-6 minutes; use --fast for a smaller forest)"""
from __future__ import annotations

import argparse
import warnings

import pandas as pd

from src import abi_config as C
from src import adapt, reset_demo
from src.config import REPORTS
from src.features import add_labels
from src.make_stream import batch_files, load_manifest
from src.warehouse import connect

warnings.filterwarnings("ignore")


def static_baseline(con, manifest, known) -> pd.DataFrame:
    """Score every batch period with the ORIGINAL v1 model and fixed threshold (never updated)."""
    v1 = con.execute("SELECT * FROM model_registry WHERE version = 'v1'").df().iloc[0]
    pipe = adapt.load_pipeline(v1["model_path"])
    cols = adapt.feature_meta()["feature_cols"]
    fails = con.execute("SELECT machine_id, event_time AS datetime FROM fact_failure").df()
    rows = []
    for b in manifest["batches"]:
        f = adapt.live_features(con, pd.Timestamp(b["start"]), pd.Timestamp(b["end"]))
        f["score"] = pipe.predict_proba(f[cols])[:, 1]
        lab = add_labels(f, fails, C.HORIZON_HOURS, last_ts=known)
        lab = lab[lab["fully_observed"] & (lab["failure_at_t"] == 0)]
        m = adapt.metrics_at(lab["y"], lab["score"], v1["threshold_high"])
        rows.append({"batch": b["batch"], **{f"static_{k}": m[k] for k in ("precision", "recall", "f1", "FP", "FN")}})
    return pd.DataFrame(rows)


def run(through: int = 18, n_estimators: int = 200) -> None:
    reset_demo.main(rebuild_stream=True)
    manifest = load_manifest()
    con = connect()
    log, decisions = [], []
    for b in manifest["batches"][:through]:
        f = batch_files(b["tag"])
        r = adapt.process_batch(con, {k: f[k] for k in ("telemetry", "errors", "maint")}, source="simulation")
        assert r["ok"], r["errors"]
        adapt.record_outcomes(con, f["outcomes"], pd.Timestamp(b["end"]))
        pc = adapt.performance_check(con)
        log.append({"batch": b["batch"], "batch_id": r["batch_id"], "week_start": b["start"][:10], "synthetic_drift": b["synthetic_drift_injected"],
                    "champion": r["model_version"], "drift_status": r["drift_status"], "max_psi": round(r["max_psi"], 3),
                    "high_risk_machines_at_week_end": r["high_risk_now"], "performance_status": pc["status"]})
        print(f"batch {b['batch']:>2} | {r['model_version']} | drift {r['drift_status']:6} (max PSI {r['max_psi']:.2f}) | "
              f"performance {pc['status']}")
        if b["batch"] in C.SIM_RETRAIN_AFTER:
            ch = adapt.train_challenger(con, n_estimators=n_estimators)
            if not ch.get("ok"):
                print("   challenger NOT trained:", ch.get("reason"))
                continue
            ev = ch["evaluation"]
            action = "promoted" if ch["passed"] else "rejected"
            if ch["passed"]:
                adapt.promote(con, ch["version"])
            else:
                adapt.reject(con, ch["version"])
            print(f"   challenger {ch['version']}: PR-AUC {ev['challenger_pr_auc']:.4f} vs champion {ev['champion_pr_auc']:.4f}, "
                  f"F1 {ev['challenger_f1']:.3f} vs {ev['champion_f1']:.3f} -> {action.upper()}")
            decisions.append({"after_batch": b["batch"], "challenger": ch["version"], "champion": ch["champion"],
                              "train_period": ch["train_period"], "val_period": ch["val_period"],
                              "champion_pr_auc": ev["champion_pr_auc"], "challenger_pr_auc": ev["challenger_pr_auc"],
                              "champion_f1": ev["champion_f1"], "challenger_f1": ev["challenger_f1"], "decision": action})

    known = pd.Timestamp(adapt.get_state(con, "outcomes_known_until"))
    adaptive = adapt.performance_by_batch(con)
    out = (pd.DataFrame(log)
           .merge(adaptive[["batch_id", "precision", "recall", "f1", "FP", "FN"]].rename(columns=lambda c: c if c == "batch_id" else f"adaptive_{c}"), on="batch_id")
           .merge(static_baseline(con, {"batches": manifest["batches"][:through]}, known), on="batch"))
    REPORTS.mkdir(exist_ok=True)
    out.round(4).to_csv(REPORTS / "adaptation_experiment_batches.csv", index=False)
    pd.DataFrame(decisions).round(4).to_csv(REPORTS / "adaptation_challenger_decisions.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n=== STATIC (v1 never updated) vs ADAPTIVE (gated retraining), same labelled hours ===")
    print(out[["batch", "champion", "drift_status", "synthetic_drift", "static_precision", "static_f1", "adaptive_precision",
               "adaptive_f1", "static_FP", "adaptive_FP"]].round(3).to_string(index=False))
    post = out[out["champion"] != "v1"]
    if len(post):
        print(f"\nBatches scored by a promoted model: {list(post['batch'])}")
        print(f"  mean F1  static {post['static_f1'].mean():.3f}  vs adaptive {post['adaptive_f1'].mean():.3f}")
        print(f"  false alarms  static {int(post['static_FP'].sum())}  vs adaptive {int(post['adaptive_FP'].sum())}")
    print("\nsaved reports/adaptation_experiment_batches.csv and reports/adaptation_challenger_decisions.csv")
    con.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--through", type=int, default=18)
    ap.add_argument("--fast", action="store_true", help="60-tree forest for quick runs")
    a = ap.parse_args()
    run(a.through, 60 if a.fast else 200)
