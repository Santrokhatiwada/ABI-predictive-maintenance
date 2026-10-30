# """Phase 11: Adaptive BI dashboard.   Launch from the project root:  streamlit run app/streamlit_app.py
# Every number on every page comes from the DuckDB warehouse (data/warehouse/pdm.duckdb)."""
# import json
# import sys
# from contextlib import contextmanager
# from pathlib import Path

# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# import numpy as np
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# import streamlit as st
# from plotly.subplots import make_subplots
# from sklearn.metrics import precision_recall_curve

# from src import abi_config as C
# from src import adapt
# from src import warehouse as W
# from src.config import FIG, REPORTS, SENSORS

# st.set_page_config(page_title="ABI Predictive Maintenance", layout="wide", page_icon="🛠️")
# BAND_COLOURS = {"high": "#d62728", "medium": "#ff9f1c", "low": "#2ca02c"}
# PAGES = ["1. Fleet overview", "2. Machine drill-down", "3. Failure & error analysis", "4. Model performance",
#          "5. Monitoring & model management", "6. New data & feedback"]


# @contextmanager
# def db():
#     """Short-lived connection: opened for one task, then closed (avoids file-lock clashes)."""
#     con = W.connect()
#     try:
#         yield con
#     finally:
#         con.close()


# def flash(kind: str, text: str):
#     st.session_state["flash"] = (kind, text)


# def show_flash():
#     if "flash" in st.session_state:
#         kind, text = st.session_state.pop("flash")
#         getattr(st, kind)(text)


# # ------------------------------------------------------------------ setup guard
# if not W.db_exists():
#     st.error("The warehouse does not exist yet. In Terminal run:  python -m src.reset_demo")
#     st.stop()
# with db() as _c:
#     try:
#         champ = adapt.get_champion(_c)
#         extent = W.q_data_extent(_c)
#         last_batch = _c.execute("SELECT * FROM batch_log ORDER BY data_end DESC LIMIT 1").df()
#         perf = adapt.performance_check(_c) if champ else None
#         known_until = adapt.get_state(_c, "outcomes_known_until")
#     except Exception as ex:   # e.g. tables missing
#         st.error(f"Warehouse is not initialised ({ex}). Run: python -m src.reset_demo")
#         st.stop()
# if champ is None:
#     st.error("No champion model registered. Run: python -m src.reset_demo")
#     st.stop()

# # ------------------------------------------------------------------ sidebar
# st.sidebar.title("🛠️ Adaptive BI")
# st.sidebar.caption("Predictive maintenance · Azure PdM dataset (simulated data)")
# page = st.sidebar.radio("Page", PAGES, key="page")
# st.sidebar.markdown("---")
# st.sidebar.markdown(f"**Champion model:** `{champ['version']}`")
# st.sidebar.markdown(f"**Data through:** {extent['last']}")
# if not last_batch.empty:
#     ds = last_batch.iloc[0]["drift_status"]
#     st.sidebar.markdown(f"**Latest batch drift:** {'🔴' if ds == 'alert' else '🟠' if ds == 'watch' else '🟢'} {ds}")
# if perf:
#     st.sidebar.markdown(f"**Performance check:** {'🔴' if perf['status'] == 'degraded' else '🟢' if perf['status'] == 'ok' else '⚪'} {perf['status']}")
# st.sidebar.markdown("---")
# st.sidebar.subheader("Decision thresholds")
# hi = st.sidebar.slider("High-risk threshold (alert)", 0.005, 0.99, float(round(champ["threshold_high"], 3)), 0.005,
#                        key=f"hi_{champ['version']}")
# med = st.sidebar.slider("Medium-risk threshold", 0.0, float(hi), float(min(round(champ["threshold_medium"], 3), hi)), 0.005,
#                         key=f"med_{champ['version']}")
# st.sidebar.caption("Defaults are the champion's thresholds, chosen on validation data. Moving the sliders re-bands the "
#                    "stored scores instantly (no re-scoring).")
# SCORE_NOTE = ("Score = the model's output between 0 and 1, used to rank machines. It is **not** a guaranteed, calibrated "
#               "failure probability. 'High' means: a failure is predicted within the next 24 hours.")
# show_flash()


# # ================================================================== 1. FLEET OVERVIEW
# def page_overview():
#     st.header("Fleet overview")
#     with db() as con:
#         risk = adapt.latest_machine_risk(con)
#         f30 = con.execute("""SELECT COUNT(*) FROM fact_failure WHERE event_time > (SELECT MAX(reading_time) FROM fact_telemetry) - INTERVAL 30 DAY""").fetchall()[0][0]
#         n_mach = con.execute("SELECT COUNT(*) FROM dim_machine").fetchall()[0][0]
#     if risk.empty:
#         st.info("No predictions yet. Open **6. New data & feedback** and load a batch.")
#         return
#     risk["band"] = adapt.risk_band(risk["score"], hi, med)
#     k = st.columns(5)
#     k[0].metric("Machines monitored", n_mach)
#     k[1].metric("High risk now", int((risk["band"] == "high").sum()))
#     k[2].metric("Medium risk now", int((risk["band"] == "medium").sum()))
#     k[3].metric("Failures, last 30 days", f30)
#     k[4].metric("Risk as of", str(risk["prediction_time"].iloc[0])[:16])
#     st.caption(SCORE_NOTE)
#     c1, c2 = st.columns([1, 2])
#     counts = risk["band"].value_counts().reindex(["high", "medium", "low"], fill_value=0).reset_index()
#     counts.columns = ["band", "machines"]
#     c1.plotly_chart(px.bar(counts, x="band", y="machines", color="band", color_discrete_map=BAND_COLOURS,
#                            title="Machines by risk band"), width="stretch")
#     top = risk.head(15)
#     c2.plotly_chart(px.bar(top, x=top["machine_id"].astype(str), y="score", color="band", color_discrete_map=BAND_COLOURS,
#                            title="Top 15 machines by current score", labels={"x": "machine"}), width="stretch")
#     st.subheader("Maintenance priority list")
#     f1, f2 = st.columns(2)
#     bands = f1.multiselect("Risk band", ["high", "medium", "low"], default=["high", "medium"])
#     models = f2.multiselect("Machine model", sorted(risk["model"].unique()), default=sorted(risk["model"].unique()))
#     view = risk[risk["band"].isin(bands) & risk["model"].isin(models)].copy()
#     view.insert(0, "priority", range(1, len(view) + 1))
#     st.dataframe(view[["priority", "machine_id", "model", "age", "score", "max_score_24h", "band", "model_version"]],
#                  width="stretch", hide_index=True,
#                  column_config={"score": st.column_config.ProgressColumn("score", min_value=0.0, max_value=1.0, format="%.3f"),
#                                 "max_score_24h": st.column_config.NumberColumn("max score, last 24h", format="%.3f")})
#     st.caption("Ranking = latest score (ties broken by the 24h maximum). Maintenance decisions still need human judgement: "
#                "the system has no cost or downtime data.")


# # ================================================================== 2. MACHINE DRILL-DOWN
# def page_machine():
#     st.header("Machine drill-down")
#     with db() as con:
#         ids = [r[0] for r in con.execute("SELECT machine_id FROM dim_machine ORDER BY 1").fetchall()]
#     c = st.columns([1, 1, 2])
#     m = c[0].selectbox("Machine", ids)
#     days = c[1].slider("History shown (days)", 2, 60, 14)
#     with db() as con:
#         info = con.execute("SELECT model, age FROM dim_machine WHERE machine_id = ?", [m]).fetchall()[0]
#         sens = W.q_machine_sensors(con, m, days * 24)
#         fails = W.q_machine_failure_history(con, m)
#         errs = W.q_machine_errors(con, m)
#         mnt = W.q_machine_maintenance(con, m)
#         pred = con.execute("""SELECT prediction_time, score, model_version FROM predictions WHERE machine_id = ?
#                               AND prediction_time > (SELECT MAX(prediction_time) FROM predictions) - INTERVAL (?) HOUR
#                               ORDER BY prediction_time""", [m, days * 24]).df()
#     c[2].markdown(f"**Model:** {info[0]} · **Age:** {info[1]} · **Failures on record:** {len(fails)}")
#     fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.03,
#                         subplot_titles=SENSORS + ["Risk score (champion thresholds shown as dashed lines)"])
#     for i, s in enumerate(SENSORS, 1):
#         fig.add_trace(go.Scatter(x=sens["reading_time"], y=sens[s], mode="lines", line=dict(width=1), name=s), row=i, col=1)
#     if len(pred):
#         fig.add_trace(go.Scatter(x=pred["prediction_time"], y=pred["score"], mode="lines", line=dict(color="#d62728", width=1.5),
#                                  name="risk score"), row=5, col=1)
#         fig.add_hline(y=hi, line_dash="dash", line_color="#d62728", row=5, col=1)
#         fig.add_hline(y=med, line_dash="dash", line_color="#ff9f1c", row=5, col=1)
#     lo_t = sens["reading_time"].min() if len(sens) else None
#     for t in fails["event_time"]:
#         if lo_t is not None and t >= lo_t:
#             fig.add_shape(type="line", xref="x", yref="paper", x0=t, x1=t, y0=0, y1=1, line=dict(color="black", dash="dot", width=1))
#     fig.update_layout(height=900, showlegend=False, margin=dict(t=40))
#     st.plotly_chart(fig, width="stretch")
#     st.caption("Black dotted lines = recorded failures. Scores exist only for hours that have been scored.")
#     a, b, d = st.columns(3)
#     a.subheader("Recent errors"); a.dataframe(errs, hide_index=True, width="stretch")
#     b.subheader("Maintenance history"); b.dataframe(mnt, hide_index=True, width="stretch")
#     d.subheader("Failure history"); d.dataframe(fails, hide_index=True, width="stretch")


# # ================================================================== 3. FAILURE & ERROR ANALYSIS
# def page_failures():
#     st.header("Failure & error analysis (warehouse queries)")
#     lo, hi_d = extent["first"].date(), extent["last"].date()
#     rng = st.date_input("Date range", (lo, hi_d), min_value=pd.Timestamp("2014-06-01").date(), max_value=hi_d)
#     if not isinstance(rng, tuple) or len(rng) != 2:
#         st.info("Select an end date."); return
#     d0, d1 = pd.Timestamp(rng[0]), pd.Timestamp(rng[1]) + pd.Timedelta(days=1)
#     with db() as con:
#         bc, bm, bmod, be, mt = (W.q_failures_by_component(con, d0, d1), W.q_failures_by_month(con, d0, d1),
#                                 W.q_failures_by_model(con, d0, d1), W.q_errors_by_type(con, d0, d1), W.q_mtbf(con, d0, d1))
#     k = st.columns(3)
#     k[0].metric("Failures in range", int(mt["failures"]))
#     k[1].metric("Machine-hours observed", f"{int(mt['machine_hours']):,}")
#     k[2].metric("Fleet MTBF (hours per failure)", f"{mt['machine_hours'] / mt['failures']:.0f}" if mt["failures"] else "n/a")
#     c1, c2 = st.columns(2)
#     c1.plotly_chart(px.bar(bc, x="component", y="failures", title="Failures by component"), width="stretch")
#     if len(bm):
#         bm["month"] = bm["year"].astype(str) + "-" + bm["month"].astype(str).str.zfill(2)
#         c2.plotly_chart(px.bar(bm, x="month", y="failures", color="component", title="Failures per month"), width="stretch")
#     bmod["failures_per_machine"] = bmod["failures"] / bmod["machines_in_model"]
#     c3, c4 = st.columns(2)
#     c3.plotly_chart(px.bar(bmod, x="model", y="failures_per_machine", title="Failures per machine, by model"), width="stretch")
#     c4.plotly_chart(px.bar(be, x="error_type", y="events", title="Error events by type"), width="stretch")
#     st.caption("Component, error and model codes are anonymised in the dataset; no physical meaning is assumed. "
#                "MTBF here is observed machine-hours ÷ failures, a simple descriptive figure.")


# # ================================================================== 4. MODEL PERFORMANCE
# def page_performance():
#     st.header("Model performance")
#     st.subheader("Offline evaluation (Phase 8, held-out test period)")
#     mc = REPORTS / "model_comparison.csv"
#     if mc.exists():
#         t = pd.read_csv(mc)
#         cols = [c for c in ["model", "threshold_from_val", "test_pr_auc", "test_precision", "test_recall", "test_f1",
#                             "test_accuracy", "test_FP", "test_FN"] if c in t.columns]
#         st.dataframe(t[cols], hide_index=True, width="stretch")
#         st.caption("Dummy accuracy shows why accuracy alone is misleading for rare failures.")
#     imgs = [FIG / n for n in ("eval_pr_curves.png", "eval_confusion_matrix.png", "eval_feature_importance.png") if (FIG / n).exists()]
#     if imgs:
#         for col, p in zip(st.columns(len(imgs)), imgs):
#             col.image(str(p), width="stretch")
#     st.subheader("Live performance on matured predictions (feedback loop)")
#     with db() as con:
#         d = adapt.matured_predictions(con)
#         reg = con.execute("SELECT version FROM model_registry ORDER BY created_at").df()["version"].tolist()
#     if d.empty:
#         st.info("No predictions have matured yet. Score a batch, then upload its outcomes in page 6."); return
#     versions = st.multiselect("Model versions included", sorted(d["model_version"].unique()), default=sorted(d["model_version"].unique()))
#     d = d[d["model_version"].isin(versions)]
#     thr = st.slider("What-if alert threshold", 0.005, 0.99, float(hi), 0.005, key="whatif")
#     m = adapt.metrics_at(d["actual_outcome"], d["score"], thr)
#     k = st.columns(5)
#     k[0].metric("Precision", f"{m['precision']:.3f}" if not np.isnan(m["precision"]) else "n/a")
#     k[1].metric("Recall", f"{m['recall']:.3f}" if not np.isnan(m["recall"]) else "n/a")
#     k[2].metric("F1", f"{m['f1']:.3f}" if not np.isnan(m["f1"]) else "n/a")
#     k[3].metric("False alarms (hours)", m["FP"])
#     k[4].metric("Missed (hours)", m["FN"])
#     c1, c2 = st.columns(2)
#     cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
#     c1.plotly_chart(px.imshow(cm, text_auto=",", x=["no alert", "alert"], y=["no failure", "failure"], color_continuous_scale="Blues",
#                               title="Confusion matrix (machine-hours)"), width="stretch")
#     prec, rec, th = precision_recall_curve(d["actual_outcome"], d["score"])
#     fig = go.Figure([go.Scatter(x=th, y=prec[:-1], name="precision"), go.Scatter(x=th, y=rec[:-1], name="recall")])
#     fig.add_vline(x=thr, line_dash="dash")
#     fig.update_layout(title="Threshold trade-off on live matured data", xaxis_title="threshold")
#     c2.plotly_chart(fig, width="stretch")
#     st.caption(f"Based on {len(d):,} matured machine-hours ({int(d['actual_outcome'].sum()):,} positive). Fewer false alarms "
#                "(higher threshold) usually means more missed failures; the slider shows that trade-off on real outcomes. "
#                "Hours are autocorrelated, so treat small differences with caution.")


# # ================================================================== 5. MONITORING & MODEL MANAGEMENT
# def page_monitoring():
#     st.header("Monitoring & model management")
#     with db() as con:
#         dl = con.execute("SELECT * FROM drift_log ORDER BY checked_at").df()
#         bl = con.execute("SELECT * FROM batch_log ORDER BY data_start").df()
#         pb = adapt.performance_by_batch(con)
#         reg = con.execute("SELECT version,status,algorithm,created_at,train_start,train_end,threshold_high,threshold_medium,val_pr_auc,val_f1 FROM model_registry ORDER BY created_at").df()
#         ev = con.execute("SELECT * FROM model_events ORDER BY event_time DESC").df()
#         prev = adapt.get_state(con, "previous_champion", "")
#         chal = con.execute("SELECT * FROM model_registry WHERE status = 'challenger' ORDER BY created_at DESC LIMIT 1").df()
#     a, b = st.columns(2)
#     if not bl.empty:
#         s = bl.iloc[-1]
#         (a.error if s["drift_status"] == "alert" else a.warning if s["drift_status"] == "watch" else a.success)(
#             f"**Data drift (no labels needed):** latest batch is **{s['drift_status']}** (max PSI {s['max_psi']:.2f}).")
#     if perf:
#         txt = {"degraded": f"**Performance (needs labels):** F1 {perf.get('f1', float('nan')):.3f} is more than {C.PERF_F1_DROP} below the validation F1 {perf['baseline_f1']:.3f}.",
#                "ok": f"**Performance (needs labels):** F1 {perf.get('f1', float('nan')):.3f} is within {C.PERF_F1_DROP} of validation F1 {perf['baseline_f1']:.3f}.",
#                "insufficient_labels": "**Performance (needs labels):** not enough matured outcomes yet to judge."}[perf["status"]]
#         (b.error if perf["status"] == "degraded" else b.success if perf["status"] == "ok" else b.info)(txt)
#     with st.expander("How to read this: drift vs. performance degradation"):
#         st.markdown(f"""
# * **Data drift** = the inputs now look different from the training data. Measured with the Population Stability Index (PSI) per feature:
#   < {C.PSI_WATCH} stable, {C.PSI_WATCH}–{C.PSI_ALERT} watch, > {C.PSI_ALERT} alert (common rules of thumb). It needs **no labels**, so it can warn immediately.
# * **Performance degradation** = predictions are now worse. It needs **actual outcomes**, which arrive late (a prediction can only be judged once its 24h window has passed).
# * Drift does **not** prove accuracy has fallen, and good accuracy does not prove there is no drift. Treat drift as a prompt to investigate.
# * **Retraining** is gated: a challenger replaces the champion only if it beats it by ≥ {C.PROMOTE_MARGIN} PR-AUC on a newer labelled block (and F1 is not worse).""")
#     if not dl.empty:
#         piv = dl.pivot_table(index="feature", columns="batch_id", values="psi", aggfunc="max")
#         piv = piv[[c for c in bl["batch_id"] if c in piv.columns]]
#         fig = go.Figure(go.Heatmap(z=np.minimum(piv.values, 1.0), x=[c[:8] for c in piv.columns], y=piv.index, text=piv.values.round(2),
#                                    texttemplate="%{text}", colorscale="YlOrRd", zmin=0, zmax=1, colorbar=dict(title="PSI (capped at 1)")))
#         fig.update_layout(title="Drift per feature per batch (cell text = PSI, colour capped at 1)", height=380)
#         st.plotly_chart(fig, width="stretch")
#     if not pb.empty:
#         long = pb.melt(id_vars=["week_start", "model_version"], value_vars=["precision", "recall", "f1"])
#         fig = px.line(long, x="week_start", y="value", color="variable", markers=True, hover_data=["model_version"],
#                       title="Live performance per batch (matured predictions only; hover shows model version)")
#         fig.update_yaxes(range=[0, 1.02])
#         st.plotly_chart(fig, width="stretch")
#     st.subheader("Model registry")
#     st.dataframe(reg, hide_index=True, width="stretch")
#     st.markdown("**Controlled retraining (champion / challenger)**")
#     n_est = st.selectbox("Forest size for the challenger", [60, 200], index=1, help="60 trees is faster; 200 matches the champion's recipe.")
#     if st.button("Train challenger on recent labelled data", type="primary", key="train_btn"):
#         with st.spinner("Building features and training... (about 1–3 minutes)"):
#             with db() as con:
#                 res = adapt.train_challenger(con, n_estimators=n_est)
#         flash("success" if res.get("ok") else "warning",
#               f"Challenger {res['version']} trained." if res.get("ok") else f"No challenger trained: {res.get('reason')}")
#         st.rerun()
#     if not chal.empty:
#         row = chal.iloc[0]
#         e = json.loads(row["evaluation_json"])
#         st.markdown(f"**Latest challenger: `{row['version']}`**  (trained on {str(row['train_start'])[:10]} → {str(row['train_end'])[:10]})")
#         cmp = pd.DataFrame({"metric": ["PR-AUC", "F1 at own best threshold"],
#                             "champion": [e["champion_pr_auc"], e["champion_f1"]], "challenger": [e["challenger_pr_auc"], e["challenger_f1"]]})
#         st.dataframe(cmp.round(4), hide_index=True, width="stretch")
#         for k_, v_ in e["criteria"].items():
#             st.write(("✅ " if v_ else "❌ ") + k_.replace("_", " "))
#         c1, c2 = st.columns(2)
#         if c1.button("Promote challenger", disabled=not e["passed"], key="promote_btn"):
#             with db() as con:
#                 r = adapt.promote(con, row["version"])
#             flash("success" if r["ok"] else "error", f"Promoted {r.get('promoted')}; {r.get('replaced')} retired." if r["ok"] else r["reason"])
#             st.rerun()
#         if c2.button("Reject challenger", key="reject_btn"):
#             with db() as con:
#                 adapt.reject(con, row["version"])
#             flash("info", f"{row['version']} rejected. Champion unchanged."); st.rerun()
#         if not e["passed"]:
#             st.caption("Promotion is disabled because the predefined criteria were not met.")
#     if prev and st.button(f"Roll back to {prev}", key="rollback_btn"):
#         with db() as con:
#             r = adapt.rollback(con)
#         flash("warning", f"Restored {r.get('restored')}; {r.get('withdrawn')} withdrawn."); st.rerun()
#     st.subheader("Audit trail")
#     st.dataframe(ev, hide_index=True, width="stretch")
#     st.subheader("Batch log")
#     st.dataframe(bl, hide_index=True, width="stretch")


# # ================================================================== 6. NEW DATA & FEEDBACK
# def _next_batch():
#     p = C.STREAM_DIR / "manifest.json"
#     if not p.exists():
#         return None, None
#     man = json.loads(p.read_text())
#     for b in man["batches"]:
#         if pd.Timestamp(b["start"]) > pd.Timestamp(extent["last"]):
#             return b, man
#     return None, man


# def _show_result(res: dict):
#     for w in res.get("warnings", []):
#         st.warning(w)
#     for e in res.get("errors", []):
#         st.error(e)


# def page_data():
#     st.header("New data & feedback")
#     with db() as con:
#         cnt = con.execute("SELECT outcome_status, COUNT(*) FROM predictions GROUP BY 1").df()
#     st.caption(f"Data stored through **{extent['last']}**. Failures are known up to **{known_until}**. "
#                "Predictions: " + ", ".join(f"{r[0]} {r[1]:,}" for r in cnt.itertuples(index=False)) if len(cnt) else
#                f"Data stored through **{extent['last']}**.")

#     st.subheader("A. Simulated arrival of the next weekly batch")
#     nb, man = _next_batch()
#     if nb is None:
#         st.info("No further stream batches (or no stream files; run `python -m src.make_stream`).")
#     else:
#         st.write(f"Next batch: **{nb['tag']}** ({nb['start'][:10]} → {nb['end'][:10]}, {nb['rows']:,} readings, "
#                  f"{nb['errors']} errors, {nb['maint']} maintenance events)")
#         if nb.get("synthetic_drift_injected"):
#             st.warning(f"This batch contains the **synthetic** {man['drift_sensor']} offset (+{man['drift_offset']}) used to test drift monitoring.")
#         use_spike = use_burst = False
#         if nb.get("variants"):
#             st.info("Demo variants exist for this week (synthetic): "
#                     f"sensor spike on machine {nb['variants']['spike_machine']}, error burst on machine {nb['variants']['error_burst_machine']}.")
#             use_spike = st.checkbox("Use the sensor-spike telemetry variant", key="spike")
#             use_burst = st.checkbox("Use the error-burst variant", key="burst")
#         if st.button(f"Load {nb['tag']} → validate, store, score", type="primary", key="load_next"):
#             tag = nb["tag"]
#             raw = {"telemetry": pd.read_csv(C.STREAM_DIR / (f"{tag}_telemetry_SPIKE.csv" if use_spike else f"{tag}_telemetry.csv")),
#                    "errors": pd.read_csv(C.STREAM_DIR / (f"{tag}_errors_BURST.csv" if use_burst else f"{tag}_errors.csv")),
#                    "maint": pd.read_csv(C.STREAM_DIR / f"{tag}_maint.csv")}
#             with st.spinner("Processing batch..."), db() as con:
#                 res = adapt.process_batch(con, raw, source=f"stream:{tag}")
#             if res["ok"]:
#                 flash("success", f"Batch {res['batch_id']} processed: {res['n_scored']:,} predictions with {res['model_version']}, "
#                                  f"drift {res['drift_status']} (max PSI {res['max_psi']:.2f}), {res['high_risk_now']} machine(s) high risk now "
#                                  f"({res['seconds']}s). Outcomes are NOT known yet; see section C.")
#                 st.rerun()
#             _show_result(res)

#     st.subheader("B. Upload your own files")
#     st.caption("Required columns, as in the original dataset: telemetry `datetime, machineID, volt, rotate, pressure, vibration`; "
#                "errors `datetime, machineID, errorID`; maintenance `datetime, machineID, comp`.")
#     u = st.columns(3)
#     f_t, f_e, f_m = u[0].file_uploader("Telemetry CSV (required)", type="csv"), u[1].file_uploader("Errors CSV (optional)", type="csv"), \
#         u[2].file_uploader("Maintenance CSV (optional)", type="csv")
#     cc = st.columns(2)
#     do_check, do_run = cc[0].button("Validate only (stores nothing)", key="val_only"), cc[1].button("Validate, store & score", type="primary", key="run_upload")
#     if (do_check or do_run) and f_t is None:
#         st.error("Please choose a telemetry file.")
#     elif do_check or do_run:
#         try:
#             raw = {"telemetry": pd.read_csv(f_t), "errors": pd.read_csv(f_e) if f_e else None, "maint": pd.read_csv(f_m) if f_m else None}
#         except Exception as ex:
#             st.error(f"The file could not be read as CSV: {ex}"); raw = None
#         if raw:
#             if do_check:
#                 with db() as con:
#                     allw, alle = [], []
#                     for k_, d_ in raw.items():
#                         if d_ is not None:
#                             _, e_, w_ = adapt.validate_upload(k_, d_, con); alle += e_; allw += w_
#                 _show_result({"errors": alle, "warnings": allw})
#                 if not alle:
#                     st.success("Validation passed. Nothing was stored.")
#             else:
#                 with st.spinner("Processing..."), db() as con:
#                     res = adapt.process_batch(con, raw, source="upload")
#                 if res["ok"]:
#                     flash("success", f"Stored and scored {res['n_scored']:,} machine-hours; drift {res['drift_status']}; "
#                                      f"{res['high_risk_now']} high-risk machine(s) now.")
#                     st.rerun()
#                 _show_result(res)

#     st.subheader("C. Feedback: record what actually happened")
#     st.caption("Predictions can only be judged once their 24-hour window has fully passed *and* you declare failures complete up to that time. "
#                "Absence of a failure row then means 'no failure'.")
#     default_known = pd.Timestamp(extent["last"])
#     last_tag, last_end = None, None
#     if man:
#         for b in man["batches"]:
#             if pd.Timestamp(b["start"]) <= pd.Timestamp(extent["last"]):
#                 last_tag, last_end = b["tag"], pd.Timestamp(b["end"])
#     k_txt = st.text_input("Failures are complete up to (YYYY-MM-DD HH:MM:SS) - used for uploaded files",
#                           value=str(default_known), key=f"known_txt_{default_known}")
#     fo = st.file_uploader("Outcomes CSV: `datetime, machineID, failure`", type="csv", key="outcomes_up")
#     c1, c2 = st.columns(2)
#     run_up = c1.button("Record uploaded outcomes", key="rec_up")
#     run_stream = c2.button(f"Record outcomes of the last loaded stream batch ({last_tag})", key="rec_stream", disabled=last_tag is None)
#     if run_up or run_stream:
#         try:
#             ku = pd.Timestamp(k_txt) if run_up else last_end      # stream batch: complete up to the batch end
#             raw_o = pd.read_csv(fo) if (run_up and fo) else (pd.read_csv(C.STREAM_DIR / f"{last_tag}_outcomes.csv") if run_stream else None)
#             if raw_o is None:
#                 st.error("Please choose an outcomes file.")
#             else:
#                 with db() as con:
#                     res = adapt.record_outcomes(con, raw_o, ku)
#                 if res["ok"]:
#                     flash("success", f"Recorded {res['failures_added']} failure(s); {res['matured']:,} prediction(s) matured. "
#                                      "Open pages 4 and 5 to see updated performance.")
#                     st.rerun()
#                 _show_result(res)
#         except Exception as ex:
#             st.error(f"Could not process outcomes: {ex}")


# {PAGES[0]: page_overview, PAGES[1]: page_machine, PAGES[2]: page_failures, PAGES[3]: page_performance,
#  PAGES[4]: page_monitoring, PAGES[5]: page_data}[page]()
"""Phase 11: Adaptive BI dashboard.   Launch from the project root:  streamlit run app/streamlit_app.py
Every number on every page comes from the DuckDB warehouse (data/warehouse/pdm.duckdb)."""
import json
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from plotly.subplots import make_subplots
from sklearn.metrics import precision_recall_curve

from src import abi_config as C
from src import adapt
from src import warehouse as W
from src.config import FIG, REPORTS, SENSORS

st.set_page_config(page_title="ABI Predictive Maintenance", layout="wide", page_icon="🔧")
BAND_COLOURS = {"high": "#ee6666", "medium": "#fac858", "low": "#91cc75"}

# ------------------------------------------------------------------ look and feel
# Charts follow the Apache ECharts default palette and style: white background, thin axes,
# light horizontal grid lines, small left-aligned titles.
ECHARTS = ["#5470c6", "#91cc75", "#fac858", "#ee6666", "#73c0de", "#3ba272", "#fc8452", "#9a60b4", "#ea7ccc"]
pio.templates["echarts"] = go.layout.Template(layout=dict(
    colorway=ECHARTS,
    font=dict(family="Helvetica Neue, Helvetica, Arial, sans-serif", size=12, color="#464646"),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    title=dict(font=dict(size=15, color="#333333"), x=0, xanchor="left"),
    margin=dict(l=48, r=16, t=56, b=40),
    xaxis=dict(showgrid=False, zeroline=False, linecolor="#6e7079", ticks="outside", tickcolor="#6e7079"),
    yaxis=dict(showgrid=True, gridcolor="#e0e6f1", zeroline=False, showline=False),
    legend=dict(orientation="h", y=1.08, x=1, xanchor="right", bgcolor="rgba(0,0,0,0)"),
    hoverlabel=dict(bgcolor="white", bordercolor="#cccccc", font=dict(color="#464646")),
))
pio.templates.default = "echarts"

_plotly_chart = DeltaGenerator.plotly_chart
def _plotly_chart_no_theme(self, *a, **k):
    k.setdefault("theme", None)          # keep our template instead of Streamlit's own chart theme
    return _plotly_chart(self, *a, **k)
DeltaGenerator.plotly_chart = _plotly_chart_no_theme

st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {
    padding-top: 2.5rem;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: 100%;
}
html, body, [class*="css"] {font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;}
[data-testid="stSidebar"] {background: #f7f8fa; border-right: 1px solid #e6e8ee;}
.brand {font-size: 16px; font-weight: 600; color: #333; margin: 4px 0 2px;}
.side-h {font-size: 12px; color: #6e7079; text-transform: uppercase; letter-spacing: .04em; margin: 6px 0 -6px;}
.kv {font-size: 13px; color: #333; margin: 2px 0; display: flex; gap: 8px; align-items: center;}
.kv span {color: #6e7079; min-width: 92px;}
.dot {display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 2px;}
.dot.green {background: #91cc75;} .dot.amber {background: #fac858;} .dot.red {background: #ee6666;} .dot.grey {background: #c4c7d0;}
.page-title {font-size: 22px; font-weight: 600; color: #333; margin: 0;}
.page-sub {font-size: 14px; color: #6e7079; margin: 2px 0 18px;}
h2, h3 {font-size: 16px !important; font-weight: 600 !important; color: #333; margin-top: 1.4rem !important;}
[data-testid="stMetric"] {background: #fff; border: 1px solid #e6e8ee; border-radius: 4px; padding: 10px 14px;}
[data-testid="stMetricLabel"] {color: #6e7079; font-size: 12px;}
[data-testid="stMetricValue"] {font-size: 24px; font-weight: 600; color: #333;}
.stButton > button {border-radius: 4px; font-size: 14px;}
.stButton > button[kind="primary"] {background: #5470c6; border-color: #5470c6; color: #fff;}
.stButton > button[kind="primary"]:hover {background: #4561b3; border-color: #4561b3;}
[data-testid="stAlert"] {border-radius: 4px;}
</style>
""", unsafe_allow_html=True)


def page_title(title: str, sub: str = "") -> None:
    st.markdown(f"<div class='page-title'>{title}</div><div class='page-sub'>{sub}</div>", unsafe_allow_html=True)

PAGES = ["Overview", "Machines", "Failures and errors", "Model accuracy",  "System architecture", "Model updates", "Add data"]
DRIFT_TXT = {"stable": "no change", "watch": "small change", "alert": "big change"}
PERF_TXT = {"ok": "working well", "degraded": "getting worse", "insufficient_labels": "not enough results yet"}


@contextmanager
def db():
    """Short-lived connection: opened for one task, then closed (avoids file-lock clashes)."""
    con = W.connect()
    try:
        yield con
    finally:
        con.close()


def flash(kind: str, text: str):
    st.session_state["flash"] = (kind, text)


def show_flash():
    if "flash" in st.session_state:
        kind, text = st.session_state.pop("flash")
        getattr(st, kind)(text)


# ------------------------------------------------------------------ setup guard
if not W.db_exists():
    st.error("The app's data store has not been set up yet. Run this once in Terminal:  python -m src.reset_demo")
    st.stop()
with db() as _c:
    try:
        champ = adapt.get_champion(_c)
        extent = W.q_data_extent(_c)
        last_batch = _c.execute("SELECT * FROM batch_log ORDER BY data_end DESC LIMIT 1").df()
        perf = adapt.performance_check(_c) if champ else None
        known_until = adapt.get_state(_c, "outcomes_known_until")
    except Exception as ex:   # e.g. tables missing
        st.error(f"The data store is not ready ({ex}). Run this once in Terminal:  python -m src.reset_demo")
        st.stop()
if champ is None:
    st.error("No model is set up yet. Run this once in Terminal:  python -m src.reset_demo")
    st.stop()

# ------------------------------------------------------------------ sidebar
st.sidebar.markdown("<div class='brand'>Machine Health Monitor</div>", unsafe_allow_html=True)
st.sidebar.caption("Predicts failures 24 hours ahead. Simulated sample data.")
page = st.sidebar.radio("Pages", PAGES, key="page", label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.markdown(f"<div class='kv'><span>Model</span>{champ['version']}</div>", unsafe_allow_html=True)
st.sidebar.markdown(f"<div class='kv'><span>Data up to</span>{str(extent['last'])[:16]}</div>", unsafe_allow_html=True)
if not last_batch.empty:
    ds = last_batch.iloc[0]["drift_status"]
    _c = {"alert": "red", "watch": "amber"}.get(ds, "green")
    st.sidebar.markdown(f"<div class='kv'><span>Incoming data</span><i class='dot {_c}'></i>{DRIFT_TXT.get(ds, ds)}</div>", unsafe_allow_html=True)
if perf:
    _c = {"degraded": "red", "ok": "green"}.get(perf["status"], "grey")
    st.sidebar.markdown(f"<div class='kv'><span>Model accuracy</span><i class='dot {_c}'></i>{PERF_TXT.get(perf['status'], perf['status'])}</div>", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown("<div class='side-h'>Alert levels</div>", unsafe_allow_html=True)
hi = st.sidebar.slider("High risk above", 0.005, 0.99, float(round(champ["threshold_high"], 3)), 0.005,
                       key=f"hi_{champ['version']}")
med = st.sidebar.slider("Medium risk above", 0.0, float(hi), float(min(round(champ["threshold_medium"], 3), hi)), 0.005,
                        key=f"med_{champ['version']}")
st.sidebar.caption("Starting values come from testing on past data. A lower level gives more alerts.")
SCORE_NOTE = ("The score runs from 0 to 1. Higher means the machine looks more likely to fail soon. "
              "It ranks machines and is not an exact probability. High risk means a failure is expected within 24 hours.")
show_flash()


# ================================================================== 1. FLEET OVERVIEW
def page_overview():
    page_title("Overview", "Machines ranked by risk of failing in the next 24 hours.")
    with db() as con:
        risk = adapt.latest_machine_risk(con)
        f30 = con.execute("""SELECT COUNT(*) FROM fact_failure WHERE event_time > (SELECT MAX(reading_time) FROM fact_telemetry) - INTERVAL 30 DAY""").fetchall()[0][0]
        n_mach = con.execute("SELECT COUNT(*) FROM dim_machine").fetchall()[0][0]
    if risk.empty:
        st.info("No predictions yet. Go to **Add data** and load a batch.")
        return
    risk["band"] = adapt.risk_band(risk["score"], hi, med)
    k = st.columns(5)
    k[0].metric("Machines", n_mach)
    k[1].metric("High risk", int((risk["band"] == "high").sum()))
    k[2].metric("Medium risk", int((risk["band"] == "medium").sum()))
    k[3].metric("Failures in last 30 days", f30)
    k[4].metric("Last updated", str(risk["prediction_time"].iloc[0])[:16])
    st.caption(SCORE_NOTE)
    c1, c2 = st.columns([1, 2])
    counts = risk["band"].value_counts().reindex(["high", "medium", "low"], fill_value=0).reset_index()
    counts.columns = ["band", "machines"]
    c1.plotly_chart(px.bar(counts, x="band", y="machines", color="band", color_discrete_map=BAND_COLOURS,
                           title="How many machines at each risk level", labels={"band": "risk level", "machines": "number of machines"}), width="stretch")
    top = risk.head(15)
    c2.plotly_chart(px.bar(top, x=top["machine_id"].astype(str), y="score", color="band", color_discrete_map=BAND_COLOURS,
                           title="The 15 machines with the highest risk score", labels={"x": "machine number", "score": "risk score"}), width="stretch")
    st.subheader("Check these machines first")
    f1, f2 = st.columns(2)
    bands = f1.multiselect("Risk level", ["high", "medium", "low"], default=["high", "medium"])
    models = f2.multiselect("Machine type", sorted(risk["model"].unique()), default=sorted(risk["model"].unique()))
    view = risk[risk["band"].isin(bands) & risk["model"].isin(models)].copy()
    view.insert(0, "priority", range(1, len(view) + 1))
    st.dataframe(view[["priority", "machine_id", "model", "age", "score", "max_score_24h", "band", "model_version"]],
                 width="stretch", hide_index=True,
                 column_config={"score": st.column_config.ProgressColumn("score", min_value=0.0, max_value=1.0, format="%.3f"),
                                "max_score_24h": st.column_config.NumberColumn("max score, last 24h", format="%.3f")})
    st.caption("Machines are ranked by their latest score. This tool helps you decide, but a person should make the final call. "
               "It does not know repair costs or how much downtime each machine can afford.")


# ================================================================== 2. MACHINE DRILL-DOWN
def page_machine():
    page_title("Machines", "Sensor readings, risk score and history for one machine.")
    with db() as con:
        ids = [r[0] for r in con.execute("SELECT machine_id FROM dim_machine ORDER BY 1").fetchall()]
    c = st.columns([1, 1, 2])
    m = c[0].selectbox("Machine number", ids)
    days = c[1].slider("Days of history to show", 2, 60, 14)
    with db() as con:
        info = con.execute("SELECT model, age FROM dim_machine WHERE machine_id = ?", [m]).fetchall()[0]
        sens = W.q_machine_sensors(con, m, days * 24)
        fails = W.q_machine_failure_history(con, m)
        errs = W.q_machine_errors(con, m)
        mnt = W.q_machine_maintenance(con, m)
        pred = con.execute("""SELECT prediction_time, score, model_version FROM predictions WHERE machine_id = ?
                              AND prediction_time > (SELECT MAX(prediction_time) FROM predictions) - INTERVAL (?) HOUR
                              ORDER BY prediction_time""", [m, days * 24]).df()
    c[2].markdown(f"**Type:** {info[0]} · **Age:** {info[1]} years · **Past failures:** {len(fails)}")
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        subplot_titles=SENSORS + ["Risk score (dashed lines mark the alert levels)"])
    for i, s in enumerate(SENSORS, 1):
        fig.add_trace(go.Scatter(x=sens["reading_time"], y=sens[s], mode="lines", line=dict(width=1), name=s), row=i, col=1)
    if len(pred):
        fig.add_trace(go.Scatter(x=pred["prediction_time"], y=pred["score"], mode="lines", line=dict(color="#ee6666", width=1.5),
                                 name="risk score"), row=5, col=1)
        fig.add_hline(y=hi, line_dash="dash", line_color="#ee6666", row=5, col=1)
        fig.add_hline(y=med, line_dash="dash", line_color="#fac858", row=5, col=1)
    lo_t = sens["reading_time"].min() if len(sens) else None
    for t in fails["event_time"]:
        if lo_t is not None and t >= lo_t:
            fig.add_shape(type="line", xref="x", yref="paper", x0=t, x1=t, y0=0, y1=1, line=dict(color="black", dash="dot", width=1))
    fig.update_layout(height=900, showlegend=False, margin=dict(t=40))
    st.plotly_chart(fig, width="stretch")
    st.caption("Black dotted lines show when this machine actually failed. A score only appears for hours the model has already checked.")
    a, b, d = st.columns(3)
    a.subheader("Recent errors"); a.dataframe(errs, hide_index=True, width="stretch")
    b.subheader("Maintenance"); b.dataframe(mnt, hide_index=True, width="stretch")
    d.subheader("Past failures"); d.dataframe(fails, hide_index=True, width="stretch")


# ================================================================== 3. FAILURE & ERROR ANALYSIS
def page_failures():
    page_title("Failures and errors", "What failed, when, and how often.")
    lo, hi_d = extent["first"].date(), extent["last"].date()
    rng = st.date_input("Pick dates", (lo, hi_d), min_value=pd.Timestamp("2014-06-01").date(), max_value=hi_d)
    if not isinstance(rng, tuple) or len(rng) != 2:
        st.info("Please pick an end date too."); return
    d0, d1 = pd.Timestamp(rng[0]), pd.Timestamp(rng[1]) + pd.Timedelta(days=1)
    with db() as con:
        bc, bm, bmod, be, mt = (W.q_failures_by_component(con, d0, d1), W.q_failures_by_month(con, d0, d1),
                                W.q_failures_by_model(con, d0, d1), W.q_errors_by_type(con, d0, d1), W.q_mtbf(con, d0, d1))
    k = st.columns(3)
    k[0].metric("Failures in these dates", int(mt["failures"]))
    k[1].metric("Hours of machine running time", f"{int(mt['machine_hours']):,}")
    k[2].metric("⏱Average hours between failures", f"{mt['machine_hours'] / mt['failures']:.0f}" if mt["failures"] else "n/a")
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(bc, x="component", y="failures", title="Failures by part", labels={"component": "part", "failures": "number of failures"}), width="stretch")
    if len(bm):
        bm["month"] = bm["year"].astype(str) + "-" + bm["month"].astype(str).str.zfill(2)
        c2.plotly_chart(px.bar(bm, x="month", y="failures", color="component", title="Failures each month", labels={"month": "month", "failures": "number of failures", "component": "part"}), width="stretch")
    bmod["failures_per_machine"] = bmod["failures"] / bmod["machines_in_model"]
    c3, c4 = st.columns(2)
    c3.plotly_chart(px.bar(bmod, x="model", y="failures_per_machine", title="Failures per machine, by machine type", labels={"model": "machine type", "failures_per_machine": "failures per machine"}), width="stretch")
    c4.plotly_chart(px.bar(be, x="error_type", y="events", title="Number of errors, by error type", labels={"error_type": "error type", "events": "number of errors"}), width="stretch")
    st.caption("Part, error and machine-type names are codes in this dataset, so we make no guess about what they mean. "
               "The average time between failures is simply total running hours divided by number of failures.")


# ================================================================== 4. MODEL PERFORMANCE
def page_performance():
    page_title("Model accuracy", "How the model did on data it had not seen.")
    st.subheader("Test on past data (Nov–Dec 2015)")
    mc = REPORTS / "model_comparison.csv"
    if mc.exists():
        t = pd.read_csv(mc)
        cols = [c for c in ["model", "threshold_from_val", "test_pr_auc", "test_precision", "test_recall", "test_f1",
                            "test_accuracy", "test_FP", "test_FN"] if c in t.columns]
        nice = {"model": "Model", "threshold_from_val": "Alert level", "test_pr_auc": "PR-AUC", "test_precision": "Precision",
                "test_recall": "Recall", "test_f1": "F1", "test_accuracy": "Accuracy", "test_FP": "False alarms", "test_FN": "Missed"}
        st.dataframe(t[cols].rename(columns=nice), hide_index=True, width="stretch")
       
    imgs = [FIG / n for n in ("eval_pr_curves.png", "eval_confusion_matrix.png", "eval_feature_importance.png") if (FIG / n).exists()]
    if imgs:
        for col, p in zip(st.columns(len(imgs)), imgs):
            col.image(str(p), width="stretch")
    st.subheader("Results on new data (checked against what really happened)")
    with db() as con:
        d = adapt.matured_predictions(con)
        reg = con.execute("SELECT version FROM model_registry ORDER BY created_at").df()["version"].tolist()
    if d.empty:
        st.info("No results to check yet. Load a batch, then record what really happened on **Add data**."); return
    versions = st.multiselect("Model versions to include", sorted(d["model_version"].unique()), default=sorted(d["model_version"].unique()))
    d = d[d["model_version"].isin(versions)]
    thr = st.slider("Try a different alert level", 0.005, 0.99, float(hi), 0.005, key="whatif")
    m = adapt.metrics_at(d["actual_outcome"], d["score"], thr)
    k = st.columns(5)
    k[0].metric("Precision", f"{m['precision']:.3f}" if not np.isnan(m["precision"]) else "n/a")
    k[1].metric("Recall", f"{m['recall']:.3f}" if not np.isnan(m["recall"]) else "n/a")
    k[2].metric("F1", f"{m['f1']:.3f}" if not np.isnan(m["f1"]) else "n/a")
    k[3].metric("False alarms", m["FP"])
    k[4].metric("Missed failures", m["FN"])
    c1, c2 = st.columns(2)
    cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
    c1.plotly_chart(px.imshow(cm, text_auto=",", x=["No alert", "Alert"], y=["No failure", "Failure"], color_continuous_scale="Blues",
                              title="Alerts vs what really happened (counted in machine-hours)"), width="stretch")
    prec, rec, th = precision_recall_curve(d["actual_outcome"], d["score"])
    fig = go.Figure([go.Scatter(x=th, y=prec[:-1], name="Precision (alerts that were right)"), go.Scatter(x=th, y=rec[:-1], name="Recall (failures caught)")])
    fig.add_vline(x=thr, line_dash="dash")
    fig.update_layout(title="Fewer false alarms or fewer missed failures?", xaxis_title="alert level")
    c2.plotly_chart(fig, width="stretch")
    st.caption(f"Based on {len(d):,} machine-hours, of which {int(d['actual_outcome'].sum()):,} were followed by a failure. A higher alert level gives fewer false alarms "
               "but usually misses more failures. Use the slider to see the trade-off. Neighbouring hours are similar, so do not read too much into small differences.")



# ==================================================================
# 5. SYSTEM ARCHITECTURE
def page_architecture():
    page_title(
        "System architecture",
        "Visual overview of the predictive maintenance system, data warehouse, "
        "and adaptive learning process."
    )

    architecture_figures = [
        (
            "Figure 1. Overall system architecture",
            "figure_A_architecture.png",
            "Shows how machine sensor data, data processing, the DuckDB warehouse, "
            "machine-learning predictions, and the Streamlit dashboard connect."
        ),
        (
            "Figure 2. Data warehouse architecture",
            "figure_B_warehouse.png",
            "Shows how raw telemetry, error, maintenance, and failure records are "
            "processed and organised for analytics and model predictions."
        ),
        (
            "Figure 3. Adaptive learning workflow",
            "figure_C_adaptive.png",
            "Shows how new data is monitored for drift, predictions are evaluated "
            "against actual outcomes, and a replacement model is tested before promotion."
        ),
    ]

    
    
    for title, filename, description in architecture_figures:
        figure_path = FIG / filename

        st.subheader(title)
        st.caption(description)

        if figure_path.exists():
            if filename in (
                "figure_A_architecture.png",
                "figure_C_adaptive.png",
            ):
                # Smaller image, centred on the page
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    st.image(
                        str(figure_path),
                        width=250,
                        output_format="PNG"
                    )
            else:
                # Figure B keeps its larger size
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image(
                        str(figure_path),
                        width="stretch",
                        output_format="PNG"
                    )
        else:
            st.warning(
                f"Figure not found: {figure_path}. "
                "Save the diagram using this exact filename."
            )

        st.divider()



# ================================================================== 5. MONITORING & MODEL MANAGEMENT
def page_monitoring():
    page_title("Model updates", "Whether the model still works, and training a replacement.")
    with db() as con:
        dl = con.execute("SELECT * FROM drift_log ORDER BY checked_at").df()
        bl = con.execute("SELECT * FROM batch_log ORDER BY data_start").df()
        pb = adapt.performance_by_batch(con)
        reg = con.execute("SELECT version,status,algorithm,created_at,train_start,train_end,threshold_high,threshold_medium,val_pr_auc,val_f1 FROM model_registry ORDER BY created_at").df()
        ev = con.execute("SELECT * FROM model_events ORDER BY event_time DESC").df()
        prev = adapt.get_state(con, "previous_champion", "")
        chal = con.execute("SELECT * FROM model_registry WHERE status = 'challenger' ORDER BY created_at DESC LIMIT 1").df()
    a, b = st.columns(2)
    if not bl.empty:
        s = bl.iloc[-1]
        (a.error if s["drift_status"] == "alert" else a.warning if s["drift_status"] == "watch" else a.success)(
            f"**Is the incoming data changing?** Latest week: **{DRIFT_TXT.get(s['drift_status'], s['drift_status'])}** (change score {s['max_psi']:.2f}).")
    if perf:
        txt = {"degraded": f"**Is the model still accurate?** Its score (F1) is {perf.get('f1', float('nan')):.3f}, more than {C.PERF_F1_DROP} lower than the {perf['baseline_f1']:.3f} it got in testing.",
               "ok": f"**Is the model still accurate?** Yes. Its score (F1) is {perf.get('f1', float('nan')):.3f}, close to the {perf['baseline_f1']:.3f} it got in testing.",
               "insufficient_labels": "**Is the model still accurate?** Too early to say. We need more real outcomes first."}[perf["status"]]
        (b.error if perf["status"] == "degraded" else b.success if perf["status"] == "ok" else b.info)(txt)
    with st.expander("ℹWhat do these two checks mean?"):
        st.markdown(f"""
* **Is the incoming data changing?** Checks whether new sensor readings look different from the data the model learned from. We measure this with a "change score" (PSI): below {C.PSI_WATCH} is stable, {C.PSI_WATCH} to {C.PSI_ALERT} is a small change, above {C.PSI_ALERT} is a big change. This can warn us straight away, because we do not need to wait for failures to happen.
* **Is the model still accurate?** Checks whether the alerts are still right. We can only know this after the next 24 hours have passed and we know what really happened.
* A big change in the data does not always mean the model is wrong, so treat it as a reason to take a closer look.
* **Updating the model is controlled.** A new model replaces the current one only if it is clearly better on newer data (at least {C.PROMOTE_MARGIN} higher PR-AUC and F1 not worse).""")
    if not dl.empty:
        piv = dl.pivot_table(index="feature", columns="batch_id", values="psi", aggfunc="max")
        piv = piv[[c for c in bl["batch_id"] if c in piv.columns]]
        fig = go.Figure(go.Heatmap(z=np.minimum(piv.values, 1.0), x=[c[:8] for c in piv.columns], y=piv.index, text=piv.values.round(2),
                                   texttemplate="%{text}", colorscale=[[0, "#f4f7ff"], [1, "#5470c6"]], zmin=0, zmax=1, colorbar=dict(title="Change")))
        fig.update_layout(title="How much each reading changed, week by week (darker = bigger change)", height=380)
        st.plotly_chart(fig, width="stretch")
    if not pb.empty:
        long = pb.melt(id_vars=["week_start", "model_version"], value_vars=["precision", "recall", "f1"])
        fig = px.line(long, x="week_start", y="value", color="variable", markers=True, hover_data=["model_version"],
                      title="Model accuracy week by week (hover to see the model version)")
        fig.update_yaxes(range=[0, 1.02])
        st.plotly_chart(fig, width="stretch")
    st.subheader("All model versions")
    st.dataframe(reg, hide_index=True, width="stretch")
    st.subheader("Train a new model")
    st.write("Train a new model on recent data and compare it with the one in use. You decide whether to switch.")
    n_est = st.selectbox("Model size (number of trees)", [60, 200], index=1, help="60 is faster. 200 is the same size as the current model.")
    if st.button("Train a new model on recent data", type="primary", key="train_btn"):
        with st.spinner("Training the new model... this takes about 1 to 3 minutes."):
            with db() as con:
                res = adapt.train_challenger(con, n_estimators=n_est)
        flash("success" if res.get("ok") else "warning",
              f"New model {res['version']} is ready to compare." if res.get("ok") else f"Could not train a new model: {res.get('reason')}")
        st.rerun()
    if not chal.empty:
        row = chal.iloc[0]
        e = json.loads(row["evaluation_json"])
        st.markdown(f"**Newest model: `{row['version']}`**  (trained on data from {str(row['train_start'])[:10]} to {str(row['train_end'])[:10]})")
        cmp = pd.DataFrame({"measure": ["PR-AUC (higher is better)", "F1 score (higher is better)"],
                            "current model": [e["champion_pr_auc"], e["champion_f1"]], "new model": [e["challenger_pr_auc"], e["challenger_f1"]]})
        st.dataframe(cmp.round(4), hide_index=True, width="stretch")
        for k_, v_ in e["criteria"].items():
            st.markdown((":green[Pass]  " if v_ else ":red[Fail]  ") + k_.replace("_", " "))
        c1, c2 = st.columns(2)
        if c1.button("Use the new model", disabled=not e["passed"], key="promote_btn"):
            with db() as con:
                r = adapt.promote(con, row["version"])
            flash("success" if r["ok"] else "error", f"Now using {r.get('promoted')}. {r.get('replaced')} has been retired." if r["ok"] else r["reason"])
            st.rerun()
        if c2.button("Keep the current model", key="reject_btn"):
            with db() as con:
                adapt.reject(con, row["version"])
            flash("info", f"{row['version']} was not used. The current model stays in place."); st.rerun()
        if not e["passed"]:
            st.caption("You cannot switch yet because the new model did not meet the rules above.")
    if prev and st.button(f"Go back to the previous model ({prev})", key="rollback_btn"):
        with db() as con:
            r = adapt.rollback(con)
        flash("warning", f"Switched back to {r.get('restored')}. {r.get('withdrawn')} has been removed."); st.rerun()
    st.subheader("History of model changes")
    st.dataframe(ev, hide_index=True, width="stretch")
    st.subheader("Weekly data log")
    st.dataframe(bl, hide_index=True, width="stretch")


# ================================================================== 6. NEW DATA & FEEDBACK
def _next_batch():
    p = C.STREAM_DIR / "manifest.json"
    if not p.exists():
        return None, None
    man = json.loads(p.read_text())
    for b in man["batches"]:
        if pd.Timestamp(b["start"]) > pd.Timestamp(extent["last"]):
            return b, man
    return None, man


def _show_result(res: dict):
    for w in res.get("warnings", []):
        st.warning(w)
    for e in res.get("errors", []):
        st.error(e)


def page_data():
    page_title("Add data", "Load a new week of readings, then record what actually happened.")
    with db() as con:
        cnt = con.execute("SELECT outcome_status, COUNT(*) FROM predictions GROUP BY 1").df()
    st.caption(f"Data saved up to **{extent['last']}**. Real failures are known up to **{known_until}**. "
               "Predictions made: " + ", ".join(f"{r[0]} {r[1]:,}" for r in cnt.itertuples(index=False)) if len(cnt) else
               f"Data saved up to **{extent['last']}**.")

    st.subheader("1. Load the next week of sample data")
    nb, man = _next_batch()
    if nb is None:
        st.info("No more sample weeks to load. If this is unexpected, run `python -m src.make_stream` in Terminal.")
    else:
        st.write(f"Next week: **{nb['tag']}** ({nb['start'][:10]} to {nb['end'][:10]}). Contains {nb['rows']:,} readings, "
                 f"{nb['errors']} errors and {nb['maint']} maintenance visits.")
        if nb.get("synthetic_drift_injected"):
            st.warning(f"This week has an artificial shift added to the **{man['drift_sensor']}** readings (+{man['drift_offset']}) to test whether the app notices changing data.")
        use_spike = use_burst = False
        if nb.get("variants"):
            st.info("Optional demo cases for this week (made up for testing): "
                    f"a sensor spike on machine {nb['variants']['spike_machine']} and a burst of errors on machine {nb['variants']['error_burst_machine']}.")
            use_spike = st.checkbox("Include the sensor spike", key="spike")
            use_burst = st.checkbox("Include the burst of errors", key="burst")
        if st.button(f"Load {nb['tag']} and check for risks", type="primary", key="load_next"):
            tag = nb["tag"]
            raw = {"telemetry": pd.read_csv(C.STREAM_DIR / (f"{tag}_telemetry_SPIKE.csv" if use_spike else f"{tag}_telemetry.csv")),
                   "errors": pd.read_csv(C.STREAM_DIR / (f"{tag}_errors_BURST.csv" if use_burst else f"{tag}_errors.csv")),
                   "maint": pd.read_csv(C.STREAM_DIR / f"{tag}_maint.csv")}
            with st.spinner("Working on it..."), db() as con:
                res = adapt.process_batch(con, raw, source=f"stream:{tag}")
            if res["ok"]:
                flash("success", f"Done. {res['n_scored']:,} predictions made with model {res['model_version']}. "
                                 f"Data change: {DRIFT_TXT.get(res['drift_status'], res['drift_status'])} (score {res['max_psi']:.2f}). "
                                 f"{res['high_risk_now']} machine(s) are high risk now. We do not know what really happened yet, so do step 3 next.")
                st.rerun()
            _show_result(res)

    st.subheader("2. Or upload your own files")
    st.caption("Your files need these columns. Sensor readings: `datetime, machineID, volt, rotate, pressure, vibration`. "
               "Errors: `datetime, machineID, errorID`. Maintenance: `datetime, machineID, comp`.")
    u = st.columns(3)
    f_t, f_e, f_m = u[0].file_uploader("Sensor readings (CSV, required)", type="csv"), u[1].file_uploader("Errors (CSV, optional)", type="csv"), \
        u[2].file_uploader("Maintenance (CSV, optional)", type="csv")
    cc = st.columns(2)
    do_check, do_run = cc[0].button("Just check my files (saves nothing)", key="val_only"), cc[1].button("Check, save and score", type="primary", key="run_upload")
    if (do_check or do_run) and f_t is None:
        st.error("Please choose a sensor readings file first.")
    elif do_check or do_run:
        try:
            raw = {"telemetry": pd.read_csv(f_t), "errors": pd.read_csv(f_e) if f_e else None, "maint": pd.read_csv(f_m) if f_m else None}
        except Exception as ex:
            st.error(f"We could not read that file as a CSV: {ex}"); raw = None
        if raw:
            if do_check:
                with db() as con:
                    allw, alle = [], []
                    for k_, d_ in raw.items():
                        if d_ is not None:
                            _, e_, w_ = adapt.validate_upload(k_, d_, con); alle += e_; allw += w_
                _show_result({"errors": alle, "warnings": allw})
                if not alle:
                    st.success("Your files look good. Nothing was saved.")
            else:
                with st.spinner("Working on it..."), db() as con:
                    res = adapt.process_batch(con, raw, source="upload")
                if res["ok"]:
                    flash("success", f"Saved and scored {res['n_scored']:,} machine-hours. Data change: {DRIFT_TXT.get(res['drift_status'], res['drift_status'])}. "
                                     f"{res['high_risk_now']} machine(s) are high risk now.")
                    st.rerun()
                _show_result(res)

    st.subheader("3. Record what really happened")
    st.caption("A prediction can only be marked right or wrong after its 24 hours are over and you confirm the failure list is complete up to that time. "
               "After that, a machine with no failure listed counts as 'did not fail'.")
    default_known = pd.Timestamp(extent["last"])
    last_tag, last_end = None, None
    if man:
        for b in man["batches"]:
            if pd.Timestamp(b["start"]) <= pd.Timestamp(extent["last"]):
                last_tag, last_end = b["tag"], pd.Timestamp(b["end"])
    k_txt = st.text_input("My failure list is complete up to (YYYY-MM-DD HH:MM:SS). Used for uploaded files.",
                          value=str(default_known), key=f"known_txt_{default_known}")
    fo = st.file_uploader("What really happened (CSV with `datetime, machineID, failure`)", type="csv", key="outcomes_up")
    c1, c2 = st.columns(2)
    run_up = c1.button("Save my uploaded results", key="rec_up")
    run_stream = c2.button(f"Save the real results for the sample week I just loaded ({last_tag})", key="rec_stream", disabled=last_tag is None)
    if run_up or run_stream:
        try:
            ku = pd.Timestamp(k_txt) if run_up else last_end      # stream batch: complete up to the batch end
            raw_o = pd.read_csv(fo) if (run_up and fo) else (pd.read_csv(C.STREAM_DIR / f"{last_tag}_outcomes.csv") if run_stream else None)
            if raw_o is None:
                st.error("Please choose a results file first.")
            else:
                with db() as con:
                    res = adapt.record_outcomes(con, raw_o, ku)
                if res["ok"]:
                    flash("success", f"Saved {res['failures_added']} failure(s). {res['matured']:,} prediction(s) can now be checked. "
                                     "See the results on the Model accuracy and Model updates pages.")
                    st.rerun()
                _show_result(res)
        except Exception as ex:
            st.error(f"We could not use that results file: {ex}")

{
    "Overview": page_overview,
    "Machines": page_machine,
    "Failures and errors": page_failures,
    "Model accuracy": page_performance,
    "System architecture": page_architecture,
    "Model updates": page_monitoring,
    "Add data": page_data,
}[page]()