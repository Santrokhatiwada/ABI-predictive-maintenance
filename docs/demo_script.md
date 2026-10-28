# 15-minute demo script (rehearse it; keep a screen recording as a backup)

Before you start: stop any running dashboard, run `python -m src.reset_demo`, then
`streamlit run app/streamlit_app.py`. Each "Load next batch" + "Record outcomes" pair takes about 5 seconds,
so you can click through weeks 1-4 before the audience arrives (stationary data, drift stays green).

| Min | Page | What you do | What you say |
|---|---|---|---|
| 0-2 | slides | Problem, dataset, A1 link | simulated data, 100 machines, hourly sensors |
| 2-4 | slides | Architecture diagram, star schema | warehouse vs raw CSV vs ML feature table |
| 4-5 | 1 Overview | KPIs, priority list, move the threshold slider | ranking score is not a calibrated probability |
| 5-6 | 2 Drill-down | Pick the top machine | sensors, errors, maintenance, score history |
| 6-8 | 6 New data | Load batch 2 with BOTH variants ticked | scenario 2 (sensor spike -> medium) and 3 (error burst -> high), no retraining, immediate |
| 8-9 | 6 New data | Upload a broken CSV (drop a column), then "Validate only" | invalid files are rejected with a clear message, nothing stored |
| 9-10 | 6 + 4 | Record outcomes | delayed labels: last 24h stay pending; live precision/recall now measurable |
| 10-12 | 5 Monitoring | Load a drifted batch (batch 5+), show PSI heatmap | drift needs no labels; degradation needs labels; they are different |
| 12-14 | 5 Mgmt | Train challenger, show criteria, promote, show drift clears, show rollback button | gated retraining, audit trail, rollback |
| 14-15 | slides | Limits and conclusion | synthetic data and synthetic drift; no cost or downtime claims |

Reaching the drifted state live: keep clicking Load + Record outcomes. From week 5 the synthetic volt offset
starts (drift = alert at once; "performance degraded" appears after outcomes arrive). By week 12 there is
enough data for a good challenger. Use the 60-tree option on page 5 to keep training short.
`python -m src.simulate --fast` reproduces the whole story offline and writes the report tables in reports/.

Fallback if anything fails: screen recording, then the saved CSVs in reports/ and the figures in reports/figures/.
