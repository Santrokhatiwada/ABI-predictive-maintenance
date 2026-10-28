# Phases 9-11 package

Unzip into the ROOT of your abi_pdm project (folders src/, app/, tests/, docs/ merge with yours).
It adds files only; it does not replace your config.py, features.py or any Phase 3-8 file.

New files
- src/abi_config.py   settings (cutoff date, thresholds, drift, retraining rules)
- src/warehouse.py    DuckDB star schema + BI queries         (Phase 9)
- src/adapt.py        adaptive mechanism                       (Phase 10)
- src/make_stream.py  weekly "new data" files + demo variants  (Phase 10)
- src/reset_demo.py   one-command reset                        (Phase 10)
- src/simulate.py     reproducible adaptation experiment       (Phase 10/13)
- app/streamlit_app.py  the dashboard                          (Phase 11)
- tests/test_warehouse.py, tests/test_adapt.py
- docs/diagrams.md, docs/demo_script.md
