"""Settings for Phases 9-11 (warehouse, adaptive mechanism, dashboard).
Edit values HERE only. Paths come from src/config.py so nothing is hard-coded twice."""
import pandas as pd

from src.config import PROJECT

# ---------------- storage ----------------
DB_PATH = PROJECT / "data" / "warehouse" / "pdm.duckdb"
STREAM_DIR = PROJECT / "data" / "stream"
HORIZON_HOURS = 24

# The warehouse is initially loaded with history BEFORE this moment. Everything
# after it is held back as a simulated stream of weekly batches. (It equals the
# start of the validation period, so the v1 champion never saw this data.)
LIVE_CUTOFF = pd.Timestamp("2015-09-01 00:00:00")
BATCH_HOURS = 168                      # one batch = one week of hourly readings
HISTORY_HOURS = 168                    # history re-read from the warehouse when scoring a batch

# ---------------- monitoring rules (documented, tunable) ----------------
PSI_WATCH = 0.10                       # rule of thumb: <0.10 stable, 0.10-0.25 watch, >0.25 alert
PSI_ALERT = 0.25
PERF_WINDOW_DAYS = 14                  # performance is judged on matured predictions of the last N days
PERF_MIN_POSITIVES = 150               # need this many positive machine-hours before judging
PERF_F1_DROP = 0.05                    # alert if F1 falls more than this below the champion's validation F1

# ---------------- retraining rules ----------------
MIN_NEW_EVENTS = 30                    # new failure events since champion's training end
VAL_DAYS = 14                          # newest labelled days kept aside to compare champion vs challenger
TRAIN_WINDOW_DAYS = 45                 # challenger trains on the most recent N days before the validation block
PROMOTE_MARGIN = 0.005                 # challenger PR-AUC must beat champion by at least this
MEDIUM_NEG_QUANTILE = 0.99             # 'medium' band starts at the 99th percentile of validation scores on non-failure hours

# ---------------- simulated sensor drift (clearly labelled as synthetic) ----------------
DRIFT_SENSOR = "volt"
DRIFT_OFFSET = 20.0                    # added to every volt reading from DRIFT_START_BATCH onwards
DRIFT_START_BATCH = 5

# ---------------- scripted experiment (src/simulate.py) ----------------
SIM_RETRAIN_AFTER = (4, 12)            # try a retrain after these batch numbers (4 = stationary, 12 = after drift)
