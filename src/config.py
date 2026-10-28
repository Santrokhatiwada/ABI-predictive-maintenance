"""Shared settings. Change horizon / split dates HERE only."""
from pathlib import Path
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent

CLEAN = PROJECT / "data" / "clean"
FEATURES = PROJECT / "data" / "features"
MODELS = PROJECT / "models"
REPORTS = PROJECT / "reports"
FIG = REPORTS / "figures"

# CLEAN = Path("/Users/santrokhatiwada/Documents/MAT/SEM 1/Data Analytics/Assignment 2/abi_pdm/data/clean")
# FEATURES = Path("/Users/santrokhatiwada/Documents/MAT/SEM 1/Data Analytics/Assignment 2/abi_pdm/data/features")
# MODELS = Path("/Users/santrokhatiwada/Documents/MAT/SEM 1/Data Analytics/Assignment 2/abi_pdm/models")
# REPORTS = Path("/Users/santrokhatiwada/Documents/MAT/SEM 1/Data Analytics/Assignment 2/abi_pdm/reports")
# FIG = REPORTS / "figures"

SENSORS = ["volt", "rotate", "pressure", "vibration"]
HORIZON_HOURS = 24                      # predict: failure within the next 24 hours
ROLL_SENSOR_WINDOWS = (3, 24)           # hours
ERROR_WINDOWS = (24, 168)               # hours (1 day, 7 days)

# Chronological split. 'purged' rows sit in the 24h gap before each later
# period, because their labels look into that later period.
VAL_START = pd.Timestamp("2015-09-01 00:00:00")
TEST_START = pd.Timestamp("2015-11-01 00:00:00")

SEED = 42
VERSION = "v1"


def assign_split(ts: pd.Series) -> pd.Series:
    gap = pd.Timedelta(hours=HORIZON_HOURS)
    out = pd.Series("purged", index=ts.index, dtype="object")
    out[ts < VAL_START - gap] = "train"
    out[(ts >= VAL_START) & (ts < TEST_START - gap)] = "val"
    out[ts >= TEST_START] = "test"
    return out

    