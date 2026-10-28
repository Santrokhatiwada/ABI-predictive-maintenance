import platform
import sys
import pandas
import numpy
import sklearn
import matplotlib
import plotly
import streamlit
import duckdb
import joblib

print("Python  :", sys.version.split()[0])
print("Machine :", platform.machine())

for m in (pandas, numpy, sklearn, matplotlib, plotly, streamlit, duckdb, joblib):
    print(f"{m.__name__:12}", m.__version__)

print("DuckDB test:", duckdb.sql("SELECT 42 AS answer").fetchall())
