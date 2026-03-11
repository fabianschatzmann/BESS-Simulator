# 9_Debug_Engineering_View.py
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 25 19:50:37 2025

@author: fabia
"""

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Debug View", layout="wide")
st.title("Debug / Engineering View")

if "results" not in st.session_state:
    st.warning("Keine Ergebnisse vorhanden. Bitte zuerst auf der Startseite **Run** ausführen.")
    st.stop()

res = st.session_state["results"]
df = res["timeseries"].copy()

st.subheader("Constraint diagnostics (placeholder)")
st.write("Diese Seite ist dafür gedacht, später Solver-Logs, Infeasibilities, SOC-Heatmaps, Throughput etc. zu sehen.")

st.markdown("---")
st.subheader("SOC distribution")
st.histogram = st.bar_chart(df["soc"].value_counts(bins=20).sort_index())

st.markdown("---")
st.subheader("Raw timeseries (first 200 rows)")
st.dataframe(df.head(200), use_container_width=True)
