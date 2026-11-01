"""
pages/1_Dashboard.py
---------------------
Home Dashboard — KPI cards, top charts, and dataset summary.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.helpers import (
    init_session_state, get_df, page_header, metric_card,
    no_data_warning, get_data_path, load_sample_data_cached,
    set_df, ensure_src_on_path,
)

ensure_src_on_path()

st.set_page_config(page_title="Dashboard", page_icon="🏠", layout="wide")
init_session_state()

page_header("Crime Dashboard")

df = get_df()

# --- Auto-load sample data if nothing is loaded yet ---
if df is None:
    sample_path = get_data_path("sample_crime_data.csv")
    if os.path.exists(sample_path):
        with st.spinner("Loading sample dataset…"):
            df, warnings = load_sample_data_cached(sample_path)
            st.session_state["df_processed"] = df
            st.session_state["df_featured"] = df
            st.session_state["data_loaded"] = True
            if warnings:
                st.session_state["warnings"] = warnings
    else:
        no_data_warning()
        st.stop()

# ---------------------------------------------------------------------------
# KPI Cards — Row 1
# ---------------------------------------------------------------------------
st.markdown("#### 📊 Key Performance Indicators")
c1, c2, c3, c4, c5, c6 = st.columns(6)

risk_col = "predicted_label" if "predicted_label" in df.columns else None
if "risk_label" in df.columns and risk_col is None:
    lmap = {2: "High Risk", 1: "Medium Risk", 0: "Low Risk"}
    df["_risk_text"] = df["risk_label"].map(lmap)
    risk_col = "_risk_text"

high_count   = (df[risk_col] == "High Risk").sum()   if risk_col else 0
medium_count = (df[risk_col] == "Medium Risk").sum() if risk_col else 0
low_count    = (df[risk_col] == "Low Risk").sum()    if risk_col else 0

total_records = len(df)
pct_high = f"{high_count/total_records*100:.1f}% of records" if total_records > 0 else "0%"
pct_med  = f"{medium_count/total_records*100:.1f}% of records" if total_records > 0 else "0%"
pct_low  = f"{low_count/total_records*100:.1f}% of records" if total_records > 0 else "0%"

with c1:
    metric_card("Total Records", f"{total_records:,}", "crime incidents", "#1565c0")
with c2:
    metric_card("Crime Categories", df["crime_type"].nunique() if total_records > 0 else 0, "distinct types", "#1976d2")
with c3:
    metric_card("🔴 High Risk", f"{high_count:,}", pct_high, "#c62828")
with c4:
    metric_card("🟠 Medium Risk", f"{medium_count:,}", pct_med, "#e65100")
with c5:
    metric_card("🟢 Low Risk", f"{low_count:,}", pct_low, "#2e7d32")
with c6:
    if total_records > 0 and pd.notna(df['date'].min()) and pd.notna(df['date'].max()):
        date_range = f"{df['date'].min().strftime('%b %Y')} – {df['date'].max().strftime('%b %Y')}"
    else:
        date_range = "N/A"
    metric_card("Date Range", date_range, "coverage period", "#37474f")

st.markdown("---")

# ---------------------------------------------------------------------------
# Charts — Row 1
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("##### 📦 Crime by Category")
    cat_counts = df["crime_type"].value_counts().reset_index()
    cat_counts.columns = ["Crime Type", "Count"]
    fig = px.bar(
        cat_counts, x="Count", y="Crime Type", orientation="h",
        color="Count", color_continuous_scale="Blues",
        height=380,
    )
    fig.update_layout(showlegend=False, plot_bgcolor="white",
                      margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.markdown("##### 📈 Monthly Crime Trend")
    try:
        trend = df.set_index("date").resample("ME").size().reset_index()
    except Exception:
        trend = df.set_index("date").resample("M").size().reset_index()
    trend.columns = ["date", "count"]
    fig2 = px.area(
        trend, x="date", y="count", height=380,
        color_discrete_sequence=["#1565c0"],
    )
    fig2.update_traces(line=dict(width=2.5), fillcolor="rgba(21,101,192,0.15)")
    fig2.update_layout(plot_bgcolor="white",
                       margin=dict(l=0, r=0, t=10, b=0),
                       xaxis_title="", yaxis_title="Incidents")
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# Charts — Row 2
# ---------------------------------------------------------------------------
col_a, col_b, col_c = st.columns([1, 1, 1])

with col_a:
    st.markdown("##### 🎯 Risk Distribution")
    if risk_col:
        risk_counts = df[risk_col].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]
        fig3 = px.pie(
            risk_counts, names="Risk Level", values="Count",
            hole=0.45, height=320,
            color="Risk Level",
            color_discrete_map={"High Risk": "#d32f2f",
                                "Medium Risk": "#f57c00",
                                "Low Risk": "#388e3c"},
        )
        fig3.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Train the model to see risk distribution.")

with col_b:
    st.markdown("##### ⏰ Crime by Hour of Day")
    if "hour" in df.columns:
        hourly = df.groupby("hour").size().reset_index(name="count")
        fig4 = px.bar(hourly, x="hour", y="count", height=320,
                      color="count", color_continuous_scale="OrRd")
        fig4.update_layout(showlegend=False, plot_bgcolor="white",
                           margin=dict(l=0, r=0, t=10, b=0),
                           xaxis_title="Hour", yaxis_title="Incidents")
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("No time column detected.")

with col_c:
    st.markdown("##### 📅 Crime by Day of Week")
    if "day_of_week" in df.columns:
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        dow = df.groupby("day_of_week").size().reset_index(name="count")
        dow["day_name"] = dow["day_of_week"].apply(lambda x: days[int(x)] if int(x) < 7 else "?")
        fig5 = px.bar(dow, x="day_name", y="count", height=320,
                      color="count", color_continuous_scale="Blues")
        fig5.update_layout(showlegend=False, plot_bgcolor="white",
                           margin=dict(l=0, r=0, t=10, b=0),
                           xaxis_title="Day", yaxis_title="Incidents")
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info("Date column required for weekday analysis.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Top 10 hotspot areas
# ---------------------------------------------------------------------------
st.markdown("#### 🔥 Top 10 Crime Hotspot Locations")
if "location" in df.columns and df["location"].notna().any():
    top_locs = (
        df.groupby("location")
        .agg(crime_count=("crime_type", "count"),
             top_crime=("crime_type", lambda x: x.value_counts().idxmax() if len(x) else "N/A"))
        .reset_index()
        .sort_values("crime_count", ascending=False)
        .head(10)
    )
    top_locs.index = range(1, len(top_locs) + 1)
    st.dataframe(top_locs, use_container_width=True)
else:
    top_coords = (
        df.assign(lat_r=df["latitude"].round(2), lon_r=df["longitude"].round(2))
        .groupby(["lat_r", "lon_r"])
        .agg(crime_count=("crime_type", "count"),
             top_crime=("crime_type", lambda x: x.value_counts().idxmax() if len(x) else "N/A"))
        .reset_index()
        .sort_values("crime_count", ascending=False)
        .head(10)
        .rename(columns={"lat_r": "Latitude", "lon_r": "Longitude"})
    )
    top_coords.index = range(1, len(top_coords) + 1)
    st.dataframe(top_coords, use_container_width=True)

st.caption("⚠️ Data shown is SYNTHETIC and for academic demonstration only.")
