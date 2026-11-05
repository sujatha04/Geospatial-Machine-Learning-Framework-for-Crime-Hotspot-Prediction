"""
pages/2_Crime_Analysis.py
--------------------------
Dataset Ingestion, Spatio-Temporal Filtering, DBSCAN Clustering & Analytics.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import io

from utils.helpers import (
    init_session_state, page_header, metric_card, no_data_warning,
    get_data_path, load_sample_data_cached, process_uploaded_data,
    df_to_csv_bytes, ensure_src_on_path, apply_custom_css,
)

ensure_src_on_path()

from src.data_preprocessing import compute_risk_label
from src.geospatial_analysis import run_dbscan, get_cluster_summary, compute_grid_density
from src.visualization import (
    plot_crime_by_category, plot_crime_trend, plot_crime_by_hour,
    plot_risk_distribution, plot_monthly_heatmap,
)

st.set_page_config(page_title="Crime Analysis & Clustering | GeoCrime AI", page_icon="📊", layout="wide")
init_session_state()
apply_custom_css()

page_header(
    title="Crime Analysis & Spatial Clustering",
    subtitle="Ingest raw CSV records, run DBSCAN density clustering, explore spatio-temporal trends, and export clean datasets.",
    badge="Data & Analytics"
)

# ---------------------------------------------------------------------------
# Data Loading Section
# ---------------------------------------------------------------------------
st.markdown("### 📂 Dataset Ingestion Hub")
load_tab, upload_tab = st.tabs(["📥 Built-in Synthetic Dataset", "⬆️ Upload Custom CSV Dataset"])

with load_tab:
    st.markdown("""
    The system includes a pre-packaged **synthetic benchmark dataset** containing 2,000 geolocated crime incidents
    with realistic urban spatial clusters, temporal distributions, and multi-class risk labels.
    """)
    if st.button("🚀 Load / Reset Benchmark Dataset", type="primary"):
        sample_path = get_data_path("sample_crime_data.csv")
        if not os.path.exists(sample_path):
            with st.spinner("Generating sample dataset…"):
                gen_script = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "generate_sample_data.py"
                )
                import subprocess
                result = subprocess.run(
                    [sys.executable, gen_script], capture_output=True, text=True
                )
                if result.returncode != 0:
                    st.error(f"Failed to generate sample data:\n{result.stderr}")
                    st.stop()
        with st.spinner("Parsing coordinates and computing spatio-temporal features…"):
            df, warnings = load_sample_data_cached(sample_path)
        st.session_state["df_processed"] = df
        st.session_state["df_featured"] = df
        st.session_state["data_loaded"] = True
        st.session_state["warnings"] = warnings
        st.success(f"✅ Successfully loaded {len(df):,} synthetic spatial records.")
        if warnings:
            for w in warnings:
                st.warning(w)

with upload_tab:
    st.markdown("""
    Upload your custom CSV file containing incident coordinates. Minimum expected schema:
    
    * `latitude`: Float decimal degrees (e.g. `40.7128`)
    * `longitude`: Float decimal degrees (e.g. `-74.0060`)
    * `date`: ISO timestamp or YYYY-MM-DD
    * `crime_type`: Categorical label (e.g. `Theft`, `Burglary`, `Assault`)
    """)
    uploaded = st.file_uploader("Upload CSV file for analysis", type=["csv"])
    if uploaded:
        with st.spinner("Validating coordinate ranges and engineering features…"):
            try:
                content = uploaded.read()
                df, warnings = process_uploaded_data(content)
                st.session_state["df_processed"] = df
                st.session_state["df_featured"] = df
                st.session_state["data_loaded"] = True
                st.session_state["warnings"] = warnings
                st.success(f"✅ Ingested {len(df):,} records successfully!")
                if warnings:
                    for w in warnings:
                        st.warning(w)
            except ValueError as e:
                st.error(f"❌ Validation Error: {e}")

# ---------------------------------------------------------------------------
# Dataset preview
# ---------------------------------------------------------------------------
df = st.session_state.get("df_processed")
if df is None:
    no_data_warning()
    st.stop()

st.write("")

# ---------------------------------------------------------------------------
# KPI Metrics
# ---------------------------------------------------------------------------
st.markdown("### 🗂 Dataset Schema & Telemetry")

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Total Observations", f"{len(df):,}", "Verified records", "#2563eb", icon="📑")
with c2:
    metric_card("Feature Columns", len(df.columns), "Spatial & temporal engineered", "#0284c7", icon="📐")
with c3:
    metric_card("Crime Classes", df["crime_type"].nunique(), "Distinct categories", "#7c3aed", icon="🏷️")
with c4:
    metric_card("Missing Values", int(df.isna().sum().sum()), "Zero critical nulls", "#059669" if df.isna().sum().sum() == 0 else "#dc2626", icon="✨")

st.write("")

# Filters
st.markdown("#### 🔍 Interactive Query & Table Filter")
col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
with col_f1:
    crime_types = ["All"] + sorted(df["crime_type"].unique().tolist())
    selected_type = st.selectbox("Filter Crime Modality", crime_types, key="da_crime_filter")
with col_f2:
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    date_range = st.date_input(
        "Date Range Filter",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="da_date_range",
    )
with col_f3:
    max_rows = st.number_input("Max Preview Rows", 10, 500, 100, step=25, key="da_max_rows")

# Apply filters
display_df = df.copy()
if selected_type != "All":
    display_df = display_df[display_df["crime_type"] == selected_type]
if len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    display_df = display_df[(display_df["date"] >= start) & (display_df["date"] <= end)]

st.dataframe(display_df.head(int(max_rows)), use_container_width=True, height=300)

# Download processed data
csv_bytes = df_to_csv_bytes(display_df)
st.download_button(
    "⬇️ Export Filtered Dataset (CSV)",
    data=csv_bytes,
    file_name="geocrime_filtered_dataset.csv",
    mime="text/csv",
)

st.write("")

# ---------------------------------------------------------------------------
# Geospatial Clustering
# ---------------------------------------------------------------------------
st.markdown("### 🌍 DBSCAN Spatial Density Clustering")
st.markdown(
    "DBSCAN detects continuous spatial density regions and groups nearby incidents while segregating sparse outliers as noise."
)

with st.expander("⚙️ DBSCAN Hyperparameter Tuning", expanded=False):
    col_eps, col_min = st.columns(2)
    with col_eps:
        eps_km = st.slider("Epsilon Radius (km)", 0.1, 5.0, 0.5, 0.1, help="Maximum spherical distance between two points to be considered neighbors.")
    with col_min:
        min_samples = st.slider("Min Samples per Cluster", 2, 20, 5, help="Minimum number of incidents required to form a dense core cluster.")

if st.button("⚡ Execute DBSCAN Spatial Clustering", type="primary"):
    with st.spinner("Calculating Haversine distance matrix and cluster centroids…"):
        df_clustered = run_dbscan(df, eps_km=eps_km, min_samples=min_samples)
        cluster_summary = get_cluster_summary(df_clustered)
        grid_density = compute_grid_density(df)
        st.session_state["df_clustered"] = df_clustered
        st.session_state["cluster_summary"] = cluster_summary
        st.session_state["grid_density"] = grid_density

    n_clusters = cluster_summary["cluster_id"].nunique() if len(cluster_summary) > 0 else 0
    st.success(f"✅ Identified {n_clusters} dense spatial clusters with epsilon = {eps_km} km.")

cluster_summary = st.session_state.get("cluster_summary")
if cluster_summary is not None and len(cluster_summary) > 0:
    st.markdown("#### 🔵 Cluster Centroids & Severity Breakdown")

    cr1, cr2, cr3 = st.columns(3)
    with cr1:
        metric_card("Identified Clusters", len(cluster_summary), "Dense epicenters", "#2563eb", icon="🎯")
    with cr2:
        high_clusters = (cluster_summary["risk_level"].isin(["High", "High Risk"])).sum()
        metric_card("High-Risk Hotspots", high_clusters, "Critical clusters", "#dc2626", icon="🚨")
    with cr3:
        clustered_crimes = cluster_summary["crime_count"].sum()
        metric_card("Clustered Incidents", f"{clustered_crimes:,}", f"{(clustered_crimes/len(df)*100):.1f}% of total crimes", "#059669", icon="📍")

    display_cols = ["cluster_id", "crime_count", "lat_center", "lon_center",
                    "top_crime", "risk_level"]
    display_cols = [c for c in display_cols if c in cluster_summary.columns]
    st.dataframe(cluster_summary[display_cols].head(20), use_container_width=True)

    import plotly.express as px
    fig_cl = px.scatter(
        cluster_summary,
        x="lon_center", y="lat_center",
        size="crime_count", color="risk_level",
        color_discrete_map={"High": "#dc2626", "High Risk": "#dc2626", "Medium": "#d97706", "Medium Risk": "#d97706", "Low": "#16a34a", "Low Risk": "#16a34a"},
        hover_data=["crime_count", "top_crime"],
        title="Spatial Coordinates of DBSCAN Cluster Centroids",
        labels={"lon_center": "Longitude", "lat_center": "Latitude"},
        size_max=36, height=400,
        template="plotly_white",
    )
    fig_cl.update_layout(
        plot_bgcolor="rgba(248,250,252,0.6)",
        paper_bgcolor="white",
        xaxis=dict(gridcolor="#e2e8f0"),
        yaxis=dict(gridcolor="#e2e8f0")
    )
    st.plotly_chart(fig_cl, use_container_width=True)

# ---------------------------------------------------------------------------
# Analytics Charts
# ---------------------------------------------------------------------------
st.write("")
st.markdown("### 📈 Comprehensive Visual Analytics Suite")

t1, t2, t3, t4, t5 = st.tabs(
    ["📊 Category Distribution", "📉 Temporal Trend", "⏰ Diurnal Hourly Curve",
     "🎯 Risk Proportions", "🗓️ Calendar Seasonality Heatmap"]
)

with t1:
    st.plotly_chart(plot_crime_by_category(df), use_container_width=True)

with t2:
    freq = st.radio("Temporal Granularity", ["Monthly", "Yearly"], horizontal=True, key="trend_freq")
    code = "ME" if freq == "Monthly" else "YE"
    st.plotly_chart(plot_crime_trend(df, freq=code), use_container_width=True)

with t3:
    if "hour" in df.columns:
        st.plotly_chart(plot_crime_by_hour(df), use_container_width=True)
    else:
        st.info("No 'hour' column found in current dataset.")

with t4:
    st.plotly_chart(plot_risk_distribution(df), use_container_width=True)

with t5:
    st.plotly_chart(plot_monthly_heatmap(df), use_container_width=True)

st.write("")
st.caption("⚠️ All records and analyses are synthetic — generated for academic machine learning research.")
