"""
pages/2_Crime_Analysis.py
--------------------------
Dataset Management + Geospatial Analysis + Analytics Charts.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import io

from utils.helpers import (
    init_session_state, page_header, metric_card, no_data_warning,
    get_data_path, load_sample_data_cached, process_uploaded_data,
    df_to_csv_bytes, ensure_src_on_path,
)

ensure_src_on_path()

from src.data_preprocessing import compute_risk_label
from src.geospatial_analysis import run_dbscan, get_cluster_summary, compute_grid_density
from src.visualization import (
    plot_crime_by_category, plot_crime_trend, plot_crime_by_hour,
    plot_risk_distribution, plot_monthly_heatmap,
)

st.set_page_config(page_title="Crime Analysis", page_icon="📊", layout="wide")
init_session_state()

page_header("Crime Dataset Management")

# ---------------------------------------------------------------------------
# Data Loading Section
# ---------------------------------------------------------------------------
st.markdown("### 📂 Data Source")
load_tab, upload_tab = st.tabs(["📥 Load Sample Dataset", "⬆️ Upload Your CSV"])

with load_tab:
    st.markdown("""
    The built-in **synthetic** sample dataset contains 2,000 computer-generated crime
    records across a fictional urban grid. It is provided solely for demonstration and testing.
    """)
    if st.button("Load Sample Dataset", type="primary"):
        sample_path = get_data_path("sample_crime_data.csv")
        if not os.path.exists(sample_path):
            # Generate it on first run
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
        with st.spinner("Loading and processing…"):
            df, warnings = load_sample_data_cached(sample_path)
        st.session_state["df_processed"] = df
        st.session_state["df_featured"] = df
        st.session_state["data_loaded"] = True
        st.session_state["warnings"] = warnings
        st.success(f"✅ Loaded {len(df):,} synthetic records.")
        if warnings:
            for w in warnings:
                st.warning(w)

with upload_tab:
    st.markdown("""
    Upload a CSV file with **at minimum** these columns:

    | Column | Example |
    |--------|---------|
    | `latitude` | 40.7128 |
    | `longitude` | -74.0060 |
    | `date` | 2023-05-12 |
    | `crime_type` | Theft |

    Optional columns: `time`, `location`, `description`
    """)
    uploaded = st.file_uploader("Choose CSV file", type=["csv"])
    if uploaded:
        with st.spinner("Validating and processing your dataset…"):
            try:
                content = uploaded.read()
                df, warnings = process_uploaded_data(content)
                st.session_state["df_processed"] = df
                st.session_state["df_featured"] = df
                st.session_state["data_loaded"] = True
                st.session_state["warnings"] = warnings
                st.success(f"✅ Uploaded and processed {len(df):,} records.")
                if warnings:
                    for w in warnings:
                        st.warning(w)
            except ValueError as e:
                st.error(f"❌ Validation failed: {e}")

# ---------------------------------------------------------------------------
# Dataset preview
# ---------------------------------------------------------------------------
df = st.session_state.get("df_processed")
if df is None:
    no_data_warning()
    st.stop()

st.markdown("---")
st.markdown("### 🗂 Dataset Preview")

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Total Records", f"{len(df):,}", "", "#1565c0")
with c2:
    metric_card("Columns", len(df.columns), "", "#1976d2")
with c3:
    metric_card("Crime Types", df["crime_type"].nunique(), "", "#0288d1")
with c4:
    metric_card("Missing Values", int(df.isna().sum().sum()), "", "#37474f")

# Filters
col_f1, col_f2, col_f3 = st.columns([2, 2, 1])
with col_f1:
    crime_types = ["All"] + sorted(df["crime_type"].unique().tolist())
    selected_type = st.selectbox("Filter by Crime Type", crime_types, key="da_crime_filter")
with col_f2:
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    date_range = st.date_input(
        "Filter by Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="da_date_range",
    )
with col_f3:
    max_rows = st.number_input("Rows to show", 10, 500, 100, step=10, key="da_max_rows")

# Apply filters
display_df = df.copy()
if selected_type != "All":
    display_df = display_df[display_df["crime_type"] == selected_type]
if len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    display_df = display_df[(display_df["date"] >= start) & (display_df["date"] <= end)]

st.dataframe(display_df.head(int(max_rows)), use_container_width=True, height=320)

# Download processed data
csv_bytes = df_to_csv_bytes(display_df)
st.download_button(
    "⬇️ Download Filtered Data (CSV)",
    data=csv_bytes,
    file_name="filtered_crime_data.csv",
    mime="text/csv",
)

# ---------------------------------------------------------------------------
# Geospatial Clustering
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 🌍 Geospatial Clustering Analysis")

with st.expander("⚙️ DBSCAN Clustering Parameters", expanded=False):
    col_eps, col_min = st.columns(2)
    with col_eps:
        eps_km = st.slider("Neighbourhood Radius (km)", 0.1, 5.0, 0.5, 0.1)
    with col_min:
        min_samples = st.slider("Min Samples per Cluster", 2, 20, 5)

if st.button("Run Geospatial Clustering", type="primary"):
    with st.spinner("Running DBSCAN clustering…"):
        from src.geospatial_analysis import run_dbscan, get_cluster_summary, compute_grid_density
        df_clustered = run_dbscan(df, eps_km=eps_km, min_samples=min_samples)
        cluster_summary = get_cluster_summary(df_clustered)
        grid_density = compute_grid_density(df)
        st.session_state["df_clustered"] = df_clustered
        st.session_state["cluster_summary"] = cluster_summary
        st.session_state["grid_density"] = grid_density

    n_clusters = cluster_summary["cluster_id"].nunique() if len(cluster_summary) > 0 else 0
    st.success(f"✅ Found {n_clusters} crime clusters.")

cluster_summary = st.session_state.get("cluster_summary")
if cluster_summary is not None and len(cluster_summary) > 0:
    st.markdown("#### 🔵 Cluster Summary")

    cr1, cr2, cr3 = st.columns(3)
    with cr1:
        metric_card("Total Clusters", len(cluster_summary), "", "#1565c0")
    with cr2:
        high_clusters = (cluster_summary["risk_level"] == "High").sum()
        metric_card("High-Risk Clusters", high_clusters, "", "#c62828")
    with cr3:
        clustered_crimes = cluster_summary["crime_count"].sum()
        metric_card("Clustered Crimes", f"{clustered_crimes:,}", "", "#2e7d32")

    display_cols = ["cluster_id", "crime_count", "lat_center", "lon_center",
                    "top_crime", "risk_level"]
    display_cols = [c for c in display_cols if c in cluster_summary.columns]
    st.dataframe(cluster_summary[display_cols].head(20), use_container_width=True)

    import plotly.express as px
    fig_cl = px.scatter(
        cluster_summary,
        x="lon_center", y="lat_center",
        size="crime_count", color="risk_level",
        color_discrete_map={"High": "#d32f2f", "Medium": "#f57c00", "Low": "#388e3c"},
        hover_data=["crime_count", "top_crime"],
        title="Crime Cluster Centres",
        labels={"lon_center": "Longitude", "lat_center": "Latitude"},
        size_max=40, height=400,
    )
    fig_cl.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig_cl, use_container_width=True)

# Grid density
grid_density = st.session_state.get("grid_density")
if grid_density is not None and len(grid_density) > 0:
    st.markdown("#### 🗺 Grid Density Map (Top 20 Cells)")
    st.dataframe(grid_density.head(20), use_container_width=True)

# ---------------------------------------------------------------------------
# Analytics Charts
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 📈 Analytics & Charts")

t1, t2, t3, t4, t5 = st.tabs(
    ["Category Distribution", "Crime Trend", "Hourly Pattern",
     "Risk Distribution", "Monthly Heatmap"]
)

with t1:
    st.plotly_chart(plot_crime_by_category(df), use_container_width=True)

with t2:
    freq = st.radio("Frequency", ["Monthly", "Yearly"], horizontal=True, key="trend_freq")
    code = "ME" if freq == "Monthly" else "YE"
    st.plotly_chart(plot_crime_trend(df, freq=code), use_container_width=True)

with t3:
    if "hour" in df.columns:
        st.plotly_chart(plot_crime_by_hour(df), use_container_width=True)
    else:
        st.info("No 'time' column found — hourly chart unavailable.")

with t4:
    st.plotly_chart(plot_risk_distribution(df), use_container_width=True)

with t5:
    st.plotly_chart(plot_monthly_heatmap(df), use_container_width=True)

st.markdown("---")
st.caption("⚠️ All records are synthetic — for academic use only.")
