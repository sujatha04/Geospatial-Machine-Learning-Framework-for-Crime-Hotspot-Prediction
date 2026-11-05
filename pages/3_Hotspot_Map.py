"""
pages/3_Hotspot_Map.py
-----------------------
Interactive Folium map with heatmap, clusters, and location search.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

from utils.helpers import (
    init_session_state, page_header, metric_card, no_data_warning,
    get_data_path, load_sample_data_cached, ensure_src_on_path, risk_badge,
    apply_custom_css,
)

ensure_src_on_path()

from src.geospatial_analysis import (
    run_dbscan, get_cluster_summary, get_nearby_crimes, location_risk_summary,
)
from src.visualization import build_crime_map, add_location_marker

st.set_page_config(page_title="Hotspot Map | GeoCrime AI", page_icon="🗺", layout="wide")
init_session_state()
apply_custom_css()

page_header(
    title="Interactive Geospatial Hotspot Map",
    subtitle="High-density kernel heatmaps, multi-layer marker clustering, and on-demand radial point risk querying.",
    badge="Cartographic View"
)

# Auto-load sample data if needed
df = st.session_state.get("df_processed")
if df is None:
    sample_path = get_data_path("sample_crime_data.csv")
    if os.path.exists(sample_path):
        with st.spinner("Loading dataset…"):
            df, warnings = load_sample_data_cached(sample_path)
            st.session_state["df_processed"] = df
            st.session_state["df_featured"] = df
    else:
        no_data_warning()
        st.stop()

# ---------------------------------------------------------------------------
# Sidebar Map Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🗺 Map Layer Controls")

    crime_types = ["All"] + sorted(df["crime_type"].unique().tolist())
    crime_filter = st.selectbox("Filter Crime Modality", crime_types, key="map_crime_filter")

    date_min = df["date"].min().date()
    date_max = df["date"].max().date()
    map_date_range = st.date_input(
        "Date Range Filter",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max,
        key="map_date_range",
    )

    show_heatmap   = st.checkbox("🌡 Density Heatmap Layer", value=True)
    show_markers   = st.checkbox("📍 Crime Incidents Cluster", value=True)
    show_clusters  = st.checkbox("🔵 DBSCAN Cluster Centroids", value=True)

    st.markdown("---")
    st.markdown("### 🔍 Point Radius Assessment")
    search_lat = st.number_input("Latitude",  value=float(df["latitude"].mean()),
                                 format="%.5f", key="search_lat")
    search_lon = st.number_input("Longitude", value=float(df["longitude"].mean()),
                                 format="%.5f", key="search_lon")
    search_radius = st.slider("Radial Buffer (km)", 0.5, 10.0, 1.0, 0.5)
    do_search = st.button("⚡ Query Location Risk", type="primary")

# ---------------------------------------------------------------------------
# Filter data
# ---------------------------------------------------------------------------
filtered_df = df.copy()
if crime_filter != "All":
    filtered_df = filtered_df[filtered_df["crime_type"] == crime_filter]
if len(map_date_range) == 2:
    s, e = pd.Timestamp(map_date_range[0]), pd.Timestamp(map_date_range[1])
    filtered_df = filtered_df[(filtered_df["date"] >= s) & (filtered_df["date"] <= e)]

# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
cluster_summary = st.session_state.get("cluster_summary")
if cluster_summary is None:
    with st.spinner("Computing spatial clusters…"):
        df_clustered = run_dbscan(df, eps_km=0.5, min_samples=5)
        cluster_summary = get_cluster_summary(df_clustered)
        st.session_state["df_clustered"] = df_clustered
        st.session_state["cluster_summary"] = cluster_summary

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
with m1:
    metric_card("Rendered Points", f"{len(filtered_df):,}", "Active on map", "#2563eb", icon="📌")
with m2:
    metric_card("DBSCAN Epicenters", len(cluster_summary) if cluster_summary is not None else 0,
                "Spatial clusters", "#0284c7", icon="🎯")
with m3:
    metric_card("Geographic Centroid",
                f"{filtered_df['latitude'].mean():.3f}, {filtered_df['longitude'].mean():.3f}",
                "Grid center", "#475569", icon="🌐")
with m4:
    high_cl = (cluster_summary["risk_level"].isin(["High", "High Risk"])).sum() if cluster_summary is not None else 0
    metric_card("High-Risk Hotspots", high_cl, "Immediate attention", "#dc2626", icon="🚨")

st.write("")

# ---------------------------------------------------------------------------
# Location search result
# ---------------------------------------------------------------------------
if do_search:
    with st.spinner("Executing spatial radius query…"):
        nearby = get_nearby_crimes(df, search_lat, search_lon, radius_km=search_radius)
        search_result = location_risk_summary(nearby)
        st.session_state["search_result"] = search_result
        st.session_state["search_coords"] = (search_lat, search_lon, search_radius)

search_result = st.session_state.get("search_result")
search_coords = st.session_state.get("search_coords")

if search_result and search_coords:
    s_lat, s_lon, s_rad = search_coords
    col_res1, col_res2 = st.columns([1, 2])
    with col_res1:
        risk = search_result["risk_level"]
        color_map = {
            "High Risk": "#dc2626", "High": "#dc2626",
            "Medium Risk": "#d97706", "Medium": "#d97706",
            "Low Risk": "#16a34a", "Low": "#16a34a"
        }
        c = color_map.get(risk, "#475569")
        st.markdown(
            f"""
            <div style='background: white; border: 1px solid #e2e8f0; border-left: 5px solid {c};
                        border-radius: 12px; padding: 18px 22px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);'>
                <div style='font-size: 11px; color: #64748b; font-weight: 700; text-transform: uppercase;'>SPATIAL RISK ASSESSMENT</div>
                <div style='font-size: 26px; font-weight: 800; color: {c}; margin: 4px 0 8px 0;'>{risk}</div>
                <div style='font-size: 13px; color: #334155; line-height: 1.6;'>
                    🎯 Assessment Confidence: <b>{search_result['confidence']:.0%}</b><br>
                    🔢 Incidents within Buffer: <b>{search_result['crime_count']}</b><br>
                    🏷️ Dominant Crime: <b>{search_result['top_crime']}</b><br>
                    📏 Search Radius: <b>{s_rad} km</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_res2:
        if search_result.get("crime_breakdown"):
            import plotly.express as px
            breakdown_df = pd.DataFrame(
                list(search_result["crime_breakdown"].items()),
                columns=["Crime Type", "Count"]
            ).sort_values("Count", ascending=False)
            fig = px.bar(
                breakdown_df, x="Count", y="Crime Type", orientation="h",
                color="Count", color_continuous_scale="Reds", height=240,
                title=f"Incident Breakdown ({s_rad} km radius)",
                template="plotly_white"
            )
            fig.update_layout(showlegend=False, margin=dict(l=0, r=10, t=30, b=0),
                              plot_bgcolor="rgba(248,250,252,0.6)", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

    st.write("")

# ---------------------------------------------------------------------------
# Build and display map
# ---------------------------------------------------------------------------
st.markdown("### 🗺 Geospatial Map Visualization")

with st.spinner("Rendering vector tiles and heat layers…"):
    m = build_crime_map(
        df=filtered_df,
        cluster_df=cluster_summary if show_clusters else None,
        show_heatmap=show_heatmap,
        show_markers=show_markers,
        show_clusters=show_clusters,
        crime_filter=crime_filter,
    )
    if search_result and search_coords:
        s_lat, s_lon, _ = search_coords
        m = add_location_marker(m, s_lat, s_lon, search_result)
        m.location = [s_lat, s_lon]

map_data = st_folium(m, width="100%", height=600, returned_objects=["last_clicked"])

# Show click info
if map_data and map_data.get("last_clicked"):
    clicked_lat = map_data["last_clicked"]["lat"]
    clicked_lon = map_data["last_clicked"]["lng"]
    nearby = get_nearby_crimes(df, clicked_lat, clicked_lon, radius_km=0.5)
    result = location_risk_summary(nearby)
    st.info(
        f"📍 Selected Coordinates: ({clicked_lat:.4f}, {clicked_lon:.4f}) — "
        f"**{result['risk_level']}** | {result['crime_count']} historical crimes detected within 0.5 km radius."
    )

# ---------------------------------------------------------------------------
# Cluster table below map
# ---------------------------------------------------------------------------
if cluster_summary is not None and len(cluster_summary) > 0:
    with st.expander("📋 Inspect Cluster Telemetry Details", expanded=False):
        cols = [c for c in ["cluster_id","crime_count","lat_center","lon_center","top_crime","risk_level"]
                if c in cluster_summary.columns]
        st.dataframe(cluster_summary[cols], use_container_width=True)

st.caption("⚠️ All coordinates and incidents are synthetic — for research and academic evaluation.")
