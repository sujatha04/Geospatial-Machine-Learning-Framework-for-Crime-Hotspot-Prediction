"""
visualization.py
----------------
Folium map builders and Plotly/Matplotlib chart helpers.
"""

import folium
from folium.plugins import HeatMap, MarkerCluster
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib
import io
import base64
import logging

matplotlib.use("Agg")
logger = logging.getLogger(__name__)

RISK_COLORS = {"High Risk": "#d32f2f", "Medium Risk": "#f57c00", "Low Risk": "#388e3c"}
RISK_MARKER_COLORS = {"High Risk": "red", "Medium Risk": "orange", "Low Risk": "green"}


# ---------------------------------------------------------------------------
# Folium Maps
# ---------------------------------------------------------------------------

def build_crime_map(
    df: pd.DataFrame,
    cluster_df: pd.DataFrame = None,
    show_heatmap: bool = True,
    show_clusters: bool = True,
    show_markers: bool = True,
    crime_filter: str = "All",
    zoom_start: int = 12,
) -> folium.Map:
    """
    Build an interactive Folium map with heatmap, cluster, and marker layers.

    Parameters
    ----------
    df            : crime records (must have latitude, longitude, crime_type, date)
    cluster_df    : optional cluster summary from geospatial_analysis.get_cluster_summary()
    show_heatmap  : add heatmap layer
    show_clusters : add DBSCAN cluster circles
    show_markers  : add individual crime markers
    crime_filter  : filter by crime_type ("All" = no filter)
    """
    center_lat = df["latitude"].mean()
    center_lon = df["longitude"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles="OpenStreetMap",
    )

    # Additional tile layers (free without API key)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        name="Light Gray Canvas",
        attr="Tiles &copy; Esri",
    ).add_to(m)
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        name="Topographic",
        attr="Tiles &copy; Esri",
    ).add_to(m)

    filtered = df.copy()
    if crime_filter != "All" and "crime_type" in df.columns:
        filtered = df[df["crime_type"] == crime_filter]

    # --- Heatmap Layer ---
    if show_heatmap and len(filtered) > 0:
        heat_data = filtered[["latitude", "longitude"]].dropna().values.tolist()
        heatmap_layer = folium.FeatureGroup(name="🌡 Crime Heatmap", show=True)
        HeatMap(heat_data, radius=15, blur=12, max_zoom=13,
                gradient={"0.2": "blue", "0.4": "lime", "0.6": "orange", "1": "red"}
                ).add_to(heatmap_layer)
        heatmap_layer.add_to(m)

    # --- Marker Cluster Layer ---
    if show_markers and len(filtered) > 0:
        marker_group = folium.FeatureGroup(name="📍 Crime Markers", show=True)
        mc = MarkerCluster().add_to(marker_group)
        for _, row in filtered.iterrows():
            if pd.isna(row["latitude"]) or pd.isna(row["longitude"]):
                continue
            risk_label = row.get("predicted_label", row.get("risk_label_text", "Unknown"))
            if isinstance(risk_label, (int, float)):
                risk_label = {2: "High Risk", 1: "Medium Risk", 0: "Low Risk"}.get(int(risk_label), "Unknown")

            color = RISK_MARKER_COLORS.get(str(risk_label), "blue")
            date_str = row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"])
            popup_html = f"""
                <div style='font-family:Arial;font-size:13px;min-width:180px'>
                    <b>🚨 {row.get('crime_type','N/A')}</b><br>
                    📅 {date_str}<br>
                    📍 ({row['latitude']:.4f}, {row['longitude']:.4f})<br>
                    📌 {row.get('location','N/A')}<br>
                    ⚠️ Risk: <b style='color:{RISK_COLORS.get(str(risk_label),"#333")}'>{risk_label}</b>
                </div>"""
            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                popup=folium.Popup(popup_html, max_width=250),
                icon=folium.Icon(color=color, icon="info-sign", prefix="glyphicon"),
                tooltip=f"{row.get('crime_type','?')} — {risk_label}",
            ).add_to(mc)
        marker_group.add_to(m)

    # --- DBSCAN Cluster Summary Circles ---
    if show_clusters and cluster_df is not None and len(cluster_df) > 0:
        cluster_group = folium.FeatureGroup(name="🔵 Crime Clusters", show=True)
        for _, row in cluster_df.iterrows():
            size = int(row.get("crime_count", 5))
            radius = max(100, min(size * 30, 2000))
            risk = str(row.get("risk_level", "Medium"))
            color = RISK_COLORS.get(f"{risk} Risk", "#1565c0")
            popup_html = f"""
                <div style='font-family:Arial;font-size:13px'>
                    <b>Cluster #{int(row['cluster_id'])}</b><br>
                    🔢 Crimes: {size}<br>
                    🏆 Top Crime: {row.get('top_crime','N/A')}<br>
                    ⚠️ Risk: <b style='color:{color}'>{risk}</b>
                </div>"""
            folium.Circle(
                location=[row["lat_center"], row["lon_center"]],
                radius=radius,
                color=color,
                fill=True,
                fill_opacity=0.25,
                popup=folium.Popup(popup_html, max_width=220),
                tooltip=f"Cluster {int(row['cluster_id'])}: {size} crimes",
            ).add_to(cluster_group)
        cluster_group.add_to(m)

    folium.LayerControl(position="topright", collapsed=False).add_to(m)
    return m


def build_hotspot_prediction_map(
    grid_df: pd.DataFrame,
    actual_df: pd.DataFrame = None,
    center_lat: float = None,
    center_lon: float = None,
) -> folium.Map:
    """
    Map showing model-predicted risk zones as coloured circles.
    Optionally overlays actual crime markers.
    """
    if center_lat is None:
        center_lat = grid_df["latitude"].mean()
    if center_lon is None:
        center_lon = grid_df["longitude"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles="OpenStreetMap",
    )
    folium.TileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        name="Light Gray Canvas",
        attr="Tiles &copy; Esri",
    ).add_to(m)

    color_map = {2: "#d32f2f", 1: "#f57c00", 0: "#388e3c"}
    label_map = {2: "High Risk", 1: "Medium Risk", 0: "Low Risk"}

    pred_group = folium.FeatureGroup(name="🗺 Predicted Risk Zones", show=True)
    for _, row in grid_df.iterrows():
        risk = int(row.get("predicted_risk", 0))
        color = color_map.get(risk, "#888888")
        label = label_map.get(risk, "Unknown")
        conf = float(row.get("confidence", 0.5))
        popup_html = f"""
            <div style='font-family:Arial;font-size:13px'>
                <b>Predicted Zone</b><br>
                📍 ({row['latitude']:.4f}, {row['longitude']:.4f})<br>
                ⚠️ Risk: <b style='color:{color}'>{label}</b><br>
                📊 Confidence: {conf:.0%}
            </div>"""
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=8,
            color=color,
            fill=True,
            fill_opacity=0.55,
            popup=folium.Popup(popup_html, max_width=200),
            tooltip=f"{label} ({conf:.0%})",
        ).add_to(pred_group)
    pred_group.add_to(m)

    if actual_df is not None and len(actual_df) > 0:
        actual_group = folium.FeatureGroup(name="📍 Actual Crimes", show=False)
        mc = MarkerCluster().add_to(actual_group)
        for _, row in actual_df.iterrows():
            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                icon=folium.Icon(color="blue", icon="info-sign", prefix="glyphicon"),
                tooltip=str(row.get("crime_type", "Crime")),
            ).add_to(mc)
        actual_group.add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:12px 16px;border-radius:8px;
                box-shadow:2px 2px 8px rgba(0,0,0,0.3);font-family:Arial;font-size:13px">
        <b>Risk Legend</b><br>
        <span style='color:#d32f2f'>●</span> High Risk<br>
        <span style='color:#f57c00'>●</span> Medium Risk<br>
        <span style='color:#388e3c'>●</span> Low Risk
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(position="topright", collapsed=False).add_to(m)
    return m


def add_location_marker(m: folium.Map, lat: float, lon: float,
                        result: dict) -> folium.Map:
    """Add a prominent marker for a searched location on the map."""
    risk = result.get("risk_level", "Unknown")
    risk_formatted = risk if "Risk" in risk else f"{risk} Risk"
    color_lookup = {
        "High Risk": "red", "High": "red",
        "Medium Risk": "orange", "Medium": "orange",
        "Low Risk": "green", "Low": "green",
    }
    color = color_lookup.get(risk, "blue")
    hex_color = RISK_COLORS.get(risk_formatted, RISK_COLORS.get(risk, "#1565c0"))

    popup_html = f"""
        <div style='font-family:Arial;font-size:14px;min-width:200px'>
            <b>🔍 Searched Location</b><br>
            📍 ({lat:.5f}, {lon:.5f})<br>
            ⚠️ Risk: <b style='color:{hex_color}'>{risk_formatted}</b><br>
            📊 Confidence: {result.get('confidence',0):.0%}<br>
            🔢 Nearby Crimes: {result.get('crime_count', 'N/A')}
        </div>"""
    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(popup_html, max_width=240),
        icon=folium.Icon(color=color, icon="star", prefix="glyphicon"),
        tooltip=f"📍 Searched Location ({risk_formatted})",
    ).add_to(m)
    folium.Circle(
        location=[lat, lon],
        radius=1000,
        color=hex_color,
        fill=False,
        weight=2,
        dash_array="5",
        tooltip="1 km radius",
    ).add_to(m)
    return m


# ---------------------------------------------------------------------------
# Plotly Charts
# ---------------------------------------------------------------------------

def plot_crime_by_category(df: pd.DataFrame) -> go.Figure:
    counts = df["crime_type"].value_counts().reset_index()
    counts.columns = ["Crime Type", "Count"]
    fig = px.bar(
        counts, x="Count", y="Crime Type", orientation="h",
        color="Count", color_continuous_scale="Blues",
        title="Crime Distribution by Category",
        labels={"Count": "Number of Incidents"},
    )
    fig.update_layout(showlegend=False, height=420,
                      plot_bgcolor="white", paper_bgcolor="white",
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig


def plot_crime_trend(df: pd.DataFrame, freq: str = "M") -> go.Figure:
    """Monthly or yearly crime trend line chart."""
    # Use pandas 2.2+ offset aliases (ME = month end, YE = year end)
    freq_alias = {"M": "ME", "Y": "YE"}.get(freq, freq)
    try:
        trend = df.set_index("date").resample(freq_alias).size().reset_index()
    except Exception:
        trend = df.set_index("date").resample(freq).size().reset_index()
    trend.columns = ["date", "count"]
    label = "Monthly" if freq in ("M", "ME") else "Yearly"
    fig = px.line(
        trend, x="date", y="count",
        title=f"{label} Crime Trend",
        labels={"date": "Date", "count": "Incidents"},
        markers=True,
    )
    fig.update_traces(line=dict(color="#1565c0", width=2.5))
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                      height=380, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def plot_crime_by_hour(df: pd.DataFrame) -> go.Figure:
    """Crime frequency by hour of day."""
    if "hour" not in df.columns:
        return go.Figure()
    hourly = df.groupby("hour").size().reset_index(name="count")
    fig = px.bar(
        hourly, x="hour", y="count",
        color="count", color_continuous_scale="Reds",
        title="Crime Frequency by Hour of Day",
        labels={"hour": "Hour (24h)", "count": "Incidents"},
    )
    fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                      height=360, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def plot_risk_distribution(df: pd.DataFrame) -> go.Figure:
    """Pie chart of risk levels."""
    col = None
    if "predicted_label" in df.columns:
        col = "predicted_label"
    elif "risk_label" in df.columns:
        label_map = {2: "High Risk", 1: "Medium Risk", 0: "Low Risk"}
        df = df.copy()
        df["_risk_text"] = df["risk_label"].map(label_map)
        col = "_risk_text"
    if col is None:
        return go.Figure()

    counts = df[col].value_counts().reset_index()
    counts.columns = ["Risk Level", "Count"]
    fig = px.pie(
        counts, names="Risk Level", values="Count",
        color="Risk Level",
        color_discrete_map={"High Risk": "#d32f2f", "Medium Risk": "#f57c00", "Low Risk": "#388e3c"},
        title="Risk Level Distribution",
        hole=0.4,
    )
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def plot_confusion_matrix(cm: np.ndarray) -> go.Figure:
    """Heatmap of confusion matrix."""
    labels = ["Low Risk", "Medium Risk", "High Risk"]
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale="Blues",
        text=cm, texttemplate="%{text}",
        showscale=True,
    ))
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        height=400,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def plot_feature_importance(importances: dict) -> go.Figure:
    """Bar chart of feature importances."""
    items = list(importances.items())[:10]
    features, values = zip(*items) if items else ([], [])
    fig = px.bar(
        x=list(values), y=list(features),
        orientation="h",
        color=list(values),
        color_continuous_scale="Greens",
        title="Top Feature Importances",
        labels={"x": "Importance", "y": "Feature"},
    )
    fig.update_layout(showlegend=False, height=400,
                      plot_bgcolor="white", paper_bgcolor="white",
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig


def plot_monthly_heatmap(df: pd.DataFrame) -> go.Figure:
    """Month × Weekday crime heatmap."""
    df = df.copy()
    df["month_name"] = df["date"].dt.strftime("%b")
    df["weekday"] = df["date"].dt.strftime("%a")
    pivot = df.groupby(["month_name", "weekday"]).size().unstack(fill_value=0)
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pivot = pivot.reindex(index=[m for m in month_order if m in pivot.index],
                          columns=[d for d in day_order if d in pivot.columns])
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="YlOrRd",
        showscale=True,
    ))
    fig.update_layout(
        title="Crime Frequency Heatmap (Month × Weekday)",
        xaxis_title="Day of Week",
        yaxis_title="Month",
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig
