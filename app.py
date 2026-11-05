"""
app.py
------
Main Landing & Executive Command Center for Geospatial ML Crime Hotspot Prediction.
Run with:  streamlit run app.py
"""

import sys
import os

# Ensure project root is on path so pages can import src.* and utils.*
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Crime Hotspot AI | Geospatial ML Platform",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.helpers import (
    init_session_state, apply_custom_css, get_df,
    get_data_path, load_sample_data_cached, metric_card
)

init_session_state()
apply_custom_css()

# Auto-load sample dataset if not yet loaded so the user immediately sees a live system
df = get_df()
if df is None:
    sample_path = get_data_path("sample_crime_data.csv")
    if os.path.exists(sample_path):
        try:
            df, warnings = load_sample_data_cached(sample_path)
            st.session_state["df_processed"] = df
            st.session_state["df_featured"] = df
            st.session_state["data_loaded"] = True
            if warnings:
                st.session_state["warnings"] = warnings
        except Exception:
            pass

# -------------------------------------------------------------------
# Sidebar Branding & Navigation Guide
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding: 18px 0 12px;'>
            <div style='font-size: 42px; filter: drop-shadow(0 4px 12px rgba(56, 189, 248, 0.4));'>🗺️</div>
            <div style='font-size: 17px; font-weight: 800; color: #f8fafc; letter-spacing: -0.01em; margin-top: 4px;'>
                GeoCrime <span style='color: #38bdf8;'>AI Studio</span>
            </div>
            <div style='font-size: 11px; color: #94a3b8; font-weight: 500; margin-top: 2px;'>
                Geospatial Machine Learning Framework
            </div>
            <div style='display: inline-block; margin-top: 10px; background: rgba(34, 197, 94, 0.15); border: 1px solid rgba(34, 197, 94, 0.35); color: #4ade80; padding: 2px 10px; border-radius: 9999px; font-size: 11px; font-weight: 600;'>
                ● System Active
            </div>
        </div>
        <hr style='border-color: rgba(255,255,255,0.08); margin: 8px 0 16px;'>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 🧭 Core Modules")
    st.markdown(
        """
        Navigate seamlessly across the platform:
        
        * 🏠 **[1_Dashboard]**: High-level KPIs & executive crime pulse
        * 📊 **[2_Crime_Analysis]**: Ingest data, filter records & DBSCAN
        * 🗺️ **[3_Hotspot_Map]**: Interactive Folium map & radius query
        * 🔮 **[4_Prediction]**: Train Random Forest & predict risk grid
        * 📈 **[5_Model_Performance]**: ROC curves, confusion matrix & reports
        """
    )

    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 16px 0;'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 8px; padding: 10px 12px;'>
            <div style='font-size: 11px; font-weight: 700; color: #cbd5e1; text-transform: uppercase;'>Academic Research</div>
            <div style='font-size: 10.5px; color: #94a3b8; line-height: 1.4; margin-top: 4px;'>
                Uses synthetic crime spatial distributions for algorithmic benchmarking.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------------------------------------------------
# Hero Banner
# -------------------------------------------------------------------
st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #090d16 0%, #0f172a 40%, #1e3a8a 80%, #2563eb 100%);
        padding: 34px 38px;
        border-radius: 18px;
        margin-bottom: 26px;
        box-shadow: 0 15px 35px -5px rgba(15, 23, 42, 0.35), 0 0 0 1px rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    ">
        <div style="position: absolute; right: -20px; top: -20px; width: 220px; height: 220px; background: radial-gradient(circle, rgba(56, 189, 248, 0.18) 0%, transparent 70%); border-radius: 50%; pointer-events: none;"></div>
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
            <span style="background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4); font-size: 11px; font-weight: 700; padding: 3px 12px; border-radius: 9999px; text-transform: uppercase; letter-spacing: 0.6px;">
                GEOSPATIAL AI 2.0
            </span>
            <span style="color: #93c5fd; font-size: 12px; font-weight: 600;">• Enterprise Analytics & Spatial Intelligence</span>
        </div>
        <h1 style="color: #ffffff; margin: 0 0 10px 0; font-size: 32px; font-weight: 800; letter-spacing: -0.025em; line-height: 1.2;">
            Crime Hotspot Prediction & Spatial ML Framework
        </h1>
        <p style="color: #cbd5e1; margin: 0; font-size: 14.5px; font-weight: 400; max-width: 820px; line-height: 1.55;">
            Detect high-risk geographical zones, extract spatio-temporal crime patterns with <b>DBSCAN</b>, and predict future safety levels using ensemble <b>Random Forest classifiers</b>.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# Telemetry Status Matrix
# -------------------------------------------------------------------
total_records = len(df) if df is not None else 0
categories_count = df["crime_type"].nunique() if df is not None else 0

if df is not None and "predicted_label" in df.columns:
    high_count = (df["predicted_label"] == "High Risk").sum()
elif df is not None and "risk_label" in df.columns:
    high_count = (df["risk_label"] == 2).sum()
else:
    high_count = 0

model_is_trained = st.session_state.get("model_trained", False)

col1, col2, col3, col4 = st.columns(4)

with col1:
    metric_card(
        title="Dataset Ingestion",
        value=f"{total_records:,}" if total_records > 0 else "0",
        subtitle="✅ Active synthetic records" if total_records > 0 else "Upload CSV to begin",
        color="#2563eb",
        icon="📂"
    )
with col2:
    metric_card(
        title="Incident Types",
        value=f"{categories_count}" if categories_count > 0 else "—",
        subtitle="Distinct crime classifications",
        color="#0284c7",
        icon="🏷️"
    )
with col3:
    metric_card(
        title="Critical Hotspots",
        value=f"{high_count:,}" if high_count > 0 else "0",
        subtitle=f"{(high_count/total_records*100):.1f}% classified as high risk" if total_records > 0 else "Pending classification",
        color="#dc2626",
        icon="🔥"
    )
with col4:
    metric_card(
        title="Predictive Engine",
        value="Ready" if model_is_trained else "Standby",
        subtitle="Random Forest v1.0" if model_is_trained else "Train via Prediction tab",
        color="#059669" if model_is_trained else "#d97706",
        icon="🤖"
    )

st.write("")

# -------------------------------------------------------------------
# Interactive Platform Feature Cards (Command Center)
# -------------------------------------------------------------------
st.markdown("### ⚡ Interactive Feature Command Center")
st.markdown("Select an operation from the sidebar or jump right into the specialized workflows:")

card_c1, card_c2, card_c3 = st.columns(3)

with card_c1:
    st.markdown(
        """
        <div style="
            background: white;
            border-radius: 14px;
            padding: 22px 20px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        ">
            <div>
                <div style="width: 44px; height: 44px; border-radius: 10px; background: #eff6ff; display: flex; align-items: center; justify-content: center; font-size: 22px; margin-bottom: 14px;">
                    🗺️
                </div>
                <h4 style="margin: 0 0 6px 0; font-size: 16px; font-weight: 700; color: #0f172a;">
                    Hotspot Density Map
                </h4>
                <p style="font-size: 13px; color: #64748b; line-height: 1.5; margin: 0 0 14px 0;">
                    Visualize crime spatial clusters with dynamic heatmaps, Leaflet marker clustering, and DBSCAN density boundary contours.
                </p>
            </div>
            <div style="background: #f8fafc; border-radius: 8px; padding: 8px 12px; font-size: 11.5px; font-weight: 600; color: #2563eb; border: 1px solid #e2e8f0;">
                👉 Open <b>3_Hotspot_Map</b> in sidebar
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with card_c2:
    st.markdown(
        """
        <div style="
            background: white;
            border-radius: 14px;
            padding: 22px 20px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        ">
            <div>
                <div style="width: 44px; height: 44px; border-radius: 10px; background: #fdf2f8; display: flex; align-items: center; justify-content: center; font-size: 22px; margin-bottom: 14px;">
                    🔮
                </div>
                <h4 style="margin: 0 0 6px 0; font-size: 16px; font-weight: 700; color: #0f172a;">
                    ML Risk Forecaster
                </h4>
                <p style="font-size: 13px; color: #64748b; line-height: 1.5; margin: 0 0 14px 0;">
                    Train a multi-class Random Forest model to predict risk levels for any custom coordinate or generate a full urban grid prediction.
                </p>
            </div>
            <div style="background: #fdf2f8; border-radius: 8px; padding: 8px 12px; font-size: 11.5px; font-weight: 600; color: #db2777; border: 1px solid #fce7f3;">
                👉 Open <b>4_Prediction</b> in sidebar
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with card_c3:
    st.markdown(
        """
        <div style="
            background: white;
            border-radius: 14px;
            padding: 22px 20px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        ">
            <div>
                <div style="width: 44px; height: 44px; border-radius: 10px; background: #ecfdf5; display: flex; align-items: center; justify-content: center; font-size: 22px; margin-bottom: 14px;">
                    📈
                </div>
                <h4 style="margin: 0 0 6px 0; font-size: 16px; font-weight: 700; color: #0f172a;">
                    Model Performance & Audit
                </h4>
                <p style="font-size: 13px; color: #64748b; line-height: 1.5; margin: 0 0 14px 0;">
                    Evaluate Precision, Recall, Confusion Matrix, and Gini Feature Importance. Export comprehensive security analysis audit reports.
                </p>
            </div>
            <div style="background: #ecfdf5; border-radius: 8px; padding: 8px 12px; font-size: 11.5px; font-weight: 600; color: #059669; border: 1px solid #d1fae5;">
                👉 Open <b>5_Model_Performance</b> in sidebar
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")

# -------------------------------------------------------------------
# Architectural Pipeline Flowchart
# -------------------------------------------------------------------
st.markdown("### 🔬 Geospatial Machine Learning Pipeline")
st.markdown(
    """
    <div style="
        background: white;
        border-radius: 14px;
        padding: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.04);
        margin-bottom: 20px;
    ">
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px;">
            <div style="padding: 14px; background: #f8fafc; border-radius: 10px; border-left: 3px solid #2563eb;">
                <div style="font-size: 11px; font-weight: 700; color: #2563eb; text-transform: uppercase;">Step 1</div>
                <div style="font-weight: 700; font-size: 14px; color: #0f172a; margin: 2px 0 4px;">Data Ingestion</div>
                <div style="font-size: 12px; color: #64748b;">Coordinate validation, timestamp parsing, schema verification & synthetic fallback.</div>
            </div>
            <div style="padding: 14px; background: #f8fafc; border-radius: 10px; border-left: 3px solid #0284c7;">
                <div style="font-size: 11px; font-weight: 700; color: #0284c7; text-transform: uppercase;">Step 2</div>
                <div style="font-weight: 700; font-size: 14px; color: #0f172a; margin: 2px 0 4px;">Feature Engineering</div>
                <div style="font-size: 12px; color: #64748b;">Spatial grid indexing, temporal cyclics (hour, weekday, month), and risk severity scoring.</div>
            </div>
            <div style="padding: 14px; background: #f8fafc; border-radius: 10px; border-left: 3px solid #7c3aed;">
                <div style="font-size: 11px; font-weight: 700; color: #7c3aed; text-transform: uppercase;">Step 3</div>
                <div style="font-weight: 700; font-size: 14px; color: #0f172a; margin: 2px 0 4px;">DBSCAN Clustering</div>
                <div style="font-size: 12px; color: #64748b;">Haversine metric density clustering to isolate dense incident epicenters from spatial noise.</div>
            </div>
            <div style="padding: 14px; background: #f8fafc; border-radius: 10px; border-left: 3px solid #059669;">
                <div style="font-size: 11px; font-weight: 700; color: #059669; text-transform: uppercase;">Step 4</div>
                <div style="font-weight: 700; font-size: 14px; color: #0f172a; margin: 2px 0 4px;">Predictive Modeling</div>
                <div style="font-size: 12px; color: #64748b;">Multi-class Random Forest with balanced class weights predicting Low, Medium & High risk zones.</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------------------
# Disclaimer & Academic Safety Note
# -------------------------------------------------------------------
st.markdown(
    """
    <div style="
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
        border-radius: 10px;
        padding: 14px 18px;
        margin-top: 24px;
    ">
        <div style="font-size: 12.5px; font-weight: 700; color: #b45309; display: flex; align-items: center; gap: 6px;">
            <span>⚠️</span> RESEARCH & ACADEMIC USE ONLY
        </div>
        <div style="font-size: 12px; color: #92400e; margin-top: 4px; line-height: 1.5;">
            All datasets and location coordinates within this framework are synthetically generated for machine learning research, algorithm validation, and interface evaluation. This system is not calibrated for live municipal policing or operational law enforcement deployment.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
