"""
app.py
------
Entry point for the Geospatial ML Crime Hotspot Prediction dashboard.
Run with:  streamlit run app.py
"""

import sys
import os

# Ensure project root is on path so pages can import src.* and utils.*
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

st.set_page_config(
    page_title="Crime Hotspot Prediction",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Global CSS ---
st.markdown(
    """
    <style>
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #1b2a3b 100%);
        color: #e3f2fd;
    }
    [data-testid="stSidebar"] .css-1d391kg, [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {color: #e3f2fd !important;}
    /* Main bg */
    .main {background-color: #f5f7fa;}
    /* Cards */
    div[data-testid="metric-container"] {
        background: white; border-radius: 8px; padding: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1565c0, #0d47a1);
        color: white; border: none; border-radius: 6px;
        font-weight: 600; padding: 8px 18px;
    }
    .stButton > button:hover {background: #1976d2;}
    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-weight: 600; color: #1565c0;
    }
    /* Tables */
    .dataframe {font-size: 12px !important;}
    /* Success/Warning box */
    .success-box {
        background: #e8f5e9; border-left: 4px solid #388e3c;
        border-radius: 6px; padding: 12px 16px; margin: 8px 0;
    }
    .warning-box {
        background: #fff8e1; border-left: 4px solid #f9a825;
        border-radius: 6px; padding: 12px 16px; margin: 8px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

from utils.helpers import init_session_state
init_session_state()

# -------------------------------------------------------------------
# Sidebar branding
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center;padding:16px 0 8px'>
            <div style='font-size:40px'>🗺️</div>
            <div style='font-size:16px;font-weight:700;color:#90caf9'>Crime Hotspot Prediction</div>
            <div style='font-size:11px;color:#78909c;margin-top:4px'>
                Geospatial ML Framework
            </div>
        </div>
        <hr style='border-color:#2a3f5f;margin:8px 0'>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📂 Navigation")
    st.markdown(
        """
        Use the **pages** in the sidebar (above) to navigate:

        | Page | Description |
        |------|-------------|
        | 🏠 Dashboard | Overview & KPIs |
        | 📊 Crime Analysis | Dataset & Charts |
        | 🗺 Hotspot Map | Interactive Map |
        | 🔮 Prediction | ML Predictions |
        | 📈 Model Performance | Metrics & Report |
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr style='border-color:#2a3f5f'>", unsafe_allow_html=True)
    st.caption("⚠️ Uses synthetic data for academic demonstration.")
    st.caption("Not for operational law-enforcement use.")

# -------------------------------------------------------------------
# Home page
# -------------------------------------------------------------------
st.markdown(
    """
    <div style="background:linear-gradient(90deg,#1565c0,#0d47a1);
                padding:28px 32px;border-radius:12px;margin-bottom:24px;text-align:center">
        <h1 style="color:white;margin:0;font-size:28px">
            🗺️ Geospatial ML Crime Hotspot Prediction
        </h1>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "👈 Use the sidebar to navigate between pages. "
    "Start by loading data on the **Crime Analysis** page, then train the model and explore predictions.",
    icon="ℹ️",
)

# Quick status
col1, col2, col3, col4 = st.columns(4)
df = st.session_state.get("df_processed")

with col1:
    st.metric("📋 Dataset Status",
              "✅ Loaded" if df is not None else "❌ Not Loaded",
              f"{len(df):,} records" if df is not None else "Load data first")
with col2:
    st.metric("🤖 Model Status",
              "✅ Trained" if st.session_state.get("model_trained") else "❌ Not Trained",
              "Ready for predictions" if st.session_state.get("model_trained") else "Train on ML page")
with col3:
    cats = df["crime_type"].nunique() if df is not None else 0
    st.metric("📂 Crime Categories", cats if cats > 0 else "—")
with col4:
    if df is not None and "predicted_label" in df.columns:
        high = (df["predicted_label"] == "High Risk").sum()
    elif df is not None and "risk_label" in df.columns:
        high = (df["risk_label"] == 2).sum()
    else:
        high = 0
    st.metric("🔴 High-Risk Records", high if high > 0 else "—")

st.markdown("---")

# Feature overview
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown("""
    #### 🔍 What this system does
    - Loads & validates crime CSV datasets
    - Engineers spatial & temporal features
    - Clusters crimes with DBSCAN / K-Means
    - Trains a Random Forest classifier
    - Predicts risk levels across a spatial grid
    """)
with col_b:
    st.markdown("""
    #### 📊 Visualisations available
    - Interactive Folium map with heatmap
    - Crime trend & category charts
    - Confusion matrix & feature importance
    - Monthly × weekday crime heatmap
    - Risk distribution pie chart
    """)
with col_c:
    st.markdown("""
    #### 🚀 Quick start
    1. Go to **Crime Analysis** → load sample data
    2. Open **Hotspot Map** to explore clusters
    3. Go to **Prediction** → train model
    4. View metrics on **Model Performance**
    5. Download report from **Model Performance**
    """)

st.markdown("---")
st.caption(
    "⚠️ **DISCLAIMER** — All datasets used in this project are SYNTHETIC (computer-generated) "
    "and created solely for academic purposes. They do not represent real police records, "
    "actual crime statistics, or any real persons or locations. "
    "Do not use this system for operational law enforcement decisions."
)
