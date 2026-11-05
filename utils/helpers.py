"""
helpers.py
----------
Shared utility functions for session state, caching, exports, and UI helpers.
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
import os
import sys
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_project_root() -> str:
    """Return the absolute path to the crime_hotspot_prediction package root."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_data_path(filename: str = "sample_crime_data.csv") -> str:
    return os.path.join(get_project_root(), "data", filename)


def get_model_path(filename: str = "crime_model.pkl") -> str:
    return os.path.join(get_project_root(), "models", filename)


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

def init_session_state():
    """Initialise all session-state keys with safe defaults."""
    defaults = {
        "df_raw": None,
        "df_processed": None,
        "df_featured": None,
        "df_clustered": None,
        "cluster_summary": None,
        "grid_density": None,
        "model": None,
        "scaler": None,
        "train_results": None,
        "model_trained": False,
        "data_loaded": False,
        "warnings": [],
        "selected_crime_filter": "All",
        "selected_date_range": None,
        "search_result": None,
        "search_coords": None,
        "single_prediction_results": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def get_df() -> pd.DataFrame:
    """Return the processed DataFrame from session state, or None."""
    return st.session_state.get("df_processed", None)


def get_featured_df() -> pd.DataFrame:
    return st.session_state.get("df_featured", None)


def set_df(df: pd.DataFrame, key: str = "df_processed"):
    st.session_state[key] = df


# ---------------------------------------------------------------------------
# Data loading with caching
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_sample_data_cached(path: str):
    """Load and cache the sample dataset."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_preprocessing import load_and_validate, engineer_features, compute_risk_label
    df, warnings = load_and_validate(path)
    df = engineer_features(df)
    df = compute_risk_label(df)
    return df, warnings


@st.cache_data(show_spinner=False)
def process_uploaded_data(content: bytes):
    """Parse and process an uploaded CSV file."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_preprocessing import load_and_validate, engineer_features, compute_risk_label
    df_raw = pd.read_csv(io.BytesIO(content))
    df, warnings = load_and_validate(df_raw)
    df = engineer_features(df)
    df = compute_risk_label(df)
    return df, warnings


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert a DataFrame to UTF-8 CSV bytes for download."""
    return df.to_csv(index=False).encode("utf-8")


def generate_report_text(df: pd.DataFrame, train_results: dict = None) -> str:
    """Generate a plain-text crime analysis report."""
    lines = [
        "=" * 60,
        "  GEOSPATIAL CRIME HOTSPOT PREDICTION — ANALYSIS REPORT",
        "=" * 60,
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  DISCLAIMER: Synthetic / demonstration data only.",
        "=" * 60,
        "",
        "DATASET SUMMARY",
        "-" * 40,
        f"Total Records     : {len(df)}",
        f"Crime Categories  : {df['crime_type'].nunique()}",
        f"Date Range        : {df['date'].min().strftime('%Y-%m-%d') if pd.notna(df['date'].min()) else 'N/A'} to {df['date'].max().strftime('%Y-%m-%d') if pd.notna(df['date'].max()) else 'N/A'}",
        "",
        "CRIME TYPE BREAKDOWN",
        "-" * 40,
    ]
    for crime, count in df["crime_type"].value_counts().items():
        pct = count / len(df) * 100
        lines.append(f"  {crime:<25} {count:>5}  ({pct:.1f}%)")

    if "risk_label" in df.columns or "predicted_label" in df.columns:
        lines += ["", "RISK LEVEL DISTRIBUTION", "-" * 40]
        col = "predicted_label" if "predicted_label" in df.columns else "risk_label"
        if col == "risk_label":
            label_map = {2: "High Risk", 1: "Medium Risk", 0: "Low Risk"}
            risk_series = df[col].map(label_map)
        else:
            risk_series = df[col]
        for level, count in risk_series.value_counts().items():
            pct = count / len(df) * 100
            lines.append(f"  {level:<20} {count:>5}  ({pct:.1f}%)")

    if train_results:
        lines += [
            "",
            "MODEL PERFORMANCE",
            "-" * 40,
            f"Algorithm         : Random Forest Classifier",
            f"Training Records  : {train_results.get('train_size', 'N/A')}",
            f"Testing Records   : {train_results.get('test_size', 'N/A')}",
            f"Accuracy          : {train_results.get('accuracy', 0):.4f}",
            f"Precision         : {train_results.get('precision', 0):.4f}",
            f"Recall            : {train_results.get('recall', 0):.4f}",
            f"F1 Score          : {train_results.get('f1_score', 0):.4f}",
            f"CV F1 (mean±std)  : {train_results.get('cv_mean', 0):.4f} ± {train_results.get('cv_std', 0):.4f}",
        ]

    lines += ["", "=" * 60,
              "  END OF REPORT — ACADEMIC USE ONLY", "=" * 60]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def apply_custom_css():
    """Inject modern polished styling across all Streamlit pages."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

        /* Global Typography */
        html, body, [class*="css"], [class*="st-"] {
            font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }

        /* Modern background */
        .stApp {
            background-color: #f8fafc;
            background-image: 
                radial-gradient(at 0% 0%, rgba(37, 99, 235, 0.03) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(6, 182, 212, 0.03) 0px, transparent 50%);
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(185deg, #090d16 0%, #0f172a 45%, #1e293b 100%) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.06);
        }
        [data-testid="stSidebar"] * {
            color: #e2e8f0 !important;
        }
        [data-testid="stSidebarNav"] a {
            border-radius: 8px;
            margin: 3px 8px;
            transition: all 0.2s ease;
            padding: 8px 14px !important;
        }
        [data-testid="stSidebarNav"] a:hover {
            background: rgba(255, 255, 255, 0.08) !important;
            transform: translateX(3px);
        }
        [data-testid="stSidebarNav"] a[aria-current="page"] {
            background: linear-gradient(90deg, rgba(37, 99, 235, 0.4), rgba(6, 182, 212, 0.2)) !important;
            border-left: 3px solid #38bdf8 !important;
            font-weight: 600 !important;
        }

        /* Primary Button */
        .stButton > button {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            font-size: 0.875rem !important;
            padding: 0.55rem 1.25rem !important;
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.25) !important;
            transition: all 0.2s ease !important;
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
            box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4) !important;
            transform: translateY(-1px) !important;
        }

        /* Download Button */
        .stDownloadButton > button {
            background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 14px rgba(5, 150, 105, 0.25) !important;
            transition: all 0.2s ease !important;
        }
        .stDownloadButton > button:hover {
            background: linear-gradient(135deg, #047857 0%, #065f46 100%) !important;
            box-shadow: 0 6px 18px rgba(5, 150, 105, 0.4) !important;
            transform: translateY(-1px) !important;
        }

        /* Form Controls */
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div {
            border-radius: 8px !important;
            border: 1.5px solid #cbd5e1 !important;
            background-color: white !important;
        }
        div[data-baseweb="select"] > div:focus-within,
        div[data-baseweb="input"] > div:focus-within {
            border-color: #2563eb !important;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12) !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px !important;
            border-bottom: 2px solid #e2e8f0 !important;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 6px !important;
            padding: 8px 16px !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            color: #64748b !important;
        }
        .stTabs [aria-selected="true"] {
            color: #2563eb !important;
            background: #eff6ff !important;
        }

        /* Dataframes */
        div[data-testid="stDataFrame"] {
            border-radius: 12px !important;
            overflow: hidden !important;
            border: 1px solid #e2e8f0 !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(title: str, value, subtitle: str = "", color: str = "#2563eb", icon: str = ""):
    """Render a styled metric card using st.markdown."""
    st.markdown(
        f"""
        <div style="
            background: white;
            border-radius: 12px;
            padding: 16px 18px;
            margin: 4px 0;
            border: 1px solid #e2e8f0;
            border-left: 4px solid {color};
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04), 0 2px 4px -2px rgba(0, 0, 0, 0.04);
            transition: all 0.2s ease;
        ">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
                <span style="font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.6px;">
                    {title}
                </span>
                {"<span style='font-size: 15px;'>" + icon + "</span>" if icon else ""}
            </div>
            <div style="font-size: 26px; font-weight: 800; color: {color}; line-height: 1.1; letter-spacing: -0.02em;">
                {value}
            </div>
            {"<div style='font-size: 11px; color: #94a3b8; font-weight: 500; margin-top: 5px;'>" + subtitle + "</div>" if subtitle else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_badge(risk_level: str) -> str:
    colors = {
        "High Risk": "#dc2626", "High": "#dc2626",
        "Medium Risk": "#ea580c", "Medium": "#ea580c",
        "Low Risk": "#16a34a", "Low": "#16a34a"
    }
    c = colors.get(risk_level, "#64748b")
    label = risk_level if "Risk" in str(risk_level) else f"{risk_level} Risk"
    return (
        f'<span style="background:{c};color:white;padding:3px 10px;'
        f'border-radius:9999px;font-size:11px;font-weight:700;letter-spacing:0.3px;">{label}</span>'
    )


def page_header(title: str, subtitle: str = "", badge: str = "Geospatial ML"):
    apply_custom_css()
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #2563eb 100%);
            padding: 24px 28px;
            border-radius: 14px;
            margin-bottom: 24px;
            box-shadow: 0 10px 25px -5px rgba(30, 58, 138, 0.25), 0 8px 10px -6px rgba(30, 58, 138, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.1);
            position: relative;
            overflow: hidden;
        ">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                <span style="background: rgba(255, 255, 255, 0.15); color: #93c5fd; font-size: 11px; font-weight: 700; padding: 2px 10px; border-radius: 9999px; text-transform: uppercase; letter-spacing: 0.5px; border: 1px solid rgba(255, 255, 255, 0.2);">
                    {badge}
                </span>
                <span style="color: #60a5fa; font-size: 12px; font-weight: 600;">• Crime Hotspot Framework</span>
            </div>
            <h1 style="color: white; margin: 0; font-size: 24px; font-weight: 800; letter-spacing: -0.02em;">
                {title}
            </h1>
            {"<p style='color: #bfdbfe; margin: 6px 0 0; font-size: 13px; font-weight: 400; max-width: 800px;'>" + subtitle + "</p>" if subtitle else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def no_data_warning():
    st.warning(
        "⚠️ No dataset loaded. Please go to the **Crime Data Management** page and "
        "upload a CSV file or load the sample dataset.",
        icon="📂",
    )


def ensure_src_on_path():
    """Add project root to sys.path so `src.*` imports resolve correctly."""
    root = get_project_root()
    if root not in sys.path:
        sys.path.insert(0, root)
