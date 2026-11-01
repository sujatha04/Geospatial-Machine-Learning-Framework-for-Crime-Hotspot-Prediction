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

def metric_card(title: str, value, subtitle: str = "", color: str = "#1565c0"):
    """Render a styled metric card using st.markdown."""
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {color}22, {color}11);
            border-left: 4px solid {color};
            border-radius: 8px;
            padding: 14px 18px;
            margin: 4px 0;
        ">
            <div style="font-size:12px;color:#555;font-weight:600;text-transform:uppercase;
                        letter-spacing:0.5px">{title}</div>
            <div style="font-size:26px;font-weight:700;color:{color};line-height:1.2">{value}</div>
            <div style="font-size:11px;color:#777;margin-top:2px">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_badge(risk_level: str) -> str:
    colors = {
        "High Risk": "#d32f2f", "High": "#d32f2f",
        "Medium Risk": "#e65100", "Medium": "#e65100",
        "Low Risk": "#2e7d32", "Low": "#2e7d32"
    }
    c = colors.get(risk_level, "#555")
    label = risk_level if "Risk" in str(risk_level) else f"{risk_level} Risk"
    return (
        f'<span style="background:{c};color:white;padding:3px 10px;'
        f'border-radius:12px;font-size:12px;font-weight:600">{label}</span>'
    )


def page_header(title: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div style="background:linear-gradient(90deg,#1565c0,#0d47a1);
                    padding:20px 24px;border-radius:10px;margin-bottom:20px">
            <h2 style="color:white;margin:0;font-size:22px">{title}</h2>
            {"<p style='color:#bbdefb;margin:4px 0 0;font-size:13px'>"+subtitle+"</p>" if subtitle else ""}
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
