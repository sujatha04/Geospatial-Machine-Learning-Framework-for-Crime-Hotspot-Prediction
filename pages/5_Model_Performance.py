"""
pages/5_Model_Performance.py
-----------------------------
Model evaluation metrics, confusion matrix, feature importance, and audit report export.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.helpers import (
    init_session_state, page_header, metric_card, no_data_warning,
    get_data_path, load_sample_data_cached, ensure_src_on_path,
    df_to_csv_bytes, generate_report_text, apply_custom_css,
)

ensure_src_on_path()

from src.visualization import (
    plot_confusion_matrix, plot_feature_importance,
    plot_risk_distribution, plot_crime_by_category,
)

st.set_page_config(page_title="Model Performance & Diagnostics | GeoCrime AI", page_icon="📈", layout="wide")
init_session_state()
apply_custom_css()

page_header(
    title="Model Diagnostics & Performance Evaluation",
    subtitle="Evaluate multi-class Random Forest metrics, confusion matrices, Gini impurity feature rankings, and generate security audit reports.",
    badge="Diagnostic Studio"
)

# Auto-load data
df = st.session_state.get("df_processed")
if df is None:
    sample_path = get_data_path("sample_crime_data.csv")
    if os.path.exists(sample_path):
        df, _ = load_sample_data_cached(sample_path)
        st.session_state["df_processed"] = df
    else:
        no_data_warning()
        st.stop()

results = st.session_state.get("train_results")

# If results are not in session state, check if saved on disk
if results is None:
    from src.model_training import load_train_results, load_model
    saved_results = load_train_results()
    if saved_results is not None:
        saved_model, saved_scaler = load_model()
        if saved_model is not None:
            st.session_state["model"] = saved_model
            st.session_state["scaler"] = saved_scaler
            st.session_state["train_results"] = saved_results
            st.session_state["model_trained"] = True
            results = saved_results

if not st.session_state.get("model_trained") or results is None:
    st.warning(
        "⚠️ Model weights have not been trained yet in this session. "
        "Initialize model training below to compute diagnostic metrics.",
        icon="🤖",
    )

    # Quick train button
    if st.button("🚀 Train Model Now", type="primary"):
        from src.model_training import train_model
        with st.spinner("Training Random Forest ensemble…"):
            results = train_model(df, test_size=0.25)
            st.session_state["model"] = results["model"]
            st.session_state["scaler"] = results["scaler"]
            st.session_state["train_results"] = results
            st.session_state["model_trained"] = True
            st.success("✅ Model trained!")
            st.rerun()
    st.stop()

# ---------------------------------------------------------------------------
# Overview Metrics
# ---------------------------------------------------------------------------
st.markdown("#### 📊 Core Diagnostic Metrics (Holdout Test Set)")

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    metric_card("Accuracy",  f"{results['accuracy']:.1%}",  "Overall correct classifications", "#2563eb", icon="🎯")
with c2:
    metric_card("Precision", f"{results['precision']:.1%}", "Macro weighted precision", "#0284c7", icon="📐")
with c3:
    metric_card("Recall",    f"{results['recall']:.1%}",    "Macro weighted recall", "#059669", icon="🔍")
with c4:
    metric_card("F1 Score",  f"{results['f1_score']:.1%}",  "Balanced harmonic mean", "#7c3aed", icon="⚖️")
with c5:
    metric_card("CV F1 Score",
                f"{results['cv_mean']:.1%}",
                f"5-fold CV (±{results['cv_std']:.3f})", "#475569", icon="🔁")

st.write("")

# ---------------------------------------------------------------------------
# Confusion Matrix + Feature Importance
# ---------------------------------------------------------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("##### 🔢 Multi-Class Confusion Matrix")
    cm = results.get("confusion_matrix")
    if cm is not None:
        fig_cm = plot_confusion_matrix(cm)
        st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.info("Confusion matrix not available.")

with col_right:
    st.markdown("##### 🌳 Gini Feature Importance Ranking")
    importances = results.get("feature_importances", {})
    if importances:
        fig_fi = plot_feature_importance(importances)
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("Feature importances not available.")

st.write("")

# ---------------------------------------------------------------------------
# Classification Report Table
# ---------------------------------------------------------------------------
st.markdown("#### 📋 Stratified Classification Report")

report = results.get("classification_report", {})
if report:
    main_classes = {k: v for k, v in report.items()
                    if k not in ("accuracy", "macro avg", "weighted avg")}
    avgs = {k: v for k, v in report.items()
            if k in ("macro avg", "weighted avg")}

    col_rep1, col_rep2 = st.columns(2)
    with col_rep1:
        st.markdown("**Per-Class Breakdown**")
        df_report = pd.DataFrame(main_classes).T.round(3)
        st.dataframe(df_report, use_container_width=True)
    with col_rep2:
        st.markdown("**Macro & Weighted Aggregates**")
        df_avgs = pd.DataFrame(avgs).T.round(3)
        st.dataframe(df_avgs, use_container_width=True)

st.write("")

# ---------------------------------------------------------------------------
# Cross-Validation Details
# ---------------------------------------------------------------------------
st.markdown("#### 🔄 5-Fold Stratified Cross-Validation")

cv_scores = results.get("cv_scores", [])
if len(cv_scores) > 0:
    fold_df = pd.DataFrame({
        "Fold": [f"Fold {i+1}" for i in range(len(cv_scores))],
        "F1 Score": cv_scores,
    })
    fig_cv = px.bar(
        fold_df, x="Fold", y="F1 Score",
        color="F1 Score", color_continuous_scale="Blues",
        title="Cross-Validation F1 Scores Across Folds",
        text="F1 Score", height=320,
        template="plotly_white",
    )
    fig_cv.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_cv.update_layout(
        showlegend=False,
        plot_bgcolor="rgba(248,250,252,0.6)",
        paper_bgcolor="white",
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(range=[0, 1.1], gridcolor="#e2e8f0"),
        xaxis=dict(gridcolor="#e2e8f0")
    )
    fig_cv.add_hline(
        y=results["cv_mean"], line_dash="dash", line_color="#ef4444",
        annotation_text=f"Mean F1 = {results['cv_mean']:.3f}",
        annotation_position="top right",
    )
    st.plotly_chart(fig_cv, use_container_width=True)

st.write("")

# ---------------------------------------------------------------------------
# Export Full Audit Report
# ---------------------------------------------------------------------------
st.markdown("### 📄 Audit Report & Artifact Export")
rep_col1, rep_col2 = st.columns([3, 1])
with rep_col1:
    st.markdown("Generate and export a comprehensive plain-text diagnostic summary report with all evaluation metrics, category breakdowns, and system configurations.")
with rep_col2:
    report_text = generate_report_text(df, results)
    st.download_button(
        label="⬇️ Export Audit Report (.txt)",
        data=report_text,
        file_name="geocrime_model_audit_report.txt",
        mime="text/plain",
    )

st.write("")
st.caption("⚠️ Metrics and models are benchmarked over synthetic demonstration data for academic research.")
