"""
pages/5_Model_Performance.py
-----------------------------
Model evaluation metrics, confusion matrix, feature importance, and report download.
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
    df_to_csv_bytes, generate_report_text,
)

ensure_src_on_path()

from src.visualization import (
    plot_confusion_matrix, plot_feature_importance,
    plot_risk_distribution, plot_crime_by_category,
)

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")
init_session_state()

page_header("Model Performance")

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
        "⚠️ Model has not been trained yet. Please click the button below "
        "or go to the **Prediction** page to train the model.",
        icon="🤖",
    )

    # Quick train button
    if st.button("🚀 Train Model Now", type="primary"):
        from src.model_training import train_model
        with st.spinner("Training model…"):
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
st.markdown("#### 📊 Model Performance Overview")

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    metric_card("Accuracy",  f"{results['accuracy']:.1%}",  "Overall correct predictions", "#1565c0")
with c2:
    metric_card("Precision", f"{results['precision']:.1%}", "Weighted avg precision", "#1976d2")
with c3:
    metric_card("Recall",    f"{results['recall']:.1%}",    "Weighted avg recall", "#0288d1")
with c4:
    metric_card("F1 Score",  f"{results['f1_score']:.1%}",  "Weighted avg F1", "#00838f")
with c5:
    metric_card("CV F1",
                f"{results['cv_mean']:.1%}",
                f"5-fold, ±{results['cv_std']:.3f}", "#37474f")

st.markdown("---")

# ---------------------------------------------------------------------------
# Confusion Matrix + Feature Importance
# ---------------------------------------------------------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("##### 🔢 Confusion Matrix")
    cm = results.get("confusion_matrix")
    if cm is not None:
        fig_cm = plot_confusion_matrix(cm)
        st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.info("Confusion matrix not available.")

with col_right:
    st.markdown("##### 🌳 Feature Importances")
    importances = results.get("feature_importances", {})
    if importances:
        fig_fi = plot_feature_importance(importances)
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.info("Feature importances not available.")

# ---------------------------------------------------------------------------
# Classification Report Table
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("#### 📋 Classification Report")

report = results.get("classification_report", {})
if report:
    # Remove averages for clarity in display
    main_classes = {k: v for k, v in report.items()
                    if k not in ("accuracy", "macro avg", "weighted avg")}
    avgs = {k: v for k, v in report.items()
            if k in ("macro avg", "weighted avg")}

    col_rep1, col_rep2 = st.columns(2)
    with col_rep1:
        st.markdown("**Per-class Metrics**")
        df_report = pd.DataFrame(main_classes).T.round(3)
        st.dataframe(df_report, use_container_width=True)
    with col_rep2:
        st.markdown("**Aggregate Metrics**")
        df_avgs = pd.DataFrame(avgs).T.round(3)
        st.dataframe(df_avgs, use_container_width=True)

# ---------------------------------------------------------------------------
# Cross-Validation Details
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("#### 🔄 Cross-Validation Results")

cv_scores = results.get("cv_scores", [])
if len(cv_scores) > 0:
    fold_df = pd.DataFrame({
        "Fold": [f"Fold {i+1}" for i in range(len(cv_scores))],
        "F1 Score": cv_scores,
    })
    fig_cv = px.bar(
        fold_df, x="Fold", y="F1 Score",
        color="F1 Score", color_continuous_scale="Greens",
        title="5-Fold Cross-Validation F1 Scores",
        text="F1 Score", height=340,
    )
    fig_cv.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_cv.update_layout(showlegend=False, plot_bgcolor="white",
                         margin=dict(l=10, r=10, t=40, b=10),
                         yaxis=dict(range=[0, 1.05]))
    # Add mean line
    fig_cv.add_hline(
        y=results["cv_mean"], line_dash="dash", line_color="red",
        annotation_text=f"Mean = {results['cv_mean']:.3f}",
        annotation_position="top right",
    )
    st.plotly_chart(fig_cv, use_container_width=True)

st.info(
    f"📊 Mean CV F1: **{results['cv_mean']:.4f}** | "
    f"Std: **{results['cv_std']:.4f}** | "
    f"Range: [{min(cv_scores):.3f} – {max(cv_scores):.3f}]"
    if len(cv_scores) > 0 else "CV scores unavailable"
)

# ---------------------------------------------------------------------------
# Training Summary
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("#### ℹ️ Training Configuration")

train_col1, train_col2 = st.columns(2)
with train_col1:
    st.markdown(f"""
    | Parameter | Value |
    |-----------|-------|
    | Algorithm | Random Forest |
    | N Estimators | 200 |
    | Max Depth | 12 |
    | Class Weight | Balanced |
    | Training Samples | {results.get('train_size', 'N/A'):,} |
    | Testing Samples | {results.get('test_size', 'N/A'):,} |
    """)
with train_col2:
    class_dist = results.get("class_distribution", {})
    lmap = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
    dist_data = [{"Risk Level": lmap.get(k, k), "Count": v}
                 for k, v in sorted(class_dist.items())]
    if dist_data:
        fig_dist = px.pie(
            pd.DataFrame(dist_data), names="Risk Level", values="Count",
            hole=0.4, height=280,
            color="Risk Level",
            color_discrete_map={"High Risk": "#d32f2f",
                                "Medium Risk": "#f57c00",
                                "Low Risk": "#388e3c"},
            title="Training Label Distribution",
        )
        fig_dist.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_dist, use_container_width=True)

# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("#### ⬇️ Download Reports")

dl_col1, dl_col2, dl_col3 = st.columns(3)

# Full text report
with dl_col1:
    report_text = generate_report_text(df, results)
    st.download_button(
        "📄 Download Analysis Report (TXT)",
        data=report_text,
        file_name="crime_hotspot_report.txt",
        mime="text/plain",
    )

# Dataset with predictions
with dl_col2:
    if st.session_state.get("model_trained"):
        from src.model_training import predict_dataframe
        model  = st.session_state["model"]
        scaler = st.session_state["scaler"]
        df_pred = predict_dataframe(df, model, scaler)
        st.download_button(
            "📊 Download Dataset with Predictions (CSV)",
            data=df_to_csv_bytes(df_pred),
            file_name="crime_data_with_predictions.csv",
            mime="text/csv",
        )
    else:
        st.button("📊 Train model first", disabled=True)

# Grid predictions
with dl_col3:
    grid_df = st.session_state.get("grid_predictions")
    if grid_df is not None:
        st.download_button(
            "🗺 Download Grid Predictions (CSV)",
            data=df_to_csv_bytes(grid_df),
            file_name="grid_predictions.csv",
            mime="text/csv",
        )
    else:
        st.button("🗺 Generate grid predictions first", disabled=True)

st.caption("⚠️ All metrics are from a model trained on SYNTHETIC data — for academic use only.")
