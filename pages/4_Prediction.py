"""
pages/4_Prediction.py
----------------------
Machine Learning model training, spatial grid predictions, and single-location prediction.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
from streamlit_folium import st_folium

from utils.helpers import (
    init_session_state, page_header, metric_card, no_data_warning,
    get_data_path, load_sample_data_cached, ensure_src_on_path, risk_badge,
    apply_custom_css,
)

ensure_src_on_path()

from src.model_training import train_model, load_model, predict_dataframe
from src.hotspot_prediction import predict_hotspots, predict_single_location, get_top_hotspots
from src.visualization import build_hotspot_prediction_map
from src.geospatial_analysis import get_nearby_crimes, location_risk_summary

st.set_page_config(page_title="ML Prediction & Risk Forecast | GeoCrime AI", page_icon="🔮", layout="wide")
init_session_state()
apply_custom_css()

page_header(
    title="Machine Learning Predictive Hotspot Engine",
    subtitle="Train Random Forest classifiers with spatial feature cross-validation, generate continuous grid risk heatmaps, or forecast any arbitrary point.",
    badge="ML Core Engine"
)

# Auto-load data
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
# Tab Layout
# ---------------------------------------------------------------------------
tab_train, tab_grid, tab_single = st.tabs(
    ["🤖 Train / Configure Classifier", "🗺 Urban Grid Risk Simulation", "📍 Arbitrary Point Forecaster"]
)

# ============================================================
# TAB 1 — Train Model
# ============================================================
with tab_train:
    st.markdown("#### 🤖 Random Forest Classifier Architecture")
    st.info(
        "The classifier models crime risk severity as a 3-class distribution (**High**, **Medium**, **Low**) "
        "leveraging spatial coordinates, periodic trigonometric cyclic encodings (hour & month), and crime density. "
        "Labels are calculated independently without data leakage.",
        icon="ℹ️",
    )

    col_train1, col_train2 = st.columns([2, 1])
    with col_train1:
        test_size = st.slider("Validation Holdout Ratio (%)", 10, 40, 25, 5)
        st.markdown(f"Training Partition: **{100 - test_size}%** | Testing Validation: **{test_size}%**")
    with col_train2:
        st.markdown("")
        load_existing = st.checkbox("Auto-restore persisted model weights", value=True)

    # Check for saved model
    if load_existing:
        model, scaler = load_model()
        if model is not None and not st.session_state.get("model_trained"):
            from src.model_training import load_train_results
            st.session_state["model"] = model
            st.session_state["scaler"] = scaler
            st.session_state["train_results"] = load_train_results()
            st.session_state["model_trained"] = True
            st.success("✅ Restored pre-trained model weights and artifacts from disk.")

    if st.button("🚀 Train / Re-fit Random Forest Model", type="primary"):
        with st.spinner("Training ensemble trees and running 5-fold cross validation…"):
            try:
                results = train_model(df, test_size=test_size / 100)
                st.session_state["model"] = results["model"]
                st.session_state["scaler"] = results["scaler"]
                st.session_state["train_results"] = results
                st.session_state["model_trained"] = True
                st.success("✅ Random Forest ensemble trained and validated successfully!")
            except Exception as e:
                st.error(f"❌ Training Exception: {e}")
                st.stop()

    if st.session_state.get("model_trained"):
        results = st.session_state.get("train_results")
        if results:
            st.write("")
            st.markdown("#### 📊 Evaluation Metrics (Out-of-Sample Test Set)")

            r1, r2, r3, r4, r5 = st.columns(5)
            with r1:
                metric_card("Accuracy",  f"{results['accuracy']:.1%}",  "Overall accuracy", "#2563eb", icon="🎯")
            with r2:
                metric_card("Precision", f"{results['precision']:.1%}", "Macro avg precision", "#0284c7", icon="📐")
            with r3:
                metric_card("Recall",    f"{results['recall']:.1%}",    "Macro avg recall", "#059669", icon="🔍")
            with r4:
                metric_card("F1 Score",  f"{results['f1_score']:.1%}",  "Harmonic mean F1", "#7c3aed", icon="⚖️")
            with r5:
                metric_card("CV F1 Score",
                            f"{results['cv_mean']:.1%}",
                            f"±{results['cv_std']:.3f} 5-fold CV", "#475569", icon="🔁")

            st.write("")
            # Class distribution
            with st.expander("📋 Detailed Stratified Breakdown & Classification Report", expanded=False):
                class_dist = results.get("class_distribution", {})
                lmap = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
                c_dist_cols = st.columns(3)
                for i, (k, v) in enumerate(class_dist.items()):
                    with c_dist_cols[i % 3]:
                        st.markdown(f"**{lmap.get(k, k)}**: `{v:,}` samples")

                report = results.get("classification_report", {})
                if report:
                    report_df = pd.DataFrame(report).T.round(3)
                    st.dataframe(report_df, use_container_width=True)
        else:
            st.success("✅ Model is initialized and ready.")


# ============================================================
# TAB 2 — Grid Prediction Map
# ============================================================
with tab_grid:
    st.markdown("#### 🗺 Continuous Spatial Risk Grid Simulation")

    if not st.session_state.get("model_trained"):
        st.warning("⚠️ Please train or restore the model in the 'Train Classifier' tab first.")
        st.stop()

    model  = st.session_state["model"]
    scaler = st.session_state["scaler"]

    # Bounding box from data
    lat_min = df["latitude"].min()
    lat_max = df["latitude"].max()
    lon_min = df["longitude"].min()
    lon_max = df["longitude"].max()

    with st.expander("⚙️ Spatial Extent & Grid Resolution", expanded=False):
        col_g1, col_g2, col_g3, col_g4 = st.columns(4)
        with col_g1:
            lat_min_inp = st.number_input("Lat Min", value=float(lat_min), format="%.4f")
        with col_g2:
            lat_max_inp = st.number_input("Lat Max", value=float(lat_max), format="%.4f")
        with col_g3:
            lon_min_inp = st.number_input("Lon Min", value=float(lon_min), format="%.4f")
        with col_g4:
            lon_max_inp = st.number_input("Lon Max", value=float(lon_max), format="%.4f")
        grid_step = st.select_slider(
            "Spatial Step Resolution (degrees)",
            options=[0.005, 0.01, 0.02, 0.03, 0.05],
            value=0.01,
        )

    show_actual = st.checkbox("Overlay recorded incidents as reference markers", value=False)

    if st.button("⚡ Generate Risk Grid Forecast", type="primary"):
        with st.spinner("Simulating multi-class inference over spatial grid…"):
            grid_df = predict_hotspots(
                model, scaler,
                lat_min=lat_min_inp, lat_max=lat_max_inp,
                lon_min=lon_min_inp, lon_max=lon_max_inp,
                step=grid_step, reference_df=df,
            )
            st.session_state["grid_prediction_df"] = grid_df
            st.session_state["grid_step"] = grid_step
            st.success(f"✅ Generated predictions across {len(grid_df):,} grid nodes.")

    grid_df = st.session_state.get("grid_prediction_df")
    if grid_df is not None:
        grid_step_val = st.session_state.get("grid_step", 0.01)

        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            metric_card("Grid Nodes", f"{len(grid_df):,}", "Evaluated coordinates", "#2563eb", icon="🌐")
        with k2:
            h_count = (grid_df["predicted_label"] == "High Risk").sum()
            metric_card("High Risk Cells", f"{h_count:,}", f"{(h_count/len(grid_df)*100):.1f}% of grid", "#dc2626", icon="🔥")
        with k3:
            m_count = (grid_df["predicted_label"] == "Medium Risk").sum()
            metric_card("Medium Risk Cells", f"{m_count:,}", f"{(m_count/len(grid_df)*100):.1f}% of grid", "#d97706", icon="⚡")
        with k4:
            l_count = (grid_df["predicted_label"] == "Low Risk").sum()
            metric_card("Low Risk Cells", f"{l_count:,}", f"{(l_count/len(grid_df)*100):.1f}% of grid", "#16a34a", icon="🛡️")

        st.write("")

        # Render Map
        with st.spinner("Rendering risk prediction polygons…"):
            pred_map = build_hotspot_prediction_map(
                grid_df,
                reference_df=df if show_actual else None,
                grid_step=grid_step_val,
            )
        st_folium(pred_map, width="100%", height=580, returned_objects=[])

        # Top Hotspots
        with st.expander("🔥 Highest Risk Simulated Coordinates", expanded=False):
            top_h = get_top_hotspots(grid_df, top_n=15)
            st.dataframe(top_h, use_container_width=True)


# ============================================================
# TAB 3 — Single Location Prediction
# ============================================================
with tab_single:
    st.markdown("#### 📍 Point Risk Forecaster & Historical Synthesis")

    if not st.session_state.get("model_trained"):
        st.warning("⚠️ Please train or restore the model in the 'Train Classifier' tab first.")
        st.stop()

    model  = st.session_state["model"]
    scaler = st.session_state["scaler"]

    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        pred_lat = st.number_input("Target Latitude",  value=float(df["latitude"].mean()), format="%.5f")
        pred_lon = st.number_input("Target Longitude", value=float(df["longitude"].mean()), format="%.5f")
    with col_in2:
        months_list = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        pred_month_name = st.selectbox("Forecast Month", months_list, index=5)
        pred_month = months_list.index(pred_month_name) + 1
        pred_hour = st.slider("Forecast Hour (0–23)", 0, 23, 14)
    with col_in3:
        all_crimes = sorted(df["crime_type"].unique().tolist())
        pred_crime = st.selectbox("Contextual Crime Type", all_crimes)
        pred_radius = st.slider("Historical Buffer Radius (km)", 0.2, 5.0, 1.0, 0.1)

    if st.button("🔮 Evaluate Coordinate Risk", type="primary"):
        with st.spinner("Synthesizing Random Forest inference & spatial radius historical stats…"):
            crime_cats = sorted(df["crime_type"].unique().tolist())
            crime_enc = crime_cats.index(pred_crime) if pred_crime in crime_cats else 0
            day_of_week = 2  # default Wednesday

            ml_result = predict_single_location(
                model, scaler,
                lat=pred_lat, lon=pred_lon,
                month=pred_month, hour=pred_hour,
                day_of_week=day_of_week,
                crime_type_encoded=crime_enc,
                reference_df=df,
            )
            nearby = get_nearby_crimes(df, pred_lat, pred_lon, radius_km=pred_radius)
            geo_result = location_risk_summary(nearby)
            st.session_state["single_prediction_results"] = {
                "ml_result": ml_result,
                "geo_result": geo_result,
                "nearby": nearby,
                "pred_lat": pred_lat,
                "pred_lon": pred_lon,
                "pred_month": pred_month,
                "pred_hour": pred_hour,
                "pred_crime": pred_crime,
                "pred_radius": pred_radius,
            }

    single_data = st.session_state.get("single_prediction_results")
    if single_data is not None:
        ml_result = single_data["ml_result"]
        geo_result = single_data["geo_result"]
        nearby = single_data["nearby"]
        p_lat = single_data["pred_lat"]
        p_lon = single_data["pred_lon"]
        p_month = single_data["pred_month"]
        p_hour = single_data["pred_hour"]
        p_crime = single_data["pred_crime"]
        p_radius = single_data["pred_radius"]

        st.write("")
        st.markdown("#### 📊 Synthesis Results & Comparative Assessment")

        res_col1, res_col2, res_col3 = st.columns(3)

        # ML Model result
        ml_risk = ml_result["risk_level"]
        ml_color = ml_result["color"]
        with res_col1:
            st.markdown(
                f"""
                <div style='background: white; border: 1px solid #e2e8f0; border-left: 5px solid {ml_color};
                            border-radius: 12px; padding: 18px 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); height: 100%;'>
                    <div style='font-size: 11px; color: #64748b; font-weight: 700; text-transform: uppercase;'>RANDOM FOREST INFERENCE</div>
                    <div style='font-size: 24px; font-weight: 800; color: {ml_color}; margin: 4px 0 8px 0;'>{ml_risk}</div>
                    <div style='font-size: 12.5px; color: #334155; line-height: 1.6;'>
                        Confidence: <b>{ml_result['confidence']:.0%}</b><br>
                        P(Low): <span style='color: #16a34a; font-weight: 600;'>{ml_result['probability_low']:.2f}</span> |
                        P(Med): <span style='color: #d97706; font-weight: 600;'>{ml_result['probability_medium']:.2f}</span> |
                        P(High): <span style='color: #dc2626; font-weight: 600;'>{ml_result['probability_high']:.2f}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Geospatial result
        geo_risk = geo_result["risk_level"]
        color_lookup = {
            "High Risk": "#dc2626", "High": "#dc2626",
            "Medium Risk": "#d97706", "Medium": "#d97706",
            "Low Risk": "#16a34a", "Low": "#16a34a"
        }
        geo_color = color_lookup.get(geo_risk, "#475569")
        with res_col2:
            st.markdown(
                f"""
                <div style='background: white; border: 1px solid #e2e8f0; border-left: 5px solid {geo_color};
                            border-radius: 12px; padding: 18px 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); height: 100%;'>
                    <div style='font-size: 11px; color: #64748b; font-weight: 700; text-transform: uppercase;'>HISTORICAL RADIAL ANALYSIS</div>
                    <div style='font-size: 24px; font-weight: 800; color: {geo_color}; margin: 4px 0 8px 0;'>{geo_risk}</div>
                    <div style='font-size: 12.5px; color: #334155; line-height: 1.6;'>
                        Confidence: <b>{geo_result['confidence']:.0%}</b><br>
                        Nearby Recorded Crimes: <b>{geo_result['crime_count']}</b><br>
                        Primary Incident Type: <b>{geo_result['top_crime']}</b><br>
                        Radial Buffer: <b>{p_radius} km</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Combined / Overall
        with res_col3:
            risk_scores = {
                "Low Risk": 0, "Low": 0,
                "Medium Risk": 1, "Medium": 1,
                "High Risk": 2, "High": 2
            }
            avg_score = (risk_scores.get(ml_risk, 1) + risk_scores.get(geo_risk, 1)) / 2.0
            if avg_score >= 1.5:
                final = "High Risk"; final_color = "#dc2626"
            elif avg_score >= 0.5:
                final = "Medium Risk"; final_color = "#d97706"
            else:
                final = "Low Risk"; final_color = "#16a34a"

            st.markdown(
                f"""
                <div style='background: white; border: 1px solid #e2e8f0; border-left: 5px solid {final_color};
                            border-radius: 12px; padding: 18px 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); height: 100%;'>
                    <div style='font-size: 11px; color: #64748b; font-weight: 700; text-transform: uppercase;'>COMBINED SYNTHESIS</div>
                    <div style='font-size: 24px; font-weight: 800; color: {final_color}; margin: 4px 0 8px 0;'>{final}</div>
                    <div style='font-size: 12.5px; color: #334155; line-height: 1.6;'>
                        Target: <b>({p_lat:.4f}, {p_lon:.4f})</b><br>
                        Context: {months_list[p_month-1]} @ {p_hour:02d}:00<br>
                        Crime Modality: <b>{p_crime}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Nearby crime breakdown
        if geo_result.get("crime_breakdown"):
            st.write("")
            st.markdown(f"##### 📋 Incident Distribution within {p_radius} km Buffer")
            breakdown = pd.DataFrame(
                list(geo_result["crime_breakdown"].items()),
                columns=["Crime Type", "Count"]
            ).sort_values("Count", ascending=False)

            import plotly.express as px
            fig = px.bar(
                breakdown, x="Count", y="Crime Type", orientation="h",
                color="Count", color_continuous_scale="Reds",
                height=260, template="plotly_white",
            )
            fig.update_layout(showlegend=False, plot_bgcolor="rgba(248,250,252,0.6)", paper_bgcolor="white",
                              margin=dict(l=0, r=10, t=20, b=0))
            st.plotly_chart(fig, use_container_width=True)

        # Mini map
        st.write("")
        st.markdown("##### 🗺 Spatial Vicinity Map")
        from src.visualization import build_crime_map, add_location_marker
        nearby_display = nearby.head(100) if len(nearby) > 0 else df.sample(min(50, len(df)))
        mini_map = build_crime_map(
            df=nearby_display, show_heatmap=True,
            show_clusters=False, show_markers=True,
        )
        mini_map = add_location_marker(mini_map, p_lat, p_lon, ml_result)
        mini_map.location = [p_lat, p_lon]
        mini_map.zoom_start = 13
        st_folium(mini_map, width="100%", height=420, returned_objects=[])

st.caption("⚠️ Predictions are computed over synthetic training features — strictly for academic evaluation.")
