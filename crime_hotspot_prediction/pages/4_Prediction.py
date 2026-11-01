"""
pages/4_Prediction.py
----------------------
Machine Learning model training, grid predictions, and single-location prediction.
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
)

ensure_src_on_path()

from src.model_training import train_model, load_model, predict_dataframe
from src.hotspot_prediction import predict_hotspots, predict_single_location, get_top_hotspots
from src.visualization import build_hotspot_prediction_map
from src.geospatial_analysis import get_nearby_crimes, location_risk_summary

st.set_page_config(page_title="Prediction", page_icon="🔮", layout="wide")
init_session_state()

page_header("🔮 Crime Hotspot Prediction")

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
    ["🤖 Train Model", "🗺 Grid Prediction Map", "📍 Single Location Prediction"]
)

# ============================================================
# TAB 1 — Train Model
# ============================================================
with tab_train:
    st.markdown("#### 🤖 Random Forest Classifier Training")
    st.info(
        "The model is trained to classify spatial grid cells as **High / Medium / Low** risk "
        "based on crime density. Features include spatial coordinates, temporal cyclical encodings, "
        "and crime category. No data leakage — labels are computed independently from the training split.",
        icon="ℹ️",
    )

    col_train1, col_train2 = st.columns([2, 1])
    with col_train1:
        test_size = st.slider("Test Set Size (%)", 10, 40, 25, 5)
        st.markdown(f"Training: **{100 - test_size}%** | Testing: **{test_size}%**")
    with col_train2:
        st.markdown("")
        load_existing = st.checkbox("Load existing model (if available)", value=True)

    # Check for saved model
    if load_existing:
        model, scaler = load_model()
        if model is not None and not st.session_state.get("model_trained"):
            from src.model_training import load_train_results
            st.session_state["model"] = model
            st.session_state["scaler"] = scaler
            st.session_state["train_results"] = load_train_results()
            st.session_state["model_trained"] = True
            st.success("✅ Loaded saved model from disk.")

    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training Random Forest… this may take 20–40 seconds…"):
            try:
                results = train_model(df, test_size=test_size / 100)
                st.session_state["model"] = results["model"]
                st.session_state["scaler"] = results["scaler"]
                st.session_state["train_results"] = results
                st.session_state["model_trained"] = True
                st.success("✅ Model trained successfully!")
            except Exception as e:
                st.error(f"❌ Training failed: {e}")
                st.stop()

    if st.session_state.get("model_trained"):
        results = st.session_state.get("train_results")
        if results:
            st.markdown("---")
            st.markdown("#### 📊 Training Results")

            r1, r2, r3, r4, r5 = st.columns(5)
            with r1:
                metric_card("Accuracy",  f"{results['accuracy']:.1%}",  "", "#1565c0")
            with r2:
                metric_card("Precision", f"{results['precision']:.1%}", "", "#1976d2")
            with r3:
                metric_card("Recall",    f"{results['recall']:.1%}",    "", "#0288d1")
            with r4:
                metric_card("F1 Score",  f"{results['f1_score']:.1%}",  "", "#00838f")
            with r5:
                metric_card("CV F1",
                            f"{results['cv_mean']:.1%} ±{results['cv_std']:.3f}",
                            "5-fold CV", "#37474f")

            st.markdown(f"""
            - **Training set**: {results['train_size']:,} samples
            - **Testing set**: {results['test_size']:,} samples
            - **Cross-validation**: 5-fold stratified CV on training set
            """)

            # Class distribution
            with st.expander("📋 Class Distribution & Report", expanded=False):
                class_dist = results.get("class_distribution", {})
                lmap = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
                for k, v in class_dist.items():
                    st.write(f"- **{lmap.get(k,k)}**: {v} records")

                report = results.get("classification_report", {})
                if report:
                    report_df = pd.DataFrame(report).T.round(3)
                    st.dataframe(report_df, use_container_width=True)
        else:
            st.success("✅ Model is ready. Train again to see metrics.")


# ============================================================
# TAB 2 — Grid Prediction Map
# ============================================================
with tab_grid:
    st.markdown("#### 🗺 Predicted Risk Zone Map")

    if not st.session_state.get("model_trained"):
        st.warning("⚠️ Please train the model first (Train Model tab).")
        st.stop()

    model  = st.session_state["model"]
    scaler = st.session_state["scaler"]

    # Bounding box from data
    lat_min = df["latitude"].min()
    lat_max = df["latitude"].max()
    lon_min = df["longitude"].min()
    lon_max = df["longitude"].max()

    with st.expander("⚙️ Grid Settings", expanded=False):
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
            "Grid Step (degrees)",
            options=[0.005, 0.01, 0.02, 0.03, 0.05],
            value=0.01,
        )

    show_actual = st.checkbox("Overlay actual crime locations (as blue markers)", value=False)

    if st.button("Generate Prediction Map", type="primary"):
        with st.spinner("Predicting across spatial grid…"):
            grid_df = predict_hotspots(
                model, scaler,
                lat_min=lat_min_inp, lat_max=lat_max_inp,
                lon_min=lon_min_inp, lon_max=lon_max_inp,
                step=grid_step, reference_df=df,
            )
            st.session_state["grid_predictions"] = grid_df

    grid_df = st.session_state.get("grid_predictions")
    if grid_df is not None and len(grid_df) > 0:
        # KPI
        g1, g2, g3, g4 = st.columns(4)
        with g1:
            metric_card("Grid Cells", f"{len(grid_df):,}", "", "#1565c0")
        with g2:
            metric_card("High-Risk Cells", f"{(grid_df['predicted_risk']==2).sum():,}", "", "#c62828")
        with g3:
            metric_card("Medium-Risk Cells", f"{(grid_df['predicted_risk']==1).sum():,}", "", "#e65100")
        with g4:
            metric_card("Low-Risk Cells", f"{(grid_df['predicted_risk']==0).sum():,}", "", "#2e7d32")

        st.markdown("##### 🗺 Predicted Hotspot Map")
        st.caption("Red = High Risk | Orange = Medium Risk | Green = Low Risk")
        with st.spinner("Rendering map…"):
            pred_map = build_hotspot_prediction_map(
                grid_df=grid_df,
                actual_df=df if show_actual else None,
                center_lat=df["latitude"].mean(),
                center_lon=df["longitude"].mean(),
            )
        st_folium(pred_map, width="100%", height=580, returned_objects=[])

        # Top hotspots table
        st.markdown("##### 🔥 Top High-Risk Zones")
        top = get_top_hotspots(grid_df, top_n=10)
        top.index = range(1, len(top) + 1)
        st.dataframe(top, use_container_width=True)

        # Download
        from utils.helpers import df_to_csv_bytes
        st.download_button(
            "⬇️ Download Grid Predictions (CSV)",
            data=df_to_csv_bytes(grid_df),
            file_name="hotspot_predictions.csv",
            mime="text/csv",
        )
    else:
        st.info("Click **Generate Prediction Map** to run predictions.")


# ============================================================
# TAB 3 — Single Location Prediction
# ============================================================
with tab_single:
    st.markdown("#### 📍 Single Location Risk Prediction")
    st.markdown(
        "Enter a latitude/longitude to get the predicted risk level from the trained model, "
        "combined with nearby historical crime analysis."
    )

    if not st.session_state.get("model_trained"):
        st.warning("⚠️ Please train the model first (Train Model tab).")
        st.stop()

    model  = st.session_state["model"]
    scaler = st.session_state["scaler"]

    col_in1, col_in2 = st.columns(2)
    with col_in1:
        pred_lat = st.number_input("Latitude",  value=float(df["latitude"].mean()),
                                   format="%.5f", key="pred_lat")
        pred_month = st.selectbox(
            "Month", range(1, 13),
            format_func=lambda m: ["Jan","Feb","Mar","Apr","May","Jun",
                                    "Jul","Aug","Sep","Oct","Nov","Dec"][m-1],
            index=5, key="pred_month",
        )
        pred_hour = st.slider("Hour of Day", 0, 23, 14, key="pred_hour")
    with col_in2:
        pred_lon = st.number_input("Longitude", value=float(df["longitude"].mean()),
                                   format="%.5f", key="pred_lon")
        crime_types_list = sorted(df["crime_type"].unique().tolist())
        pred_crime = st.selectbox("Crime Type (context)", crime_types_list, key="pred_crime")
        pred_radius = st.slider("Historical Search Radius (km)", 0.5, 5.0, 1.0, 0.5,
                                key="pred_radius")

    if st.button("🔮 Predict Risk Level", type="primary"):
        # Encode crime type
        type_to_int = {t: i for i, t in enumerate(sorted(df["crime_type"].unique()))}
        crime_enc = type_to_int.get(pred_crime, 0)

        day_of_week = 2  # Wednesday default

        with st.spinner("Running prediction…"):
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

        st.markdown("---")
        st.markdown("#### 📊 Prediction Results")

        res_col1, res_col2, res_col3 = st.columns(3)

        # ML Model result
        ml_risk = ml_result["risk_level"]
        ml_color = ml_result["color"]
        with res_col1:
            st.markdown(
                f"""
                <div style='background:{ml_color}22;border-left:4px solid {ml_color};
                            border-radius:8px;padding:16px 20px;height:100%'>
                    <div style='font-size:11px;color:#555;font-weight:600'>ML MODEL PREDICTION</div>
                    <div style='font-size:22px;font-weight:700;color:{ml_color}'>{ml_risk}</div>
                    <div style='font-size:12px;color:#333;margin-top:4px'>
                        Confidence: <b>{ml_result['confidence']:.0%}</b><br>
                        P(Low): {ml_result['probability_low']:.2f}<br>
                        P(Med): {ml_result['probability_medium']:.2f}<br>
                        P(High): {ml_result['probability_high']:.2f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Geospatial result
        geo_risk = geo_result["risk_level"]
        color_lookup = {
            "High Risk": "#d32f2f", "High": "#d32f2f",
            "Medium Risk": "#e65100", "Medium": "#e65100",
            "Low Risk": "#2e7d32", "Low": "#2e7d32"
        }
        geo_color = color_lookup.get(geo_risk, "#555")
        with res_col2:
            st.markdown(
                f"""
                <div style='background:{geo_color}22;border-left:4px solid {geo_color};
                            border-radius:8px;padding:16px 20px;height:100%'>
                    <div style='font-size:11px;color:#555;font-weight:600'>HISTORICAL ANALYSIS</div>
                    <div style='font-size:22px;font-weight:700;color:{geo_color}'>{geo_risk}</div>
                    <div style='font-size:12px;color:#333;margin-top:4px'>
                        Confidence: <b>{geo_result['confidence']:.0%}</b><br>
                        Nearby Crimes: <b>{geo_result['crime_count']}</b><br>
                        Top Crime: <b>{geo_result['top_crime']}</b><br>
                        Radius: <b>{p_radius} km</b>
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
                final = "High Risk"; final_color = "#d32f2f"
            elif avg_score >= 0.5:
                final = "Medium Risk"; final_color = "#e65100"
            else:
                final = "Low Risk"; final_color = "#2e7d32"

            st.markdown(
                f"""
                <div style='background:{final_color}22;border-left:4px solid {final_color};
                            border-radius:8px;padding:16px 20px;height:100%'>
                    <div style='font-size:11px;color:#555;font-weight:600'>COMBINED ASSESSMENT</div>
                    <div style='font-size:22px;font-weight:700;color:{final_color}'>{final}</div>
                    <div style='font-size:12px;color:#333;margin-top:4px'>
                        Location: ({p_lat:.4f}, {p_lon:.4f})<br>
                        Month: {["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][p_month-1]}<br>
                        Hour: {p_hour:02d}:00<br>
                        Crime Context: {p_crime}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Nearby crime breakdown
        if geo_result.get("crime_breakdown"):
            st.markdown("---")
            st.markdown(f"##### 📋 Nearby Crime Breakdown ({p_radius} km radius)")
            breakdown = pd.DataFrame(
                list(geo_result["crime_breakdown"].items()),
                columns=["Crime Type", "Count"]
            ).sort_values("Count", ascending=False)

            import plotly.express as px
            fig = px.bar(
                breakdown, x="Count", y="Crime Type", orientation="h",
                color="Count", color_continuous_scale="Reds",
                height=300, title=f"Crimes within {p_radius} km",
            )
            fig.update_layout(showlegend=False, plot_bgcolor="white",
                              margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

        # Mini map
        st.markdown("---")
        st.markdown("##### 🗺 Location Preview")
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

    st.caption("⚠️ Predictions are based on SYNTHETIC training data — for academic use only.")
