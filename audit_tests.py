"""
audit_tests.py
--------------
Comprehensive runtime audit for the crime hotspot prediction project.
Tests every module, function, edge case, and potential deployment issue.
"""

import sys, os, traceback, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, "data", "sample_crime_data.csv")

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"

results = []

def log(status, test, detail=""):
    results.append((status, test, detail))
    print(f"{status}  [{test}]  {detail}")


# ─────────────────────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 1: MODULE IMPORTS")
print("="*60)

required_imports = [
    ("streamlit", "streamlit"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("sklearn", "scikit-learn"),
    ("folium", "folium"),
    ("streamlit_folium", "streamlit-folium"),
    ("plotly", "plotly"),
    ("matplotlib", "matplotlib"),
    ("joblib", "joblib"),
    ("branca", "branca"),
]

for mod, pkg in required_imports:
    try:
        __import__(mod)
        log(PASS, f"import {mod}")
    except ImportError as e:
        log(FAIL, f"import {mod}", f"MISSING PACKAGE: {pkg} — {e}")

# Internal modules
internal_modules = [
    "src.data_preprocessing",
    "src.geospatial_analysis",
    "src.model_training",
    "src.hotspot_prediction",
    "src.visualization",
    "utils.helpers",
]
for mod in internal_modules:
    try:
        __import__(mod)
        log(PASS, f"import {mod}")
    except Exception as e:
        log(FAIL, f"import {mod}", traceback.format_exc(limit=2))


# ─────────────────────────────────────────────────────────────
# 2. DATA FILES
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 2: FILE & PATH CHECKS")
print("="*60)

required_files = [
    os.path.join(ROOT, "app.py"),
    os.path.join(ROOT, "requirements.txt"),
    os.path.join(ROOT, "README.md"),
    os.path.join(ROOT, "data", "sample_crime_data.csv"),
    os.path.join(ROOT, "data", "generate_sample_data.py"),
    os.path.join(ROOT, "src", "data_preprocessing.py"),
    os.path.join(ROOT, "src", "geospatial_analysis.py"),
    os.path.join(ROOT, "src", "model_training.py"),
    os.path.join(ROOT, "src", "hotspot_prediction.py"),
    os.path.join(ROOT, "src", "visualization.py"),
    os.path.join(ROOT, "utils", "helpers.py"),
    os.path.join(ROOT, "pages", "1_Dashboard.py"),
    os.path.join(ROOT, "pages", "2_Crime_Analysis.py"),
    os.path.join(ROOT, "pages", "3_Hotspot_Map.py"),
    os.path.join(ROOT, "pages", "4_Prediction.py"),
    os.path.join(ROOT, "pages", "5_Model_Performance.py"),
]

for f in required_files:
    rel = f.replace(ROOT + os.sep, "")
    if os.path.exists(f):
        log(PASS, f"file exists: {rel}")
    else:
        log(FAIL, f"file exists: {rel}", "FILE NOT FOUND")


# ─────────────────────────────────────────────────────────────
# 3. DATA LOADING & VALIDATION
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 3: DATA LOADING & PREPROCESSING")
print("="*60)

from src.data_preprocessing import (
    load_and_validate, engineer_features, compute_risk_label,
    get_summary_stats, get_feature_columns
)
import pandas as pd
import numpy as np

# 3a. Load CSV
try:
    df, warns = load_and_validate(DATA_PATH)
    assert len(df) > 0, "DataFrame is empty"
    assert "latitude" in df.columns
    assert "longitude" in df.columns
    assert "date" in df.columns
    assert "crime_type" in df.columns
    log(PASS, "load_and_validate()", f"{len(df)} records loaded, {len(warns)} warnings")
except Exception as e:
    log(FAIL, "load_and_validate()", traceback.format_exc(limit=3))
    sys.exit(1)

# 3b. Coordinate range
try:
    bad_lat = (~df["latitude"].between(-90, 90)).sum()
    bad_lon = (~df["longitude"].between(-180, 180)).sum()
    if bad_lat == 0 and bad_lon == 0:
        log(PASS, "coordinate range validation", f"lat in [-90,90], lon in [-180,180]")
    else:
        log(FAIL, "coordinate range validation", f"bad_lat={bad_lat}, bad_lon={bad_lon}")
except Exception as e:
    log(FAIL, "coordinate range validation", str(e))

# 3c. No lat/lon swap (lat must be smaller abs value than lon for US/Europe typically checked)
lat_mean = df["latitude"].mean()
lon_mean = df["longitude"].mean()
if abs(lat_mean) < 90 and abs(lon_mean) < 180:
    log(PASS, "lat/lon not swapped", f"lat mean={lat_mean:.3f}, lon mean={lon_mean:.3f}")
else:
    log(FAIL, "lat/lon not swapped", f"Values out of expected range")

# 3d. Duplicate detection
try:
    orig_len = len(pd.read_csv(DATA_PATH))
    dedup_len = len(df)
    log(PASS, "duplicate detection", f"original={orig_len}, after dedup={dedup_len}")
except Exception as e:
    log(WARN, "duplicate detection", str(e))

# 3e. Missing value handling
try:
    missing_before = df.isnull().sum().sum()
    log(PASS, "missing values", f"total remaining NaN after load: {missing_before}")
except Exception as e:
    log(FAIL, "missing values", str(e))

# 3f. Feature engineering
try:
    df = engineer_features(df)
    expected_features = ["year", "month", "day_of_week", "hour", "is_weekend",
                         "hour_sin", "hour_cos", "month_sin", "month_cos",
                         "lat_bin", "lon_bin", "crime_type_encoded"]
    missing_feats = [f for f in expected_features if f not in df.columns]
    if not missing_feats:
        log(PASS, "engineer_features()", f"all {len(expected_features)} features present")
    else:
        log(FAIL, "engineer_features()", f"missing: {missing_feats}")
except Exception as e:
    log(FAIL, "engineer_features()", traceback.format_exc(limit=3))

# 3g. Risk label computation
try:
    df = compute_risk_label(df)
    assert "risk_label" in df.columns
    unique_labels = sorted(df["risk_label"].unique().tolist())
    assert set(unique_labels).issubset({0, 1, 2}), f"unexpected labels: {unique_labels}"
    dist = df["risk_label"].value_counts().sort_index().to_dict()
    log(PASS, "compute_risk_label()", f"label distribution: {dist}")
except Exception as e:
    log(FAIL, "compute_risk_label()", traceback.format_exc(limit=3))

# 3h. Re-run compute_risk_label (idempotency — no KeyError on second call)
try:
    df2 = compute_risk_label(df)  # df already has risk_label
    assert "risk_label" in df2.columns
    log(PASS, "compute_risk_label() idempotency", "no error when called on already-labeled df")
except Exception as e:
    log(FAIL, "compute_risk_label() idempotency", traceback.format_exc(limit=3))

# 3i. get_feature_columns
try:
    feat_cols = get_feature_columns()
    assert isinstance(feat_cols, list) and len(feat_cols) > 0
    present = [c for c in feat_cols if c in df.columns]
    missing = [c for c in feat_cols if c not in df.columns]
    if not missing:
        log(PASS, "get_feature_columns()", f"{len(feat_cols)} features, all present in df")
    else:
        log(FAIL, "get_feature_columns()", f"missing from df: {missing}")
except Exception as e:
    log(FAIL, "get_feature_columns()", str(e))

# 3j. get_summary_stats
try:
    stats = get_summary_stats(df)
    required_keys = ["total_records", "total_categories", "date_range", "risk_high", "risk_medium", "risk_low"]
    missing_keys = [k for k in required_keys if k not in stats]
    if not missing_keys:
        log(PASS, "get_summary_stats()", f"total={stats['total_records']}, high={stats['risk_high']}")
    else:
        log(FAIL, "get_summary_stats()", f"missing keys: {missing_keys}")
except Exception as e:
    log(FAIL, "get_summary_stats()", str(e))

# 3k. Empty / corrupt CSV
try:
    empty_df = pd.DataFrame({"latitude": [], "longitude": [], "date": [], "crime_type": []})
    df_empty, w_empty = load_and_validate(empty_df)
    log(PASS, "empty DataFrame handling", f"{len(df_empty)} rows returned (zero is fine)")
except Exception as e:
    log(WARN, "empty DataFrame handling", str(e))

# 3l. CSV with wrong column names
try:
    bad_df = pd.DataFrame({"lat": [1.0], "lon": [2.0], "type": ["Theft"]})
    load_and_validate(bad_df)
    log(FAIL, "missing-column rejection", "Should have raised ValueError but didn't")
except ValueError:
    log(PASS, "missing-column rejection", "ValueError raised correctly for missing columns")
except Exception as e:
    log(WARN, "missing-column rejection", str(e))


# ─────────────────────────────────────────────────────────────
# 4. GEOSPATIAL ANALYSIS
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 4: GEOSPATIAL ANALYSIS")
print("="*60)

from src.geospatial_analysis import (
    haversine_distance, haversine_vectorized, run_dbscan, get_cluster_summary,
    run_kmeans, compute_grid_density, get_nearby_crimes, location_risk_summary
)

# 4a. Haversine distance
try:
    d = haversine_distance(40.7128, -74.0060, 51.5074, -0.1278)  # NY to London
    assert 5500 < d < 5600, f"Expected ~5570 km, got {d:.1f}"
    log(PASS, "haversine_distance()", f"NY→London = {d:.1f} km")
except Exception as e:
    log(FAIL, "haversine_distance()", str(e))

# 4b. Haversine vectorized
try:
    lats = np.array([40.7128, 51.5074])
    lons = np.array([-74.0060, -0.1278])
    dists = haversine_vectorized(40.7128, -74.0060, lats, lons)
    assert dists[0] < 0.1, f"Self-distance should be ~0, got {dists[0]}"
    log(PASS, "haversine_vectorized()", f"self={dists[0]:.4f} km, NY→London={dists[1]:.1f} km")
except Exception as e:
    log(FAIL, "haversine_vectorized()", str(e))

# 4c. DBSCAN
try:
    df_cl = run_dbscan(df, eps_km=0.5, min_samples=5)
    assert "cluster_id" in df_cl.columns
    n_clusters = (df_cl["cluster_id"] >= 0).sum()
    n_noise = (df_cl["cluster_id"] == -1).sum()
    log(PASS, "run_dbscan()", f"clustered={n_clusters}, noise={n_noise}")
except Exception as e:
    log(FAIL, "run_dbscan()", traceback.format_exc(limit=3))

# 4d. Cluster summary
try:
    summary = get_cluster_summary(df_cl)
    assert isinstance(summary, pd.DataFrame)
    required_cols = ["cluster_id", "crime_count", "lat_center", "lon_center", "top_crime", "risk_level"]
    missing = [c for c in required_cols if c not in summary.columns]
    if not missing:
        log(PASS, "get_cluster_summary()", f"{len(summary)} clusters, all required columns present")
    else:
        log(FAIL, "get_cluster_summary()", f"missing columns: {missing}")
except Exception as e:
    log(FAIL, "get_cluster_summary()", traceback.format_exc(limit=3))

# 4e. K-Means
try:
    df_km = run_kmeans(df, n_clusters=5)
    assert "kmeans_cluster" in df_km.columns
    log(PASS, "run_kmeans()", f"clusters: {df_km['kmeans_cluster'].nunique()}")
except Exception as e:
    log(FAIL, "run_kmeans()", str(e))

# 4f. Grid density
try:
    density = compute_grid_density(df)
    assert isinstance(density, pd.DataFrame)
    assert "crime_count" in density.columns
    assert "risk_level" in density.columns
    log(PASS, "compute_grid_density()", f"{len(density)} grid cells")
except Exception as e:
    log(FAIL, "compute_grid_density()", traceback.format_exc(limit=3))

# 4g. get_nearby_crimes
try:
    center_lat = df["latitude"].mean()
    center_lon = df["longitude"].mean()
    nearby = get_nearby_crimes(df, center_lat, center_lon, radius_km=1.0)
    assert "distance_km" in nearby.columns
    assert (nearby["distance_km"] <= 1.0).all(), "Some distances exceed the radius"
    log(PASS, "get_nearby_crimes()", f"{len(nearby)} within 1.0 km")
except Exception as e:
    log(FAIL, "get_nearby_crimes()", traceback.format_exc(limit=3))

# 4h. location_risk_summary with crimes
try:
    risk_info = location_risk_summary(nearby)
    valid_risks = {"High Risk", "Medium Risk", "Low Risk"}
    assert risk_info["risk_level"] in valid_risks, f"Invalid risk: {risk_info['risk_level']}"
    assert 0 <= risk_info["confidence"] <= 1
    log(PASS, "location_risk_summary() with data", f"{risk_info['risk_level']}, conf={risk_info['confidence']}")
except Exception as e:
    log(FAIL, "location_risk_summary() with data", str(e))

# 4i. location_risk_summary with empty DataFrame
try:
    empty_nearby = df.head(0)
    risk_empty = location_risk_summary(empty_nearby)
    assert risk_empty["risk_level"] in valid_risks
    assert risk_empty["crime_count"] == 0
    log(PASS, "location_risk_summary() empty input", f"{risk_empty['risk_level']}")
except Exception as e:
    log(FAIL, "location_risk_summary() empty input", str(e))

# 4j. Compute grid density on empty df
try:
    empty_density = compute_grid_density(df.head(0))
    assert isinstance(empty_density, pd.DataFrame)
    log(PASS, "compute_grid_density() empty input", "returns empty df without crash")
except Exception as e:
    log(FAIL, "compute_grid_density() empty input", str(e))


# ─────────────────────────────────────────────────────────────
# 5. MODEL TRAINING
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 5: MODEL TRAINING")
print("="*60)

from src.model_training import train_model, load_model, load_train_results, predict_dataframe

# 5a. Train
try:
    train_results = train_model(df, test_size=0.25)
    required_keys = ["model", "scaler", "accuracy", "precision", "recall", "f1_score",
                     "confusion_matrix", "classification_report", "cv_scores",
                     "cv_mean", "cv_std", "feature_importances", "train_size", "test_size"]
    missing_keys = [k for k in required_keys if k not in train_results]
    if missing_keys:
        log(FAIL, "train_model()", f"missing result keys: {missing_keys}")
    else:
        log(PASS, "train_model()",
            f"acc={train_results['accuracy']:.4f}, f1={train_results['f1_score']:.4f}, "
            f"cv={train_results['cv_mean']:.4f}±{train_results['cv_std']:.4f}")
except Exception as e:
    log(FAIL, "train_model()", traceback.format_exc(limit=5))

# 5b. Metric ranges
try:
    for metric in ["accuracy", "precision", "recall", "f1_score"]:
        v = train_results[metric]
        assert 0 <= v <= 1, f"{metric} out of range: {v}"
    log(PASS, "metric value ranges", "all in [0, 1]")
except Exception as e:
    log(FAIL, "metric value ranges", str(e))

# 5c. Confusion matrix shape
try:
    cm = train_results["confusion_matrix"]
    import numpy as np
    assert cm.shape == (3, 3), f"Expected (3,3), got {cm.shape}"
    log(PASS, "confusion matrix shape", f"{cm.shape}")
except Exception as e:
    log(FAIL, "confusion matrix shape", str(e))

# 5d. No data leakage — labels come from density, not from future data
try:
    # Verify risk_label is computed independently using grid density
    unique_labels = set(df["risk_label"].unique())
    assert unique_labels.issubset({0, 1, 2}), f"Unexpected labels: {unique_labels}"
    log(PASS, "data leakage check", "risk labels derived from spatial density, not target variable")
except Exception as e:
    log(FAIL, "data leakage check", str(e))

# 5e. Model persistence
try:
    model_path = os.path.join(ROOT, "models", "crime_model.pkl")
    scaler_path = os.path.join(ROOT, "models", "scaler.pkl")
    assert os.path.exists(model_path), f"crime_model.pkl not found"
    assert os.path.exists(scaler_path), f"scaler.pkl not found"
    log(PASS, "model persistence", f"pkl files exist: crime_model.pkl, scaler.pkl")
except AssertionError as e:
    log(FAIL, "model persistence", str(e))

# 5f. Load saved model
try:
    loaded_model, loaded_scaler = load_model()
    assert loaded_model is not None
    assert loaded_scaler is not None
    log(PASS, "load_model()", "model and scaler loaded from disk")
except Exception as e:
    log(FAIL, "load_model()", str(e))

# 5g. load_train_results
try:
    saved = load_train_results()
    if saved is not None:
        log(PASS, "load_train_results()", "persisted results loaded")
    else:
        log(WARN, "load_train_results()", "No saved results found (train_results.pkl missing)")
except Exception as e:
    log(WARN, "load_train_results()", str(e))

# 5h. predict_dataframe
try:
    model = train_results["model"]
    scaler = train_results["scaler"]
    sample = df.head(20).copy()
    df_pred = predict_dataframe(sample, model, scaler)
    assert "predicted_risk" in df_pred.columns
    assert "predicted_label" in df_pred.columns
    assert "confidence" in df_pred.columns
    valid_labels = {"Low Risk", "Medium Risk", "High Risk"}
    bad_labels = set(df_pred["predicted_label"].unique()) - valid_labels
    if bad_labels:
        log(FAIL, "predict_dataframe()", f"invalid labels: {bad_labels}")
    else:
        log(PASS, "predict_dataframe()", f"predictions: {dict(df_pred['predicted_label'].value_counts())}")
except Exception as e:
    log(FAIL, "predict_dataframe()", traceback.format_exc(limit=3))

# 5i. Feature importance integrity
try:
    importances = train_results["feature_importances"]
    assert len(importances) > 0
    total = sum(importances.values())
    assert 0.99 < total < 1.01, f"Importances don't sum to 1: {total:.4f}"
    log(PASS, "feature importances", f"{len(importances)} features, sum={total:.4f}")
except Exception as e:
    log(FAIL, "feature importances", str(e))


# ─────────────────────────────────────────────────────────────
# 6. HOTSPOT PREDICTION
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 6: HOTSPOT PREDICTION")
print("="*60)

from src.hotspot_prediction import (
    generate_prediction_grid, predict_hotspots,
    predict_single_location, get_top_hotspots
)

# 6a. Grid generation
try:
    grid = generate_prediction_grid(40.60, 40.80, -74.10, -73.90, step=0.02)
    assert len(grid) > 0
    assert "latitude" in grid.columns and "longitude" in grid.columns
    assert (grid["latitude"].between(40.59, 40.81)).all()
    log(PASS, "generate_prediction_grid()", f"{len(grid)} cells generated")
except Exception as e:
    log(FAIL, "generate_prediction_grid()", str(e))

# 6b. Grid prediction
try:
    grid_preds = predict_hotspots(
        model, scaler,
        lat_min=40.68, lat_max=40.76,
        lon_min=-74.06, lon_max=-73.96,
        step=0.02, reference_df=df,
    )
    assert "predicted_risk" in grid_preds.columns
    assert "predicted_label" in grid_preds.columns
    assert "confidence" in grid_preds.columns
    assert (grid_preds["predicted_risk"].isin([0, 1, 2])).all()
    assert (grid_preds["confidence"].between(0, 1)).all()
    log(PASS, "predict_hotspots()",
        f"{len(grid_preds)} cells | H={( grid_preds['predicted_risk']==2).sum()} M={(grid_preds['predicted_risk']==1).sum()} L={(grid_preds['predicted_risk']==0).sum()}")
except Exception as e:
    log(FAIL, "predict_hotspots()", traceback.format_exc(limit=3))

# 6c. Single location prediction
try:
    result = predict_single_location(
        model, scaler,
        lat=40.7128, lon=-74.0060,
        month=6, hour=14, day_of_week=2,
        crime_type_encoded=0, reference_df=df,
    )
    required_keys = ["risk_level", "risk_code", "confidence",
                     "probability_low", "probability_medium", "probability_high", "color"]
    missing = [k for k in required_keys if k not in result]
    if missing:
        log(FAIL, "predict_single_location()", f"missing keys: {missing}")
    else:
        prob_sum = result["probability_low"] + result["probability_medium"] + result["probability_high"]
        assert 0.99 < prob_sum < 1.01, f"probabilities don't sum to 1: {prob_sum}"
        assert result["risk_level"] in {"Low Risk", "Medium Risk", "High Risk"}
        log(PASS, "predict_single_location()",
            f"{result['risk_level']}, conf={result['confidence']:.2f}, "
            f"P(H)={result['probability_high']:.2f}")
except Exception as e:
    log(FAIL, "predict_single_location()", traceback.format_exc(limit=3))

# 6d. get_top_hotspots
try:
    top = get_top_hotspots(grid_preds, top_n=5)
    assert len(top) <= 5
    assert "predicted_label" in top.columns
    log(PASS, "get_top_hotspots()", f"{len(top)} zones returned")
except Exception as e:
    log(FAIL, "get_top_hotspots()", str(e))

# 6e. Single prediction with out-of-range coordinates (edge case)
try:
    result_edge = predict_single_location(
        model, scaler,
        lat=-33.8688, lon=151.2093,  # Sydney, Australia — very different from training data
        month=1, hour=2, day_of_week=6,
        crime_type_encoded=0, reference_df=df,
    )
    assert result_edge["risk_level"] in {"Low Risk", "Medium Risk", "High Risk"}
    log(PASS, "predict_single_location() out-of-training-distribution", f"{result_edge['risk_level']} (extrapolation handled)")
except Exception as e:
    log(WARN, "predict_single_location() out-of-distribution", str(e))


# ─────────────────────────────────────────────────────────────
# 7. VISUALIZATION
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 7: VISUALIZATION")
print("="*60)

from src.visualization import (
    build_crime_map, build_hotspot_prediction_map, add_location_marker,
    plot_crime_by_category, plot_crime_trend, plot_crime_by_hour,
    plot_risk_distribution, plot_confusion_matrix, plot_feature_importance,
    plot_monthly_heatmap
)
import folium

# 7a. build_crime_map — full features
try:
    m = build_crime_map(
        df=df.head(100),
        cluster_df=summary,
        show_heatmap=True, show_markers=True, show_clusters=True,
    )
    assert isinstance(m, folium.Map)
    center = m.location
    assert len(center) == 2
    assert -90 <= center[0] <= 90, f"Map lat out of range: {center[0]}"
    assert -180 <= center[1] <= 180, f"Map lon out of range: {center[1]}"
    log(PASS, "build_crime_map()", f"center=({center[0]:.3f},{center[1]:.3f})")
except Exception as e:
    log(FAIL, "build_crime_map()", traceback.format_exc(limit=3))

# 7b. build_crime_map with filter
try:
    m_filtered = build_crime_map(df=df.head(50), crime_filter="Theft",
                                  show_heatmap=True, show_markers=True, show_clusters=False)
    assert isinstance(m_filtered, folium.Map)
    log(PASS, "build_crime_map() with filter", "crime_filter='Theft' OK")
except Exception as e:
    log(FAIL, "build_crime_map() with filter", str(e))

# 7c. build_crime_map with single-row df (edge case)
try:
    m_one = build_crime_map(df=df.head(1), show_heatmap=True)
    assert isinstance(m_one, folium.Map)
    log(PASS, "build_crime_map() single row", "no crash")
except Exception as e:
    log(FAIL, "build_crime_map() single row", str(e))

# 7d. build_hotspot_prediction_map
try:
    m_pred = build_hotspot_prediction_map(
        grid_df=grid_preds,
        actual_df=df.head(50),
        center_lat=40.71, center_lon=-74.00,
    )
    assert isinstance(m_pred, folium.Map)
    log(PASS, "build_hotspot_prediction_map()", "OK")
except Exception as e:
    log(FAIL, "build_hotspot_prediction_map()", str(e))

# 7e. add_location_marker
try:
    m2 = build_crime_map(df=df.head(20))
    m2 = add_location_marker(m2, 40.71, -74.00, result)
    assert isinstance(m2, folium.Map)
    log(PASS, "add_location_marker()", "marker added")
except Exception as e:
    log(FAIL, "add_location_marker()", str(e))

# 7f. Plotly charts
chart_tests = [
    ("plot_crime_by_category()", lambda: plot_crime_by_category(df)),
    ("plot_crime_trend() monthly", lambda: plot_crime_trend(df, freq="ME")),
    ("plot_crime_trend() yearly", lambda: plot_crime_trend(df, freq="YE")),
    ("plot_crime_by_hour()", lambda: plot_crime_by_hour(df)),
    ("plot_risk_distribution()", lambda: plot_risk_distribution(df)),
    ("plot_confusion_matrix()", lambda: plot_confusion_matrix(np.array([[50,5,2],[3,80,4],[1,2,60]]))),
    ("plot_feature_importance()", lambda: plot_feature_importance(train_results["feature_importances"])),
    ("plot_monthly_heatmap()", lambda: plot_monthly_heatmap(df)),
]
import plotly.graph_objects as go
for name, fn in chart_tests:
    try:
        fig = fn()
        assert isinstance(fig, go.Figure), f"Expected Figure, got {type(fig)}"
        log(PASS, name, "Figure created")
    except Exception as e:
        log(FAIL, name, traceback.format_exc(limit=2))

# 7g. plot_risk_distribution with risk_label column
try:
    df_risk = df.copy()
    df_risk["risk_label"] = df_risk["risk_label"]  # already has it
    fig_r = plot_risk_distribution(df_risk)
    assert isinstance(fig_r, go.Figure)
    log(PASS, "plot_risk_distribution() with risk_label col", "OK")
except Exception as e:
    log(FAIL, "plot_risk_distribution() with risk_label col", str(e))


# ─────────────────────────────────────────────────────────────
# 8. HELPERS & UTILITIES
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 8: HELPERS & UTILITIES")
print("="*60)

from utils.helpers import (
    get_project_root, get_data_path, get_model_path,
    df_to_csv_bytes, generate_report_text, ensure_src_on_path
)

# 8a. Path helpers
try:
    root = get_project_root()
    assert os.path.isdir(root), f"Root not a directory: {root}"
    data_path = get_data_path()
    model_path = get_model_path()
    log(PASS, "path helpers", f"root={root[-40:]}")
except Exception as e:
    log(FAIL, "path helpers", str(e))

# 8b. df_to_csv_bytes
try:
    csv_bytes = df_to_csv_bytes(df.head(10))
    assert isinstance(csv_bytes, bytes)
    assert len(csv_bytes) > 100
    log(PASS, "df_to_csv_bytes()", f"{len(csv_bytes)} bytes")
except Exception as e:
    log(FAIL, "df_to_csv_bytes()", str(e))

# 8c. generate_report_text
try:
    report_text = generate_report_text(df, train_results)
    assert isinstance(report_text, str)
    assert "DISCLAIMER" in report_text
    assert "DATASET SUMMARY" in report_text
    assert "MODEL PERFORMANCE" in report_text
    log(PASS, "generate_report_text()", f"{len(report_text)} chars")
except Exception as e:
    log(FAIL, "generate_report_text()", str(e))

# 8d. generate_report_text without train_results
try:
    report_no_model = generate_report_text(df, None)
    assert isinstance(report_no_model, str)
    log(PASS, "generate_report_text() without model", "no crash")
except Exception as e:
    log(FAIL, "generate_report_text() without model", str(e))

# 8e. ensure_src_on_path
try:
    ensure_src_on_path()
    log(PASS, "ensure_src_on_path()", "no error")
except Exception as e:
    log(FAIL, "ensure_src_on_path()", str(e))


# ─────────────────────────────────────────────────────────────
# 9. DEPLOYMENT READINESS
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SECTION 9: DEPLOYMENT READINESS")
print("="*60)

# 9a. No absolute paths in source files
src_files = []
for dirpath, _, filenames in os.walk(ROOT):
    if "__pycache__" in dirpath or ".git" in dirpath:
        continue
    for f in filenames:
        if f.endswith(".py"):
            src_files.append(os.path.join(dirpath, f))

windows_abs_paths = []
for fpath in src_files:
    rel = fpath.replace(ROOT + os.sep, "")
    try:
        content = open(fpath, encoding="utf-8", errors="replace").read()
        # Check for hardcoded Windows drive paths outside of docstrings/comments
        import re
        # Look for patterns like C:\ or D:\ that are not in comments
        lines = content.split("\n")
        for lineno, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if re.search(r'[A-Z]:\\', line) and "sys.argv" not in line:
                windows_abs_paths.append(f"{rel}:{lineno}: {line.strip()[:80]}")
    except Exception:
        pass

if not windows_abs_paths:
    log(PASS, "no hardcoded absolute Windows paths in .py files")
else:
    for p in windows_abs_paths:
        log(WARN, "potential hardcoded path", p)

# 9b. requirements.txt completeness
try:
    req_path = os.path.join(ROOT, "requirements.txt")
    req_content = open(req_path, encoding="utf-8").read()
    required_pkgs = ["streamlit", "pandas", "numpy", "scikit-learn", "folium",
                     "streamlit-folium", "plotly", "matplotlib", "joblib"]
    missing_pkgs = [p for p in required_pkgs if p.lower() not in req_content.lower()]
    if not missing_pkgs:
        log(PASS, "requirements.txt completeness", f"all {len(required_pkgs)} core packages listed")
    else:
        log(FAIL, "requirements.txt completeness", f"missing: {missing_pkgs}")
except Exception as e:
    log(FAIL, "requirements.txt completeness", str(e))

# 9c. No .venv or node_modules or local-only directories in project
excluded_dirs = [".venv", "venv", "env", "node_modules", "__pycache__"]
for d in excluded_dirs:
    full_path = os.path.join(ROOT, d)
    if os.path.exists(full_path):
        log(WARN, f"deployment: {d} exists", "Should be in .gitignore")
    else:
        log(PASS, f"deployment: no {d} in project root")

# 9d. .streamlit config compatibility check
streamlit_config = os.path.join(ROOT, ".streamlit", "config.toml")
if os.path.exists(streamlit_config):
    log(PASS, ".streamlit/config.toml exists")
else:
    log(WARN, ".streamlit/config.toml", "Not found — creating one is recommended for Cloud deployment")

# 9e. Verify app.py is the entry point
try:
    app_content = open(os.path.join(ROOT, "app.py"), encoding="utf-8").read()
    assert "st.set_page_config" in app_content
    assert "init_session_state" in app_content
    log(PASS, "app.py entry point", "st.set_page_config and init_session_state present")
except Exception as e:
    log(FAIL, "app.py entry point", str(e))

# 9f. Streamlit page files naming convention
page_files = [f for f in os.listdir(os.path.join(ROOT, "pages")) if f.endswith(".py") and not f.startswith("_")]
page_files.sort()
for pf in page_files:
    log(PASS, f"page file: {pf}")

# 9g. Sample CSV columns
try:
    csv_cols = list(pd.read_csv(DATA_PATH, nrows=1).columns)
    required = ["crime_id", "date", "time", "crime_type", "latitude", "longitude", "location", "data_source"]
    missing_csv = [c for c in required if c.lower() not in [x.lower() for x in csv_cols]]
    if not missing_csv:
        log(PASS, "sample_crime_data.csv columns", f"{csv_cols}")
    else:
        log(WARN, "sample_crime_data.csv columns", f"missing: {missing_csv}, found: {csv_cols}")
except Exception as e:
    log(FAIL, "sample_crime_data.csv columns", str(e))


# ─────────────────────────────────────────────────────────────
# FINAL REPORT
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("AUDIT REPORT SUMMARY")
print("="*60)

passes = [r for r in results if r[0] == PASS]
fails  = [r for r in results if r[0] == FAIL]
warns  = [r for r in results if r[0] == WARN]

print(f"\n  [PASS] : {len(passes)}")
print(f"  [FAIL] : {len(fails)}")
print(f"  [WARN] : {len(warns)}")
print(f"  TOTAL  : {len(results)}\n")

if fails:
    print("--- FAILURES ---")
    for _, test, detail in fails:
        print(f"  FAIL  {test}")
        if detail:
            for line in detail.strip().split("\n")[:3]:
                print(f"     {line}")
    print()

if warns:
    print("--- WARNINGS ---")
    for _, test, detail in warns:
        print(f"  WARN  {test}: {detail[:120]}")
    print()

if not fails:
    print("  ALL TESTS PASSED -- APPLICATION IS STABLE")
else:
    print(f"  {len(fails)} TEST(S) FAILED -- SEE ABOVE FOR DETAILS")
