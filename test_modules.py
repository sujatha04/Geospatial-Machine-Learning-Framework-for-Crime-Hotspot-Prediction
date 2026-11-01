"""Quick smoke test for all core modules."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "sample_crime_data.csv")

print("=== Testing data_preprocessing ===")
from src.data_preprocessing import load_and_validate, engineer_features, compute_risk_label, get_summary_stats
df, warns = load_and_validate(DATA_PATH)
df = engineer_features(df)
df = compute_risk_label(df)
stats = get_summary_stats(df)
print("Records:", stats["total_records"])
print("Categories:", stats["total_categories"])
print("Risk High:", stats["risk_high"], "Med:", stats["risk_medium"], "Low:", stats["risk_low"])
print("PASS\n")

print("=== Testing geospatial_analysis ===")
from src.geospatial_analysis import run_dbscan, get_cluster_summary, compute_grid_density, get_nearby_crimes, location_risk_summary
df_cl = run_dbscan(df, eps_km=0.5, min_samples=5)
summary = get_cluster_summary(df_cl)
print("DBSCAN clusters:", len(summary))
density = compute_grid_density(df)
print("Grid cells:", len(density))
nearby = get_nearby_crimes(df, df["latitude"].mean(), df["longitude"].mean(), radius_km=1.0)
risk_info = location_risk_summary(nearby)
print("Location risk:", risk_info["risk_level"], "| Crimes:", risk_info["crime_count"])
print("PASS\n")

print("=== Testing model_training ===")
from src.model_training import train_model, load_model, predict_dataframe
results = train_model(df, test_size=0.25)
print("Accuracy:", results["accuracy"])
print("F1 Score:", results["f1_score"])
print("CV F1:", results["cv_mean"], "+-", results["cv_std"])
print("Train samples:", results["train_size"], "| Test:", results["test_size"])

model = results["model"]
scaler = results["scaler"]
df_pred = predict_dataframe(df.head(10), model, scaler)
print("Predictions:", df_pred["predicted_label"].tolist())
print("PASS\n")

print("=== Testing hotspot_prediction ===")
from src.hotspot_prediction import predict_hotspots, predict_single_location, get_top_hotspots
lat_c = df["latitude"].mean()
lon_c = df["longitude"].mean()
grid = predict_hotspots(
    model, scaler,
    lat_min=lat_c - 0.05, lat_max=lat_c + 0.05,
    lon_min=lon_c - 0.05, lon_max=lon_c + 0.05,
    step=0.02, reference_df=df,
)
print("Grid cells:", len(grid))
top = get_top_hotspots(grid, top_n=5)
print("Top hotspots:\n", top.to_string())

single = predict_single_location(model, scaler, lat=lat_c, lon=lon_c, reference_df=df)
print("Single prediction:", single["risk_level"], "confidence:", single["confidence"])
print("PASS\n")

print("=== Testing visualization (no display) ===")
from src.visualization import (
    build_crime_map, plot_crime_by_category, plot_crime_trend,
    plot_risk_distribution, plot_confusion_matrix, plot_feature_importance,
)
import numpy as np
m = build_crime_map(df.head(50), show_heatmap=True, show_markers=True, show_clusters=False)
print("Folium map built:", type(m).__name__)
fig1 = plot_crime_by_category(df)
print("Category chart:", type(fig1).__name__)
fig2 = plot_crime_trend(df)
print("Trend chart:", type(fig2).__name__)
cm_test = np.array([[50, 5, 2], [3, 80, 4], [1, 2, 60]])
fig3 = plot_confusion_matrix(cm_test)
print("Confusion matrix:", type(fig3).__name__)
print("PASS\n")

print("=" * 50)
print("ALL MODULES PASSED SMOKE TEST")
print("=" * 50)
