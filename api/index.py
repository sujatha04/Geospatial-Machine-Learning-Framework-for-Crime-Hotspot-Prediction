"""
api/index.py
------------
Serverless backend API for Vercel deployment.
Provides endpoints for dataset analytics, DBSCAN clustering,
Random Forest hotspot prediction, location risk assessment, and reports.
"""

import os
import sys
import json
import logging
from datetime import datetime

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np

from src.data_preprocessing import load_and_validate, engineer_features, compute_risk_label, get_summary_stats
from src.geospatial_analysis import run_dbscan, get_cluster_summary, compute_grid_density, get_nearby_crimes, location_risk_summary
from src.model_training import train_model, load_model, load_train_results, predict_dataframe
from src.hotspot_prediction import predict_hotspots, predict_single_location, get_top_hotspots
from utils.helpers import get_data_path, generate_report_text

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Global In-Memory State
# ---------------------------------------------------------------------------
DATA_CACHE = {
    "df": None,
    "model": None,
    "scaler": None,
    "train_results": None,
    "clusters": None,
}


def _convert_numpy(obj):
    """Recursively convert NumPy data types to standard Python types for JSON."""
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.ndarray, list)):
        return [_convert_numpy(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    return obj


def get_dataset():
    """Load or retrieve processed dataset."""
    if DATA_CACHE["df"] is None:
        csv_path = get_data_path("sample_crime_data.csv")
        if not os.path.exists(csv_path):
            from data.generate_sample_data import main as gen_data
            df = gen_data()
        else:
            df, _ = load_and_validate(csv_path)
            df = engineer_features(df)
            df = compute_risk_label(df)
        DATA_CACHE["df"] = df
    return DATA_CACHE["df"]


def get_trained_model():
    """Load persisted model or train a fresh one in memory if needed."""
    if DATA_CACHE["model"] is None or DATA_CACHE["scaler"] is None:
        try:
            m, s = load_model()
            res = load_train_results()
            if m is not None and s is not None:
                DATA_CACHE["model"] = m
                DATA_CACHE["scaler"] = s
                DATA_CACHE["train_results"] = res
        except Exception as exc:
            logger.warning("Could not load model from disk: %s", exc)

    if DATA_CACHE["model"] is None or DATA_CACHE["scaler"] is None:
        df = get_dataset()
        res = train_model(df, test_size=0.25)
        DATA_CACHE["model"] = res["model"]
        DATA_CACHE["scaler"] = res["scaler"]
        DATA_CACHE["train_results"] = res

    return DATA_CACHE["model"], DATA_CACHE["scaler"], DATA_CACHE["train_results"]


# ---------------------------------------------------------------------------
# Frontend Static Routing (for local running & standalone servers)
# ---------------------------------------------------------------------------

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    public_dir = os.path.join(PROJECT_ROOT, "public")
    if path != "" and os.path.exists(os.path.join(public_dir, path)):
        return send_from_directory(public_dir, path)
    return send_from_directory(public_dir, "index.html")


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "Crime Hotspot Prediction API", "time": datetime.utcnow().isoformat()})


@app.route("/api/stats", methods=["GET"])
def stats():
    df = get_dataset()
    summary = get_summary_stats(df)
    
    # Calculate percentages safely
    total = summary["total_records"]
    high_pct = round((summary["risk_high"] / total) * 100, 1) if total > 0 else 0
    med_pct = round((summary["risk_medium"] / total) * 100, 1) if total > 0 else 0
    low_pct = round((summary["risk_low"] / total) * 100, 1) if total > 0 else 0

    response_data = {
        "total_records": total,
        "total_categories": summary["total_categories"],
        "date_range": summary["date_range"],
        "top_crime": summary["top_crime"],
        "unique_locations": summary["unique_locations"],
        "risk_high": summary["risk_high"],
        "risk_medium": summary["risk_medium"],
        "risk_low": summary["risk_low"],
        "risk_high_pct": high_pct,
        "risk_medium_pct": med_pct,
        "risk_low_pct": low_pct,
    }
    return jsonify(_convert_numpy(response_data))


@app.route("/api/analytics", methods=["GET"])
def analytics():
    df = get_dataset()
    
    # Crime by category
    cat_series = df["crime_type"].value_counts()
    categories = [{"type": k, "count": int(v)} for k, v in cat_series.items()]

    # Monthly trend
    try:
        trend_s = df.set_index("date").resample("ME").size()
    except Exception:
        trend_s = df.set_index("date").resample("M").size()
    trend = [{"date": idx.strftime("%b %Y"), "count": int(val)} for idx, val in trend_s.items()]

    # Hourly pattern
    hourly = []
    if "hour" in df.columns:
        h_s = df.groupby("hour").size()
        hourly = [{"hour": int(h), "count": int(c)} for h, c in h_s.items()]

    # Day of week pattern
    dow = []
    if "day_of_week" in df.columns:
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        d_s = df.groupby("day_of_week").size()
        dow = [{"day": day_names[int(d)] if int(d) < 7 else "?", "count": int(c)} for d, c in d_s.items()]

    # Top hotspot areas
    top_locs = (
        df.groupby("location")
        .agg(crime_count=("crime_type", "count"),
             top_crime=("crime_type", lambda x: x.value_counts().idxmax() if len(x) else "N/A"))
        .reset_index()
        .sort_values("crime_count", ascending=False)
        .head(10)
    ).to_dict(orient="records")

    return jsonify(_convert_numpy({
        "categories": categories,
        "trend": trend,
        "hourly": hourly,
        "day_of_week": dow,
        "top_locations": top_locs,
    }))


@app.route("/api/data", methods=["GET"])
def get_data():
    df = get_dataset()
    crime_filter = request.args.get("crime_type", "All")
    date_start = request.args.get("date_start")
    date_end = request.args.get("date_end")
    limit = int(request.args.get("limit", 100))

    filtered = df.copy()
    if crime_filter != "All" and "crime_type" in filtered.columns:
        filtered = filtered[filtered["crime_type"] == crime_filter]

    if date_start and date_end:
        try:
            s = pd.Timestamp(date_start)
            e = pd.Timestamp(date_end)
            filtered = filtered[(filtered["date"] >= s) & (filtered["date"] <= e)]
        except Exception:
            pass

    records = filtered.head(limit).to_dict(orient="records")
    return jsonify(_convert_numpy({
        "total": len(filtered),
        "crime_types": sorted(df["crime_type"].unique().tolist()),
        "records": records,
    }))


@app.route("/api/map-data", methods=["GET"])
def map_data():
    df = get_dataset()
    crime_filter = request.args.get("crime_type", "All")
    limit = int(request.args.get("limit", 1500))

    filtered = df.copy()
    if crime_filter != "All":
        filtered = filtered[filtered["crime_type"] == crime_filter]

    cols = ["latitude", "longitude", "crime_type", "date", "location", "risk_label"]
    cols = [c for c in cols if c in filtered.columns]
    
    label_map = {2: "High Risk", 1: "Medium Risk", 0: "Low Risk"}
    records = []
    for row in filtered.head(limit)[cols].to_dict(orient="records"):
        r_lbl = row.get("risk_label", 1)
        row["risk_text"] = label_map.get(int(r_lbl), "Medium Risk")
        records.append(row)

    center_lat = float(filtered["latitude"].mean()) if len(filtered) else 40.7128
    center_lon = float(filtered["longitude"].mean()) if len(filtered) else -74.0060

    return jsonify(_convert_numpy({
        "count": len(records),
        "center": [center_lat, center_lon],
        "points": records,
    }))


@app.route("/api/clusters", methods=["GET"])
def clusters():
    df = get_dataset()
    eps = float(request.args.get("eps", 0.5))
    min_samples = int(request.args.get("min_samples", 5))

    df_clustered = run_dbscan(df, eps_km=eps, min_samples=min_samples)
    summary = get_cluster_summary(df_clustered)
    
    return jsonify(_convert_numpy({
        "total_clusters": len(summary),
        "clusters": summary.to_dict(orient="records"),
    }))


@app.route("/api/search-location", methods=["POST"])
def search_location():
    data = request.get_json() or {}
    lat = float(data.get("lat", 40.7128))
    lon = float(data.get("lon", -74.0060))
    radius = float(data.get("radius", 1.0))

    df = get_dataset()
    nearby = get_nearby_crimes(df, lat, lon, radius_km=radius)
    summary = location_risk_summary(nearby)
    
    return jsonify(_convert_numpy(summary))


@app.route("/api/predict-single", methods=["POST"])
def predict_single():
    data = request.get_json() or {}
    lat = float(data.get("lat", 40.7128))
    lon = float(data.get("lon", -74.0060))
    month = int(data.get("month", 6))
    hour = int(data.get("hour", 14))
    day_of_week = int(data.get("day_of_week", 2))
    crime_type = data.get("crime_type", "Theft")
    radius = float(data.get("radius", 1.0))

    df = get_dataset()
    model, scaler, _ = get_trained_model()

    type_to_int = {t: i for i, t in enumerate(sorted(df["crime_type"].unique()))}
    crime_enc = type_to_int.get(crime_type, 0)

    ml_result = predict_single_location(
        model, scaler,
        lat=lat, lon=lon,
        month=month, hour=hour,
        day_of_week=day_of_week,
        crime_type_encoded=crime_enc,
        reference_df=df,
    )

    nearby = get_nearby_crimes(df, lat, lon, radius_km=radius)
    geo_result = location_risk_summary(nearby)

    risk_scores = {"Low Risk": 0, "Low": 0, "Medium Risk": 1, "Medium": 1, "High Risk": 2, "High": 2}
    ml_score = risk_scores.get(ml_result["risk_level"], 1)
    geo_score = risk_scores.get(geo_result["risk_level"], 1)
    avg_score = (ml_score + geo_score) / 2.0

    if avg_score >= 1.5:
        final_risk = "High Risk"
        final_color = "#d32f2f"
    elif avg_score >= 0.5:
        final_risk = "Medium Risk"
        final_color = "#e65100"
    else:
        final_risk = "Low Risk"
        final_color = "#2e7d32"

    return jsonify(_convert_numpy({
        "ml": ml_result,
        "geospatial": geo_result,
        "combined": {
            "risk_level": final_risk,
            "color": final_color,
            "score": avg_score,
            "nearby_count": len(nearby),
        }
    }))


@app.route("/api/predict-grid", methods=["POST"])
def predict_grid():
    data = request.get_json() or {}
    df = get_dataset()
    model, scaler, _ = get_trained_model()

    lat_min = float(data.get("lat_min", df["latitude"].min()))
    lat_max = float(data.get("lat_max", df["latitude"].max()))
    lon_min = float(data.get("lon_min", df["longitude"].min()))
    lon_max = float(data.get("lon_max", df["longitude"].max()))
    step = float(data.get("step", 0.01))

    grid = predict_hotspots(
        model, scaler,
        lat_min=lat_min, lat_max=lat_max,
        lon_min=lon_min, lon_max=lon_max,
        step=step, reference_df=df
    )
    
    top = get_top_hotspots(grid, top_n=10)

    return jsonify(_convert_numpy({
        "total_cells": len(grid),
        "high_risk_cells": int((grid["predicted_risk"] == 2).sum()),
        "med_risk_cells": int((grid["predicted_risk"] == 1).sum()),
        "low_risk_cells": int((grid["predicted_risk"] == 0).sum()),
        "grid": grid.to_dict(orient="records"),
        "top_hotspots": top.to_dict(orient="records"),
    }))


@app.route("/api/model-performance", methods=["GET"])
def model_performance():
    _, _, results = get_trained_model()
    if results is None:
        df = get_dataset()
        results = train_model(df, test_size=0.25)
        DATA_CACHE["train_results"] = results

    clean_results = {
        "accuracy": results["accuracy"],
        "precision": results["precision"],
        "recall": results["recall"],
        "f1_score": results["f1_score"],
        "cv_mean": results["cv_mean"],
        "cv_std": results["cv_std"],
        "cv_scores": list(results.get("cv_scores", [])),
        "confusion_matrix": results["confusion_matrix"].tolist() if hasattr(results["confusion_matrix"], "tolist") else results["confusion_matrix"],
        "feature_importances": results.get("feature_importances", {}),
        "classification_report": results.get("classification_report", {}),
        "train_size": results.get("train_size", 0),
        "test_size": results.get("test_size", 0),
    }
    return jsonify(_convert_numpy(clean_results))


@app.route("/api/train", methods=["POST"])
def retrain():
    data = request.get_json() or {}
    test_size = float(data.get("test_size", 0.25))

    df = get_dataset()
    results = train_model(df, test_size=test_size)
    DATA_CACHE["model"] = results["model"]
    DATA_CACHE["scaler"] = results["scaler"]
    DATA_CACHE["train_results"] = results

    return jsonify(_convert_numpy({
        "message": "Model re-trained successfully",
        "accuracy": results["accuracy"],
        "f1_score": results["f1_score"],
        "cv_mean": results["cv_mean"],
        "train_size": results["train_size"],
        "test_size": results["test_size"],
    }))


@app.route("/api/export-report", methods=["GET"])
def export_report():
    df = get_dataset()
    _, _, results = get_trained_model()
    text = generate_report_text(df, results)
    return Response(
        text,
        mimetype="text/plain",
        headers={"Content-Disposition": "attachment;filename=crime_hotspot_report.txt"}
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

