"""
hotspot_prediction.py
----------------------
Grid-based crime hotspot prediction across a geographical area.
Generates prediction grid cells and classifies each as Low/Medium/High risk.
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional

from src.data_preprocessing import engineer_features, get_feature_columns

logger = logging.getLogger(__name__)

RISK_LABELS = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
RISK_COLORS = {"High Risk": "#d32f2f", "Medium Risk": "#f57c00", "Low Risk": "#388e3c"}


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------

def generate_prediction_grid(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float,
    step: float = 0.01
) -> pd.DataFrame:
    """
    Create a regular lat/lon grid covering the given bounding box.

    Parameters
    ----------
    step : grid spacing in degrees (≈ 1 km at mid-latitudes)
    """
    if lat_min > lat_max:
        lat_min, lat_max = lat_max, lat_min
    if lon_min > lon_max:
        lon_min, lon_max = lon_max, lon_min
    if lat_min == lat_max:
        lat_max += step
    if lon_min == lon_max:
        lon_max += step

    lats = np.arange(lat_min, lat_max + step, step)
    lons = np.arange(lon_min, lon_max + step, step)
    grid_lats, grid_lons = np.meshgrid(lats, lons)
    grid = pd.DataFrame({
        "latitude": grid_lats.ravel(),
        "longitude": grid_lons.ravel(),
    })
    return grid


def _add_default_temporal_features(grid: pd.DataFrame, reference_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Add synthetic temporal features to the prediction grid."""
    if reference_df is not None and "month" in reference_df.columns:
        # Use the modal month/hour from the training data
        modal_month = int(reference_df["month"].mode()[0])
        modal_hour = int(reference_df["hour"].mode()[0]) if "hour" in reference_df.columns else 12
        modal_dow = int(reference_df["day_of_week"].mode()[0]) if "day_of_week" in reference_df.columns else 2
    else:
        modal_month, modal_hour, modal_dow = 6, 14, 2

    grid = grid.copy()
    grid["year"] = 2023
    grid["month"] = modal_month
    grid["day_of_week"] = modal_dow
    grid["is_weekend"] = int(modal_dow >= 5)
    grid["hour"] = modal_hour
    grid["hour_sin"] = np.sin(2 * np.pi * modal_hour / 24)
    grid["hour_cos"] = np.cos(2 * np.pi * modal_hour / 24)
    grid["month_sin"] = np.sin(2 * np.pi * modal_month / 12)
    grid["month_cos"] = np.cos(2 * np.pi * modal_month / 12)
    grid["crime_type_encoded"] = 0  # unknown type for grid prediction

    # Spatial bins (relative to grid extents)
    grid["lat_bin"] = pd.cut(grid["latitude"], bins=20, labels=False).astype(float).fillna(0)
    grid["lon_bin"] = pd.cut(grid["longitude"], bins=20, labels=False).astype(float).fillna(0)
    return grid


# ---------------------------------------------------------------------------
# Predict on grid
# ---------------------------------------------------------------------------

def predict_hotspots(
    model,
    scaler,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    step: float = 0.01,
    reference_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Run model predictions on a spatial grid.

    Returns
    -------
    DataFrame with columns: latitude, longitude, predicted_risk, predicted_label, confidence
    """
    grid = generate_prediction_grid(lat_min, lat_max, lon_min, lon_max, step)
    grid = _add_default_temporal_features(grid, reference_df)

    feature_cols = get_feature_columns()
    feature_cols = [c for c in feature_cols if c in grid.columns]

    X = grid[feature_cols].fillna(0).values
    X_scaled = scaler.transform(X)

    preds = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)

    grid["predicted_risk"] = preds
    grid["predicted_label"] = [RISK_LABELS.get(p, "Unknown") for p in preds]
    grid["confidence"] = probs.max(axis=1).round(3)

    logger.info("Grid prediction: %d cells, High=%d, Med=%d, Low=%d",
                len(grid),
                (preds == 2).sum(), (preds == 1).sum(), (preds == 0).sum())
    return grid


# ---------------------------------------------------------------------------
# Single point prediction
# ---------------------------------------------------------------------------

def predict_single_location(
    model,
    scaler,
    lat: float,
    lon: float,
    month: int = 6,
    hour: int = 14,
    day_of_week: int = 2,
    crime_type_encoded: int = 0,
    reference_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Predict risk level for a single coordinate.
    """
    from src.data_preprocessing import get_feature_columns
    import numpy as np

    # Build a single-row feature vector
    row = {
        "latitude": lat,
        "longitude": lon,
        "year": 2023,
        "month": month,
        "day_of_week": day_of_week,
        "is_weekend": int(day_of_week >= 5),
        "hour": hour,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "month_sin": np.sin(2 * np.pi * month / 12),
        "month_cos": np.cos(2 * np.pi * month / 12),
        "crime_type_encoded": crime_type_encoded,
        "lat_bin": 10,
        "lon_bin": 10,
    }

    # Refine lat/lon bins using reference data extents
    if reference_df is not None:
        lat_min_r = reference_df["latitude"].min()
        lat_max_r = reference_df["latitude"].max()
        lon_min_r = reference_df["longitude"].min()
        lon_max_r = reference_df["longitude"].max()
        lat_range = max(lat_max_r - lat_min_r, 1e-6)
        lon_range = max(lon_max_r - lon_min_r, 1e-6)
        row["lat_bin"] = int(((lat - lat_min_r) / lat_range) * 19)
        row["lon_bin"] = int(((lon - lon_min_r) / lon_range) * 19)

    feature_cols = get_feature_columns()
    X = np.array([[row.get(c, 0) for c in feature_cols]])
    X_scaled = scaler.transform(X)

    pred = int(model.predict(X_scaled)[0])
    probs = model.predict_proba(X_scaled)[0]
    confidence = float(probs.max())

    risk_label = RISK_LABELS.get(pred, "Unknown")
    color = RISK_COLORS.get(risk_label, "#888888")

    # Safely map probabilities by class index
    classes = getattr(model, "classes_", [0, 1, 2])
    prob_dict = {cls: float(p) for cls, p in zip(classes, probs)}

    return {
        "risk_level": risk_label,
        "risk_code": pred,
        "confidence": round(confidence, 3),
        "probability_low": round(prob_dict.get(0, 0.0), 3),
        "probability_medium": round(prob_dict.get(1, 0.0), 3),
        "probability_high": round(prob_dict.get(2, 0.0), 3),
        "color": color,
    }


# ---------------------------------------------------------------------------
# Top hotspot zones
# ---------------------------------------------------------------------------

def get_top_hotspots(grid_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return the top-N highest-confidence High-Risk grid cells."""
    high = grid_df[grid_df["predicted_risk"] == 2].copy()
    if high.empty:
        high = grid_df.nlargest(top_n, "confidence")
    else:
        high = high.nlargest(top_n, "confidence")
    return high[["latitude", "longitude", "predicted_label", "confidence"]].reset_index(drop=True)
