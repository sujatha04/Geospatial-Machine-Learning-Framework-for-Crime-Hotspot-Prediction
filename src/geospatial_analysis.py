"""
geospatial_analysis.py
----------------------
Spatial analytics: clustering, density, and grid-based crime analysis.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in km between two points."""
    rlat1, rlon1 = np.radians(lat1), np.radians(lon1)
    rlat2, rlon2 = np.radians(lat2), np.radians(lon2)
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def haversine_vectorized(lat1: float, lon1: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Return the great-circle distance in km between (lat1, lon1) and arrays of coordinates."""
    rlat1, rlon1 = np.radians(lat1), np.radians(lon1)
    rlats, rlons = np.radians(lats), np.radians(lons)
    dlat = rlats - rlat1
    dlon = rlons - rlon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(rlat1) * np.cos(rlats) * np.sin(dlon / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# DBSCAN Clustering
# ---------------------------------------------------------------------------

def run_dbscan(df: pd.DataFrame, eps_km: float = 0.5, min_samples: int = 5) -> pd.DataFrame:
    """
    Run DBSCAN on crime coordinates and attach cluster labels.

    Parameters
    ----------
    eps_km      : neighbourhood radius in km (converted to radians for haversine metric)
    min_samples : minimum points to form a core point

    Returns
    -------
    DataFrame with added 'cluster_id' column. Label -1 means noise/outlier.
    """
    df = df.copy()
    coords = np.radians(df[["latitude", "longitude"]].values)
    eps_rad = eps_km / EARTH_RADIUS_KM

    model = DBSCAN(
        eps=eps_rad,
        min_samples=min_samples,
        algorithm="ball_tree",
        metric="haversine",
    )
    df["cluster_id"] = model.fit_predict(coords)

    n_clusters = len(set(df["cluster_id"])) - (1 if -1 in df["cluster_id"].values else 0)
    n_noise = (df["cluster_id"] == -1).sum()
    logger.info("DBSCAN: %d clusters, %d noise points (eps=%.2f km)", n_clusters, n_noise, eps_km)
    return df


def get_cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise each DBSCAN cluster: centroid, size, dominant crime type.
    """
    if "cluster_id" not in df.columns:
        raise ValueError("Run run_dbscan first to add 'cluster_id'.")

    clusters = df[df["cluster_id"] >= 0].copy()
    if len(clusters) == 0:
        return pd.DataFrame(columns=["cluster_id", "crime_count", "lat_center", "lon_center", "top_crime", "risk_level"])

    summary = (
        clusters.groupby("cluster_id")
        .agg(
            crime_count=("crime_type", "count"),
            lat_center=("latitude", "mean"),
            lon_center=("longitude", "mean"),
            top_crime=("crime_type", lambda x: x.value_counts().idxmax() if len(x) else "N/A"),
        )
        .reset_index()
    )
    # Assign risk level based on cluster size
    q66 = summary["crime_count"].quantile(0.66)
    q33 = summary["crime_count"].quantile(0.33)

    def risk(c):
        if pd.isna(c):
            return "Medium"
        if c >= q66:
            return "High"
        elif c >= q33:
            return "Medium"
        return "Low"

    summary["risk_level"] = summary["crime_count"].apply(risk)
    return summary.sort_values("crime_count", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# K-Means Clustering
# ---------------------------------------------------------------------------

def run_kmeans(df: pd.DataFrame, n_clusters: int = 8) -> pd.DataFrame:
    """
    Run K-Means on scaled coordinates; attaches 'kmeans_cluster' column.
    """
    df = df.copy()
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(df[["latitude", "longitude"]])
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["kmeans_cluster"] = km.fit_predict(coords_scaled)
    logger.info("K-Means: %d clusters assigned", n_clusters)
    return df


# ---------------------------------------------------------------------------
# Spatial Grid Density
# ---------------------------------------------------------------------------

def compute_grid_density(df: pd.DataFrame, grid_size: float = 0.02) -> pd.DataFrame:
    """
    Divide the bounding box into a grid and count crimes per cell.

    Parameters
    ----------
    grid_size : degree size of each grid cell (≈ 2 km at mid-latitudes)

    Returns
    -------
    DataFrame with grid cell centre coordinates and crime counts.
    """
    if df.empty or "latitude" not in df.columns or "longitude" not in df.columns:
        return pd.DataFrame(columns=["lat_grid", "lon_grid", "crime_count", "risk_level"])

    df = df.copy()
    df["lat_grid"] = (df["latitude"] // grid_size) * grid_size + grid_size / 2
    df["lon_grid"] = (df["longitude"] // grid_size) * grid_size + grid_size / 2

    density = (
        df.groupby(["lat_grid", "lon_grid"])
        .agg(crime_count=("crime_type", "count"))
        .reset_index()
    )

    if density.empty:
        density["risk_level"] = []
        return density

    q_high = density["crime_count"].quantile(0.75)
    q_low = density["crime_count"].quantile(0.40)

    def classify(c):
        if pd.isna(c):
            return "Medium"
        if c >= q_high:
            return "High"
        elif c >= q_low:
            return "Medium"
        return "Low"

    density["risk_level"] = density["crime_count"].apply(classify)
    return density.sort_values("crime_count", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Nearby Crimes
# ---------------------------------------------------------------------------

def get_nearby_crimes(df: pd.DataFrame, lat: float, lon: float,
                      radius_km: float = 1.0) -> pd.DataFrame:
    """Return all crime records within `radius_km` of (lat, lon)."""
    if df.empty or "latitude" not in df.columns or "longitude" not in df.columns:
        return df.copy()
    df = df.copy()
    df["distance_km"] = haversine_vectorized(lat, lon, df["latitude"].values, df["longitude"].values)
    nearby = df[df["distance_km"] <= radius_km].sort_values("distance_km")
    return nearby


def location_risk_summary(nearby_df: pd.DataFrame) -> dict:
    """
    Summarise risk for a searched location based on nearby crimes.
    Returns consistent 'High Risk', 'Medium Risk', or 'Low Risk'.
    """
    count = len(nearby_df)
    if count == 0:
        return {
            "risk_level": "Low Risk",
            "crime_count": 0,
            "top_crime": "None",
            "crime_breakdown": {},
            "confidence": 0.30,
            "message": "No crimes found in this area."
        }

    top_crime = nearby_df["crime_type"].value_counts().idxmax()
    crime_breakdown = nearby_df["crime_type"].value_counts().to_dict()

    # Heuristic risk thresholds based on radius
    if count >= 15:
        risk = "High Risk"
        confidence = min(0.95, 0.70 + count * 0.005)
    elif count >= 5:
        risk = "Medium Risk"
        confidence = min(0.85, 0.55 + count * 0.01)
    else:
        risk = "Low Risk"
        confidence = max(0.30, 0.50 - (5 - count) * 0.04)

    return {
        "risk_level": risk,
        "crime_count": count,
        "top_crime": top_crime,
        "crime_breakdown": crime_breakdown,
        "confidence": round(confidence, 2),
        "message": f"{count} incidents within the selected radius.",
    }
