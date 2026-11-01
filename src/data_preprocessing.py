"""
data_preprocessing.py
---------------------
Handles loading, validating, cleaning, and feature-engineering crime datasets.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = {"latitude", "longitude", "date", "crime_type"}
OPTIONAL_COLUMNS = {"time", "location", "description"}

CRIME_CATEGORY_MAP = {
    # Normalise common aliases
    "theft": "Theft",
    "burglary": "Burglary",
    "assault": "Assault",
    "vandalism": "Vandalism",
    "drug offense": "Drug Offense",
    "drug": "Drug Offense",
    "robbery": "Robbery",
    "auto theft": "Auto Theft",
    "vehicle theft": "Auto Theft",
    "fraud": "Fraud",
    "trespassing": "Trespassing",
    "disorderly conduct": "Disorderly Conduct",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_and_validate(filepath_or_df) -> Tuple[pd.DataFrame, list]:
    """
    Load a CSV file (or accept a DataFrame) and return a validated DataFrame.

    Returns
    -------
    df : pd.DataFrame  — cleaned data
    warnings : list[str] — human-readable issues found (not fatal)
    """
    warnings = []

    # --- Load ---
    if isinstance(filepath_or_df, pd.DataFrame):
        df = filepath_or_df.copy()
    else:
        try:
            df = pd.read_csv(filepath_or_df)
        except Exception as exc:
            raise ValueError(f"Could not read file: {exc}")

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # --- Check required columns ---
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            f"Columns found: {list(df.columns)}"
        )

    # --- Drop full duplicates ---
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped:
        warnings.append(f"Removed {dropped} duplicate rows.")

    # --- Parse dates ---
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    bad_dates = df["date"].isna().sum()
    if bad_dates:
        warnings.append(f"{bad_dates} rows had unparseable dates and will be dropped.")
        df = df.dropna(subset=["date"])

    # --- Cast coordinates ---
    for col in ["latitude", "longitude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    bad_coords = df[["latitude", "longitude"]].isna().any(axis=1).sum()
    if bad_coords:
        warnings.append(f"{bad_coords} rows had invalid coordinates and will be dropped.")
        df = df.dropna(subset=["latitude", "longitude"])

    # --- Validate coordinate ranges ---
    invalid_lat = (~df["latitude"].between(-90, 90)).sum()
    invalid_lon = (~df["longitude"].between(-180, 180)).sum()
    if invalid_lat:
        warnings.append(f"{invalid_lat} rows have latitudes outside [-90, 90] — dropping.")
        df = df[df["latitude"].between(-90, 90)]
    if invalid_lon:
        warnings.append(f"{invalid_lon} rows have longitudes outside [-180, 180] — dropping.")
        df = df[df["longitude"].between(-180, 180)]

    # --- Normalise crime_type ---
    df["crime_type"] = (
        df["crime_type"]
        .astype(str)
        .str.strip()
        .apply(lambda x: CRIME_CATEGORY_MAP.get(x.lower(), x.title()))
    )

    # --- Ensure optional columns exist ---
    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df = df.reset_index(drop=True)
    logger.info("Dataset loaded: %d records, %d columns", len(df), len(df.columns))
    return df, warnings


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal and spatial features used by the ML model.

    New columns added
    -----------------
    year, month, day_of_week, hour, is_weekend,
    hour_sin, hour_cos, month_sin, month_cos,
    lat_bin, lon_bin, crime_type_encoded
    """
    df = df.copy()

    # --- Temporal ---
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek  # 0=Mon … 6=Sun
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Parse time if available
    if df["time"].notna().any():
        df["hour"] = pd.to_datetime(
            df["time"].astype(str), format="%H:%M", errors="coerce"
        ).dt.hour.fillna(12).astype(int)
    else:
        df["hour"] = 12  # default noon if time absent

    # Cyclical encoding to capture periodicity
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # --- Spatial binning ---
    df["lat_bin"] = pd.cut(df["latitude"], bins=20, labels=False).astype(float)
    df["lon_bin"] = pd.cut(df["longitude"], bins=20, labels=False).astype(float)

    # --- Encode crime type ---
    crime_types = sorted(df["crime_type"].unique())
    type_to_int = {t: i for i, t in enumerate(crime_types)}
    df["crime_type_encoded"] = df["crime_type"].map(type_to_int).fillna(0).astype(int)

    return df


def get_feature_columns() -> list:
    """Return the feature columns used for model training."""
    return [
        "latitude", "longitude",
        "hour_sin", "hour_cos",
        "month_sin", "month_cos",
        "day_of_week", "is_weekend",
        "lat_bin", "lon_bin",
        "crime_type_encoded",
        "year",
    ]


def compute_risk_label(df: pd.DataFrame, percentile_high: float = 75,
                        percentile_low: float = 40) -> pd.DataFrame:
    """
    Assign a 'risk_label' to each grid cell based on local crime density.
    Uses a spatial grid approach — NOT based on crime type severity alone.

    Labels
    ------
    2 = High Risk, 1 = Medium Risk, 0 = Low Risk
    """
    df = df.copy()

    # Drop any pre-existing risk columns to avoid merge conflicts
    for col in ["risk_label", "cell_count"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Round to 2 decimal places to create ~1km grid cells
    df["lat_grid"] = df["latitude"].round(2)
    df["lon_grid"] = df["longitude"].round(2)

    density = (
        df.groupby(["lat_grid", "lon_grid"])
        .size()
        .reset_index(name="cell_count")
    )
    q_high = density["cell_count"].quantile(percentile_high / 100)
    q_low = density["cell_count"].quantile(percentile_low / 100)

    def label(count):
        if count >= q_high:
            return 2  # High
        elif count >= q_low:
            return 1  # Medium
        else:
            return 0  # Low

    density["risk_label"] = density["cell_count"].apply(label)
    df = df.merge(density[["lat_grid", "lon_grid", "risk_label", "cell_count"]],
                  on=["lat_grid", "lon_grid"], how="left")
    df["risk_label"] = df["risk_label"].fillna(0).astype(int)
    return df


def get_summary_stats(df: pd.DataFrame) -> dict:
    """Return a summary statistics dictionary for the dashboard."""
    risk_counts = {"High": 0, "Medium": 0, "Low": 0}
    if "risk_label" in df.columns:
        mapping = {2: "High", 1: "Medium", 0: "Low"}
        counts = df["risk_label"].map(mapping).value_counts().to_dict()
        for k, v in counts.items():
            risk_counts[k] = v

    return {
        "total_records": len(df),
        "total_categories": df["crime_type"].nunique(),
        "date_range": (
            df["date"].min().strftime("%Y-%m-%d"),
            df["date"].max().strftime("%Y-%m-%d"),
        ) if not df["date"].isna().all() else ("N/A", "N/A"),
        "top_crime": df["crime_type"].value_counts().idxmax() if len(df) else "N/A",
        "unique_locations": df["location"].nunique() if "location" in df.columns else 0,
        "risk_high": risk_counts["High"],
        "risk_medium": risk_counts["Medium"],
        "risk_low": risk_counts["Low"],
    }
