"""
Script to generate a realistic synthetic crime dataset for testing.
NOTE: All records are SYNTHETIC / SIMULATED data created for academic purposes only.
      These are NOT real police reports or actual crime records.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)

# --- Configuration ---
N_RECORDS = 2000
START_DATE = datetime(2021, 1, 1)
END_DATE = datetime(2023, 12, 31)

# Simulate a mid-sized urban area (loosely inspired by a generic US city grid)
# Coordinates represent a fictional urban grid — not tied to any real location.
CENTER_LAT = 40.7128
CENTER_LON = -74.0060

CRIME_CATEGORIES = [
    "Theft", "Burglary", "Assault", "Vandalism", "Drug Offense",
    "Robbery", "Auto Theft", "Fraud", "Trespassing", "Disorderly Conduct"
]

LOCATION_NAMES = [
    "Downtown", "Midtown", "Eastside", "Westside", "North District",
    "South District", "Harbor Area", "Market Street", "Central Park Zone",
    "Industrial Zone", "University Area", "Residential Block A",
    "Residential Block B", "Shopping District", "Transit Hub"
]

TIME_SLOTS = [
    "00:00", "01:00", "02:00", "03:00", "04:00", "05:00",
    "06:00", "07:00", "08:00", "09:00", "10:00", "11:00",
    "12:00", "13:00", "14:00", "15:00", "16:00", "17:00",
    "18:00", "19:00", "20:00", "21:00", "22:00", "23:00"
]

# Crime hotspot clusters — biased coordinates simulating realistic spatial clustering
HOTSPOT_CENTERS = [
    (CENTER_LAT + 0.02, CENTER_LON + 0.03, 0.35),   # Downtown cluster
    (CENTER_LAT - 0.03, CENTER_LON - 0.02, 0.25),   # Southside cluster
    (CENTER_LAT + 0.05, CENTER_LON - 0.04, 0.20),   # Northside cluster
    (CENTER_LAT - 0.01, CENTER_LON + 0.06, 0.10),   # Eastside cluster
    (CENTER_LAT + 0.08, CENTER_LON + 0.01, 0.10),   # Outlier cluster
]

def generate_coordinates():
    """Generate lat/lon biased toward hotspot clusters."""
    rand_val = np.random.random()
    cumulative = 0.0
    for lat, lon, weight in HOTSPOT_CENTERS:
        cumulative += weight
        if rand_val < cumulative:
            # Spread within the cluster
            lat_offset = np.random.normal(0, 0.015)
            lon_offset = np.random.normal(0, 0.015)
            return round(lat + lat_offset, 6), round(lon + lon_offset, 6)
    # Fallback: random scatter
    return (
        round(CENTER_LAT + np.random.uniform(-0.10, 0.10), 6),
        round(CENTER_LON + np.random.uniform(-0.10, 0.10), 6),
    )

def generate_date():
    """Generate a random date between START_DATE and END_DATE."""
    delta = END_DATE - START_DATE
    random_days = np.random.randint(0, delta.days)
    return (START_DATE + timedelta(days=random_days)).strftime("%Y-%m-%d")

def generate_time():
    """Generate a time, biased toward evening/night hours."""
    weights = [3, 3, 2, 2, 1, 1, 1, 2, 3, 3, 3, 3,
               3, 3, 3, 4, 4, 5, 5, 6, 6, 6, 5, 4]
    weights_norm = [w / sum(weights) for w in weights]
    hour = np.random.choice(range(24), p=weights_norm)
    minute = np.random.choice([0, 15, 30, 45])
    return f"{hour:02d}:{minute:02d}"

def assign_location(lat, lon):
    """Assign a location name based on proximity to cluster centers."""
    distances = [
        abs(lat - c[0]) + abs(lon - c[1]) for c in HOTSPOT_CENTERS
    ]
    idx = distances.index(min(distances))
    location_map = {
        0: "Downtown",
        1: "South District",
        2: "North District",
        3: "Eastside",
        4: "Westside",
    }
    return location_map.get(idx, np.random.choice(LOCATION_NAMES))

def generate_crime_category():
    """Generate crime category with realistic frequency weights."""
    weights = [0.20, 0.12, 0.14, 0.10, 0.08, 0.10, 0.09, 0.07, 0.06, 0.04]
    return np.random.choice(CRIME_CATEGORIES, p=weights)

def main():
    records = []
    for i in range(N_RECORDS):
        lat, lon = generate_coordinates()
        record = {
            "crime_id": f"SYNTH-{i+1:05d}",
            "date": generate_date(),
            "time": generate_time(),
            "crime_type": generate_crime_category(),
            "latitude": lat,
            "longitude": lon,
            "location": assign_location(lat, lon),
            "description": "Synthetic record — generated for academic demonstration only.",
            "data_source": "SYNTHETIC"
        }
        records.append(record)

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    output_path = os.path.join(os.path.dirname(__file__), "sample_crime_data.csv")
    df.to_csv(output_path, index=False)
    print(f"[INFO] Sample dataset saved to: {output_path}")
    print(f"[INFO] Total records: {len(df)}")
    print(f"[INFO] Date range: {df['date'].min()} to {df['date'].max()}")
    print("[DISCLAIMER] This dataset is SYNTHETIC and created for academic purposes only.")
    return df

if __name__ == "__main__":
    main()
