# Geospatial Machine Learning Framework for Crime Hotspot Prediction

> **Academic Project** — Final-Year Engineering Demonstration  
> ⚠️ All data used is **SYNTHETIC** (computer-generated) and is provided for academic purposes only.  
> It does not represent real police records, actual crime statistics, or any real person/location.

---

## Project Overview

This is a complete, end-to-end web-based crime hotspot prediction system that:

- Loads and validates crime CSV datasets
- Engineers spatial and temporal machine learning features
- Clusters crime incidents using **DBSCAN** and **K-Means**
- Trains a **Random Forest** classifier for risk level prediction
- Predicts **High / Medium / Low** risk across a spatial grid
- Visualises everything on an interactive **Folium** map with heatmap layers
- Provides downloadable reports and CSV exports

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Frontend / Dashboard | Streamlit |
| Machine Learning | Scikit-learn (Random Forest, DBSCAN, K-Means) |
| Data Processing | Pandas, NumPy |
| Geospatial Analysis | GeoPandas, Shapely |
| Map Visualisation | Folium + streamlit-folium |
| Charts | Plotly, Matplotlib |
| Model Persistence | joblib |

---

## Project Structure

```
crime_hotspot_prediction/
├── app.py                      # Main Streamlit entry point
├── requirements.txt
├── README.md
├── DEPLOYMENT.md
├── Dockerfile
├── Procfile
├── data/
│   ├── sample_crime_data.csv   # Auto-generated synthetic dataset
│   └── generate_sample_data.py # Dataset generator script
├── models/
│   ├── crime_model.pkl         # Saved Random Forest model
│   ├── scaler.pkl              # Saved StandardScaler
│   └── train_results.pkl       # Saved metrics
├── src/
│   ├── data_preprocessing.py   # Load, validate, clean, feature engineering
│   ├── geospatial_analysis.py  # DBSCAN, K-Means, grid density, nearby search
│   ├── model_training.py       # Train, evaluate, persist Random Forest
│   ├── hotspot_prediction.py   # Grid prediction & single-location prediction
│   └── visualization.py        # Folium maps, Plotly charts
├── pages/
│   ├── 1_Dashboard.py          # KPI cards + overview charts
│   ├── 2_Crime_Analysis.py     # Dataset management + analytics
│   ├── 3_Hotspot_Map.py        # Interactive Folium map
│   ├── 4_Prediction.py         # ML training + grid + single prediction
│   └── 5_Model_Performance.py  # Metrics, confusion matrix, reports
└── utils/
    └── helpers.py              # Session state, caching, UI utilities
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip

### Step 1 — Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> If you encounter issues installing `geopandas` on Windows, use:
> ```bash
> pip install geopandas --find-links https://girder.github.io/large_image_wheels
> ```
> Or install from conda-forge:
> ```bash
> conda install -c conda-forge geopandas
> ```

---

## Running the Application

From the `crime_hotspot_prediction/` directory:

```bash
streamlit run app.py
```

The dashboard will open at **http://localhost:8501** in your browser.

---

## Deployment

To deploy this application to the cloud (Streamlit Community Cloud, Render, Hugging Face Spaces, or Docker), see the comprehensive [Deployment Guide](DEPLOYMENT.md).

---

## Quick Start Guide

1. **Home** — Read the overview, check the system status banner.
2. **Crime Analysis (Page 2)** — Click **"Load Sample Dataset"** to load synthetic data. Or upload your own CSV.
3. **Hotspot Map (Page 3)** — Explore the interactive Folium map. Use the sidebar to filter by crime type or date. Try the location search.
4. **Prediction (Page 4)** — Click **"Train Model"**. Then generate the grid prediction map and try single-location prediction.
5. **Model Performance (Page 5)** — View accuracy, confusion matrix, feature importances, and download reports.

---

## CSV Dataset Format

If uploading your own data, the CSV must include:

| Column | Type | Example |
|--------|------|---------|
| `latitude` | float | 40.7128 |
| `longitude` | float | -74.0060 |
| `date` | date string | 2023-05-12 |
| `crime_type` | string | Theft |
| `time` *(optional)* | HH:MM | 14:30 |
| `location` *(optional)* | string | Downtown |

---

## Machine Learning Details

### Feature Engineering
- Cyclical encoding of hour and month (sin/cos)
- Spatial grid binning (lat_bin, lon_bin)
- Day of week + weekend flag
- Crime type integer encoding

### Risk Labelling
Risk labels are derived from **spatial crime density** using grid-cell counts and percentile thresholds — not from crime severity alone. This avoids label leakage.

### Model
- **Algorithm**: Random Forest Classifier (200 trees, max_depth=12, class_weight='balanced')
- **Split**: Stratified 75/25 train/test
- **Cross-validation**: 5-fold stratified CV on training set
- **Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

---

## Disclaimer

> **All records in the sample dataset are SYNTHETIC** — computer-generated for academic testing.  
> They are not derived from real police data, do not describe real crimes, and do not identify real people or locations.  
> **Do not use this system for operational law enforcement decisions.**

---

## Authors

Final-Year Engineering Team — AI & Data Science  
Academic Year 2025–2026
