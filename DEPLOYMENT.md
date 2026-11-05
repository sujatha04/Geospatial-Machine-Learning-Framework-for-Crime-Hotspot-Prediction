# 🚀 Vercel Deployment Guide — Crime Hotspot Prediction

This application is configured for deployment on **Vercel** with a high-performance **Python Serverless API** (`api/index.py`) and a modern **interactive Leaflet / Tailwind frontend** (`public/index.html`).

---

## 🌟 Deploy on Vercel (1-Click & 100% Free)

### Step 1: Push latest code to GitHub

In your PowerShell / terminal:

```powershell
# 1. Navigate into the project folder
cd "crime_hotspot_prediction"

# 2. Stage all files
git add .

# 3. Commit changes
git commit -m "Deploy: Full-Stack Vercel Serverless Architecture"

# 4. Push to GitHub
git push origin main
```

---

### Step 2: Deploy in Vercel Dashboard

1. Go to **[vercel.com](https://vercel.com)** and log in with your GitHub account.
2. Click **"Add New..."** → **"Project"**.
3. Under **Import Git Repository**, find:
   `sujatha04/Geospatial-Machine-Learning-Framework-for-Crime-Hotspot-Prediction`
4. Click **"Import"**.
5. Configure the project:
   - **Framework Preset:** Leave as `Other`
   - **Root Directory:** Leave as `./` (or click Edit and select `./` if prompted)
6. Click **"Deploy"**!

Within ~45 seconds, Vercel will build your serverless backend and provide a permanent global live URL (e.g., `https://geospatial-machine-learning-framework-for-crime-hotspot.vercel.app`).

---

## 💻 Local Testing & Development

You can run the full-stack web application locally using Python:

```powershell
cd crime_hotspot_prediction
..\venv\Scripts\python.exe api/index.py
```
Open `http://localhost:5000` or open `public/index.html` in your browser!

Or using the Vercel CLI:
```powershell
npx vercel dev
```

---

## 📁 Architecture Overview

- `vercel.json` — Vercel routing configuration directing `/api/*` to Python serverless and `/*` to static web frontend.
- `api/index.py` — Flask Serverless Backend providing endpoints for DBSCAN clustering, ML hotspot prediction, and KPI metrics.
- `public/index.html` — Interactive single-page web dashboard with TailwindCSS, Leaflet.js interactive maps, and Chart.js.
- `src/` — Preprocessing, geospatial calculations, Random Forest machine learning, and cyclical time encodings.
- `data/` — Synthetic crime dataset.
- `models/` — Persisted models with automatic in-memory fallback.
