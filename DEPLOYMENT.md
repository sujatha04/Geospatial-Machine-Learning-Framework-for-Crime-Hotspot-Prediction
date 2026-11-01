# 🚀 Deployment Guide — Crime Hotspot Prediction

This guide provides step-by-step instructions to deploy your Streamlit Geospatial ML application to the cloud for free.

---

## 🌟 Method 1: Streamlit Community Cloud (Recommended & Easiest)

Streamlit Community Cloud is **100% free**, officially maintained by Streamlit, and provides automatic continuous deployment whenever you push changes to GitHub.

### Step 1: Initialize Git and Push to GitHub

Open a terminal (PowerShell or Bash) in the `crime_hotspot_prediction` folder:

```bash
cd "crime_hotspot_prediction"

# Initialize git repository (if not already done)
git init
git add .
git commit -m "Initial commit: Crime Hotspot Prediction App"

# Rename branch to main
git branch -M main

# Add your GitHub remote repository (replace with your repo URL)
git remote add origin https://github.com/<your-username>/<your-repo-name>.git

# Push to GitHub
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with your GitHub account.
2. Click **"New app"** (or **"Create app"**).
3. Select **"Use existing repo"**.
4. Configure the deployment settings:
   - **Repository:** `<your-username>/<your-repo-name>`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **"Deploy!"**
6. Streamlit Cloud will automatically build your app and generate a live public URL (e.g., `https://crime-hotspot-prediction.streamlit.app`).

---

## ☁️ Method 2: Render (Free Web Service / Docker)

Render offers a free tier for hosting web applications.

### Option A: Using Git Repository
1. Push your code to GitHub as shown in Method 1.
2. Go to **[render.com](https://render.com)** and sign in.
3. Click **"New +"** → **"Web Service"**.
4. Connect your GitHub repository.
5. Set:
   - **Environment:** `Python`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
6. Choose the **Free** instance type and click **"Create Web Service"**.

---

## 🤗 Method 3: Hugging Face Spaces (Free ML Hosting)

Hugging Face Spaces supports Streamlit apps out of the box with dedicated hardware.

1. Create a free account at **[huggingface.co](https://huggingface.co)**.
2. Click **"New Space"**.
3. Choose:
   - **Space SDK:** `Streamlit`
   - **License:** `MIT` / `Apache 2.0`
   - **Space hardware:** `Free (CPU basic)`
4. Clone the space repository locally or upload your files (`app.py`, `pages/`, `src/`, `utils/`, `data/`, `models/`, `requirements.txt`).
5. Hugging Face will automatically install `requirements.txt` and run `app.py`.

---

## 🐳 Method 4: Docker Container (Local / AWS / DigitalOcean / GCP)

A ready-to-use [`Dockerfile`](file:///c:/Users/sujathathennarasu/Documents/AI&DS(VIPS.Tech)/Team%20E/Geospatial%20Machine%20Learning%20Crime%20Hotspot%20prediction/crime_hotspot_prediction/Dockerfile) is included in the project.

### 1. Build the Docker Image
```bash
docker build -t crime-hotspot-prediction .
```

### 2. Run the Container
```bash
docker run -d -p 8501:8501 --name crime-hotspot-app crime-hotspot-prediction
```

Open `http://localhost:8501` in your browser.

---

## 📋 Pre-Deployment Checklist

- [x] Python dependencies specified in `requirements.txt`
- [x] Streamlit server configured in `.streamlit/config.toml` (Headless = true)
- [x] Models and sample data included in `models/` and `data/`
- [x] Fallback auto-generation script included (`data/generate_sample_data.py`)
- [x] Tested with `python test_modules.py` (all tests passing)
- [x] Cross-platform path handling using `os.path`

