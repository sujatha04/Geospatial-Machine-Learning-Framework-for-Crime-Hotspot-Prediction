"""
model_training.py
-----------------
Trains a Random Forest classifier for crime hotspot risk classification.
Avoids data leakage by engineering features before train/test split.
"""

import os
import joblib
import numpy as np
import pandas as pd
import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

from src.data_preprocessing import get_feature_columns, compute_risk_label, engineer_features

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "crime_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
RESULTS_PATH = os.path.join(MODEL_DIR, "train_results.pkl")
LABEL_ENC_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train_model(df: pd.DataFrame, test_size: float = 0.25,
                random_state: int = 42) -> dict:
    """
    Train a Random Forest classifier and persist model + scaler.

    Returns
    -------
    results : dict with model, scaler, metrics, and feature importances.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Ensure feature engineering is completed
    if "hour_sin" not in df.columns:
        df = engineer_features(df)

    # --- Assign risk labels (spatial density-based — no leakage from target) ---
    df = compute_risk_label(df)

    feature_cols = get_feature_columns()
    # Keep only columns that actually exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].fillna(0).values
    y = df["risk_label"].values

    # Stratified split to preserve class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Random Forest
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    report = classification_report(
        y_test, y_pred,
        labels=[0, 1, 2],
        target_names=["Low Risk", "Medium Risk", "High Risk"],
        zero_division=0,
        output_dict=True,
    )

    # Cross-validation (on training set only)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=cv,
                                scoring="f1_weighted", n_jobs=-1)

    # Feature importances
    importances = dict(zip(feature_cols, clf.feature_importances_))
    importances = dict(sorted(importances.items(), key=lambda x: -x[1]))

    # Persist
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    logger.info("Model saved to %s", MODEL_PATH)
    logger.info("Scaler saved to %s", SCALER_PATH)

    results = {
        "model": clf,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm,
        "classification_report": report,
        "cv_scores": cv_scores,
        "cv_mean": round(cv_scores.mean(), 4),
        "cv_std": round(cv_scores.std(), 4),
        "feature_importances": importances,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "class_distribution": dict(pd.Series(y).value_counts().sort_index()),
    }
    try:
        joblib.dump(results, RESULTS_PATH)
        logger.info("Train results saved to %s", RESULTS_PATH)
    except Exception as exc:
        logger.warning("Could not persist train results: %s", exc)

    return results


# ---------------------------------------------------------------------------
# Load saved model & results
# ---------------------------------------------------------------------------

def load_model():
    """Load persisted model and scaler from disk. Returns (model, scaler) or (None, None)."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as exc:
        logger.warning("Could not load model: %s", exc)
        return None, None


def load_train_results():
    """Load persisted training results from disk. Returns dict or None."""
    if not os.path.exists(RESULTS_PATH):
        return None
    try:
        return joblib.load(RESULTS_PATH)
    except Exception as exc:
        logger.warning("Could not load train results: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Batch predict on a DataFrame
# ---------------------------------------------------------------------------

def predict_dataframe(df: pd.DataFrame, model, scaler) -> pd.DataFrame:
    """
    Add 'predicted_risk', 'predicted_label', and 'confidence' columns to df.
    """
    if "hour_sin" not in df.columns:
        df = engineer_features(df)

    feature_cols = get_feature_columns()
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].fillna(0).values
    X_scaled = scaler.transform(X)

    preds = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)

    label_map = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
    df = df.copy()
    df["predicted_risk"] = preds
    df["predicted_label"] = [label_map.get(p, "Unknown") for p in preds]
    df["confidence"] = probs.max(axis=1).round(3)
    return df
