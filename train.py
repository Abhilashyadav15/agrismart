"""
AgriSmart - Model Training Pipeline
Trains Decision Tree and Random Forest classifiers for:
1. Crop Recommendation (using soil NPK, pH, moisture/humidity, rainfall, temp)
2. Fertilizer Recommendation (using soil NPK, pH, moisture, soil type, crop type)

Saves trained models and serialization artifacts in models/ directory.
"""

import sys
import os
import pickle

# Ensure UTF-8 output encoding for Windows terminals
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train_crop_models():
    print("=" * 60)
    print("[1] TRAINING CROP RECOMMENDATION MODELS")
    print("=" * 60)

    df_crop = pd.read_csv("data/crop_recommendation.csv")
    print(f"Loaded Crop Dataset: {df_crop.shape[0]} samples, {df_crop['label'].nunique()} crop classes.")

    X = df_crop[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
    y = df_crop["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1. Decision Tree Classifier
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    acc_dt = accuracy_score(y_test, y_pred_dt)

    # 2. Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf, average="weighted")

    print(f"-> Decision Tree Test Accuracy : {acc_dt * 100:.2f}%")
    print(f"-> Random Forest Test Accuracy : {acc_rf * 100:.2f}% (Weighted F1: {f1_rf:.4f})")

    # Save artifacts
    os.makedirs("models", exist_ok=True)
    with open("models/crop_dt_model.pkl", "wb") as f:
        pickle.dump(dt_model, f)
    with open("models/crop_rf_model.pkl", "wb") as f:
        pickle.dump(rf_model, f)

    print("-> Saved crop models to models/crop_dt_model.pkl and models/crop_rf_model.pkl\n")
    return {"dt_acc": acc_dt, "rf_acc": acc_rf}


def train_fertilizer_models():
    print("=" * 60)
    print("[2] TRAINING FERTILIZER RECOMMENDATION MODELS")
    print("=" * 60)

    df_fert = pd.read_csv("data/fertilizer_dataset.csv")
    print(f"Loaded Fertilizer Dataset: {df_fert.shape[0]} samples, {df_fert['Fertilizer'].nunique()} fertilizer classes.")

    # Encode categorical features
    le_soil = LabelEncoder()
    le_crop = LabelEncoder()
    
    df_fert["Soil_Type_Enc"] = le_soil.fit_transform(df_fert["Soil_Type"])
    df_fert["Crop_Type_Enc"] = le_crop.fit_transform(df_fert["Crop_Type"])

    feature_cols = [
        "Nitrogen", "Phosphorus", "Potassium", "pH", "Moisture",
        "Temperature", "Humidity", "Soil_Type_Enc", "Crop_Type_Enc"
    ]
    
    X = df_fert[feature_cols]
    y = df_fert["Fertilizer"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1. Decision Tree Classifier
    dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    acc_dt = accuracy_score(y_test, y_pred_dt)

    # 2. Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf, average="weighted")

    print(f"-> Decision Tree Test Accuracy : {acc_dt * 100:.2f}%")
    print(f"-> Random Forest Test Accuracy : {acc_rf * 100:.2f}% (Weighted F1: {f1_rf:.4f})")

    # Save artifacts and encoders
    with open("models/fertilizer_dt_model.pkl", "wb") as f:
        pickle.dump(dt_model, f)
    with open("models/fertilizer_rf_model.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    with open("models/fertilizer_encoders.pkl", "wb") as f:
        pickle.dump({
            "soil_encoder": le_soil,
            "crop_encoder": le_crop,
            "feature_cols": feature_cols
        }, f)

    print("-> Saved fertilizer models to models/fertilizer_dt_model.pkl and models/fertilizer_rf_model.pkl\n")
    return {"dt_acc": acc_dt, "rf_acc": acc_rf}


if __name__ == "__main__":
    print("STARTING AGRISMART TRAINING PIPELINE...\n")
    crop_res = train_crop_models()
    fert_res = train_fertilizer_models()

    print("=" * 60)
    print("SUMMARY COMPARISON TABLE (DECISION TREE vs RANDOM FOREST)")
    print("=" * 60)
    print(f"{'Task':<25} | {'Decision Tree':<15} | {'Random Forest (Selected)':<25}")
    print("-" * 68)
    print(f"{'Crop Recommendation':<25} | {crop_res['dt_acc']*100:>13.2f}% | {crop_res['rf_acc']*100:>23.2f}%")
    print(f"{'Fertilizer Prediction':<25} | {fert_res['dt_acc']*100:>13.2f}% | {fert_res['rf_acc']*100:>23.2f}%")
    print("=" * 68)
    print("All models and encoders serialized successfully to ./models/\n")
