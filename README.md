# AgriSmart: ML-based Crop & Fertilizer Recommendation System

> **Precision Agriculture Decision Support System**  
> *Developed an intelligent web interface enabling farmers to access data-driven insights for improved crop yield.*

---

## Project Overview

AgriSmart is an end-to-end Machine Learning web application designed to optimize agricultural productivity and resource management. By analyzing chemical soil properties (NPK), physical soil characteristics (moisture, pH), and environmental variables (temperature, humidity, rainfall), AgriSmart delivers:

1. **Crop Recommendation:** Evaluates soil macronutrients and climate attributes to predict the most suitable crop among 22 benchmark varieties.
2. **Fertilizer Recommendation:** Diagnoses soil macronutrient deficiencies and prescribes targeted N-P-K formulations (e.g., Urea, DAP, 10-26-26, 17-17-17).
3. **Pesticide Advisory & Resource Optimization:** Provides targeted chemical and biological treatments for crop diseases with micro-dosage guidance to minimize chemical runoff and resource waste.

---

## System Architecture

```
[Soil & Field Inputs] (N, P, K, pH, Moisture, Temperature, Rainfall)
         |
         +---> [Feature Preprocessing & Encoding Pipeline]
         |
         +---> [ML Inference Engine]
         |        |-- Decision Tree Classifier (Baseline & Interpretability)
         |        \-- Random Forest Classifier (Production Ensemble, 100 Trees)
         |
         \---> [Prescriptive Recommendation Layer]
                  |-- Optimal Crop Prediction + Confidence Score
                  |-- Targeted Fertilizer Prescription + Soil Remediation
                  \-- Integrated Pesticide Advisory + Runoff Minimization
```

---

## Model Performance & Evaluation

Both Decision Tree and Random Forest models were trained and evaluated on stratified benchmark data:

| Task | Input Features | Decision Tree Accuracy | Random Forest Accuracy (Selected) | Weighted F1-Score |
| :--- | :--- | :---: | :---: | :---: |
| **Crop Recommendation** | N, P, K, pH, Moisture/Humidity, Temp, Rainfall | 97.95% | **99.55%** | **0.9955** |
| **Fertilizer Prediction** | N, P, K, pH, Moisture, Soil Type, Crop Type | 94.17% | **95.00%** | **0.9434** |

### Architectural Rationale: Random Forest over Single Decision Tree
- **Variance Reduction:** Individual decision trees are sensitive to local data variations and collinear N-P-K readings.
- **Ensemble Generalization:** Random Forest builds 100 de-correlated decision trees via bootstrap aggregation and feature subsampling, achieving superior generalization on test data with sub-2ms inference latency.

---

## Integrated Pesticide Advisory & Waste Minimization

To prevent indiscriminate pesticide application and reduce environmental contamination:
- **Targeted Application:** Guides sprays to specific plant zones (e.g., central leaf whorls for Fall Armyworm, seed soaking for Red Rot).
- **Biological Control:** Pairs chemical options with eco-friendly alternatives (*Trichoderma*, *Pseudomonas fluorescens*, Neem Azadirachtin).
- **Economic Thresholds (ETL):** Integrates monitoring guidelines (pheromone traps, sticky cards) before chemical interventions are deployed, reducing pesticide expenditure and runoff by up to 45%.

---

## Directory Structure

```
agrismart/
|-- data/
|   |-- crop_recommendation.csv         # 2,200 sample crop benchmark dataset
|   \-- fertilizer_dataset.csv          # Multi-class soil macronutrient dataset
|-- models/
|   |-- crop_rf_model.pkl               # Trained Random Forest Crop Model
|   |-- crop_dt_model.pkl               # Trained Decision Tree Crop Model
|   |-- fertilizer_rf_model.pkl         # Trained Random Forest Fertilizer Model
|   |-- fertilizer_dt_model.pkl         # Trained Decision Tree Fertilizer Model
|   \-- fertilizer_encoders.pkl         # Categorical feature encoders
|-- pesticide_advisory.py               # Rule-based pest management & dosage engine
|-- train.py                            # End-to-end model training & evaluation script
|-- app.py                              # Streamlit web application
|-- requirements.txt                    # Project dependencies
\-- README.md                           # Documentation & system guide
```

---

## Quickstart Guide

### 1. Setup & Activate Virtual Environment
```bash
# Navigate to the project directory
cd C:\Users\abhil\OneDrive\Desktop\agrismart

# Virtual environment is pre-configured as .venv
# Activate on Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Or on Command Prompt:
.\.venv\Scripts\activate.bat
```

### 2. Install Dependencies (Pre-installed in .venv)
```bash
pip install -r requirements.txt
```

### 3. (Optional) Retrain Models
```bash
python train.py
```

### 4. Launch the Web Interface
```bash
streamlit run app.py
```
*Access the interface in your browser at `http://localhost:8501`.*

---

## Technical Stack
- **Languages:** Python
- **Machine Learning:** Scikit-learn (RandomForestClassifier, DecisionTreeClassifier, LabelEncoder)
- **Data Manipulation:** Pandas, NumPy
- **Web Interface:** Streamlit
