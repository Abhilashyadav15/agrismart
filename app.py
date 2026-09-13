"""
AgriSmart - ML-based Crop & Fertilizer Recommendation System
Precision Agriculture Decision Support System
"""

import os
import base64
import pickle
import streamlit as st
import pandas as pd
import numpy as np
from pesticide_advisory import (
    PESTICIDE_DATABASE,
    get_supported_crops,
    get_diseases_for_crop,
    get_recommendation
)

# ------------------------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="AgriSmart - Precision Agriculture System",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")


def set_page_background(image_filename: str):
    """
    Injects:
    1. Bright full-screen landscape background for the main page.
    2. Real farmer with agriculture tool background image for the sidebar.
    """
    image_path = os.path.join(ASSETS_DIR, image_filename)
    sidebar_path = os.path.join(ASSETS_DIR, "sidebar_farmer.jpg")

    encoded_img = ""
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            encoded_img = base64.b64encode(f.read()).decode()

    encoded_sidebar = ""
    if os.path.exists(sidebar_path):
        with open(sidebar_path, "rb") as f:
            encoded_sidebar = base64.b64encode(f.read()).decode()

    css = f"""
    <style>
    /* Vibrant, bright full-screen landscape background */
    [data-testid="stAppViewContainer"] {{
        background: linear-gradient(
            rgba(255, 255, 255, 0.32),
            rgba(255, 255, 255, 0.42)
        ), url("data:image/jpeg;base64,{encoded_img}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    [data-testid="stHeader"] {{
        background: rgba(255, 255, 255, 0.35);
        backdrop-filter: blur(6px);
    }}

    /* Farmer with agriculture instrument sidebar background */
    [data-testid="stSidebar"] {{
        background: linear-gradient(
            rgba(12, 32, 16, 0.62),
            rgba(6, 20, 10, 0.76)
        ), url("data:image/jpeg;base64,{encoded_sidebar}") !important;
        background-size: cover !important;
        background-position: center top !important;
        background-repeat: no-repeat !important;
        border-right: 1px solid #1b4d24 !important;
    }}

    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] label {{
        color: #ffffff !important;
    }}

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{
        color: #c8e6c9 !important;
    }}

    [data-testid="stSidebar"] div[role="radiogroup"] label p {{
        color: #f1f8e9 !important;
        font-weight: 500;
    }}

    [data-testid="stSidebar"] hr {{
        border-color: rgba(255, 255, 255, 0.2) !important;
    }}

    .sidebar-info-box {{
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.22);
        border-radius: 8px;
        padding: 0.9rem;
        color: #e8f5e9;
        font-size: 0.83rem;
        line-height: 1.6;
    }}
    .sidebar-info-box strong {{
        color: #ffffff;
    }}

    /* Clean, frosted glass cards for perfect content readability */
    .content-card {{
        background: rgba(255, 255, 255, 0.93);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.85);
        border-radius: 12px;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.07);
    }}

    .feature-box {{
        background: rgba(255, 255, 255, 0.93);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.8);
        border-top: 4px solid #2e7d32;
        border-radius: 10px;
        padding: 1.4rem;
        height: 100%;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.06);
    }}

    .feature-title {{
        font-size: 1.15rem;
        font-weight: 600;
        color: #1b5e20;
        margin-bottom: 0.5rem;
    }}

    /* Form styling with frosted glass */
    .stForm {{
        background: rgba(255, 255, 255, 0.93) !important;
        backdrop-filter: blur(12px) !important;
        border-radius: 12px !important;
        padding: 1.8rem !important;
        border: 1px solid rgba(255, 255, 255, 0.85) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08) !important;
    }}

    /* Results and diagnosis panels */
    .result-panel {{
        background: rgba(241, 248, 233, 0.95);
        border-left: 5px solid #2e7d32;
        border-radius: 8px;
        padding: 1.4rem;
        margin-top: 1.2rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
    }}

    .protocol-panel {{
        background: rgba(255, 253, 231, 0.96);
        border-left: 5px solid #fbc02d;
        border-radius: 8px;
        padding: 1.2rem;
        margin-top: 1rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
    }}

    .meta-tag {{
        display: inline-block;
        background-color: #f0fdf4;
        color: #166534;
        border: 1px solid #bbf7d0;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 0.25rem 0.7rem;
        border-radius: 4px;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# MODEL LOADING
# ------------------------------------------------------------------------------
@st.cache_resource
def load_models():
    """Load serialized Decision Tree & Random Forest models and encoders."""
    models = {}
    models_dir = os.path.join(BASE_DIR, "models")

    with open(os.path.join(models_dir, "crop_rf_model.pkl"), "rb") as f:
        models["crop_rf"] = pickle.load(f)
    with open(os.path.join(models_dir, "crop_dt_model.pkl"), "rb") as f:
        models["crop_dt"] = pickle.load(f)

    with open(os.path.join(models_dir, "fertilizer_rf_model.pkl"), "rb") as f:
        models["fertilizer_rf"] = pickle.load(f)
    with open(os.path.join(models_dir, "fertilizer_dt_model.pkl"), "rb") as f:
        models["fertilizer_dt"] = pickle.load(f)

    with open(os.path.join(models_dir, "fertilizer_encoders.pkl"), "rb") as f:
        models["fertilizer_meta"] = pickle.load(f)

    return models


try:
    MODELS = load_models()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}. Please run 'python train.py' first.")
    st.stop()


# ------------------------------------------------------------------------------
# SIDEBAR NAVIGATION (Formal & Clean - Zero Emojis)
# ------------------------------------------------------------------------------
st.sidebar.markdown("### AgriSmart")
st.sidebar.caption("Precision Agriculture Decision Support System")
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Navigation Menu",
    [
        "System Overview",
        "Crop Recommendation",
        "Fertilizer Recommendation",
        "Pesticide Advisory",
        "Model Benchmarks"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Inference Engine**")
model_choice = st.sidebar.selectbox(
    "Active Classifier",
    ["Random Forest (Production)", "Decision Tree (Baseline)"],
    index=0
)
selected_engine = "rf" if "Random Forest" in model_choice else "dt"

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div class="sidebar-info-box">
        <strong>Architecture Details:</strong><br>
        Framework: Scikit-learn, Streamlit<br>
        Input Features: Soil NPK, pH, Moisture<br>
        Models: Random Forest & Decision Tree<br>
        Environment: Isolated Virtual Environment
    </div>
    """,
    unsafe_allow_html=True
)


# ==============================================================================
# 0. SYSTEM OVERVIEW
# ==============================================================================
if app_mode == "System Overview":
    set_page_background("home_bg.jpg")

    st.markdown("""
    <div class="content-card">
        <span class="meta-tag">Machine Learning System</span>
        <span class="meta-tag">Supervised Learning</span>
        <span class="meta-tag">Precision Agriculture</span>
        <h1 style="color: #1b5e20; margin-top: 0.4rem; font-size: 2.3rem; margin-bottom: 0.3rem;">AgriSmart</h1>
        <h4 style="color: #2e7d32; font-weight: 500; margin-top: 0;">Precision Agriculture & Crop Yield Decision Support Platform</h4>
        <p style="color: #334155; font-size: 1.05rem; line-height: 1.6; margin-top: 0.8rem;">
            AgriSmart is an end-to-end machine learning system engineered to support data-driven agricultural decision-making. 
            The platform processes chemical soil composition (Nitrogen, Phosphorus, Potassium), physical metrics (pH, soil moisture), 
            and meteorological parameters to provide predictive crop selection, tailored fertilizer prescriptions, 
            and targeted pest management guidance.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Core System Modules")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-box">
            <div class="feature-title">Crop Recommendation</div>
            <p style="color: #475569; font-size: 0.92rem; line-height: 1.5;">
                Multi-class classification pipeline evaluating 7 soil and climate features against 22 crop varieties.
            </p>
            <p style="color: #64748b; font-size: 0.85rem; margin-bottom: 0;">
                • Algorithm: Random Forest (99.55% accuracy)<br>
                • Baseline: Decision Tree (97.95%)<br>
                • Features: N, P, K, pH, moisture, temp, rain
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-box">
            <div class="feature-title">Fertilizer Recommendation</div>
            <p style="color: #475569; font-size: 0.92rem; line-height: 1.5;">
                Nutrient deficit identification model prescribing targeted NPK chemical formulations.
            </p>
            <p style="color: #64748b; font-size: 0.85rem; margin-bottom: 0;">
                • Prescriptions: Urea, DAP, 10-26-26, 17-17-17<br>
                • Accuracy: 95.00% validation accuracy<br>
                • Inputs: Soil NPK, pH, moisture, soil type, crop
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-box">
            <div class="feature-title">Pesticide & Pest Advisory</div>
            <p style="color: #475569; font-size: 0.92rem; line-height: 1.5;">
                Prescriptive decision logic for targeted chemical and biological pest treatments.
            </p>
            <p style="color: #64748b; font-size: 0.85rem; margin-bottom: 0;">
                • Biological and chemical alternatives<br>
                • Precise dosage and application guidance<br>
                • Resource optimization & runoff reduction
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Performance Summary")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Crop Accuracy (RF)", "99.55%")
    m2.metric("Fertilizer Accuracy (RF)", "95.00%")
    m3.metric("Supported Crops", "22 Classes")
    m4.metric("Inference Latency", "< 2 ms")


# ==============================================================================
# 1. CROP RECOMMENDATION
# ==============================================================================
elif app_mode == "Crop Recommendation":
    set_page_background("crop_bg.jpg")

    st.markdown("""
    <div class="content-card">
        <h2 style="color: #1b5e20; margin: 0;">Crop Recommendation Module</h2>
        <p style="color: #475569; font-size: 0.98rem; margin-top: 0.3rem; margin-bottom: 0;">
            Predict optimal crops based on soil nutrients and climatic parameters.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_preset, _ = st.columns([2, 1])
    preset = col_preset.selectbox(
        "Load Test Scenario (Optional)",
        ["Manual Entry", "Scenario 1: Paddy / Rice Conditions", "Scenario 2: Cotton Dryland", "Scenario 3: Chickpea / Legume", "Scenario 4: Coffee Plantation"]
    )

    defaults = {
        "Manual Entry": (90, 42, 43, 6.5, 80.0, 24.0, 200.0),
        "Scenario 1: Paddy / Rice Conditions": (85, 45, 40, 6.6, 82.0, 23.5, 230.0),
        "Scenario 2: Cotton Dryland": (118, 46, 20, 6.9, 80.0, 24.0, 70.0),
        "Scenario 3: Chickpea / Legume": (40, 65, 80, 7.3, 17.0, 19.0, 75.0),
        "Scenario 4: Coffee Plantation": (105, 30, 30, 6.7, 58.0, 26.0, 160.0),
    }[preset]

    with st.form("crop_form"):
        st.markdown("**1. Soil Chemical Composition (NPK & pH)**")
        c1, c2, c3, c4 = st.columns(4)
        n_val = c1.number_input("Nitrogen (N) [kg/ha]", min_value=0, max_value=150, value=int(defaults[0]))
        p_val = c2.number_input("Phosphorus (P) [kg/ha]", min_value=0, max_value=150, value=int(defaults[1]))
        k_val = c3.number_input("Potassium (K) [kg/ha]", min_value=0, max_value=210, value=int(defaults[2]))
        ph_val = c4.number_input("Soil pH Level (0-14)", min_value=3.5, max_value=9.5, value=float(defaults[3]), step=0.1)

        st.markdown("**2. Environmental & Moisture Conditions**")
        c5, c6, c7 = st.columns(3)
        hum_val = c5.number_input("Relative Humidity (%)", min_value=10.0, max_value=100.0, value=float(defaults[4]), step=0.5)
        temp_val = c6.number_input("Average Temperature (°C)", min_value=5.0, max_value=55.0, value=float(defaults[5]), step=0.5)
        rain_val = c7.number_input("Annual Rainfall (mm)", min_value=10.0, max_value=400.0, value=float(defaults[6]), step=1.0)

        submit_crop = st.form_submit_button("Run Crop Prediction", use_container_width=True)

    if submit_crop:
        input_data = pd.DataFrame([[n_val, p_val, k_val, temp_val, hum_val, ph_val, rain_val]],
                                  columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
        
        model = MODELS["crop_rf"] if selected_engine == "rf" else MODELS["crop_dt"]
        prediction = model.predict(input_data)[0]
        
        probabilities = model.predict_proba(input_data)[0]
        confidence = float(np.max(probabilities)) * 100

        st.success(f"Recommended Crop: **{prediction.capitalize()}**")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Active Model", "Random Forest" if selected_engine == "rf" else "Decision Tree")
        m2.metric("Prediction Confidence", f"{confidence:.2f}%")
        m3.metric("Soil Reaction", "Neutral / Optimal" if 6.0 <= ph_val <= 7.5 else ("Acidic" if ph_val < 6.0 else "Alkaline"))

        st.markdown(f"""
        <div class="result-panel">
            <strong>Agronomic Summary:</strong><br>
            For the specified soil composition (N={n_val}, P={p_val}, K={k_val}, pH={ph_val}) and environmental profile 
            (Humidity={hum_val}%, Rainfall={rain_val} mm), <strong>{prediction.capitalize()}</strong> exhibits optimal biological compatibility 
            and nutrient absorption characteristics.
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# 2. FERTILIZER RECOMMENDATION
# ==============================================================================
elif app_mode == "Fertilizer Recommendation":
    set_page_background("fertilizer_bg.jpg")

    st.markdown("""
    <div class="content-card">
        <h2 style="color: #1b5e20; margin: 0;">Fertilizer Recommendation Module</h2>
        <p style="color: #475569; font-size: 0.98rem; margin-top: 0.3rem; margin-bottom: 0;">
            Diagnose nutrient imbalances and determine targeted fertilizer prescriptions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_preset, _ = st.columns([2, 1])
    fert_preset = col_preset.selectbox(
        "Load Test Scenario (Optional)",
        ["Manual Entry", "Scenario 1: High Nitrogen Deficit", "Scenario 2: Phosphorus Deficit", "Scenario 3: Potassium Deficit", "Scenario 4: Balanced Depletion"]
    )

    defaults_f = {
        "Manual Entry": (20, 50, 45, 6.5, 40.0, 26.0, 60.0, "Loamy", "Rice"),
        "Scenario 1: High Nitrogen Deficit": (15, 60, 50, 6.4, 45.0, 27.0, 65.0, "Black", "Wheat"),
        "Scenario 2: Phosphorus Deficit": (55, 12, 45, 6.8, 38.0, 25.0, 55.0, "Loamy", "Cotton"),
        "Scenario 3: Potassium Deficit": (45, 45, 10, 7.0, 50.0, 28.0, 70.0, "Clayey", "Sugarcane"),
        "Scenario 4: Balanced Depletion": (25, 25, 25, 6.5, 42.0, 26.0, 62.0, "Red", "Maize"),
    }[fert_preset]

    with st.form("fert_form"):
        st.markdown("**1. Soil Macronutrients & Chemistry**")
        f1, f2, f3, f4 = st.columns(4)
        n_fert = f1.number_input("Nitrogen (N) [kg/ha]", min_value=0, max_value=150, value=int(defaults_f[0]))
        p_fert = f2.number_input("Phosphorus (P) [kg/ha]", min_value=0, max_value=150, value=int(defaults_f[1]))
        k_fert = f3.number_input("Potassium (K) [kg/ha]", min_value=0, max_value=150, value=int(defaults_f[2]))
        ph_fert = f4.number_input("Soil pH Level", min_value=4.0, max_value=9.5, value=float(defaults_f[3]), step=0.1)

        st.markdown("**2. Field & Crop Parameters**")
        f5, f6, f7, f8, f9 = st.columns(5)
        moist_fert = f5.number_input("Soil Moisture (%)", min_value=10.0, max_value=90.0, value=float(defaults_f[4]))
        temp_fert = f6.number_input("Soil Temp (°C)", min_value=15.0, max_value=45.0, value=float(defaults_f[5]))
        hum_fert = f7.number_input("Atmospheric Humidity (%)", min_value=20.0, max_value=95.0, value=float(defaults_f[6]))
        
        soil_types = list(MODELS["fertilizer_meta"]["soil_encoder"].classes_)
        crop_types = list(MODELS["fertilizer_meta"]["crop_encoder"].classes_)

        soil_choice = f8.selectbox("Soil Type", soil_types, index=soil_types.index(defaults_f[7]) if defaults_f[7] in soil_types else 0)
        crop_choice = f9.selectbox("Target Crop", crop_types, index=crop_types.index(defaults_f[8]) if defaults_f[8] in crop_types else 0)

        submit_fert = st.form_submit_button("Predict Fertilizer Formulation", use_container_width=True)

    if submit_fert:
        meta = MODELS["fertilizer_meta"]
        soil_enc = meta["soil_encoder"].transform([soil_choice])[0]
        crop_enc = meta["crop_encoder"].transform([crop_choice])[0]

        row = pd.DataFrame([[n_fert, p_fert, k_fert, ph_fert, moist_fert, temp_fert, hum_fert, soil_enc, crop_enc]],
                           columns=meta["feature_cols"])

        model_fert = MODELS["fertilizer_rf"] if selected_engine == "rf" else MODELS["fertilizer_dt"]
        fertilizer_pred = model_fert.predict(row)[0]
        fert_conf = float(np.max(model_fert.predict_proba(row)[0])) * 100

        st.success(f"Prescribed Formulation: **{fertilizer_pred}**")

        c_a, c_b, c_c = st.columns(3)
        c_a.metric("Formulation", fertilizer_pred)
        c_b.metric("Confidence Score", f"{fert_conf:.2f}%")
        c_c.metric("Active Model", "Random Forest" if selected_engine == "rf" else "Decision Tree")

        descriptions = {
            "Urea": "High nitrogen formulation (46-0-0). Indicated for vegetative foliage recovery and nitrogen deficiency remediation.",
            "DAP": "Diammonium Phosphate (18-46-0). High phosphorus source to support root proliferation and early plant establishment.",
            "10-26-26": "Enriched Potash-Phosphate blend. Promotes drought tolerance, vascular strength, and grain/fruit development.",
            "17-17-17": "Balanced universal N-P-K formulation. Restores depleted soils requiring proportional macronutrient replenishment.",
            "28-28-0": "High nitrogen and phosphorus complex without potassium. Appropriate for soils with adequate potash reserves.",
            "14-35-14": "Phosphorus-dominant starter blend for early root system anchoring.",
            "20-20-0": "Balanced nitrogen-phosphorus blend for early vegetative expansion."
        }

        desc = descriptions.get(fertilizer_pred, "Targeted chemical blend for balanced soil remediation.")
        st.markdown(f"""
        <div class="result-panel">
            <strong>Prescription Details:</strong><br>
            <strong>{fertilizer_pred}:</strong> {desc}<br><br>
            <strong>Application Note:</strong> Apply during secondary tillage or as targeted top-dressing adjacent to the root zone at present soil moisture ({moist_fert}%). Avoid broadcast application over dry topsoil to minimize volatilization losses.
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# 3. PESTICIDE ADVISORY
# ==============================================================================
elif app_mode == "Pesticide Advisory":
    set_page_background("pesticide_bg.jpg")

    st.markdown("""
    <div class="content-card">
        <h2 style="color: #1b5e20; margin: 0;">Pesticide & Disease Advisory Module</h2>
        <p style="color: #475569; font-size: 0.98rem; margin-top: 0.3rem; margin-bottom: 0;">
            Decision logic for targeted pest management, biological alternatives, and chemical runoff reduction.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    selected_crop = col1.selectbox("Crop Selection", get_supported_crops())
    diseases = get_diseases_for_crop(selected_crop)
    selected_disease = col2.selectbox("Observed Symptom / Pathogen", diseases)

    if st.button("Generate Advisory Report", use_container_width=True):
        advisory = get_recommendation(selected_crop, selected_disease)

        if advisory["found"]:
            st.markdown(f"""
            <div class="result-panel">
                <h4 style="color: #1b5e20; margin-top:0;">Diagnosis: {advisory['disease']} ({advisory['crop']})</h4>
                <p><strong>Clinical Symptoms:</strong> {advisory['symptoms']}</p>
                <hr style="border: 0; border-top: 1px solid #cbd5e1; margin: 0.8rem 0;">
                <p><strong>Targeted Chemical Solution:</strong><br>{advisory['chemical_solution']}</p>
                <p><strong>Biological / Eco-friendly Alternative:</strong><br>{advisory['biological_alternative']}</p>
                <p><strong>Recommended Dosage:</strong><br>{advisory['dosage']}</p>
            </div>
            <div class="protocol-panel">
                <strong>Resource Optimization Protocol:</strong><br>
                {advisory['resource_saving_tip']}
            </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# 4. MODEL BENCHMARKS
# ==============================================================================
elif app_mode == "Model Benchmarks":
    set_page_background("benchmark_bg.jpg")

    st.markdown("""
    <div class="content-card">
        <h2 style="color: #1b5e20; margin: 0;">Model Benchmarks & Performance Metrics</h2>
        <p style="color: #475569; font-size: 0.98rem; margin-top: 0.3rem; margin-bottom: 0;">
            Quantitative comparison between single Decision Tree baseline and Random Forest ensemble models.
        </p>
    </div>
    """, unsafe_allow_html=True)

    b1, b2 = st.columns(2)

    with b1:
        st.subheader("Crop Recommendation Task")
        df_bench_crop = pd.DataFrame({
            "Metric": ["Test Accuracy", "Weighted F1-Score", "Inference Latency", "Variance Characteristic"],
            "Decision Tree": ["97.95%", "0.9795", "< 0.5 ms", "Sensitive to localized feature fluctuations"],
            "Random Forest (Selected)": ["99.55%", "0.9955", "~ 1.5 ms", "Strong variance reduction via ensemble voting"]
        })
        st.dataframe(df_bench_crop, use_container_width=True, hide_index=True)

    with b2:
        st.subheader("Fertilizer Prediction Task")
        df_bench_fert = pd.DataFrame({
            "Metric": ["Test Accuracy", "Weighted F1-Score", "Inference Latency", "Variance Characteristic"],
            "Decision Tree": ["94.17%", "0.9402", "< 0.5 ms", "Prone to leaf over-specialization on NPK"],
            "Random Forest (Selected)": ["95.00%", "0.9434", "~ 1.8 ms", "De-correlated bagging across 100 decision trees"]
        })
        st.dataframe(df_bench_fert, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="result-panel">
        <strong>Architectural Analysis:</strong>
        <ul style="margin-top: 0.5rem; margin-bottom: 0;">
            <li><strong>Decision Tree Baseline:</strong> Provides transparent, interpretable threshold partitions along soil macronutrient axes (N, P, K) and pH.</li>
            <li><strong>Random Forest Selection:</strong> Agricultural features exhibit complex, non-linear multi-collinearity. By ensembling 100 bootstrap-aggregated decision trees with randomized feature sub-sampling, the Random Forest model eliminated individual tree variance, achieving <strong>99.55%</strong> accuracy on the crop benchmark and <strong>95.00%</strong> on fertilizer recommendation.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
