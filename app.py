import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- PENGATURAN HALAMAN ---
st.set_page_config(page_title="MetS Multi-Model Independent", layout="wide")
st.title("🔬 Analisis Mandiri Sindrom Metabolik")

# --- 1. DAFTAR FITUR & MUAT MODEL ---
SELECTED_FEATURES = [
    'TyG_Index', 'Jenis_Kelamin', 'newhdl', 'Waist Measurement', 
    'Systolic', 'Weight', 'newtg', 'Diastolic', 'Usia', 'newua', 'Height'
]

@st.cache_resource
def load_all_assets():
    path = "model_deployments"
    scaler = joblib.load(os.path.join(path, 'scaler.pkl'))
    rf = joblib.load(os.path.join(path, 'rf_model.pkl'))
    mlp = joblib.load(os.path.join(path, 'mlp_model.pkl'))
    meta = joblib.load(os.path.join(path, 'meta_model.pkl'))
    return scaler, rf, mlp, meta

scaler, rf_model, mlp_model, meta_model = load_all_assets()

# --- 2. INISIALISASI SESSION STATE (Sekali saja di awal) ---
DEFAULT_INPUTS = {
    'Usia': 50, 'Jenis_Kelamin': 0, 'Height': 1.65, 'Weight': 65.0,
    'Waist Measurement': 90.0, 'Systolic': 120.0, 'Diastolic': 80.0,
    'TyG_Index': 8.0, 'newtg': 150.0, 'newhdl': 45.0, 'newua': 5.0
}

for key, value in DEFAULT_INPUTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- 3. FUNGSI INPUT COMPONENT (Tanpa callback - otomatis sync via key!) ---
def get_user_inputs():
    """Form input dengan session_state otomatis via parameter 'key'"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.number_input("Usia", 18, 100, key="Usia")
        st.selectbox("Jenis Kelamin", [0, 1], 
                    format_func=lambda x: "Wanita" if x == 1 else "Pria",
                    key="Jenis_Kelamin")
        st.number_input("Tinggi (m)", 1.0, 2.5, key="Height")
        st.number_input("Berat (kg)", 30.0, 200.0, key="Weight")
    
    with col2:
        st.number_input("Lingkar Pinggang (cm)", 50.0, 150.0, key="Waist Measurement")
        st.number_input("Systolic", 80.0, 250.0, key="Systolic")
        st.number_input("Diastolic", 40.0, 150.0, key="Diastolic")
    
    with col3:
        st.number_input("TyG Index", 5.0, 15.0, key="TyG_Index")
        st.number_input("Trigliserida", 50.0, 1000.0, key="newtg")
        st.number_input("HDL", 10.0, 200.0, key="newhdl")
        st.number_input("Asam Urat", 1.0, 15.0, key="newua")
    
    # Ambil semua nilai dari session_state
    return {key: st.session_state[key] for key in DEFAULT_INPUTS.keys()}

# --- 4. INPUT FORM DI LUAR TAB ---
st.subheader("📋 Input Data Pasien")
user_data = get_user_inputs()

# Tombol reset
if st.button("🔄 Reset Semua Input"):
    for key, value in DEFAULT_INPUTS.items():
        st.session_state[key] = value
    st.rerun()

st.divider()

# --- 5. PEMBUATAN TABS ---
tab1, tab2, tab3 = st.tabs(["🌳 Random Forest", "🧠 MLP Network", "🔗 Stacking Model"])

# --- TAB 1: RANDOM FOREST ---
with tab1:
    st.header("Prediksi: Random Forest")
    if st.button("Prediksi Status MetS", key="btn_rf"):
        with st.spinner("Sedang memproses..."):
            input_df = pd.DataFrame([user_data], columns=SELECTED_FEATURES)
            X_scaled = scaler.transform(input_df.values)
            prob = rf_model.predict_proba(X_scaled)[:, 1][0]
            st.write("---")
            col_res1, col_res2 = st.columns(2)
            col_res1.metric("Probabilitas MetS (RF)", f"{prob:.4f}")
            if prob > 0.5: 
                col_res2.error("Hasil: POSITIF SINDROM METABOLIK")
            else: 
                col_res2.success("Hasil: NON-SINDROM METABOLIK")

# --- TAB 2: MLP ---
with tab2:
    st.header("Prediksi: MLP Network")
    if st.button("Prediksi Status MetS", key="btn_mlp"):
        with st.spinner("Sedang memproses..."):
            input_df = pd.DataFrame([user_data], columns=SELECTED_FEATURES)
            X_scaled = scaler.transform(input_df.values)
            prob = mlp_model.predict_proba(X_scaled)[:, 1][0]
            st.write("---")
            col_res1, col_res2 = st.columns(2)
            col_res1.metric("Probabilitas MetS (MLP)", f"{prob:.4f}")
            if prob > 0.5: 
                col_res2.error("Hasil: POSITIF SINDROM METABOLIK")
            else: 
                col_res2.success("Hasil: NON-SINDROM METABOLIK")

# --- TAB 3: STACKING ---
with tab3:
    st.header("Prediksi: Stacking Model")
    st.info("Model ini akan menjalankan RF dan MLP secara internal sebagai basis input bagi Meta-Model.")
    if st.button("Prediksi Status MetS", key="btn_stack"):
        with st.spinner("Sedang memproses..."):
            input_df = pd.DataFrame([user_data], columns=SELECTED_FEATURES)
            X_scaled = scaler.transform(input_df.values)
            
            # Alur Stacking
            rf_p = rf_model.predict_proba(X_scaled)[:, 1]
            mlp_p = mlp_model.predict_proba(X_scaled)[:, 1]
            meta_X = np.column_stack((rf_p, mlp_p))
            prob = meta_model.predict_proba(meta_X)[:, 1][0]
            
            st.write("---")
            col_res1, col_res2 = st.columns(2)
            col_res1.metric("Final Probabilitas (Stacking)", f"{prob:.4f}")
            if prob > 0.5: 
                col_res2.error("Hasil Akhir: POSITIF SINDROM METABOLIK")
            else: 
                col_res2.success("Hasil Akhir: NON-SINDROM METABOLIK")

st.write("---")
st.caption("Dibuat oleh Abisatya")