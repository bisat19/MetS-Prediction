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

# --- 2. FUNGSI INPUT COMPONENT ---
def get_user_inputs(key_suffix):
    """Fungsi untuk membuat form input yang bisa dipanggil di setiap tab"""
    col1, col2, col3 = st.columns(3)
    inputs = {}
    with col1:
        inputs['Usia'] = st.number_input("Usia", 18, 100, 50, key=f"usia_{key_suffix}")
        inputs['Jenis_Kelamin'] = st.selectbox("Jenis Kelamin", [0, 1], format_func=lambda x: "Wanita" if x == 1 else "Pria", key=f"jk_{key_suffix}")
        inputs['Height'] = st.number_input("Tinggi (m)", 1.0, 2.5, 1.65, key=f"h_{key_suffix}")
        inputs['Weight'] = st.number_input("Berat (kg)", 30.0, 200.0, 65.0, key=f"w_{key_suffix}")
    with col2:
        inputs['Waist Measurement'] = st.number_input("Lingkar Pinggang (cm)", 50.0, 150.0, 90.0, key=f"wm_{key_suffix}")
        inputs['Systolic'] = st.number_input("Systolic", 80.0, 250.0, 120.0, key=f"sys_{key_suffix}")
        inputs['Diastolic'] = st.number_input("Diastolic", 40.0, 150.0, 80.0, key=f"dia_{key_suffix}")
    with col3:
        inputs['TyG_Index'] = st.number_input("TyG Index", 5.0, 15.0, 8.0, key=f"tyg_{key_suffix}")
        inputs['newtg'] = st.number_input("Trigliserida", 50.0, 1000.0, 150.0, key=f"tg_{key_suffix}")
        inputs['newhdl'] = st.number_input("HDL", 10.0, 200.0, 45.0, key=f"hdl_{key_suffix}")
        inputs['newua'] = st.number_input("Asam Urat", 1.0, 15.0, 5.0, key=f"ua_{key_suffix}")
    return inputs

# --- 3. PEMBUATAN TABS ---
tab1, tab2, tab3 = st.tabs(["🌳 Random Forest", "🧠 MLP Network", "🔗 Stacking Model"])

# --- TAB 1: RANDOM FOREST ---
with tab1:
    st.header("Prediksi Individual: Random Forest")
    data_rf = get_user_inputs("rf")
    if st.button("Hitung Random Forest", key="btn_rf"):
        input_df = pd.DataFrame([data_rf], columns=SELECTED_FEATURES)
        X_scaled = scaler.transform(input_df.values)
        prob = rf_model.predict_proba(X_scaled)[:, 1][0]
        st.write("---")
        st.metric("Probabilitas MetS (RF)", f"{prob:.4f}")
        if prob > 0.5: st.error("Hasil: Positif MetS")
        else: st.success("Hasil: Negatif MetS")

# --- TAB 2: MLP ---
with tab2:
    st.header("Prediksi Individual: MLP Network")
    data_mlp = get_user_inputs("mlp")
    if st.button("Hitung MLP", key="btn_mlp"):
        input_df = pd.DataFrame([data_mlp], columns=SELECTED_FEATURES)
        X_scaled = scaler.transform(input_df.values)
        prob = mlp_model.predict_proba(X_scaled)[:, 1][0]
        st.write("---")
        st.metric("Probabilitas MetS (MLP)", f"{prob:.4f}")
        if prob > 0.5: st.error("Hasil: Positif MetS")
        else: st.success("Hasil: Negatif MetS")

# --- TAB 3: STACKING ---
with tab3:
    st.header("Prediksi Kompleks: Stacking Model")
    st.info("Model ini akan menjalankan RF dan MLP secara internal sebagai basis input bagi Meta-Model.")
    data_stack = get_user_inputs("stack")
    if st.button("Hitung Stacking", key="btn_stack"):
        input_df = pd.DataFrame([data_stack], columns=SELECTED_FEATURES)
        X_scaled = scaler.transform(input_df.values)
        
        # Alur Stacking
        rf_p = rf_model.predict_proba(X_scaled)[:, 1]
        mlp_p = mlp_model.predict_proba(X_scaled)[:, 1]
        meta_X = np.column_stack((rf_p, mlp_p))
        prob = meta_model.predict_proba(meta_X)[:, 1][0]
        
        st.write("---")
        col_res1, col_res2 = st.columns(2)
        col_res1.metric("Final Probabilitas (Stacking)", f"{prob:.4f}")
        if prob > 0.5: col_res2.error("Hasil Akhir: Positif MetS")
        else: col_res2.success("Hasil Akhir: Negatif MetS")

st.write("---")
st.caption("Dibuat oleh Abisatya")