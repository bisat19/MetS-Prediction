import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time

# --- PENGATURAN HALAMAN ---
st.set_page_config(
    page_title="Prediksi Sindrom Metabolik (MetS) Multi-Model",
    layout="wide"
)
st.title("🔬 Prediksi Sindrom Metabolik (MetS)")
st.markdown("Aplikasi Web untuk Prediksi MetS menggunakan Model Individu dan Stacking.")

# --- 1. DAFTAR FITUR ---
SELECTED_FEATURES = [
    'TyG_Index', 'Jenis_Kelamin', 'newhdl', 'Waist Measurement', 
    'Systolic', 'Weight', 'newtg', 'Diastolic', 'Usia', 'newua', 'Height'
]

# --- 2. MUAT MODEL ---
@st.cache_resource
def load_models():
    model_path = "model_deployments"
    try:
        scaler = joblib.load(os.path.join(model_path, 'scaler.pkl'))
        rf_model = joblib.load(os.path.join(model_path, 'rf_model.pkl'))
        mlp_model = joblib.load(os.path.join(model_path, 'mlp_model.pkl'))
        meta_model = joblib.load(os.path.join(model_path, 'meta_model.pkl'))
        return scaler, rf_model, mlp_model, meta_model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

scaler, rf_model, mlp_model, meta_model = load_models()

# --- 3. FUNGSI UNTUK TAMPILAN HASIL ---
def display_result(name, prob, class_pred):
    """Fungsi pembantu untuk menampilkan box hasil prediksi"""
    st.subheader(f"Hasil {name}")
    if class_pred == 1:
        st.error(f"**{name}: POSITIF METS**")
    else:
        st.success(f"**{name}: NEGATIF METS**")
    st.metric(label="Probabilitas", value=f"{prob:.4f}")

# --- 4. INTERFACE INPUT ---
st.subheader("📝 Masukkan Data Pasien")
col1, col2, col3 = st.columns(3)
user_input_dict = {}

with col1:
    user_input_dict['Usia'] = st.number_input("Usia (tahun)", 18, 100, 50)
    user_input_dict['Jenis_Kelamin'] = st.selectbox("Jenis Kelamin", [0, 1], format_func=lambda x: "Wanita" if x == 1 else "Pria")
    user_input_dict['Height'] = st.number_input("Tinggi Badan (m)", 1.0, 2.5, 1.65)
    user_input_dict['Weight'] = st.number_input("Berat Badan (kg)", 30.0, 200.0, 65.0)
with col2:
    user_input_dict['Waist Measurement'] = st.number_input("Lingkar Pinggang (cm)", 50.0, 150.0, 90.0)
    user_input_dict['Systolic'] = st.number_input("Systolic (mmHg)", 80.0, 250.0, 120.0)
    user_input_dict['Diastolic'] = st.number_input("Diastolic (mmHg)", 40.0, 150.0, 80.0)
    user_input_dict['TyG_Index'] = st.number_input("TyG Index", 5.0, 15.0, 8.0)
with col3:
    user_input_dict['newtg'] = st.number_input("Trigliserida", 50.0, 1000.0, 150.0)
    user_input_dict['newhdl'] = st.number_input("HDL", 10.0, 200.0, 45.0)
    user_input_dict['newua'] = st.number_input("Asam Urat", 1.0, 15.0, 5.0)

st.markdown("---")

# --- 5. LOGIKA PREDIKSI & TAB ---
if st.button("🚀 Jalankan Semua Prediksi"):
    input_df = pd.DataFrame([user_input_dict], columns=SELECTED_FEATURES)
    X_scaled = scaler.transform(input_df.values)
    
    with st.spinner('Menghitung prediksi...'):
        # Prediksi RF
        rf_prob = rf_model.predict_proba(X_scaled)[:, 1][0]
        rf_class = (rf_prob > 0.5).astype(int)
        
        # Prediksi MLP
        mlp_prob = mlp_model.predict_proba(X_scaled)[:, 1][0]
        mlp_class = (mlp_prob > 0.5).astype(int)
        
        # Prediksi Stacking (Meta Model)
        meta_X = np.column_stack((rf_prob, mlp_prob))
        stack_prob = meta_model.predict_proba(meta_X)[:, 1][0]
        stack_class = (stack_prob > 0.5).astype(int)

    # PEMBUATAN TAB
    tab1, tab2, tab3 = st.tabs(["🌳 Random Forest", "🧠 MLP Network", "🔗 Stacking Model"])

    with tab1:
        display_result("Random Forest", rf_prob, rf_class)
        st.info("Ini adalah hasil prediksi tunggal dari algoritma Random Forest.")

    with tab2:
        display_result("Multi-Layer Perceptron", mlp_prob, mlp_class)
        st.info("Ini adalah hasil prediksi tunggal dari arsitektur Neural Network (MLP).")

    with tab3:
        display_result("Stacking (Meta-Model)", stack_prob, stack_class)
        st.info("Stacking menggabungkan probabilitas dari RF dan MLP untuk keputusan akhir yang lebih kuat.")

# Footer
st.write("---")
st.caption("Dibuat oleh Abisatya")