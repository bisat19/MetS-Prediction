import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time

# --- PENGATURAN HALAMAN DAN JUDUL ---
st.set_page_config(
    page_title="Prediksi Sindrom Metabolik (MetS) Stacking Model",
    layout="wide"
)
st.title("🔬 Prediksi Sindrom Metabolik (MetS)")
st.markdown("Aplikasi Web Berbasis Model Stacking untuk Prediksi Sindrom Metabolik (MetS)")

# --- 1. DAFTAR FITUR TERPILIH ---
SELECTED_FEATURES = [
    'TyG_Index', 'Jenis_Kelamin', 'newhdl', 'Waist Measurement', 
    'Systolic', 'Weight', 'newtg', 'Diastolic', 'Usia', 'newua', 'Height'
]

# --- 2. FUNGSI MUAT MODEL ---
# Menggunakan cache Streamlit agar model hanya dimuat sekali
@st.cache_resource
def load_models():
    """Memuat semua model dan preprocessor yang sudah di-fit."""
    model_path = "model_deployments"
    try:
        scaler = joblib.load(os.path.join(model_path, 'scaler.pkl'))
        rf_model = joblib.load(os.path.join(model_path, 'rf_model.pkl'))
        mlp_model = joblib.load(os.path.join(model_path, 'mlp_model.pkl'))
        meta_model = joblib.load(os.path.join(model_path, 'meta_model.pkl'))
        return scaler, rf_model, mlp_model, meta_model
    
    except FileNotFoundError as e:
        st.error(f"File model tidak ditemukan: {e}. Pastikan semua file .pkl ada di folder '{model_path}'.")
        st.stop()
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

scaler, rf_model, mlp_model, meta_model = load_models()


# --- 3. FUNGSI INFERENSI/PREDIKSI ---
def predict_mets(input_data_df, scaler, rf_model, mlp_model, meta_model):
    """Melakukan prediksi menggunakan alur Stacking Model."""
    X_unscaled = input_data_df.values
    X_scaled = scaler.transform(X_unscaled) 

    rf_user_prob = rf_model.predict_proba(X_scaled)[:, 1]
    mlp_user_prob = mlp_model.predict_proba(X_scaled)[:, 1]

    meta_X_user = np.column_stack((rf_user_prob, mlp_user_prob))

    final_prediction_prob = meta_model.predict_proba(meta_X_user)[:, 1]
    final_prediction_class = (final_prediction_prob > 0.5).astype(int)
    
    return final_prediction_class[0], final_prediction_prob[0]


# --- 4. INTERFACE INPUT STREAMLIT ---
st.subheader("Masukkan Nilai 11 Fitur Pasien")

# ***PERUBAHAN UTAMA DI SINI: MEMBUAT 4 KOLOM***
# Input dibagi ke col1, col2, col3. Hasil prediksi akan ditampilkan di col4.
col1, col2, col3, col4 = st.columns([1, 1, 1, 1.2]) # Memberi bobot sedikit lebih besar ke kolom hasil

# Dictionary untuk menyimpan input pengguna
user_input_dict = {}

# --- KOLOM 1: Input ---
with col1:
    user_input_dict['Usia'] = st.number_input("Usia (tahun)", min_value=18, max_value=100, value=50, step=1)
    user_input_dict['Jenis_Kelamin'] = st.selectbox("Jenis Kelamin", options=[0, 1], format_func=lambda x: "Wanita" if x == 1 else "Pria")
    user_input_dict['Height'] = st.number_input("Tinggi Badan (m)", min_value=1.00, max_value=2.50, value=1.65, step=0.01)
    user_input_dict['Weight'] = st.number_input("Berat Badan (kg)", min_value=30.0, max_value=200.0, value=65.0, step=0.1)

# --- KOLOM 2: Input ---
with col2:
    user_input_dict['Waist Measurement'] = st.number_input("Lingkar Pinggang (cm)", min_value=50.0, max_value=150.0, value=90.0, step=0.1)
    user_input_dict['Systolic'] = st.number_input("Systolic (mmHg)", min_value=80.0, max_value=250.0, value=120.0, step=1.0)
    user_input_dict['Diastolic'] = st.number_input("Diastolic (mmHg)", min_value=40.0, max_value=150.0, value=80.0, step=1.0)
    user_input_dict['TyG_Index'] = st.number_input("TyG Index", min_value=5.0, max_value=15.0, value=8.0, step=0.1)

# --- KOLOM 3: Input ---
with col3:
    user_input_dict['newtg'] = st.number_input("Trigliserida (newtg)", min_value=50.0, max_value=1000.0, value=150.0, step=1.0)
    user_input_dict['newhdl'] = st.number_input("HDL (newhdl)", min_value=10.0, max_value=200.0, value=45.0, step=1.0)
    user_input_dict['newua'] = st.number_input("Asam Urat (newua)", min_value=1.0, max_value=15.0, value=5.0, step=0.1)


# --- 5. LOGIKA PREDIKSI & TOMBOL ---

# Tombol Prediksi diletakkan di bawah input, namun hasil akan ditampilkan di col4
st.markdown("---")
if st.button("Prediksi Status MetS"):
    
    # 1. Konversi input ke DataFrame
    try:
        input_df = pd.DataFrame([user_input_dict], columns=SELECTED_FEATURES)
    except Exception as e:
        st.error("Gagal memproses input. Pastikan semua kolom terisi dengan benar.")
        # Pesan error akan muncul di bagian utama, bukan di col4
        st.stop()

    # 2. Jalankan prediksi
    with st.spinner('Memproses data dan menjalankan model stacking...'):
        time.sleep(1) 
        predicted_class, predicted_prob = predict_mets(input_df, scaler, rf_model, mlp_model, meta_model)

    # 3. Tampilkan Hasil di KOLOM 4
    with col4:
        st.subheader("✅ Hasil Prediksi")
        if predicted_class == 1:
            st.info("STATUS: BERPOTENSI METABOLIK SINDROM (METS)")
            st.metric(label="Probabilitas MetS (Kelas 1)", value=f"{predicted_prob:.4f}")
        else:
            st.info("STATUS: NON-SINDROM METABOLIK")
            st.metric(label="Probabilitas MetS (Kelas 1)", value=f"{predicted_prob:.4f}")
            
# Footer/Catatan Kaki
st.write("---")
st.caption("Catatan: Prediksi ini dihasilkan oleh model Machine Learning dan bukan merupakan diagnosis medis. Selalu konsultasikan dengan profesional kesehatan.")
st.write("---")
st.caption("dibuat oleh Abisatya")