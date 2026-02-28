import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- PENGATURAN HALAMAN ---
st.set_page_config(page_title="MetS Multi-Model Independent", layout="wide")
st.title("🔬 Analisis Mandiri Sindrom Metabolik")
st.markdown("""
    <style>
    .stTooltipIcon { cursor: help; }
    </style>
""", unsafe_allow_html=True)

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

# --- 2. INISIALISASI SESSION STATE ---
DEFAULT_INPUTS = {
    'Usia': 50, 'Jenis_Kelamin': 0, 'Height': 1.65, 'Weight': 65.0,
    'Waist Measurement': 90.0, 'Systolic': 120.0, 'Diastolic': 80.0,
    'TyG_Index': 8.0, 'newtg': 150.0, 'newhdl': 45.0, 'newua': 5.0
}

for key, value in DEFAULT_INPUTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- 3. FUNGSI HELPER UNTUK TOOLTIP ---
def tooltip_help(text):
    """Membuat ikon help dengan tooltip"""
    return f"""
    <span style="cursor: help; color: #0068c9;" title="{text}">ⓘ</span>
    """

# --- 4. FUNGSI INPUT COMPONENT ---
def get_user_inputs():
    """Form input dengan label Bahasa Indonesia + tooltip penjelasan"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.number_input(
            "Usia (tahun)", 
            min_value=18, max_value=100, 
            key="Usia",
            help="Usia pasien dalam tahun"
        )
        st.selectbox(
            "Jenis Kelamin", 
            options=[0, 1], 
            format_func=lambda x: "Wanita" if x == 1 else "Pria",
            key="Jenis_Kelamin",
            help="Pilih jenis kelamin pasien"
        )
        st.number_input(
            "Tinggi Badan (m)", 
            min_value=1.0, max_value=2.5, step=0.01,
            key="Height",
            help="Tinggi badan dalam meter (contoh: 1.65)"
        )
        st.number_input(
            "Berat Badan (kg)", 
            min_value=30.0, max_value=200.0, step=0.1,
            key="Weight",
            help="Berat badan dalam kilogram"
        )
    
    with col2:
        st.number_input(
            "Lingkar Pinggang (cm)", 
            min_value=50.0, max_value=150.0, step=0.1,
            key="Waist Measurement",
            help="Lingkar pinggang diukur pada titik tersempit (dalam cm)"
        )
        st.number_input(
            "Tekanan Darah Sistolik (mmHg)", 
            min_value=80.0, max_value=250.0, step=1.0,
            key="Systolic",
            help="Tekanan darah saat jantung berdetak (angka atas)"
        )
        st.number_input(
            "Tekanan Darah Diastolik (mmHg)", 
            min_value=40.0, max_value=150.0, step=1.0,
            key="Diastolic",
            help="Tekanan darah saat jantung beristirahat (angka bawah)"
        )
    
    with col3:
        st.number_input(
            "Indeks TyG (Triglyceride-Glucose) 📊", 
            min_value=5.0, max_value=15.0, step=0.01,
            key="TyG_Index",
            help="Indikator resistensi insulin. Dihitung dari: ln[(Trigliserida mg/dL × Glukosa mg/dL)/2]. Nilai >8.5 mengindikasikan risiko tinggi."
        )
        st.number_input(
            "Trigliserida (mg/dL) 🩸", 
            min_value=50.0, max_value=1000.0, step=1.0,
            key="newtg",
            help="Kadar lemak dalam darah. Normal: <150 mg/dL"
        )
        st.number_input(
            "HDL - Kolesterol 'Baik' (mg/dL) 💙", 
            min_value=10.0, max_value=200.0, step=1.0,
            key="newhdl",
            help="High-Density Lipoprotein. Semakin tinggi semakin baik. Minimal: Pria >40, Wanita >50 mg/dL"
        )
        st.number_input(
            "Asam Urat (mg/dL) 🦴", 
            min_value=1.0, max_value=15.0, step=0.1,
            key="newua",
            help="Kadar asam urat dalam darah. Normal: Pria 3.4-7.0, Wanita 2.4-6.0 mg/dL"
        )
    
    # Ambil semua nilai dari session_state
    return {key: st.session_state[key] for key in DEFAULT_INPUTS.keys()}

# --- 5. INPUT FORM DI LUAR TAB ---
st.subheader("📋 Input Data Pasien")
st.markdown("""
    <div style="background-color: #e8f4fd; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
    <strong>💡 Petunjuk:</strong> Isi data sesuai hasil pemeriksaan terakhir. 
    Klik ikon <span style="color: #0068c9;">ⓘ</span> untuk melihat penjelasan setiap parameter.
    </div>
""", unsafe_allow_html=True)

user_data = get_user_inputs()

# Tombol reset
col_reset, col_empty = st.columns([1, 5])
with col_reset:
    if st.button("🔄 Reset Input"):
        for key, value in DEFAULT_INPUTS.items():
            st.session_state[key] = value
        st.rerun()

st.divider()

# --- 6. PEMBUATAN TABS ---
tab1, tab2, tab3 = st.tabs(["🌳 Random Forest", "🧠 MLP Network", "🔗 Stacking Model"])

# --- TAB 1: RANDOM FOREST ---
with tab1:
    st.header("🌳 Prediksi: Random Forest")
    st.caption("Model ensemble berbasis pohon keputusan yang robust terhadap outlier.")
    
    if st.button("▶️ Hitung Prediksi", key="btn_rf", type="primary"):
        with st.spinner("🔄 Sedang memproses prediksi..."):
            input_df = pd.DataFrame([user_data], columns=SELECTED_FEATURES)
            X_scaled = scaler.transform(input_df.values)
            prob = rf_model.predict_proba(X_scaled)[:, 1][0]
            
            st.write("---")
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.metric("📊 Probabilitas MetS", f"{prob:.2%}")
                st.progress(prob)
            with col_res2:
                if prob > 0.5: 
                    st.error("🚨 Hasil: **POSITIF** Sindrom Metabolik")
                    st.caption("Disarankan konsultasi ke dokter untuk evaluasi lebih lanjut.")
                else: 
                    st.success("✅ Hasil: **NON-Sindrom** Metabolik")
                    st.caption("Pertahankan pola hidup sehat!")

# --- TAB 2: MLP ---
with tab2:
    st.header("🧠 Prediksi: MLP Neural Network")
    st.caption("Model deep learning dengan arsitektur multi-layer perceptron.")
    
    if st.button("▶️ Hitung Prediksi", key="btn_mlp", type="primary"):
        with st.spinner("🔄 Sedang memproses prediksi..."):
            input_df = pd.DataFrame([user_data], columns=SELECTED_FEATURES)
            X_scaled = scaler.transform(input_df.values)
            prob = mlp_model.predict_proba(X_scaled)[:, 1][0]
            
            st.write("---")
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.metric("📊 Probabilitas MetS", f"{prob:.2%}")
                st.progress(prob)
            with col_res2:
                if prob > 0.5: 
                    st.error("🚨 Hasil: **POSITIF** Sindrom Metabolik")
                    st.caption("Disarankan konsultasi ke dokter untuk evaluasi lebih lanjut.")
                else: 
                    st.success("✅ Hasil: **NON-Sindrom** Metabolik")
                    st.caption("Pertahankan pola hidup sehat!")

# --- TAB 3: STACKING ---
with tab3:
    st.header("🔗 Prediksi: Stacking Ensemble Model")
    st.info("""
    **Bagaimana cara kerja Stacking?**  
    Model ini menggabungkan prediksi dari Random Forest dan MLP Neural Network, 
    kemudian diproses oleh Meta-Model untuk menghasilkan prediksi akhir yang lebih akurat.
    """)
    
    if st.button("▶️ Hitung Prediksi", key="btn_stack", type="primary"):
        with st.spinner("🔄 Sedang memproses prediksi ensemble..."):
            input_df = pd.DataFrame([user_data], columns=SELECTED_FEATURES)
            X_scaled = scaler.transform(input_df.values)
            
            # Alur Stacking
            rf_p = rf_model.predict_proba(X_scaled)[:, 1]
            mlp_p = mlp_model.predict_proba(X_scaled)[:, 1]
            meta_X = np.column_stack((rf_p, mlp_p))
            prob = meta_model.predict_proba(meta_X)[:, 1][0]
            
            st.write("---")
            # Tampilkan kontribusi masing-masing model
            st.subheader("📈 Kontribusi Model")
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("🌳 Random Forest", f"{rf_p[0]:.2%}")
            col_m2.metric("🧠 MLP Network", f"{mlp_p[0]:.2%}")
            
            st.write("---")
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.metric("🎯 Final Probabilitas", f"{prob:.2%}")
                st.progress(prob)
            with col_res2:
                if prob > 0.5: 
                    st.error("🚨 Hasil Akhir: **POSITIF** Sindrom Metabolik")
                    st.caption("Disarankan konsultasi ke dokter untuk evaluasi lebih lanjut.")
                else: 
                    st.success("✅ Hasil Akhir: **NON-Sindrom** Metabolik")
                    st.caption("Pertahankan pola hidup sehat!")

# --- FOOTER ---
st.write("---")
st.caption("🔬 Dibuat oleh Abisatya | Model untuk tujuan edukasi dan skrining awal. Konsultasikan hasil ke tenaga medis profesional.")