import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import datetime
import os

# === Setup ===
IMG_SIZE = (128, 128)
CLASS_NAMES = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']

st.set_page_config(page_title="Deteksi Penyakit Mata", layout="centered")

# === Load Model ===
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('eye_disease_classification_model.h5', compile=False)
    return model

model = load_model()

# === UI: Upload ===
st.markdown("## Unggah Gambar Citra Mata", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    detect_button = st.button("🔍 Deteksi Sekarang", use_container_width=True)

with col2:
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_container_width=True)

# === Deteksi ===
if uploaded_file and detect_button:
    with st.spinner('Memproses gambar...'):
        img_resized = image.resize(IMG_SIZE)
        img_array = np.array(img_resized) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

    # Scroll otomatis ke hasil
    st.markdown("---")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Gambar yang Diperiksa", width=300)

    with col2:
        st.markdown("### Hasil Deteksi")
        st.markdown("Kemungkinan: ")
        st.markdown(f"**<span style='font-size: 26px;'>{predicted_class}</span>**", unsafe_allow_html=True)
        st.markdown(f"<span style='font-size: 32px; color: red; font-weight: bold;'>{confidence:.0f}%</span>", unsafe_allow_html=True)

    # Tombol simpan hasil dan deteksi ulang
    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("💾 Simpan Hasil"):
            # Membuat direktori jika tidak ada
            if not os.path.exists("riwayat_deteksi"):
                os.makedirs("riwayat_deteksi")
            
            # Menyimpan hasil ke dalam file .txt
            file_path = os.path.join("riwayat_deteksi", "riwayat_deteksi.txt")
            with open(file_path, "a") as f:
                f.write(f"{datetime.datetime.now()}: {predicted_class} ({confidence:.2f}%)\n")
            st.success(f"Hasil berhasil disimpan di {file_path}!")

    with colB:
        if st.button("🔁 Deteksi Ulang"):
            st.experimental_rerun()
