import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load model yang sudah dilatih
model = tf.keras.models.load_model('eye_disease_classification_model.h5')

# Daftar kelas penyakit mata
class_names = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']

# Fungsi preprocessing gambar
def preprocess_image(image):
    image = image.resize((224, 224))  # Sesuaikan ukuran input model
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    img_array = preprocess_input(img_array)
    return img_array

# Tampilan UI
st.markdown("<h3 style='text-align: center;'>Unggah Gambar Citra Mata</h3>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar Citra Fundus", use_column_width=True)

with col2:
    if uploaded_file:
        st.write("Pratinjau Gambar Citra Mata")
        st.image(image, use_column_width=True)

st.markdown("<div style='text-align: center; margin-top: 20px;'>", unsafe_allow_html=True)
if st.button("Deteksi Sekarang"):
    if uploaded_file is None:
        st.warning("Silakan unggah gambar terlebih dahulu.")
    else:
        img = preprocess_image(image)
        prediction = model.predict(img)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)
        st.success(f"🧠 Prediksi: **{predicted_class}**")
        st.info(f"📈 Tingkat Keyakinan: **{confidence * 100:.2f}%**")
st.markdown("</div>", unsafe_allow_html=True)
