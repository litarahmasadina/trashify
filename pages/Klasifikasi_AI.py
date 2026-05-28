import os
import streamlit as st
from PIL import Image
import numpy as np

# Import tf_keras 
from tensorflow import keras
import tensorflow as tf

st.set_page_config(page_title="Demo AI", page_icon="📸")
st.title("📸 Deteksi Kategori Sampah")

# Load model menggunakan tf_keras
@st.cache_resource
def load_keras_model():
    try:
        model = keras.models.load_model('model/model_trashid_v3.keras', compile=False)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

model = load_keras_model()
class_names = ['Anorganik', 'Organik', 'Residu'] 

uploaded_file = st.file_uploader("Unggah foto sampah (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Gambar yang diunggah.', width=300)
    
    if st.button("Analisis Gambar"):
        if model is None:
            st.error("⚠️ File model 'model_trashid_v3.keras' belum dimasukkan ke folder 'model'.")
        else:
            with st.spinner("AI sedang menganalisis..."):
                img_resized = image.resize((224, 224))
                img_array = np.array(img_resized)
                
                img_array = img_array / 255.0 
                img_array = np.expand_dims(img_array, axis=0)

                predictions = model.predict(img_array)
                score = tf.nn.softmax(predictions[0])
                predicted_class = class_names[np.argmax(score)]
                confidence = 100 * np.max(score)

                st.success(f"**Hasil Deteksi: {predicted_class}**")
                st.write(f"Tingkat Kepercayaan: {confidence:.2f}%")
