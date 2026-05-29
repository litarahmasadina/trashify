import os
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

st.set_page_config(page_title="Demo AI", page_icon="📸")
st.title("📸 Deteksi Kategori Sampah")

# Load model
@st.cache_resource
def load_keras_model():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))

        model_path = os.path.join(
            current_dir,
            "..",
            "model",
            "model_trashid_v3.keras"
        )

        # Pakai tf.keras langsung
        model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

        return model

    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None


model = load_keras_model()

class_names = ['Anorganik', 'Organik', 'Residu']

uploaded_file = st.file_uploader(
    "Unggah foto sampah (JPG/PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Gambar yang diunggah",
        width=300
    )

    if st.button("Analisis Gambar"):

        if model is None:

            st.error(
                "⚠️ Model gagal dimuat. "
                "Pastikan file model ada di folder model/"
            )

        else:

            with st.spinner("AI sedang menganalisis..."):

                # Resize gambar
                img_resized = image.resize((224, 224))

                # Convert ke array
                img_array = np.array(img_resized)

                # Normalisasi
                img_array = img_array.astype("float32") / 255.0

                # Tambah batch dimension
                img_array = np.expand_dims(img_array, axis=0)

                # Prediksi
                predictions = model.predict(img_array)

                predicted_index = np.argmax(predictions[0])

                confidence = float(
                    np.max(predictions[0]) * 100
                )

                predicted_class = class_names[predicted_index]

                st.success(
                    f"**Hasil Deteksi: {predicted_class}**"
                )

                st.write(
                    f"Tingkat Kepercayaan: {confidence:.2f}%"
                )
