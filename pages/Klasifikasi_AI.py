```python
import os
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# =========================
# CONFIG PAGE
# =========================
st.set_page_config(
    page_title="Demo AI",
    page_icon="📸"
)

st.title("📸 Deteksi Kategori Sampah")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_keras_model():
    try:

        # Lokasi folder saat ini
        current_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        # Path model
        model_path = os.path.join(
            current_dir,
            "..",
            "model",
            "model_trashid_v3.keras"
        )

        # DEBUG PATH
        st.write("📂 Path model:", model_path)

        # Cek file ada atau tidak
        if not os.path.exists(model_path):
            st.error("❌ File model tidak ditemukan!")
            return None

        st.success("✅ File model ditemukan!")

        # Load model
        model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

        st.success("✅ Model berhasil dimuat!")

        return model

    except Exception as e:

        st.error(f"❌ Gagal memuat model: {e}")

        return None


# Load model
model = load_keras_model()

# =========================
# CLASS LABEL
# =========================
class_names = [
    "Anorganik",
    "Organik",
    "Residu"
]

# =========================
# UPLOAD GAMBAR
# =========================
uploaded_file = st.file_uploader(
    "Unggah foto sampah (JPG/PNG)",
    type=["jpg", "jpeg", "png"]
)

# =========================
# PROSES PREDIKSI
# =========================
if uploaded_file is not None:

    # Buka gambar
    image = Image.open(
        uploaded_file
    ).convert("RGB")

    # Tampilkan gambar
    st.image(
        image,
        caption="Gambar yang diunggah",
        width=300
    )

    # Tombol prediksi
    if st.button("Analisis Gambar"):

        # Kalau model gagal dimuat
        if model is None:

            st.error(
                "⚠️ Model gagal dimuat."
            )

        else:

            with st.spinner(
                "AI sedang menganalisis..."
            ):

                # Resize
                img_resized = image.resize(
                    (224, 224)
                )

                # Convert array
                img_array = np.array(
                    img_resized
                )

                # Normalisasi
                img_array = (
                    img_array.astype("float32")
                    / 255.0
                )

                # Tambah dimensi batch
                img_array = np.expand_dims(
                    img_array,
                    axis=0
                )

                # Prediksi
                predictions = model.predict(
                    img_array
                )

                # Ambil hasil
                predicted_index = np.argmax(
                    predictions[0]
                )

                confidence = float(
                    np.max(predictions[0]) * 100
                )

                predicted_class = class_names[
                    predicted_index
                ]

                # Output
                st.success(
                    f"🗑️ Hasil Deteksi: "
                    f"{predicted_class}"
                )

                st.write(
                    f"🎯 Tingkat Kepercayaan: "
                    f"{confidence:.2f}%"
                )
```
