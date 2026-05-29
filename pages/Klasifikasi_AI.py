```python
import os
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Demo AI",
    page_icon="📸"
)

st.title("📸 Deteksi Kategori Sampah")

# ======================
# LOAD MODEL
# ======================
@st.cache_resource
def load_keras_model():

    try:

        current_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        model_path = os.path.join(
            current_dir,
            "..",
            "model",
            "model_trashid_v3.keras"
        )

        st.write("📂 Path Model:", model_path)

        # cek file ada atau tidak
        if not os.path.exists(model_path):

            st.error("❌ File model tidak ditemukan!")

            return None

        st.success("✅ File model ditemukan!")

        # load model
        model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

        st.success("✅ Model berhasil dimuat!")

        return model

    except Exception as e:

        st.error(f"❌ Error load model: {e}")

        return None


# ======================
# LOAD MODEL KE MEMORY
# ======================
model = load_keras_model()

# ======================
# LABEL KELAS
# ======================
class_names = [
    "Anorganik",
    "Organik",
    "Residu"
]

# ======================
# UPLOAD FILE
# ======================
uploaded_file = st.file_uploader(
    "Unggah foto sampah",
    type=["jpg", "jpeg", "png"]
)

# ======================
# PREDIKSI
# ======================
if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Gambar yang diunggah",
        width=300
    )

    if st.button("Analisis Gambar"):

        if model is None:

            st.error(
                "⚠️ Model gagal dimuat"
            )

        else:

            with st.spinner(
                "AI sedang menganalisis..."
            ):

                # resize gambar
                img_resized = image.resize(
                    (224, 224)
                )

                # convert array
                img_array = np.array(
                    img_resized
                )

                # normalisasi
                img_array = (
                    img_array.astype("float32")
                    / 255.0
                )

                # tambah dimensi
                img_array = np.expand_dims(
                    img_array,
                    axis=0
                )

                # prediksi
                predictions = model.predict(
                    img_array
                )

                predicted_index = np.argmax(
                    predictions[0]
                )

                confidence = float(
                    np.max(predictions[0]) * 100
                )

                predicted_class = class_names[
                    predicted_index
                ]

                # output
                st.success(
                    f"🗑️ Hasil Deteksi: {predicted_class}"
                )

                st.write(
                    f"🎯 Tingkat Kepercayaan: "
                    f"{confidence:.2f}%"
                )
```
