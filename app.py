"""Streamlit inference interface for trained brain MRI image classifiers."""
import json
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

IMG_SIZE = (224, 224)
MODEL_OPTIONS = {
    "MobileNetV2 (Transfer Learning)": Path("models/mobilenetv2_transfer.keras"),
    "Custom CNN": Path("models/custom_cnn.h5"),
}

st.set_page_config(page_title="Brain MRI Classifier", page_icon="🧠", layout="centered")
st.title("🧠 Brain Tumor MRI Image Classification")
st.write("Upload an MRI image to see the selected model's predicted dataset class and scores.")
st.warning(
    "Educational demonstration only — not a medical device. This prediction is not a diagnosis "
    "and must not guide treatment or triage. Consult a qualified medical professional."
)

classes_path = Path("models/class_names.json")
if not classes_path.exists():
    st.error("Class names are missing. Train a model first; training creates models/class_names.json.")
    st.stop()

with classes_path.open(encoding="utf-8") as f:
    class_names = json.load(f)

available = {name: path for name, path in MODEL_OPTIONS.items() if path.exists()}
if not available:
    st.info("No trained model was found. Download and arrange the dataset, then run the training commands in README.md.")
    st.stop()

default_model = next((name for name in available if name.startswith("MobileNetV2")), next(iter(available)))
choice = st.selectbox("Model", list(available.keys()), index=list(available.keys()).index(default_model))
uploaded = st.file_uploader("Upload a brain MRI image", type=["jpg", "jpeg", "png"])

@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

if uploaded:
    try:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded MRI image", use_container_width=True)
        resized = image.resize(IMG_SIZE)
        array = np.asarray(resized, dtype=np.float32)
        batch = np.expand_dims(array, axis=0)
        model = load_model(str(available[choice]))
        scores = model.predict(batch, verbose=0)[0]
        if len(scores) != len(class_names):
            st.error("The selected model's output count does not match class_names.json. Retrain models together.")
            st.stop()
        index = int(np.argmax(scores))
        st.subheader("Prediction")
        st.metric("Predicted dataset class", class_names[index])
        st.caption(f"Highest model score: {scores[index]:.1%} (not a calibrated clinical probability)")
        st.subheader("Scores for each class")
        order = np.argsort(scores)[::-1]
        for i in order:
            st.write(f"**{class_names[int(i)]}** — {scores[int(i)]:.1%}")
            st.progress(float(scores[int(i)]))
    except Exception as exc:
        st.error(f"Could not process this image: {exc}")
else:
    st.caption("Supported formats: JPG, JPEG, PNG.")
