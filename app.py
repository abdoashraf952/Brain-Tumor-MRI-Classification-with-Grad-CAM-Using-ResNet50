import streamlit as st
import numpy as np
import tensorflow as tf
import cv2

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.applications.resnet50 import preprocess_input

# ===============================
# Page config
# ===============================
st.set_page_config(
    page_title="Brain Tumor MRI Classification",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Tumor MRI Classification with Grad-CAM")
st.write("Upload an MRI image and see the prediction + where the model focused.")

# ===============================
# Classes
# ===============================
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# ===============================
# Load model (cached)
# ===============================
@st.cache_resource
def load_model_cached():
    return tf.keras.models.load_model(
        "brain_tumor_resnet50_with_preprocessing.keras",
        custom_objects={"preprocess_input": preprocess_input}
    )

model = load_model_cached()

# ===============================
# Build Grad-CAM models (cached)
# ===============================
@st.cache_resource
def build_gradcam_models(model):
    seq = model.get_layer('sequential')
    backbone = seq.get_layer('resnet50')
    last_conv = backbone.get_layer("conv5_block3_out")

    # Feature extractor
    last_conv_model = Model(backbone.input, last_conv.output)

    # Classifier head
    classifier_layer_names = [
        "global_average_pooling2d",
        "dense",
        "batch_normalization",
        "dropout",
        "dense_1",
        "batch_normalization_1",
        "dropout_1",
        "output"
    ]

    classifier_input = Input(shape=last_conv.output.shape[1:])
    x = classifier_input
    for name in classifier_layer_names:
        x = seq.get_layer(name)(x)

    classifier_model = Model(classifier_input, x)

    return last_conv_model, classifier_model

last_conv_model, classifier_model = build_gradcam_models(model)

# ===============================
# Grad-CAM functions
# ===============================
def make_gradcam(img_array):
    with tf.GradientTape() as tape:
        conv_out = last_conv_model(img_array)
        tape.watch(conv_out)

        preds = classifier_model(conv_out)
        top_idx = tf.argmax(preds[0])
        top_class = preds[:, top_idx]

    grads = tape.gradient(top_class, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2)).numpy()

    conv_out = conv_out.numpy()[0]
    for i in range(pooled_grads.shape[-1]):
        conv_out[:, :, i] *= pooled_grads[i]

    heatmap = np.sum(conv_out, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    return heatmap


def overlay_heatmap(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = np.uint8(heatmap * alpha + img)
    return superimposed

# ===============================
# Upload image
# ===============================
uploaded_file = st.file_uploader(
    "📤 Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Show original image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, 1)
    st.image(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB),
             caption="Uploaded MRI Image",
             use_container_width=True)

    # ===============================
    # Preprocess
    # ===============================
    img = cv2.resize(original_img, (224, 224))
    img_array = np.expand_dims(img, axis=0)
    img_array = preprocess_input(img_array)

    # ===============================
    # Prediction
    # ===============================
    with st.spinner("🧠 Analyzing MRI..."):
        preds = model.predict(img_array)
        predicted_class = class_names[np.argmax(preds)]
        confidence = np.max(preds) * 100

    st.success("✅ Prediction Completed")
    st.markdown(f"### 🧬 Predicted Tumor Type: **{predicted_class.upper()}**")
    st.markdown(f"### 📊 Confidence: **{confidence:.2f}%**")

    # Probabilities
    st.subheader("📈 Class Probabilities")
    for i, name in enumerate(class_names):
        st.write(f"{name}: {preds[0][i]*100:.2f}%")
        st.progress(float(preds[0][i]))

    # ===============================
    # Grad-CAM
    # ===============================
    st.subheader("🧭 Model Attention (Grad-CAM)")
    heatmap = make_gradcam(img_array)
    gradcam_img = overlay_heatmap(img, heatmap)

    st.image(cv2.cvtColor(gradcam_img, cv2.COLOR_BGR2RGB),
             caption="Grad-CAM Visualization",
             use_container_width=True)

else:
    st.info("👆 Please upload an MRI image to start.")
