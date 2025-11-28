# app.py (Streamlit App)
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Page config
st.set_page_config(
    page_title="Aerial Object Classifier",
    page_icon="🦅",
    layout="wide"
)

# Title
st.title("🦅 Aerial Object Classification & Detection")
st.markdown("**Upload an image to classify as Bird or Drone, or use YOLOv8 for detection.**")

# Sidebar
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Custom CNN", "MobileNetV2", "ResNet50", "EfficientNetB0", "YOLOv8 Detection"]
)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)

# Load classification models
@st.cache_resource
def load_classification_model(model_name):
    model_paths = {
        "Custom CNN": "models/custom_cnn_best.h5",
        "MobileNetV2": "models/mobilenet_best.h5",
        "ResNet50": "models/resnet50_best.h5",
        "EfficientNetB0": "models/efficientnet_best.h5"
    }
    return tf.keras.models.load_model(model_paths[model_name])

# Load YOLO model
@st.cache_resource
def load_yolo_model():
    from ultralytics import YOLO
    return YOLO("models/yolov8_aerial/weights/best.pt")

# Preprocess image for classification
def preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Classification function
def classify_image(model, image):
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img, verbose=0)[0][0]

    if prediction > confidence_threshold:
        label = "Drone"
        confidence = prediction
    else:
        label = "Bird"
        confidence = 1 - prediction
    
    return label, confidence

# YOLO detection function
def detect_objects_yolo(model, image):
    img_array = np.array(image)
    results = model.predict(
        source=img_array,
        conf=confidence_threshold,
        verbose=False
    )
    annotated_img = results[0].plot()  # BGR array
    return annotated_img, results[0]

# File uploader
st.markdown("---")
uploaded_file = st.file_uploader(
    "Upload an aerial image",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear image of a bird or drone"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # YOLO detection
    if model_choice == "YOLOv8 Detection":
        st.subheader("🔍 YOLOv8 Object Detection Results")
        model = load_yolo_model()
        annotated_img, results = detect_objects_yolo(model, image)
        st.image(annotated_img, caption="YOLOv8 Detection Output", use_column_width=True)

        # Show detected boxes
        st.write("Detection Data:")
        st.write(results.boxes.data.tolist())

    # CNN classification
    else:
        st.subheader("🔎 CNN Classification Results")
        model = load_classification_model(model_choice)
        label, confidence = classify_image(model, image)

        st.success(f"Prediction: **{label}** (Confidence: **{confidence:.3f}**)")


