# app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from PIL import Image
import os

# Page Configuration

st.set_page_config(
    page_title="Rare Neurological Disease Classifier",
    page_icon="🧠",
    layout="wide"
)

# Sidebar

st.sidebar.title("About This Project")
st.sidebar.info("""
**Neurological Disease Classification** using ResNet50.

Classifies images into 5 categories:  
Fukuyama Muscular Dystrophy, Hallervorden-Spatz Disease, Moyamoya Disease, Pachygyria & Cerebellar Hypoplasia, Walker-Warburg Syndrome.

Preprocessing: resized to 224×224, normalized with ResNet50 preprocessing.

Test Accuracy: 97.14%. Upload an image to see the predicted class and confidence.

Purpose: Demonstrates model selection and real-time prediction for a college project.
""")

# Header

st.title("🧠 Rare Neurological Disease Classifier")
st.subheader("Using ResNet50 (Best Model)")
st.markdown("---")

st.markdown("## 🔗 Useful Links")

col1, col2 = st.columns(2)

with col1:
    st.link_button(
        "📊 View Kaggle training notebook",
        "https://www.kaggle.com/code/xr1nc3/efficientnetb0-62-86-vs-resnet50-97-14"
    )

with col2:
    st.link_button(
        "📥 Download sample images to test model (Dataset)",
        "https://www.kaggle.com/datasets/ahsanneural/rare-neurological-diseases-mri-curated-edition"
    )

import gdown
import os

model_path = "final_model.keras"

if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?id=1t6LKd1f7NgXdeM9ohuX7dcHCujeeE5sP"
    gdown.download(url, model_path, quiet=False)

# Load Model

import keras

model = keras.models.load_model(model_path, compile=False)


# Class Names

class_names = [
    'Fukuyama Muscular Dystrophy',
    'Hallervorden Spatz Disease',
    'Moyamoya Disease',
    'Pachygyria Cerebellar Hypoplasia',
    'Walker Warburg Syndrome'
    ]

# Upload Image

uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=["jpg", "jpeg", "png"],
    key="resnet_file_uploader"
)

if uploaded_file is not None:
    # Create two columns: left = image, right = prediction
    col1, col2 = st.columns([1, 1])
    
    # Left Column: Show Image
    
    with col1:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption='Uploaded Image', width=500)

    # Preprocess Image

    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction[0])
    confidence = np.max(prediction[0])
    predicted_class = class_names[class_idx]

    # Right Column: Show Result
    
    with col2:
        st.markdown("### Prediction Result")
        st.success(f"Predicted Class: **{predicted_class}**")
        st.info(f"Confidence: {confidence*100:.2f}%")

        # Optional: Show all class probabilities in expandable section
        with st.expander("Show Full Class Probabilities"):
            for i, prob in enumerate(prediction[0]):
                st.write(f"{class_names[i]}: {prob*100:.2f}%")