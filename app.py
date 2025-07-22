# ---------------------------------------------
# Streamlit GUI for Plant Disease Detection
# Author: Bokketi Pranitha
# This app uses a trained CNN (Keras with TensorFlow)
# to classify plant leaf diseases from uploaded images.
# ---------------------------------------------

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Load the trained model
model = load_model("plant_disease_model.h5")

# Class names (adjust based on your dataset)
class_names = ['Tomato_Healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Potato_Healthy']

# Streamlit UI
st.set_page_config(page_title="🌿 Plant Disease Detector", layout="centered")
st.title("🌱 Plant Disease Detection from Leaf Image")

uploaded_file = st.file_uploader("Upload a Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = load_img(uploaded_file, target_size=(128, 128))  # same size used during training
   
    st.image(image, caption='Uploaded Leaf Image', use_container_width=True)

    # Preprocess the image
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Show result
    st.success(f"Prediction: **{predicted_class}**")
    st.info(f"Confidence: {confidence:.2f}")
