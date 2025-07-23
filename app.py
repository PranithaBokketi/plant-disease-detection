import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import os

# Title and description
st.set_page_config(page_title="Plant Disease Detection", layout="centered")
st.title(" Plant Disease Detection")
st.write("Upload a leaf image to detect plant diseases using a CNN model built with Keras.")

# Load the model
@st.cache_resource
def load_cnn_model():
    try:
        model = load_model("plant_disease_model.h5")
        return model
    except Exception as e:
        st.error(f" Error loading model: {e}")
        return None

model = load_cnn_model()

# Class names - ensure this matches your training classes
class_names = ['Apple Scab', 'Apple Black Rot', 'Corn Gray Leaf Spot', 'Healthy', 'Potato Early Blight', 'Potato Late Blight']  # example classes

# File uploader
uploaded_file = st.file_uploader(" Upload Leaf Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Load and display image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Leaf Image", use_container_width=True)

        # Preprocess
        

        img = image.resize((128, 128))  # match the model's expected input
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # shape becomes (1, 128, 128, 3)



        # Predict
        prediction = model.predict(img_array)

        if prediction is not None and prediction.shape[0] > 0:
            predicted_class = class_names[np.argmax(prediction[0])]
            confidence = np.max(prediction[0]) * 100
            st.success(f" Predicted: **{predicted_class}** with {confidence:.2f}% confidence.")
        else:
            st.warning("⚠️ Unable to make a prediction. Try a different image.")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Footer
st.markdown("---")
st.markdown("Made with love by pranitha pranay  using CNN & Streamlit")
