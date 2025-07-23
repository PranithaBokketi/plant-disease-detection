#  Plant Disease Detection from Leaf Images

##  Objective
Detect and classify plant diseases from leaf images using a CNN model built with Keras and deploy it via a Streamlit GUI.

---

##  Tools & Technologies
- Python
- TensorFlow / Keras
- OpenCV
- NumPy / Matplotlib
- Streamlit
- PlantVillage Dataset

---

##  Dataset
- Source: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- Used 4–5 disease classes for training
- Dataset split into: `train/`, `val/`

---

##  Model Summary
- CNN (Convolutional Neural Network) built using Keras
- Image size: 128x128
- Image normalization applied
- Trained for ~10 epochs
- Model saved as `plant_disease_model.h5`

---

##  Image Preprocessing
- Resize to 128x128
- Normalize pixel values
- Augmentation (flip, rotation) applied to training set

---

##  Streamlit GUI
- Upload leaf image via Streamlit
- Display prediction: Disease Class
- Shows uploaded image and result
- Run the app:
  ```bash
  streamlit run app.py


## Project Structure

plant-disease-detection/


 PlantVillage/             # Raw dataset
 
 
 plant_disease.py          # Model training code
 
 app.py                    # Streamlit GUI
 
 split_data.py             # Script to split dataset
 
 plant_disease_model.h5    # Trained CNN model
 
 README.txt                # Project overview


## How to Run

1. Install dependencies:

  pip install tensorflow opencv-python streamlit

2. Run model training (if not already trained)

    python plant_disease.py

3. Launch GUI:

   streamlit run app.py

##  Demo Video

Watch the video demo of the Plant Disease Detection project:

 [Click to watch the demo](https://www.loom.com/share/b6d01347b3e143cb9711d04f5416a15f?sid=4927af9e-2b46-4f4d-b72a-845ceca379a8)



 Author
 
Bokketi Pranitha
