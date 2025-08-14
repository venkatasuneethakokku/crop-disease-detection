import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# Load model
model = tf.keras.models.load_model('model.h5')

# Load class labels
class_names = os.listdir('data/')
class_names.sort()  # Ensure labels match training order

# Streamlit UI
st.title("🌿 Crop Disease Detection")
uploaded_file = st.file_uploader("Upload a leaf image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    st.subheader("Prediction:")
    st.write(f"**Class:** {class_names[class_index]}")
    st.write(f"**Confidence:** {confidence*100:.2f}%")
