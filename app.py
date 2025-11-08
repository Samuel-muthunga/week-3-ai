# Run this as app.py to deploy MNIST model
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

st.title("🧠 MNIST Digit Classifier")

model = tf.keras.models.load_model("mnist_model.h5")

uploaded_file = st.file_uploader("Upload a handwritten digit image", type=["png","jpg","jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("L").resize((28,28))
    img_array = np.expand_dims(np.array(img)/255.0, axis=(0, -1))
    prediction = np.argmax(model.predict(img_array))
    st.image(img, caption=f"Predicted Digit: {prediction}", width=150)
