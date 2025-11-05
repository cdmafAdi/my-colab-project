import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("cat_dog_model.h5")

st.title("🐾 Cat vs Dog Classifier")
st.write("Upload an image and let the model predict whether it's a cat or a dog!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize((256, 256))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]
    label = "🐶 Dog" if prediction > 0.5 else "🐱 Cat"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    st.markdown(f"### Prediction: **{label}**")
    st.progress(float(confidence))
    st.markdown(f"Confidence: **{confidence*100:.2f}%**")
