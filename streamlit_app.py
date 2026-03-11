import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
model = tf.keras.models.load_model("brain_tumor_model.h5")

# Title
st.title("🧠 Brain Tumor Detection App")

# Upload image
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

def preprocess(image):
    image = image.resize((224,224))
    image = np.array(image)/255.0
    image = np.expand_dims(image, axis=0)
    return image

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = preprocess(image)
    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        st.error("Tumor Detected ❌")
    else:
        st.success("No Tumor ✅")
