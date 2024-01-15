import streamlit as st
import numpy as np
from PIL import Image
import joblib

# Load the trained model
model = joblib.load('image_compression_model.joblib')

# Title
st.title('Image Compression')

# User input for image upload
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file)
    image = np.array(image)
    pixels = image.reshape(-1, 3)

    # Make predictions on the input data
    compressed_pixels = model.cluster_centers_[model.labels_]
    compressed_image = compressed_pixels.reshape(image.shape)

    # Display the original and compressed images
    st.image(image, caption='Original Image', use_column_width=True)
    st.image(compressed_image, caption='Compressed Image', use_column_width=True)
