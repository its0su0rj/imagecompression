import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import joblib
import streamlit as st
from io import BytesIO

# Load the trained model
model = joblib.load('image.joblib')

# Title
st.title('Image Compression')

# User input for image upload
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file)
    image = np.array(image)

    # Reshape the image to be a list of RGB values
    pixels = image.reshape(-1, 3)

    # Make predictions on the input data
    compressed_pixels = model.cluster_centers_[model.predict(pixels)]
    compressed_image = compressed_pixels.reshape(image.shape)

    # Normalize pixel values to [0, 255]
    compressed_image = np.clip(compressed_image, 0, 255).astype(np.uint8)

    # Display the original and compressed images
    st.image(image, caption='Original Image', use_column_width=True)
    st.image(compressed_image, caption='Compressed Image', use_column_width=True)

    # Download button for the compressed image
    compressed_image_path = "compressed_image.jpg"
    compressed_image_bytes = BytesIO()
    Image.fromarray(compressed_image).convert("RGB").save(compressed_image_bytes, format="JPEG")
    st.download_button(
        label="Download Compressed Image",
        data=compressed_image_bytes.getvalue(),
        file_name=compressed_image_path,
        mime="image/jpeg"
    )
