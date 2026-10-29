import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import streamlit as st
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
        color: white;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #e8f5e9;
        border: 1px solid #c8e6c9;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Cache the model loading so it loads only once for speed
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("plant_disease_model.keras")
    return model

with st.spinner("Loading model... Please wait."):
    model = load_model()

# Exact 38 class names
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Sidebar Content
with st.sidebar:
    st.image("https://img.icons8.com/color/96/plant-under-rain.png", width=80)
    st.title("About App")
    st.info(
        "This application uses a deep learning **EfficientNet** model "
        "trained to identify 38 different plant leaf diseases. "
        "Upload a clear image of a leaf to get instant diagnostics."
    )
    st.markdown("---")
    st.markdown("### Supported Crops")
    st.markdown("Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato")

# Main Interface Header
st.title("🌿 Plant Disease Detection System")
st.write("Upload an image of a plant leaf to detect potential diseases instantly.")

# File Uploader Widget
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        st.subheader("Uploaded Image")
        img = Image.open(uploaded_file)
        st.image(img, use_container_width=True)

    with col2:
        st.subheader("Analysis")
        if st.button("Predict Disease"):
            with st.spinner("Analyzing leaf patterns..."):
                # Preprocess image
                img_resized = img.resize((128, 128))
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0)

                # Prediction
                predictions = model.predict(img_array)
                predicted_index = np.argmax(predictions[0])
                confidence = np.max(predictions[0]) * 100

                predicted_disease = class_names[predicted_index]

            st.success("Analysis Complete!")
            
            # Display clean results card
            st.markdown(f"""
                <div class="prediction-card">
                    <h3 style="color: #1b5e20; margin-bottom: 5px;">Prediction Result</h3>
                    <p style="font-size: 20px; font-weight: bold; color: #2e7d32;">{predicted_disease}</p>
                    <hr style="border-top: 1px solid #c8e6c9;">
                    <p style="font-size: 16px; color: #333;">Confidence Score: <b>{confidence:.2f}%</b></p>
                </div>
            """, unsafe_allow_html=True)