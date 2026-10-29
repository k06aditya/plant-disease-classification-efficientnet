import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# 1. Load your saved model
model = tf.keras.models.load_model("plant_disease_model.keras")
print("Model loaded successfully!")

# 2. Your exact 38 class names matching your training categories
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

print(f"Loaded {len(class_names)} disease categories.")

# 3. Prompt you in the terminal to type or paste the image path
img_path = input("\nEnter the path to your test leaf image: ").strip().strip("'\"")

# 4. Load and resize the image to 128x128 pixels
img = image.load_img(img_path, target_size=(128, 128))

# 5. Convert image to an array and expand dimensions for the model
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# 6. Make a prediction
predictions = model.predict(img_array)
predicted_index = np.argmax(predictions[0])
confidence = np.max(predictions[0]) * 100

# 7. Print out the final result
predicted_disease = class_names[predicted_index]
print(f"\n--- Prediction Results ---")
print(f"Predicted Disease: {predicted_disease}")
print(f"Confidence Score: {confidence:.2f}%")