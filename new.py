import tensorflow as tf

# Load your model
model = tf.keras.models.load_model("plant_disease_model.keras")

# Check model structure
model.summary()
