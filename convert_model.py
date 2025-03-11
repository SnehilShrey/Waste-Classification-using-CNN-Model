import tensorflow as tf

# Load the original Keras model
model = tf.keras.models.load_model("waste_classification_model.keras")

# Convert to TFLite with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Apply quantization
quantized_model = converter.convert()

# Save the compressed model
with open("waste_classification_model.tflite", "wb") as f:
    f.write(quantized_model)

print("✅ Quantized model saved as waste_classification_model.tflite")