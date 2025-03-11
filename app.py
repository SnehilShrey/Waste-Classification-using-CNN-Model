import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64

# Set page configuration (MUST be first)
st.set_page_config(page_title="Waste Classification", page_icon="♻️", layout="centered")

# Function to set a solid background image
def set_background(image_file):
    with open(image_file, "rb") as img:
        encoded_string = base64.b64encode(img.read()).decode()
    
    bg_css = f"""
    <style>
    .stApp {{
        background: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(bg_css, unsafe_allow_html=True)

# Call function to set background (Use your saved image file name)
set_background("background.jpg")  

# Load TFLite model
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="waste_classification_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define class labels
class_labels = ["Organic", "Recyclable"]

# Function to make predictions using TFLite model
def predict_waste(image):
    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    
    class_index = np.argmax(predictions)
    confidence = predictions[class_index] * 100
    return class_labels[class_index], confidence

# Streamlit UI with refined spacing
st.markdown("""
    <div style="background-color:#FF9933; padding:8px; border-radius:10px; text-align:center;">
        <h1 style="color:white; margin:5px 0;">♻️ Waste Classification System</h1>
    </div>
""", unsafe_allow_html=True)

# First subtitle with no gap from the heading
st.markdown("""
    <div style="background-color:#C850C0; padding:5px; border-radius:10px; text-align:center; margin-top:0;">
        <p style="color:white; font-size:14px; margin:5px 0;">Upload an image to classify waste</p>
    </div>
""", unsafe_allow_html=True)

# Increased gap before second subtitle
st.markdown("""
    <div style="background-color:#FFD700; padding:6px; border-radius:10px; text-align:center; margin-top:20px; margin-bottom:0;">
        <h4 style="color:black; margin:4px 0;">Choose an image file to upload</h4>
    </div>
""", unsafe_allow_html=True)

# File uploader (now directly below second subtitle)
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    label, confidence = predict_waste(image)
    
    # Enhanced UI layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
            <div style="background-color:#e0f7fa; padding:20px; border-radius:12px; 
                    width: 100%; text-align: left; 
                    box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.2);">
            <h2 style="color:#00796B; margin: 5px 0; font-size: 26px;">
               Prediction: <span style="font-weight: bold;">{label}</span>
            </h2>
            <h3 style="color:#004D40; margin: 5px 0; font-size: 22px;">
                Confidence: <span style="font-weight: bold;">{confidence:.2f}%</span>
            </h3>
        </div>
    """, unsafe_allow_html=True)
        
    # Color-coded result
    if label == "Organic":
        st.success("🌿 This waste is **Organic** and can be composted! ♻️")
    else:
        st.info("🔄 This waste is **Recyclable** and should be sent to a recycling facility! ♻️")
    
    # Styling
    st.markdown("""
        <style>
        .stProgress > div > div > div > div {
            background-color: #4CAF50 !important;
        }
        </style>
    """, unsafe_allow_html=True)
