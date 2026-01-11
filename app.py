import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# 1. Page Configuration
st.set_page_config(page_title="Fighter Jet Classifier", page_icon="✈️")

# 2. Load the Model (Cached for speed)
@st.cache_resource
def load_classifier():
    # Make sure this filename matches exactly what you uploaded
    return load_model('fighter_jet_deploy.h5', compile=False)

model = load_classifier()

# 3. Define the 20 Class Names (Must match your training folder order)
CLASS_NAMES = [
    'An225', 'B1B', 'B2', 'B52', 'C5', 'E2d', 'EA18G', 'EA6b', 'F117', 
    'F22', 'F35', 'H6k', 'J10', 'J20', 'RQ4', 'Rafal', 'T50', 'Tu160', 
    'U2', 'V22'
]

# 4. App UI
st.title("✈️ Fighter Jet Identity System")
st.markdown("Upload an image of a fighter jet to identify its model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display Image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.write("Analyzing...")

    # 5. Preprocessing (Strictly for EfficientNet)
    # Resize to 224x224
    img = image.resize((224, 224))
    # Convert to array
    img_array = np.array(img)
    
    # Ensure 3 channels (RGB)
    if img_array.shape[-1] != 3:
         img_array = np.stack((img_array,)*3, axis=-1)

    # Expand dimensions to (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # EfficientNet Preprocessing (Critical!)
    img_array = preprocess_input(img_array)

    # 6. Prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0]) # Get confidence scores
    
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)

    # 7. Results
    st.success(f"**Prediction:** {predicted_class}")
    st.info(f"**Confidence:** {confidence}%")
    
    # Optional: Show breakdown
    # st.bar_chart(predictions[0])