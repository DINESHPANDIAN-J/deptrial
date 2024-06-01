import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

# Load your pre-trained model
model = load_model('model.h5')

# Function to predict the class probabilities of a single image
def predict_image_probabilities(model, image, target_size=(224, 224)):
    # Convert the image to an array
    img_array = img_to_array(image)
    
    # Normalize the image
    img_array = img_array / 255.0
    
    # Expand dimensions to match the input shape of the model
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make a prediction
    predictions = model.predict(img_array)
    
    return predictions[0]

# Streamlit application
st.title("Disease Prediction from Image")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Preprocess the image
    image = image.resize((224, 224))  # Ensure the image matches the target size
    
    # Predict button
    if st.button('Predict'):
        # Make prediction
        probabilities = predict_image_probabilities(model, image)
        
        # Display the result
        st.write("Class Probabilities:")
        for idx, probability in enumerate(probabilities):
            st.write(f"Class {idx}: {probability:.4f}")

