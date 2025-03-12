import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Page configuration
st.set_page_config(page_title="CIFAR-10 Image Classifier", page_icon="🖼️")

# Load the pre-trained model and cache it
@st.cache_resource
def load_model():
    try:
        # First try to load the fixed model if it exists
        if os.path.exists('cifar10_cnn_model_fixed.h5'):
            return tf.keras.models.load_model('cifar10_cnn_model_fixed.h5')
        else:
            # If fixed model doesn't exist, try the original model
            return tf.keras.models.load_model('cifar10_cnn_model.h5')
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please run the fix_h5_model.py script to fix the model file.")
        return None

# Define CIFAR-10 class names
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Set up the Streamlit app
st.title('CIFAR-10 Image Classifier')
st.write('Upload an image and the model will predict its class')

# Add sidebar with information about the app
with st.sidebar:
    st.header("About")
    st.write("This app uses a CNN trained on the CIFAR-10 dataset to classify images into 10 categories.")
    st.write("The model can recognize: " + ", ".join(class_names))
    
    st.header("Instructions")
    st.write("1. Upload an image using the file uploader")
    st.write("2. The model will automatically classify the image")
    st.write("3. View the results and confidence levels")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def preprocess_image(image):
    image = image.convert('RGB')
    image = image.resize((32, 32))
    image_array = np.array(image)
    image_array = image_array.astype('float32') / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# If an image is uploaded, make a prediction
if uploaded_file is not None:
    # Create two columns for layout
    col1, col2 = st.columns(2)
    
    # Display the uploaded image
    image = Image.open(uploaded_file)
    with col1:
        st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Load the model
    model = load_model()
    
    # Only proceed if model was loaded successfully
    if model is not None:
        # Preprocess the image and make prediction
        image_array = preprocess_image(image)
        
        with st.spinner('Running prediction...'):
            predictions = model.predict(image_array)
        
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = np.max(predictions[0]) * 100
        
        # Display results
        with col2:
            st.subheader("Prediction Results")
            st.write(f"**Prediction:** {predicted_class}")
            st.write(f"**Confidence:** {confidence:.2f}%")
        
        # Display bar chart of all predictions
        st.subheader("Confidence Levels")
        prediction_dict = {class_names[i]: float(predictions[0][i]) for i in range(10)}
        st.bar_chart(prediction_dict)
    else:
        st.error("Model could not be loaded. Please fix the model file first.")

else:
    # Display example images if no file is uploaded
    st.info("Please upload an image to get a prediction")
    
    # Optionally, show example images here
    st.write("Examples of CIFAR-10 classes:")
    examples_col1, examples_col2 = st.columns(2)
    with examples_col1:
        st.write("• Airplane: Small passenger or military aircraft")
        st.write("• Automobile: Cars, sedans, and other passenger vehicles")
        st.write("• Bird: Various species of birds")
        st.write("• Cat: Domestic and wild cats")
        st.write("• Deer: Various species of deer")
    with examples_col2:
        st.write("• Dog: Various breeds of dogs")
        st.write("• Frog: Frogs, toads, and similar amphibians")
        st.write("• Horse: Horses and ponies")
        st.write("• Ship: Boats, ships, and other watercraft")
        st.write("• Truck: Pickup trucks, delivery trucks, etc.")