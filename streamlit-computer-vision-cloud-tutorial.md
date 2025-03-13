# Building an Image Classifier Web App with Streamlit

#streamlit #IronHack #deployment 

## What is Streamlit?
Streamlit is a Python library that turns your Python scripts into interactive web applications. It allows you to create a website without knowing HTML, CSS, or JavaScript. You can run it locally or you can publish it on Streamlit platform.
https://streamlit.io/

## Why Streamlit is Great for AI Apps
- You write pure Python code
- Your app updates automatically as users interact with it
- It's free to host basic applications
- You don't need to know web development

## Installing Streamlit
Using Anaconda: https://docs.streamlit.io/get-started/installation/anaconda-distribution
Using pip: `pip install streamlit`

## Writing your first app
### The Main Script
Every Streamlit app starts with a Python file (usually called `app.py` or `streamlit_app.py`). This file contains both your AI logic and your interface code. Here's a simple example:

**Try running it on your computer!**
1. Create new directory `my_cnn_app` (or something like that)
2. Create an `.py` file named `app.py` 
3. Copy into into it the code below
```python
import streamlit as st # importing streamlit library
  
st.title("Image Classifier") # this text will appear big
```

### Running Your First App
4. Open terminal/command prompt
5. Run your app:
```bash
python -m streamlit run app.py
```

### How Streamlit Works
Every time you interact with your Streamlit app (type in a textbox, move a slider, click a button), **Streamlit runs your entire script again from top to bottom. This means any variables you create will be reset!**

Streamlit works in an unusual way for a webapp. Every time you do anything in a Streamlit app - click a button, type text, upload a file - Streamlit runs your entire program again from start to finish. It's like closing and reopening your jupyter notebook. All the variables are forgotten.

*We will learn how to save the variables between re-runs in a few days.*

## Building an Image Classifier App with Streamlit

Now let's build a more advanced app: a CIFAR-10 image classifier that can recognize 10 different types of objects in uploaded images.

### Step 1: Project Setup
1. Create a new directory for your project
2. Create a new file called `app.py`
3. Make sure you have your pre-trained model file (`cifar10_cnn_model.h5`) in the same directory

### Step 2: Import Required Libraries
The first part of our app imports all the necessary libraries:

```python
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
```

Let's understand what each library does:
- `streamlit`: Creates our web interface
- `tensorflow`: Loads and runs our machine learning model
- `numpy`: Helps us work with arrays (for image processing)
- `PIL`: Python Imaging Library - helps us work with images

### Step 3: Load the Pre-trained Model
This section loads our pre-trained CNN model:

```python
# Load the pre-trained model and cache it (so that we don't have to load it every time)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('cifar10_cnn_model.h5')

```
- `@` here above the function name signifies something called `decorator`. A decorator in Python is a special kind of function that modifis behavior of a given function without changing function code. Think of decorators like adding a special ability to a function. The decorator intercepts the function call, adds some extra functionality, and then proceeds with the original function.
- `@st.cache_resource`: This is a special decorator that tells Streamlit to save the model in memory after loading it the first time. Remember how Streamlit reruns the entire script for every interaction? Without caching, it would reload the model every time you click anything!
```python
# Define CIFAR-10 class names
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
```
- `class_names`: This list contains the names of the 10 classes our model can predict.

### Step 4: Create the Web Interface
Now we set up the basic elements of our web app:

```python
# Set up the Streamlit app
st.title('CIFAR-10 Image Classifier')
st.write('Upload an image and the model will predict its class')

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
```

- `st.title()`: Creates a big heading for our app
- `st.write()`: Adds some text to explain what the app does
- `st.file_uploader()`: Creates a button that lets users upload image files

### Step 5: Image Preprocessing Function
Before feeding an image to our model, we need to prepare it:

```python
def preprocess_image(image):
    image = image.convert('RGB')
    image = image.resize((32, 32)) # has to be the same as your model input!!!
    image_array = np.array(image)
    image_array = image_array.astype('float32') / 255.0
    image_array = np.expand_dims(image_array, axis=0) 
    return image_array
```

This function:
1. Converts the image to RGB format (in case it's in another format)
2. Resizes it to 32x32 pixels (what our CIFAR-10 model expects)
    1. If you are using a different model you have resize it to your model input size
3. Converts the image to a numpy array
4. Scales pixel values to be between 0 and 1
    1. We did this during training because neural networks work better with smaller numbers. Values between 0-1 are less likely to cause computational issues than values between 0-255.
    2. If you didn't do this traing, you definitely should go back and to it.
5. Adds a batch dimension (our model expects a batch of images)

### Step 6: Making Predictions
The final part of our code handles the prediction process:

```python
# If an image is uploaded, we are going to make a prediction and display the results
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    image_array = preprocess_image(image)
    
    # Load model and make prediction
    model = load_model()
    predictions = model.predict(image_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100
    
    # Display results
    st.write(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")
    
    # Display bar chart of all predictions
    st.bar_chart({class_names[i]: float(predictions[0][i]) for i in range(10)})
```

Let's break this down:
1. The `if uploaded_file is not None:` checks if a user has uploaded an image
2. If an image is uploaded:
   - We display the uploaded image with `st.image()`
   - We preprocess the image using our function
   - We load the model and make a prediction
   - We find the class with the highest probability
   - We calculate the confidence (probability * 100)
   - We display the prediction and confidence
   - We create a bar chart showing the probabilities for all classes

### Step 7: Putting It All Together
Let's look at the complete code for our CIFAR-10 image classifier app:

```python
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(page_title="CIFAR-10 Image Classifier", page_icon="🖼️")

# Load the pre-trained model and cache it
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('cifar10_cnn_model.h5')

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
    image = image.resize((224, 224))
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
    
    # Preprocess the image and make prediction
    image_array = preprocess_image(image)
    model = load_model()
    
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
```

Save this code as `app.py` and run it with `streamlit run app.py`.

## Deploying Your App to Streamlit Cloud

Now that you've built a working image classifier app, let's deploy it to Streamlit Cloud so you can share it with the world!

### Step 1: Create a GitHub Repository
Streamlit Cloud deploys your app directly from a GitHub repository:

1. Create a new repository on GitHub
2. Upload your app files to the repository:
   - `app.py`: Your Streamlit app code
   - `cifar10_cnn_model.h5`: Your pre-trained model file
   - `requirements.txt`: A file listing your app's dependencies

### Step 2: Create a Requirements File
Create a file named `requirements.txt` in your repository with the required packages:

```
Pillow
streamlit
pandas>=2.0.3
numpy>=1.24.3
matplotlib>=3.7.2
tensorflow==2.19.0
```

The specific versions might need to be adjusted based on your environment.

### Step 3: Deploy to Streamlit Cloud
1. Create an account on [Streamlit Cloud](https://streamlit.io/cloud)
2. Click "New app"
3. Enter your GitHub repository details:
   - Repository: `yourusername/your-repo-name`
   - Branch: `main` (or whichever branch you're using)
   - Main file path: `app.py`
4. Click "Deploy"

After a few minutes, your app will be live at a URL like `https://yourusername-your-repo-name.streamlit.app`!

### Step 4: Troubleshooting Common Deployment Issues

If your app doesn't deploy correctly, check for these common issues:

1. **Missing Requirements**: Make sure all required packages are listed in your `requirements.txt` file
2. **File Size Limits**: Streamlit Cloud has a 1GB file size limit - large model files might need to be hosted elsewhere
3. **Path Issues**: Ensure all file paths in your code are correct (relative to the app.py file)
4. **Memory Limits**: If your app uses too much memory, try optimizing your code or using smaller models

## Next Steps

Congratulations! You've built and deployed an image classifier app with Streamlit. Here are some ways you could extend it:

1. **Improve the UI**: Add more interactive elements, improve the layout, or customize the appearance
2. **Use a More Powerful Model**: Replace the CIFAR-10 model with a more advanced model like MobileNet or EfficientNet
3. **Add More Features**: Allow users to select different models, save results, or process multiple images
4. **Monitor Usage**: Set up analytics to track how people are using your app

Remember that Streamlit is a powerful tool for quickly turning your data science and machine learning projects into interactive web applications. The more you practice, the more impressive your apps will become!

## Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-cloud)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Streamlit Gallery](https://streamlit.io/gallery) - For inspiration from other Streamlit apps