from flask import Flask, request, jsonify, render_template
from keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)

# Load the model with the custom DepthwiseConv2D layer
class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        # Remove 'groups' from kwargs if passed
        groups = kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)
        self.groups = groups

# Load model (ensure you use the custom DepthwiseConv2D layer)
try:
    print("Attempting to load model...")
    model = load_model("keras_Model.h5", compile=False, custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Ensure that model is set to None if loading fails

# Check if model is successfully loaded before proceeding
if model is None:
    print("Model loading failed. Exiting.")
    exit()

# Load class names (labels)
try:
    print("Loading class labels...")
    class_names = open("labels.txt", "r").readlines()
    print("Labels loaded successfully.")
except Exception as e:
    print(f"Error loading labels: {e}")
    class_names = []  # Set labels to empty list if loading fails

# Image preprocessing function to ensure consistency
def prepare_image(image_path):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Open the image file
    image = Image.open(image_path).convert("RGB")

    # Resize the image to 224x224 and crop from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Convert the image to a numpy array
    image_array = np.asarray(image)

    # Normalize the image (ensure this matches training pipeline)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the data array
    data[0] = normalized_image_array

    return data

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if file is part of the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save the uploaded image
        image_path = 'uploaded_image.jpg'
        file.save(image_path)

        # Prepare the image
        image_data = prepare_image(image_path)

        # Predict using the model
        prediction = model.predict(image_data)

        # Get the predicted class and confidence score
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = prediction[0][index]

        # Convert confidence_score to a regular Python float before returning
        confidence_score = float(confidence_score)

        # Return result with better formatting and confidence thresholding
        return jsonify({
            'class': class_name,
            'confidence_score': confidence_score,
            'message': 'Prediction successful.'
        })

    except Exception as e:
        return jsonify({'error': str(e), 'message': 'Prediction failed'}), 500

# Render the HTML page for the frontend
@app.route('/')
def index():
    return render_template('index.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
