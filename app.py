import numpy as np
from keras.models import load_model
from keras.layers import DepthwiseConv2D
from keras import layers
from PIL import Image, ImageOps

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Subclass the DepthwiseConv2D layer to allow for the 'groups' argument
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, groups=1, **kwargs):
        # Pass the groups argument to the superclass (DepthwiseConv2D) constructor
        super().__init__(**kwargs)
        self.groups = groups  # Add groups as an attribute

# Load the model with the custom DepthwiseConv2D layer
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

# Prepare the image to feed into the model
def prepare_image(image_path):
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Open the image file
    image = Image.open(image_path).convert("RGB")

    # Resize the image to 224x224 and crop from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Convert the image to a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the data array
    data[0] = normalized_image_array

    return data

# Load and preprocess the image
image_path = "E:/Training/Image_1.jpg"
image_data = prepare_image(image_path)

# Check if class names are loaded and the model is available before prediction
if len(class_names) == 0:
    print("Class names are missing. Exiting.")
    exit()

# Predict using the model
try:
    print("Making prediction...")
    prediction = model.predict(image_data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()  # Strip any leading/trailing spaces from the label
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name)
    print("Confidence Score:", confidence_score)

except Exception as e:
    print(f"Error during prediction: {e}")
