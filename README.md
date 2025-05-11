🍎 Food Image Classification using CNN

This project leverages a dataset sourced from Kaggle to build a Convolutional Neural Network (CNN) for classifying food images into four categories: **grapes**, **apples**, **bananas**, and **oranges**. The project demonstrates image preprocessing, deep learning model training, evaluation, and deployment of a prediction script for real-world use cases.

---

## 📁 Dataset

- Source: [Kaggle Food Image Dataset]
- Classes: Grapes, Apples, Bananas, Oranges
- Structure your dataset directory like this:
```

dataset/
├── grapes/
├── apples/
├── bananas/
└── oranges/

````

---

## ⚙️ Steps to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
````

### 2. Set Up the Environment

Install the required libraries:

```bash
pip install tensorflow keras numpy matplotlib opencv-python
```

(Optional) Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Preprocess the Images

* Resize images to 128x128 pixels
* Normalize pixel values (0-1)
* Split into training, validation, and test sets

### 4. Build the CNN Model

* Use layers like `Conv2D`, `MaxPooling2D`, `Flatten`, and `Dense`
* Compile with `categorical_crossentropy` and `Adam` optimizer

### 5. Train the Model

* Train on the training set
* Monitor validation accuracy and loss
* Plot training history using Matplotlib

### 6. Evaluate the Model

* Test the model on unseen data
* Print accuracy, confusion matrix, and classification report

### 7. Predict Using a Script

Use the prediction script to classify new images:

```bash
python predict.py --image path_to_image.jpg
```



## ✅ Project Highlights

* End-to-end CNN implementation for food classification
* Image preprocessing and augmentation
* Real-world deployment-ready prediction script
* Extendable to more classes or datasets

 📌 Requirements

* Python 3.x
* TensorFlow / Keras
* NumPy
* Matplotlib
* OpenCV

## 📬 Contact

Feel free to reach out for feedback or collaboration!
*Adinath Nage** – [your.email@example.com](mailto:adinathnage@gmail.com]
