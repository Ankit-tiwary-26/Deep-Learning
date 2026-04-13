import numpy as np
import cv2
from tensorflow import keras

# Load trained model
model = keras.models.load_model("mnist_model.keras")

def preprocess_image(image):
    # Convert to grayscale
    img = np.array(image.convert('L'))

    # Resize to 28x28
    img = cv2.resize(img, (28, 28))

    # Apply Gaussian Blur (reduce noise)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Apply OTSU thresholding (auto threshold)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert colors (MNIST format: white digit on black)
    img = 255 - img

    # Normalize
    img = img / 255.0

    # Reshape (VERY IMPORTANT)
    img = img.reshape(1, 28, 28, 1)

    return img


def predict_digit(image):
    processed = preprocess_image(image)

    prediction = model.predict(processed)

    digit = np.argmax(prediction)
    confidence = float(np.max(prediction))

    return digit, confidence, processed