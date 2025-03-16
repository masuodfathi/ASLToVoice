# scripts/predict.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from preprocess import normalize_images

def predict_image(model, image_path, label_encoder):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))
    img = normalize_images(np.array([img]))
    prediction = model.predict(img)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]

if __name__ == "__main__":
    model = load_model('models/asl_model.h5')
    label_encoder = ...  # Load or recreate the label encoder used during training
    image_path = "path/to/image.jpg"  # Update with the path to the image you want to predict
    predicted_label = predict_image(model, image_path, label_encoder)
    print(f"Predicted label: {predicted_label}")