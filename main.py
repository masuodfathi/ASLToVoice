import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_images_from_folder(folder):
    images = []
    labels = []
    for label in os.listdir(folder):
        label_path = os.path.join(folder, label)
        if os.path.isdir(label_path):
            for filename in os.listdir(label_path):
                img_path = os.path.join(label_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (64, 64))  # Resize to 64x64
                    images.append(img)
                    labels.append(label)
    return np.array(images), np.array(labels)

def normalize_images(images):
    return images / 255.0  # Normalize pixel values to [0, 1]

def encode_labels(labels):
    le = LabelEncoder()
    return le.fit_transform(labels), le

def preprocess_data(folder):
    images, labels = load_images_from_folder(folder)
    images = normalize_images(images)
    labels, label_encoder = encode_labels(labels)
    return images, labels, label_encoder

if __name__ == "__main__":
    dataset_folder = "dataset/train"  # Update this path to your dataset location
    images, labels, label_encoder = preprocess_data(dataset_folder)
    print(f"Processed {len(images)} images.")
    print(f"Labels: {label_encoder.classes_}")
