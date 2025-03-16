# scripts/preprocess.py

import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
import mediapipe as mp

class HandPreprocessor:
    def __init__(self):
        # Initialize MediaPipe Hand detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )

    def detect_and_crop_hand(self, image):
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect hands
        results = self.hands.process(rgb_image)
        
        if results.multi_hand_landmarks:
            # Get hand landmarks
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Get bounding box of hand
            h, w, _ = image.shape
            x_min = w
            x_max = 0
            y_min = h
            y_max = 0
            
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                x_min = min(x_min, x)
                x_max = max(x_max, x)
                y_min = min(y_min, y)
                y_max = max(y_max, y)
            
            # Add padding
            padding = 20
            x_min = max(0, x_min - padding)
            x_max = min(w, x_max + padding)
            y_min = max(0, y_min - padding)
            y_max = min(h, y_max + padding)
            
            # Crop hand region
            if x_min < x_max and y_min < y_max:
                hand_region = image[y_min:y_max, x_min:x_max]
                return hand_region, True
                
        return image, False

    def close(self):
        self.hands.close()

def load_images_from_folder(folder):
    images = []
    labels = []
    hand_preprocessor = HandPreprocessor()
    
    print("Processing images...")
    total_images = sum([len(files) for r, d, files in os.walk(folder)])
    processed_images = 0
    
    for label in os.listdir(folder):
        label_path = os.path.join(folder, label)
        if os.path.isdir(label_path):
            for filename in os.listdir(label_path):
                img_path = os.path.join(label_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    # Detect and crop hand
                    hand_img, hand_detected = hand_preprocessor.detect_and_crop_hand(img)
                    if hand_detected:
                        # Resize the cropped hand image
                        hand_img = cv2.resize(hand_img, (64, 64))
                        images.append(hand_img)
                        labels.append(label)
                    
                    processed_images += 1
                    if processed_images % 100 == 0:
                        print(f"Processed {processed_images}/{total_images} images")
    
    hand_preprocessor.close()
    return np.array(images), np.array(labels)

def normalize_images(images):
    return images / 255.0

def encode_labels(labels):
    le = LabelEncoder()
    return le.fit_transform(labels), le

def preprocess_data(folder):
    print("Loading and preprocessing images...")
    images, labels = load_images_from_folder(folder)
    print(f"Found {len(images)} valid hand images")
    
    print("Normalizing images...")
    images = normalize_images(images)
    
    print("Encoding labels...")
    labels, label_encoder = encode_labels(labels)
    
    return images, labels, label_encoder

if __name__ == "__main__":
    dataset_folder = "../data/train"
    images, labels, label_encoder = preprocess_data(dataset_folder)
    print(f"Preprocessing complete. Dataset contains {len(images)} images")
    print(f"Labels: {label_encoder.classes_}")
