# scripts/train.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from preprocess import preprocess_data

def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    dataset_folder = "data/raw/train"  # Update this path to your dataset location
    images, labels, label_encoder = preprocess_data(dataset_folder)
    input_shape = (64, 64, 3)
    num_classes = len(label_encoder.classes_)
    
    model = build_model(input_shape, num_classes)
    model.fit(images, labels, epochs=10, validation_split=0.2)
    model.save('models/asl_model.h5')