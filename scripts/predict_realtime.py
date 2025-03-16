import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import os
import mediapipe as mp
import time
import pyttsx3
from collections import deque


def preprocess_frame(frame):
    # Resize and normalize the frame
    img = cv2.resize(frame, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

class HandSignPredictor:
    def __init__(self, model_path, encoder_path):
        # Load the model and label encoder
        self.model = load_model(model_path)
        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        
        # Initialize word formation with longer cooldowns
        self.current_word = []
        self.display_word = ""
        self.letter_cooldown = 2.0  # 2 seconds cooldown
        self.no_hand_time = time.time()
        self.space_threshold = 3.0  # 3 seconds for word completion
        
        # Initialize timing and prediction variables
        self.last_letter_time = time.time()
        self.min_confidence = 85.0
        self.prediction_queue = deque(maxlen=3)
        self.last_prediction = None
        self.consecutive_matches = 0
        self.required_matches = 3
        self.current_letter_start = None  # New: track when we start seeing a letter
        self.current_candidate = None  # New: track current letter candidate

    def preprocess_hand(self, frame, hand_landmarks):
        # Get frame dimensions
        h, w, _ = frame.shape
        
        # Get bounding box of hand
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
        
        # Extract and preprocess hand region
        if x_min < x_max and y_min < y_max:
            hand_region = frame[y_min:y_max, x_min:x_max]
            hand_region = cv2.resize(hand_region, (64, 64))
            hand_region = hand_region / 255.0
            return hand_region, (x_min, y_min, x_max, y_max)
        
        return None, None

    def get_smooth_prediction(self, prediction):
        current_time = time.time()
        
        # If we see a new letter, reset the timing
        if prediction != self.current_candidate:
            self.current_candidate = prediction
            self.current_letter_start = current_time
            self.consecutive_matches = 1
            return None
        
        # If we're seeing the same letter, increment matches
        self.consecutive_matches += 1
        
        # Only return prediction if we've held the same letter for 2 seconds AND have enough matches
        time_holding_letter = current_time - self.current_letter_start if self.current_letter_start else 0
        if (time_holding_letter >= self.letter_cooldown and 
            self.consecutive_matches >= self.required_matches):
            return prediction
            
        return None

    def update_word_buffer(self, letter, confidence):
        if confidence >= self.min_confidence:
            self.current_word.append(letter)
            self.display_word = ''.join(self.current_word)
            self.last_letter_time = time.time()
            self.consecutive_matches = 0
            self.current_letter_start = None  # Reset timing for next letter
            self.current_candidate = None

    def check_for_space(self, hand_detected):
        current_time = time.time()
        if not hand_detected:
            if self.current_word and self.current_word[-1] != ' ':
                time_without_hand = current_time - self.no_hand_time
                if time_without_hand >= self.space_threshold:
                    word = ''.join(self.current_word).strip()
                    if word:  # Only speak if there's a word
                        self.speak_text(word)
                        # Clear the current word after speaking it
                        self.current_word = []
                        self.display_word = ""
                    self.no_hand_time = current_time
        else:
            self.no_hand_time = current_time

    def speak_text(self, text):
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"Text-to-speech error: {e}")

    def predict_realtime(self):
        cap = cv2.VideoCapture(0)
        
        print("Controls:")
        print("'q' - Quit")
        print("'c' - Clear text")
        print("'b' - Backspace")
        print("\nInstructions:")
        print("1. Show your hand sign clearly")
        print("2. Hold the sign steady for 2 seconds")
        print("3. Wait for the green bar to fill completely")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Calculate cooldown progress based on current letter detection
            current_time = time.time()
            if self.current_letter_start and self.current_candidate:
                time_holding_letter = current_time - self.current_letter_start
                cooldown_progress = min(time_holding_letter / self.letter_cooldown, 1.0)
            else:
                cooldown_progress = 0
            
            # Draw cooldown progress bar
            bar_width = 200
            progress_width = int(bar_width * cooldown_progress)
            cv2.rectangle(frame, (10, 130), (210, 150), (0, 0, 255), 2)
            cv2.rectangle(frame, (10, 130), (10 + progress_width, 150), (0, 255, 0), -1)
            
            # Draw cooldown status
            if self.current_candidate:
                remaining_time = max(0, self.letter_cooldown - (current_time - self.current_letter_start))
                status = f"Hold '{self.current_candidate}' for {remaining_time:.1f}s"
            else:
                status = "Show a sign"
            cv2.putText(frame, status, (220, 145),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Initialize variables
            hand_detected = False
            predicted_letter = None
            confidence = 0
            bbox = None
            
            if results.multi_hand_landmarks:
                hand_detected = True
                hand_landmarks = results.multi_hand_landmarks[0]
                
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                hand_img, bbox = self.preprocess_hand(frame, hand_landmarks)
                
                if hand_img is not None:
                    prediction = self.model.predict(np.expand_dims(hand_img, axis=0), verbose=0)
                    predicted_class = np.argmax(prediction[0])
                    predicted_letter = self.label_encoder.inverse_transform([predicted_class])[0]
                    confidence = prediction[0][predicted_class] * 100
                    
                    smooth_letter = self.get_smooth_prediction(predicted_letter)
                    
                    if smooth_letter is not None:
                        self.update_word_buffer(smooth_letter, confidence)
            
            # Check for space when no hand is detected
            self.check_for_space(hand_detected)
            
            # Draw predictions and information
            if predicted_letter is not None:
                cv2.putText(frame, f"Letter: {predicted_letter}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {confidence:.2f}%", (10, 70),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Matches: {self.consecutive_matches}/{self.required_matches}", 
                          (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if bbox is not None:
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # Display the current word
            cv2.putText(frame, f"Text: {self.display_word}", (10, 110),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            cv2.imshow('ASL Recognition', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.current_word = []
                self.display_word = ""
                self.consecutive_matches = 0
                self.last_prediction = None
            elif key == ord('b'):
                if self.current_word:
                    self.current_word.pop()
                    self.display_word = ''.join(self.current_word)
        
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()

if __name__ == "__main__":
    try:
        model_path = '../models/asl_model.h5'
        encoder_path = '../models/label_encoder.pkl'
        
        predictor = HandSignPredictor(model_path, encoder_path)
        predictor.predict_realtime()
        
    except Exception as e:
        print(f"Error: {e}")

