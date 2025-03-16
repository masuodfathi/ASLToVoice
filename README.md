# ASL Recognition System

A real-time American Sign Language (ASL) recognition system that detects hand signs and converts them to text and speech.

## Features

- Real-time ASL letter recognition using webcam
- Hand detection and tracking using MediaPipe
- High-accuracy prediction with confidence threshold (85%)
- 2-second detection window for stable letter recognition
- Automatic word formation with space detection
- Text-to-speech output for detected words
- Visual feedback with progress bar and confidence display
- User-friendly interface with real-time status updates

## Requirements

- Python 3.8+
- OpenCV
- TensorFlow 2.x
- MediaPipe
- pyttsx3 (for text-to-speech)
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Masuod.fathi@gmail.com/asl-recognition.git
cd asl-recognition
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the real-time prediction script:
```bash
python scripts/predict_realtime.py
```

2. Position your hand in front of the camera
3. Hold each letter sign steady for 2 seconds
4. Remove hand from view to complete a word
5. System will speak the detected word

## How It Works

1. **Hand Detection**: Uses MediaPipe to detect and isolate hand regions in real-time
2. **Letter Recognition**: CNN model analyzes hand signs and predicts letters
3. **Confidence Check**: Only accepts predictions with 85%+ confidence
4. **Word Formation**: Combines detected letters into words when hand is removed
5. **Audio Output**: Converts formed words to speech

## Project Structure

```
asl-recognition/
├── scripts/
│   ├── predict_realtime.py    # Real-time prediction script
│   ├── train.py              # Model training script
│   └── utils.py             # Utility functions
├── models/                   # Trained model files
├── data/                    # Training data
└── requirements.txt         # Project dependencies
```

## Development Process

1. Initial Implementation: Basic image-based model for static ASL signs
2. Real-time Camera Integration: Live video processing
3. Hand Detection: MediaPipe integration for precise hand tracking
4. Training Optimization: Enhanced model training with hand detection
5. Frame-Based Detection: Stable frame-by-frame analysis
6. Accuracy Improvements: Confidence thresholds and time windows
7. Word Formation: Automatic word detection and speech output
8. User Interface: Visual feedback and status indicators

## Future Improvements

- Support for full ASL sentences
- Dynamic gesture recognition
- Custom gesture mapping
- Mobile application support
- Improved accuracy in varying lighting conditions
- Real-time translation of continuous signing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MediaPipe team for hand detection framework
- TensorFlow team for machine learning tools
- ASL community for guidance and feedback
