import cv2
from keras.models import load_model
import numpy as np

# Load pre-trained facial expression recognition model
model = load_model('FER_model.h5')

# Define emotions
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]


# Function to preprocess frames
def preprocess_frame(frame):
    # Resize frame to fit model input size
    resized_frame = cv2.resize(frame, (48, 48))
    # Convert frame to grayscale
    grayscale_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    # Normalize pixel values
    normalized_frame = grayscale_frame / 255.0
    # Expand dimensions to match model input shape
    processed_frame = np.expand_dims(normalized_frame, axis=0)
    processed_frame = np.expand_dims(processed_frame, axis=-1)
    return processed_frame


# Function to detect facial emotion from frame
def detect_emotion(frame):
    # Preprocess frame
    processed_frame = preprocess_frame(frame)
    # Predict emotion probabilities
    emotion_probabilities = model.predict(processed_frame)[0]
    # Get index of dominant emotion
    dominant_emotion_index = np.argmax(emotion_probabilities)
    # Get dominant emotion label
    dominant_emotion = EMOTIONS[dominant_emotion_index]
    return dominant_emotion


# Process video frames
def process_video(video_path):
        # Detect facial emotion
        emotion = detect_emotion(video_path)
        return emotion



