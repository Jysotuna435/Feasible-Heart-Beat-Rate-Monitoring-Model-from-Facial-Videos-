import cv2
import numpy as np
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('TkAgg')
def extract_heart_rate(video_path):
    # Load the video
    cap = cv2.VideoCapture(video_path)
    # Initialize variables
    frames = []
    times = []
    prev_frame = None
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Loop through video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Conert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Store the grayscale frame
        frames.append(gray_frame)
        # Get the timestamp of the frame
        time = cap.get(cv2.CAP_PROP_POS_MSEC)
        times.append(time)
    # Release video capture
    cap.release()
    cv2.destroyAllWindows()
    # Convert frames to numpy array
    frames = np.array(frames)
    # Calculate average pixel intensity
    avg_intensity = np.mean(frames, axis=(1, 2))
    # Find peaks in intensity signal
    peaks, signal = find_peaks(avg_intensity, height=avg_intensity.mean(), distance=fps * 0.5)
    signals = signal['peak_heights']
    return signals


#

