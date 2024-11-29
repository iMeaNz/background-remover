import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from utils import rgb_to_grayscale, apply_binary_mask, gaussian_blur

def main():
    # Open a connection to the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Parameters for the average calculation
    buffer = None
    frame_count = 0
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = fps * 10

    print(f"Starting to average the first {10} seconds of frames...")

    while frame_count < total_frames:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Here we convert the frame to float for a better averaging calculation
        frame_float = frame.astype(np.float32)

        if buffer is None:
            buffer = np.zeros_like(frame_float)

        buffer += frame_float
        frame_count += 1

        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if frame_count > 0:
        average_frame = (buffer / frame_count).astype(np.uint8)
        grayscale_buffer = rgb_to_grayscale(average_frame)
        print(f"Finished averaging {frame_count} frames.")

    threshold = 25 
    print("Resuming webcam feed, showing differences...")
    print("Press UP/DOWN to adjust the threshold. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        current_frame_gray = rgb_to_grayscale(frame)

        diff_mask = np.abs(current_frame_gray.astype(np.int16) - grayscale_buffer.astype(np.int16)).astype(np.uint8)

        diff_mask = gaussian_blur(diff_mask, 5, 1)

        thresholded_mask = apply_binary_mask(diff_mask, threshold, 255)

        cv2.imshow('Difference Mask', thresholded_mask)

        # Handle keyboard input for threshold adjustment
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 82:  # UP arrow
            threshold = min(threshold + 5, 255)
            print(f"Threshold increased to: {threshold}")
        elif key == 84:  # DOWN arrow
            threshold = max(threshold - 5, 0)
            print(f"Threshold decreased to: {threshold}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
