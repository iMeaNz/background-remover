import cv2
import numpy as np
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

    print("Step 1: Calculating the background model.")
    print("- The webcam feed will display for 10 seconds.")
    print("- These frames will be averaged to compute a reference background.")
    print("- Use q to stop the process early.")

    # Inform the user about the initial setup
    input("Press Enter to begin averaging frames for background calculation...")

    while frame_count < total_frames:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        frame_float = frame.astype(np.float32)

        if buffer is None:
            buffer = np.zeros_like(frame_float)

        buffer += frame_float
        frame_count += 1

        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Averaging process interrupted by user.")
            break

    if frame_count > 0:
        average_frame = (buffer / frame_count).astype(np.uint8)
        grayscale_buffer = rgb_to_grayscale(average_frame)
        print(f"Background model computed from {frame_count} frames.")

    print("\nStep 2: Live difference detection.")
    print("- The webcam feed will now show areas of motion or change.")
    print("- Adjust sensitivity using UP/DOWN arrows.")
    print("- Press 'q' to quit at any time.")

    threshold = 25
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
