import cv2
import numpy as np
from utils import rgb_to_grayscale, gaussian_blur, p_tile_threshold, normalize
from canny import compute_gradient, compute_magnitude_angle, non_maximum_suppression


def compute_bounding_box(binary_image):
    """
    Compute the bounding box around the non-zero pixels in a binary image.
    :param binary_image: Binary image (2D numpy array)
    :return: Bounding box as (x, y, w, h) or None if no shape is detected
    """
    non_zero_coords = np.argwhere(binary_image > 0)

    if non_zero_coords.size == 0:
        return None

    rows, cols = non_zero_coords[:, 0], non_zero_coords[:, 1]

    x_min, x_max = cols.min(), cols.max()
    y_min, y_max = rows.min(), rows.max()

    return x_min, y_min, x_max - x_min + 1, y_max - y_min + 1


def draw_bounding_box(image, bounding_box):
    """
    Draw a bounding box on an image.
    :param image: Input image (3D numpy array)
    :param bounding_box: Bounding box as (x, y, w, h)
    :return: Image with bounding box drawn
    """
    if bounding_box is not None:
        x, y, w, h = bounding_box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image


def main():
    # Open a connection to the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")
    print("Press 'm' to toggle magnitude visualization.")
    print("Press 'g' to toggle gradient visualizations (Gx, Gy).")
    print("Press 's' to toggle suppressed visualization.")
    print("Press '+' or '-' to adjust the threshold.")

    threshold_percent = 10

    show_magnitude = False
    show_gradient = False
    show_suppressed = False

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break
        frame = cv2.resize(frame, (256, 256))

        current_frame_gray = rgb_to_grayscale(frame)
        smoothed_frame = gaussian_blur(current_frame_gray)
        Gx, Gy = compute_gradient(smoothed_frame)
        magnitude, angle = compute_magnitude_angle(Gx, Gy)

        suppressed = non_maximum_suppression(magnitude, angle)

        edges = p_tile_threshold(suppressed, threshold_percent)

        bounding_box = compute_bounding_box(edges)

        frame_with_box = draw_bounding_box(frame.copy(), bounding_box)

        cv2.imshow('Bounding Box', frame_with_box)
        cv2.imshow('Binary Edges', edges)

        if show_magnitude:
            magnitude_norm = normalize(magnitude)
            cv2.imshow("Magnitude", magnitude_norm)
        else:
            cv2.destroyWindow("Magnitude")

        if show_gradient:
            cv2.imshow("Gradient X", Gx)
            cv2.imshow("Gradient Y", Gy)
        else:
            cv2.destroyWindow("Gradient X")
            cv2.destroyWindow("Gradient Y")

        if show_suppressed:
            suppressed_norm = normalize(suppressed)
            cv2.imshow("Suppressed", suppressed_norm)
        else:
            cv2.destroyWindow("Suppressed")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+'):
            threshold_percent = min(100, threshold_percent + 1)
            print(f"Threshold increased to {threshold_percent}%")
        elif key == ord('-'):
            threshold_percent = max(1, threshold_percent - 1)
            print(f"Threshold decreased to {threshold_percent}%")
        elif key == ord('m'):
            show_magnitude = not show_magnitude
            print("Toggled Magnitude Visualization:", show_magnitude)
        elif key == ord('g'):
            show_gradient = not show_gradient
            print("Toggled Gradient Visualization:", show_gradient)
        elif key == ord('s'):
            show_suppressed = not show_suppressed
            print("Toggled Suppressed Visualization:", show_suppressed)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
