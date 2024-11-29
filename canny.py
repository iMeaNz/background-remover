import numpy as np
from scipy.signal import convolve2d

# Prewitt kernels for gradient computation
prewittX = (1.0 / 3.0) * np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
prewittY = (1.0 / 3.0) * np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

def compute_gradient(image):
    """Compute gradients using Prewitt operators."""
    # Use 2D convolution to compute gradients
    Gx = convolve2d(image, prewittX, mode='same', boundary='symm')
    Gy = convolve2d(image, prewittY, mode='same', boundary='symm')
    
    return Gx, Gy

def compute_magnitude_angle(Gx, Gy):
    """Compute gradient magnitude and angle."""
    magnitude = np.sqrt(Gx**2 + Gy**2)
    angle = np.arctan2(Gy, Gx) * 180 / np.pi
    angle[angle < 0] += 360
    return magnitude, angle

def non_maximum_suppression(magnitude, angle):
    """Apply non-maximum suppression using vectorized operations."""
    angle = angle % 180  # Map angles to [0, 180)
    height, width = magnitude.shape
    suppressed = np.zeros_like(magnitude, dtype=np.float32)

    # Define shifted neighbors for each angle range
    # Horizontal (0 or 180 degrees)
    neighbor1_h = np.pad(magnitude[:, 1:], ((0, 0), (0, 1)), mode='constant')
    neighbor2_h = np.pad(magnitude[:, :-1], ((0, 0), (1, 0)), mode='constant')

    # Vertical (90 degrees)
    neighbor1_v = np.pad(magnitude[1:, :], ((0, 1), (0, 0)), mode='constant')
    neighbor2_v = np.pad(magnitude[:-1, :], ((1, 0), (0, 0)), mode='constant')

    # Diagonal (45 degrees)
    neighbor1_d1 = np.pad(magnitude[:-1, 1:], ((1, 0), (0, 1)), mode='constant')
    neighbor2_d1 = np.pad(magnitude[1:, :-1], ((0, 1), (1, 0)), mode='constant')

    # Anti-diagonal (135 degrees)
    neighbor1_d2 = np.pad(magnitude[1:, 1:], ((0, 1), (0, 1)), mode='constant')
    neighbor2_d2 = np.pad(magnitude[:-1, :-1], ((1, 0), (1, 0)), mode='constant')

    horizontal_mask = ((0 <= angle) & (angle < 22.5)) | ((157.5 <= angle) & (angle <= 180))
    vertical_mask = (67.5 <= angle) & (angle < 112.5)
    diagonal_mask = (22.5 <= angle) & (angle < 67.5)
    anti_diagonal_mask = (112.5 <= angle) & (angle < 157.5)

    suppressed = np.where(
        horizontal_mask & (magnitude >= neighbor1_h) & (magnitude >= neighbor2_h), magnitude, suppressed
    )
    suppressed = np.where(
        vertical_mask & (magnitude >= neighbor1_v) & (magnitude >= neighbor2_v), magnitude, suppressed
    )
    suppressed = np.where(
        diagonal_mask & (magnitude >= neighbor1_d1) & (magnitude >= neighbor2_d1), magnitude, suppressed
    )
    suppressed = np.where(
        anti_diagonal_mask & (magnitude >= neighbor1_d2) & (magnitude >= neighbor2_d2), magnitude, suppressed
    )

    return suppressed
