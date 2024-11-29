import numpy as np
import cv2

def rgb_to_grayscale(image):
    """
    Converts an RGB image to grayscale using the formula:
    Gray = 0.299 * R + 0.587 * G + 0.114 * B
    """
    # Extract the RGB channels
    r, g, b = image[..., 2], image[..., 1], image[..., 0]

    # Calculate grayscale values using the formula
    gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)

    return gray

def apply_binary_mask(image, threshold_value, max_value=255):
    """
    Apply a binary threshold to the image.
    :param image: Input grayscale image (2D array).
    :param threshold_value: Threshold value for binarization.
    :param max_value: Value to assign for pixels above the threshold.
    :return: Thresholded binary image.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Input image must be a NumPy array.")
    
    if image.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale image.")
    
    binary_image = np.zeros_like(image, dtype=np.uint8)
    binary_image[image > threshold_value] = max_value
    
    return binary_image

def create_gaussian_kernel(size, sigma):
    """
    Create a 1D Gaussian kernel.
    :param size: Kernel size (must be odd).
    :param sigma: Standard deviation of the Gaussian distribution.
    :return: 1D Gaussian kernel.
    """
    k = size // 2
    x = np.arange(-k, k + 1, dtype=np.float32)
    kernel = np.exp(-x**2 / (2 * sigma**2))
    kernel /= kernel.sum()  # Normalize
    return kernel

    def gaussian_blur(image, kernel_size=5, sigma=1.0):
        """
        Apply a separable Gaussian blur to a grayscale image.
        :param image: Input image (2D grayscale).
        :param kernel_size: Size of the Gaussian kernel (must be odd).
        :param sigma: Standard deviation of the Gaussian kernel.
        :return: Blurred image.
        """
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd.")
        if sigma <= 0:
            raise ValueError("Sigma must be greater than zero.")

        kernel = create_gaussian_kernel(kernel_size, sigma)

        if image.ndim != 2:
            raise ValueError("Image must be 2D (grayscale) or 3D (color).")
        
        k = len(kernel) // 2
        padded_image = np.pad(image, ((k, k), (k, k)), mode='reflect')
        
        temp_result = np.zeros_like(image, dtype=np.float32)
        for i in range(image.shape[0]):
            temp_result[i, :] = np.convolve(padded_image[i, :], kernel, mode='valid')

        padded_temp = np.pad(temp_result, ((k, k), (0, 0)), mode='reflect')
        blurred_image = np.zeros_like(image, dtype=np.float32)
        for j in range(image.shape[1]):
            blurred_image[:, j] = np.convolve(padded_temp[:, j], kernel, mode='valid')

        return np.clip(blurred_image, 0, 255).astype(np.uint8)

def slightly_optimized_gaussian_blur(image, kernel_size=5, sigma=1.0):
    """
    Apply a separable Gaussian blur to an image.
    :param image: Input grayscale image (2D array).
    :param kernel_size: Size of the Gaussian kernel (must be odd).
    :param sigma: Standard deviation of the Gaussian kernel.
    :return: Blurred image.
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")
    if sigma <= 0:
        raise ValueError("Sigma must be greater than zero.")

    kernel = create_gaussian_kernel(kernel_size, sigma)

    k = kernel_size // 2
    padded_image = np.pad(image, ((0, 0), (k, k)), mode='reflect')
    temp_result = np.zeros_like(image, dtype=np.float32)
    # We process rows
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            temp_result[i, j] = np.sum(padded_image[i, j:j + kernel_size] * kernel)

    # We process columns
    padded_result = np.pad(temp_result, ((k, k), (0, 0)), mode='reflect')
    blurred_image = np.zeros_like(image, dtype=np.float32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            blurred_image[i, j] = np.sum(padded_result[i:i + kernel_size, j] * kernel)

    return np.clip(blurred_image, 0, 255).astype(np.uint8)

def unoptimized_gaussian_blur(image, kernel_size=5, sigma=1.0):
    """
    Apply a Gaussian blur to an image using direct 2D convolution.
    :param image: Input grayscale image (2D array).
    :param kernel_size: Size of the Gaussian kernel (must be odd).
    :param sigma: Standard deviation of the Gaussian kernel.
    :return: Blurred image.
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")
    if sigma <= 0:
        raise ValueError("Sigma must be greater than zero.")
    
    kernel = create_gaussian_kernel(kernel_size, sigma)
    k = kernel_size // 2

    padded_image = np.pad(image, ((k, k), (k, k)), mode='reflect')
    blurred_image = np.zeros_like(image, dtype=np.float32)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extract the k x k neighborhood around the pixel
            region = padded_image[i:i + kernel_size, j:j + kernel_size]

            # Perform element-wise multiplication and sum the result
            blurred_image[i, j] = np.sum(region * kernel)

    return np.clip(blurred_image, 0, 255).astype(np.uint8)

def p_tile_threshold(image, percent):
    """
    Applies p-tile method of automatic thresholding to find the best threshold value and then apply it
    to the image to create a binary image.
    
    :param image: Input grayscale image array (2D numpy array)
    :param percent: Percentage of non-zero pixels to be above the threshold
    :return: Binary image array with p-tile thresholding applied
    """
    histogram, _ = np.histogram(image, bins=256, range=(0, 256))
    
    total_non_zero_pixels = np.sum(histogram[1:])
    
    threshold_pixel_count = np.around(total_non_zero_pixels * (percent / 100)).astype(int)

    cumulative_sum = 0
    threshold_value = 255
    for intensity in range(255, -1, -1):
        cumulative_sum += histogram[intensity]
        if cumulative_sum >= threshold_pixel_count:
            threshold_value = intensity
            break

    binary_image = np.where(image >= threshold_value, 255, 0).astype(np.uint8)

    return binary_image

def normalize(image):
    """Normalize a floating-point image to an 8-bit range [0, 255]."""
    min_val = np.min(image)
    max_val = np.max(image)

    # Avoid division by zero if the image is flat (all pixels have the same value)
    if max_val - min_val == 0:
        return np.zeros_like(image, dtype=np.uint8)

    # Normalize to [0, 1], then scale to [0, 255]
    normalized = (image - min_val) / (max_val - min_val) * 255
    return normalized.astype(np.uint8)