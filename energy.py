import cv2
import numpy as np


def compute_energy(img: np.ndarray) -> np.ndarray:
    """
    Compute energy map using Sobel gradient magnitude on grayscale image.
    Returns a 2D float64 array of shape (H, W).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    energy = np.abs(sobel_x) + np.abs(sobel_y)
    return energy
