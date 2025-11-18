# energy.py
import numpy as np
import cv2

def compute_energy(img: np.ndarray) -> np.ndarray:
    """
    Compute the energy map of an image using gradient magnitude.

    Parameters
    ----------
    img : np.ndarray
        Input image, H x W x 3 (BGR) or H x W (grayscale).

    Returns
    -------
    energy : np.ndarray
        2D array (H x W) of energy values (float64).
    """
    # Convert to grayscale if needed
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    gray = gray.astype(np.float64)

    # Sobel gradients in x and y
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Energy = gradient magnitude (L1 norm)
    energy = np.abs(grad_x) + np.abs(grad_y)
    return energy
