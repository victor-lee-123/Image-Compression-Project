#seam_removal.py for dynamic programming

import numpy as np


def remove_vertical_seam(img: np.ndarray, seam):
    """
    Remove a vertical seam from the image.
    seam: list of (row, col) for each row.
    Returns new image of shape (H, W-1, C).
    """
    H, W, C = img.shape
    out = np.zeros((H, W - 1, C), dtype=img.dtype)

    for (r, c) in seam:
        out[r, :, :] = np.delete(img[r, :, :], c, axis=0)

    return out


def remove_horizontal_seam(img: np.ndarray, seam):
    """
    Remove a horizontal seam from the image.
    seam: list of (row, col) for each column (seam[c] matches col c).
    Returns new image of shape (H-1, W, C).
    """
    H, W, C = img.shape
    out = np.zeros((H - 1, W, C), dtype=img.dtype)

    for col in range(W):
        row = seam[col][0]  # (row, col)
        out[:, col, :] = np.delete(img[:, col, :], row, axis=0)

    return out
