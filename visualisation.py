import cv2
import numpy as np

# IMPORT THE DP + ENERGY FUNCTIONS
from dp_seam import (
    compute_energy,
    vertical_seam_dp,
    remove_vertical_seam,
)


def draw_vertical_seam(img, seam, color=(0, 0, 255)):
    out = img.copy()
    for (r, c) in seam:
        out[r, c] = color
    return out


def draw_horizontal_seam(img, seam, color=(0, 0, 255)):
    out = img.copy()
    for (r, c) in seam:
        out[r, c] = color
    return out


def draw_multiple_seams(img, seams):
    out = img.copy()
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 255, 255),
        (0, 128, 255),
        (0, 0, 255),
    ]
    for i, seam in enumerate(seams):
        color = colors[i % len(colors)]
        for (r, c) in seam:
            out[r, c] = color
    return out


def animate_vertical_carving(img, n, delay=40):
    """
    Animate carving n vertical seams, showing each seam before removal.
    """
    temp = img.copy()
    for _ in range(n):
        energy = compute_energy(temp)
        seam = vertical_seam_dp(energy)
        vis = draw_vertical_seam(temp, seam)
        cv2.imshow("Carving Animation", vis)
        cv2.waitKey(delay)
        temp = remove_vertical_seam(temp, seam)

    cv2.imshow("Final Carved", temp)
    cv2.waitKey(0)
