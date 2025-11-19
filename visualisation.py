import cv2
import numpy as np

from energy import compute_energy
from dp_seam import vertical_seam_dp
from greedy_seam import vertical_seam_greedy
from seam_removal import remove_vertical_seam


def draw_vertical_seam(img, seam, color=(0, 0, 255)):
    """
    Draw a vertical seam on a copy of the image.
    """
    out = img.copy()
    for (r, c) in seam:
        out[r, c] = color
    return out


def visualize_dp_seam(img):
    """
    Return (image_with_DP_seam, energy_map).
    """
    energy = compute_energy(img)
    seam = vertical_seam_dp(energy)
    vis = draw_vertical_seam(img, seam)
    return vis, energy


def visualize_greedy_seam(img):
    """
    Return (image_with_greedy_seam, energy_map).
    """
    energy = compute_energy(img)
    seam = vertical_seam_greedy(energy)
    vis = draw_vertical_seam(img, seam)
    return vis, energy


def animate_dp_vertical(img, n=30, delay=1):
    """
    Fast DP animation using vectorized DP.
    """
    temp = img.copy()
    for _ in range(n):
        energy = compute_energy(temp)
        seam = vertical_seam_dp(energy)
        vis = draw_vertical_seam(temp, seam)

        cv2.imshow("DP Animation", vis)
        cv2.waitKey(delay)

        temp = remove_vertical_seam(temp, seam)

    cv2.imshow("DP Final", temp)
    cv2.waitKey(0)



def animate_greedy_vertical(img, n=30, delay=40):
    """
    Animate Greedy vertical seam removal.
    """
    temp = img.copy()
    for _ in range(n):
        energy = compute_energy(temp)
        seam = vertical_seam_greedy(energy)
        vis = draw_vertical_seam(temp, seam)
        cv2.imshow("Greedy Animation", vis)
        cv2.waitKey(delay)
        temp = remove_vertical_seam(temp, seam)
    cv2.imshow("Greedy Final", temp)
    cv2.waitKey(0)
