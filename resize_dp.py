import time
import sys

from energy import compute_energy
from dp_seam import vertical_seam_dp, horizontal_seam_dp
from seam_removal import remove_vertical_seam, remove_horizontal_seam


def carve_vertical_n_dp(img, n: int):
    """
    Remove n vertical seams using DP.
    """
    for _ in range(n):
        energy = compute_energy(img)
        seam = vertical_seam_dp(energy)
        img = remove_vertical_seam(img, seam)
    return img


def carve_horizontal_n_dp(img, n: int):
    """
    Remove n horizontal seams using DP.
    """
    for _ in range(n):
        energy = compute_energy(img)
        seam = horizontal_seam_dp(energy)
        img = remove_horizontal_seam(img, seam)
    return img


def seam_carve_to_size_dp(img, new_h, new_w):
    H, W, _ = img.shape
    total_v = W - new_w
    total_h = H - new_h
    start = time.time()

    # --- Vertical seams ---
    for i in range(total_v):
        energy = compute_energy(img)
        seam = vertical_seam_dp(energy)
        img = remove_vertical_seam(img, seam)

        elapsed = time.time() - start
        progress = (i+1) / total_v * 100

        sys.stdout.write(f"\rVertical seams: {i+1}/{total_v} ({progress:.1f}%) | {elapsed:.2f}s")
        sys.stdout.flush()

    # --- Horizontal seams ---
    H, W, _ = img.shape
    for i in range(total_h):
        energy = compute_energy(img)
        seam = horizontal_seam_dp(energy)
        img = remove_horizontal_seam(img, seam)

        elapsed = time.time() - start
        progress = (i+1) / total_h * 100

        sys.stdout.write(f"\rHorizontal seams: {i+1}/{total_h} ({progress:.1f}%) | {elapsed:.2f}s")
        sys.stdout.flush()

    print()  # newline after completion
    return img