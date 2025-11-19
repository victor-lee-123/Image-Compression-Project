import numpy as np


def vertical_seam_greedy(energy: np.ndarray):
    """
    Greedy VERTICAL seam:
    At each row, pick the locally minimum neighbor (same, left, right).
    Returns list of (row, col) tuples.
    """
    H, W = energy.shape
    seam = []

    # Start at global min of top row
    c = int(np.argmin(energy[0]))
    seam.append((0, c))

    for r in range(1, H):
        candidates = [c]
        if c > 0:
            candidates.append(c - 1)
        if c < W - 1:
            candidates.append(c + 1)

        # pick the best local candidate in this row
        c = min(candidates, key=lambda x: energy[r, x])
        seam.append((r, c))

    return seam


def horizontal_seam_greedy(energy: np.ndarray):
    """
    Greedy HORIZONTAL seam via transpose trick.
    Returns list of (row, col) tuples, one per column.
    """
    seam_T = vertical_seam_greedy(energy.T)  # (r_T, c_T)
    seam = []
    for r_T, c_T in seam_T:
        col = r_T
        row = c_T
        seam.append((row, col))
    return seam
