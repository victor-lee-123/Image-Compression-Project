import numpy as np
import cv2


# ----------------------------------------------------
# ENERGY FUNCTION (SOBEL — HIGH QUALITY)
# ----------------------------------------------------
def compute_energy(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    energy = np.abs(sobel_x) + np.abs(sobel_y)
    return energy


# ----------------------------------------------------
# DP SEAM FINDER (VERTICAL)
# ----------------------------------------------------
def vertical_seam_dp(energy: np.ndarray):
    H, W = energy.shape
    dp = energy.copy().astype(np.float64)
    parent = np.zeros((H, W), dtype=np.int32)

    for r in range(1, H):
        for c in range(W):

            candidates = [dp[r - 1, c]]

            if c > 0:
                candidates.append(dp[r - 1, c - 1])
            else:
                candidates.append(np.inf)

            if c < W - 1:
                candidates.append(dp[r - 1, c + 1])
            else:
                candidates.append(np.inf)

            best = np.argmin(candidates)

            if best == 0:
                parent[r, c] = c
            elif best == 1:
                parent[r, c] = c - 1
            else:
                parent[r, c] = c + 1

            dp[r, c] += candidates[best]

    seam = []
    c = int(np.argmin(dp[-1]))

    for r in reversed(range(H)):
        seam.append((r, c))
        c = parent[r, c]

    seam.reverse()
    return seam


# ----------------------------------------------------
# DP SEAM FINDER (HORIZONTAL)
# ----------------------------------------------------
def horizontal_seam_dp(energy: np.ndarray):
    seam_T = vertical_seam_dp(energy.T)
    seam = [(c, r) for (r, c) in seam_T]
    return seam


# ----------------------------------------------------
# REMOVE VERTICAL SEAM
# ----------------------------------------------------
def remove_vertical_seam(img: np.ndarray, seam):
    H, W, C = img.shape
    out = np.zeros((H, W - 1, C), dtype=img.dtype)

    for (r, c) in seam:
        out[r, :, :] = np.delete(img[r, :, :], c, axis=0)

    return out


# ----------------------------------------------------
# REMOVE HORIZONTAL SEAM
# ----------------------------------------------------
def remove_horizontal_seam(img: np.ndarray, seam):
    H, W, C = img.shape
    out = np.zeros((H - 1, W, C), dtype=img.dtype)

    for (r, c) in seam:
        out[:, c, :] = np.delete(img[:, c, :], r, axis=0)

    return out


# ----------------------------------------------------
# MULTIPLE SEAMS: VERTICAL
# ----------------------------------------------------
def carve_vertical_n(img: np.ndarray, n: int):
    H, W, _ = img.shape
    n = max(0, min(n, W - 1))  # safety clamp

    for _ in range(n):
        energy = compute_energy(img)
        seam = vertical_seam_dp(energy)
        img = remove_vertical_seam(img, seam)
    return img


# ----------------------------------------------------
# MULTIPLE SEAMS: HORIZONTAL
# ----------------------------------------------------
def carve_horizontal_n(img: np.ndarray, n: int):
    H, W, _ = img.shape
    n = max(0, min(n, H - 1))  # safety clamp

    for _ in range(n):
        energy = compute_energy(img)
        seam = horizontal_seam_dp(energy)
        img = remove_horizontal_seam(img, seam)
    return img


# ----------------------------------------------------
# RESIZE TO TARGET SIZE
# ----------------------------------------------------
def seam_carve_to_size(img: np.ndarray, new_h: int, new_w: int):
    H, W, _ = img.shape

    if new_w > W or new_h > H:
        raise ValueError("New size must be <= original size for seam carving")

    while W > new_w:
        energy = compute_energy(img)
        seam = vertical_seam_dp(energy)
        img = remove_vertical_seam(img, seam)
        H, W, _ = img.shape

    while H > new_h:
        energy = compute_energy(img)
        seam = horizontal_seam_dp(energy)
        img = remove_horizontal_seam(img, seam)
        H, W, _ = img.shape

    return img
