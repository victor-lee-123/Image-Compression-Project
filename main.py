# main.py
import cv2
import numpy as np

from energy import compute_energy
from greedy_seam import find_vertical_seam_greedy


def draw_vertical_seam(img: np.ndarray, seam: np.ndarray,
                       color=(0, 0, 255)) -> np.ndarray:
    """
    Draw a vertical seam on the image.

    Parameters
    ----------
    img : np.ndarray
        Original image (H x W x 3).
    seam : np.ndarray
        1D array of length H, seam[i] = column index in row i.
    color : tuple
        BGR color to draw seam (default red).

    Returns
    -------
    vis : np.ndarray
        Image with seam drawn.
    """
    vis = img.copy()
    H = img.shape[0]
    for row in range(H):
        col = int(seam[row])
        # safety clamp
        col = max(0, min(col, img.shape[1] - 1))
        vis[row, col] = color
    return vis


def main():
    img = cv2.imread("images/test.jpg")

    if img is None:
        print("Failed to load image! Make sure images/test.jpg exists.")
        return

    print("Image loaded successfully:", img.shape)

    # 1. Compute energy map
    energy_map = compute_energy(img)
    print("Energy map shape:", energy_map.shape)

    # 2. Find greedy vertical seam
    seam = find_vertical_seam_greedy(energy_map)
    print("Greedy vertical seam (col per row):", seam)
    print("Seam length:", len(seam), "expected height:", img.shape[0])

    # 3. Visualize seam on the image
    img_with_seam = draw_vertical_seam(img, seam)

    # Optionally normalize energy for display
    energy_norm = energy_map / energy_map.max()
    energy_norm = energy_norm.astype(np.float32)

    cv2.imshow("Original", img)
    cv2.imshow("Energy map (normalized)", energy_norm)
    cv2.imshow("Greedy vertical seam", img_with_seam)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
