import cv2
import numpy as np

from energy import compute_energy
from dp_seam import vertical_seam_dp, horizontal_seam_dp
from greedy_seam import vertical_seam_greedy, horizontal_seam_greedy
from seam_removal import remove_vertical_seam, remove_horizontal_seam

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

# ============================================================
#  DP RESIZING ANIMATION  (Width + Height)
# ============================================================

def animate_resize_dp(img, new_h, new_w, delay=10):
    import time, sys
    temp = img.copy()
    H, W, _ = temp.shape

    total_v = W - new_w
    total_h = H - new_h
    total = total_v + total_h

    print(f"\nAnimating DP resize: removing {total} seams total...")
    start = time.time()
    count = 0

    # ===============================
    #   VERTICAL RESIZING
    # ===============================
    for i in range(total_v):
        energy = compute_energy(temp)
        seam = vertical_seam_dp(energy)

        vis = draw_vertical_seam(temp, seam)
        cv2.imshow("DP Resize Animation", vis)
        cv2.waitKey(delay)

        temp = remove_vertical_seam(temp, seam)

        # ---- TIMER + PROGRESS ----
        count += 1
        elapsed = time.time() - start
        pct = count / total * 100
        eta = elapsed / count * (total - count)

        sys.stdout.write(f"\rDP Resize: {count}/{total} seams ({pct:.1f}%) | "
                         f"{elapsed:.2f}s elapsed | ETA {eta:.2f}s")
        sys.stdout.flush()

    # ===============================
    #   HORIZONTAL RESIZING
    # ===============================
    H, W, _ = temp.shape
    for i in range(total_h):
        energy = compute_energy(temp)
        seam = horizontal_seam_dp(energy)

        vis = temp.copy()
        for (r, c) in seam:
            vis[r, c] = (0, 0, 255)

        cv2.imshow("DP Resize Animation", vis)
        cv2.waitKey(delay)

        temp = remove_horizontal_seam(temp, seam)

        # ---- TIMER + PROGRESS ----
        count += 1
        elapsed = time.time() - start
        pct = count / total * 100
        eta = elapsed / count * (total - count)

        sys.stdout.write(f"\rDP Resize: {count}/{total} seams ({pct:.1f}%) | "
                         f"{elapsed:.2f}s elapsed | ETA {eta:.2f}s")
        sys.stdout.flush()

    print("\nDP resizing completed! Result saved under test result folder\n")
    cv2.imshow("DP Final Resized", temp)
    cv2.waitKey(0)
    return temp


# ============================================================
#  GREEDY RESIZING ANIMATION  (Width + Height)
# ============================================================

def animate_resize_greedy(img, new_h, new_w, delay=10):
    import time, sys
    temp = img.copy()
    H, W, _ = temp.shape

    total_v = W - new_w
    total_h = H - new_h
    total = total_v + total_h

    print(f"\nAnimating Greedy resize: removing {total} seams total...")
    start = time.time()
    count = 0

    # ===============================
    #   VERTICAL RESIZING
    # ===============================
    for i in range(total_v):
        energy = compute_energy(temp)
        seam = vertical_seam_greedy(energy)

        vis = draw_vertical_seam(temp, seam)
        cv2.imshow("Greedy Resize Animation", vis)
        cv2.waitKey(delay)

        temp = remove_vertical_seam(temp, seam)

        # ---- TIMER + PROGRESS ----
        count += 1
        elapsed = time.time() - start
        pct = count / total * 100
        eta = elapsed / count * (total - count)

        sys.stdout.write(f"\rGreedy Resize: {count}/{total} seams ({pct:.1f}%) | "
                         f"{elapsed:.2f}s elapsed | ETA {eta:.2f}s")
        sys.stdout.flush()

    # ===============================
    #   HORIZONTAL RESIZING
    # ===============================
    H, W, _ = temp.shape
    for i in range(total_h):
        energy = compute_energy(temp)
        seam = horizontal_seam_greedy(energy)

        vis = temp.copy()
        for (r, c) in seam:
            vis[r, c] = (0, 0, 255)

        cv2.imshow("Greedy Resize Animation", vis)
        cv2.waitKey(delay)

        temp = remove_horizontal_seam(temp, seam)

        # ---- TIMER + PROGRESS ----
        count += 1
        elapsed = time.time() - start
        pct = count / total * 100
        eta = elapsed / count * (total - count)

        sys.stdout.write(f"\rGreedy Resize: {count}/{total} seams ({pct:.1f}%) | "
                         f"{elapsed:.2f}s elapsed | ETA {eta:.2f}s")
        sys.stdout.flush()

    print("\nGreedy resizing completed! Result saved under test result folder\n")
    cv2.imshow("Greedy Final Resized", temp)
    cv2.waitKey(0)
    return temp