# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.image import imread


# # ---------- Energy function ----------

# def compute_energy(image: np.ndarray) -> np.ndarray:
#     """Simple energy: gradient magnitude on grayscale image."""
#     gray = np.mean(image, axis=2)

#     dx = np.abs(np.diff(gray, axis=1, append=gray[:, -1:]))
#     dy = np.abs(np.diff(gray, axis=0, append=gray[-1:, :]))

#     return dx + dy


# # ---------- DP for seams ----------

# def _vertical_seam_dp(energy: np.ndarray):
#     """Internal: minimum-energy vertical seam on energy map."""
#     H, W = energy.shape

#     dp = energy.astype(np.float64).copy()
#     parent = np.zeros((H, W), dtype=np.int32)

#     for y in range(1, H):
#         for x in range(W):
#             best_x = x
#             best_val = dp[y - 1, x]

#             if x > 0 and dp[y - 1, x - 1] < best_val:
#                 best_val = dp[y - 1, x - 1]
#                 best_x = x - 1

#             if x < W - 1 and dp[y - 1, x + 1] < best_val:
#                 best_val = dp[y - 1, x + 1]
#                 best_x = x + 1

#             dp[y, x] += best_val
#             parent[y, x] = best_x

#     end_x = int(np.argmin(dp[-1]))
#     min_cost = float(dp[-1, end_x])

#     seam = []
#     x = end_x
#     for y in reversed(range(H)):
#         seam.append((y, x))
#         if y > 0:
#             x = parent[y, x]

#     seam.reverse()
#     return seam, min_cost


# def vertical_seam_dp(energy: np.ndarray):
#     return _vertical_seam_dp(energy)


# def horizontal_seam_dp(energy: np.ndarray):
#     """Use vertical DP on the transposed energy map."""
#     seam_T, cost = _vertical_seam_dp(energy.T)
#     # map back to (row, col)
#     seam = [(c, r) for (r, c) in seam_T]
#     return seam, cost


# # ---------- Seam removal ----------

# def remove_vertical_seam(img: np.ndarray, seam):
#     H, W, C = img.shape
#     new_img = np.zeros((H, W - 1, C), dtype=img.dtype)

#     for (y, x) in seam:
#         new_img[y, :x, :] = img[y, :x, :]
#         new_img[y, x:, :] = img[y, x + 1:, :]

#     return new_img


# def remove_horizontal_seam(img: np.ndarray, seam):
#     H, W, C = img.shape
#     new_img = np.zeros((H - 1, W, C), dtype=img.dtype)

#     for (r, c) in seam:
#         new_img[:r, c, :] = img[:r, c, :]
#         new_img[r:, c, :] = img[r + 1:, c, :]

#     return new_img


# # ---------- Convenience: carve many seams ----------

# def carve_vertical_n(img: np.ndarray, n: int):
#     for _ in range(n):
#         energy = compute_energy(img)
#         seam, _ = vertical_seam_dp(energy)
#         img = remove_vertical_seam(img, seam)
#     return img


# def carve_horizontal_n(img: np.ndarray, n: int):
#     for _ in range(n):
#         energy = compute_energy(img)
#         seam, _ = horizontal_seam_dp(energy)
#         img = remove_horizontal_seam(img, seam)
#     return img


# # ---------- Main demo ----------

# def main():
#     # Load image
#     img = imread("test.jpg").astype(np.float64)

#     print("Original shape:", img.shape)

#     # How many seams to remove (change these if you want)
#     num_v = 80   # vertical seams
#     num_h = 80   # horizontal seams

#     # Carve copies so we don’t modify the original
#     v_carved = carve_vertical_n(img.copy(), num_v)
#     h_carved = carve_horizontal_n(img.copy(), num_h)

#     print("After vertical carving:", v_carved.shape)
#     print("After horizontal carving:", h_carved.shape)

#     # Show results
#     plt.figure(figsize=(15, 4))

#     plt.subplot(1, 3, 1)
#     plt.title("Original")
#     plt.imshow(img.astype(np.uint8))
#     plt.axis("off")

#     plt.subplot(1, 3, 2)
#     plt.title(f"After {num_v} Vertical Seams")
#     plt.imshow(v_carved.astype(np.uint8))
#     plt.axis("off")

#     plt.subplot(1, 3, 3)
#     plt.title(f"After {num_h} Horizontal Seams")
#     plt.imshow(h_carved.astype(np.uint8))
#     plt.axis("off")

#     plt.tight_layout()
#     plt.show()


# if __name__ == "__main__":
#     main()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread


# ---------- Energy function ----------

def compute_energy(image: np.ndarray) -> np.ndarray:
    """Simple energy: gradient magnitude on grayscale image."""
    gray = np.mean(image, axis=2)

    dx = np.abs(np.diff(gray, axis=1, append=gray[:, -1:]))
    dy = np.abs(np.diff(gray, axis=0, append=gray[-1:, :]))

    return dx + dy


# ---------- DP for seams ----------

def _vertical_seam_dp(energy: np.ndarray):
    """Internal: minimum-energy vertical seam on energy map."""
    H, W = energy.shape

    dp = energy.astype(np.float64).copy()
    parent = np.zeros((H, W), dtype=np.int32)

    for y in range(1, H):
        for x in range(W):
            best_x = x
            best_val = dp[y - 1, x]

            if x > 0 and dp[y - 1, x - 1] < best_val:
                best_val = dp[y - 1, x - 1]
                best_x = x - 1

            if x < W - 1 and dp[y - 1, x + 1] < best_val:
                best_val = dp[y - 1, x + 1]
                best_x = x + 1

            dp[y, x] += best_val
            parent[y, x] = best_x

    end_x = int(np.argmin(dp[-1]))
    min_cost = float(dp[-1, end_x])

    seam = []
    x = end_x
    for y in reversed(range(H)):
        seam.append((y, x))
        if y > 0:
            x = parent[y, x]

    seam.reverse()
    return seam, min_cost


def vertical_seam_dp(energy: np.ndarray):
    return _vertical_seam_dp(energy)


def horizontal_seam_dp(energy: np.ndarray):
    """Use vertical DP on the transposed energy map."""
    seam_T, cost = _vertical_seam_dp(energy.T)
    # map back to (row, col)
    seam = [(c, r) for (r, c) in seam_T]
    return seam, cost


# ---------- Seam removal ----------

def remove_vertical_seam(img: np.ndarray, seam):
    H, W, C = img.shape
    new_img = np.zeros((H, W - 1, C), dtype=img.dtype)

    for (y, x) in seam:
        new_img[y, :x, :] = img[y, :x, :]
        new_img[y, x:, :] = img[y, x + 1:, :]

    return new_img


def remove_horizontal_seam(img: np.ndarray, seam):
    H, W, C = img.shape
    new_img = np.zeros((H - 1, W, C), dtype=img.dtype)

    for (r, c) in seam:
        new_img[:r, c, :] = img[:r, c, :]
        new_img[r:, c, :] = img[r + 1:, c]

    return new_img


# ---------- Convenience: carve many seams ----------

def carve_vertical_n(img: np.ndarray, n: int):
    H, W, _ = img.shape
    n = min(n, max(W - 1, 0))  # never carve down to width 0

    for _ in range(n):
        energy = compute_energy(img)
        seam, _ = vertical_seam_dp(energy)
        img = remove_vertical_seam(img, seam)
    return img


def carve_horizontal_n(img: np.ndarray, n: int):
    H, W, _ = img.shape
    n = min(n, max(H - 1, 0))  # never carve down to height 0

    for _ in range(n):
        energy = compute_energy(img)
        seam, _ = horizontal_seam_dp(energy)
        img = remove_horizontal_seam(img, seam)
    return img


# ---------- Seam carving to desired size (step c) ----------

def seam_carve_to_size(img: np.ndarray, new_h: int, new_w: int):
    """
    Resize image to (new_h, new_w) using seam carving.
    Assumes new_h <= current_h and new_w <= current_w.
    """
    H, W, _ = img.shape

    if new_w > W or new_h > H:
        raise ValueError("New size must be smaller than or equal to original size")

    # First adjust width using vertical seams
    while W > new_w:
        energy = compute_energy(img)
        seam, _ = vertical_seam_dp(energy)
        img = remove_vertical_seam(img, seam)
        H, W, _ = img.shape

    # Then adjust height using horizontal seams
    while H > new_h:
        energy = compute_energy(img)
        seam, _ = horizontal_seam_dp(energy)
        img = remove_horizontal_seam(img, seam)
        H, W, _ = img.shape

    return img


# ---------- Main demo ----------
import os  # ensure this is at the top of your file

def main():
    print("Running in:", os.getcwd())

    # Load image
    img_path = "monke.jpg"
    if not os.path.exists(img_path):
        print(f"ERROR: Cannot find {img_path} in this folder.")
        return

    img = imread(img_path).astype(np.float64)
    print("Loaded image:", img.shape)
    H, W, _ = img.shape

    # --- Ask user for independent X and Y percentages ---
    try:
        pct_x_str = input("Enter X scale percentage (width, e.g. 70 for 70%): ")
        pct_y_str = input("Enter Y scale percentage (height, e.g. 50 for 50%): ")
        scale_x = float(pct_x_str) / 100.0
        scale_y = float(pct_y_str) / 100.0
    except Exception:
        print("Invalid input, using 100% for both X and Y (no scaling).")
        scale_x = 1.0
        scale_y = 1.0

    # Clamp scales between 0.1 and 1.0 (10%–100%)
    if scale_x > 1.0:
        print("X scale > 100% is not supported for seam carving. Using 100% for X.")
        scale_x = 1.0
    if scale_y > 1.0:
        print("Y scale > 100% is not supported for seam carving. Using 100% for Y.")
        scale_y = 1.0
    if scale_x < 0.1:
        print("X scale < 10% is too small. Using 10% for X.")
        scale_x = 0.1
    if scale_y < 0.1:
        print("Y scale < 10% is too small. Using 10% for Y.")
        scale_y = 0.1

    target_w = max(1, int(W * scale_x))
    target_h = max(1, int(H * scale_y))

    print(f"Target size: width {target_w} ({int(scale_x*100)}%), "
          f"height {target_h} ({int(scale_y*100)}%)")

    # --- Seam carve to target size ---
    carved = seam_carve_to_size(img.copy(), target_h, target_w)
    print("Carved shape:", carved.shape)

    # --- Plot and save result ---
    fig = plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(img.astype(np.uint8))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(f"Seam Carved (X {int(scale_x*100)}%, Y {int(scale_y*100)}%)")
    plt.imshow(carved.astype(np.uint8))
    plt.axis("off")

    fig.tight_layout()
    fig.savefig("result.png")
    print("Saved result to result.png")

    try:
        plt.show()
    except Exception as e:
        print("Could not display window:", e)


if __name__ == "__main__":
    main()
