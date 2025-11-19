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


def seam_carve_to_size_dp(img, new_h: int, new_w: int):
    """
    Resize image to (new_h, new_w) using DP-based seam carving.
    Assumes new_h <= H and new_w <= W.
    """
    H, W, _ = img.shape

    # Adjust width first
    while W > new_w:
        energy = compute_energy(img)
        seam = vertical_seam_dp(energy)
        img = remove_vertical_seam(img, seam)
        H, W, _ = img.shape

    # Then adjust height
    while H > new_h:
        energy = compute_energy(img)
        seam = horizontal_seam_dp(energy)
        img = remove_horizontal_seam(img, seam)
        H, W, _ = img.shape

    return img
