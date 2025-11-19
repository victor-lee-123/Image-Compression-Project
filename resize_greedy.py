from energy import compute_energy
from greedy_seam import vertical_seam_greedy, horizontal_seam_greedy
from seam_removal import remove_vertical_seam, remove_horizontal_seam


def carve_vertical_n_greedy(img, n: int):
    """
    Remove n vertical seams using Greedy.
    """
    for _ in range(n):
        energy = compute_energy(img)
        seam = vertical_seam_greedy(energy)
        img = remove_vertical_seam(img, seam)
    return img


def carve_horizontal_n_greedy(img, n: int):
    """
    Remove n horizontal seams using Greedy.
    """
    for _ in range(n):
        energy = compute_energy(img)
        seam = horizontal_seam_greedy(energy)
        img = remove_horizontal_seam(img, seam)
    return img


def seam_carve_to_size_greedy(img, new_h: int, new_w: int):
    """
    Resize image to (new_h, new_w) using Greedy seam carving.
    Visually worse than DP by design – good for comparison.
    """
    H, W, _ = img.shape

    # Adjust width first
    while W > new_w:
        energy = compute_energy(img)
        seam = vertical_seam_greedy(energy)
        img = remove_vertical_seam(img, seam)
        H, W, _ = img.shape

    # Then adjust height
    while H > new_h:
        energy = compute_energy(img)
        seam = horizontal_seam_greedy(energy)
        img = remove_horizontal_seam(img, seam)
        H, W, _ = img.shape

    return img
