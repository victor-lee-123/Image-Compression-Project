import numpy as np

def find_vertical_seam_greedy(energy_map: np.ndarray) -> np.ndarray:
    """
    Find a vertical seam using a greedy strategy.

    Parameters
    ----------
    energy_map : np.ndarray
        2D array (H x W) of energy values.

    Returns
    -------
    seam : np.ndarray
        1D array of length H.
        seam[i] is the column index of the chosen pixel in row i.
    """
    H, W = energy_map.shape

    # Store the chosen column index in each row
    seam = np.zeros(H, dtype=np.int32)

    # Start from the top row: choose the minimum-energy pixel
    j = int(np.argmin(energy_map[0]))
    seam[0] = j

    # Go row by row, always picking the locally best next pixel
    for i in range(1, H):
        # Candidate columns in the next row: j-1, j, j+1 (clamped to image bounds)
        candidates = [j]
        if j > 0:
            candidates.append(j - 1)
        if j < W - 1:
            candidates.append(j + 1)

        # Pick the candidate with the smallest energy in row i
        best_col = min(candidates, key=lambda c: energy_map[i, c])
        seam[i] = best_col
        j = best_col

    return seam


def find_horizontal_seam_greedy(energy_map: np.ndarray) -> np.ndarray:
    """
    Find a horizontal seam using a greedy strategy.

    We reuse the vertical seam code by transposing the energy map.
    After transposing:
      - "rows" become original columns
      - "columns" become original rows

    Parameters
    ----------
    energy_map : np.ndarray
        2D array (H x W) of energy values.

    Returns
    -------
    seam : np.ndarray
        1D array of length W.
        seam[j] is the row index of the chosen pixel in column j.
    """
    # Apply vertical greedy on the transposed energy map
    # energy_map.T has shape (W x H)
    vertical_seam_on_transpose = find_vertical_seam_greedy(energy_map.T)

    # Now:
    #   index in vertical_seam_on_transpose = column index in original image
    #   value in vertical_seam_on_transpose = row index in original image
    # So this is exactly the horizontal seam representation we want.
    return vertical_seam_on_transpose
