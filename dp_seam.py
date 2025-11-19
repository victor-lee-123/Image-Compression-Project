import numpy as np
from energy import compute_energy


# ---------------------------------------------------------
# VECTORIZED DP — FAST VERSION
# ---------------------------------------------------------

def vertical_seam_dp(energy):
    """
    Fast vectorized DP algorithm.
    Input:  energy (H, W)
    Output: list of (row, col) seam positions.
    """
    H, W = energy.shape

    # dp table
    dp = energy.astype(np.float64).copy()
    parent = np.zeros((H, W), dtype=np.int32)

    for r in range(1, H):
        prev = dp[r - 1]

        # shift arrays for up-left, up, up-right
        left  = np.roll(prev, 1)
        right = np.roll(prev, -1)
        middle = prev

        # very important: invalidate wrapped edges
        left[0] = np.inf
        right[-1] = np.inf

        # compute best parents
        candidates = np.vstack([middle, left, right])
        best = np.argmin(candidates, axis=0)

        # store parents
        parent[r] = np.where(best == 0, np.arange(W),
                      np.where(best == 1, np.arange(W) - 1,
                                         np.arange(W) + 1))

        # accumulate dp cost
        dp[r] = energy[r] + candidates[best, np.arange(W)]

    # traceback (backwards)
    seam = []
    c = int(np.argmin(dp[-1]))
    for r in range(H - 1, -1, -1):
        seam.append((r, c))
        c = parent[r, c]
    seam.reverse()
    return seam


def horizontal_seam_dp(energy):
    seam_T = vertical_seam_dp(energy.T)
    return [(c, r) for (r, c) in seam_T]
