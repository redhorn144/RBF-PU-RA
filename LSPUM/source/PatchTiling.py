import numpy as np
from scipy.spatial import cKDTree

#------------------------------------------------------------------------------------
# This file consisits of a function that generates a tiling of patches for the LS-RBF-PUM
# different choices are given for how to tile the domain
#------------------------------------------------------------------------------------


def GenPatchTiling(eval_interior, eval_boundary, tiling_choice=("grid",)):
    if tiling_choice[0] == "grid":
        overlap = tiling_choice[1] if len(tiling_choice) > 1 else 1.5
        n_side  = tiling_choice[2] if len(tiling_choice) > 2 else None
        return grid_tiling(eval_interior, eval_boundary, overlap=overlap, n_side=n_side)
    else:
        raise ValueError("Unsupported tiling choice: choose 'grid'.")


def grid_tiling(eval_interior, eval_boundary, overlap=1.5, n_side=None):
    """
    Cover the domain with circles whose centers lie on a uniform Cartesian grid.

    Every domain node (interior or boundary) is guaranteed to fall inside at
    least one patch.  Grid cells that contain no domain nodes are discarded,
    so non-convex and irregular domains are handled correctly.

    Parameters
    ----------
    eval_interior : (N_i, d) array
        Interior evaluation nodes.
    eval_boundary : (N_b, d) array
        Boundary evaluation nodes.
    overlap : float
        Overlap factor (> 1).  Patch radius is ``overlap * h*sqrt(d)/2``,
        where ``h`` is the grid spacing.  Larger values give more inter-patch
        overlap at the cost of larger local systems.  Default: 1.5.
    n_side : int or None
        Number of grid intervals along the longest axis.  If None, defaults
        to ``max(2, int(sqrt(N) / 2))`` where N is the total node count.

    Returns
    -------
    centers : (M, d) array
        Patch center coordinates (one row per active patch).
    r : float
        Patch radius (same for every patch).
    """
    nodes = np.vstack([eval_interior, eval_boundary])
    d = nodes.shape[1]
    N = len(nodes)

    lo = nodes.min(axis=0)
    hi = nodes.max(axis=0)

    if n_side is None:
        n_side = max(2, int(np.sqrt(N) / 2))

    h = (hi - lo).max() / n_side

    # Minimum radius to cover every point in a grid cell, times the overlap factor
    r = overlap * h * np.sqrt(d) / 2

    # Candidate centers: one per grid point in the bounding box
    axes = [np.arange(lo[k], hi[k] + h, h) for k in range(d)]
    grids = np.meshgrid(*axes, indexing='ij')
    candidates = np.column_stack([g.ravel() for g in grids])

    # Discard centers with no domain nodes within radius r
    tree = cKDTree(nodes)
    counts = tree.query_ball_point(candidates, r=r, return_length=True)
    centers = candidates[counts > 0]

    return centers, r
