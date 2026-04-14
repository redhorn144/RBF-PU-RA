import numpy as np
from scipy.spatial import cKDTree

#---------------------------------------------------------------------------
# This file contains utilities for tiling the domain with patches
# ---------------------------------------------------------------------------

def BoxGridTiling2D(eval_nodes, groups, n_interp, eval_density, oversample_factor, overlap):
    x_min, y_min = np.min(eval_nodes, axis=0)
    x_max, y_max = np.max(eval_nodes, axis=0)

    r = np.sqrt(oversample_factor*n_interp / (np.pi * eval_density))

    x_centers = np.arange(x_min, x_max + r, r * (1 - overlap))
    y_centers = np.arange(y_min, y_max + r, r * (1 - overlap))
    centers = np.array([(x, y) for x in x_centers for y in y_centers])
    
    return centers, r






    