import numpy as np

#---------------------------------------------------------------------------
# This file contains utilities for tiling the domain with patches
# ---------------------------------------------------------------------------

def BoxGridTiling2D(eval_nodes, n_interp, oversample_factor, overlap):
    x_min, y_min = np.min(eval_nodes, axis=0)
    x_max, y_max = np.max(eval_nodes, axis=0)

    Lx = x_max - x_min
    Ly = y_max - y_min
    n_eval = eval_nodes.shape[0]
    eval_density = n_eval / (Lx * Ly)

    # Set r so each circular patch contains ~oversample_factor * n_interp nodes:
    #   pi * r^2 * eval_density = oversample_factor * n_interp
    r = np.sqrt(oversample_factor * n_interp / (np.pi * eval_density))

    # overlap = r / h, so h = r / overlap; larger overlap -> denser patch grid
    h = r / overlap
    Px = np.ceil(Lx / h).astype(int)
    Py = np.ceil(Ly / h).astype(int)

    x_centers = np.array([x_min + h/2 + i*h for i in range(Px)])
    y_centers = np.array([y_min + h/2 + i*h for i in range(Py)])
    centers = np.array([(x, y) for x in x_centers for y in y_centers])

    return centers, r

def ManualTiling2D(nx, ny, x_min, x_max, y_min, y_max, r):
    x_centers = np.linspace(x_min + r/2, x_max - r/2, nx, endpoint=True)
    y_centers = np.linspace(y_min + r/2, y_max - r/2, ny, endpoint=True)
    centers = np.array([(x, y) for x in x_centers for y in y_centers])
    return centers, r

def LarssonBox2D(H, xrange, yrange, delta):
    x_min, x_max = xrange
    y_min, y_max = yrange
    centers = []

    xcenters = np.arange(x_min + H/2, x_max, H)
    ycenters = np.arange(y_min + H/2, y_max, H)
    for x in xcenters:
        for y in ycenters:
            centers.append((x, y))
    r = (1 + delta) * np.sqrt(2) * H / 2
    return np.array(centers), r


    