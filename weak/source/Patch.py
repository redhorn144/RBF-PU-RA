from dataclasses import dataclass
import numpy as np
import numba as nb

@dataclass
class Patch:
    """Data container for one PU-RBF patch. All fields are numpy arrays."""
    center: np.ndarray          # (d,)           patch center
    radius: float               # support radius
    node_indices: np.ndarray    # (n_local,)     global indices of nodes in this patch
    nodes: np.ndarray           # (n_local, d)   local node coordinates
    normals: np.ndarray         # (n_local, d)   normal vectors at local nodes
    Phi: np.ndarray             # (n_local, n_local) interpolation matrix
    D: np.ndarray               # (d, n_local, n_local) gradient matrices [D_x0, D_x1, ...]
    L: np.ndarray               # (n_local, n_local) Laplacian matrix
    # PU weights at each local node (precomputed, normalized)
    w_bar: np.ndarray           # (n_local,)     normalized C2 PU weight
    gw_bar: np.ndarray          # (n_local, d)   gradient of normalized weight
    lw_bar: np.ndarray          # (n_local,)     Laplacian of normalized weight
    q_hat: np.ndarray = None    # (n_local,)     local quadrature weights (set by Quadrature)
    # Non-nodal Gauss-point evaluation data (set by SetupGaussEval)
    gauss_local_idx: np.ndarray = None   # (M_local,) indices into global gauss_pts
    E_gauss:         np.ndarray = None   # (M_local, n_local) PHS eval matrix
    GE_gauss:        np.ndarray = None   # (d, M_local, n_local) PHS gradient eval
    w_bar_gauss:     np.ndarray = None   # (M_local,) normalized PU weight at Gauss pts
    gw_bar_gauss:    np.ndarray = None   # (M_local, d) PU weight gradient at Gauss pts