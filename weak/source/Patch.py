from dataclasses import dataclass, field
from typing import Optional
import numpy as np

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
    # Shape parameter from RA method (needed to evaluate RBF at quad points)
    shape_param: float = 0.0
    # Galerkin quadrature fields (populated by GenQuadrature in Setup)
    quad_pts:     Optional[np.ndarray] = None  # (n_q, d)       quadrature points (interior)
    quad_weights: Optional[np.ndarray] = None  # (n_q,)         weights with Jacobian
    Phi_q:        Optional[np.ndarray] = None  # (n_q, n_local) RBF basis at quad pts
    dPhi_q:       Optional[np.ndarray] = None  # (d, n_q, n_local) RBF grad at quad pts
    w_bar_q:      Optional[np.ndarray] = None  # (n_q,)         normalized PU weight at quad pts
    gw_bar_q:     Optional[np.ndarray] = None  # (n_q, d)       grad of normalized PU weight at quad pts