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
    # Oversampling fields (None unless SetupOversampled was used)
    eval_indices: np.ndarray    # (m_local,)     global indices into eval_pts array
    eval_nodes: np.ndarray      # (m_local, d)   collocation point coordinates
    Phi_eval: np.ndarray        # (m_local, n_local) RBF kernel at eval pts
    D_eval: np.ndarray          # (d, m_local, n_local) gradient at eval pts
    L_eval: np.ndarray          # (m_local, n_local) Laplacian at eval pts
    w_bar_eval: np.ndarray      # (m_local,)     normalized PU weight at eval pts
    gw_bar_eval: np.ndarray     # (m_local, d)   PU weight gradient at eval pts
    lw_bar_eval: np.ndarray     # (m_local,)     PU weight Laplacian at eval pts