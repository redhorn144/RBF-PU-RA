from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class Patch:
    """
    Data container for one LS-RBF-PUM patch.

    Geometry
    --------
    center       : (d,)              patch centre
    radius       : float             support radius
    node_indices : (n_eval,)         global indices into the full eval-node array
    nodes        : (n_eval, d)       coordinates of eval nodes in this patch
    normals      : (n_eval, d)       outward normal at each eval node (zero for interior)
    interp_nodes : (n_interp, d)     fixed RBF interpolation nodes (e.g. Vogel points)

    Boundary information
    --------------------
    is_boundary  : (n_eval,) bool    True for boundary eval nodes
    bc_values    : (n_eval,)         Dirichlet value at boundary nodes, NaN for interior

    RBF evaluation matrices  (n_eval × n_interp, built at shape param e_ref)
    -------------------------
    Phi          : (n_eval, n_interp)
    D            : (d, n_eval, n_interp)   spatial derivatives
    L            : (n_eval, n_interp)      Laplacian

    PU weights  (filled by NormalizeWeights after all patches are built)
    ----------
    w_bar        : (n_eval,)         normalised Wendland C2 weight
    gw_bar       : (n_eval, d)       gradient of normalised weight
    lw_bar       : (n_eval,)         Laplacian of normalised weight
    """
    # --- geometry ---
    center       : np.ndarray
    radius       : float
    node_indices : np.ndarray
    nodes        : np.ndarray
    normals      : np.ndarray
    interp_nodes : np.ndarray

    # --- boundary info ---
    is_boundary  : np.ndarray
    bc_values    : np.ndarray

    # --- RBF matrices ---
    Phi          : np.ndarray
    D            : np.ndarray
    L            : np.ndarray

    # --- PU weights (populated later) ---
    w_bar        : Optional[np.ndarray] = field(default=None)
    gw_bar       : Optional[np.ndarray] = field(default=None)
    lw_bar       : Optional[np.ndarray] = field(default=None)
