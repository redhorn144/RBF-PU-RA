import numpy as np
#########################################
#Helpful wrappers for the base functions
#########################################

def GenMatrices(x, e):
    d = x.shape[1]
    n = x.shape[0]
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]   # (n, n, d)
    r = np.linalg.norm(diff, axis=2)                     # (n, n)
    Phi = np.exp(-(e * r) ** 2)

    # Build all d RHS matrices at once: (d, n, n)
    PhiXk = -2 * e**2 * diff.transpose(2, 0, 1) * Phi   # diff_k * Phi for each k
    PhiL  = 2 * e**2 * Phi * (2 * e**2 * r**2 - d)

    # Stack all RHS matrices: (n, n*(d+1))
    RHS = np.hstack([PhiXk.reshape(d * n, n).T, PhiL])   # (n, n*d + n)
    Sol = np.linalg.solve(Phi, RHS)                        # single LAPACK call

    # Grad[k] = (Phi^{-1} PhiXk_k^T)^T = PhiXk_k Phi^{-1}  (PhiXk is anti-symmetric)
    Grad = Sol[:, :d * n].T.reshape(d, n, n)
    # Lap = (Phi^{-1} PhiL)^T = PhiL Phi^{-1}              (PhiL is symmetric)
    Lap  = Sol[:, d * n:].T
    return Phi, Grad, Lap

#########################################
# Interpolation matrices for the kernel
# and the derivative matrives 
# (weight derivatives not nodal derivatives).
#########################################
def GenPhi(x, e):
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    return np.exp(-(e * r) ** 2)

def GenPhixk(x, e, k):
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    diff_k = diff[:, :, k]
    return -2 * e**2 * diff_k * np.exp(-(e * r) ** 2)

def GenPhiL(x, e):
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    d = x.shape[1]
    return 2 * e**2 * np.exp(-(e * r) ** 2) * (2 * e**2 * r**2 - d)

def GenEvalPhi(eval_points, x, e):
    diff = eval_points[:, np.newaxis, :] - x[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    return np.exp(-(e * r) ** 2)

def GenMatricesEval(eval_pts, nodes, e):
    """
    Generate rectangular RBF differentiation matrices for oversampled collocation.

    Parameters
    ----------
    eval_pts : (m, d) array — collocation/evaluation points
    nodes    : (n, d) array — RBF centers (solution nodes)
    e        : scalar — shape parameter

    Returns
    -------
    Phi_nn : (n, n) — interpolation matrix at solution nodes (used for inversion)
    D_eval : (d, m, n) — gradient matrices at eval_pts with centers at nodes
    L_eval : (m, n) — Laplacian matrix at eval_pts with centers at nodes
    """
    d = nodes.shape[1]
    n = nodes.shape[0]
    m = eval_pts.shape[0]

    # Square interpolation matrix at solution nodes
    diff_nn = nodes[:, np.newaxis, :] - nodes[np.newaxis, :, :]   # (n, n, d)
    r_nn = np.linalg.norm(diff_nn, axis=2)                         # (n, n)
    Phi_nn = np.exp(-(e * r_nn) ** 2)

    # Rectangular kernel matrices at eval_pts (rows) with centers at nodes (cols)
    diff_eval = eval_pts[:, np.newaxis, :] - nodes[np.newaxis, :, :]  # (m, n, d)
    r_eval = np.linalg.norm(diff_eval, axis=2)                         # (m, n)
    Phi_eval = np.exp(-(e * r_eval) ** 2)                              # (m, n)

    # Gradient kernel: ∂Φ/∂y_k(y_i, x_j) where y=eval_pts, x=nodes
    PhiXk_eval = -2 * e**2 * diff_eval.transpose(2, 0, 1) * Phi_eval  # (d, m, n)

    # Laplacian kernel: ΔᵧΦ(y_i, x_j)
    PhiL_eval = 2 * e**2 * Phi_eval * (2 * e**2 * r_eval**2 - d)      # (m, n)

    # Differentiation matrices: D_eval[k] = PhiXk_eval[k] @ Phi_nn^{-1},  L_eval = PhiL_eval @ Phi_nn^{-1}
    # Solve Phi_nn @ X = RHS where RHS columns come from the eval kernels
    # Stack RHS: (n, d*m + m)
    RHS = np.hstack([PhiXk_eval.reshape(d * m, n).T, PhiL_eval.T])  # (n, m*(d+1))
    Sol = np.linalg.solve(Phi_nn, RHS)                                # (n, m*(d+1))

    D_eval = Sol[:, :d * m].T.reshape(d, m, n)   # (d, m, n)
    L_eval = Sol[:, d * m:].T                     # (m, n)

    return Phi_nn, D_eval, L_eval

