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

def GenEvalPhixk(eval_points, x, e, k):
    """(M x N) derivative of Gaussian kernel w.r.t. x_k, evaluated at eval_points."""
    diff = eval_points[:, np.newaxis, :] - x[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    diff_k = diff[:, :, k]
    return -2 * e**2 * diff_k * np.exp(-(e * r) ** 2)

# ---------------------------------------------------------------------------
# Polyharmonic spline (PHS) helpers — parameter-free off-node evaluation.
# ---------------------------------------------------------------------------

def phs_kernel(pts_a, pts_b, s):
    """(M, N) PHS kernel matrix phi_ij = ||pts_a_i - pts_b_j||^s."""
    diff = pts_a[:, np.newaxis, :] - pts_b[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    return r ** s

def phs_kernel_grad(eval_pts, nodes, s, k):
    """(M, N) gradient of PHS kernel w.r.t. x_k of the eval point."""
    diff = eval_pts[:, np.newaxis, :] - nodes[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    diff_k = diff[:, :, k]
    safe_r = np.where(r == 0, 1.0, r)
    result = s * safe_r ** (s - 2) * diff_k
    result[r == 0] = 0.0
    return result

def phs_poly_block(pts, deg):
    """(M, q) Pascal-triangle monomial block up to total degree deg."""
    cols = []
    for total in range(deg + 1):
        for px in range(total + 1):
            py = total - px
            cols.append(pts[:, 0] ** px * pts[:, 1] ** py)
    return np.column_stack(cols)

def phs_poly_block_grad(pts, deg, k):
    """(M, q) derivative of the monomial block w.r.t. x_k."""
    cols = []
    for total in range(deg + 1):
        for px in range(total + 1):
            py = total - px
            if k == 0:
                cols.append(np.zeros(len(pts)) if px == 0
                            else px * pts[:, 0] ** (px - 1) * pts[:, 1] ** py)
            else:
                cols.append(np.zeros(len(pts)) if py == 0
                            else py * pts[:, 0] ** px * pts[:, 1] ** (py - 1))
    return np.column_stack(cols)

