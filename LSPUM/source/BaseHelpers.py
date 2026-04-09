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
    diff = eval_points[:, np.newaxis, :] - x[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    diff_k = diff[:, :, k]
    return -2 * e**2 * diff_k * np.exp(-(e * r) ** 2)

def GenEvalPhiL(eval_points, x, e):
    diff = eval_points[:, np.newaxis, :] - x[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    d = x.shape[1]
    return 2 * e**2 * np.exp(-(e * r) ** 2) * (2 * e**2 * r**2 - d)