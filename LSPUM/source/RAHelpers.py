import numpy as np
from .BaseHelpers import *
from scipy import optimize
from scipy.linalg import lu_factor, lu_solve, solve_triangular

#--------------------------------------------------------------------------------
# RBF-RA-PUM: LS version, identical patch nodes in every patch.
#
# Phi(nodes, nodes) is complex symmetric (Gaussian RBF at complex epsilon).
# We pre-compute an LU factorisation of Phi once per contour point and reuse
# it across every patch, whose eval nodes vary but whose patch nodes are fixed.
#--------------------------------------------------------------------------------
def PhiFactors(nodes, K=64):
    """
    Pre-factorise Phi(nodes, nodes) at every contour epsilon.

    Returns
    -------
    phi_lus : list of (lu, piv) tuples from scipy.linalg.lu_factor, length K/2
    Er      : float  – reference shape parameter
    Es      : (K/2,) complex – contour points
    """
    Er = GenEr(nodes)
    Es = GenEs(K)
    phi_lus = [lu_factor(GenPhi(nodes, Es[i] * Er)) for i in range(len(Es))]
    return phi_lus, Er, Es


def StableMatricesLS(patch_eval_points, nodes, phi_lus, Er, Es,
                     n=16, m=48, eval_epsilon=0):
    """
    Compute stable interpolation/differentiation matrices for one patch's
    eval points, reusing the LU factors of Phi(nodes, nodes) that are
    identical across all patches.

    Parameters
    ----------
    patch_eval_points : (n_eval, d) array
        Evaluation points for this patch.
    nodes : (N, d) array
        Patch nodes (same for every patch).
    phi_lus : list of (lu, piv) tuples
        Output of PhiFactors – LU factors of Phi(nodes,nodes) at each epsilon.
    Er : float
        Reference shape parameter from GenEr(nodes).
    Es : (K/2,) complex array
        Contour points from GenEs(K).
    n, m : int
        Denominator / numerator degrees for the rational approximant.
    eval_epsilon : float
        If 0, return flat-limit matrices; otherwise evaluate RA at this epsilon.

    Returns
    -------
    phi_stable  : (n_eval, N)
    grad_stable : (d, n_eval, N)
    lap_stable  : (n_eval, N)
    """
    n_eval  = patch_eval_points.shape[0]
    N       = nodes.shape[0]
    d_space = nodes.shape[1]
    K       = len(Es)

    E_flat  = np.empty((K, n_eval * N), dtype=complex)
    grads_flat = np.empty((K, d_space, n_eval * N), dtype=complex)
    laps_flat  = np.empty((K, n_eval * N), dtype=complex)

    for i in range(K):
        eps = Es[i] * Er

        # eval-point kernel matrices — change per patch, Phi(nodes,nodes) does not
        phi_eval  = GenEvalPhi(patch_eval_points, nodes, eps)   # (n_eval, N)
        lap_eval  = GenEvalPhiL(patch_eval_points, nodes, eps)  # (n_eval, N)
        grad_eval = np.stack([                                   # (d, n_eval, N)
            GenEvalPhixk(patch_eval_points, nodes, eps, k)
            for k in range(d_space)
        ])

        # Solve Phi @ X = RHS for X = Phi^{-1} @ RHS.
        # RHS columns: [phi_eval | grad_eval[0] | ... | grad_eval[d-1] | lap_eval]
        # transposed so shape is (N, n_eval*(d+2))
        rhs = np.hstack(
            [phi_eval.T] +
            [grad_eval[k].T for k in range(d_space)] +
            [lap_eval.T]
        )  # (N, n_eval*(d_space+2))

        X = lu_solve(phi_lus[i], rhs)   # (N, n_eval*(d_space+2))

        # unpack: X.T rows are the eval matrices
        E_flat[i]  = X[:, :n_eval].T.reshape(-1)
        for k in range(d_space):
            grads_flat[i, k] = X[:, n_eval*(1+k):n_eval*(2+k)].T.reshape(-1)
        laps_flat[i] = X[:, n_eval*(1+d_space):].T.reshape(-1)

    # rational approximant fit
    a_E, b_E = GenRAab(E_flat, Es, n, m)
    a_lap, b_lap = GenRAab(laps_flat, Es, n, m)
    a_grad = np.empty((d_space, m+1, n_eval*N))
    b_grad = np.empty((d_space, n+1))
    for k in range(d_space):
        a_grad[k], b_grad[k] = GenRAab(grads_flat[:, k, :], Es, n, m)

    if eval_epsilon == 0:
        E_stable  = a_E[0].real.reshape(n_eval, N)
        lap_stable  = a_lap[0].real.reshape(n_eval, N)
        grad_stable = a_grad[:, 0, :].real.reshape(d_space, n_eval, N)
    else:
        eps_scaled  = eval_epsilon * Er
        E_stable  = EvalRA(a_E, b_E, eps_scaled).reshape(n_eval, N)
        lap_stable  = EvalRA(a_lap, b_lap, eps_scaled).reshape(n_eval, N)
        grad_stable = np.empty((d_space, n_eval, N))
        for k in range(d_space):
            grad_stable[k] = EvalRA(a_grad[k], b_grad[k], eps_scaled).reshape(n_eval, N)

    return E_stable, grad_stable, lap_stable


######################################
# Helpers for RA method
######################################
def GenEr(x):
    minimizer = optimize.fminbound(lambda e: ConditionObjective(e, x), 0.1, 20)
    return minimizer

def ConditionObjective(e, x):
    re_phi = GenPhi(x, e)
    im_phi = GenPhi(x, e * 1j)
    try:
        inv_re_phi = np.linalg.inv(re_phi)
    except np.linalg.LinAlgError:
        return np.inf
    return np.linalg.norm(im_phi, ord=np.inf)/np.linalg.norm(inv_re_phi, ord=np.inf)

def GenEs(K):
    K = int(K)
    k = 2 * np.arange(1, K // 2 + 1)
    thetas = (np.pi / 2) * (k / K)
    es = np.exp(1j * thetas)
    return es

def GenRAab(fj_mat, es, n, m):
    """
    Solve for the rational approximant coefficients a and b
    using the Figure 5 algorithm from Wright & Fornberg (2017).

    Parameters
    ----------
    fj_mat : (K/2, M) complex array
        Function evaluations at contour points. fj_mat[k, j] = f_j(ε_k).
    es : (K/2,) complex array
        Contour evaluation points in the first quadrant.
    n : int
        Denominator degree (in ε^2).
    m : int
        Numerator degree (in ε^2), so m+1 numerator coefficients per component.

    Returns
    -------
    a : (m+1, M) real array
        Numerator coefficients for each component.
    b : (n+1,) real array
        Denominator coefficients (b[0] = 1).
    """
    n = int(n)
    m = int(m)
    Khalf = len(es)          # K/2 complex points
    K = 2 * Khalf            # K real rows after splitting real/imag
    M = fj_mat.shape[1]      # number of components

    # --- Step 1: Row normalization ---
    # For each contour point, find max magnitude across all M components
    fmax = np.max(np.abs(fj_mat), axis=1)  # (K/2,)

    # Build the scaled E matrix using even powers of ε
    # E columns: [1/fmax, ε^2/fmax, ε^4/fmax, ..., ε^(2m)/fmax]
    e2 = es ** 2  # ε_k^2
    E = np.zeros((Khalf, m + 1), dtype=complex)
    E[:, 0] = 1.0 / fmax
    for col in range(1, m + 1):
        E[:, col] = E[:, col - 1] * e2

    # Build F matrices and RHS g for each component
    # F_j uses the first n+1 columns of E scaled by f_j values
    # Then g = F(:,0,:) and F = -F(:,1:n+1,:)
    Eblock = E[:, :n + 1]  # (K/2, n+1)
    # Broadcast: (K/2, n+1, 1) * (K/2, 1, M) -> (K/2, n+1, M)
    F_all = Eblock[:, :, np.newaxis] * fj_mat[:, np.newaxis, :]
    g_all = F_all[:, 0, :]             # (K/2, M) — the RHS
    F_all = -F_all[:, 1:n + 1, :]      # (K/2, n, M) — the F_j blocks

    # Split complex rows into real and imaginary parts → K real rows
    ER = np.vstack([E.real, E.imag])            # (K, m+1)
    FR = np.concatenate([F_all.real, F_all.imag], axis=0)  # (K, n, M)
    gr = np.vstack([g_all.real, g_all.imag])    # (K, M)

    # QR factorization of E
    Q, R_mat = np.linalg.qr(ER, mode='complete')
    QT = Q.T
    R_mat = R_mat[:m + 1, :]   # (m+1, m+1) upper triangular

    # --- Step 2: Left-multiply all F_j and g_j by Q^T ---
    # FR is (K, n, M), gr is (K, M)
    for j in range(M):
        FR[:, :, j] = QT @ FR[:, :, j]
        gr[:, j] = QT @ gr[:, j]

    # --- Step 3: Separate top (rows 0..m) and bottom (rows m+1..K-1) ---
    FT = FR[:m + 1, :, :]         # (m+1, n, M) — for numerator back-sub
    FB = FR[m + 1:, :, :]         # (K-m-1, n, M) — for denominator least-squares
    gt = gr[:m + 1, :]            # (m+1, M)
    gb = gr[m + 1:, :]            # (K-m-1, M)

    # Stack all M bottom blocks into one tall system for b
    # FB: (K-m-1, n, M) → reshape to (M*(K-m-1), n)
    FB_stacked = np.transpose(FB, (2, 0, 1)).reshape(M * (K - m - 1), n)
    gb_stacked = np.transpose(gb, (1, 0)).reshape(M * (K - m - 1))

    # --- Step 4: Least-squares solve for denominator coefficients b ---
    b_coeffs, _, _, _ = np.linalg.lstsq(FB_stacked, gb_stacked, rcond=None)
    # b_coeffs has shape (n,)

    # --- Step 5: Back-substitute for each numerator a_j ---
    # R * a_j = gt[:, j] - FT[:, :, j] @ b_coeffs
    v = gt - np.einsum('ijk,j->ik', FT, b_coeffs)  # (m+1, M)
    a = solve_triangular(R_mat, v)                   # (m+1, M)

    # Prepend 1 to denominator: b = [1, b1, b2, ..., bn]
    b = np.concatenate([[1.0], b_coeffs])

    return a, b

def polyval2(p, x):
    """
    Evaluate the even polynomial
      y = p[0] + p[1]*x^2 + p[2]*x^4 + ... + p[N]*x^(2N)
    using Horner's method.
    """
    y = np.zeros_like(x, dtype=complex if np.iscomplexobj(p) else float)
    x2 = x ** 2
    for j in range(len(p) - 1, -1, -1):
        y = x2 * y + p[j]
    return y


def EvalRA(a, b, epsilon):
    """
    Evaluate the vector-valued rational approximant at given epsilon values.

    Parameters
    ----------
    a : (m+1, M) array
        Numerator coefficients (from GenRAab).
    b : (n+1,) array
        Denominator coefficients with b[0]=1 (from GenRAab).
    epsilon : scalar or 1-D array
        Shape parameter value(s) to evaluate at (real, scaled by Er).

    Returns
    -------
    R : (M, len(epsilon)) array
        Rational approximant evaluated at each epsilon for each component.
    """
    epsilon = np.atleast_1d(np.asarray(epsilon, dtype=float))
    M = a.shape[1]
    denom = polyval2(b, epsilon)  # (len(epsilon),)
    R = np.zeros((M, len(epsilon)))
    for j in range(M):
        R[j, :] = polyval2(a[:, j], epsilon) / denom
    return R
