import numpy as np
import scipy.linalg as spla
from BaseHelpers import GenPhi
from BaseHelpers import GenMatrices
from scipy import optimize

######################################
# StableFlatMatrices generates the stable
# matrices from the RBF-RA method in the 
# flat limit.
# Called on each rank after the patches 
# are generated and boadcast.
######################################
def StableFlatMatrices(nodes, K = 64, n = 16, m = 48, eval_epsilon = 0):
    Er = GenEr(nodes)
    es = GenEs(K)
    d = nodes.shape[1]
    N = nodes.shape[0]

    #generate the matrices at each contour point
    phis = np.empty((len(es), N, N), dtype=complex)
    grads = np.empty((len(es), d, N, N), dtype=complex)
    laps = np.empty((len(es), N, N), dtype=complex)

    for i in range(len(es)):
        phis[i], grads[i], laps[i] = GenMatrices(nodes, es[i] * Er)

    #flatten all matrices into (K/2, N^2) for GenRAab
    phis_flat = phis.reshape(len(es), -1)
    grads_flat = grads.reshape(len(es), d, -1)
    laps_flat = laps.reshape(len(es), -1)

    #generate the rational approximant coefficients
    a_phi, b_phi = GenRAab(phis_flat, es, n, m)
    a_lap, b_lap = GenRAab(laps_flat, es, n, m)

    a_grad = np.empty((d, m+1, N*N))
    b_grad = np.empty((d, n+1))
    for i in range(d):
        a_grad[i], b_grad[i] = GenRAab(grads_flat[:, i, :], es, n, m)

    if eval_epsilon == 0:
        # flat limit = a_0 coefficients, cast to real (imag parts are numerical noise)
        phi_stable = a_phi[0].real.reshape(N, N)
        lap_stable = a_lap[0].real.reshape(N, N)
        grad_stable = a_grad[:, 0, :].real.reshape(d, N, N)
    else:
        # evaluate rational approximant at eval_epsilon, scaled by Er
        eps_scaled = eval_epsilon * Er
        phi_stable = EvalRA(a_phi, b_phi, eps_scaled).reshape(N, N)
        lap_stable = EvalRA(a_lap, b_lap, eps_scaled).reshape(N, N)
        grad_stable = np.empty((d, N, N))
        for i in range(d):
            grad_stable[i] = EvalRA(a_grad[i], b_grad[i], eps_scaled).reshape(N, N)

    return phi_stable, grad_stable, lap_stable

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
    a = spla.solve_triangular(R_mat, v)              # (m+1, M)

    # Prepend 1 to denominator: b = [1, b1, b2, ..., bn]
    b = np.concatenate([[1.0], b_coeffs])

    return a, b

def StableFlatMatricesOS(eval_pts, nodes, K=64, n=16, m=48, eval_epsilon=0):
    """
    Combined oversampled RA setup: one GenEr call, one contour loop, one solve.

    Computes stable flat-limit matrices for BOTH the square (solution-node) system
    and the rectangular (eval-point) system in a single pass.

    Parameters
    ----------
    eval_pts : (m_pts, d) — collocation/evaluation points
    nodes    : (n_pts, d) — RBF centers (solution nodes)

    Returns
    -------
    Phi_nn    : (n_pts, n_pts)       — square interpolation matrix (flat limit)
    D_nn      : (d, n_pts, n_pts)    — square gradient differentiation matrices
    L_nn      : (n_pts, n_pts)       — square Laplacian differentiation matrix
    Interp    : (m_pts, n_pts)       — evaluation matrix Φ_eval Φ_nn⁻¹ at eval pts
    D_eval    : (d, m_pts, n_pts)    — rectangular gradient differentiation matrices
    L_eval    : (m_pts, n_pts)       — rectangular Laplacian differentiation matrix
    """
    Er = GenEr(nodes)
    es = GenEs(K)
    d = nodes.shape[1]
    n_pts = nodes.shape[0]
    m_pts = eval_pts.shape[0]

    # Precompute geometry — independent of ε
    diff_nn    = nodes[:, np.newaxis, :] - nodes[np.newaxis, :, :]         # (n, n, d)
    r_nn       = np.linalg.norm(diff_nn, axis=2)                            # (n, n)
    diff_nn_T  = diff_nn.transpose(2, 0, 1)                                 # (d, n, n)

    diff_eval  = eval_pts[:, np.newaxis, :] - nodes[np.newaxis, :, :]      # (m, n, d)
    r_eval     = np.linalg.norm(diff_eval, axis=2)                          # (m, n)
    diff_eval_T = diff_eval.transpose(2, 0, 1)                              # (d, m, n)

    # Column layout of the combined RHS per contour point:
    #   [D_nn cols: d*n | L_nn cols: n | Interp cols: m | D_eval cols: d*m | L_eval cols: m]
    n_cols_nn   = d * n_pts + n_pts
    n_cols_eval = m_pts + d * m_pts + m_pts
    n_cols_tot  = n_cols_nn + n_cols_eval

    nn_grads   = np.empty((len(es), d, n_pts, n_pts), dtype=complex)
    nn_laps    = np.empty((len(es), n_pts, n_pts),    dtype=complex)
    nn_phis    = np.empty((len(es), n_pts, n_pts),    dtype=complex)
    ev_interp  = np.empty((len(es), m_pts, n_pts),    dtype=complex)
    ev_grads   = np.empty((len(es), d, m_pts, n_pts), dtype=complex)
    ev_laps    = np.empty((len(es), m_pts, n_pts),    dtype=complex)

    for i, e in enumerate(es):
        eps  = e * Er
        eps2 = eps ** 2

        Phi_nn   = np.exp(-(eps * r_nn)   ** 2)                             # (n, n)
        Phi_ev   = np.exp(-(eps * r_eval) ** 2)                             # (m, n)

        PhiXk_nn = -2.0 * eps2 * diff_nn_T   * Phi_nn                      # (d, n, n)
        PhiL_nn  =  2.0 * eps2 * Phi_nn  * (2.0 * eps2 * r_nn**2   - d)   # (n, n)

        PhiXk_ev = -2.0 * eps2 * diff_eval_T * Phi_ev                      # (d, m, n)
        PhiL_ev  =  2.0 * eps2 * Phi_ev  * (2.0 * eps2 * r_eval**2 - d)   # (m, n)

        # One combined solve: Phi_nn @ X = RHS, shape (n, n_cols_tot)
        RHS = np.empty((n_pts, n_cols_tot), dtype=complex)
        RHS[:, :d*n_pts]                          = PhiXk_nn.reshape(d*n_pts, n_pts).T
        RHS[:, d*n_pts:d*n_pts+n_pts]             = PhiL_nn
        RHS[:, n_cols_nn:n_cols_nn+m_pts]         = Phi_ev.T
        RHS[:, n_cols_nn+m_pts:n_cols_nn+m_pts+d*m_pts] = PhiXk_ev.reshape(d*m_pts, n_pts).T
        RHS[:, n_cols_nn+m_pts+d*m_pts:]          = PhiL_ev.T

        Sol = np.linalg.solve(Phi_nn, RHS)                                  # (n, n_cols_tot)

        nn_phis[i]   = Phi_nn
        nn_grads[i]  = Sol[:, :d*n_pts].T.reshape(d, n_pts, n_pts)
        nn_laps[i]   = Sol[:, d*n_pts:d*n_pts+n_pts].T
        ev_interp[i] = Sol[:, n_cols_nn:n_cols_nn+m_pts].T
        ev_grads[i]  = Sol[:, n_cols_nn+m_pts:n_cols_nn+m_pts+d*m_pts].T.reshape(d, m_pts, n_pts)
        ev_laps[i]   = Sol[:, n_cols_nn+m_pts+d*m_pts:].T

    def _ra(arr_flat):
        a, b = GenRAab(arr_flat, es, n, m)
        return a, b

    def _extract(a, shape, eps_scaled):
        if eval_epsilon == 0:
            return a[0].real.reshape(shape)
        return EvalRA(a, _ra(None)[1], eps_scaled).reshape(shape)  # placeholder — handled below

    eps_scaled = None if eval_epsilon == 0 else eval_epsilon * Er

    def _flat_or_eval(data_4d, shape):
        """data_4d: (K/2, *shape_flat) — fit RA and extract."""
        flat = data_4d.reshape(len(es), -1)
        a, b = GenRAab(flat, es, n, m)
        if eval_epsilon == 0:
            return a[0].real.reshape(shape)
        return EvalRA(a, b, eps_scaled).reshape(shape)

    def _flat_or_eval_grad(data_5d, d_dim, s0, s1):
        """data_5d: (K/2, d, s0, s1)"""
        result = np.empty((d_dim, s0, s1))
        for k in range(d_dim):
            flat = data_5d[:, k, :, :].reshape(len(es), -1)
            a, b = GenRAab(flat, es, n, m)
            if eval_epsilon == 0:
                result[k] = a[0].real.reshape(s0, s1)
            else:
                result[k] = EvalRA(a, b, eps_scaled).reshape(s0, s1)
        return result

    phi_nn_stable   = _flat_or_eval(nn_phis,   (n_pts, n_pts))
    lap_nn_stable   = _flat_or_eval(nn_laps,   (n_pts, n_pts))
    grad_nn_stable  = _flat_or_eval_grad(nn_grads, d, n_pts, n_pts)

    interp_stable   = _flat_or_eval(ev_interp, (m_pts, n_pts))
    lap_ev_stable   = _flat_or_eval(ev_laps,   (m_pts, n_pts))
    grad_ev_stable  = _flat_or_eval_grad(ev_grads, d, m_pts, n_pts)

    return phi_nn_stable, grad_nn_stable, lap_nn_stable, interp_stable, grad_ev_stable, lap_ev_stable


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