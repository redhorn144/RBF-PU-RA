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
def StableFlatMatrices(nodes, K = 64, n = 16, m = 48):
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
    a_phi, _ = GenRAab(phis_flat, es, n, m)
    a_lap, _ = GenRAab(laps_flat, es, n, m)

    a_grad = np.empty((d, m+1, N*N))
    for i in range(d):
        a_grad[i], _ = GenRAab(grads_flat[:, i, :], es, n, m)

    # flat limit = a_0 coefficients, cast to real (imag parts are numerical noise)
    phi_stable = a_phi[0].real.reshape(N, N)
    lap_stable = a_lap[0].real.reshape(N, N)
    grad_stable = a_grad[:, 0, :].real.reshape(d, N, N)
    
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
    return np.linalg.norm(im_phi, ord=np.inf)/np.linalg.norm(np.linalg.inv(re_phi), ord=np.inf)

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