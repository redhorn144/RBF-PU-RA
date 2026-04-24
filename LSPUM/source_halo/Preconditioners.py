import numpy as np
from scipy.linalg import cholesky, solve_triangular

#---------------------------------------------------------------------------
# Preconditioners for the LS-RBF-PUM system.
#
# Each factory returns an (apply_Pinv, apply_PinvT) pair of closures that
# act on the rank-local DOF vector v_local.  The driver (Solvers.py) wraps
# them around the matrix-free matvec / rmatvec as a right preconditioner:
#
#     matvec_prec(y) = matvec(apply_PinvT(y))      # A · P^{-1}
#     rmatvec_prec(u) = apply_Pinv(rmatvec(u))     # P^{-T} · A^T
#
# After the solve the driver recovers x = P^{-1} y = apply_PinvT(y).
# All factories are patch-local; no communication.
#---------------------------------------------------------------------------

def GenBlockJacobi(Rs, n_interp, ridge=1e-14):
    """
    Block-Jacobi preconditioner in factored form.

    For each patch p, M_p = R_p^T R_p is the (n_interp, n_interp) diagonal
    block of A^T A.  Store its Cholesky L_p (lower) so M_p = L_p L_p^T and
    take P = block_diag(L_p^T), giving diag-block-identity for the
    preconditioned normal matrix (P^{-T} A^T A P^{-1})_{pp} = I.  The
    remaining patch-to-patch coupling from overlap is untouched.

    Parameters
    ----------
    Rs       : list[(n_eval_p, n_interp) array]  row matrices (PoissonRowMatrices)
    n_interp : int                               DOFs per patch
    ridge    : float                             tiny diagonal stabiliser

    Returns
    -------
    apply_Pinv  : callable  v_local -> block_diag(L_p^{-1})  v_local
    apply_PinvT : callable  v_local -> block_diag(L_p^{-T})  v_local
    """
    Ls = []
    Ieye = ridge * np.eye(n_interp)
    for R in Rs:
        Mp = R.T @ R + Ieye
        Ls.append(cholesky(Mp, lower=True))

    def apply_PinvT(v_local):
        out = np.empty_like(v_local)
        for pi, L in enumerate(Ls):
            out[pi*n_interp:(pi+1)*n_interp] = solve_triangular(
                L, v_local[pi*n_interp:(pi+1)*n_interp], lower=True, trans=1)
        return out

    def apply_Pinv(v_local):
        out = np.empty_like(v_local)
        for pi, L in enumerate(Ls):
            out[pi*n_interp:(pi+1)*n_interp] = solve_triangular(
                L, v_local[pi*n_interp:(pi+1)*n_interp], lower=True)
        return out

    return apply_Pinv, apply_PinvT


def GenDiagEquil(Rs, n_interp):
    """
    Column-equilibration preconditioner  P = diag(||A[:,j]||).

    Each DOF column lives entirely in its owning patch's rows, so the
    column norm is purely patch-local: ||A[:,j]|| = ||R_p[:,k]||.  P is
    diagonal so P^{-1} = P^{-T} and the two returned closures are the
    same function.

    Parameters
    ----------
    Rs       : list[(n_eval_p, n_interp) array]
    n_interp : int

    Returns
    -------
    apply_Pinv, apply_PinvT : callable  (both apply v_local -> (1/d) v_local)
    """
    d = np.empty(len(Rs) * n_interp)
    for pi, R in enumerate(Rs):
        d[pi*n_interp:(pi+1)*n_interp] = np.linalg.norm(R, axis=0)
    inv_d = np.where(d > 0, 1.0 / d, 1.0)

    def apply(v_local):
        return inv_d * v_local

    return apply, apply
