import numpy as np
from mpi4py import MPI

###############################################################################
# Distributed LSQR  (Paige & Saunders, 1982) — halo-exchange variant
#
#   • x  is LOCAL   — rank r owns the block x_r  of size (n_local,)
#   • b  is OWNED   — rank r holds b[owned_indices], size (n_owned,)
#   • u  is OWNED   — same distribution as b; produced by halo-exchange matvec
#   • v  is LOCAL   — column iterate; each rank holds its block
#
# Communication per iteration
# ---------------------------
#   1. halo exchange(s)   inside matvec  (point-to-point, O(8 * n_halo) floats)
#   2. Allreduce(scalar)  for ||u_new||  (beta normalisation — NEW vs. source/)
#   3. halo exchange(s)   inside rmatvec
#   4. Allreduce(scalar)  for ||v_new||  (alpha normalisation, unchanged)
#   5. Allreduce(scalar)  for ||x||      (convergence check, unchanged)
#
# The O(M) Allreduce from the old matvec is eliminated.
###############################################################################


def lsqr(comm, matvec, rmatvec, b,
         atol=1e-8, btol=1e-8, maxiter=None, show=False, reorth=False):
    """
    Distributed LSQR solver (halo-exchange variant).

    Parameters
    ----------
    comm     : MPI communicator
    matvec   : callable  (n_local,) -> (n_owned,)   forward  A @ v_local
    rmatvec  : callable  (n_owned,) -> (n_local,)   adjoint  A^T @ u_owned
    b        : (n_owned,)  rank-local owned slice of the right-hand side
    atol     : float  tolerance on ||A^T r|| / (||A|| ||r||)
    btol     : float  tolerance on ||r|| / ||b||
    maxiter  : int    iteration cap  (default: 4 * global M, estimated via Allreduce)
    show     : bool   print convergence info on rank 0
    reorth   : bool   full reorthogonalization against iteration history;
                      costs O(itn * n_owned) memory and one Allreduce of size
                      itn per iteration for both u and v

    Returns
    -------
    x_local : (n_local,)  rank-local block of the solution
    itn     : int         iterations taken
    rnorm   : float       final ||b - A x||  estimate
    """
    rank = comm.Get_rank()
    n_owned = len(b)

    # Global problem size (sum of n_owned across ranks) — used only for maxiter default
    M_global = comm.allreduce(n_owned, op=MPI.SUM)
    if maxiter is None:
        maxiter = 4 * M_global

    # ------------------------------------------------------------------ init
    u     = b.copy()
    beta1 = np.sqrt(comm.allreduce(np.dot(u, u), op=MPI.SUM))  # now needs Allreduce
    u    /= beta1

    v     = rmatvec(u)
    alpha = np.sqrt(comm.allreduce(np.dot(v, v), op=MPI.SUM))
    v    /= alpha

    w     = v.copy()
    x     = np.zeros_like(v)

    rhobar = alpha
    phibar = beta1
    norm_b = beta1
    norm_A = alpha
    alpha1 = alpha

    eps       = np.finfo(float).eps
    beta_tol  = eps * beta1
    alpha_tol = eps * alpha1

    rnorm   = phibar
    norm_Ar = alpha
    istop   = 0

    if reorth:
        U_hist = [u.copy()]   # list of (n_owned,) owned u vectors
        V_hist = [v.copy()]   # list of (n_local,) v vectors

    if show and rank == 0:
        print(f"{'itn':>6}  {'rnorm':>12}  {'||Ar||/||A||||r||':>20}")

    for itn in range(1, maxiter + 1):

        # ---- bidiagonalisation ----
        u_new = matvec(v) - alpha * u   # (n_owned,)

        if reorth:
            # Batch all u dot products into one Allreduce
            dots_local = np.array([np.dot(u_prev, u_new) for u_prev in U_hist])
            dots = np.empty_like(dots_local)
            comm.Allreduce(dots_local, dots, op=MPI.SUM)
            for coeff, u_prev in zip(dots, U_hist):
                u_new -= coeff * u_prev

        beta = np.sqrt(comm.allreduce(np.dot(u_new, u_new), op=MPI.SUM))

        if beta <= beta_tol:
            istop = 1
            break

        u = u_new / beta

        v_new = rmatvec(u) - beta * v

        if reorth:
            dots_local = np.array([np.dot(v_prev, v_new) for v_prev in V_hist])
            dots = np.empty_like(dots_local)
            comm.Allreduce(dots_local, dots, op=MPI.SUM)
            for coeff, v_prev in zip(dots, V_hist):
                v_new -= coeff * v_prev

        alpha = np.sqrt(comm.allreduce(np.dot(v_new, v_new), op=MPI.SUM))

        if alpha <= alpha_tol:
            istop = 2
            break

        v = v_new / alpha

        if reorth:
            U_hist.append(u.copy())
            V_hist.append(v.copy())

        # ---- plane rotation (QR step) ----
        rho    = np.hypot(rhobar, beta)
        c      = rhobar / rho
        s      = beta   / rho
        theta  = s * alpha
        rhobar = -c * alpha
        phi    = c * phibar
        phibar = s * phibar

        # ---- update solution and search direction ----
        x  += (phi / rho) * w
        w   = v - (theta / rho) * w

        # ---- convergence estimates ----
        norm_A  = np.hypot(norm_A, alpha)
        rnorm   = phibar
        norm_Ar = abs(phibar * alpha * c)

        test1 = rnorm   / (norm_b  + 1e-300)
        test2 = norm_Ar / (norm_A * rnorm + 1e-300)

        if show and rank == 0 and (itn <= 10 or itn % 10 == 0):
            print(f"{itn:>6}  {rnorm:>12.4e}  {test2:>20.4e}")

        if test1 < btol or test2 < atol:
            istop = 3
            break

    if show and rank == 0:
        msg = {0: 'maxiter reached', 1: 'beta=0 (exact)',
               2: 'alpha=0 (exact)', 3: 'converged'}[istop]
        print(f"Stopped ({msg}) after {itn} iterations,  ||r|| = {rnorm:.4e}")

    return x, itn, rnorm
