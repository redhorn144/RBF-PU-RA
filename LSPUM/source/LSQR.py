import numpy as np
from mpi4py import MPI

###############################################################################
# Distributed LSQR  (Paige & Saunders, 1982)
#
# Solves the least-squares problem
#
#       min_x  ||A x - b||_2
#
# where A is distributed across MPI ranks in the COLUMN dimension:
#
#   • x  is LOCAL  — rank r owns the block x_r  of size (n_local,)
#   • b  is GLOBAL — every rank holds the full M-vector (identical copies)
#   • u  is GLOBAL — the current row iterate; always identical on all ranks
#                    because matvec produces it via Allreduce
#   • v  is LOCAL  — the current column iterate; each rank holds its block
#
# Communication per iteration
# ---------------------------
#   1. Allreduce(M)       inside matvec  (forward pass, unavoidable)
#   2. Allreduce(scalar)  for ||v_new||  (alpha normalisation)
#   3. Allreduce(scalar)  for ||x||      (convergence check, same cost)
#
# Compare to wrapping scipy lsqr with a globalised rmatvec:
#   1. Allreduce(M)          forward
#   2. Allreduce(P_global*n) adjoint  ← eliminated here
#
# API
# ---
#   matvec(v_local)  ->  (M,)  same on every rank  [uses Allreduce internally]
#   rmatvec(u)       ->  (n_local,)  rank-local result  [no communication]
###############################################################################


def lsqr(comm, matvec, rmatvec, b,
         atol=1e-8, btol=1e-8, maxiter=None, show=False, reorth=False):
    """
    Distributed LSQR solver.

    Parameters
    ----------
    comm     : MPI communicator
    matvec   : callable  (n_local,) -> (M,)   forward operator  A @ v_local
    rmatvec  : callable  (M,)      -> (n_local,)  adjoint A^T @ u  (local output)
    b        : (M,)  right-hand side, same on all ranks
    atol     : float  tolerance on ||A^T r|| / (||A|| ||r||)
    btol     : float  tolerance on ||r|| / ||b||
    maxiter  : int    iteration cap  (default: 4 * len(b))
    show     : bool   print convergence info on rank 0
    reorth   : bool   if True, reorthogonalize u and v against the complete
                      iteration history (MGS-style) to counter finite-precision
                      loss of orthogonality; costs O(itn*M) memory and one
                      extra Allreduce of size itn per iteration for v

    Returns
    -------
    x_local : (n_local,)  rank-local block of the solution
    itn     : int         iterations taken
    rnorm   : float       final ||b - A x||  estimate
    """
    rank = comm.Get_rank()
    M    = len(b)

    if maxiter is None:
        maxiter = 4 * M

    # ------------------------------------------------------------------ init
    u     = b.copy()
    beta1 = np.sqrt(np.dot(u, u))        # u identical on all ranks → no Allreduce
    u    /= beta1

    v     = rmatvec(u)                   # local block of A^T u
    alpha = np.sqrt(comm.allreduce(np.dot(v, v), op=MPI.SUM))
    v    /= alpha

    w     = v.copy()
    x     = np.zeros_like(v)

    rhobar = alpha
    phibar = beta1
    norm_b = beta1
    norm_A = alpha      # running Frobenius estimate ||A||_F
    alpha1 = alpha      # saved for breakdown threshold

    # Absolute thresholds below which we treat a norm as zero.
    # Scaled to the problem so floating-point noise in a zero vector doesn't
    # trigger a false breakdown.
    eps = np.finfo(float).eps
    beta_tol = eps * beta1
    alpha_tol = eps * alpha1

    rnorm   = phibar
    norm_Ar = alpha
    norm_x  = 0.0
    istop   = 0         # 0=running, 1=beta breakdown, 2=alpha breakdown, 3=converged

    if reorth:
        U_hist = [u.copy()]   # global (M,) u vectors, identical on all ranks
        V_hist = [v.copy()]   # local (n_local,) v vectors

    if show and rank == 0:
        print(f"{'itn':>6}  {'rnorm':>12}  {'||Ar||/||A||||r||':>20}")

    for itn in range(1, maxiter + 1):

        # ---- bidiagonalisation ----
        u_new = matvec(v) - alpha * u   # M-vector, identical on all ranks

        if reorth:
            for u_prev in U_hist:
                u_new -= np.dot(u_prev, u_new) * u_prev

        beta  = np.sqrt(np.dot(u_new, u_new))   # identical on all ranks → no Allreduce

        if beta <= beta_tol:
            # A v_k = alpha u_k  exactly — the LS solution is in the current
            # Krylov subspace; x already holds the optimal iterate.
            istop = 1
            break

        u = u_new / beta

        v_new = rmatvec(u) - beta * v   # local block

        if reorth:
            dots_local = np.array([np.dot(v_prev, v_new) for v_prev in V_hist])
            dots = np.empty_like(dots_local)
            comm.Allreduce(dots_local, dots, op=MPI.SUM)
            for coeff, v_prev in zip(dots, V_hist):
                v_new -= coeff * v_prev

        alpha = np.sqrt(comm.allreduce(np.dot(v_new, v_new), op=MPI.SUM))

        if alpha <= alpha_tol:
            # A^T u_{k+1} = beta v_k  exactly — similar termination.
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
        rnorm   = phibar                          # ||b - A x|| estimate
        norm_Ar = abs(phibar * alpha * c)         # ||A^T r|| estimate
        norm_x  = np.sqrt(comm.allreduce(np.dot(x, x), op=MPI.SUM))

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
