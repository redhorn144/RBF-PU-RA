import numpy as np
import numba as nb
from mpi4py import MPI

###############################
# GMRES solver for the global system
###############################

def gmres(comm, matvec, b, x0=None, tol=1e-6, restart=30, maxiter=None, precond=None):
    """
    Distributed GMRES with restart.
    
    Parameters
    ----------
    comm     : MPI communicator
    matvec   : callable, distributed A @ x
    b        : local portion of RHS
    x0       : initial guess (local portion)
    restart  : Krylov subspace size before restart (GMRES(m))
    precond  : callable, applies preconditioner M^-1 @ x
    """
    
    x = x0 if x0 is not None else np.zeros_like(b)
    total_iters = 0

    # Relative tolerance: ||r|| / ||b|| < tol
    b_norm = distributed_norm(comm, b)
    if b_norm == 0.0:
        b_norm = 1.0
    abs_tol = tol * b_norm

    for outer in range(maxiter):
        # --- Initial residual ---
        r = b - matvec(x)
        if precond:
            r = precond(r)
        
        beta = distributed_norm(comm, r)
        if beta < abs_tol:
            break
        
        # --- Arnoldi / GMRES inner loop ---
        x, converged, iters = gmres_cycle(comm, matvec, b, x, beta, restart, abs_tol, precond)
        total_iters += iters

        if converged:
            break
    
    return x, total_iters

def gmres_cycle(comm, matvec, b, x, beta, m, tol, precond):
    """Single restart cycle — builds Krylov subspace of size m."""
    
    rank = comm.Get_rank()
    
    # Hessenberg matrix (small, replicated on all ranks)
    H = np.zeros((m + 1, m))
    
    # Krylov basis vectors (distributed)
    r = b - matvec(x)
    V = [r / beta]
    
    # For least squares solve at the end
    g = np.zeros(m + 1)
    g[0] = beta
    
    # Givens rotations (replicated)
    cs = np.zeros(m)
    sn = np.zeros(m)
    
    for j in range(m):
        # --- Arnoldi step ---
        w = matvec(V[j])
        if precond:
            w = precond(w)
        
        # Modified Gram-Schmidt (distributed inner products)
        for i in range(j + 1):
            H[i, j] = distributed_dot(comm, V[i], w)
            w = w - H[i, j] * V[i]
        
        H[j + 1, j] = distributed_norm(comm, w)
        
        if H[j + 1, j] < 1e-14:  # Breakdown
            m = j + 1
            break
        
        V.append(w / H[j + 1, j])
        
        # --- Apply previous Givens rotations ---
        for i in range(j):
            H[i:i+2, j] = apply_givens(cs[i], sn[i], H[i:i+2, j])
        
        # --- New Givens rotation ---
        cs[j], sn[j] = compute_givens(H[j, j], H[j + 1, j])
        H[j, j]     =  cs[j] * H[j, j] + sn[j] * H[j + 1, j]
        H[j + 1, j] = 0.0
        g[j + 1]    = -sn[j] * g[j]
        g[j]        =  cs[j] * g[j]
        
        residual = abs(g[j + 1])
        if residual < tol:
            m = j + 1
            break
    
    # --- Solve upper triangular system (small, local) ---
    y = np.linalg.solve(H[:m, :m], g[:m])
    
    # --- Update solution (distributed) ---
    for i in range(m):
        x = x + y[i] * V[i]
    
    converged = residual < tol
    return x, converged, j + 1

def apply_givens(cs, sn, v):
    """Apply Givens rotation to a 2-element vector."""
    return np.array([cs * v[0] + sn * v[1], -sn * v[0] + cs * v[1]])

def compute_givens(a, b):
    """Compute Givens rotation coefficients."""
    if b == 0:
        return 1.0, 0.0
    r = np.sqrt(a**2 + b**2)
    return a / r, b / r

def distributed_dot(comm, u, v):
    """Global dot product across all ranks."""
    local_dot = np.dot(u, v)
    global_dot = comm.allreduce(local_dot, op=MPI.SUM)
    return global_dot

def distributed_norm(comm, v):
    return np.sqrt(distributed_dot(comm, v, v))


def lsqr(comm, op, op_T, b, tol=1e-6, maxiter=None):
    """
    Distributed LSQR (Paige & Saunders 1982).

    Solves min‖op(u) - b‖ using op (N→M) and op_T (M→N) without
    forming the normal equations. Converges at condition number κ(A)
    rather than κ(A)² as with GMRES on (AᵀA)u = Aᵀb.

    Parameters
    ----------
    comm    : MPI communicator
    op      : callable  u (N,) → v (M,)   — forward operator
    op_T    : callable  v (M,) → u (N,)   — adjoint operator
    b       : (M,) RHS in eval-point space
    tol     : relative residual tolerance  ‖r‖/‖b‖ < tol
    maxiter : maximum iterations (default 10*N)

    Returns
    -------
    x      : (N,) solution
    niters : number of iterations taken
    """
    # --- Initialize bidiagonalization ---
    b_norm = distributed_norm(comm, b)
    u = b / b_norm              # M-space unit vector
    beta = b_norm

    v = op_T(u)                 # N-space
    alpha = distributed_norm(comm, v)
    v = v / alpha

    w = v.copy()                # search direction (N-space)
    x = np.zeros_like(v)       # solution (N-space)

    phi_bar = beta
    rho_bar = alpha

    if maxiter is None:
        maxiter = 10 * len(x)

    niters = maxiter
    for i in range(maxiter):
        # --- Lanczos bidiagonalization ---
        u = op(v) - alpha * u   # M-space
        beta = distributed_norm(comm, u)
        u = u / beta

        v = op_T(u) - beta * v  # N-space
        alpha = distributed_norm(comm, v)
        v = v / alpha

        # --- QR step on 2×2 bidiagonal block ---
        rho     = np.sqrt(rho_bar**2 + beta**2)
        c       = rho_bar / rho
        s       = beta    / rho
        theta   = s * alpha
        rho_bar = -c * alpha
        phi     = c * phi_bar
        phi_bar = s * phi_bar

        # --- Update solution and search direction (N-space) ---
        x = x + (phi / rho) * w
        w = v - (theta / rho) * w

        # --- Stopping criterion: |φ̄| / β₁ tracks ‖r‖/‖b‖ ---
        if abs(phi_bar) / b_norm < tol:
            niters = i + 1
            break

    return x, niters