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