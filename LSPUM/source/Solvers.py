import numpy as np
from mpi4py import MPI

from .Operators import PoissonRowMatrices, GenMatFreeOps
from .Preconditioners import GenBlockJacobi, GenDiagEquil
from .LSQR import lsqr

#---------------------------------------------------------------------------
# Parallel iterative solver for the LS-RBF-PUM Poisson system.
#
# One call assembles the matrix-free operator, attaches a preconditioner,
# and hands the preconditioned system to the distributed LSQR in LSQR.py.
# Usage mirrors the other Gen* factories:
#
#     solve  = GenIterativeSolver(comm, patches, M, n_interp)
#     local_cs, itn, rnorm = solve(f)
#
# solve(f) returns a list of per-patch coefficient vectors (one (n_interp,)
# array per local patch, same order as `patches`).
#---------------------------------------------------------------------------

def GenIterativeSolver(comm, patches, M, n_interp,
                       bc_scale=100.0, preconditioner='block_jacobi',
                       atol=1e-10, btol=1e-10, maxiter=None, show=False):
    """
    Build a parallel iterative least-squares solver for the LS-PUM
    Poisson problem (interior rows = PUM Laplacian, Dirichlet rows = PUM
    interpolant, boundary-row weighting bc_scale).

    Parameters
    ----------
    comm     : MPI communicator
    patches  : list[Patch]   local patches on this rank (from Setup)
    M        : int           global number of eval nodes
    n_interp : int           DOFs per patch
    bc_scale : float         Dirichlet row weight (see Operators.PoissonRowMatrices)
    preconditioner : str
        'block_jacobi'  per-patch Cholesky of R_p^T R_p   (default; fastest)
        'equilibrate'   column-norm scaling
        'none'          plain LSQR
    atol, btol, maxiter, show
        Forwarded to source.LSQR.lsqr.

    Returns
    -------
    solve : callable
        solve(f) -> (local_cs, itn, rnorm)
        local_cs : list of (n_interp,) arrays, one per local patch.
        itn, rnorm : iteration count and final residual estimate from LSQR.
    """
    Rs = PoissonRowMatrices(patches, bc_scale)
    matvec, rmatvec = GenMatFreeOps(comm, patches, Rs, M, n_interp)

    if preconditioner == 'block_jacobi':
        apply_Pinv, apply_PinvT = GenBlockJacobi(Rs, n_interp)
    elif preconditioner == 'equilibrate':
        apply_Pinv, apply_PinvT = GenDiagEquil(Rs, n_interp)
    elif preconditioner in (None, 'none'):
        identity = lambda v: v
        apply_Pinv, apply_PinvT = identity, identity
    else:
        raise ValueError(f"Unknown preconditioner: {preconditioner!r}")

    def mv(y):  return matvec(apply_PinvT(y))
    def rm(u):  return apply_Pinv(rmatvec(u))

    def solve(f):
        y, itn, rnorm = lsqr(comm, mv, rm, f,
                             atol=atol, btol=btol, maxiter=maxiter, show=show)
        x = apply_PinvT(y)
        return [x[pi*n_interp:(pi+1)*n_interp] for pi in range(len(patches))], itn, rnorm

    return solve
