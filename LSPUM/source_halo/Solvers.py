import numpy as np
from mpi4py import MPI

from .Operators import PoissonRowMatrices, GenMatFreeOps
from .Preconditioners import GenBlockJacobi, GenDiagEquil
from .LSQR import lsqr

#---------------------------------------------------------------------------
# Parallel iterative solver for the LS-RBF-PUM Poisson system.
#
# Halo-exchange variant: b and u vectors are distributed (each rank holds
# only the eval nodes in its home boxes).  The solver call changes to:
#
#     solve = GenIterativeSolver(comm, patches, halo, n_interp)
#     local_cs, itn, rnorm = solve(f_owned)
#
# where f_owned = f[halo.owned_indices] (the caller slices before passing).
#---------------------------------------------------------------------------

def GenIterativeSolver(comm, patches, halo, n_interp,
                       bc_scale=100.0, preconditioner='block_jacobi',
                       atol=1e-10, btol=1e-10, maxiter=None, show=False,
                       reorth=False):
    """
    Build a halo-exchange iterative least-squares solver.

    Parameters
    ----------
    comm     : MPI communicator
    patches  : list[Patch]   local patches (w_bar etc. already set by Setup)
    halo     : HaloComm      from Setup — owns the communication graph
    n_interp : int           DOFs per patch
    bc_scale : float         Dirichlet row weight
    preconditioner : str     'block_jacobi' | 'equilibrate' | 'none'
    atol, btol, maxiter, show, reorth
        Forwarded to LSQR.lsqr.

    Returns
    -------
    solve : callable
        solve(f_owned) -> (local_cs, itn, rnorm)
        f_owned  : (n_owned,) slice of the global RHS (f[halo.owned_indices])
        local_cs : list of (n_interp,) arrays, one per local patch
    """
    Rs = PoissonRowMatrices(patches, bc_scale)
    matvec, rmatvec = GenMatFreeOps(patches, Rs, halo, n_interp)

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

    def solve(f_owned):
        y, itn, rnorm = lsqr(comm, mv, rm, f_owned,
                             atol=atol, btol=btol, maxiter=maxiter,
                             show=show, reorth=reorth)
        x = apply_PinvT(y)
        return [x[pi*n_interp:(pi+1)*n_interp] for pi in range(len(patches))], itn, rnorm

    return solve
