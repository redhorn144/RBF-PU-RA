import numpy as np
from mpi4py import MPI

from .Operators import GenMatFreeOps
from .Preconditioners import GenBlockJacobi, GenDiagEquil, GenSAS
from .LSQR import lsqr
from .PCG import pcg


def GenIterativeSolver(comm, patches, halo, n_interp, Rs,
                       preconditioner='block_jacobi',
                       atol=1e-10, btol=1e-10, maxiter=None, show=False,
                       reorth=False):
    """
    Build a halo-exchange iterative least-squares solver.

    Parameters
    ----------
    comm     : MPI communicator
    patches  : list[Patch]
    halo     : HaloComm
    n_interp : int
    Rs       : list[(n_eval_p, n_interp) ndarray]  row matrices (e.g. PoissonRowMatrices)
    preconditioner : 'block_jacobi' | 'equilibrate' | 'none' | 'sas'
        'sas' uses PCG on the normal equations with a symmetric additive Schwarz
        left preconditioner; the others use right-preconditioned LSQR.

    Returns
    -------
    solve(f_owned) -> (local_cs, itn, rnorm)
        f_owned  : (n_owned,) owned slice of the RHS
        local_cs : list of (n_interp,) coefficient arrays, one per local patch
    """
    matvec, rmatvec = GenMatFreeOps(patches, Rs, halo, n_interp)

    if preconditioner == 'sas':
        apply_sas = GenSAS(comm, patches, Rs, n_interp)

        def solve(f_owned):
            x, itn, rnorm = pcg(comm, matvec, rmatvec, f_owned,
                                 M_inv=apply_sas, atol=atol, maxiter=maxiter,
                                 show=show)
            return [x[pi*n_interp:(pi+1)*n_interp] for pi in range(len(patches))], itn, rnorm

        return solve

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
