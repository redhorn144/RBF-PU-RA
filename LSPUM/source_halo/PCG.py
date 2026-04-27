import numpy as np
from mpi4py import MPI

###############################################################################
# Preconditioned Conjugate Gradient on the normal equations A^T A x = A^T b.
#
# This is the correct solver for an additive Schwarz (RAS) left preconditioner.
# Each iteration applies:
#   1. matvec(p)   — A p  (owned space, with halo exchange)
#   2. rmatvec(Ap) — A^T (A p)  (local space, with halo exchange)
#   3. M_inv(r)    — left preconditioner apply (local → local)
#   4. Two global Allreduces for α and β
#
# Distributed layout:
#   x, r, z, p : LOCAL space  (n_local = n_patches_local * n_interp)
#   b           : OWNED space  (n_owned eval nodes)
###############################################################################


def pcg(comm, matvec, rmatvec, b,
        M_inv=None, atol=1e-10, maxiter=None, show=False):
    """
    Preconditioned CG on the normal equations A^T A x = A^T b.

    Parameters
    ----------
    comm     : MPI communicator
    matvec   : callable (n_local,) -> (n_owned,)   forward operator A
    rmatvec  : callable (n_owned,) -> (n_local,)   adjoint  operator A^T
    b        : (n_owned,) local slice of the RHS
    M_inv    : callable (n_local,) -> (n_local,)   left preconditioner M^{-1}
               (default: identity)
    atol     : convergence tolerance on ||A^T r|| / ||A^T b||
    maxiter  : iteration limit
    show     : print residual every 100 iterations

    Returns
    -------
    x     : (n_local,) solution
    itn   : iterations taken
    rnorm : ||Ax - b|| / ||b||  (for comparability with LSQR output)
    """
    if M_inv is None:
        M_inv = lambda v: v

    AtB = rmatvec(b)
    AtB_norm2 = comm.allreduce(float(np.dot(AtB, AtB)), op=MPI.SUM)
    b_norm2   = comm.allreduce(float(np.dot(b, b)),   op=MPI.SUM)
    AtB_norm  = np.sqrt(AtB_norm2)

    if AtB_norm == 0.0:
        return np.zeros_like(AtB), 0, 0.0

    if maxiter is None:
        maxiter = max(len(AtB) * 10, 1000)

    x   = np.zeros_like(AtB)
    r   = AtB.copy()            # r = A^T b  (x = 0)
    z   = M_inv(r)              # z = M^{-1} r
    p   = z.copy()
    rz  = comm.allreduce(float(np.dot(r, z)), op=MPI.SUM)

    rnorm_ls = 1.0

    for itn in range(1, maxiter + 1):
        Ap    = matvec(p)
        AtAp  = rmatvec(Ap)
        denom = comm.allreduce(float(np.dot(p, AtAp)), op=MPI.SUM)

        if denom <= 0.0:
            break

        alpha = rz / denom
        x = x + alpha * p
        r = r - alpha * AtAp

        # Convergence: ||A^T r|| / ||A^T b||
        r_norm2  = comm.allreduce(float(np.dot(r, r)), op=MPI.SUM)
        rnorm_ne = np.sqrt(r_norm2) / (AtB_norm + 1e-300)

        if show and itn % 100 == 0:
            if comm.Get_rank() == 0:
                print(f"  PCG itn={itn:5d}  ||A^T r||/||A^T b||={rnorm_ne:.3e}")

        if rnorm_ne <= atol:
            break

        z_new  = M_inv(r)
        rz_new = comm.allreduce(float(np.dot(r, z_new)), op=MPI.SUM)
        beta   = rz_new / rz
        p      = z_new + beta * p
        rz     = rz_new

    # Report ||Ax - b|| / ||b|| for consistency with LSQR
    Ax_final = matvec(x)
    res2     = comm.allreduce(float(np.dot(b - Ax_final, b - Ax_final)), op=MPI.SUM)
    rnorm_ls = np.sqrt(res2) / (np.sqrt(b_norm2) + 1e-300)

    return x, itn, rnorm_ls
