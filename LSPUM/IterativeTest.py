"""
IterativeTest.py — prototype matrix-free parallel LSQR for LS-RBF-PUM Poisson.

Goal
----
Make an iterative solver actually reach the paper's error floor on this
ill-conditioned LS system by comparing:

  1. plain LSQR (no preconditioner)
  2. LSQR + column equilibration
  3. LSQR + block-Jacobi preconditioner
  4. LSQR + block-Jacobi + full reorthogonalization

All operator and preconditioner logic lives in source/; this file only
handles problem setup, timing, and error reporting.
"""

from mpi4py import MPI
import numpy as np

from nodes.SquareDomain import MinEnergySquareOne
from source.PatchTiling import LarssonBox2D
from source.LSSetup import Setup
from source.Solvers import GenIterativeSolver


# ---------------------------------------------------------------------------
# PUM interpolant for post-solve evaluation.
# Not yet in source/, so kept here.
# ---------------------------------------------------------------------------
def global_interp(comm, patches, v_local, M, n_interp):
    out_local = np.zeros(M)
    for pi, p in enumerate(patches):
        c_p = v_local[pi*n_interp:(pi+1)*n_interp]
        out_local[p.eval_node_indices] += p.w_bar * (p.E @ c_p)
    out = np.empty(M)
    comm.Allreduce(out_local, out, op=MPI.SUM)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import time

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # --- problem ---
    M        = 8000
    n_interp = 40
    H        = 0.1
    delta    = 0.2
    bc_scale = 100.0
    maxiter  = 20000
    atol = btol = 1e-10

    if rank == 0:
        eval_nodes, normals, groups = MinEnergySquareOne(M)
        centers, r = LarssonBox2D(H=H, xrange=(0, 1), yrange=(0, 1), delta=delta)
    else:
        eval_nodes = normals = groups = centers = r = None
    eval_nodes = comm.bcast(eval_nodes, root=0)
    normals    = comm.bcast(normals, root=0)
    groups     = comm.bcast(groups, root=0)
    centers    = comm.bcast(centers, root=0)
    r          = comm.bcast(r, root=0)

    bc_flags = np.empty(len(eval_nodes), dtype=str)
    bc_flags[groups["boundary:all"]] = 'd'
    bc_flags[groups["interior"]]     = 'i'

    if rank == 0:
        print(f"M={M}  n_interp={n_interp}  patches={len(centers)}  "
              f"bc_scale={bc_scale}  ranks={size}")

    patches = Setup(comm, eval_nodes, normals, bc_flags, centers, r,
                    n_interp=n_interp, node_layout='vogel',
                    assignment='round_robin',
                    K=64, n=16, m=48, eval_epsilon=0)

    # RHS
    f = np.zeros(M)
    ii = groups["interior"]
    xi = eval_nodes[ii]
    f[ii] = -2 * np.pi**2 * np.sin(np.pi * xi[:, 0]) * np.sin(np.pi * xi[:, 1])

    u_exact = np.sin(np.pi * eval_nodes[:, 0]) * np.sin(np.pi * eval_nodes[:, 1])

    cases = [
        ('plain LSQR',                      'none',         True),
        ('LSQR + column equilibration',     'equilibrate',  False),
        ('LSQR + block-Jacobi',             'block_jacobi', False),
        ('LSQR + block-Jacobi + reorth',    'block_jacobi', True),
    ]

    for label, precond, reorth in cases:
        if rank == 0:
            print(f"\n== {label} ==")

        solve = GenIterativeSolver(comm, patches, M, n_interp,
                                   bc_scale=bc_scale, preconditioner=precond,
                                   atol=atol, btol=btol, maxiter=maxiter,
                                   reorth=reorth)
        t0 = time.time()
        local_cs, itn, rnorm = solve(f)
        elapsed = time.time() - t0

        v_local = np.concatenate(local_cs)
        U = global_interp(comm, patches, v_local, M, n_interp)

        if rank == 0:
            err_inf = np.max(np.abs(U - u_exact))
            err_l2  = np.sqrt(np.mean((U - u_exact) ** 2))
            print(f"   iters={itn}  rnorm={rnorm:.3e}  time={elapsed:.2f}s  "
                  f"max|U-u_ex|={err_inf:.3e}  L2={err_l2:.3e}")
