"""
IterativeTest.py — prototype matrix-free parallel LSQR for LS-RBF-PUM Poisson.

Goal
----
Make an iterative solver actually reach the paper's error floor on this
ill-conditioned LS system by adding:

  1. matrix-free forward/adjoint operators built from per-patch R_p matrices
     (no global A assembly — one Allreduce per forward apply),
  2. boundary-row weighting (Dirichlet rows multiplied by bc_scale),
  3. column equilibration (||A[:,j]|| normalised to 1 before LSQR, rescaled
     after the solve).

If this prototype works, the new helpers can move into source/ later.
Touches no source files.
"""

from mpi4py import MPI
import numpy as np
from scipy.linalg import cholesky, solve_triangular

from nodes.SquareDomain import MinEnergySquareOne
from source.PatchTiling import LarssonBox2D
from source.LSSetup import Setup
from source.PUWeights import NormalizeWeights
from source.LSQR import lsqr as lsqr_mpi


# ---------------------------------------------------------------------------
# Per-patch row matrix
#
# R_p has shape (n_eval_p, n_interp) and encodes this patch's contribution
# to every row of the global Poisson-LS matrix A:
#
#   interior  i :  R_p[i,:] = w_bar[i]*L[i,:]  +  2 gw_bar[i]·D[:,i,:]
#                           + lw_bar[i]*E[i,:]
#   Dirichlet i :  R_p[i,:] = bc_scale * w_bar[i] * E[i,:]
#
# Once built, matvec/rmatvec/column-norms are all one-liners on R_p.
# AdjustBoundaryMatrices already zeroed D,L at Dirichlet rows so the
# interior formula collapses to lw_bar*E there; we overwrite those rows.
# Neumann not handled in this prototype.
# ---------------------------------------------------------------------------
def build_row_mats(patches, bc_scale):
    Rs = []
    for p in patches:
        R = (p.w_bar[:, None] * p.L
             + 2.0 * np.einsum('id,dij->ij', p.gw_bar, p.D)
             + p.lw_bar[:, None] * p.E)
        bc_mask = (p.bc_flags == 'd')
        if bc_mask.any():
            R[bc_mask] = bc_scale * p.w_bar[bc_mask, None] * p.E[bc_mask]
        Rs.append(R)
    return Rs


# ---------------------------------------------------------------------------
# Flat-DOF matvec / rmatvec for a rank's local patches.
#
# Layout
# ------
#   v_local has shape (len(patches) * n_interp,); the pi-th local patch
#   owns block v_local[pi*n_interp : (pi+1)*n_interp].  Global DOF
#   ordering (by patch.global_pid) is only used when gluing to/from
#   rank-0 dense solves; the iterative path never needs it.
# ---------------------------------------------------------------------------
def make_ops(comm, patches, Rs, M, n_interp):
    def matvec(v_local):
        out_local = np.zeros(M)
        for pi, (p, R) in enumerate(zip(patches, Rs)):
            c_p = v_local[pi*n_interp:(pi+1)*n_interp]
            out_local[p.eval_node_indices] += R @ c_p
        out = np.empty(M)
        comm.Allreduce(out_local, out, op=MPI.SUM)
        return out

    def rmatvec(u):
        v_local = np.empty(len(patches) * n_interp)
        for pi, (p, R) in enumerate(zip(patches, Rs)):
            y_p = u[p.eval_node_indices]
            v_local[pi*n_interp:(pi+1)*n_interp] = R.T @ y_p
        return v_local

    return matvec, rmatvec


# ---------------------------------------------------------------------------
# Column norms of A (no communication).
#
# Column (j,k) is non-zero only at rows where x_i ∈ patch j, and those rows
# live on the rank that owns patch j.  ||A[:,jk]|| is therefore purely
# patch-local: sqrt(sum over i of R_p[i,k]^2).
# ---------------------------------------------------------------------------
def column_norms(Rs, n_interp):
    out = np.empty(len(Rs) * n_interp)
    for pi, R in enumerate(Rs):
        out[pi*n_interp:(pi+1)*n_interp] = np.linalg.norm(R, axis=0)
    return out


# ---------------------------------------------------------------------------
# Block-Jacobi preconditioner, factored form.
#
# Per patch p, M_p = R_p^T R_p is the (n_interp, n_interp) diagonal block
# of A^T A.  Store its Cholesky L_p (lower) so M_p = L_p L_p^T.  Using the
# change of variable y = L^T x turns LSQR into a solve on
#   A_prec = A · P^{-1},   P = block_diag(L_p^T)
# which makes every diagonal block of A_prec^T A_prec = I.  Off-diagonal
# (patch-to-patch overlap) coupling is untouched but typically much smaller.
#
# No communication — each patch's factor and its triangular solves are
# entirely rank-local.
# ---------------------------------------------------------------------------
def build_block_jacobi(Rs, n_interp, ridge=1e-14):
    Ls = []
    Ieye = ridge * np.eye(n_interp)
    for R in Rs:
        Mp = R.T @ R + Ieye
        Ls.append(cholesky(Mp, lower=True))
    return Ls


def apply_Linv_T(v_local, Ls, n_interp):
    """Block-diagonal L_p^{-T} — applied inside matvec (= P^{-1})."""
    out = np.empty_like(v_local)
    for pi, L in enumerate(Ls):
        out[pi*n_interp:(pi+1)*n_interp] = solve_triangular(
            L, v_local[pi*n_interp:(pi+1)*n_interp], lower=True, trans=1)
    return out


def apply_Linv(v_local, Ls, n_interp):
    """Block-diagonal L_p^{-1} — applied after rmatvec (= P^{-T})."""
    out = np.empty_like(v_local)
    for pi, L in enumerate(Ls):
        out[pi*n_interp:(pi+1)*n_interp] = solve_triangular(
            L, v_local[pi*n_interp:(pi+1)*n_interp], lower=True)
    return out


# ---------------------------------------------------------------------------
# PUM interpolant for post-solve evaluation.
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
    M        = 2000
    n_interp = 40
    H        = 0.2
    delta    = 0.2
    bc_scale = 100.0

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
    NormalizeWeights(comm, patches, M)

    Rs = build_row_mats(patches, bc_scale)
    matvec, rmatvec = make_ops(comm, patches, Rs, M, n_interp)

    # RHS
    f = np.zeros(M)
    ii = groups["interior"]
    xi = eval_nodes[ii]
    f[ii] = -2 * np.pi**2 * np.sin(np.pi * xi[:, 0]) * np.sin(np.pi * xi[:, 1])
    # boundary rows: f=0 (sin vanishes on the unit-square boundary);
    # scaling them by bc_scale does not change 0.

    u_exact = np.sin(np.pi * eval_nodes[:, 0]) * np.sin(np.pi * eval_nodes[:, 1])

    maxiter = 20000
    atol = btol = 1e-10

    # ---- 1. plain LSQR (expected to stall) ----
    if rank == 0:
        print("\n== plain LSQR ==")
    t0 = time.time()
    x, itn, rnorm = lsqr_mpi(comm, matvec, rmatvec, f,
                             atol=atol, btol=btol, maxiter=maxiter, show=False)
    t_plain = time.time() - t0
    U = global_interp(comm, patches, x, M, n_interp)
    if rank == 0:
        err_inf = np.max(np.abs(U - u_exact))
        err_l2  = np.sqrt(np.mean((U - u_exact) ** 2))
        print(f"   iters={itn}  rnorm={rnorm:.3e}  time={t_plain:.2f}s  "
              f"max|U-u_ex|={err_inf:.3e}  L2={err_l2:.3e}")

    # ---- 2. LSQR with column equilibration ----
    if rank == 0:
        print("\n== LSQR + column equilibration ==")
    d_local = column_norms(Rs, n_interp)
    d_safe  = np.where(d_local > 0, d_local, 1.0)
    inv_d   = 1.0 / d_safe

    def mv_eq(v):   return matvec(inv_d * v)           # A @ diag(1/d) @ v
    def rm_eq(u):   return inv_d * rmatvec(u)          # diag(1/d) @ A^T @ u

    t0 = time.time()
    y, itn, rnorm = lsqr_mpi(comm, mv_eq, rm_eq, f,
                             atol=atol, btol=btol, maxiter=maxiter, show=False)
    t_eq = time.time() - t0
    x_eq = inv_d * y
    U_eq = global_interp(comm, patches, x_eq, M, n_interp)
    if rank == 0:
        err_inf = np.max(np.abs(U_eq - u_exact))
        err_l2  = np.sqrt(np.mean((U_eq - u_exact) ** 2))
        print(f"   iters={itn}  rnorm={rnorm:.3e}  time={t_eq:.2f}s  "
              f"max|U-u_ex|={err_inf:.3e}  L2={err_l2:.3e}")

    # ---- 3. LSQR + block-Jacobi preconditioner ----
    if rank == 0:
        print("\n== LSQR + block-Jacobi preconditioner ==")
    Ls = build_block_jacobi(Rs, n_interp)

    def mv_bj(y):   return matvec(apply_Linv_T(y, Ls, n_interp))
    def rm_bj(u):   return apply_Linv(rmatvec(u), Ls, n_interp)

    t0 = time.time()
    y, itn, rnorm = lsqr_mpi(comm, mv_bj, rm_bj, f,
                             atol=atol, btol=btol, maxiter=maxiter, show=False)
    t_bj = time.time() - t0
    x_bj = apply_Linv_T(y, Ls, n_interp)   # recover x from y
    U_bj = global_interp(comm, patches, x_bj, M, n_interp)
    if rank == 0:
        err_inf = np.max(np.abs(U_bj - u_exact))
        err_l2  = np.sqrt(np.mean((U_bj - u_exact) ** 2))
        print(f"   iters={itn}  rnorm={rnorm:.3e}  time={t_bj:.2f}s  "
              f"max|U-u_ex|={err_inf:.3e}  L2={err_l2:.3e}")

