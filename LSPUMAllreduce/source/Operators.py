import numpy as np
from mpi4py import MPI


# ---------------------------------------------------------------------------
# Row-matrix constructors — one per supported PDE/operator.
# Each returns a list of (n_eval_p, n_interp) arrays, one per local patch,
# ready to be passed into GenMatFreeOps.
# ---------------------------------------------------------------------------

def PoissonRowMatrices(patches, bc_scale=1.0):
    """
    Per-patch row matrices for the LS-PUM Poisson (Laplacian) system.

        interior  i :  R_p[i,:] = w_bar[i]*L[i,:]
                                + 2 gw_bar[i]·D[:,i,:]
                                + lw_bar[i]*E[i,:]
        Dirichlet i :  R_p[i,:] = bc_scale * w_bar[i] * E[i,:]
    """
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


def AdvectionDiffusionRowMatrices(patches, a, nu=1.0, bc_scale=1.0):
    """
    Per-patch row matrices for the LS-PUM advection-diffusion system  ν·Δu − a·∇u.

    Discretises the standard transport form  du/dt + a·∇u = ν·Δu, so that a
    bump in u advects in the direction of +a.

    Applies the product rule to each PUM-weighted basis function ψ_p φ_j:

        ν·Δ(ψ_p φ_j) = ν·(w_bar·L  +  2 gw_bar·D  +  lw_bar·E)   [diffusion]
        a·∇(ψ_p φ_j) = (a·D_w)·E  +  w_bar·(a·D)                  [advection]

    where  D_w = a · gw_bar  (scalar per node) and  a·D = Σ_k a_k D[k].

        interior  i :  R_p[i,:] = ν*(w_bar*L + 2 gw_bar·D + lw_bar*E)[i,:]
                                 − w_bar[i]*(a·D)[i,:]
                                 − (a·gw_bar[i])*E[i,:]
        Dirichlet i :  R_p[i,:] = bc_scale * w_bar[i] * E[i,:]

    Parameters
    ----------
    patches  : list[Patch]
    a        : (d,) array-like   advection velocity
    nu       : float             diffusion coefficient
    bc_scale : float             Dirichlet row weight
    """
    a = np.asarray(a, dtype=float)
    Rs = []
    for p in patches:
        R = (nu * (p.w_bar[:, None] * p.L
                   + 2.0 * np.einsum('id,dij->ij', p.gw_bar, p.D)
                   + p.lw_bar[:, None] * p.E)
             - p.w_bar[:, None] * np.einsum('d,dij->ij', a, p.D)
             - np.einsum('d,id->i', a, p.gw_bar)[:, None] * p.E)
        bc_mask = (p.bc_flags == 'd')
        if bc_mask.any():
            R[bc_mask] = bc_scale * p.w_bar[bc_mask, None] * p.E[bc_mask]
        Rs.append(R)
    return Rs


def InterpolationRowMatrices(patches, bc_scale=1.0):
    """
    Per-patch row matrices for a pure PUM interpolation/projection system.

        all i :  R_p[i,:] = w_bar[i] * E[i,:]

    Useful as a standalone operator or as the boundary block in a coupled
    system where interior rows are handled by a different operator.
    """
    Rs = []
    for p in patches:
        R = p.w_bar[:, None] * p.E
        bc_mask = (p.bc_flags == 'd')
        if bc_mask.any():
            R[bc_mask] = bc_scale * p.w_bar[bc_mask, None] * p.E[bc_mask]
        Rs.append(R)
    return Rs


# ---------------------------------------------------------------------------
# Matrix-free LS-PUM operator (operator-agnostic).
#
# Takes any list of row matrices Rs produced by one of the constructors above
# and returns a (matvec, rmatvec) pair suitable for an iterative solver.
# ---------------------------------------------------------------------------

def GenMatFreeOps(comm, patches, Rs, M, n_interp):
    """
    Matrix-free forward / adjoint operators for a LS-PUM system.

    Parameters
    ----------
    comm     : MPI communicator
    patches  : list[Patch]                 local patches on this rank
    Rs       : list[(n_eval_p, n_interp)]  row matrices from a *RowMatrices fn
    M        : int                         global number of eval nodes
    n_interp : int                         DOFs per patch

    Returns
    -------
    matvec  : callable  v_local -> (M,) global, identical on every rank
    rmatvec : callable  (M,) global -> v_local, rank-local
    """
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


def assemble_dense(comm, patches, Rs, M, N_patches, n_interp):
    """
    Assemble the global dense matrix A of shape (M, N_patches * n_interp)
    from per-patch row matrices Rs, ordered by global_pid.

    Each rank writes its local patch columns into the full column range
    [global_pid * n_interp, (global_pid+1) * n_interp]; an Allreduce sums
    contributions so every rank holds the identical complete matrix.
    """
    N_col = N_patches * n_interp
    A_local = np.zeros((M, N_col))
    for p, R in zip(patches, Rs):
        c0 = p.global_pid * n_interp
        A_local[p.eval_node_indices, c0:c0 + n_interp] += R
    A = np.zeros((M, N_col))
    comm.Allreduce(A_local, A, op=MPI.SUM)
    return A
