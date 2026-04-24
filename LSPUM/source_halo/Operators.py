import numpy as np
from mpi4py import MPI


# ---------------------------------------------------------------------------
# Row-matrix constructors (unchanged from source/)
# ---------------------------------------------------------------------------

def PoissonRowMatrices(patches, bc_scale=1.0):
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
    a = np.asarray(a, dtype=float)
    Rs = []
    for p in patches:
        R = (nu * (p.w_bar[:, None] * p.L
                   + 2.0 * np.einsum('id,dij->ij', p.gw_bar, p.D)
                   + p.lw_bar[:, None] * p.E)
             + p.w_bar[:, None] * np.einsum('d,dij->ij', a, p.D)
             + np.einsum('d,id->i', a, p.gw_bar)[:, None] * p.E)
        bc_mask = (p.bc_flags == 'd')
        if bc_mask.any():
            R[bc_mask] = bc_scale * p.w_bar[bc_mask, None] * p.E[bc_mask]
        Rs.append(R)
    return Rs


def InterpolationRowMatrices(patches, bc_scale=1.0):
    Rs = []
    for p in patches:
        R = p.w_bar[:, None] * p.E
        bc_mask = (p.bc_flags == 'd')
        if bc_mask.any():
            R[bc_mask] = bc_scale * p.w_bar[bc_mask, None] * p.E[bc_mask]
        Rs.append(R)
    return Rs


# ---------------------------------------------------------------------------
# Matrix-free LS-PUM operator — halo-exchange version.
#
# Communication per iteration
# ---------------------------
#   matvec:
#     1. Allreduce(scalar)  alpha norm   (unchanged)
#     2. point-to-point     halo contributions to neighbor-owned nodes
#        → at most 8 neighbors × (halo_nodes_per_nbr,) floats
#   rmatvec:
#     1. point-to-point     halo u values from neighbor-owned nodes
#        → symmetric with matvec halo
#     2. Allreduce(scalar)  ||x|| convergence check  (unchanged)
#
# Both matvec and rmatvec operate on distributed u vectors of size n_owned
# (each rank holds only the eval nodes in its home boxes) rather than the
# full M-vector, eliminating the O(M) Allreduce in the forward pass.
# ---------------------------------------------------------------------------

def GenMatFreeOps(patches, Rs, halo, n_interp):
    """
    Halo-exchange matrix-free forward / adjoint operators.

    Parameters
    ----------
    patches  : list[Patch]
    Rs       : list[(n_eval_p, n_interp)]  row matrices
    halo     : HaloComm
    n_interp : int

    Returns
    -------
    matvec  : callable  v_local (n_local,) -> u_owned (n_owned,)
    rmatvec : callable  u_owned (n_owned,) -> v_local (n_local,)
    """
    n_owned = len(halo.owned_indices)

    def matvec(v_local):
        u_owned   = np.zeros(n_owned)
        send_bufs = {s: np.zeros(len(halo.mv_send_gidx[s]))
                     for s in halo.neighbor_ranks}

        for pi, (p, R, ph) in enumerate(zip(patches, Rs, halo.patch_halo)):
            contrib = R @ v_local[pi * n_interp:(pi + 1) * n_interp]

            u_owned[ph.home_lidx] += contrib[ph.home_mask]

            for s in ph.nbr_ranks:
                send_bufs[s][ph.nbr_buf_idx[s]] += contrib[ph.nbr_mask[s]]

        recv = halo.mv_exchange(send_bufs, tag=50)
        for s, buf in recv.items():
            u_owned[halo.mv_recv_lidx[s]] += buf

        return u_owned

    def rmatvec(u_owned):
        # Fetch halo u values from neighbor-owned nodes
        halo_u = halo.rmv_exchange_1d(u_owned, tag=51)

        v_local = np.empty(len(patches) * n_interp)
        for pi, (p, R, ph) in enumerate(zip(patches, Rs, halo.patch_halo)):
            u_p = np.empty(len(p.eval_node_indices))

            u_p[ph.home_mask] = u_owned[ph.home_lidx]

            for s in ph.nbr_ranks:
                u_p[ph.nbr_mask[s]] = halo_u[s][ph.nbr_buf_idx[s]]

            v_local[pi * n_interp:(pi + 1) * n_interp] = R.T @ u_p

        return v_local

    return matvec, rmatvec


def assemble_dense(comm, patches, Rs, M, N_patches, n_interp):
    """Assemble the global dense matrix A (unchanged from source/)."""
    N_col   = N_patches * n_interp
    A_local = np.zeros((M, N_col))
    for p, R in zip(patches, Rs):
        c0 = p.global_pid * n_interp
        A_local[p.eval_node_indices, c0:c0 + n_interp] += R
    A = np.zeros((M, N_col))
    comm.Allreduce(A_local, A, op=MPI.SUM)
    return A
