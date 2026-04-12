import numpy as np
from mpi4py import MPI

#####################################
# Matrix-free operators for the global LS-PUM system.
#
# Global system:  A(Y, X) U(X) = F(Y)
#
#   Y : (M, d)    global evaluation nodes
#   X : (P*n, d)  union of all patch interpolation nodes  (P patches, n each)
#   U : (P*n,)    solution vector — COLUMN-DISTRIBUTED across ranks
#   F : (M,)      right-hand side — same on every rank
#
# MPI layout — FORWARD operators:
#   Each rank owns a LOCAL block of U (n_local * n_interp entries for its
#   patches).  It reads u_local[p*n:(p+1)*n] for p = 0 … n_local-1, accumulates
#   into result_local, then Allreduce gives the full M-vector on every rank.
#
# MPI layout — ADJOINT operators:
#   The input v is an M-vector already identical on all ranks (produced by a
#   prior forward Allreduce).  Each rank computes A_j^T v[idx_j] for its local
#   patches and returns a LOCAL block of size (n_local * n_interp,).
#   No Allreduce is needed — the column space is partitioned across ranks.
#
# This design pairs with the custom distributed LSQR in source/LSQR.py, which
# reduces per-iteration communication to:
#   1 × Allreduce(M)      — forward pass
#   2 × Allreduce(scalar) — ||v|| and ||x|| norms
# instead of the 1 × Allreduce(M) + 1 × Allreduce(P_global*n) that scipy's
# lsqr requires when rmatvec must return a globally-sized vector.
#####################################


# ---------------------------------------------------------------------------
# Forward operators  (u_local → M-vector via Allreduce)
# ---------------------------------------------------------------------------

def ApplyDeriv(comm, patches, M, k):
    """
    Build the forward LS-PUM directional derivative operator ∂_k.

    Parameters
    ----------
    comm    : MPI communicator
    patches : list[Patch]  local patches (w_bar / gw_bar populated).
    M       : int          total number of global evaluation nodes.
    k       : int          spatial direction index (0=x, 1=y, …).

    Returns
    -------
    deriv : callable  deriv(u_local) — (n_local*n_interp,) → (M,)
    """
    n_interp = patches[0].E.shape[1] if patches else 0

    def deriv(u_local):
        result_local = np.zeros(M)
        for p, patch in enumerate(patches):
            u_j = u_local[p * n_interp : (p + 1) * n_interp]
            idx = patch.node_indices
            du  = patch.D[k] @ u_j
            E_u = patch.E    @ u_j
            result_local[idx] += patch.w_bar * du + patch.gw_bar[:, k] * E_u
        result = np.zeros(M)
        comm.Allreduce(result_local, result, op=MPI.SUM)
        return result

    return deriv


def ApplyLap(comm, patches, M):
    """
    Build the forward LS-PUM Laplacian operator Δ.

    Interior eval nodes use the PUM product-rule Laplacian; Dirichlet boundary
    nodes use the PUM interpolation operator (w_bar * E @ u) for BC enforcement.

    Parameters
    ----------
    comm    : MPI communicator
    patches : list[Patch]  local patches (w_bar / gw_bar / lw_bar populated).
    M       : int          total number of global evaluation nodes.

    Returns
    -------
    lap : callable  lap(u_local) — (n_local*n_interp,) → (M,)
    """
    n_interp = patches[0].E.shape[1] if patches else 0
    d        = patches[0].D.shape[0]  if patches else 0

    def lap(u_local):
        result_local = np.zeros(M)
        for p, patch in enumerate(patches):
            u_j   = u_local[p * n_interp : (p + 1) * n_interp]
            idx   = patch.node_indices
            E_u   = patch.E @ u_j
            grad  = np.column_stack([patch.D[k] @ u_j for k in range(d)])
            lap_u = patch.L @ u_j

            int_mask = (patch.bc_flags == 'i')
            bnd_mask = (patch.bc_flags == 'd')

            result_local[idx[int_mask]] += (
                patch.w_bar[int_mask]  * lap_u[int_mask]
                + 2.0 * np.sum(patch.gw_bar[int_mask] * grad[int_mask], axis=1)
                + patch.lw_bar[int_mask] * E_u[int_mask]
            )
            result_local[idx[bnd_mask]] += patch.w_bar[bnd_mask] * E_u[bnd_mask]

        result = np.zeros(M)
        comm.Allreduce(result_local, result, op=MPI.SUM)
        return result

    return lap


def ApplyInterp(comm, patches, M):
    """
    Build the forward LS-PUM interpolation operator.

    Parameters
    ----------
    comm    : MPI communicator
    patches : list[Patch]  local patches (w_bar populated).
    M       : int          total number of global evaluation nodes.

    Returns
    -------
    interp : callable  interp(u_local) — (n_local*n_interp,) → (M,)
    """
    n_interp = patches[0].E.shape[1] if patches else 0

    def interp(u_local):
        result_local = np.zeros(M)
        for p, patch in enumerate(patches):
            u_j = u_local[p * n_interp : (p + 1) * n_interp]
            idx = patch.node_indices
            result_local[idx] += patch.w_bar * (patch.E @ u_j)
        result = np.zeros(M)
        comm.Allreduce(result_local, result, op=MPI.SUM)
        return result

    return interp


# ---------------------------------------------------------------------------
# Adjoint operators  (M-vector → local block, no communication)
#
# The input v is the global M-vector, identical on all ranks.
# Each rank independently computes A_j^T contributions for its local patches
# and returns a LOCAL block of size (n_local * n_interp,).
# No Allreduce is needed because the column space is rank-partitioned.
# ---------------------------------------------------------------------------

def ApplyDerivT(patches, k):
    """
    Build the adjoint of the LS-PUM directional derivative operator ∂_k.

    Returns
    -------
    derivT : callable  derivT(v) — (M,) → (n_local*n_interp,)
    """
    n_interp = patches[0].E.shape[1] if patches else 0

    def derivT(v):
        out = np.zeros(len(patches) * n_interp)
        for p, patch in enumerate(patches):
            idx   = patch.node_indices
            v_idx = v[idx]
            block = slice(p * n_interp, (p + 1) * n_interp)
            out[block] = (patch.D[k].T @ (patch.w_bar       * v_idx)
                        + patch.E.T    @ (patch.gw_bar[:, k] * v_idx))
        return out

    return derivT


def ApplyLapT(patches):
    """
    Build the adjoint of the LS-PUM Laplacian operator Δ.

    Mirrors ApplyLap: interior nodes use the PUM-Laplacian adjoint;
    Dirichlet boundary nodes use the PUM-interpolation adjoint.

    Returns
    -------
    lapT : callable  lapT(v) — (M,) → (n_local*n_interp,)
    """
    n_interp = patches[0].E.shape[1] if patches else 0
    d        = patches[0].D.shape[0]  if patches else 0

    def lapT(v):
        out = np.zeros(len(patches) * n_interp)
        for p, patch in enumerate(patches):
            idx   = patch.node_indices
            v_idx = v[idx]
            block = slice(p * n_interp, (p + 1) * n_interp)

            int_mask = (patch.bc_flags == 'i')
            bnd_mask = (patch.bc_flags == 'd')
            vi = v_idx[int_mask]
            vb = v_idx[bnd_mask]

            adj = (patch.L[int_mask].T @ (patch.w_bar[int_mask]  * vi)
                   + 2.0 * sum(patch.D[k][int_mask].T @ (patch.gw_bar[int_mask, k] * vi)
                                for k in range(d))
                   + patch.E[int_mask].T @ (patch.lw_bar[int_mask] * vi))
            adj += patch.E[bnd_mask].T @ (patch.w_bar[bnd_mask] * vb)

            out[block] = adj
        return out

    return lapT


def ApplyInterpT(patches):
    """
    Build the adjoint of the LS-PUM interpolation operator.

    Returns
    -------
    interpT : callable  interpT(v) — (M,) → (n_local*n_interp,)
    """
    n_interp = patches[0].E.shape[1] if patches else 0

    def interpT(v):
        out = np.zeros(len(patches) * n_interp)
        for p, patch in enumerate(patches):
            idx   = patch.node_indices
            block = slice(p * n_interp, (p + 1) * n_interp)
            out[block] = patch.E.T @ (patch.w_bar * v[idx])
        return out

    return interpT


def ApplyBCs(rhs, bc_flags, bc_values):
    """
    Apply boundary conditions to the right-hand side vector (in-place).

    Parameters
    ----------
    rhs       : (N,) array   right-hand side at all evaluation nodes.
    bc_flags  : (N,) array   'i' interior, 'd' Dirichlet, 'n' Neumann.
    bc_values : (N,) array   prescribed values.
    """
    for i in range(len(rhs)):
        if bc_flags[i] in ('d', 'n'):
            rhs[i] = bc_values[i]
