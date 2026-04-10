import numpy as np
from mpi4py import MPI

#####################################
# Matrix-free operators for the global LS-PUM system.
#
# Global system:  A(Y, X) U(X) = F(Y)
#
#   Y : (M, d)  global evaluation nodes
#   X : (N, d)  union of all patch interpolation nodes,  N = P * n_interp
#   U : (N,)    function values at the patch interpolation nodes
#   F : (M,)    right-hand side at the evaluation nodes
#
# Each patch j contributes n_interp columns and n_eval_j rows.  The local
# sub-matrix for the Laplacian operator is:
#
#   A_j = diag(w_bar_j) @ L_j
#       + 2 Σ_k diag(gw_bar_j[:,k]) @ D_j[k]
#       + diag(lw_bar_j) @ E_j          shape (n_eval_j, n_interp)
#
# where E_j = phi_eval @ Phi_n^{-1} maps function values at interp nodes to
# function values at eval nodes (the Phi_n^{-1} solve is pre-baked into E).
#
# MPI layout — FORWARD operators:
#   Each rank owns a subset of patches.  u_local is the stacked function-
#   value vector [u_1; ...; u_P_local], one n_interp block per local patch.
#   Each rank accumulates into result_local, then Allreduce gives the full
#   M-vector on every rank.
#
# MPI layout — ADJOINT operators:
#   The input v is an M-vector that is already global (same on all ranks,
#   e.g. from a prior Allreduce).  Each rank reads v[idx] for its patches
#   and writes into its own output block.  No Allreduce is needed.
#####################################


# ---------------------------------------------------------------------------
# Forward operators
# ---------------------------------------------------------------------------

def ApplyDeriv(comm, patches, M, k):
    """
    Build the forward LS-PUM directional derivative operator ∂_k.

    For patch j at eval node y_i (PUM product rule):

        result[idx[i]] += w_bar_j[i]    * (D_j[k] @ u_j)[i]
                        + gw_bar_j[i,k] * (E_j     @ u_j)[i]

    Parameters
    ----------
    comm    : MPI communicator
    patches : list[Patch]  local patches with w_bar / gw_bar populated.
    M       : int          total number of global evaluation nodes.
    k       : int          spatial direction index (0=x, 1=y, …).

    Returns
    -------
    deriv : callable
        deriv(u_local) — u_local is (P_local*n_interp,); returns (M,).
    """
    n_interp = patches[0].E.shape[1] if patches else 0

    def deriv(u_local):
        result_local = np.zeros(M)
        for p, patch in enumerate(patches):
            u_j = u_local[p * n_interp : (p + 1) * n_interp]
            idx = patch.node_indices
            du  = patch.D[k] @ u_j      # (n_eval,)
            E_u = patch.E    @ u_j      # (n_eval,)
            result_local[idx] += patch.w_bar * du + patch.gw_bar[:, k] * E_u
        result = np.zeros(M)
        comm.Allreduce(result_local, result, op=MPI.SUM)
        return result

    return deriv


def ApplyLap(comm, patches, M):
    """
    Build the forward LS-PUM Laplacian operator Δ.

    For patch j at eval node y_i (PUM product rule):

        result[idx[i]] += w_bar_j[i]  * (L_j   @ u_j)[i]
                        + 2 gw_bar_j[i] · (D_j @ u_j)[:,i]
                        + lw_bar_j[i]  * (E_j   @ u_j)[i]

    Parameters
    ----------
    comm    : MPI communicator
    patches : list[Patch]  local patches with w_bar / gw_bar / lw_bar populated.
    M       : int          total number of global evaluation nodes.

    Returns
    -------
    lap : callable
        lap(u_local) — u_local is (P_local*n_interp,); returns (M,).
    """
    n_interp = patches[0].E.shape[1] if patches else 0
    d        = patches[0].D.shape[0]  if patches else 0

    def lap(u_local):
        result_local = np.zeros(M)
        for p, patch in enumerate(patches):
            u_j   = u_local[p * n_interp : (p + 1) * n_interp]
            idx   = patch.node_indices
            E_u   = patch.E @ u_j                                    # (n_eval,)
            grad  = np.column_stack([patch.D[k] @ u_j                # (n_eval, d)
                                     for k in range(d)])
            lap_u = patch.L @ u_j                                    # (n_eval,)
            result_local[idx] += (patch.w_bar  * lap_u
                                  + 2.0 * np.sum(patch.gw_bar * grad, axis=1)
                                  + patch.lw_bar * E_u)
        result = np.zeros(M)
        comm.Allreduce(result_local, result, op=MPI.SUM)
        return result

    return lap


def ApplyInterp(comm, patches, M):
    """
    Build the forward LS-PUM interpolation operator.

    Maps U(X) (function values at patch interp nodes) to the M eval nodes:

        result[idx[i]] += w_bar_j[i] * (E_j @ u_j)[i]

    Parameters
    ----------
    comm    : MPI communicator
    patches : list[Patch]  local patches with w_bar populated.
    M       : int          total number of global evaluation nodes.

    Returns
    -------
    interp : callable
        interp(u_local) — u_local is (P_local*n_interp,); returns (M,).
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
# Adjoint operators
#
# For the sub-matrix  A_j  of patch j (shape n_eval_j × n_interp), the
# adjoint A_j^T maps a weighted residual vector at the eval nodes back into
# the n_interp-block owned by patch j.
#
# The input v (M,) is the full global vector — already the same on all ranks
# (e.g. from a prior Allreduce in the forward pass or from LSMR/LSQR).
# Each rank independently computes its own output block; no Allreduce needed.
# ---------------------------------------------------------------------------

def ApplyDerivT(patches, k):
    """
    Build the adjoint of the LS-PUM directional derivative operator ∂_k.

    A_j = diag(w_bar_j) @ D_j[k]  +  diag(gw_bar_j[:,k]) @ E_j

    so the adjoint contribution from patch j is:

        out[j-block] = D_j[k]^T @ (w_bar_j    * v[idx])
                     + E_j^T    @ (gw_bar_j[:,k] * v[idx])

    Parameters
    ----------
    patches  : list[Patch]  local patches with w_bar / gw_bar populated.
    k        : int          spatial direction index (0=x, 1=y, …).

    Returns
    -------
    derivT : callable
        derivT(v) — v is the global (M,) vector; returns (P_local*n_interp,).
    """
    n_interp = patches[0].E.shape[1] if patches else 0

    def derivT(v):
        out = np.zeros(len(patches) * n_interp)
        for p, patch in enumerate(patches):
            idx    = patch.node_indices
            v_idx  = v[idx]                              # (n_eval,)
            block  = slice(p * n_interp, (p + 1) * n_interp)
            out[block] = (patch.D[k].T @ (patch.w_bar      * v_idx)
                        + patch.E.T    @ (patch.gw_bar[:, k] * v_idx))
        return out

    return derivT


def ApplyLapT(patches):
    """
    Build the adjoint of the LS-PUM Laplacian operator Δ.

    A_j = diag(w_bar_j) @ L_j
        + 2 Σ_k diag(gw_bar_j[:,k]) @ D_j[k]
        + diag(lw_bar_j) @ E_j

    so the adjoint contribution from patch j is:

        out[j-block] = L_j^T @ (w_bar_j  * v[idx])
                     + 2 Σ_k D_j[k]^T @ (gw_bar_j[:,k] * v[idx])
                     + E_j^T @ (lw_bar_j * v[idx])

    Parameters
    ----------
    patches  : list[Patch]  local patches with w_bar / gw_bar / lw_bar populated.

    Returns
    -------
    lapT : callable
        lapT(v) — v is the global (M,) vector; returns (P_local*n_interp,).
    """
    n_interp = patches[0].E.shape[1] if patches else 0
    d        = patches[0].D.shape[0]  if patches else 0

    def lapT(v):
        out = np.zeros(len(patches) * n_interp)
        for p, patch in enumerate(patches):
            idx    = patch.node_indices
            v_idx  = v[idx]                              # (n_eval,)
            block  = slice(p * n_interp, (p + 1) * n_interp)
            out[block] = (patch.L.T @ (patch.w_bar  * v_idx)
                        + 2.0 * sum(patch.D[k].T @ (patch.gw_bar[:, k] * v_idx)
                                    for k in range(d))
                        + patch.E.T @ (patch.lw_bar * v_idx))
        return out

    return lapT


def ApplyInterpT(patches):
    """
    Build the adjoint of the LS-PUM interpolation operator.

    A_j = diag(w_bar_j) @ E_j

    so the adjoint contribution from patch j is:

        out[j-block] = E_j^T @ (w_bar_j * v[idx])

    Parameters
    ----------
    patches  : list[Patch]  local patches with w_bar populated.

    Returns
    -------
    interpT : callable
        interpT(v) — v is the global (M,) vector; returns (P_local*n_interp,).
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
