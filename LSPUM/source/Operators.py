import numpy as np
from mpi4py import MPI
from scipy.sparse import csr_matrix

def GenLap(comm, patches, M):

    d        = patches[0].D.shape[0]  if patches else 0

    def lap(local_us):
        result_local = np.zeros(M)
        for p, patch in enumerate(patches):
            #pid = patch.global_pid
            idx   = patch.eval_node_indices
            u_j = local_us[p]
            E_u   = patch.E @ u_j
            grad  = np.column_stack([patch.D[k] @ u_j for k in range(d)])
            lap_u = patch.L @ u_j

            result_local[idx] += patch.w_bar * lap_u + 2.0 * np.sum(patch.gw_bar * grad, axis=1) + patch.lw_bar * E_u

        result = np.zeros(M)
        comm.Allreduce(result_local, result, op=MPI.SUM)
        return result
    
    return lap

def GenInterp(comm, patches, M):


    def interp(local_us):
        result_local = np.zeros(M)
        for p, patch in enumerate(patches):
            idx = patch.eval_node_indices
            u_j = local_us[p]
            E_u   = patch.E @ u_j

            result_local[idx] += patch.w_bar * E_u

        result = np.zeros(M)
        comm.Allreduce(result_local, result, op=MPI.SUM)
        return result
    
    return interp

def GenLapMatrix(comm, patches, M, N_patches, n_interp, bc_indices):
    """
    Build the PUM system matrix via batched matvecs.

    For each patch j, applies both operators to the n_interp identity columns
    of that DOF block, producing n_interp matrix columns per MPI round.

    Interior rows : PUM Laplacian   (w_bar*L + 2*gw_bar·D + lw_bar*E)
    Dirichlet rows: PUM interpolant (w_bar*E)

    Returns CSR matrix on rank 0, None on other ranks.
    """
    rank = comm.Get_rank()
    nc   = N_patches * n_interp

    A_lap = np.zeros((M, nc)) if rank == 0 else None
    A_itp = np.zeros((M, nc)) if rank == 0 else None

    patch_by_pid = {p.global_pid: p for p in patches}

    for j in range(N_patches):
        c0 = j * n_interp
        c1 = c0 + n_interp

        loc_lap = np.zeros((M, n_interp))
        loc_itp = np.zeros((M, n_interp))

        if j in patch_by_pid:
            patch = patch_by_pid[j]
            idx = patch.eval_node_indices
            EU  = patch.E                                                   # (n_eval, n_interp)
            DU  = patch.D                                                   # (d, n_eval, n_interp)
            LU  = patch.L                                                   # (n_eval, n_interp)

            loc_lap[idx] = (patch.w_bar[:, None] * LU
                            + 2.0 * np.einsum('id,dij->ij', patch.gw_bar, DU)
                            + patch.lw_bar[:, None] * EU)
            loc_itp[idx] = patch.w_bar[:, None] * EU

        buf_lap = np.zeros((M, n_interp))
        buf_itp = np.zeros((M, n_interp))
        comm.Allreduce(loc_lap, buf_lap, op=MPI.SUM)
        comm.Allreduce(loc_itp, buf_itp, op=MPI.SUM)

        if rank == 0:
            A_lap[:, c0:c1] = buf_lap
            A_itp[:, c0:c1] = buf_itp

    if rank == 0:
        A_lap[bc_indices, :] = A_itp[bc_indices, :]
        return csr_matrix(A_lap)
    return None


#---------------------------------------------------------------------------
# Matrix-free LS-Poisson operator.
#
# The same system built by GenLapMatrix, but never assembled: each patch
# stores a per-patch row matrix R_p (n_eval_p, n_interp) and forward /
# adjoint apply reduce to one dense product per patch plus a single
# Allreduce in the forward direction.
#---------------------------------------------------------------------------

def GenRowMatrices(patches, bc_scale=1.0):
    """
    Per-patch row matrices R_p for the LS-PUM Poisson system.

    R_p has shape (n_eval_p, n_interp) and encodes patch p's contribution
    to every global row of A:

        interior  i :  R_p[i,:] = w_bar[i]*L[i,:]
                                + 2 gw_bar[i]·D[:,i,:]
                                + lw_bar[i]*E[i,:]
        Dirichlet i :  R_p[i,:] = bc_scale * w_bar[i] * E[i,:]

    AdjustBoundaryMatrices has already zeroed D and L at Dirichlet rows,
    so the 'interior' formula collapses to lw_bar*E there before the
    Dirichlet rows are overwritten.  Neumann rows are not yet handled.

    Parameters
    ----------
    patches  : list[Patch]
    bc_scale : float
        Weight on Dirichlet rows.  Interior (Laplacian) entries scale like
        1/r^2 while interpolant entries scale like 1, so unweighted LSQR
        ignores the boundary.  bc_scale balances the two row norms.

    Returns
    -------
    Rs : list of (n_eval_p, n_interp) arrays, one per local patch.
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


def GenMatFreeOps(comm, patches, Rs, M, n_interp):
    """
    Matrix-free forward / adjoint operators for the LS-PUM Poisson system.

    Layout
    ------
    v_local has shape (len(patches) * n_interp,); the pi-th local patch
    owns block v_local[pi*n_interp : (pi+1)*n_interp].

    Parameters
    ----------
    comm     : MPI communicator
    patches  : list[Patch]                 local patches on this rank
    Rs       : list[(n_eval_p, n_interp)]  row matrices from GenRowMatrices
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