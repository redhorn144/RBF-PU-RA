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