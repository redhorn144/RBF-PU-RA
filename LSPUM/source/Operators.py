import numpy as np
from mpi4py import MPI

def GenLap(comm, patches, M):

    n_interp = patches[0].E.shape[1] if patches else 0
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

            result_local[idx] = patch.w_bar * lap_u + 2.0 * np.sum(patch.gw_bar * grad, axis=1) + patch.lw_bar * E_u

        result = np.zeros(M)
        comm.Allreduce(result_local, result, op=MPI.SUM)
        return result