import numpy as np
from scipy.spatial import cKDTree
from mpi4py import MPI

from .Patch import Patch
from .PatchNodes import GenPatchNodes
from .RAHelpers import PhiFactors, StableMatricesLS

def Setup(comm, eval_nodes, normals, bc_flags, centers, r, n_interp=30, node_layout='vogel', assignment='round_robin',
          K=64, n=16, m=48, eval_epsilon=0):
    
    rank = comm.Get_rank()
    size = comm.Get_size()

    patch_nodes_base = GenPatchNodes(n_interp, r, eval_nodes.shape[1], node_layout)

    if rank == 0:
        phi_lus, Er, Es    = PhiFactors(patch_nodes_base, K=K)
    else:
        phi_lus = None
        Er = None
        Es = None
    phi_lus = comm.bcast(phi_lus, root=0)
    Er = comm.bcast(Er, root=0)
    Es = comm.bcast(Es, root=0)

    if assignment == 'round_robin':
        local_patch_indices = RoundRobin(rank, size, len(patch_nodes_base))
    else:
        raise ValueError("Unknown assignment method")
    
    if rank == 0:
        tree = cKDTree(eval_nodes)
    else:
        tree = None
    
    comm.bcast(tree, root=0)

    local_patches = []
    for i in local_patch_indices:
        
        c = centers[i]
        patch_nodes = patch_nodes_base + c
        eval_idxs = np.asarray(tree.query_ball_point(c, r=r), dtype=int)

        local_eval_nodes = eval_nodes[eval_idxs]
        
        E, D, L = StableMatricesLS(
            local_eval_nodes, patch_nodes, phi_lus, Er, Es,
            n=n, m=m, eval_epsilon=eval_epsilon,
        )

        AdjustBoundaryMatrices(E, D, L, bc_flags[eval_idxs], normals[eval_idxs])

        local_patches.append(Patch(
            center       = c,
            radius       = r,
            eval_node_indices = eval_idxs,
            eval_nodes        = eval_nodes[eval_idxs],
            normals      = normals[eval_idxs],
            interp_nodes = patch_nodes,
            bc_flags     = bc_flags[eval_idxs],
            E            = E,                                     # (n_eval, n_interp)
            D            = D,                                       # (d, n_eval, n_interp)
            L            = L,                                       # (n_eval, n_interp)
            global_pid   = i,                                    # (int)
        ))

    return local_patches

def RoundRobin(rank, size, num_patches):
    return [i for i in range(num_patches) if i % size == rank]

def AdjustBoundaryMatrices(E, D, L, bc_flags, full_normals):
    for i in range(E.shape[0]):
        if bc_flags[i] == 'd':  # Dirichlet
            D[:,i,:] = 0
            L[i,:] = 0
        elif bc_flags[i] == 'n':  # Neumann
            D[:,i,:] = np.einsum('ki,k->i', D[:,i,:], full_normals[i])
            E[i,:] = 0
            L[i,:] = 0