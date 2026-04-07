import numpy as np
from mpi4py import MPI

#####################################
# Matrix free operators for the global system
# to be applied in the GMRES solver
#####################################

#####################################
# Derivative operator
#####################################
def ApplyDeriv(comm, patches, N, k, boundary_groups, BCs):
    """
    Apply the PU partial derivative operator in coordinate direction k to a vector u.

    The PU derivative at node i is given by the product rule:
        (∂/∂x_k u)(x_i) = sum_p [ w_bar_p * (D_p[k] u_p)
                                  + gw_bar_p[:, k] * (Phi_p u_p) ]

    where the sum is over patches p covering node i.

    Parameters
    ----------
    comm : MPI communicator
    patches : list of Patch objects owned by this rank
    N : int, total number of nodes
    k : int, coordinate direction index (0 for x, 1 for y, ...)
    boundary_groups : list of arrays of boundary node indices, one per BC group
    BCs : list of BC type strings, one per boundary group

    Returns
    -------
    deriv : function that takes u (N,) and returns (∂u/∂x_k) (N,)
    """
    def deriv(u):
        result_local = np.zeros(N)

        for patch in patches:
            idx = patch.node_indices
            u_local = u[idx]

            # Directional derivative: D_p[k] u_p, shape (n_local,)
            du = patch.D[k] @ u_local

            # PU assembly:
            # w_bar * (D[k] u)  +  (gw_bar[:, k]) * u
            result_local[idx] += (patch.w_bar * du
                                  + patch.gw_bar[:, k] * u_local)

        # Allreduce to sum contributions from patches on other ranks
        result = np.zeros(N)
        comm.Allreduce(result_local, result)

        # Enforce strong boundary conditions
        for group_idx, bc_nodes in enumerate(boundary_groups):
            bc_type = BCs[group_idx]

            if bc_type == 'dirichlet':
                result[bc_nodes] = u[bc_nodes]
            else:
                print(f"Warning: BC type '{bc_type}' not implemented. Defaulting to Dirichlet.")
                result[bc_nodes] = u[bc_nodes]

        return result

    return deriv

#####################################
# Laplacian operator
#####################################
def ApplyLap(comm, patches, N, boundary_groups, BCs):
    """
    Apply the PU Laplacian operator to a vector u.
    
    The PU Laplacian at node i is:
        (Lap u)(x_i) = sum_p [ w_bar_p * (L_p u_p) 
                              + 2 * gw_bar_p . (D_p u_p) 
                              + lw_bar_p * (Phi_p u_p) ]
    
    where the sum is over patches p covering node i.
    
    Parameters
    ----------
    comm : MPI communicator
    patches : list of Patch objects owned by this rank
    N : int, total number of nodes
    
    Returns
    -------
    lap : function that takes u (N,) and returns (Lap u) (N,)
    """
    def lap(u):
        result_local = np.zeros(N)

        for patch in patches:
            idx = patch.node_indices
            u_local = u[idx]

            # Gradient: D_p u_p, shape (n_local, d)
            grad = np.column_stack([D @ u_local for D in patch.D])

            # Laplacian: L_p u_p
            lap_local = patch.L @ u_local

            # PU assembly:
            # w_bar * L u  +  2 * (gw_bar . grad u)  +  lw_bar * Phi u
            result_local[idx] += (patch.w_bar * lap_local
                                  + 2.0 * np.sum(patch.gw_bar * grad, axis=1)
                                  + patch.lw_bar * u_local)

        # Allreduce to sum contributions from patches on other ranks
        result = np.zeros(N)
        comm.Allreduce(result_local, result)

        # Enforce strong boundary conditions
        for group_idx, bc_nodes in enumerate(boundary_groups):
            bc_type = BCs[group_idx]

            if bc_type == 'dirichlet':
                result[bc_nodes] = u[bc_nodes]
            else:
                print(f"Warning: BC type '{bc_type}' not implemented. Defaulting to Dirichlet.")
                result[bc_nodes] = u[bc_nodes]


        return result

    return lap