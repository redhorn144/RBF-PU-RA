import numpy as np
from mpi4py import MPI

#####################################
# Matrix free operators for the global system
# to be applied in the Iterative solver
#####################################

#####################################
# Derivative operator
#####################################
def ApplyDeriv(comm, patches, N, k):
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

        return result

    return deriv

#####################################
# Laplacian operator
#####################################
def ApplyLap(comm, patches, N):
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

        return result

    return lap


#####################################
# Mass matrix multiply operator
#####################################
def ApplyMassMul(comm, patches, N, q):
    """
    Apply the mass matrix M to a vector u, using quadrature weights q.

    Because the PU-RBF basis is cardinal (Psi_i(x_j) = delta_ij), the mass
    matrix is diagonal:
        (M u)_i = q_i * u_i

    This is just a pointwise multiply by the quadrature weights, which makes
    the integration patch-parallel (no patch loop needed).

    Parameters
    ----------
    comm : MPI communicator
    patches : list of Patch objects owned by this rank (unused, kept for API consistency)
    N : int, total number of nodes
    q : (N,) quadrature weights

    Returns
    -------
    massmul : function that takes u (N,) and returns (M u) (N,)
    """
    def massmul(u):
        return q * u

    return massmul


#####################################
# Adjoint derivative operator
#####################################
def ApplyDerivAdj(comm, patches, N, k):
    """
    Apply the adjoint (transpose) of the PU partial derivative operator D_k^T
    to a vector g.

    The adjoint at node j is:
        (D_k^T g)_j = sum_{p containing j} [ (D_k^p)^T (w_bar_p * g_p) ]_{l(j,p)}
                                            + (grad_k w_bar_p)(x_j) * g_j

    Key differences from the forward ApplyDeriv:
    1. w_bar multiplies the INPUT g before the matrix multiply (not the output)
    2. The local matrix is TRANSPOSED: (D_k^p)^T
    3. The gw_bar term is structurally identical

    Parameters
    ----------
    comm : MPI communicator
    patches : list of Patch objects owned by this rank
    N : int, total number of nodes
    k : int, coordinate direction index (0 for x, 1 for y, ...)

    Returns
    -------
    deriv_adj : function that takes g (N,) and returns (D_k^T g) (N,)
    """
    def deriv_adj(g):
        result_local = np.zeros(N)

        for patch in patches:
            idx = patch.node_indices
            g_local = g[idx]

            # Adjoint: D_p[k]^T (w_bar * g_p)
            result_local[idx] += (patch.D[k].T @ (patch.w_bar * g_local)
                                  + patch.gw_bar[:, k] * g_local)

        # Allreduce to sum contributions from patches on other ranks
        result = np.zeros(N)
        comm.Allreduce(result_local, result)

        return result

    return deriv_adj
