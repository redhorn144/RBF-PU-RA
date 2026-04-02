import numpy as np
from mpi4py import MPI

#####################################
# Non-nodal Gauss-point operators
# All four operate on the global Gauss-point index from GaussPointsAndWeights.
# Each patch stores gauss_local_idx, E_gauss, GE_gauss, w_bar_gauss,
# gw_bar_gauss (set by SetupGaussEval).
#####################################

def ApplyGaussDerivRow(comm, patches, N, M_q, k):
    """Full PU derivative of u in direction k at all M_q Gauss points. (N,)→(M_q,)"""
    def eval_k(u):
        result_local = np.zeros(M_q)
        for patch in patches:
            gidx = patch.gauss_local_idx
            if len(gidx) == 0:
                continue
            u_local = u[patch.node_indices]
            interp  = patch.E_gauss     @ u_local
            grad_k  = patch.GE_gauss[k] @ u_local
            result_local[gidx] += (patch.w_bar_gauss       * grad_k
                                   + patch.gw_bar_gauss[:, k] * interp)
        result = np.zeros(M_q)
        comm.Allreduce(result_local, result)
        return result
    return eval_k


def ApplyGaussDerivAdj(comm, patches, N, M_q, k):
    """Adjoint of ApplyGaussDerivRow. (M_q,)→(N,)"""
    def eval_k_adj(g):
        result_local = np.zeros(N)
        for patch in patches:
            gidx = patch.gauss_local_idx
            if len(gidx) == 0:
                continue
            g_local = g[gidx]
            result_local[patch.node_indices] += (
                patch.GE_gauss[k].T @ (patch.w_bar_gauss       * g_local)
              + patch.E_gauss.T    @ (patch.gw_bar_gauss[:, k] * g_local)
            )
        result = np.zeros(N)
        comm.Allreduce(result_local, result)
        return result
    return eval_k_adj


def ApplyGaussEvalRow(comm, patches, N, M_q):
    """PU interpolation at all M_q Gauss points (E_0). (N,)→(M_q,)"""
    def eval0(u):
        result_local = np.zeros(M_q)
        for patch in patches:
            gidx = patch.gauss_local_idx
            if len(gidx) == 0:
                continue
            u_local = u[patch.node_indices]
            result_local[gidx] += patch.w_bar_gauss * (patch.E_gauss @ u_local)
        result = np.zeros(M_q)
        comm.Allreduce(result_local, result)
        return result
    return eval0


def ApplyGaussEvalAdj(comm, patches, N, M_q):
    """Adjoint of E_0. (M_q,)→(N,)  Used for RHS assembly and mass matrix."""
    def eval0_adj(g):
        result_local = np.zeros(N)
        for patch in patches:
            gidx = patch.gauss_local_idx
            if len(gidx) == 0:
                continue
            g_local = g[gidx]
            result_local[patch.node_indices] += patch.E_gauss.T @ (patch.w_bar_gauss * g_local)
        result = np.zeros(N)
        comm.Allreduce(result_local, result)
        return result
    return eval0_adj


def ApplyWeakLap(comm, patches, N, gauss_wts, bc_nodes=None):
    """
    Weak Laplacian  A = Σ_l E_l^T diag(gauss_wts) E_l,  optionally with
    Dirichlet BC row replacement.  Returns a closure (N,)→(N,).
    """
    M_q = len(gauss_wts)
    d   = patches[0].nodes.shape[1] if patches else 2
    Dk  = [ApplyGaussDerivRow(comm, patches, N, M_q, k) for k in range(d)]
    DkT = [ApplyGaussDerivAdj(comm, patches, N, M_q, k) for k in range(d)]

    def weaplap(u):
        result = np.zeros(N)
        for l in range(d):
            result += DkT[l](gauss_wts * Dk[l](u))
        if bc_nodes is not None:
            result[bc_nodes] = u[bc_nodes]
        return result
    return weaplap


def ApplyGaussMassMul(comm, patches, N, gauss_wts):
    """Mass matrix  M = E_0^T diag(gauss_wts) E_0.  Returns (N,)→(N,)."""
    M_q = len(gauss_wts)
    E0  = ApplyGaussEvalRow(comm, patches, N, M_q)
    E0T = ApplyGaussEvalAdj(comm, patches, N, M_q)
    def massmul(u):
        return E0T(gauss_wts * E0(u))
    return massmul


def AssembleGaussRHS(comm, patches, N, gauss_pts, gauss_wts, f_fn):
    """
    Assemble  b = E_0^T (gauss_wts * f(gauss_pts)).

    Parameters
    ----------
    f_fn : callable (M_q, d) → (M_q,)
    """
    M_q = len(gauss_wts)
    E0T = ApplyGaussEvalAdj(comm, patches, N, M_q)
    return E0T(gauss_wts * f_fn(gauss_pts))


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
