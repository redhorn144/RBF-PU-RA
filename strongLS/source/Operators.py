import numpy as np
from mpi4py import MPI

#####################################
# Oversampled operators for least-squares strong-form collocation.
# ApplyLapOS  : ℝᴺ → ℝᴹ  (forward, M collocation points)
# ApplyLapOST : ℝᴹ → ℝᴺ  (adjoint/transpose)
#
# Compose as  ApplyLapNormal = ApplyLapOST ∘ ApplyLapOS  to get ℝᴺ → ℝᴺ
# and solve  (Aᵀ A) u = Aᵀ f  with the existing GMRES.
#####################################

def ApplyLapOS(comm, patches, N, M, eval_boundary_indices, solution_boundary_groups, BCs):
    """
    Apply the oversampled PU Laplacian operator.

    Maps u ∈ ℝᴺ (solution nodes) → v ∈ ℝᴹ (collocation/eval points).

    PU Laplacian at eval point yᵢ:
        v[i] = Σ_p [ w̄_p(yᵢ) * (L_p u_p)
                     + 2 * ∇w̄_p(yᵢ) · (D_p u_p)
                     + Δw̄_p(yᵢ) * u_p ]
    where L_p, D_p are the rectangular (m×n) patch matrices.

    At eval points that coincide with Dirichlet boundary nodes, the row is
    replaced by the identity: v[i] = u[j] for the matching solution node j.

    Parameters
    ----------
    comm                     : MPI communicator
    patches                  : list of Patch objects owned by this rank
    N                        : total number of solution nodes
    M                        : total number of eval/collocation points
    eval_boundary_indices    : array of indices into eval_pts that are boundary pts
    solution_boundary_groups : list of arrays of boundary node indices (solution space)
    BCs                      : list of BC type strings
    """
    def op(u):
        result_local = np.zeros(M)

        for patch in patches:
            sol_idx  = patch.node_indices    # (n,) indices into u
            eval_idx = patch.eval_indices    # (m,) indices into result
            u_local  = u[sol_idx]

            # PU Laplacian at eval points (product rule with rectangular matrices)
            # D_eval is (d, m, n); matmul broadcasts to (d, m), then .T → (m, d)
            grad = (patch.D_eval @ u_local).T                             # (m, d)
            lap_local = patch.L_eval @ u_local                            # (m,)

            result_local[eval_idx] += (patch.w_bar_eval * lap_local
                                       + 2.0 * np.sum(patch.gw_bar_eval * grad, axis=1)
                                       + patch.lw_bar_eval * (patch.Phi_eval @ u_local))

        result = np.zeros(M)
        comm.Allreduce(result_local, result)

        # Enforce Dirichlet BCs at eval points that are boundary solution nodes.
        # eval_boundary_indices[g] maps eval-space boundary rows;
        # solution_boundary_groups[g] maps the corresponding solution-space values.
        for g, (eval_bnd, sol_bnd) in enumerate(zip(eval_boundary_indices, solution_boundary_groups)):
            if BCs[g] == 'dirichlet':
                result[eval_bnd] = u[sol_bnd]
            else:
                result[eval_bnd] = u[sol_bnd]

        return result

    return op


def ApplyLapOSTranspose(comm, patches, N, M, eval_boundary_indices, solution_boundary_groups, BCs):
    """
    Apply the adjoint (transpose) of ApplyLapOS.

    Maps v ∈ ℝᴹ → w ∈ ℝᴺ.

    The transpose of the PU Laplacian accumulates at solution node j:
        w[j] = Σ_p Σᵢ∈eval_p [ v[i] * (L_p^T)_{j,i} * w̄_p(yᵢ) + ... ]

    This is exactly the adjoint of the forward operator, implemented by
    reversing the matrix–vector products.

    Parameters
    ----------
    (same as ApplyLapOS)
    """
    def op_T(v):
        result_local = np.zeros(N)

        for patch in patches:
            sol_idx  = patch.node_indices
            eval_idx = patch.eval_indices
            v_local  = v[eval_idx]             # (m,) residual at eval pts

            # Transpose of: w̄ * L u   →  L^T (w̄ * v)
            result_local[sol_idx] += patch.L_eval.T @ (patch.w_bar_eval * v_local)

            # Transpose of: 2 * ∇w̄ · (D u)   →  Σ_k D_k^T (2 * gw̄_k * v)
            # D_eval (d,m,n), gw_bar_eval (m,d), v_local (m,)
            # second operand is (m,d) so indices are 'md', not 'dm'
            result_local[sol_idx] += np.einsum(
                'dmn,md->n', patch.D_eval, 2.0 * patch.gw_bar_eval * v_local[:, None]
            )

            # Transpose of: Δw̄ * (Φ u)   →  Φ^T (Δw̄ * v)
            result_local[sol_idx] += patch.Phi_eval.T @ (patch.lw_bar_eval * v_local)

        result = np.zeros(N)
        comm.Allreduce(result_local, result)

        # Boundary rows of the adjoint: Aᵀ identity rows = identity columns
        for g, (eval_bnd, sol_bnd) in enumerate(zip(eval_boundary_indices, solution_boundary_groups)):
            if BCs[g] == 'dirichlet':
                result[sol_bnd] = v[eval_bnd]
            else:
                result[sol_bnd] = v[eval_bnd]

        return result

    return op_T

def ApplyInterpOS(comm, patches, N, M):
    """
    Oversampled PU interpolation operator E: ℝᴺ → ℝᴹ.

    Maps nodal values u to the PU approximant at M eval points:
        (E u)(y_i) = Σ_p w̄_p(y_i) * (Φ_eval_p · u_p)

    This is the "mass" side of the overdetermined semidiscrete system
        E · du/dt = -D · u
    whose ℓ₂ solution gives du/dt = -(EᵀE)⁻¹ Eᵀ D · u  (Tominec et al. 2024).
    """
    def op(u):
        result_local = np.zeros(M)
        for patch in patches:
            u_local = u[patch.node_indices]
            result_local[patch.eval_indices] += patch.w_bar_eval * (patch.Phi_eval @ u_local)
        result = np.zeros(M)
        comm.Allreduce(result_local, result)
        return result
    return op


def ApplyDerivOS(comm, patches, N, M, k):
    """
    Oversampled PU partial derivative in direction k.

    Maps u ∈ ℝᴺ → v ∈ ℝᴹ (eval points), no BCs enforced.

    (∂u/∂x_k)(y_i) = Σ_p [ w̄_p(y_i) * (D_eval_p[k] u_p)
                            + gw̄_p[k](y_i) * (Φ_eval_p u_p) ]
    """
    def op(u):
        result_local = np.zeros(M)
        for patch in patches:
            sol_idx  = patch.node_indices
            eval_idx = patch.eval_indices
            u_local  = u[sol_idx]
            result_local[eval_idx] += (patch.w_bar_eval * (patch.D_eval[k] @ u_local)
                                       + patch.gw_bar_eval[:, k] * (patch.Phi_eval @ u_local))
        result = np.zeros(M)
        comm.Allreduce(result_local, result)
        return result
    return op


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