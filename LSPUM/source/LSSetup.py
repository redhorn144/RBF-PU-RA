import numpy as np
from scipy.spatial import cKDTree
from mpi4py import MPI

from .Patch import Patch
from .PatchNodes import GenPatchNodes
from .RAHelpers import PhiFactors, StableMatricesLS

#------------------------------------------------------------------------------------
# LS-RBF-PUM patch setup, distributed over MPI ranks.
#
# Workflow:
#   1. Rank 0 generates patch centres via GenPatchTiling (or caller supplies them).
#   2. All ranks receive centres and radius via broadcast.
#   3. Patches are assigned round-robin (rank i owns patches i, i+size, i+2*size, …).
#   4. Each rank independently builds its Patch objects — no communication required.
#   5. PU weights are filled separately by NormalizeWeights (requires Allreduce).
#
# Why PhiFactors is computed once:
#   All patches use the same prototype node geometry (Vogel/GLL points, centred at
#   the origin, scaled by r).  Each patch's interp nodes are c + proto_nodes.
#   Because GenPhi depends only on pairwise *differences*, the square matrix
#   Φ(c + proto, c + proto) = Φ(proto, proto) for any centre c.
#   Eval points are shifted to the local frame (eval_pts - c) before calling
#   StableMatricesLS, so the same LU factors apply across all patches.
#------------------------------------------------------------------------------------

def SetupPatches(comm, eval_interior, eval_boundary, normals, bc_values,
                 centers, r, n_interp=30, node_layout='vogel',
                 K=64, n=16, m=48, eval_epsilon=0, strict=True):
    """
    Build the local list of Patch objects for this MPI rank using RA-stable matrices.

    Parameters
    ----------
    comm          : MPI communicator
    eval_interior : (N_i, d)   interior evaluation nodes
    eval_boundary : (N_b, d)   boundary evaluation nodes
    normals       : (N_b, d)   outward unit normals at boundary nodes
    bc_values     : (N_b,)     Dirichlet BC values at boundary nodes
    centers       : (M, d)     patch centres from GenPatchTiling (same on all ranks)
    r             : float      patch radius (same on all ranks)
    n_interp      : int        number of RBF interpolation nodes per patch
    node_layout   : str        interpolation node layout ('vogel' or 'polar_gll')
    K             : int        contour points for the RA method (passed to PhiFactors)
    n, m          : int        denominator / numerator degrees for the rational approximant
    eval_epsilon  : float      shape parameter to evaluate at; 0 = flat-limit
    strict        : bool       if True (default), raise if any patch has fewer than
                               n_interp eval nodes (underdetermined local system).
                               Use GenPatchTiling(..., min_nodes=n_interp) to prevent
                               this at the tiling stage.

    Returns
    -------
    local_patches : list[Patch]
        Patch objects owned by this rank.  PU weight fields (w_bar, gw_bar, lw_bar)
        are left as None and must be filled by NormalizeWeights.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    n_int = len(eval_interior)
    d     = eval_interior.shape[1]

    # Stack all eval nodes; boundary nodes sit after interior nodes
    all_nodes = np.vstack([eval_interior, eval_boundary])          # (N, d)

    # Node-level boundary mask and normal/BC arrays aligned to all_nodes
    is_bnd       = np.zeros(len(all_nodes), dtype=bool)
    is_bnd[n_int:] = True
    full_normals = np.vstack([np.zeros((n_int, d)), normals])      # (N, d)
    full_bc      = np.concatenate([np.full(n_int, np.nan), bc_values])  # (N,)

    # Prototype interpolation nodes (origin-centred, scaled by r).
    # PhiFactors pre-computes LU factorisations of Φ(proto, proto) at every
    # contour point — done once and reused for every patch.
    proto_nodes        = GenPatchNodes(n_interp, r, d, layout=node_layout)
    phi_lus, Er, Es    = PhiFactors(proto_nodes, K=K)

    # KD-tree over all eval nodes for radius queries
    tree = cKDTree(all_nodes)

    # Round-robin patch assignment
    local_patches = []
    for pid in range(rank, len(centers), size):
        c   = centers[pid]
        idx = np.asarray(tree.query_ball_point(c, r=r), dtype=int)
        if len(idx) < n_interp:
            if strict and len(idx) > 0:
                raise ValueError(
                    f"Patch {pid} has {len(idx)} eval nodes but n_interp={n_interp}. "
                    "Pass min_nodes=n_interp to GenPatchTiling to filter at the tiling stage."
                )
            continue

        # Shift eval points to the local frame of proto_nodes (centred at origin).
        # This is equivalent to using (c + proto_nodes) as interp nodes with absolute
        # eval coords, but lets us reuse the precomputed LU factors directly.
        eval_pts_local = all_nodes[idx] - c                        # (n_eval, d)

        E, D, L = StableMatricesLS(
            eval_pts_local, proto_nodes, phi_lus, Er, Es,
            n=n, m=m, eval_epsilon=eval_epsilon,
        )

        local_patches.append(Patch(
            center       = c,
            radius       = r,
            node_indices = idx,
            nodes        = all_nodes[idx],
            normals      = full_normals[idx],
            interp_nodes = c + proto_nodes,
            is_boundary  = is_bnd[idx],
            bc_values    = full_bc[idx],
            E            = E,                                     # (n_eval, n_interp)
            D            = D,                                       # (d, n_eval, n_interp)
            L            = L,                                       # (n_eval, n_interp)
        ))

    return local_patches


def PatchNodeOrdering(N, patches):
    """
    Greedy node reordering for improved diagonal dominance in iterative solvers.

    Iterates through patches in order; the first time a node is encountered it
    is assigned the next available position.  This clusters nodes that share a
    patch contiguously, so the assembled system matrix has denser diagonal blocks.

    Complexity: O(sum of patch sizes) = O(N * avg_overlap) — optimal since every
    patch membership must be read at least once.

    Parameters
    ----------
    N       : int          total number of nodes in the global system
    patches : list[Patch]  patches in the desired ordering (each has .node_indices)

    Returns
    -------
    perm     : (N,) int array
        perm[i] = original index placed at new position i.
        Reorder arrays as  x_new = x[perm].
    inv_perm : (N,) int array
        inv_perm[j] = new position of original node j.
        Reorder system rows/columns as  A_new = A[np.ix_(inv_perm, inv_perm)].
    """
    placed = np.zeros(N, dtype=bool)
    chunks = []

    for patch in patches:
        idx  = patch.node_indices
        new  = idx[~placed[idx]]          # nodes in this patch not yet placed
        if len(new):
            chunks.append(new)
            placed[new] = True

    perm             = np.concatenate(chunks)
    inv_perm         = np.empty(N, dtype=np.intp)
    inv_perm[perm]   = np.arange(N, dtype=np.intp)

    return perm, inv_perm


def ApplyNodeOrdering(patches, inv_perm):
    """
    Update every patch's node_indices to reflect a global node reordering.

    After reordering, the global node array satisfies  x_new = x_old[perm].
    Each old index j maps to new index inv_perm[j].  Applying inv_perm to
    node_indices makes each patch point at the correct rows of x_new (and of
    any global vector in the reordered system) without touching the patch-local
    arrays (nodes, normals, Phi, D, L, …), which stay in their current order.

    Call this once, on every rank's local patch list, immediately after
    PatchNodeOrdering and before assembling or solving.

    Parameters
    ----------
    patches  : list[Patch]   local patches owned by this rank
    inv_perm : (N,) int array  inv_perm[old_idx] = new_idx, from PatchNodeOrdering
    """
    for patch in patches:
        patch.node_indices = inv_perm[patch.node_indices]
