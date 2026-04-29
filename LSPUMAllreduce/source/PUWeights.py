import numpy as np
from mpi4py import MPI


def NormalizeWeights(comm, patches, N):
    """
    Compute and store normalised Wendland C2 partition-of-unity weights on every
    local patch, accounting for contributions from all ranks.

    Algorithm (two-pass with one Allreduce barrier):

      Pass 1 — accumulate raw (unnormalised) weights into rank-local arrays and
               cache them to avoid recomputation.

      Allreduce — sum the three local arrays (W, ∇W, ΔW) across all ranks so
                  every rank holds the global denominators.

      Pass 2 — divide each patch's cached raw weights by the global denominators
               using the quotient-rule expressions for ∇ψ and Δψ, and store the
               results in patch.w_bar / gw_bar / lw_bar.

    The normalised weight satisfies  Σ_j ψ_j(x) = 1  at every evaluation node,
    together with its gradient and Laplacian identities required by the PUM
    operator assembly.

    Parameters
    ----------
    comm    : MPI communicator
    patches : list[Patch]  local patches owned by this rank (from SetupPatches)
    N       : int          total number of evaluation nodes in the global system

    Returns
    -------
    W : (N,) array
        Global weight sum (should be ≥ 1 everywhere; useful for diagnostics).
    """
    d = patches[0].interp_nodes.shape[1] if patches else 0
    d = comm.allreduce(d, op=MPI.MAX)

    W_local     = np.zeros(N)
    gradW_local = np.zeros((N, d))
    lapW_local  = np.zeros(N)

    # Pass 1: accumulate raw weights; cache to avoid recomputing after Allreduce
    cache = []
    for patch in patches:
        idx = patch.eval_node_indices
        w   = C2Weight(patch.eval_nodes, patch.center, patch.radius)
        gw  = C2WeightGradient(patch.eval_nodes, patch.center, patch.radius)
        lw  = C2WeightLaplacian(patch.eval_nodes, patch.center, patch.radius)
        cache.append((w, gw, lw))

        W_local[idx]     += w
        gradW_local[idx] += gw
        lapW_local[idx]  += lw

    # Allreduce: every rank receives the global sums
    W     = np.zeros(N)
    gradW = np.zeros((N, d))
    lapW  = np.zeros(N)
    comm.Allreduce(W_local,     W,     op=MPI.SUM)
    comm.Allreduce(gradW_local, gradW, op=MPI.SUM)
    comm.Allreduce(lapW_local,  lapW,  op=MPI.SUM)

    # Pass 2: normalise using global denominators
    # Let ψ = w / W.  Quotient rule gives:
    #   ∇ψ  = ∇w/W  -  w ∇W / W²
    #   Δψ  = Δw/W  -  2 ∇w·∇W / W²  -  w ΔW / W²  +  2 w |∇W|² / W³
    for patch, (w, gw, lw) in zip(patches, cache):
        idx = patch.eval_node_indices
        Wn  = W[idx]            # (n_eval,)
        gWn = gradW[idx]        # (n_eval, d)
        lWn = lapW[idx]         # (n_eval,)

        W2 = Wn ** 2
        W3 = Wn * W2

        patch.w_bar  = w / Wn

        patch.gw_bar = gw / Wn[:, None] - w[:, None] * gWn / W2[:, None]

        patch.lw_bar = (lw / Wn
                        - 2.0 * np.einsum('ij,ij->i', gw, gWn) / W2
                        - w * lWn / W2
                        + 2.0 * w * np.einsum('ij,ij->i', gWn, gWn) / W3)

    return W


######################################
# Vectorized Wendland C2 weight functions
# x: (n, d) array of node positions
# center: (d,) patch center
# radius: scalar patch radius
######################################

def C2Weight(x, center, radius):
    """Returns (n,) array of weights."""
    rho = np.linalg.norm(x - center, axis=1) / radius
    return (1.0 - rho) ** 4 * (4.0 * rho + 1.0)


def C2WeightGradient(x, center, radius):
    """Returns (n, d) array of weight gradients."""
    diff = x - center
    r    = np.linalg.norm(diff, axis=1)
    rho  = r / radius

    # Scalar factor d(w)/dr * (1/r); zero at the centre
    safe_r = np.where(r == 0.0, 1.0, r)
    factor = -20.0 * (1.0 - rho) ** 3 * rho / (radius * safe_r)
    factor = np.where(r == 0.0, 0.0, factor)

    return factor[:, None] * diff


def C2WeightLaplacian(x, center, radius):
    """Returns (n,) array of weight Laplacians."""
    d    = x.shape[1]
    r    = np.linalg.norm(x - center, axis=1)
    rho  = r / radius

    safe_rho = np.where(rho == 0.0, 1.0, rho)
    psi_d  = -20.0 * (1.0 - rho) ** 3 * rho
    psi_dd =  20.0 * (1.0 - rho) ** 2 * (4.0 * rho - 1.0)

    result = (psi_dd + (d - 1.0) * psi_d / safe_rho) / radius ** 2

    # Exact limit at the centre: Δw(0) = -20d / radius²
    return np.where(r == 0.0, -20.0 * d / radius ** 2, result)
