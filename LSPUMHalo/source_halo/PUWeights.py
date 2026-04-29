import numpy as np


def NormalizeWeights(patches, halo):
    """
    Compute and store normalised Wendland C2 PU weights using halo exchange.

    Algorithm (two-pass):

      Pass 1  — each patch accumulates raw (w, ∇w, Δw) into:
                  • owned nodes   → W_owned / gradW_owned / lapW_owned   (local)
                  • halo nodes    → send buffers for the owner rank

      Forward exchange  — send contributions to owner ranks; receive
                          contributions from neighbors → complete sums at
                          all owned eval nodes.

      Reverse exchange  — owner ranks broadcast (W, ∇W, ΔW) back to all
                          ranks whose patches touch those nodes.

      Pass 2  — each patch assembles the denominator arrays from owned and
                received halo values, then applies the quotient-rule to get
                w_bar / gw_bar / lw_bar.

    Parameters
    ----------
    patches : list[Patch]   local patches (eval_nodes, center, radius must be set)
    halo    : HaloComm      from build_halo_comm (LSSetup calls this for you)
    """
    if not patches:
        return

    d = patches[0].interp_nodes.shape[1]
    n_owned = len(halo.owned_indices)

    # Accumulators at owned eval nodes
    W_owned     = np.zeros(n_owned)
    gradW_owned = np.zeros((n_owned, d))
    lapW_owned  = np.zeros(n_owned)

    # Halo send buffers: contributions to nodes owned by other ranks
    send_W     = {s: np.zeros(len(halo.mv_send_gidx[s])) for s in halo.neighbor_ranks}
    send_gradW = {s: np.zeros((len(halo.mv_send_gidx[s]), d)) for s in halo.neighbor_ranks}
    send_lapW  = {s: np.zeros(len(halo.mv_send_gidx[s])) for s in halo.neighbor_ranks}

    # Pass 1: accumulate raw weights; cache (w, gw, lw) per patch for Pass 2
    cache = []
    for patch, ph in zip(patches, halo.patch_halo):
        w  = C2Weight(patch.eval_nodes, patch.center, patch.radius)
        gw = C2WeightGradient(patch.eval_nodes, patch.center, patch.radius)
        lw = C2WeightLaplacian(patch.eval_nodes, patch.center, patch.radius)
        cache.append((w, gw, lw))

        # Home contributions
        W_owned[ph.home_lidx]     += w[ph.home_mask]
        gradW_owned[ph.home_lidx] += gw[ph.home_mask]
        lapW_owned[ph.home_lidx]  += lw[ph.home_mask]

        # Halo contributions
        for s in ph.nbr_ranks:
            bidx = ph.nbr_buf_idx[s]
            mask = ph.nbr_mask[s]
            send_W[s][bidx]       += w[mask]
            send_gradW[s][bidx]   += gw[mask]
            send_lapW[s][bidx]    += lw[mask]

    # Forward exchange: accumulate halo contributions at owned nodes
    recv_W     = halo.mv_exchange(send_W,     tag=10)
    recv_gradW = halo.mv_exchange(
        {s: buf.ravel() for s, buf in send_gradW.items()}, tag=11, n_components=d)
    recv_lapW  = halo.mv_exchange(send_lapW,  tag=12)

    for s, buf in recv_W.items():
        W_owned[halo.mv_recv_lidx[s]] += buf
    for s, buf in recv_gradW.items():
        gradW_owned[halo.mv_recv_lidx[s]] += buf.reshape(-1, d)
    for s, buf in recv_lapW.items():
        lapW_owned[halo.mv_recv_lidx[s]] += buf

    # Reverse exchange: broadcast (W, ∇W, ΔW) from owners to users
    halo_W     = halo.rmv_exchange_1d(W_owned,                    tag=13)
    halo_gradW = halo.rmv_exchange_nd(gradW_owned,                 tag=14)
    halo_lapW  = halo.rmv_exchange_1d(lapW_owned,                  tag=15)

    # Pass 2: normalise
    for patch, ph, (w, gw, lw) in zip(patches, halo.patch_halo, cache):
        n_eval = len(patch.eval_node_indices)

        # Assemble W, ∇W, ΔW at all eval nodes of this patch
        Wn    = np.empty(n_eval)
        gWn   = np.empty((n_eval, d))
        lWn   = np.empty(n_eval)

        Wn[ph.home_mask]    = W_owned[ph.home_lidx]
        gWn[ph.home_mask]   = gradW_owned[ph.home_lidx]
        lWn[ph.home_mask]   = lapW_owned[ph.home_lidx]

        for s in ph.nbr_ranks:
            mask = ph.nbr_mask[s]
            bidx = ph.nbr_buf_idx[s]
            Wn[mask]  = halo_W[s][bidx]
            gWn[mask] = halo_gradW[s][bidx]
            lWn[mask] = halo_lapW[s][bidx]

        W2 = Wn ** 2
        W3 = Wn * W2

        patch.w_bar  = w / Wn

        patch.gw_bar = gw / Wn[:, None] - w[:, None] * gWn / W2[:, None]

        patch.lw_bar = (lw / Wn
                        - 2.0 * np.einsum('ij,ij->i', gw, gWn) / W2
                        - w * lWn / W2
                        + 2.0 * w * np.einsum('ij,ij->i', gWn, gWn) / W3)


######################################
# Vectorized Wendland C2 weight functions
######################################

def C2Weight(x, center, radius):
    rho = np.linalg.norm(x - center, axis=1) / radius
    return (1.0 - rho) ** 4 * (4.0 * rho + 1.0)


def C2WeightGradient(x, center, radius):
    diff   = x - center
    r      = np.linalg.norm(diff, axis=1)
    rho    = r / radius
    safe_r = np.where(r == 0.0, 1.0, r)
    factor = -20.0 * (1.0 - rho) ** 3 * rho / (radius * safe_r)
    factor = np.where(r == 0.0, 0.0, factor)
    return factor[:, None] * diff


def C2WeightLaplacian(x, center, radius):
    dim    = x.shape[1]
    r      = np.linalg.norm(x - center, axis=1)
    rho    = r / radius
    safe_rho = np.where(rho == 0.0, 1.0, rho)
    psi_d  = -20.0 * (1.0 - rho) ** 3 * rho
    psi_dd =  20.0 * (1.0 - rho) ** 2 * (4.0 * rho - 1.0)
    result = (psi_dd + (dim - 1.0) * psi_d / safe_rho) / radius ** 2
    return np.where(r == 0.0, -20.0 * dim / radius ** 2, result)
