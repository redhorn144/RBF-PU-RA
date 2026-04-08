import numpy as np


def NormalizeWeights(comm, patches, nodes):
    N = nodes.shape[0]
    d = nodes.shape[1]

    W_local = np.zeros(N)
    gradW_local = np.zeros((N, d))
    lapW_local = np.zeros(N)

    for patch in patches:
        idx = patch.node_indices
        patch_nodes = nodes[idx]

        w = C2Weight(patch_nodes, patch.center, patch.radius)
        gw = C2WeightGradient(patch_nodes, patch.center, patch.radius)
        lw = C2WeightLaplacian(patch_nodes, patch.center, patch.radius)

        W_local[idx] += w
        gradW_local[idx] += gw
        lapW_local[idx] += lw

    W = np.zeros_like(W_local)
    gradW = np.zeros_like(gradW_local)
    lapW = np.zeros_like(lapW_local)

    comm.Allreduce(W_local, W)
    comm.Allreduce(gradW_local, gradW)
    comm.Allreduce(lapW_local, lapW)

    for patch in patches:
        idx = patch.node_indices
        patch_nodes = nodes[idx]

        w = C2Weight(patch_nodes, patch.center, patch.radius)
        gw = C2WeightGradient(patch_nodes, patch.center, patch.radius)
        lw = C2WeightLaplacian(patch_nodes, patch.center, patch.radius)

        Wn = W[idx]
        gWn = gradW[idx]
        lWn = lapW[idx]

        patch.w_bar = w / Wn
        patch.gw_bar = gw / Wn[:, None] - w[:, None] * gWn / Wn[:, None]**2
        patch.lw_bar = (lw / Wn
                        - 2.0 * np.sum(gw * gWn, axis=1) / Wn**2
                        - w * lWn / Wn**2
                        + 2.0 * w * np.sum(gWn * gWn, axis=1) / Wn**3)


######################################
# Vectorized Wendland C2 weight functions
# x: (n, d) array of node positions
# center: (d,) patch center
# radius: scalar patch radius
######################################

def C2Weight(x, center, radius):
    """Returns (n,) array of weights."""
    r = np.linalg.norm(x - center, axis=1) / radius
    return (1 - r)**4 * (4 * r + 1)


def C2WeightGradient(x, center, radius):
    """Returns (n, d) array of weight gradients."""
    diff = x - center
    r = np.linalg.norm(diff, axis=1)
    rho = r / radius

    # Avoid division by zero; gradient is zero at center
    safe_r = np.where(r == 0, 1.0, r)
    factor = -20.0 * (1 - rho)**3 * rho / (radius * safe_r)
    factor = np.where(r == 0, 0.0, factor)

    return factor[:, None] * diff


def C2WeightLaplacian(x, center, radius):
    """Returns (n,) array of weight Laplacians."""
    d = x.shape[1]
    r = np.linalg.norm(x - center, axis=1)
    rho = r / radius

    # At r=0: Laplacian = -20 * d / radius^2
    at_origin = -20.0 * d / radius**2

    safe_rho = np.where(rho == 0, 1.0, rho)
    psi_d = -20.0 * (1 - rho)**3 * rho
    psi_dd = 20.0 * (1 - rho)**2 * (4 * rho - 1)
    result = (psi_dd + (d - 1) * psi_d / safe_rho) / radius**2

    return np.where(r == 0, at_origin, result)