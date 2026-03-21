import numpy as np
from scipy.special import roots_jacobi, eval_legendre
from scipy.spatial import cKDTree


def gll_nodes_weights(n):
    """
    Gauss-Lobatto-Legendre nodes and weights on [-1, 1].
    n: total number of points (includes both endpoints ±1).
    Returns (xi, w) both shape (n,).
    """
    if n < 2:
        raise ValueError("GLL requires n >= 2")
    if n == 2:
        return np.array([-1.0, 1.0]), np.array([1.0, 1.0])

    # Interior n-2 points are roots of P'_{n-1} = (n-1) P_{n-2}^{(1,1)}
    xi_int, _ = roots_jacobi(n - 2, 1.0, 1.0)
    xi = np.concatenate([[-1.0], np.sort(xi_int), [1.0]])

    # Weights: w_i = 2 / (n(n-1) [P_{n-1}(xi_i)]^2)
    Pn1 = eval_legendre(n - 1, xi)
    w = 2.0 / (n * (n - 1) * Pn1**2)
    return xi, w


def polar_quad(center, radius, n_r, n_theta):
    """
    Polar quadrature rule over a disk of given radius centered at center.

    Radial direction: GLL points on [-1,1] mapped to [0, radius].
    Angular direction: n_theta uniform points on [0, 2pi).

    Returns
    -------
    pts     : (n_r * n_theta, 2)  quadrature points
    weights : (n_r * n_theta,)    weights including the polar Jacobian r dr dtheta
    """
    xi, w_gll = gll_nodes_weights(n_r)

    # Map from [-1, 1] to [0, radius]
    r = radius * (xi + 1.0) / 2.0         # (n_r,)
    w_r = w_gll * (radius / 2.0)          # scaled radial weights

    theta = 2.0 * np.pi * np.arange(n_theta) / n_theta  # (n_theta,)
    w_theta = 2.0 * np.pi / n_theta

    # Build all (r, theta) pairs
    R, T = np.meshgrid(r, theta, indexing='ij')        # (n_r, n_theta)
    Wr, _ = np.meshgrid(w_r, theta, indexing='ij')

    pts = np.stack([
        center[0] + R * np.cos(T),
        center[1] + R * np.sin(T),
    ], axis=-1).reshape(-1, 2)                         # (n_r*n_theta, 2)

    # Polar Jacobian: r dr dtheta
    weights = (Wr * w_theta * R).reshape(-1)           # (n_r*n_theta,)
    return pts, weights


def filter_interior(pts, weights, bdy_nodes, bdy_normals):
    """
    Zero the weights of quadrature points that lie outside the domain.

    For each point q, find the nearest boundary node x_b with outward
    normal n_b. The point is inside if (q - x_b) . n_b <= 0.

    Returns
    -------
    weights : (n_q,)  filtered weights (outside points zeroed)
    inside  : (n_q,)  bool mask, True for interior points
    """
    tree = cKDTree(bdy_nodes)
    _, idx = tree.query(pts)
    x_b = bdy_nodes[idx]   # (n_q, d)
    n_b = bdy_normals[idx]  # (n_q, d)

    signed = np.sum((pts - x_b) * n_b, axis=1)   # positive = outside
    inside = signed <= 0.0
    return weights * inside, inside


def eval_phi_at_pts(pts, patch_nodes, e):
    """
    Evaluate Gaussian RBF basis functions at quadrature points.
    phi_j(q) = exp(-e^2 |q - x_j|^2)

    Returns Phi_q : (n_q, n_loc)
    """
    diff = pts[:, np.newaxis, :] - patch_nodes[np.newaxis, :, :]  # (n_q, n_loc, d)
    r2 = np.sum(diff**2, axis=2)                                   # (n_q, n_loc)
    return np.exp(-(e**2) * r2)


def eval_grad_phi_at_pts(pts, patch_nodes, e):
    """
    Evaluate gradient of Gaussian RBF basis w.r.t. the evaluation point q.
    d(phi_j)/d(q_k) = -2 e^2 (q_k - x_j_k) phi_j(q)

    Returns dPhi_q : (d, n_q, n_loc)
    """
    diff = pts[:, np.newaxis, :] - patch_nodes[np.newaxis, :, :]  # (n_q, n_loc, d)
    r2 = np.sum(diff**2, axis=2)                                   # (n_q, n_loc)
    Phi_q = np.exp(-(e**2) * r2)                                   # (n_q, n_loc)
    # transpose diff to (d, n_q, n_loc) then broadcast with Phi_q
    dPhi_q = -2.0 * e**2 * diff.transpose(2, 0, 1) * Phi_q        # (d, n_q, n_loc)
    return dPhi_q
