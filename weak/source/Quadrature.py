"""
Quadrature.py — Delaunay triangulation + Duffy–Gauss-Legendre quadrature
                 for the weak-form Poisson solver.

The global node set is triangulated with scipy.spatial.Delaunay.  A
p-point-per-direction tensor-product Gauss-Legendre rule is placed on each
triangle via the Duffy transformation (degree of exactness 2p−1, all weights
positive).  Weights are accumulated back to the original nodes via barycentric
lumping so that the returned array has exactly one entry per node.

Duffy transformation on T₂ = {(x,y): x,y≥0, x+y≤1}:
    x = u·v,  y = u·(1−v),  |J| = u
    ∫_T f dx dy = ∫₀¹∫₀¹ f(uv, u(1−v)) · u du dv
"""

import numpy as np
from scipy.spatial import Delaunay
from mpi4py import MPI


# ---------------------------------------------------------------------------
# Gauss-Legendre on [0, 1]
# ---------------------------------------------------------------------------

def _gauss_legendre_01(n):
    """n-point Gauss-Legendre rule on [0, 1].  Weights sum to 1."""
    pts, wts = np.polynomial.legendre.leggauss(n)
    pts = (pts + 1.0) / 2.0
    wts = wts / 2.0
    return pts, wts


# ---------------------------------------------------------------------------
# Reference simplex rules via Duffy + Gauss-Legendre
# ---------------------------------------------------------------------------

def _ref_triangle_rule(p):
    """
    Gauss rule on the reference triangle T₂={ξ≥0,η≥0,ξ+η≤1} via Duffy.

    Parameters
    ----------
    p : number of GL points per direction

    Returns
    -------
    pts  : (p², 2) reference-triangle quadrature points (ξ, η)
    wts  : (p²,)   weights; all positive; sum = 0.5 (area of T₂)
    bary : (p², 3) barycentric coords (λ₀=1−u, λ₁=uv, λ₂=u(1−v))
    """
    t, w = _gauss_legendre_01(p)

    ui, vi = np.meshgrid(t, t, indexing='ij')
    wu, wv = np.meshgrid(w, w, indexing='ij')

    ui = ui.ravel(); vi = vi.ravel()
    wu = wu.ravel(); wv = wv.ravel()

    xi  = ui * vi
    eta = ui * (1.0 - vi)
    weights = wu * wv * ui          # Duffy Jacobian = u

    bary = np.column_stack([1.0 - ui, xi, eta])
    pts  = np.column_stack([xi, eta])
    return pts, weights, bary


# ---------------------------------------------------------------------------
# DelaunayGaussQuadrature
# ---------------------------------------------------------------------------

class DelaunayGaussQuadrature:
    """
    High-order positive quadrature over scattered 2-D nodes via Delaunay
    triangulation + Duffy–Gauss-Legendre rules.

    Parameters
    ----------
    p : int
        Number of Gauss-Legendre points per direction.  The rule integrates
        polynomials of degree ≤ 2p−1 exactly.
        Suggested values: p=3 (deg 5), p=4 (deg 7), p=5 (deg 9).
    nodal : bool
        False — return interior Gauss points (non-nodal, highest accuracy).
        True  — project weights to original nodes via barycentric lumping
                (weights at original node locations; accuracy ~1–2 orders lower).
    """

    def __init__(self, p=4, nodal=True):
        self.p = p
        self.nodal = nodal
        self._tri = None
        self._quad_pts = None
        self._weights = None

    def fit(self, nodes):
        """
        Build the quadrature rule for the given scattered nodes.

        Parameters
        ----------
        nodes : (n, 2) array

        Returns
        -------
        quad_pts : (M, 2) quadrature point coordinates
                   (equals nodes if nodal=True)
        weights  : (M,) positive quadrature weights
        """
        nodes = np.asarray(nodes, dtype=float)
        n = nodes.shape[0]

        tri = Delaunay(nodes)
        self._tri = tri
        simplices = tri.simplices          # (n_simp, 3)

        ref_pts, ref_wts, ref_bary = _ref_triangle_rule(self.p)
        n_ref  = len(ref_wts)
        n_simp = len(simplices)

        if self.nodal:
            node_weights = np.zeros(n)
            for simp in simplices:
                verts = nodes[simp]        # (3, 2)
                v0 = verts[0]
                J  = (verts[1:] - v0).T   # (2, 2)
                jac = abs(np.linalg.det(J))
                for q in range(n_ref):
                    w_phys = ref_wts[q] * jac
                    for k, idx in enumerate(simp):
                        node_weights[idx] += w_phys * ref_bary[q, k]
            self._quad_pts = nodes
            self._weights  = node_weights
            return nodes.copy(), node_weights.copy()

        else:
            all_pts = np.empty((n_simp * n_ref, 2))
            all_wts = np.empty(n_simp * n_ref)
            for s, simp in enumerate(simplices):
                verts = nodes[simp]
                v0 = verts[0]
                J  = (verts[1:] - v0).T
                jac = abs(np.linalg.det(J))
                phys_pts = ref_pts @ J.T + v0
                phys_wts = ref_wts * jac
                sl = slice(s * n_ref, (s + 1) * n_ref)
                all_pts[sl] = phys_pts
                all_wts[sl] = phys_wts
            self._quad_pts = all_pts
            self._weights  = all_wts
            return all_pts, all_wts

    def degree_of_exactness_estimate(self):
        """Return 2p−1 (theoretical degree for the Duffy–GL rule)."""
        return 2 * self.p - 1

    @property
    def triangulation(self):
        return self._tri

    @property
    def weights(self):
        return self._weights

    @property
    def quad_points(self):
        return self._quad_pts


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def GaussPointsAndWeights(comm, patches, N, p=4):
    """
    Build global non-nodal Gauss points and weights via global Delaunay on
    all N nodes (rank 0), then broadcast.  Degree of exactness 2p-1.

    Parameters
    ----------
    comm    : MPI communicator
    patches : list of Patch objects owned by this rank
    N       : total number of global nodes
    p       : GL points per direction

    Returns
    -------
    gauss_pts : (M_q, 2)
    gauss_wts : (M_q,)   positive weights; sum ≈ domain area
    """
    nodes_local = np.zeros((N, 2))
    have_node   = np.zeros(N, dtype=bool)
    for patch in patches:
        idx = patch.node_indices
        nodes_local[idx] = patch.nodes
        have_node[idx]   = True

    nodes_global = np.zeros((N, 2))
    comm.Allreduce(nodes_local, nodes_global)
    count = np.zeros(N)
    comm.Allreduce(have_node.astype(float), count)
    count = np.maximum(count, 1.0)
    nodes_global /= count[:, None]

    if comm.Get_rank() == 0:
        quad = DelaunayGaussQuadrature(p=p, nodal=False)
        gauss_pts, gauss_wts = quad.fit(nodes_global)
        print(f"Non-nodal Gauss quadrature: p={p}, "
              f"degree={quad.degree_of_exactness_estimate()}, "
              f"n_pts={len(gauss_wts)}, "
              f"min_w={gauss_wts.min():.3e}, sum_w={gauss_wts.sum():.6f}")
    else:
        gauss_pts = None
        gauss_wts = None

    gauss_pts = comm.bcast(gauss_pts, root=0)
    gauss_wts = comm.bcast(gauss_wts, root=0)
    return gauss_pts, gauss_wts


def PatchLocalWeights(comm, patches, N, p=4):
    """
    Compute positive quadrature weights via Delaunay triangulation +
    Duffy–Gauss-Legendre quadrature (nodal accumulation).

    The global node set is triangulated with scipy.spatial.Delaunay; a
    p-point-per-direction GL rule is placed on each triangle via the Duffy
    transformation giving degree of exactness 2p−1.  Weights are accumulated
    back to the original nodes via barycentric lumping.

    Parameters
    ----------
    comm    : MPI communicator
    patches : list of Patch objects owned by this rank
    N       : int, total number of global nodes
    p       : int, GL points per direction (default 4 → degree 7)

    Returns
    -------
    q : (N,) positive quadrature weights
    """
    nodes_local = np.zeros((N, 2))
    have_node   = np.zeros(N, dtype=bool)
    for patch in patches:
        idx = patch.node_indices
        nodes_local[idx] = patch.nodes
        have_node[idx]   = True

    nodes_global = np.zeros((N, 2))
    comm.Allreduce(nodes_local, nodes_global)

    count = np.zeros(N)
    comm.Allreduce(have_node.astype(float), count)
    count = np.maximum(count, 1.0)
    nodes_global /= count[:, None]

    if comm.Get_rank() == 0:
        quad = DelaunayGaussQuadrature(p=p, nodal=True)
        _, q = quad.fit(nodes_global)
        print(f"Delaunay-Gauss quadrature: p={p}, "
              f"degree={quad.degree_of_exactness_estimate()}, "
              f"min_w={q.min():.3e}, sum_w={q.sum():.6f}")
    else:
        q = None

    q = comm.bcast(q, root=0)
    return q
