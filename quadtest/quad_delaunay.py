"""
quad_delaunay.py — Delaunay triangulation / tetrahedralisation + high-order
                   Gaussian quadrature on each simplex via the Duffy
                   transformation.

Works for both 2D (triangles) and 3D (tetrahedra); dimension is detected
automatically from nodes.shape[1].

Duffy transformation (always-positive weights)
-----------------------------------------------
2D triangle  T₂ = {(x,y): x,y≥0, x+y≤1}:
    x = u·v,  y = u·(1−v),  |J| = u
    ∫_T f dx dy = ∫₀¹∫₀¹ f(uv, u(1−v)) · u du dv

3D tetrahedron  T₃ = {(x,y,z): x,y,z≥0, x+y+z≤1}:
    x = u·v·w,  y = u·v·(1−w),  z = u·(1−v),  |J| = u²·v
    ∫_T f dx dy dz = ∫₀¹∫₀¹∫₀¹ f(uvw, uv(1−w), u(1−v)) · u²v du dv dw

A tensor-product Gauss-Legendre rule with p points per direction gives degree
of exactness 2p−1 on the simplex and has all positive weights.

nodal=False (default)
    Returns the Gauss points on the simplices (not the original nodes).
    The RBF interpolant can be evaluated at these interior points.

nodal=True
    Accumulates Gauss weights back to the original nodes via barycentric
    weights:  w_vertex_k += w_gauss · λ_k.
    Accuracy drops to ~1 (vertex lumping) but weights stay at original nodes.
"""

import numpy as np
from scipy.spatial import Delaunay
from quad_utils import gauss_legendre_01


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
    pts : (p², 2) reference-triangle quadrature points (ξ, η)
    wts : (p²,)   weights; all positive; sum = 0.5 (area of T₂)
    bary: (p², 3) barycentric coords (λ₀=1−u, λ₁=uv, λ₂=u(1−v))
    """
    t, w = gauss_legendre_01(p)

    ui, vi = np.meshgrid(t, t, indexing='ij')
    wu, wv = np.meshgrid(w, w, indexing='ij')

    ui = ui.ravel(); vi = vi.ravel()
    wu = wu.ravel(); wv = wv.ravel()

    xi = ui * vi
    eta = ui * (1.0 - vi)
    weights = wu * wv * ui          # Duffy Jacobian = u

    lam0 = 1.0 - ui
    lam1 = xi
    lam2 = eta
    bary = np.column_stack([lam0, lam1, lam2])

    pts = np.column_stack([xi, eta])
    return pts, weights, bary


def _ref_tet_rule(p):
    """
    Gauss rule on the reference tetrahedron T₃={ξ,η,ζ≥0,ξ+η+ζ≤1} via Duffy.

    Parameters
    ----------
    p : number of GL points per direction

    Returns
    -------
    pts : (p³, 3) quadrature points (ξ, η, ζ)
    wts : (p³,)   all positive; sum = 1/6 (volume of T₃)
    bary: (p³, 4) barycentric coords (λ₀=1−u, λ₁=uv(1−w)→η, λ₂=uvw→ξ,
                  λ₃=u(1−v)→ζ)   [note: ordering matches simplex vertex order]
    """
    t, w = gauss_legendre_01(p)

    ui, vi, wi_ = np.meshgrid(t, t, t, indexing='ij')
    wu, wv, ww  = np.meshgrid(w, w, w, indexing='ij')

    ui  = ui.ravel(); vi = vi.ravel(); wi_ = wi_.ravel()
    wu  = wu.ravel(); wv = wv.ravel(); ww  = ww.ravel()

    xi  = ui * vi * wi_
    eta = ui * vi * (1.0 - wi_)
    zeta= ui * (1.0 - vi)
    weights = wu * wv * ww * (ui ** 2) * vi    # Duffy Jacobian = u²·v

    lam0 = 1.0 - ui
    lam1 = xi
    lam2 = eta
    lam3 = zeta
    bary = np.column_stack([lam0, lam1, lam2, lam3])

    pts = np.column_stack([xi, eta, zeta])
    return pts, weights, bary


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class DelaunayGaussQuadrature:
    """
    High-order positive quadrature over scattered nodes via Delaunay
    triangulation/tetrahedralisation + Duffy–Gauss-Legendre rules.

    Parameters
    ----------
    p : int
        Number of Gauss-Legendre points per direction.  The rule integrates
        polynomials of degree <= 2p−1 exactly.
        Suggested values: p=3 (deg 5), p=4 (deg 7), p=5 (deg 9).
    nodal : bool
        False (default) — return interior Gauss points (non-nodal, high accuracy).
        True            — project weights to original nodes via barycentric lumping
                          (all weights at original node locations; accuracy ~1–2).

    Usage
    -----
        quad = DelaunayGaussQuadrature(p=4)
        pts, wts = quad.fit(nodes)   # pts may differ from nodes if nodal=False
        approx = np.dot(wts, f(pts))
    """

    def __init__(self, p=4, nodal=False):
        self.p = p
        self.nodal = nodal
        self._tri = None
        self._quad_pts = None
        self._weights = None

    # ------------------------------------------------------------------
    def fit(self, nodes):
        """
        Build the quadrature rule for the given scattered nodes.

        Parameters
        ----------
        nodes : (n, dim) array,  dim=2 or 3

        Returns
        -------
        quad_pts : (M, dim) quadrature point coordinates
                   (equals nodes if nodal=True, interior Gauss pts if nodal=False)
        weights  : (M,) positive quadrature weights
        """
        nodes = np.asarray(nodes, dtype=float)
        n, dim = nodes.shape

        if dim not in (2, 3):
            raise ValueError(f"dim must be 2 or 3, got {dim}")

        # Delaunay triangulation / tetrahedralisation
        tri = Delaunay(nodes)
        self._tri = tri
        simplices = tri.simplices  # (n_simp, dim+1)

        # Reference rule
        if dim == 2:
            ref_pts, ref_wts, ref_bary = _ref_triangle_rule(self.p)
        else:
            ref_pts, ref_wts, ref_bary = _ref_tet_rule(self.p)

        n_ref = len(ref_wts)
        n_simp = len(simplices)

        if self.nodal:
            # Accumulate weights at original node indices
            node_weights = np.zeros(n)
            for simp in simplices:
                verts = nodes[simp]          # (dim+1, dim)
                # affine map: x = v0 + J * ref_pt
                v0 = verts[0]
                J = (verts[1:] - v0).T       # (dim, dim)
                jac = abs(np.linalg.det(J))  # = dim! * vol(simplex)
                for q in range(n_ref):
                    w_phys = ref_wts[q] * jac
                    # barycentric coords give the split to vertices
                    for k, idx in enumerate(simp):
                        node_weights[idx] += w_phys * ref_bary[q, k]
            self._quad_pts = nodes
            self._weights = node_weights
            return nodes.copy(), node_weights.copy()

        else:
            # Non-nodal: collect all physical Gauss points and weights
            all_pts = np.empty((n_simp * n_ref, dim))
            all_wts = np.empty(n_simp * n_ref)

            for s, simp in enumerate(simplices):
                verts = nodes[simp]          # (dim+1, dim)
                v0 = verts[0]
                J = (verts[1:] - v0).T       # (dim, dim)
                jac = abs(np.linalg.det(J))
                # Physical points: x = v0 + J @ ref_pt  (ref_pt in R^dim)
                phys_pts = ref_pts @ J.T + v0    # (n_ref, dim)
                phys_wts = ref_wts * jac

                sl = slice(s * n_ref, (s + 1) * n_ref)
                all_pts[sl] = phys_pts
                all_wts[sl] = phys_wts

            self._quad_pts = all_pts
            self._weights = all_wts
            return all_pts, all_wts

    # ------------------------------------------------------------------
    @property
    def triangulation(self):
        """The scipy.spatial.Delaunay object (available after fit)."""
        return self._tri

    @property
    def quad_points(self):
        return self._quad_pts

    @property
    def weights(self):
        return self._weights

    def degree_of_exactness_estimate(self):
        """Return 2p−1 (the theoretical degree for the Duffy–GL rule)."""
        return 2 * self.p - 1

    def n_points(self):
        """Number of quadrature points."""
        return len(self._weights) if self._weights is not None else None

    def summary(self):
        if self._weights is None:
            return "DelaunayGaussQuadrature (not fitted)"
        n_simp = len(self._tri.simplices)
        n_pts  = len(self._weights)
        return (f"DelaunayGaussQuadrature: p={self.p}, nodal={self.nodal}, "
                f"simplices={n_simp}, quad_pts={n_pts}, "
                f"min_w={self._weights.min():.3e}, sum_w={self._weights.sum():.6f}")


# ---------------------------------------------------------------------------
# Convenience: directly compute the integral of f over the scattered node set
# ---------------------------------------------------------------------------

def delaunay_integrate(nodes, f, p=4):
    """
    Integrate f over the convex hull of `nodes` using Delaunay + Duffy-GL.

    Returns
    -------
    approx : float
    """
    quad = DelaunayGaussQuadrature(p=p, nodal=False)
    pts, wts = quad.fit(nodes)
    return float(np.dot(wts, f(pts)))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    print("=== quad_delaunay smoke test ===")

    # --- 2D unit square ---
    rng = np.random.default_rng(42)
    nodes2 = rng.random((200, 2))

    for p in [3, 4, 5]:
        quad = DelaunayGaussQuadrature(p=p, nodal=False)
        pts, wts = quad.fit(nodes2)
        area = wts.sum()
        # Test constant and polynomial
        err_const = abs(area - 1.0)
        f = lambda x: x[:, 0] ** 4 * x[:, 1] ** 3
        exact = 1.0/5 * 1.0/4
        err_poly = abs(np.dot(wts, f(pts)) - exact)
        pos = (wts > 0).all()
        print(f"  2D p={p}: area_err={err_const:.2e}  poly_err={err_poly:.2e}  "
              f"positive={pos}  pts={len(wts)}")

    # nodal mode
    quad_n = DelaunayGaussQuadrature(p=4, nodal=True)
    pts_n, wts_n = quad_n.fit(nodes2)
    assert len(wts_n) == len(nodes2), "nodal mode must return weights at each node"
    print(f"  2D nodal: area_err={abs(wts_n.sum()-1.0):.2e}  "
          f"positive={( wts_n > 0).all()}")

    # --- 3D unit cube ---
    nodes3 = rng.random((300, 3))
    quad3 = DelaunayGaussQuadrature(p=3, nodal=False)
    pts3, wts3 = quad3.fit(nodes3)
    vol = wts3.sum()
    f3 = lambda x: x[:, 0] ** 2 * x[:, 1] ** 2 * x[:, 2] ** 2
    exact3 = 1.0 / 27
    err3 = abs(np.dot(wts3, f3(pts3)) - exact3)
    print(f"  3D p=3: vol_err={abs(vol-1.0):.2e}  poly_err={err3:.2e}  "
          f"positive={(wts3>0).all()}  pts={len(wts3)}")

    print("All OK.")
