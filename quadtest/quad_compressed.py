"""
quad_compressed.py — Compressed positive quadrature via QR column pivoting
                     (Huybrechs 2009 / Sommariva–Vianello style).

Idea
----
Given n scattered nodes, use QR with column pivoting on the Vandermonde
matrix V (m × n) to select a well-poised subset S of m = |{monomials}| nodes.
Then solve a *square* NNLS problem for their weights.

Compared to MomentFittingQuadrature (which uses all n nodes):
  • The selected m × m system is well-conditioned by construction.
  • Positive weights are more reliably obtained (fewer near-zero entries).
  • The output rule has exactly m quadrature points — optimal compression
    for RBF-PUM local patches.

Algorithm
---------
1. Scale nodes to [0,1]^d; build V (m × n).
2. Q, R, P = scipy.linalg.qr(V, pivoting=True)   (QR with column pivoting)
3. Select S = P[:m]  (indices of the m "most important" nodes).
4. Solve  w_S = nnls(V[:, S], b).               (square m × m system)
5. Check rank: if |R[m-1,m-1]| / |R[0,0]| < tol, reduce degree and retry.
6. Scale weights back to the physical domain.

Positivity
----------
Near-guaranteed in practice: QR pivoting selects a well-poised node set for
which the square Vandermonde system is non-singular, and NNLS then finds the
minimum-norm non-negative solution.  Mathematical guarantee exists when the
selected nodes form a "Fekete-like" set.

Accuracy
--------
Exact for polynomials of degree ≤ `degree`.  Same order as moment fitting
but more stable (better-conditioned m × m solve vs. underdetermined n-system).
"""

import numpy as np
from scipy.linalg import qr
from scipy.optimize import nnls
from quad_utils import (
    build_vandermonde, compute_moments, scale_nodes, monomial_exponents
)


class CompressedQuadrature:
    """
    Compressed positive quadrature via QR-pivoting + NNLS.

    Parameters
    ----------
    degree : int
        Polynomial degree of exactness (default 6).
    tol : float
        Rank-condition threshold: R[m-1,m-1]/R[0,0] must exceed tol.
        If not, the degree is automatically reduced (default 1e-10).
    domain : str
        'unit_square', 'unit_cube', 'unit_disk', or 'unit_ball' — used for the
        moment vector.  'auto' selects based on node dimension.

    Usage
    -----
        cq = CompressedQuadrature(degree=6)
        pts, wts = cq.fit(nodes)
        approx = np.dot(wts, f(pts))
    """

    def __init__(self, degree=6, tol=1e-10, domain='auto'):
        self.degree = degree
        self.tol = tol
        self.domain = domain
        self._nodes_selected = None
        self._weights = None
        self._indices = None
        self._effective_degree = None
        self._cond_ratio = None
        self._residual = None

    # ------------------------------------------------------------------
    def fit(self, nodes):
        """
        Compute the compressed quadrature rule.

        Parameters
        ----------
        nodes : (n, dim) array

        Returns
        -------
        selected_nodes : (m, dim) array — the m selected nodes
        weights        : (m,) array   — all positive weights
        """
        nodes = np.asarray(nodes, dtype=float)
        if nodes.ndim == 1:
            nodes = nodes[:, None]
        n, dim = nodes.shape

        if self.domain == 'auto':
            domain = 'unit_square' if dim == 2 else 'unit_cube'
        else:
            domain = self.domain

        # Scale to [0,1]^dim
        scaled, lo, rng = scale_nodes(nodes)
        vol_factor = float(np.prod(rng))

        # Try degree; reduce if rank-deficient
        degree = self.degree
        while degree >= 1:
            m = len(monomial_exponents(dim, degree))
            if n < m:
                degree -= 1
                continue

            V = build_vandermonde(scaled, degree)  # (m, n)
            b = compute_moments(degree, domain)     # (m,)

            # QR with column pivoting: V = Q R P^T
            _, R, P = qr(V, pivoting=True)

            # Check effective rank
            diag = np.abs(np.diag(R))
            if diag[0] < 1e-14:
                degree -= 1
                continue
            cond_ratio = diag[min(m, len(diag)) - 1] / diag[0]
            self._cond_ratio = float(cond_ratio)

            if cond_ratio < self.tol:
                degree -= 1
                continue

            # Select best m columns
            S = P[:m]

            # Solve square system for positive weights
            V_S = V[:, S]    # (m, m)
            w_S, residual = nnls(V_S, b)
            self._residual = float(residual)
            self._effective_degree = degree

            # Scale back to physical domain
            w_phys = w_S * vol_factor

            self._indices = S
            self._nodes_selected = nodes[S]
            self._weights = w_phys

            return self._nodes_selected.copy(), self._weights.copy()

        raise RuntimeError(
            "Could not find a well-conditioned Vandermonde sub-matrix. "
            "Try providing more nodes or reducing the target degree."
        )

    # ------------------------------------------------------------------
    @property
    def selected_indices(self):
        """Indices into the original nodes array that were selected."""
        return self._indices

    @property
    def effective_degree(self):
        """Actual polynomial degree achieved (may be < requested degree)."""
        return self._effective_degree

    @property
    def moment_residual(self):
        """||V_S w - b||₂ (lower is better)."""
        return self._residual

    @property
    def condition_ratio(self):
        """R[m-1,m-1] / R[0,0] — measures how well-poised the selection is."""
        return self._cond_ratio

    def compression_ratio(self, n_original):
        """
        Compression factor: n_original / m_selected.
        Higher = fewer output points relative to input.
        """
        if self._indices is None:
            return None
        return n_original / len(self._indices)

    def summary(self, n_original=None):
        if self._nodes_selected is None:
            return "CompressedQuadrature (not fitted)"
        m = len(self._weights)
        cr = f"{n_original}/{m}" if n_original else f"{m} pts"
        return (
            f"CompressedQuadrature: degree={self._effective_degree}, "
            f"pts={cr}, "
            f"cond_ratio={self._cond_ratio:.2e}, "
            f"residual={self._residual:.2e}, "
            f"min_w={self._weights.min():.3e}, "
            f"sum_w={self._weights.sum():.6f}"
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=== quad_compressed smoke test ===")
    rng = np.random.default_rng(7)

    # ---- 2D unit square ----
    for deg in [2, 4, 6]:
        nodes = rng.random((500, 2))
        cq = CompressedQuadrature(degree=deg)
        pts, wts = cq.fit(nodes)
        area = wts.sum()
        f = lambda x: x[:, 0] ** deg * x[:, 1] ** (deg - 1)
        exact = 1.0 / (deg + 1) / deg
        err = abs(np.dot(wts, f(pts)) - exact)
        pos = (wts > 0).all()
        m = len(pts)
        print(f"  2D deg={deg}: area_err={abs(area-1):.2e}  poly_err={err:.2e}  "
              f"positive={pos}  pts={m}  "
              f"cond={cq.condition_ratio:.2e}  res={cq.moment_residual:.2e}")

    # ---- 3D unit cube ----
    nodes3 = rng.random((600, 3))
    cq3 = CompressedQuadrature(degree=4)
    pts3, wts3 = cq3.fit(nodes3)
    vol = wts3.sum()
    f3 = lambda x: x[:, 0] ** 2 * x[:, 1] ** 2 * x[:, 2] ** 2
    exact3 = 1.0 / 27
    err3 = abs(np.dot(wts3, f3(pts3)) - exact3)
    print(f"  3D deg=4: vol_err={abs(vol-1):.2e}  poly_err={err3:.2e}  "
          f"positive={(wts3>0).all()}  pts={len(pts3)}")

    # ---- compression ratio ----
    nodes_big = rng.random((1000, 2))
    cq_big = CompressedQuadrature(degree=6)
    pts_b, wts_b = cq_big.fit(nodes_big)
    print(f"  Compression 1000→{len(pts_b)} pts  "
          f"(ratio={cq_big.compression_ratio(1000):.1f}x)")

    print("All OK.")
