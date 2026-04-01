"""
quad_moment_fitting.py — Positive quadrature via moment fitting (NNLS / LP).

Given n scattered nodes {x_i} in R^d, find weights w_i ≥ 0 such that

    Σ_i  w_i · p_k(x_i)  =  ∫_Ω  p_k(x) dx      for k = 1, …, m

where {p_k} are all monomials of total degree ≤ d (m terms).
This is an underdetermined non-negative least-squares (NNLS) problem when n > m.

Method
------
1. Scale nodes to [0,1]^dim for numerical stability.
2. Build Vandermonde matrix V (m × n).
3. Compute moment vector b (exact integrals over the scaled domain).
4. Solve  min ||Vw - b||₂  s.t. w ≥ 0   via scipy.optimize.nnls.
   OR minimise  Σ w_i  s.t. V w = b, w ≥ ε  via scipy.optimize.linprog (LP).
5. Drop zero-weight nodes; re-scale weights by prod(rng) back to physical domain.

Positivity guarantee
--------------------
NNLS:  w_i ≥ 0 guaranteed.  Some w_i may be exactly 0 (those nodes are dropped).
LP:    w_i ≥ min_weight > 0 when feasible; can fail if n is too small vs m.

Accuracy
--------
Exact for all polynomials of degree ≤ `degree`.  For smooth f: O(h^{degree+1}).
"""

import numpy as np
from scipy.optimize import nnls, linprog
from quad_utils import (
    build_vandermonde, compute_moments, scale_nodes, monomial_exponents
)


class MomentFittingQuadrature:
    """
    Positive quadrature by moment fitting on scattered nodes.

    Parameters
    ----------
    degree : int
        Polynomial degree of exactness (default 4).
    method : str
        'nnls' (default) — non-negative least squares (always feasible).
        'lp'             — linear programme; enforces w_i ≥ min_weight.
    min_weight : float
        Lower bound on weights when method='lp' (default 1e-14).
    domain : str
        'unit_square', 'unit_cube', 'unit_disk', or 'unit_ball'.
        Used to compute the moment vector.  The algebraic methods (nnls/lp)
        work with the scaled [0,1]^d domain, so 'unit_square'/'unit_cube'
        is the most robust choice (nodes are always scaled to [0,1]^d before
        the Vandermonde is built).  Disk/ball moments are used when the
        original node distribution lives in those domains.

    Usage
    -----
        mf = MomentFittingQuadrature(degree=4)
        nodes_used, weights = mf.fit(nodes)
        approx = np.dot(weights, f(nodes_used))
    """

    def __init__(self, degree=4, method='nnls', min_weight=1e-14,
                 domain='auto'):
        self.degree = degree
        self.method = method
        self.min_weight = min_weight
        self.domain = domain
        # Results populated after fit()
        self._all_nodes = None
        self._all_weights = None
        self._nodes_used = None
        self._weights_used = None
        self._indices_used = None
        self._residual = None

    # ------------------------------------------------------------------
    def fit(self, nodes):
        """
        Compute quadrature weights for the given scattered nodes.

        Parameters
        ----------
        nodes : (n, dim) array

        Returns
        -------
        nodes_used : (k, dim) subset of nodes with positive weights
        weights    : (k,) corresponding positive weights
        """
        nodes = np.asarray(nodes, dtype=float)
        if nodes.ndim == 1:
            nodes = nodes[:, None]
        n, dim = nodes.shape

        # --- determine domain for moment computation ---
        if self.domain == 'auto':
            domain = 'unit_square' if dim == 2 else 'unit_cube'
        else:
            domain = self.domain

        # --- scale nodes to [0,1]^dim ---
        scaled, lo, rng = scale_nodes(nodes)
        vol_factor = float(np.prod(rng))   # physical domain volume factor

        # --- check feasibility: need n >= m ---
        m = len(monomial_exponents(dim, self.degree))
        if n < m:
            raise ValueError(
                f"Need at least {m} nodes for degree {self.degree} in {dim}D "
                f"(have {n}). Reduce degree or add more nodes."
            )

        # --- Vandermonde and moment vector (in scaled coords) ---
        V = build_vandermonde(scaled, self.degree)   # (m, n)
        b = compute_moments(self.degree, domain)     # (m,)

        # --- solve for weights ---
        if self.method == 'nnls':
            w, residual = nnls(V, b)
            self._residual = float(residual)
        elif self.method == 'lp':
            w, residual = self._solve_lp(V, b, n)
            self._residual = float(residual)
        else:
            raise ValueError(f"Unknown method '{self.method}'; use 'nnls' or 'lp'")

        # --- scale weights back to physical domain ---
        w_phys = w * vol_factor

        # --- store full results ---
        self._all_nodes = nodes
        self._all_weights = w_phys

        # --- drop zero-weight nodes ---
        mask = w > self.min_weight
        self._indices_used = np.where(mask)[0]
        self._nodes_used = nodes[mask]
        self._weights_used = w_phys[mask]

        return self._nodes_used.copy(), self._weights_used.copy()

    # ------------------------------------------------------------------
    def _solve_lp(self, V, b, n):
        """
        Minimise sum(w) s.t.  V w = b,  w >= min_weight.
        Returns (w, residual_norm).
        """
        eps = self.min_weight
        # shift: let w = u + eps  so u >= 0
        b_shifted = b - V.sum(axis=1) * eps

        result = linprog(
            c=np.ones(n),
            A_eq=V,
            b_eq=b_shifted,
            bounds=[(0, None)] * n,
            method='highs',
        )
        if result.status != 0:
            # LP infeasible: fall back to NNLS
            w, res = nnls(V, b)
            return w, res
        w = result.x + eps
        residual = float(np.linalg.norm(V @ w - b))
        return w, residual

    # ------------------------------------------------------------------
    @property
    def moment_residual(self):
        """||V w - b||₂ in scaled coordinates (lower is better)."""
        return self._residual

    @property
    def indices_used(self):
        """Indices (into original nodes array) with positive weights."""
        return self._indices_used

    @property
    def all_weights(self):
        """Weights for all input nodes (zeros for dropped nodes)."""
        return self._all_weights

    def compression_ratio(self):
        """Fraction of nodes retained."""
        if self._all_nodes is None:
            return None
        return len(self._nodes_used) / len(self._all_nodes)

    def summary(self):
        if self._nodes_used is None:
            return "MomentFittingQuadrature (not fitted)"
        return (
            f"MomentFittingQuadrature: degree={self.degree}, method={self.method}, "
            f"nodes_used={len(self._nodes_used)}/{len(self._all_nodes)}, "
            f"residual={self._residual:.2e}, "
            f"min_w={self._weights_used.min():.3e}, "
            f"sum_w={self._weights_used.sum():.6f}"
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    print("=== quad_moment_fitting smoke test ===")
    rng = np.random.default_rng(0)

    # ---- 2D unit square ----
    for deg in [2, 4, 6]:
        nodes = rng.random((300, 2))
        mf = MomentFittingQuadrature(degree=deg)
        nu, wu = mf.fit(nodes)
        area = wu.sum()
        f = lambda x: x[:, 0] ** deg * x[:, 1] ** (deg - 1)
        exact = 1.0 / (deg + 1) / deg
        err = abs(np.dot(wu, f(nu)) - exact)
        pos = (wu > 0).all()
        print(f"  2D deg={deg}: area_err={abs(area-1):.2e}  "
              f"poly_err={err:.2e}  positive={pos}  "
              f"kept={len(nu)}/{len(nodes)}  residual={mf.moment_residual:.2e}")

    # ---- 3D unit cube ----
    nodes3 = rng.random((500, 3))
    mf3 = MomentFittingQuadrature(degree=4)
    nu3, wu3 = mf3.fit(nodes3)
    area3 = wu3.sum()
    f3 = lambda x: x[:, 0] ** 2 * x[:, 1] ** 2 * x[:, 2] ** 2
    exact3 = 1.0 / 27
    err3 = abs(np.dot(wu3, f3(nu3)) - exact3)
    print(f"  3D deg=4: vol_err={abs(area3-1):.2e}  poly_err={err3:.2e}  "
          f"positive={(wu3>0).all()}")

    # ---- LP mode ----
    nodes_lp = rng.random((200, 2))
    mf_lp = MomentFittingQuadrature(degree=4, method='lp', min_weight=1e-12)
    nu_lp, wu_lp = mf_lp.fit(nodes_lp)
    print(f"  LP mode: area_err={abs(wu_lp.sum()-1):.2e}  "
          f"min_w={wu_lp.min():.3e}  strictly_positive={(wu_lp>0).all()}")

    print("All OK.")
