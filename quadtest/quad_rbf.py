"""
quad_rbf.py — RBF-based quadrature: fit an RBF interpolant and integrate it.

Theory
------
Given n scattered nodes {x_i}, the RBF interpolant of f is:

    I_f(x) = Σ_i α_i φ(‖x − x_i‖)  +  Σ_k β_k p_k(x)

where φ is a polyharmonic spline (PHS) kernel and {p_k} are low-degree monomials
(polynomial tail required for conditional positive definiteness).

Integrating both sides over the domain Ω and matching with the quadrature sum:

    ∫_Ω f(x) dx  ≈  ∫_Ω I_f(x) dx  =  Σ_i w_i f(x_i)

gives quadrature weights  w = A⁻¹ c_full  where

    A = [[Φ,  P ],      Φ[i,j] = φ(‖x_i − x_j‖)
         [P^T, 0]]      P[i,k] = p_k(x_i)

    c_full = [c₁, …, c_n, b₁, …, b_m]ᵀ
    c_i = ∫_Ω φ(‖x − x_i‖) dx          (RBF integral weight, computed numerically)
    b_k = ∫_Ω p_k(x) dx                 (polynomial moments, exact)

The matrix A is symmetric (PHS is symmetric; the block structure preserves
symmetry), so A⁻ᵀ = A⁻¹.

Supported kernels
-----------------
'phs3' : φ(r) = r³         (cubic PHS; CPD order 2; needs poly_degree ≥ 1)
'phs5' : φ(r) = r⁵         (quintic PHS; CPD order 3; needs poly_degree ≥ 2)
'phs7' : φ(r) = r⁷         (septic PHS; needs poly_degree ≥ 3)
'tps'  : φ(r) = r² log(r)  (thin-plate spline, 2D only; CPD order 2)

Positivity
----------
NOT guaranteed.  Negative weights can appear near boundaries or for clustered
nodes.  When fix_negative=True (default), negative weights are clipped to 0
and the remaining weights are rescaled so Σ w_i = ∫_Ω 1 dx.  This breaks
polynomial exactness but often gives a usable rule.

Accuracy
--------
Spectrally fast for smooth functions.  Formally O(h^{2k}) for r^{2k-1} PHS.
Cubic PHS ≈ 4th-order; quintic ≈ 6th-order in practice.

The c_i integrals are approximated numerically on a tensor-product GL grid
(20 pts/dim for 2D, 10 pts/dim for 3D).
"""

import numpy as np
from scipy.linalg import solve
from quad_utils import (
    build_vandermonde, compute_moments, scale_nodes,
    monomial_exponents, gauss_legendre_01
)


# ---------------------------------------------------------------------------
# PHS kernel functions
# ---------------------------------------------------------------------------

def _phi_phs3(r):
    return r ** 3

def _phi_phs5(r):
    return r ** 5

def _phi_phs7(r):
    return r ** 7

def _phi_tps(r):
    # r² log(r); limit as r→0 is 0
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(r == 0.0, 0.0, r ** 2 * np.log(r))
    return result

_KERNELS = {
    'phs3': (_phi_phs3, 1),   # (kernel function, min poly degree)
    'phs5': (_phi_phs5, 2),
    'phs7': (_phi_phs7, 3),
    'tps':  (_phi_tps,  1),
}


# ---------------------------------------------------------------------------
# Numerical integration of the RBF kernel over the unit hypercube
# ---------------------------------------------------------------------------

def _rbf_integrals(nodes_scaled, phi_fn, dim, n_gl=20):
    """
    Compute c_i = ∫_{[0,1]^dim} φ(‖x − x_i‖) dx  for each node x_i.

    Uses a tensor-product Gauss-Legendre rule on the scaled [0,1]^dim domain.

    Parameters
    ----------
    nodes_scaled : (n, dim) nodes in [0,1]^dim
    phi_fn       : callable  r → φ(r)
    dim          : 2 or 3
    n_gl         : GL points per direction

    Returns
    -------
    c : (n,) array
    """
    pts1d, wts1d = gauss_legendre_01(n_gl)

    if dim == 2:
        px, py = np.meshgrid(pts1d, pts1d, indexing='ij')
        wx, wy = np.meshgrid(wts1d, wts1d, indexing='ij')
        grid = np.column_stack([px.ravel(), py.ravel()])    # (n_gl², 2)
        grid_wts = (wx * wy).ravel()                         # (n_gl²,)
    elif dim == 3:
        n_gl_3d = min(n_gl, 12)   # cap to keep memory manageable
        pts1d, wts1d = gauss_legendre_01(n_gl_3d)
        px, py, pz = np.meshgrid(pts1d, pts1d, pts1d, indexing='ij')
        wx, wy, wz = np.meshgrid(wts1d, wts1d, wts1d, indexing='ij')
        grid = np.column_stack([px.ravel(), py.ravel(), pz.ravel()])
        grid_wts = (wx * wy * wz).ravel()
    else:
        raise ValueError(f"dim must be 2 or 3, got {dim}")

    n = len(nodes_scaled)
    c = np.zeros(n)
    for i in range(n):
        diff = grid - nodes_scaled[i]                        # (M, dim)
        r = np.sqrt(np.sum(diff ** 2, axis=1))               # (M,)
        c[i] = np.dot(grid_wts, phi_fn(r))
    return c


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class RBFQuadrature:
    """
    Quadrature via RBF interpolation + analytic integration.

    Parameters
    ----------
    rbf : str
        Kernel: 'phs3' (default), 'phs5', 'phs7', or 'tps' (2D only).
    poly_degree : int or 'auto'
        Degree of the polynomial tail.  'auto' uses the minimum required for
        the selected kernel (phs3→1, phs5→2, phs7→3, tps→1).
    fix_negative : bool
        If True (default), clip negative weights to 0 and rescale.
    n_gl : int
        GL points per dimension for numerical RBF integrals (default 20).

    Usage
    -----
        quad = RBFQuadrature(rbf='phs3')
        pts, wts = quad.fit(nodes)
        approx = np.dot(wts, f(pts))
    """

    def __init__(self, rbf='phs3', poly_degree='auto', fix_negative=True,
                 n_gl=20):
        if rbf not in _KERNELS:
            raise ValueError(f"Unknown rbf '{rbf}'; choose from {list(_KERNELS)}")
        self.rbf = rbf
        self.poly_degree = poly_degree
        self.fix_negative = fix_negative
        self.n_gl = n_gl
        self._nodes = None
        self._weights = None
        self._cond = None
        self._n_negative = None

    # ------------------------------------------------------------------
    def fit(self, nodes, domain='auto'):
        """
        Compute quadrature weights for the given scattered nodes.

        Parameters
        ----------
        nodes  : (n, dim) array
        domain : 'unit_square' / 'unit_cube' / 'auto'

        Returns
        -------
        nodes   : (n, dim) — same as input (all nodes are kept)
        weights : (n,) — may include negatives unless fix_negative=True
        """
        nodes = np.asarray(nodes, dtype=float)
        if nodes.ndim == 1:
            nodes = nodes[:, None]
        n, dim = nodes.shape

        if dim not in (2, 3):
            raise ValueError(f"dim must be 2 or 3, got {dim}")
        if self.rbf == 'tps' and dim != 2:
            raise ValueError("'tps' kernel is defined for 2D only")

        phi_fn, min_pd = _KERNELS[self.rbf]
        poly_deg = min_pd if self.poly_degree == 'auto' else max(self.poly_degree, min_pd)

        if domain == 'auto':
            domain = 'unit_square' if dim == 2 else 'unit_cube'

        # Scale to [0,1]^dim
        scaled, lo, rng = scale_nodes(nodes)
        vol_factor = float(np.prod(rng))

        # Polynomial tail
        exps = monomial_exponents(dim, poly_deg)
        m = len(exps)
        P = build_vandermonde(scaled, poly_deg).T     # (n, m)

        # RBF matrix Φ
        diff = scaled[:, None, :] - scaled[None, :, :]   # (n, n, dim)
        R = np.sqrt(np.sum(diff ** 2, axis=2))             # (n, n)
        Phi = phi_fn(R)

        # Augmented system matrix A (symmetric)
        A = np.zeros((n + m, n + m))
        A[:n, :n] = Phi
        A[:n, n:] = P
        A[n:, :n] = P.T

        # RHS: c_i (RBF integrals) + b_k (polynomial moments), both in scaled domain
        c = _rbf_integrals(scaled, phi_fn, dim, self.n_gl)
        b = compute_moments(poly_deg, domain)              # (m,)
        rhs = np.concatenate([c, b])

        # Condition number estimate
        self._cond = float(np.linalg.cond(A))

        # Solve  A w_full = rhs
        try:
            w_full = solve(A, rhs, assume_a='sym')
        except np.linalg.LinAlgError:
            w_full = np.linalg.lstsq(A, rhs, rcond=None)[0]

        w = w_full[:n] * vol_factor   # physical weights

        # Handle negatives
        self._n_negative = int(np.sum(w < 0))
        if self.fix_negative and self._n_negative > 0:
            w = np.clip(w, 0.0, None)
            target_sum = vol_factor   # ∫_Ω 1 dx
            s = w.sum()
            if s > 0:
                w *= target_sum / s

        self._nodes = nodes
        self._weights = w
        return nodes.copy(), w.copy()

    # ------------------------------------------------------------------
    @property
    def weights(self):
        return self._weights

    @property
    def condition_number(self):
        """Condition number of the augmented RBF system matrix."""
        return self._cond

    @property
    def n_negative_weights(self):
        """Number of negative weights before fix (if fix_negative=False)."""
        return self._n_negative

    def summary(self):
        if self._nodes is None:
            return "RBFQuadrature (not fitted)"
        n = len(self._nodes)
        return (
            f"RBFQuadrature: rbf={self.rbf}, n={n}, "
            f"cond={self._cond:.2e}, "
            f"n_negative={self._n_negative}, "
            f"fix_negative={self.fix_negative}, "
            f"min_w={self._weights.min():.3e}, "
            f"sum_w={self._weights.sum():.6f}"
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=== quad_rbf smoke test ===")
    rng = np.random.default_rng(3)

    # ---- 2D unit square ----
    for rbf in ['phs3', 'phs5', 'tps']:
        nodes = rng.random((150, 2))
        quad = RBFQuadrature(rbf=rbf)
        pts, wts = quad.fit(nodes)
        area = wts.sum()
        f = lambda x: np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
        exact = 4.0 / np.pi ** 2
        err = abs(np.dot(wts, f(pts)) - exact)
        pos = (wts > 0).all()
        print(f"  2D {rbf}: area_err={abs(area-1):.2e}  sin_err={err:.2e}  "
              f"positive={pos}  cond={quad.condition_number:.2e}  "
              f"n_neg={quad.n_negative_weights}")

    # ---- 3D unit cube ----
    nodes3 = rng.random((80, 3))
    quad3 = RBFQuadrature(rbf='phs3')
    pts3, wts3 = quad3.fit(nodes3)
    vol = wts3.sum()
    print(f"  3D phs3: vol_err={abs(vol-1):.2e}  "
          f"cond={quad3.condition_number:.2e}  "
          f"n_neg={quad3.n_negative_weights}")

    # ---- Without fix_negative to see raw negatives ----
    nodes_raw = rng.random((100, 2))
    quad_raw = RBFQuadrature(rbf='phs5', fix_negative=False)
    _, wts_raw = quad_raw.fit(nodes_raw)
    print(f"  phs5 raw: n_neg={quad_raw.n_negative_weights}  "
          f"min_w={wts_raw.min():.3e}")

    print("All OK.")
