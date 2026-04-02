"""
quad_glaubitz.py — Stable high-order quadrature for scattered data (Glaubitz 2020).

Implements two methods from:
    Glaubitz, J. (2020). Stable high order quadrature rules for scattered data
    and general weight functions. SIAM J. Numer. Anal., 58(4), 2144-2164.

Two quadrature rules
--------------------
LS-QR   (method='ls')  : Least-squares quadrature rule (Section 3).
         The minimum-norm weight vector that exactly satisfies all degree-d
         polynomial exactness conditions.  Stability proven for equidistant
         points; holds empirically for scattered points.  May have negative
         weights.

NNLS-QR (method='nnls') : Non-negative LS quadrature rule (Section 4).
         Weights non-negative (for ω≥0) or sign-consistent with the weight
         function (for general ω).  Exactness is approximate but improves as
         N grows relative to d.

Core idea — Discrete Orthogonal Polynomials (DOPs)
---------------------------------------------------
Both methods replace the ill-conditioned monomial Vandermonde with a basis
{φ_k}_{k=0}^d that is orthonormal w.r.t. the discrete inner product at the
scattered nodes:

    <φ_k, φ_l>_N = Σ_n φ_k(x_n) φ_l(x_n) = δ_{kl}

With DOPs, the normal equations of the LS problem collapse to the identity,
and the minimum-norm LS weight vector is (Eq. 3.13–3.14):

    w_n^LS = Σ_k φ_k(x_n) · I[φ_k]    ⟺    w = Φᵀ m

where
    Φ[k, n] = φ_k(x_n)   is the (m × N) DOP evaluation matrix,
    m[k]    = I[φ_k]      are the DOP moments ∫_Ω φ_k(x) ω(x) dx,
computed numerically by a tensor-product Gauss-Legendre rule (Section 3,
Remark 3.1; J = n_gl^dim GL points).

DOP construction (Eq. 3.10 + Section 3.2)
------------------------------------------
Starting from the monomial Vandermonde V (m × N):

    1. Thin QR of Vᵀ (N × m):  Vᵀ = Q R,   Q orthonormal (N × m), R upper-
       triangular (m × m).
    2. DOP values at nodes:  Φ = Qᵀ  →  Φ Φᵀ = I_m.
    3. DOP values at new points x:  φ(x) = (Rᵀ)⁻¹ v(x),
       where v(x) is the monomial vector at x.  Computed via
       scipy.linalg.solve_triangular for numerical stability.

Extension to 2D / 3D
---------------------
The paper is 1-D.  We extend it by using the total-degree monomial basis
(shared with the other methods in this repo) and a tensor-product GL rule
for moment integration.  The DOP orthogonalisation via QR is dimension-
agnostic.  (The paper notes multi-D extension as future work, but the
algebraic structure transfers directly.)

Sign-consistent NNLS for general weight functions (Section 4)
-------------------------------------------------------------
For a weight function ω that may have mixed signs, define the sign matrix
S = diag(sign(ω(x_n))).  The NNLS problem becomes (Eq. 4.7):

    min  ||Φ S u − m||₂   s.t. u ≥ 0,   then  w = S u.

For ω ≡ 1 (the default), S = I and this reduces to standard NNLS on Φ.
"""

import numpy as np
from scipy.linalg import qr, solve_triangular
from scipy.optimize import nnls
from quad_utils import (
    build_vandermonde, scale_nodes, monomial_exponents, gauss_legendre_01
)


# ---------------------------------------------------------------------------
# DOP construction helpers
# ---------------------------------------------------------------------------

def _build_dops(V):
    """
    Construct Discrete Orthogonal Polynomials from the monomial Vandermonde.

    Given V (m × n), computes Φ = (Rᵀ)⁻¹ V where Vᵀ = QR (thin QR), so
    that Φ Φᵀ = Qᵀ Q = I_m.

    Also returns R so that DOPs can be evaluated at arbitrary new points via
        φ(x) = solve_triangular(R.T, v(x), lower=True)

    Parameters
    ----------
    V : (m, n) monomial Vandermonde

    Returns
    -------
    Phi : (m, n)  DOP values at the n nodes;  Phi @ Phi.T ≈ I_m
    R   : (m, m)  upper-triangular QR factor
    """
    _, R = qr(V.T, mode='economic')   # Q is n×m but we only need R
    Phi = solve_triangular(R.T, V, lower=True)   # (m, n)
    return Phi, R


def _dop_at_points(R, V_pts):
    """
    Evaluate DOPs at new points given their monomial Vandermonde V_pts.

    Parameters
    ----------
    R     : (m, m) upper-triangular factor from _build_dops
    V_pts : (m, J) monomial Vandermonde at J new points

    Returns
    -------
    Phi_pts : (m, J) DOP values at the J new points
    """
    return solve_triangular(R.T, V_pts, lower=True)


def _gl_grid(dim, n_gl):
    """
    Build a tensor-product Gauss-Legendre grid on [0,1]^dim.

    Returns
    -------
    gl_pts : (J, dim) array of GL points
    gl_wts : (J,) array of GL weights  (sum = 1)
    """
    pts1d, wts1d = gauss_legendre_01(n_gl)
    if dim == 2:
        px, py = np.meshgrid(pts1d, pts1d, indexing='ij')
        wx, wy = np.meshgrid(wts1d, wts1d, indexing='ij')
        gl_pts = np.column_stack([px.ravel(), py.ravel()])
        gl_wts = (wx * wy).ravel()
    else:  # dim == 3
        px, py, pz = np.meshgrid(pts1d, pts1d, pts1d, indexing='ij')
        wx, wy, wz = np.meshgrid(wts1d, wts1d, wts1d, indexing='ij')
        gl_pts = np.column_stack([px.ravel(), py.ravel(), pz.ravel()])
        gl_wts = (wx * wy * wz).ravel()
    return gl_pts, gl_wts


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class GlaubitzQuadrature:
    """
    Stable high-order quadrature for scattered nodes via DOPs (Glaubitz 2020).

    Parameters
    ----------
    degree : int
        Target polynomial degree of exactness (default 6).
    method : str
        'ls'   — LS-QR: minimum-norm weights, always exact to `degree`
                 (subject to numerical precision), may include negatives.
        'nnls' — NNLS-QR: non-negative weights (for uniform ω), approximately
                 exact; exactness improves as N grows relative to d.
    weight_fn : callable or None
        Weight function ω for the integral ∫ f(x) ω(x) dx.
        Signature: weight_fn(X) → (J,) where X is (J, dim) in physical coords.
        None (default) uses ω ≡ 1 (standard Lebesgue measure on [0,1]^dim).
    n_gl : int
        GL points per dimension for numerical moment computation (default 20).
        Higher values improve moment accuracy for non-smooth weight functions.
        Capped at 10 for 3-D to limit memory.
    tol_rank : float
        If |R[k,k]| / |R[0,0]| drops below this threshold the effective rank
        (and therefore effective degree) is trimmed (default 1e-12).

    Usage
    -----
        quad = GlaubitzQuadrature(degree=8, method='nnls')
        pts, wts = quad.fit(nodes)
        approx = wts @ f(pts)
    """

    def __init__(self, degree=6, method='nnls', weight_fn=None,
                 n_gl=20, tol_rank=1e-12):
        if method not in ('ls', 'nnls'):
            raise ValueError(f"method must be 'ls' or 'nnls', got '{method}'")
        self.degree = degree
        self.method = method
        self.weight_fn = weight_fn
        self.n_gl = n_gl
        self.tol_rank = tol_rank
        # populated after fit()
        self._nodes = None
        self._weights = None
        self._effective_degree = None
        self._residual = None
        self._n_negative = None
        self._cond_ratio = None

    # ------------------------------------------------------------------
    def fit(self, nodes):
        """
        Compute Glaubitz quadrature weights for the given scattered nodes.

        Parameters
        ----------
        nodes : (n, dim) array,  dim = 2 or 3

        Returns
        -------
        nodes   : (n, dim) — same as input (all nodes kept)
        weights : (n,) — LS weights (may be negative) or NNLS (non-negative
                  for uniform ω)
        """
        nodes = np.asarray(nodes, dtype=float)
        if nodes.ndim == 1:
            nodes = nodes[:, None]
        n, dim = nodes.shape

        if dim not in (2, 3):
            raise ValueError(f"dim must be 2 or 3, got {dim}")

        # --- scale to [0, 1]^dim ---
        scaled, lo, rng = scale_nodes(nodes)
        vol_factor = float(np.prod(rng))

        # --- find highest feasible degree ---
        degree = self.degree
        while degree >= 1:
            m = len(monomial_exponents(dim, degree))
            if n >= m:
                break
            degree -= 1
        if degree < 1:
            raise ValueError(
                f"Need at least {len(monomial_exponents(dim, 1))} nodes for "
                f"degree-1 in {dim}D; got {n}."
            )

        # --- monomial Vandermonde at scattered nodes: V (m × n) ---
        V = build_vandermonde(scaled, degree)   # (m, n)

        # --- build DOPs via thin QR of Vᵀ ---
        Phi, R = _build_dops(V)   # Phi (m, n), R (m, m)

        # --- check / trim effective rank ---
        diag_R = np.abs(np.diag(R))
        if diag_R[0] < 1e-14:
            raise RuntimeError(
                "Vandermonde matrix is singular; check for duplicate nodes."
            )
        self._cond_ratio = float(diag_R[-1] / diag_R[0])
        if self._cond_ratio < self.tol_rank:
            eff_rank = int(np.sum(diag_R > self.tol_rank * diag_R[0]))
            eff_rank = max(1, eff_rank)
            Phi = Phi[:eff_rank, :]
            R = R[:eff_rank, :eff_rank]

        self._effective_degree = degree

        # --- GL grid on [0, 1]^dim ---
        n_gl = self.n_gl if dim == 2 else min(self.n_gl, 10)
        gl_pts, gl_wts = _gl_grid(dim, n_gl)   # (J, dim), (J,)

        # --- DOP values at GL points: Φ_GL (m × J) ---
        V_gl = build_vandermonde(gl_pts, degree)   # (m, J)
        Phi_gl = _dop_at_points(R, V_gl)            # (m, J)

        # --- weight function at GL points (physical coords) ---
        if self.weight_fn is None:
            omega_gl = np.ones(len(gl_wts))
        else:
            gl_pts_phys = gl_pts * rng[np.newaxis, :] + lo[np.newaxis, :]
            omega_gl = np.asarray(self.weight_fn(gl_pts_phys), dtype=float)

        # --- DOP moments: m_k = Σ_j w_j^GL ω_j φ_k(x_j^GL)  (Eq. 3.17) ---
        m_vec = Phi_gl @ (gl_wts * omega_gl)   # (m,)

        # --- compute weights ---
        if self.method == 'ls':
            # minimum-norm LS solution: w = Φᵀ m  (Eq. 3.13)
            w = Phi.T @ m_vec         # (n,)
            self._residual = 0.0      # LS solution satisfies exactness exactly
            self._n_negative = int(np.sum(w < 0))

        else:  # 'nnls'
            if self.weight_fn is None:
                # ω ≡ 1: plain NNLS on Φ  (Eq. 4.1 with S = I)
                u, self._residual = nnls(Phi, m_vec)
                w = u
            else:
                # general ω: sign-consistent NNLS  (Eq. 4.7)
                nodes_phys = scaled * rng[np.newaxis, :] + lo[np.newaxis, :]
                sign_n = np.sign(
                    np.asarray(self.weight_fn(nodes_phys), dtype=float)
                )
                sign_n[sign_n == 0] = 1.0
                # minimize ||Φ S u − m||  s.t. u ≥ 0
                A_signed = Phi * sign_n[np.newaxis, :]   # (m, n)
                u, self._residual = nnls(A_signed, m_vec)
                w = sign_n * u
            self._n_negative = int(np.sum(w < 0))

        w_phys = w * vol_factor

        self._nodes = nodes
        self._weights = w_phys
        return nodes.copy(), w_phys.copy()

    # ------------------------------------------------------------------
    @property
    def effective_degree(self):
        """Actual polynomial degree used (may be < requested if n is small)."""
        return self._effective_degree

    @property
    def moment_residual(self):
        """||Φ w − m||₂ in scaled coordinates (0 for LS; small for NNLS)."""
        return self._residual

    @property
    def n_negative_weights(self):
        """Number of negative weights (0 for NNLS with uniform ω ≡ 1)."""
        return self._n_negative

    @property
    def dop_condition_ratio(self):
        """
        |R[-1,-1]| / |R[0,0]| from the QR step.
        Measures how well-posed the DOP construction is (closer to 1 is better).
        """
        return self._cond_ratio

    def summary(self):
        if self._nodes is None:
            return "GlaubitzQuadrature (not fitted)"
        return (
            f"GlaubitzQuadrature: method={self.method}, "
            f"degree={self._effective_degree}, "
            f"n={len(self._nodes)}, "
            f"dop_cond={self._cond_ratio:.2e}, "
            f"residual={self._residual:.2e}, "
            f"n_neg={self._n_negative}, "
            f"min_w={self._weights.min():.3e}, "
            f"sum_w={self._weights.sum():.6f}"
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=== quad_glaubitz smoke test ===")
    rng = np.random.default_rng(42)

    # ---- 2D unit square: both methods, several degrees ----
    for method in ['ls', 'nnls']:
        for deg in [4, 6, 8]:
            nodes = rng.random((400, 2))
            quad = GlaubitzQuadrature(degree=deg, method=method)
            pts, wts = quad.fit(nodes)

            area = wts.sum()
            f = lambda x: np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
            exact = 4.0 / np.pi ** 2
            err = abs(np.dot(wts, f(pts)) - exact)
            pos = (wts >= 0).all()

            print(f"  2D {method} deg={deg}: "
                  f"area_err={abs(area - 1):.2e}  sin_err={err:.2e}  "
                  f"positive={pos}  cond={quad.dop_condition_ratio:.2e}  "
                  f"n_neg={quad.n_negative_weights}  res={quad.moment_residual:.2e}")

    # ---- 2D: polynomial exactness check for LS ----
    print()
    nodes = rng.random((300, 2))
    quad_ls = GlaubitzQuadrature(degree=6, method='ls')
    pts_ls, wts_ls = quad_ls.fit(nodes)
    f_poly = lambda x: x[:, 0] ** 4 * x[:, 1] ** 3
    exact_poly = 1.0 / 5 * 1.0 / 4
    err_poly = abs(np.dot(wts_ls, f_poly(pts_ls)) - exact_poly)
    print(f"  LS poly exactness (x^4 y^3): err={err_poly:.2e}  "
          f"(should be ~machine-eps for deg<=6 rule with 300 nodes)")

    # ---- 3D unit cube ----
    print()
    for method in ['ls', 'nnls']:
        nodes3 = rng.random((200, 3))
        quad3 = GlaubitzQuadrature(degree=4, method=method)
        pts3, wts3 = quad3.fit(nodes3)
        vol = wts3.sum()
        f3 = lambda x: x[:, 0] ** 2 * x[:, 1] ** 2 * x[:, 2] ** 2
        exact3 = 1.0 / 27
        err3 = abs(np.dot(wts3, f3(pts3)) - exact3)
        print(f"  3D {method} deg=4: "
              f"vol_err={abs(vol - 1):.2e}  poly_err={err3:.2e}  "
              f"positive={(wts3 >= 0).all()}")

    # ---- General weight function (ω = cos(2π x₁)) on 2D ----
    print()
    omega = lambda X: np.cos(2 * np.pi * X[:, 0])
    nodes_w = rng.random((400, 2))
    quad_w = GlaubitzQuadrature(degree=6, method='nnls', weight_fn=omega)
    pts_w, wts_w = quad_w.fit(nodes_w)
    # exact: ∫₀¹∫₀¹ sin(πx)sin(πy) cos(2πx) dx dy
    # = (∫₀¹ sin(πx)cos(2πx) dx) · (∫₀¹ sin(πy) dy)
    # ∫₀¹ sin(πx)cos(2πx) dx = -4/(3π²) via trig identity  →  × 2/π
    exact_w = (-4.0 / (3 * np.pi ** 2)) * (2.0 / np.pi)
    f_w = lambda x: np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
    err_w = abs(np.dot(wts_w, f_w(pts_w)) - exact_w)
    sign_ok = np.all(
        np.sign(wts_w) == np.sign(omega(nodes_w)) + (omega(nodes_w) == 0)
    )
    print(f"  2D NNLS with ω=cos(2πx): err={err_w:.2e}  "
          f"sign_consistent={sign_ok}")

    print("\nAll OK.")
