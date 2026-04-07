"""
NNLS.py — Non-Negative Least Squares (NNLS-QR) quadrature for 2D scattered data.

Implements the NNLS-QR method from:
    Glaubitz, J. (2020). Stable high order quadrature rules for scattered data
    and general weight functions. SIAM J. Numer. Anal., 58(4), 2144–2164.

Method overview
---------------
Given N scattered nodes {x_i} in a 2D domain, the NNLS-QR rule computes
non-negative weights {w_i} approximating ∫_Ω f(x) dx ≈ Σ_i w_i f(x_i).

The key steps are:

1. Monomial Vandermonde V (m×N):  V[k,i] = p_k(x_i),  {p_k} = monomials up to degree d
2. Discrete Orthogonal Polynomials (DOPs) via thin QR of Vᵀ = QR:
       Phi = solve(Rᵀ, V)   →   Phi Phiᵀ = I_m
3. DOP moments via tensor-product Gauss-Legendre on [0,1]²:
       m_k = Σ_j w_j^GL φ_k(x_j^GL)
4. NNLS solve:  min ||Phi u - m||₂  s.t. u ≥ 0   (scipy.optimize.nnls)
5. Physical weights:  w = u · vol_factor   (vol_factor = area of original domain)

DOPs replace the ill-conditioned raw monomial basis so the NNLS problem is
well-scaled. Non-negativity guarantees stability; exactness is approximate
but improves as N grows relative to d.

Usage
-----
    from NNLS import NNLSQuadrature
    import numpy as np

    nodes = np.random.default_rng(0).random((400, 2))   # scattered in [0,1]²
    quad  = NNLSQuadrature(degree=6)
    pts, wts = quad.fit(nodes)
    approx = wts @ np.sin(np.pi * pts[:, 0]) * np.sin(np.pi * pts[:, 1])

This file is self-contained: no local quad_* imports are required.
"""

import numpy as np
from scipy.linalg import qr, solve_triangular
from scipy.optimize import nnls


# ── Inlined utilities ──────────────────────────────────────────────────────

def monomial_exponents(dim, degree):
    """
    List of multi-index tuples (a_0,...,a_{dim-1}) with sum ≤ degree,
    in graded lexicographic order.  Supports dim = 1, 2, 3.
    """
    exps = []
    if dim == 1:
        for d in range(degree + 1):
            exps.append((d,))
    elif dim == 2:
        for total in range(degree + 1):
            for a in range(total + 1):
                exps.append((a, total - a))
    elif dim == 3:
        for total in range(degree + 1):
            for a in range(total + 1):
                for b in range(total - a + 1):
                    exps.append((a, b, total - a - b))
    else:
        raise ValueError(f"dim must be 1, 2, or 3; got {dim}")
    return exps


def build_vandermonde(nodes, degree):
    """
    Monomial Vandermonde V of shape (m, n): V[k, i] = p_k(nodes[i]).

    Parameters
    ----------
    nodes  : (n, dim) array  — must be in [0,1]^dim for numerical stability
    degree : int

    Returns
    -------
    V : (m, n) float array,  m = len(monomial_exponents(dim, degree))
    """
    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim == 1:
        nodes = nodes[:, None]
    n, dim = nodes.shape
    exps = monomial_exponents(dim, degree)
    V = np.ones((len(exps), n), dtype=float)
    for k, alpha in enumerate(exps):
        for j, aj in enumerate(alpha):
            if aj > 0:
                V[k] *= nodes[:, j] ** aj
    return V


def gauss_legendre_01(n):
    """n-point Gauss-Legendre rule on [0, 1].  Weights sum to 1."""
    pts, wts = np.polynomial.legendre.leggauss(n)
    return (pts + 1.0) / 2.0, wts / 2.0


def scale_nodes(nodes):
    """
    Scale nodes to [0, 1]^dim.

    Returns
    -------
    scaled : (n, dim) in [0, 1]^dim
    lo     : (dim,) coordinate minima
    rng    : (dim,) coordinate ranges (max - min)

    Recovery: nodes_physical = scaled * rng + lo
    Weight scaling: w_physical = w_scaled * prod(rng)
    """
    nodes = np.asarray(nodes, dtype=float)
    lo  = nodes.min(axis=0)
    rng = nodes.max(axis=0) - lo
    rng[rng == 0.0] = 1.0          # guard against degenerate dimensions
    return (nodes - lo) / rng, lo, rng


# ── DOP helpers ────────────────────────────────────────────────────────────

def _build_dops(V):
    """
    Construct Discrete Orthogonal Polynomials (DOPs) from the Vandermonde V.

    Thin QR of Vᵀ (n × m) gives R (m × m); then
        Phi = solve(Rᵀ, V)   satisfies   Phi Phiᵀ ≈ I_m.

    Parameters
    ----------
    V : (m, n) float array

    Returns
    -------
    Phi : (m, n)  DOP values at the n nodes
    R   : (m, m)  upper-triangular QR factor (for evaluating at new points)
    """
    _, R = qr(V.T, mode='economic')
    Phi  = solve_triangular(R.T, V, lower=True)
    return Phi, R


def _gl_grid_2d(n_gl):
    """
    Tensor-product Gauss-Legendre grid on [0, 1]².

    Returns
    -------
    gl_pts : (J, 2)  GL points,   J = n_gl²
    gl_wts : (J,)    GL weights,  sum = 1
    """
    pts1d, wts1d = gauss_legendre_01(n_gl)
    px, py = np.meshgrid(pts1d, pts1d, indexing='ij')
    wx, wy = np.meshgrid(wts1d, wts1d, indexing='ij')
    gl_pts = np.column_stack([px.ravel(), py.ravel()])
    gl_wts = (wx * wy).ravel()
    return gl_pts, gl_wts


# ── NNLSQuadrature ─────────────────────────────────────────────────────────

class NNLSQuadrature:
    """
    Non-negative quadrature rule for 2D scattered nodes via DOPs (Glaubitz 2020).

    Parameters
    ----------
    degree : int
        Target polynomial degree of exactness (default 6).
        The rule is approximately exact for polynomials up to this degree;
        exactness improves as N grows relative to d.
    n_gl : int
        GL points per dimension used for numerical moment integration (default 20).
        n_gl = 20 gives 400 GL points on [0,1]², more than enough for degree ≤ 10.
    tol_rank : float
        Threshold for trimming near-zero diagonal entries of R from the QR step.
        If |R[k,k]| / |R[0,0]| < tol_rank, the effective degree is reduced.

    After calling fit(), diagnostic properties are available:
        .effective_degree     — actual degree used (may be < requested)
        .moment_residual      — ||Phi u - m||₂ from the NNLS solve
        .n_negative_weights   — number of negative weights (should be 0)
        .dop_condition_ratio  — |R[-1,-1]| / |R[0,0]|  (condition indicator)
    """

    def __init__(self, degree=6, n_gl=20, tol_rank=1e-12):
        self.degree   = degree
        self.n_gl     = n_gl
        self.tol_rank = tol_rank
        # populated by fit()
        self._nodes            = None
        self._weights          = None
        self._effective_degree = None
        self._residual         = None
        self._n_negative       = None
        self._cond_ratio       = None

    def fit(self, nodes):
        """
        Compute NNLS-QR quadrature weights for the given scattered nodes.

        Parameters
        ----------
        nodes : (N, 2) array of 2D node coordinates

        Returns
        -------
        nodes   : (N, 2) copy of the input nodes
        weights : (N,)   non-negative quadrature weights
        """
        nodes = np.asarray(nodes, dtype=float)
        if nodes.ndim != 2 or nodes.shape[1] != 2:
            raise ValueError(f"nodes must have shape (N, 2); got {nodes.shape}")
        n = nodes.shape[0]

        # 1. Scale to [0,1]²
        scaled, _, rng = scale_nodes(nodes)
        vol_factor = float(rng[0] * rng[1])

        # 2. Find highest feasible degree (need N ≥ m_d)
        degree = self.degree
        while degree >= 1:
            m = len(monomial_exponents(2, degree))
            if n >= m:
                break
            degree -= 1
        if degree < 1:
            raise ValueError(
                f"Need at least {len(monomial_exponents(2, 1))} nodes for "
                f"degree-1 in 2D; got {n}."
            )

        # 3. Monomial Vandermonde at scattered nodes
        V = build_vandermonde(scaled, degree)           # (m, n)

        # 4. Build DOPs via thin QR
        Phi, R = _build_dops(V)                         # (m, n), (m, m)

        # 5. Effective-rank check / trim
        diag_R = np.abs(np.diag(R))
        if diag_R[0] < 1e-14:
            raise RuntimeError("Vandermonde is singular; check for duplicate nodes.")
        self._cond_ratio = float(diag_R[-1] / diag_R[0])
        if self._cond_ratio < self.tol_rank:
            eff_rank = max(1, int(np.sum(diag_R > self.tol_rank * diag_R[0])))
            Phi = Phi[:eff_rank, :]
            R   = R[:eff_rank, :eff_rank]

        self._effective_degree = degree

        # 6. GL grid on [0,1]²
        gl_pts, gl_wts = _gl_grid_2d(self.n_gl)        # (J, 2), (J,)

        # 7. DOP values at GL points
        V_gl   = build_vandermonde(gl_pts, degree)      # (m, J)
        Phi_gl = solve_triangular(R.T, V_gl, lower=True)  # (m, J)

        # 8. DOP moments:  m_k = ∫_{[0,1]²} φ_k(x) dx  ≈  Phi_gl @ gl_wts
        m_vec = Phi_gl @ gl_wts                         # (m,)

        # 9. NNLS solve:  min ||Phi u - m||₂  s.t. u ≥ 0
        u, self._residual = nnls(Phi, m_vec)

        # 10. Scale weights back to physical domain
        w_phys = u * vol_factor
        self._n_negative = int(np.sum(w_phys < 0))

        self._nodes   = nodes
        self._weights = w_phys
        return nodes.copy(), w_phys.copy()

    # -- Diagnostic properties ---------------------------------------------

    @property
    def effective_degree(self):
        """Polynomial degree actually used (may be < requested if N is small)."""
        return self._effective_degree

    @property
    def moment_residual(self):
        """||Phi u - m||₂ from the NNLS solve (smaller is better)."""
        return self._residual

    @property
    def n_negative_weights(self):
        """Number of negative weights (should be 0 for NNLS with ω ≡ 1)."""
        return self._n_negative

    @property
    def dop_condition_ratio(self):
        """|R[-1,-1]| / |R[0,0]| — measures DOP conditioning (closer to 1 is better)."""
        return self._cond_ratio

    def summary(self):
        if self._nodes is None:
            return "NNLSQuadrature (not fitted)"
        return (
            f"NNLSQuadrature: degree={self._effective_degree}, "
            f"n={len(self._nodes)}, "
            f"dop_cond={self._cond_ratio:.2e}, "
            f"residual={self._residual:.2e}, "
            f"n_neg={self._n_negative}, "
            f"min_w={self._weights.min():.3e}, "
            f"sum_w={self._weights.sum():.6f}"
        )


# ── Test suite ─────────────────────────────────────────────────────────────

def _gauss_bump_exact():
    """High-accuracy reference value of ∫∫ exp(-10((x-½)²+(y-½)²)) dx dy over [0,1]²."""
    pts, wts = np.polynomial.legendre.leggauss(50)
    pts = (pts + 1) / 2
    wts = wts / 2
    X, Y = np.meshgrid(pts, pts, indexing='ij')
    W    = np.outer(wts, wts)
    return float(np.sum(W * np.exp(-10 * ((X - 0.5)**2 + (Y - 0.5)**2))))


def _test_integrands():
    """Dict of {name: (func, exact_value)} for integrands over [0,1]²."""
    return {
        'constant':    (lambda x: np.ones(len(x)),
                        1.0),
        'linear':      (lambda x: x[:, 0] + x[:, 1],
                        1.0),
        'poly_x4y3':   (lambda x: x[:, 0]**4 * x[:, 1]**3,
                        1.0/5 * 1.0/4),
        'sin_product': (lambda x: np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1]),
                        4.0 / np.pi**2),
        'gauss_bump':  (lambda x: np.exp(-10 * ((x[:, 0] - 0.5)**2
                                                 + (x[:, 1] - 0.5)**2)),
                        _gauss_bump_exact()),
    }


def test_basic_integrands(degree=6, n=400, seed=42, tol=1e-2):
    """
    Test each integrand and print PASS/FAIL. Returns True if all pass.

    Note: NNLS-QR is *approximately* polynomial-exact, so tol is intentionally
    loose (1e-2 default).  Errors shrink as N grows — see test_convergence().
    """
    rng       = np.random.default_rng(seed)
    nodes     = rng.random((n, 2))
    quad      = NNLSQuadrature(degree=degree)
    pts, wts  = quad.fit(nodes)
    integrands = _test_integrands()

    print(f"\n--- Basic integrands  (degree={degree}, n={n}) ---")
    print(f"  {'Integrand':<14}  {'Approx':>12}  {'Exact':>12}  {'Error':>10}  Status")
    print(f"  {'-'*60}")
    all_pass = True
    for name, (f, exact) in integrands.items():
        approx = float(wts @ f(pts))
        err    = abs(approx - exact)
        ok     = err < tol
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {name:<14}  {approx:>12.7f}  {exact:>12.7f}  {err:>10.2e}  {status}")
    return all_pass


def test_positivity(degree=6, n=400, seed=42):
    """Check all weights are non-negative and sum ≈ area. Returns True if all positive."""
    rng      = np.random.default_rng(seed)
    nodes    = rng.random((n, 2))
    quad     = NNLSQuadrature(degree=degree)
    _, wts = quad.fit(nodes)

    all_pos  = bool(np.all(wts >= 0))
    sum_err  = abs(wts.sum() - 1.0)
    status   = "PASS" if all_pos else "FAIL"
    print(f"\n--- Positivity  (degree={degree}, n={n}) ---")
    print(f"  min(w)={wts.min():.3e}  max(w)={wts.max():.3e}  "
          f"sum(w)={wts.sum():.6f}  sum_err={sum_err:.2e}  {status}")
    return all_pos


def test_degree_of_exactness(n=500, max_degree=10, seed=7, tol=1e-8):
    """
    Find the highest polynomial degree for which the rule is exactly correct.
    Returns the degree of exactness found.
    """
    rng      = np.random.default_rng(seed)
    nodes    = rng.random((n, 2))
    quad     = NNLSQuadrature(degree=6)
    pts, wts = quad.fit(nodes)

    doe = -1
    for deg in range(max_degree + 1):
        new_exps = [e for e in monomial_exponents(2, deg) if sum(e) == deg]
        passed   = True
        for alpha in new_exps:
            f     = lambda x, a=alpha: np.prod(x ** np.array(a), axis=1)
            exact = 1.0 / np.prod([a + 1 for a in alpha])
            err   = abs(float(wts @ f(pts)) - exact)
            if err > tol:
                passed = False
                break
        if passed:
            doe = deg
        else:
            break

    req    = quad.effective_degree
    status = "PASS" if doe >= req - 1 else "FAIL"
    print(f"\n--- Degree of exactness  (n={n}) ---")
    print(f"  Requested degree={req}  Achieved degree_of_exactness={doe}  {status}")
    return doe


def _estimate_order(results):
    """Estimate convergence order from the last two non-NaN data points."""
    valid = [(n, e) for n, e, *_ in results if np.isfinite(e) and e > 0]
    if len(valid) < 2:
        return float('nan')
    (n1, e1), (n2, e2) = valid[-2], valid[-1]
    # err ~ h^p, h ~ n^{-1/2}  →  err ~ n^{-p/2}  →  p ≈ -2 log(e2/e1)/log(n2/n1)
    return -2.0 * np.log(e2 / e1) / np.log(n2 / n1)


def test_convergence(degrees=(4, 6, 8), n_list=(50, 100, 200, 400, 800),
                     trials=5, seed_base=0):
    """
    Convergence study: mean absolute error on sin_product vs N, for each degree.

    Returns
    -------
    all_results : dict  {degree: [(n, mean_err, std_err), ...]}
    """
    f, exact   = _test_integrands()['sin_product']
    all_results = {}

    print(f"\n--- Convergence study  (integrand=sin(πx)sin(πy)) ---")
    for deg in degrees:
        print(f"\n  degree={deg}")
        print(f"  {'N':>6}  {'mean_err':>10}  {'std_err':>10}  {'order':>7}")
        print(f"  {'-'*40}")
        results = []
        for n in n_list:
            errs = []
            for trial in range(trials):
                rng      = np.random.default_rng(seed_base + n * 100 + trial)
                nodes    = rng.random((n, 2))
                quad     = NNLSQuadrature(degree=deg)
                pts, wts = quad.fit(nodes)
                errs.append(abs(float(wts @ f(pts)) - exact))
            me = float(np.mean(errs))
            se = float(np.std(errs))
            results.append((n, me, se))
            # estimate order from results so far
            order_str = f"{_estimate_order(results):5.2f}" if len(results) >= 2 else "    -"
            print(f"  {n:>6}  {me:>10.3e}  {se:>10.3e}  {order_str}")
        all_results[deg] = results

    return all_results


def test_weight_visualization(degree=6, n=300, seed=99, filename='nnls_weights.png'):
    """Scatter plot of nodes coloured by weight + weight histogram."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rng      = np.random.default_rng(seed)
    nodes    = rng.random((n, 2))
    quad     = NNLSQuadrature(degree=degree)
    pts, wts = quad.fit(nodes)

    _, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: scatter coloured by weight
    sc = axes[0].scatter(pts[:, 0], pts[:, 1], c=wts, cmap='viridis',
                         s=40, edgecolors='none', alpha=0.85)
    plt.colorbar(sc, ax=axes[0], label='Weight')
    axes[0].set_title(f'NNLS-QR weights  (degree={degree}, N={n})')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
    axes[0].set_aspect('equal')

    # Right: histogram
    axes[1].hist(wts, bins=30, color='steelblue', edgecolor='white', linewidth=0.5)
    axes[1].axvline(0, color='red', linestyle='--', linewidth=1, label='w=0')
    axes[1].set_xlabel('Weight value')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Weight distribution')
    info = (f"min={wts.min():.3e}\nmax={wts.max():.3e}\n"
            f"sum={wts.sum():.5f}\nn_neg={int(np.sum(wts<0))}")
    axes[1].text(0.97, 0.97, info, transform=axes[1].transAxes,
                 ha='right', va='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"\n  Saved: {filename}")


def plot_convergence(all_results, n_list, filename='nnls_convergence.png'):
    """Log-log convergence plot for all degrees."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    _, ax = plt.subplots(figsize=(8, 5))
    colors  = plt.cm.tab10(np.linspace(0, 0.5, len(all_results)))

    for (deg, results), color in zip(all_results.items(), colors):
        ns   = [r[0] for r in results]
        errs = [r[1] for r in results]
        valid = [(n, e) for n, e in zip(ns, errs) if np.isfinite(e) and e > 0]
        if not valid:
            continue
        nv, ev = zip(*valid)
        ax.loglog(nv, ev, 'o-', label=f'degree={deg}', color=color,
                  linewidth=1.8, markersize=6)

    # Reference slopes
    n_ref = np.array(n_list, dtype=float)
    for p, ls in zip((4, 6, 8), (':', '-.', '--')):
        ref = 0.1 * (n_ref / n_ref[0]) ** (-p / 2)
        ax.loglog(n_ref, ref, ls, color='gray', linewidth=0.9, alpha=0.7,
                  label=f'O(N^{{-{p}/2}})')

    ax.set_xlabel('Number of nodes N')
    ax.set_ylabel('Absolute error')
    ax.set_title('NNLS-QR convergence: ∫∫ sin(πx)sin(πy) dx dy')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved: {filename}")


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="NNLS-QR quadrature test suite")
    parser.add_argument('--quick',  action='store_true',
                        help='faster run: N ≤ 200, fewer trials')
    parser.add_argument('--noplot', action='store_true',
                        help='skip matplotlib output')
    parser.add_argument('--degree', type=int, default=6,
                        help='polynomial degree for basic tests (default 6)')
    parser.add_argument('--n',      type=int, default=400,
                        help='node count for basic tests (default 400)')
    args = parser.parse_args()

    print("=" * 60)
    print("  NNLS-QR Quadrature — Test Suite")
    print("  Glaubitz (2020), SIAM J. Numer. Anal. 58(4)")
    print("=" * 60)

    passed = failed = 0

    # 1. Basic integrands
    ok = test_basic_integrands(degree=args.degree, n=args.n)
    passed += ok; failed += (not ok)

    # 2. Positivity
    ok = test_positivity(degree=args.degree, n=args.n)
    passed += ok; failed += (not ok)

    # 3. Degree of exactness
    doe = test_degree_of_exactness(n=args.n + 100)
    ok  = (doe >= args.degree - 1)
    passed += ok; failed += (not ok)

    # 4. Convergence study
    if args.quick:
        n_list  = (50, 100, 200)
        degrees = (4, 6)
        trials  = 3
    else:
        n_list  = (50, 100, 200, 400, 800)
        degrees = (4, 6, 8)
        trials  = 5
    conv_results = test_convergence(degrees=degrees, n_list=n_list, trials=trials)

    # 5. Visualisation
    if not args.noplot:
        test_weight_visualization(degree=args.degree, n=args.n)
        plot_convergence(conv_results, n_list=n_list)

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
