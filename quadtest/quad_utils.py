"""
quad_utils.py — Shared utilities for scattered-node quadrature methods.

Provides:
  - monomial_exponents(dim, degree)  : enumerate multi-indices
  - build_vandermonde(nodes, degree)  : Vandermonde matrix for monomials
  - compute_moments(degree, domain)   : exact monomial integrals over standard domains
  - scale_nodes(nodes)               : scale to [0,1]^d; returns (scaled, lo, rng)
  - random_unit_square / disk / cube / ball : scattered-node generators
  - TEST_INTEGRANDS_2D / 3D          : dict of {name: (func, exact)} test cases
  - check_positivity(weights)        : True if all w_i > 0
  - compute_error(f, nodes, weights, exact) : absolute quadrature error
"""

import numpy as np
from math import factorial
from scipy.special import gamma


# ---------------------------------------------------------------------------
# Monomial enumeration
# ---------------------------------------------------------------------------

def monomial_exponents(dim, degree):
    """
    Return list of multi-index tuples (a_0,...,a_{dim-1}) with sum <= degree,
    in graded lexicographic order.
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


# ---------------------------------------------------------------------------
# Vandermonde matrix
# ---------------------------------------------------------------------------

def build_vandermonde(nodes, degree):
    """
    Evaluate monomials up to `degree` at each node.

    Parameters
    ----------
    nodes  : (n, dim) array of node coordinates (scale to [0,1]^dim for stability)
    degree : polynomial degree

    Returns
    -------
    V : (m, n) float array, V[k, i] = p_k(nodes[i])
        where {p_k} are the monomials in graded-lex order.
    """
    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim == 1:
        nodes = nodes[:, None]
    n, dim = nodes.shape
    exps = monomial_exponents(dim, degree)
    m = len(exps)
    V = np.ones((m, n), dtype=float)
    for k, alpha in enumerate(exps):
        for j, aj in enumerate(alpha):
            if aj > 0:
                V[k] *= nodes[:, j] ** aj
    return V


# ---------------------------------------------------------------------------
# Exact monomial integrals (moments)
# ---------------------------------------------------------------------------

def compute_moments(degree, domain='unit_square', dim=None):
    """
    Compute b_k = integral_{domain} p_k(x) dx for each monomial p_k.

    Supported domains
    -----------------
    'unit_square' (dim=2)  : [0,1]^2,  analytical
    'unit_cube'   (dim=3)  : [0,1]^3,  analytical
    'unit_disk'   (dim=2)  : x^2+y^2<=1, analytical (Gamma-function formula)
    'unit_ball'   (dim=3)  : |x|<=1,   analytical (Gamma-function formula)

    Returns
    -------
    b : (m,) float array
    """
    # --- [0,1]^d domains ---
    if domain in ('unit_square', 'unit_cube'):
        d = 2 if domain == 'unit_square' else 3
        if dim is not None and dim != d:
            raise ValueError(f"domain='{domain}' requires dim={d}")
        exps = monomial_exponents(d, degree)
        # integral of x^a over [0,1] is 1/(a+1); product over coordinates
        return np.array([1.0 / np.prod([a + 1 for a in alpha]) for alpha in exps])

    # --- unit disk (2D): integral of x^a y^b over B^2 ---
    elif domain == 'unit_disk':
        exps = monomial_exponents(2, degree)
        moments = []
        for a, b in exps:
            if a % 2 != 0 or b % 2 != 0:
                moments.append(0.0)
            else:
                # Derivation via polar coords:
                # integral = Gamma(m1+1/2) * Gamma(m2+1/2) / Gamma(m1+m2+2)
                # where m1=a/2, m2=b/2
                m1, m2 = a // 2, b // 2
                val = gamma(m1 + 0.5) * gamma(m2 + 0.5) / gamma(m1 + m2 + 2)
                moments.append(val)
        return np.array(moments)

    # --- unit ball (3D): integral of x^a y^b z^c over B^3 ---
    elif domain == 'unit_ball':
        exps = monomial_exponents(3, degree)
        moments = []
        for a, b, c in exps:
            if a % 2 != 0 or b % 2 != 0 or c % 2 != 0:
                moments.append(0.0)
            else:
                # Derivation via spherical coords:
                # integral = Gamma(m1+1/2)*Gamma(m2+1/2)*Gamma(m3+1/2) / Gamma(m1+m2+m3+5/2)
                m1, m2, m3 = a // 2, b // 2, c // 2
                val = (gamma(m1 + 0.5) * gamma(m2 + 0.5) * gamma(m3 + 0.5)
                       / gamma(m1 + m2 + m3 + 2.5))
                moments.append(val)
        return np.array(moments)

    else:
        raise ValueError(f"Unknown domain: '{domain}'")


# ---------------------------------------------------------------------------
# Node scaling
# ---------------------------------------------------------------------------

def scale_nodes(nodes):
    """
    Scale nodes to [0,1]^dim.

    Returns
    -------
    scaled : (n, dim) nodes in [0,1]^dim
    lo     : (dim,) coordinate minimums
    rng    : (dim,) coordinate ranges (hi - lo)

    To recover original coords: nodes = scaled * rng + lo
    To scale quadrature weights back: w_physical = w_scaled * prod(rng)
    """
    nodes = np.asarray(nodes, dtype=float)
    lo = nodes.min(axis=0)
    hi = nodes.max(axis=0)
    rng = hi - lo
    rng[rng == 0.0] = 1.0       # avoid divide-by-zero for degenerate dims
    scaled = (nodes - lo) / rng
    return scaled, lo, rng


# ---------------------------------------------------------------------------
# Scattered node generators
# ---------------------------------------------------------------------------

def random_unit_square(n, seed=None):
    """n random nodes uniformly distributed in [0,1]^2."""
    rng = np.random.default_rng(seed)
    return rng.random((n, 2))


def random_unit_disk(n, seed=None):
    """n random nodes uniformly distributed in the unit disk."""
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        batch = rng.random((2 * n, 2)) * 2 - 1
        mask = np.sum(batch ** 2, axis=1) <= 1.0
        pts.extend(batch[mask].tolist())
    return np.array(pts[:n])


def random_unit_cube(n, seed=None):
    """n random nodes uniformly distributed in [0,1]^3."""
    rng = np.random.default_rng(seed)
    return rng.random((n, 3))


def random_unit_ball(n, seed=None):
    """n random nodes uniformly distributed in the unit ball."""
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        batch = rng.random((2 * n, 3)) * 2 - 1
        mask = np.sum(batch ** 2, axis=1) <= 1.0
        pts.extend(batch[mask].tolist())
    return np.array(pts[:n])


# ---------------------------------------------------------------------------
# Test integrands
# ---------------------------------------------------------------------------

# Each entry: (function, exact_integral_over_unit_square/cube)
# exact=None means it must be computed numerically (handled in test_compare.py)

TEST_INTEGRANDS_2D = {
    'constant':    (lambda x: np.ones(len(x)),
                    1.0),
    'linear':      (lambda x: x[:, 0] + x[:, 1],
                    1.0),
    'poly_x4y3':   (lambda x: x[:, 0] ** 4 * x[:, 1] ** 3,
                    1.0 / 5 * 1.0 / 4),
    'poly_x6':     (lambda x: x[:, 0] ** 6,
                    1.0 / 7),
    'sin_product': (lambda x: np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1]),
                    4.0 / np.pi ** 2),
    'gauss_bump':  (lambda x: np.exp(-10 * ((x[:, 0] - 0.5) ** 2
                                            + (x[:, 1] - 0.5) ** 2)),
                    None),   # numerical exact below
}

# Pre-compute the Gauss bump exact value with high-accuracy 1D Gauss-Legendre
def _gauss_bump_exact_2d():
    pts, wts = np.polynomial.legendre.leggauss(50)
    pts = (pts + 1) / 2
    wts = wts / 2
    X, Y = np.meshgrid(pts, pts, indexing='ij')
    W = np.outer(wts, wts)
    return float(np.sum(W * np.exp(-10 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))))


TEST_INTEGRANDS_2D['gauss_bump'] = (
    TEST_INTEGRANDS_2D['gauss_bump'][0],
    _gauss_bump_exact_2d()
)

TEST_INTEGRANDS_3D = {
    'constant':     (lambda x: np.ones(len(x)),
                     1.0),
    'linear':       (lambda x: x[:, 0] + x[:, 1] + x[:, 2],
                     1.5),
    'poly_x2y2z2':  (lambda x: x[:, 0] ** 2 * x[:, 1] ** 2 * x[:, 2] ** 2,
                     1.0 / 27),
    'sin_product':  (lambda x: (np.sin(np.pi * x[:, 0])
                                * np.sin(np.pi * x[:, 1])
                                * np.sin(np.pi * x[:, 2])),
                     8.0 / np.pi ** 3),
    'gauss_bump':   (lambda x: np.exp(-10 * ((x[:, 0] - 0.5) ** 2
                                             + (x[:, 1] - 0.5) ** 2
                                             + (x[:, 2] - 0.5) ** 2)),
                     None),
}

def _gauss_bump_exact_3d():
    pts, wts = np.polynomial.legendre.leggauss(20)
    pts = (pts + 1) / 2
    wts = wts / 2
    val = 0.0
    for i, (xi, wi) in enumerate(zip(pts, wts)):
        for j, (xj, wj) in enumerate(zip(pts, wts)):
            for k, (xk, wk) in enumerate(zip(pts, wts)):
                val += wi * wj * wk * np.exp(
                    -10 * ((xi - 0.5) ** 2 + (xj - 0.5) ** 2 + (xk - 0.5) ** 2))
    return val

TEST_INTEGRANDS_3D['gauss_bump'] = (
    TEST_INTEGRANDS_3D['gauss_bump'][0],
    _gauss_bump_exact_3d()
)


# ---------------------------------------------------------------------------
# Quadrature diagnostics
# ---------------------------------------------------------------------------

def check_positivity(weights, tol=0.0):
    """Return True if all weights are > tol."""
    return bool(np.all(np.asarray(weights) > tol))


def compute_error(f, nodes, weights, exact):
    """
    Compute absolute quadrature error |sum_i w_i f(x_i) - exact|.

    Parameters
    ----------
    f      : callable, f(nodes) -> (n,) array
    nodes  : (n, dim) array
    weights: (n,) array of quadrature weights
    exact  : scalar, exact value of the integral
    """
    approx = float(np.dot(weights, f(nodes)))
    return abs(approx - exact)


def degree_of_exactness(nodes, weights, dim, max_degree=10):
    """
    Find the highest polynomial degree for which the rule is exact on [0,1]^dim.
    Tests all monomials up to max_degree; stops at the first failure (tol=1e-8).
    """
    tol = 1e-8
    exact_deg = -1
    for deg in range(max_degree + 1):
        exps = monomial_exponents(dim, deg)
        # Only test monomials of exactly this degree
        new_exps = [e for e in exps if sum(e) == deg]
        passed = True
        for alpha in new_exps:
            f = lambda x, a=alpha: np.prod(x ** np.array(a), axis=1)
            exact = 1.0 / np.prod([a + 1 for a in alpha])   # unit hypercube
            err = compute_error(f, nodes, weights, exact)
            if err > tol:
                passed = False
                break
        if passed:
            exact_deg = deg
        else:
            break
    return exact_deg


# ---------------------------------------------------------------------------
# Gauss-Legendre on [0,1]  (shared by quad_delaunay and quad_rbf)
# ---------------------------------------------------------------------------

def gauss_legendre_01(n):
    """
    Return (points, weights) for n-point Gauss-Legendre rule on [0, 1].
    Weights sum to 1.
    """
    pts, wts = np.polynomial.legendre.leggauss(n)
    pts = (pts + 1.0) / 2.0
    wts = wts / 2.0
    return pts, wts


if __name__ == '__main__':
    print("=== quad_utils smoke test ===")
    print(f"monomial_exponents(2, 2) = {monomial_exponents(2, 2)}")

    nodes2d = random_unit_square(100, seed=0)
    V = build_vandermonde(nodes2d, degree=3)
    print(f"Vandermonde (2D, deg=3): shape {V.shape}")

    b_sq = compute_moments(3, 'unit_square')
    b_dk = compute_moments(3, 'unit_disk')
    print(f"Moments unit_square deg<=3: {b_sq}")
    print(f"Moments unit_disk  deg<=3: {b_dk}")

    b_cu = compute_moments(3, 'unit_cube')
    b_bl = compute_moments(3, 'unit_ball')
    print(f"Moments unit_cube  deg<=3: {b_cu}")
    print(f"Moments unit_ball  deg<=3 (even only nonzero): {b_bl}")

    pts1d, wts1d = gauss_legendre_01(5)
    print(f"GL(5) on [0,1]: sum(w)={wts1d.sum():.6f}")

    print("Gauss bump 2D exact:", TEST_INTEGRANDS_2D['gauss_bump'][1])
    print("Gauss bump 3D exact:", TEST_INTEGRANDS_3D['gauss_bump'][1])
    print("All OK.")
