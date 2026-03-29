import numpy as np
from scipy.optimize import minimize
from mpi4py import MPI


def _max_poly_degree(n, d=2):
    """Largest m such that binom(m+d, d) <= n."""
    m = 0
    while True:
        next_dim = 1
        for i in range(1, d + 1):
            next_dim = next_dim * (m + 1 + i) // i
        if next_dim > n:
            return m
        m += 1


def _build_vandermonde_2d(nodes, m):
    """
    Build 2D polynomial Vandermonde matrix up to total degree m.
    nodes: (n, 2) coordinates
    Returns: (n, M) Vandermonde matrix, list of (a, b) exponent pairs
    """
    monoms = []
    for deg in range(m + 1):
        for b in range(deg + 1):
            a = deg - b
            monoms.append((a, b))

    P = np.empty((nodes.shape[0], len(monoms)))
    for j, (a, b) in enumerate(monoms):
        P[:, j] = (nodes[:, 0] ** a) * (nodes[:, 1] ** b)

    return P, monoms


def _square_moments(monoms):
    """
    Analytic moments of monomials x^a y^b over [0,1]^2.
    integral_[0,1]^2 x^a y^b dx dy = 1/((a+1)(b+1))
    """
    M = np.empty(len(monoms))
    for j, (a, b) in enumerate(monoms):
        M[j] = 1.0 / ((a + 1) * (b + 1))
    return M


def PatchLocalWeights(comm, patches, N):
    """
    Compute positive, polynomial-exact quadrature weights via a global
    solve over [0,1]^2.

    Uses direct KKT solve (O(N M^2) where M << N) with fallback to
    iterative SLSQP only if any weights are negative.

    Parameters
    ----------
    comm : MPI communicator
    patches : list of Patch objects owned by this rank
    N : int, total number of global nodes

    Returns
    -------
    q : (N,) global quadrature weights (positive)
    """
    # Reconstruct global node coordinates from patches
    nodes_local = np.zeros((N, 2))
    have_node = np.zeros(N, dtype=bool)
    for patch in patches:
        idx = patch.node_indices
        nodes_local[idx] = patch.nodes
        have_node[idx] = True

    nodes_global = np.zeros((N, 2))
    comm.Allreduce(nodes_local, nodes_global)

    count = np.zeros(N)
    count_local = have_node.astype(float)
    comm.Allreduce(count_local, count)
    count = np.maximum(count, 1.0)
    nodes_global /= count[:, None]

    rank = comm.Get_rank()
    if rank == 0:
        q = _solve_global_qp(nodes_global)
    else:
        q = None

    q = comm.bcast(q, root=0)
    return q


def _solve_global_qp(nodes):
    """
    Solve for quadrature weights over [0,1]^2.

    min ||q - q0||^2  s.t.  P^T q = M,  q >= 0

    First tries the direct KKT solution (O(N M^2) with M x M inverse).
    Falls back to SLSQP only if any weights are negative.

    Parameters
    ----------
    nodes : (N, 2) all node coordinates

    Returns
    -------
    q : (N,) positive quadrature weights
    """
    N = nodes.shape[0]
    m = _max_poly_degree(N, d=2)
    m = min(m, 10)

    q0 = (1.0 / N) * np.ones(N)

    # Try decreasing polynomial degrees until KKT gives all-positive weights
    while m > 0:
        P, monoms = _build_vandermonde_2d(nodes, m)
        M_moments = _square_moments(monoms)

        if np.linalg.matrix_rank(P) < len(monoms):
            m -= 1
            continue

        # Direct KKT: q = q0 + P @ inv(P^T P) @ (M - P^T q0)
        PtP = P.T @ P
        rhs = M_moments - P.T @ q0
        lam = np.linalg.solve(PtP, rhs)
        q = q0 + P @ lam

        if np.all(q > 0):
            print(f"KKT solution all-positive at degree {m}")
            return q

        m -= 1

    # Degree 0 fallback: uniform weights
    return q0