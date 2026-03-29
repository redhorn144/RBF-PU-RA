"""
Diagnostic tests for the weak-form Poisson setup.
Checks quadrature accuracy and adjoint correctness.
"""
import numpy as np
from mpi4py import MPI
from nodes.SquareDomain import PoissonSquareOne
from source.Patch import Patch
from source.Setup import Setup
from source.Operators import ApplyDeriv, ApplyDerivAdj
from source.Quadrature import PatchLocalWeights

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    nodes, normals, groups = PoissonSquareOne(0.05)
    print(f"N = {nodes.shape[0]} nodes")
else:
    nodes = normals = groups = None

nodes = comm.bcast(nodes, root=0)
normals = comm.bcast(normals, root=0)
groups = comm.bcast(groups, root=0)

N = nodes.shape[0]
bc_nodes = groups['boundary:all']

patches, patches_for_rank = Setup(comm, nodes, normals, 50)
q = PatchLocalWeights(comm, patches, N)

x, y = nodes[:, 0], nodes[:, 1]

# ============================================================
# TEST 1: Quadrature accuracy — integrate known functions
# ============================================================
if rank == 0:
    print("\n=== TEST 1: Quadrature accuracy ===")

    test_functions = [
        ("1",           np.ones(N),                     1.0),                       # int 1 over [0,1]^2
        ("x",           x,                               0.5),                       # int x
        ("y",           y,                               0.5),                       # int y
        ("x^2",         x**2,                            1/3),                       # int x^2
        ("y^2",         y**2,                            1/3),
        ("x*y",         x*y,                             0.25),                      # int xy
        ("x^2*y^2",     x**2 * y**2,                     1/9),
        ("x^3*y",       x**3 * y,                        1/8),
        ("x^4",         x**4,                            1/5),
        ("sin(2pi x)*sin(2pi y)", np.sin(2*np.pi*x)*np.sin(2*np.pi*y), 0.0),
    ]

    for name, f_vals, exact in test_functions:
        numerical = np.dot(q, f_vals)
        err = abs(numerical - exact)
        rel = err / (abs(exact) + 1e-30)
        print(f"  int({name:25s}) = {numerical:+.8e}  exact={exact:+.8e}  err={err:.2e}  rel={rel:.2e}")

    print(f"\n  Quadrature weights: min={q.min():.4e}  max={q.max():.4e}  sum={q.sum():.6f}  all>0: {np.all(q > 0)}")
    neg_count = np.sum(q <= 0)
    if neg_count > 0:
        print(f"  WARNING: {neg_count} non-positive weights!")

# ============================================================
# TEST 2: Adjoint correctness — <D_k u, v> == <u, D_k^T v>
# ============================================================
if rank == 0:
    print("\n=== TEST 2: Adjoint correctness ===")

Dk = [ApplyDeriv(comm, patches, N, k) for k in range(2)]
DkT = [ApplyDerivAdj(comm, patches, N, k) for k in range(2)]

np.random.seed(42)
u_rand = np.random.randn(N)
v_rand = np.random.randn(N)

for k in range(2):
    Dk_u = Dk[k](u_rand)
    DkT_v = DkT[k](v_rand)
    lhs = np.dot(Dk_u, v_rand)      # <D_k u, v>
    rhs = np.dot(u_rand, DkT_v)     # <u, D_k^T v>
    if rank == 0:
        rel_err = abs(lhs - rhs) / (0.5 * (abs(lhs) + abs(rhs)) + 1e-30)
        print(f"  k={k}: <D_k u, v> = {lhs:+.10e}   <u, D_k^T v> = {rhs:+.10e}   rel_err = {rel_err:.2e}")

# ============================================================
# TEST 3: Integration by parts — <D_k u, q * v> + <u, D_k^T (q * v)> consistency
# For the weak Laplacian, we need D_k^T Q D_k to be correct.
# Test: for a polynomial u where we know -Lap u exactly, check
# that sum_k <D_k u, Q D_k u> == int grad(u) . grad(u) dx
# ============================================================
if rank == 0:
    print("\n=== TEST 3: Weak Laplacian of known function ===")

# Test with u = x*(1-x)*y*(1-y) which vanishes on boundary
u_test = x * (1 - x) * y * (1 - y)
# -Lap(u) = 2*y*(1-y) + 2*x*(1-x)
neg_lap_u = 2*y*(1-y) + 2*x*(1-x)

# Weak form: A u = b  where b_i = q_i * f(x_i)
# For the exact solution, A u should equal q * (-Lap u) at interior nodes

# Compute A u = sum_k D_k^T Q D_k u
Au = np.zeros(N)
for k in range(2):
    g = Dk[k](u_test)
    g = q * g
    Au += DkT[k](g)

interior = groups['interior']

# The correct test: v^T A u = v^T (q*f) for all test vectors v
# i.e. <v, Au> = <v, q*f> in the Euclidean inner product
# This is the weak form identity, NOT a pointwise check.

# Check energy: <grad u, Q grad u> should equal int |grad u|^2 dx = 1/45
num_dux = Dk[0](u_test)
num_duy = Dk[1](u_test)
energy_num = np.dot(num_dux, q * num_dux) + np.dot(num_duy, q * num_duy)
energy_exact = 1.0 / 45.0
if rank == 0:
    print(f"  Energy <grad u, Q grad u> = {energy_num:.8e}  exact = {energy_exact:.8e}  rel_err = {abs(energy_num - energy_exact)/energy_exact:.2e}")

# Check weak form identity: for random v, <v, Au> should equal sum_k <D_k v, Q D_k u>
np.random.seed(123)
v_test = np.random.randn(N)
v_test[bc_nodes] = 0.0  # test function vanishes on boundary

lhs = np.dot(v_test, Au)
rhs_bilinear = sum(np.dot(Dk[k](v_test), q * Dk[k](u_test)) for k in range(2))
if rank == 0:
    rel = abs(lhs - rhs_bilinear) / (0.5*(abs(lhs)+abs(rhs_bilinear)))
    print(f"  <v, Au> = {lhs:.8e}   sum <D_k v, Q D_k u> = {rhs_bilinear:.8e}   rel_err = {rel:.2e}")

# Check: does solving A u_sol = q*f recover u?
from source.Solver import gmres
rhs_weak = q * neg_lap_u
rhs_weak[bc_nodes] = 0.0

def WeakLap_test(u_in):
    res = np.zeros(N)
    for k in range(2):
        g = Dk[k](u_in)
        g = q * g
        res += DkT[k](g)
    res[bc_nodes] = u_in[bc_nodes]
    return res

u_sol, iters = gmres(comm, WeakLap_test, rhs_weak, tol=1e-10, restart=100, maxiter=200)
if rank == 0:
    err_sol = np.linalg.norm(u_sol[interior] - u_test[interior]) / np.linalg.norm(u_test[interior])
    print(f"  GMRES solve error: {err_sol:.2e}  ({iters} iterations)")

# ============================================================
# TEST 4: Check that the derivative operators are accurate
# for a known function u = sin(2pi x) sin(2pi y)
# ============================================================
if rank == 0:
    print("\n=== TEST 4: Derivative operator accuracy ===")

u_sin = np.sin(2*np.pi*x) * np.sin(2*np.pi*y)
# du/dx = 2pi cos(2pi x) sin(2pi y)
# du/dy = 2pi sin(2pi x) cos(2pi y)
exact_dx = 2*np.pi * np.cos(2*np.pi*x) * np.sin(2*np.pi*y)
exact_dy = 2*np.pi * np.sin(2*np.pi*x) * np.cos(2*np.pi*y)

num_dx = Dk[0](u_sin)
num_dy = Dk[1](u_sin)

if rank == 0:
    err_dx = np.linalg.norm(num_dx[interior] - exact_dx[interior]) / np.linalg.norm(exact_dx[interior])
    err_dy = np.linalg.norm(num_dy[interior] - exact_dy[interior]) / np.linalg.norm(exact_dy[interior])
    print(f"  D_x error (interior): {err_dx:.2e}")
    print(f"  D_y error (interior): {err_dy:.2e}")

# ============================================================
# TEST 5: Symmetry and conditioning check
# All ranks must participate in matvecs (Allreduce inside).
# ============================================================
if rank == 0:
    print("\n=== TEST 5: Stiffness matrix conditioning ===")
    print(f"  Interior DOFs: {len(interior)}")

# Symmetry check — all ranks participate in matvecs
np.random.seed(99)
u1 = np.random.randn(N); u1[bc_nodes] = 0.0
u2 = np.random.randn(N); u2[bc_nodes] = 0.0
Au1 = WeakLap_test(u1)
Au2 = WeakLap_test(u2)
if rank == 0:
    lhs = np.dot(Au1, u2)
    rhs = np.dot(u1, Au2)
    sym_err = abs(lhs - rhs) / (0.5*(abs(lhs) + abs(rhs)))
    print(f"  Symmetry check |<Au,v> - <u,Av>|/avg: {sym_err:.2e}")

# Power iteration for largest eigenvalue — all ranks participate
v = np.random.randn(N); v[bc_nodes] = 0.0
v /= np.linalg.norm(v)
for _ in range(100):
    Av = WeakLap_test(v)
    Av[bc_nodes] = 0.0
    lam_max = np.dot(v, Av)
    v = Av / np.linalg.norm(Av)

# Inverse power iteration for smallest eigenvalue — use GMRES
# Solve A v = b for random b, then estimate smallest eigenvalue from Rayleigh quotient
rhs_rand = np.random.randn(N); rhs_rand[bc_nodes] = 0.0
v_inv, iters_inv = gmres(comm, WeakLap_test, rhs_rand, tol=1e-8, restart=50, maxiter=500)
Av_inv = WeakLap_test(v_inv)
Av_inv[bc_nodes] = 0.0
v_inv[bc_nodes] = 0.0
lam_min_est = np.dot(v_inv, Av_inv) / np.dot(v_inv, v_inv)

if rank == 0:
    print(f"  Largest eigenvalue (power iter): {lam_max:.4e}")
    print(f"  Smallest eigenvalue (Rayleigh): {lam_min_est:.4e}")
    print(f"  Estimated condition number: {abs(lam_max / lam_min_est):.4e}")
    print(f"  (inverse iteration GMRES took {iters_inv} iters)")

if rank == 0:
    print("\nDone.")
