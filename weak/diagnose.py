"""
diagnose.py — Layered diagnostics to pinpoint the source of noise/error
               in the RBF-PU-RA weak-form Poisson solver.

Run with:  mpiexec -n 1 python diagnose.py

Tests are ordered from highest-level to most granular:
  A — Operator consistency  (is WeakLap(u_exact) == q*f ?)
  B — Quadrature p-invariance  (is nodal lumping = Voronoi areas regardless of p?)
  C — GMRES convergence  (did the solver actually converge?)
  D — Derivative accuracy by polynomial degree
  E — Stiffness matrix eigenspectrum  (SPD? condition number?)
  F — Partition-of-unity integrity  (sum_p w_bar = 1, sum_p grad_w_bar = 0?)
  G — Spatial error map  (is the error smooth or oscillatory?)
"""

import numpy as np
from mpi4py import MPI
from nodes.SquareDomain import PoissonSquareOne
from source.Setup import Setup
from source.Operators import ApplyDeriv, ApplyDerivAdj
from source.Quadrature import PatchLocalWeights, DelaunayGaussQuadrature
from source.Solver import gmres

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# -----------------------------------------------------------------------
# Setup — coarse grid so the dense matrix build in TEST E is feasible
# -----------------------------------------------------------------------
if rank == 0:
    nodes, normals, groups = PoissonSquareOne(0.05)
    print(f"N = {nodes.shape[0]} nodes")
else:
    nodes = normals = groups = None

nodes   = comm.bcast(nodes,   root=0)
normals = comm.bcast(normals, root=0)
groups  = comm.bcast(groups,  root=0)

N        = nodes.shape[0]
d        = nodes.shape[1]
x, y     = nodes[:, 0], nodes[:, 1]
interior = groups['interior']
bc_nodes = groups['boundary:all']

patches, _ = Setup(comm, nodes, normals, 50)
q          = PatchLocalWeights(comm, patches, N)

Dk  = [ApplyDeriv(comm,    patches, N, k) for k in range(d)]
DkT = [ApplyDerivAdj(comm, patches, N, k) for k in range(d)]

def WeakLap(u):
    result = np.zeros(N)
    for l in range(d):
        g = Dk[l](u)
        g = q * g
        result += DkT[l](g)
    result[bc_nodes] = u[bc_nodes]
    return result

u_exact = np.sin(2*np.pi*x) * np.sin(2*np.pi*y)
f_exact = 8*np.pi**2 * np.sin(2*np.pi*x) * np.sin(2*np.pi*y)


# =====================================================================
# TEST A — Operator consistency
# =====================================================================
if rank == 0:
    print("\n" + "="*60)
    print("TEST A — Operator Consistency: WeakLap(u_exact) vs q*f")
    print("="*60)

b_exact = q * f_exact

Au = np.zeros(N)
for l in range(d):
    g = Dk[l](u_exact)
    g = q * g
    Au += DkT[l](g)

res     = Au[interior] - b_exact[interior]
res_rel = np.linalg.norm(res) / np.linalg.norm(b_exact[interior])
noise   = np.std(np.abs(res)) / (np.mean(np.abs(res)) + 1e-30)

if rank == 0:
    print(f"  ||WeakLap(u*) - b||/||b||  (interior): {res_rel:.3e}")
    print(f"  Max pointwise residual:                {np.max(np.abs(res)):.3e}")
    print(f"  std/mean of |residual|  (noise ratio): {noise:.2f}")
    if res_rel < 0.01:
        print("  => Operator looks CONSISTENT.  Bug likely in solver (see TEST C).")
    elif res_rel < 0.1:
        print("  => Operator has moderate inconsistency.  Check quadrature (TEST B) and derivatives (TEST D).")
    else:
        print("  => Operator is INCONSISTENT with the PDE.  Root cause likely in quadrature, derivatives, or PU (see B/D/F).")
    if noise > 1.0:
        print(f"  => Residual is OSCILLATORY (noise ratio {noise:.1f}).  Operator has per-node errors.")


# =====================================================================
# TEST B — Quadrature p-invariance
# =====================================================================
if rank == 0:
    print("\n" + "="*60)
    print("TEST B — Quadrature p-Invariance (nodal lumping vs Voronoi)")
    print("="*60)

    _, q1 = DelaunayGaussQuadrature(p=1, nodal=True).fit(nodes)
    _, q2 = DelaunayGaussQuadrature(p=2, nodal=True).fit(nodes)
    _, q4 = DelaunayGaussQuadrature(p=4, nodal=True).fit(nodes)
    _, q8 = DelaunayGaussQuadrature(p=8, nodal=True).fit(nodes)

    d12 = np.linalg.norm(q1 - q2) / np.linalg.norm(q1)
    d14 = np.linalg.norm(q1 - q4) / np.linalg.norm(q1)
    d18 = np.linalg.norm(q1 - q8) / np.linalg.norm(q1)
    print(f"  ||q(p=1)-q(p=2)||/||q||: {d12:.2e}")
    print(f"  ||q(p=1)-q(p=4)||/||q||: {d14:.2e}")
    print(f"  ||q(p=1)-q(p=8)||/||q||: {d18:.2e}")
    if max(d12, d14, d18) < 1e-10:
        print("  => CONFIRMED: nodal weights are p-INVARIANT.")
        print("     Barycentric lumping = Voronoi areas = O(h^2) rule regardless of p.")
        print("     The claimed degree 2p-1 only holds for the NON-NODAL Gauss points.")
    else:
        print("  => Weights change with p — barycentric lumping is not purely Voronoi.")

    print()
    print("  Quadrature accuracy (p=4 nodal vs p=1 nodal):")
    print(f"  {'f':30s} | exact         | err(p=4)    | err(p=1)")
    tests = [
        ("1",                    np.ones(N),           1.0),
        ("x",                    x,                    0.5),
        ("x^2",                  x**2,                 1/3),
        ("x^4",                  x**4,                 1/5),
        ("x^6",                  x**6,                 1/7),
        ("x^8",                  x**8,                 1/9),
        ("sin(2pi x)sin(2pi y)", np.sin(2*np.pi*x)*np.sin(2*np.pi*y), 0.0),
    ]
    for name, fv, ex in tests:
        e4 = abs(np.dot(q4, fv) - ex)
        e1 = abs(np.dot(q1, fv) - ex)
        print(f"  {name:30s} | {ex:+.6f}    | {e4:.3e}    | {e1:.3e}")


# =====================================================================
# TEST C — GMRES convergence
# =====================================================================
if rank == 0:
    print("\n" + "="*60)
    print("TEST C — GMRES Convergence Check")
    print("="*60)

rhs = q * f_exact
rhs[bc_nodes] = 0.0

sol, iters = gmres(comm, WeakLap, rhs, tol=1e-8, restart=300, maxiter=50)

Asol          = WeakLap(sol)
final_res     = Asol - rhs
rel_res_norm  = np.linalg.norm(final_res) / (np.linalg.norm(rhs) + 1e-30)
sol_err_int   = (np.linalg.norm(sol[interior] - u_exact[interior])
                 / np.linalg.norm(u_exact[interior]))

if rank == 0:
    print(f"  GMRES iterations:            {iters}")
    print(f"  Final relative residual:     {rel_res_norm:.3e}  (tol=1e-8)")
    print(f"  Relative solution error:     {sol_err_int:.3e}")
    if rel_res_norm < 1e-6:
        if sol_err_int < 0.01:
            print("  => Solver CONVERGED to an accurate solution.")
        else:
            print("  => Solver converged but solution error is large.")
            print("     System Ax=b was solved correctly, but b or A is wrong (see TEST A).")
    else:
        print("  => Solver DID NOT converge within budget.")
        print("     Consider ill-conditioning (see TEST E) or operator issues (TEST A).")


# =====================================================================
# TEST D — Derivative accuracy by polynomial degree
# =====================================================================
if rank == 0:
    print("\n" + "="*60)
    print("TEST D — Derivative Operator Accuracy by Polynomial Degree")
    print("="*60)
    print(f"  {'k':>3} | {'D_x(x^k) err':>14} | {'D_y(y^k) err':>14} | {'D_x(x^k*y) err':>16}")

for k in range(1, 10):
    # Pure monomial: D_x(x^k) = k*x^{k-1}
    u_xk      = x**k
    exact_xk  = k * x**(k-1)
    # Pure monomial: D_y(y^k) = k*y^{k-1}
    u_yk      = y**k
    exact_yk  = k * y**(k-1)
    # Mixed: D_x(x^k * y) = k*x^{k-1}*y
    u_xky     = x**k * y
    exact_xky = k * x**(k-1) * y

    edx  = (np.linalg.norm((Dk[0](u_xk)  - exact_xk )[interior])
            / np.linalg.norm(exact_xk [interior]))
    edy  = (np.linalg.norm((Dk[1](u_yk)  - exact_yk )[interior])
            / np.linalg.norm(exact_yk [interior]))
    edxy = (np.linalg.norm((Dk[0](u_xky) - exact_xky)[interior])
            / np.linalg.norm(exact_xky[interior]))

    if rank == 0:
        print(f"  {k:>3} | {edx:>14.3e} | {edy:>14.3e} | {edxy:>16.3e}")

# Also test the actual sin function (target of Poisson driver)
u_sin     = np.sin(2*np.pi*x) * np.sin(2*np.pi*y)
exact_dsx = 2*np.pi * np.cos(2*np.pi*x) * np.sin(2*np.pi*y)
exact_dsy = 2*np.pi * np.sin(2*np.pi*x) * np.cos(2*np.pi*y)
err_dsx   = (np.linalg.norm((Dk[0](u_sin) - exact_dsx)[interior])
             / np.linalg.norm(exact_dsx[interior]))
err_dsy   = (np.linalg.norm((Dk[1](u_sin) - exact_dsy)[interior])
             / np.linalg.norm(exact_dsy[interior]))
if rank == 0:
    print(f"\n  D_x(sin(2pi x)sin(2pi y)) interior error: {err_dsx:.3e}")
    print(f"  D_y(sin(2pi x)sin(2pi y)) interior error: {err_dsy:.3e}")
    if max(err_dsx, err_dsy) < 1e-3:
        print("  => Derivative operators appear ACCURATE for target function.")
    else:
        print("  => Derivative operators have significant error for target function.")
        print("     This directly degrades both stiffness matrix accuracy and solution quality.")


# =====================================================================
# TEST E — Stiffness matrix eigenspectrum
# =====================================================================
if rank == 0:
    print("\n" + "="*60)
    print("TEST E — Stiffness Matrix Eigenspectrum")
    print("="*60)
    print(f"  Building {N}x{N} dense matrix (interior block {len(interior)}x{len(interior)})...")

    A_dense = np.zeros((N, N))
    for j in range(N):
        ej = np.zeros(N); ej[j] = 1.0
        A_dense[:, j] = WeakLap(ej)

    A_int = A_dense[np.ix_(interior, interior)]

    # Symmetry
    sym_err = (np.linalg.norm(A_int - A_int.T, 'fro')
               / (np.linalg.norm(A_int, 'fro') + 1e-30))
    print(f"  Symmetry ||A-A^T||_F / ||A||_F: {sym_err:.3e}  (ideal: ~0)")

    # Eigenvalues of symmetrized interior block
    A_sym   = 0.5 * (A_int + A_int.T)
    eigvals = np.linalg.eigvalsh(A_sym)
    n_neg   = np.sum(eigvals < 0)
    n_zero  = np.sum(eigvals < 1e-8 * np.max(eigvals))
    cond    = np.max(eigvals) / max(np.min(eigvals[eigvals > 0]), 1e-30)

    print(f"  Eigenvalues (interior block):")
    print(f"    Min:  {np.min(eigvals):.4e}")
    print(f"    Max:  {np.max(eigvals):.4e}")
    print(f"    Condition number (max/min_pos): {cond:.4e}")
    print(f"    # negative:   {n_neg}")
    print(f"    # near-zero:  {n_zero}")

    if n_neg > 0:
        print("  => NEGATIVE eigenvalues found!  Operator is NOT SPD.")
        print("     This will cause GMRES to fail or give spurious solutions.")
    elif sym_err > 1e-4:
        print("  => Large symmetry error.  Adjoint operator may be incorrect.")
    elif cond > 1e6:
        print(f"  => Very large condition number ({cond:.2e}).  GMRES will converge slowly.")
    else:
        print("  => Eigenspectrum looks reasonable (SPD, moderate condition number).")


# =====================================================================
# TEST F — Partition-of-unity integrity
# =====================================================================
if rank == 0:
    print("\n" + "="*60)
    print("TEST F — Partition-of-Unity Integrity")
    print("="*60)

W_sum  = np.zeros(N)
gW_sum = np.zeros((N, d))
for patch in patches:
    idx = patch.node_indices
    W_sum[idx]      += patch.w_bar
    gW_sum[idx, :]  += patch.gw_bar

W_global  = np.zeros(N)
gW_global = np.zeros((N, d))
comm.Allreduce(W_sum,  W_global)
comm.Allreduce(gW_sum, gW_global)

coverage = np.zeros(N, dtype=int)
for patch in patches:
    coverage[patch.node_indices] += 1
cov_global = np.zeros(N, dtype=int)
comm.Allreduce(coverage, cov_global)

if rank == 0:
    pu_err   = np.max(np.abs(W_global - 1.0))
    pu_g_err = np.max(np.abs(gW_global))
    print(f"  max|sum_p w_bar - 1|:       {pu_err:.3e}  (ideal: ~0)")
    print(f"  max|sum_p grad_w_bar|:      {pu_g_err:.3e}  (ideal: ~0)")
    print(f"  Min node coverage:          {cov_global.min()}  (must be ≥1)")
    print(f"  Mean node coverage:         {cov_global.mean():.2f}")
    if pu_err > 1e-10:
        print("  => PU NORMALIZATION BROKEN.  Nodes may be within Wendland radius")
        print("     but excluded from the patch K-NN list (radius_scale=1.5 issue).")
    else:
        print("  => PU sums look correct.")
    if cov_global.min() < 1:
        print("  => Some nodes have ZERO patch coverage!  Those nodes will be wrong.")


# =====================================================================
# TEST G — Spatial error map
# =====================================================================
if rank == 0:
    print("\n" + "="*60)
    print("TEST G — Spatial Error Map (smooth vs oscillatory)")
    print("="*60)

# Re-use solution from TEST C
err_nodal = np.abs(sol - u_exact)

if rank == 0:
    e_int = err_nodal[interior]
    noise_ratio = e_int.std() / (e_int.mean() + 1e-30)
    print(f"  Interior solution error:")
    print(f"    max:              {e_int.max():.3e}")
    print(f"    mean:             {e_int.mean():.3e}")
    print(f"    std:              {e_int.std():.3e}")
    print(f"    std/mean (noise): {noise_ratio:.2f}  (smooth~0, noisy>>1)")

    # Gradient of error: high gradient = rapid spatial oscillation
    err_dx = Dk[0](err_nodal)
    err_dy = Dk[1](err_nodal)
    grad_err_mag = np.sqrt(err_dx**2 + err_dy**2)
    print(f"  |grad(error)| at interior nodes:")
    print(f"    max:   {grad_err_mag[interior].max():.3e}")
    print(f"    mean:  {grad_err_mag[interior].mean():.3e}")
    # Relative oscillation: |grad e| * h vs |e|  (dimensionless)
    h_approx = np.sqrt(1.0 / N)
    osc = grad_err_mag[interior].mean() * h_approx / (e_int.mean() + 1e-30)
    print(f"  Oscillation index |grad e|*h / |e| : {osc:.2f}  (smooth~1, oscillatory>>1)")

    if noise_ratio > 1.5 or osc > 3.0:
        print("  => Error is SPATIALLY OSCILLATORY (noisy).  Consistent with bad quadrature")
        print("     or inaccurate derivatives producing a stiffness matrix with wrong off-diagonals.")
    else:
        print("  => Error appears SMOOTH.  Consistent with low-order convergence, not oscillation.")

if rank == 0:
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  TEST A  operator residual rel norm: {res_rel:.3e}")
    print(f"  TEST B  quadrature p-invariance:    {max(d12,d14,d18):.3e}  (0=Voronoi)")
    print(f"  TEST C  GMRES residual:             {rel_res_norm:.3e}")
    print(f"  TEST C  solution error:             {sol_err_int:.3e}")
    print(f"  TEST D  D_x(sin) error:             {err_dsx:.3e}")
    print(f"  TEST E  symmetry error:             {sym_err:.3e}")
    print(f"  TEST E  condition number:           {cond:.3e}")
    print(f"  TEST E  # negative eigenvalues:     {n_neg}")
    print(f"  TEST F  PU sum error:               {pu_err:.3e}")
    print(f"  TEST G  noise ratio (std/mean):     {noise_ratio:.2f}")
    print("="*60)
    print("Done.")
