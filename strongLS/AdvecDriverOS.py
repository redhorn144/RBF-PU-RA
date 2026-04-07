"""
Oversampled RBF-PU-RA advection solver (Tominec et al. 2024 framework).

Solves  ∂u/∂t + c·∇u = 0  on [0,1]²  with homogeneous Dirichlet BCs.
Velocity field: solid-body rotation c = ω(-(y-0.5), (x-0.5)).

Method
------
Semidiscrete oversampled system (M > N eval points):
    E du/dt = -D u            (M equations, N unknowns)

    E (M×N): PU interpolation at eval pts — evaluates the approximant
    D (M×N): PU advection at eval pts     — evaluates c·∇u_h

ℓ₂ normal equations give the effective N×N ODE:
    du/dt = A_eff u,   A_eff = -(EᵀE)⁻¹ Eᵀ D

A_eff has eigenvalues in the left half-plane for sufficient oversampling β,
enabling a stable explicit time integrator. Compare with the square Kansa
system which has spurious positive-real eigenvalues requiring implicit methods.

Hyperviscosity
--------------
A biharmonic hyperviscosity term is added to remove residual positive-real
eigenvalues:

    du/dt = A_eff u  -  ε_hv · Δ²_eff u

where  Δ²_eff = L_eff @ L_eff,  L_eff = (EᵀE)⁻¹ Eᵀ L_dense  (N×N).

Since Δ is negative semidefinite, Δ² = (Δ)² is positive semidefinite, so
-ε_hv · Δ²_eff contributes purely negative real parts.  The k⁴ scaling
damps only high-frequency modes; physical (low-k) modes are barely affected.

Time integration: matrix exponential (exact solution to the linear ODE).
Precomputes P = expm(dt · A_eff) once via scipy.linalg.expm; each step is
u^{n+1} = P u^n — one dense matvec, no CFL constraint.

Usage
-----
    mpiexec -n 4 python AdvecDriverOS.py
"""
import numpy as np
import scipy.linalg as spla   # used for cho_factor/cho_solve in A_eff precomputation
from mpi4py import MPI
import os

from nodes.SquareDomain import PoissonSquareOne
from source.Setup import SetupOversampled
from source.Operators import ApplyDerivOS, ApplyInterpOS, ApplyLapOS
from source.Plotter import AnimateSolution

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# ── Problem parameters ────────────────────────────────────────────────────────
h_sol  = 0.07          # solution node spacing  → N ≈ 400
h_eval = 0.03        # eval point spacing     → β ≈ (0.05/0.034)² ≈ 2.16

omega  = 1.0           # angular velocity (rad/s)
T      = 2*np.pi/omega # one full revolution
dt     = 0.005      # time step; CFL ~ dt * ω/h_sol ≈ 1.0,  well inside √3
n_steps = int(round(T / dt))

# Biharmonic hyperviscosity coefficient (shift spurious positive-real eigenvalues left).
# Scales as h^4: decrease if physical modes are over-damped, increase to push the
# spectrum further into the left half-plane.
# Tuning guide: if max Re(λ) > 0 after assembly, increase eps_hv until stable.
eps_hv = 1.5e-2

# Gaussian bump initial condition
x0, y0, sigma = 0.5, 0.70, 0.07

# ── Node generation ───────────────────────────────────────────────────────────
if rank == 0:
    nodes, normals, sol_groups = PoissonSquareOne(h_sol)
    eval_all, _, eval_groups   = PoissonSquareOne(h_eval)

    interior_eval = eval_all[eval_groups['interior']]
    boundary_sol  = nodes[sol_groups['boundary:all']]
    eval_pts      = np.vstack([interior_eval, boundary_sol])

    N = nodes.shape[0]
    M = eval_pts.shape[0]
    boundary_nodes    = sol_groups['boundary:all']
    eval_interior_idx = np.arange(len(interior_eval))
    eval_boundary_idx = np.arange(len(interior_eval), M)

    print(f"N = {N} solution nodes,  M = {M} eval pts  (β = {M/N:.2f})")
    print(f"dt = {dt},  n_steps = {n_steps},  T = {T:.4f}")
else:
    nodes = normals = sol_groups = eval_pts = None
    boundary_nodes = eval_interior_idx = eval_boundary_idx = None

nodes             = comm.bcast(nodes,             root=0)
normals           = comm.bcast(normals,           root=0)
sol_groups        = comm.bcast(sol_groups,        root=0)
eval_pts          = comm.bcast(eval_pts,          root=0)
boundary_nodes    = comm.bcast(boundary_nodes,    root=0)
eval_interior_idx = comm.bcast(eval_interior_idx, root=0)
eval_boundary_idx = comm.bcast(eval_boundary_idx, root=0)

N = nodes.shape[0]
M = eval_pts.shape[0]

# ── Velocity field ────────────────────────────────────────────────────────────
cx_eval = -omega * (eval_pts[:, 1] - 0.5)
cy_eval =  omega * (eval_pts[:, 0] - 0.5)

# ── Patch setup ───────────────────────────────────────────────────────────────
if rank == 0:
    print("Setting up oversampled patches ...")
    t0 = MPI.Wtime()

patches_os, _ = SetupOversampled(comm, nodes, eval_pts, normals,
                                  nodes_per_patch=30, overlap=3, eval_epsilon=0)

if rank == 0:
    print(f"Setup complete in {MPI.Wtime()-t0:.1f}s")

# ── Operator closures ─────────────────────────────────────────────────────────
E_op      = ApplyInterpOS(comm, patches_os, N, M)       # interpolation N → M
Dx_os     = ApplyDerivOS( comm, patches_os, N, M, 0)    # ∂/∂x  N → M
Dy_os     = ApplyDerivOS( comm, patches_os, N, M, 1)    # ∂/∂y  N → M
# Raw PU Laplacian at eval pts — no BC enforcement (we zero boundary rows below).
LapOS_raw = ApplyLapOS(comm, patches_os, N, M, [], [], np.array([]))

def D_op(u):
    """c·∇u at M eval pts."""
    return cx_eval * Dx_os(u) + cy_eval * Dy_os(u)

# ── Assemble dense E, D, and L by probing with basis vectors ─────────────────
# All ranks participate (operators use Allreduce internally).
if rank == 0:
    print(f"Assembling dense E ({M}×{N}), D ({M}×{N}), L ({M}×{N}) ...")
    t0 = MPI.Wtime()

E_dense = np.zeros((M, N))
D_dense = np.zeros((M, N))
L_dense = np.zeros((M, N))   # PU Laplacian at eval pts (for hyperviscosity)

for j in range(N):
    e_j = np.zeros(N)
    e_j[j] = 1.0
    E_dense[:, j] = E_op(e_j)
    D_dense[:, j] = D_op(e_j)
    L_dense[:, j] = LapOS_raw(e_j)

if rank == 0:
    print(f"Assembly complete in {MPI.Wtime()-t0:.1f}s")

# ── Boundary treatment ────────────────────────────────────────────────────────
# Eval boundary rows in E: standard interpolation (already assembled correctly).
# Eval boundary rows in D and L: set to zero — boundary nodes have zero time
# derivative (homogeneous Dirichlet).  Boundary rows of A_eff and L_eff are
# also zeroed explicitly after construction.
D_dense[eval_boundary_idx, :] = 0.0
L_dense[eval_boundary_idx, :] = 0.0

# ── Precompute A_eff = -(EᵀE)⁻¹ EᵀD  via Cholesky of EᵀE ───────────────────
# Done on all ranks (same dense data), no communication needed here.
if rank == 0:
    print("Computing A_eff = -(EᵀE)⁻¹ EᵀD via Cholesky ...")
    t0 = MPI.Wtime()

EtE = E_dense.T @ E_dense    # (N, N)  SPD
EtD = E_dense.T @ D_dense    # (N, N)
EtL = E_dense.T @ L_dense    # (N, N)

# Cholesky factorization of EᵀE (more stable than direct inversion)
cho_L, lower = spla.cho_factor(EtE, lower=True)
A_eff = -spla.cho_solve((cho_L, lower), EtD)   # (N, N)  pure advection
L_eff =  spla.cho_solve((cho_L, lower), EtL)   # (N, N)  effective Laplacian (Δ_eff)

# Zero boundary rows: du/dt = 0 at Dirichlet nodes
A_eff[boundary_nodes, :] = 0.0
L_eff[boundary_nodes, :] = 0.0

# Biharmonic hyperviscosity:  A_eff += -ε_hv · (Δ_eff)²
# Δ_eff has negative eigenvalues ⟹ (Δ_eff)² has positive eigenvalues
# ⟹ -ε_hv · (Δ_eff)² contributes negative real parts → pushes spectrum left.
# k⁴ scaling: high-frequency modes are damped, low-k physical modes barely touched.
A_eff -= eps_hv * (L_eff @ L_eff)

if rank == 0:
    # Report eigenvalue max real part as a stability check
    eigs = np.linalg.eigvals(A_eff)
    max_re = np.max(np.real(eigs))
    max_im = np.max(np.abs(np.imag(eigs)))
    print(f"A_eff precomputed in {MPI.Wtime()-t0:.1f}s")
    print(f"  max Re(λ) = {max_re:+.3e}  {'(stable ✓)' if max_re < 1e-8 else '(UNSTABLE — increase eps_hv)'}")
    print(f"  max |Im(λ)| = {max_im:.3e}  (no CFL constraint with matrix exponential)")

# ── Precompute matrix exponential propagator ──────────────────────────────────
# P = expm(dt · A_eff) propagates the ODE u' = A_eff u exactly over one step.
# This is the exact solution regardless of how large Im(λ) is — no stability
# constraint on dt.  Cost: one O(N³) expm call here, then O(N²) per step.
if rank == 0:
    print(f"Computing matrix exponential propagator P = expm(dt · A_eff) ...")
    t0 = MPI.Wtime()

P = spla.expm(dt * A_eff)
# Boundary rows: u stays zero at Dirichlet nodes every step.
P[boundary_nodes, :] = 0.0
P[boundary_nodes, boundary_nodes] = 1.0   # identity at boundary (u_bnd = 0 persists)

if rank == 0:
    print(f"expm complete in {MPI.Wtime()-t0:.1f}s")

# ── Initial condition ─────────────────────────────────────────────────────────
u = np.exp(-((nodes[:, 0]-x0)**2 + (nodes[:, 1]-y0)**2) / (2*sigma**2))
u[boundary_nodes] = 0.0

# ── Matrix exponential time stepping ─────────────────────────────────────────
# u^{n+1} = P u^n  — exact solution of the linear ODE, one matvec per step.

snapshots = [u.copy()]
times     = [0.0]

if rank == 0:
    print(f"\nStarting matrix-exponential time stepping ({n_steps} steps, dt={dt}) ...")
    t0 = MPI.Wtime()

for step in range(n_steps):
    u = P @ u

    t = (step + 1) * dt
    snapshots.append(u.copy())
    times.append(t)

    if rank == 0 and (step + 1) % 10 == 0:
        idx = np.argmax(u)
        print(f"  step {step+1:4d}/{n_steps}  t={t:.3f}  "
              f"peak={u.max():.4f}  loc=({nodes[idx,0]:.3f},{nodes[idx,1]:.3f})")

if rank == 0:
    t_wall = MPI.Wtime() - t0
    print(f"\nTime stepping complete: {n_steps} steps in {t_wall:.2f}s  "
          f"({t_wall/n_steps*1000:.1f}ms/step)")

    err = np.linalg.norm(u - snapshots[0]) / np.linalg.norm(snapshots[0])
    print(f"Relative L² error after one revolution: {err:.3e}")

    os.makedirs("figures", exist_ok=True)
    AnimateSolution(nodes, snapshots, times,
                    savepath="figures/advec_os.gif", fps=10)
    print("Done.")
