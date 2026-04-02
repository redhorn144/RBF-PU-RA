"""
AdvectionDriver.py — Weak-form PU-RBF advection with RK4 time stepping.

Semi-discrete system:
    M du/dt + C(b) u = 0

Spatial discretisation (non-nodal Gauss quadrature, same as PoissonDriver):
    C u  = E₀ᵀ [w * (b·∇u_h)]   =  Σ_l E₀ᵀ(w * b_l(x_q) * E_l u)
    Cᵀ u = Σ_l E_l^T(w * b_l(x_q) * E₀ u)

Skew-symmetric advection operator (½(C - Cᵀ)):
    - Valid for any b (divergence-free or not)
    - Produces an anti-Hermitian discrete operator → purely imaginary eigenvalues
    - RK4 is stable on the imaginary axis up to |λ dt| ≤ 2.83 (ideal pairing)

Mass treatment (lumped):
    m_lump[i] = (E₀ᵀ w)_i = Σ_q w_q Ψ_i(x_q)
    Valid because the PU property Σ_j Ψ_j(x) = 1 gives M·1 = E₀ᵀ w.
    This makes the mass diagonal → no linear solve per RK stage.

Time integrator:
    Classical RK4 (4th order, 4 function evaluations per step).
    CFL guideline:  dt ≤ 2.83 * h / |b|_max

Test problem (solid-body rotation benchmark):
    b(x, y) = (-(y − 0.5),  x − 0.5)   (angular velocity ω = 1)
    u₀ = Gaussian at (0.25, 0.5),  σ = 0.05
    Orbit radius 0.25 — stays well inside [0,1]² for T = 2π (one full rotation)
    Exact solution at T = 2π equals the initial condition.
"""

import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from nodes.SquareDomain import PoissonSquareOne
from source.Setup import Setup, SetupGaussEval
from source.Operators import (ApplyGaussEvalRow, ApplyGaussEvalAdj,
                               ApplyGaussDerivRow, ApplyGaussDerivAdj)
from source.Quadrature import GaussPointsAndWeights

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
H           = 0.03        # target node spacing
N_PATCH     = 50          # nodes per patch
P_QUAD      = 4           # GL points per direction  (degree 2p-1 = 7)
DT          = 0.01        # time step
T_END       = 2 * np.pi   # one full solid-body rotation
PRINT_EVERY = 50          # console update frequency (steps)

# Initial condition: Gaussian centred at (x0, y0), orbit radius r0 = dist to (0.5, 0.5)
X0, Y0, SIGMA = 0.25, 0.50, 0.05   # orbit radius ≈ 0.25, stays well inside domain

# ---------------------------------------------------------------------------
# Velocity field and exact solution
# ---------------------------------------------------------------------------
def b_field(pts):
    """Solid-body rotation around (0.5, 0.5). b = (-(y-0.5), x-0.5)."""
    return np.column_stack([-(pts[:, 1] - 0.5),
                              pts[:, 0] - 0.5])

def u_exact(pts, t):
    """Rotate the Gaussian centre by angle t and evaluate."""
    cx, cy = 0.5, 0.5
    dx0, dy0 = X0 - cx, Y0 - cy
    ct, st = np.cos(t), np.sin(t)
    xt = cx + ct * dx0 - st * dy0
    yt = cy + st * dx0 + ct * dy0
    r2 = (pts[:, 0] - xt)**2 + (pts[:, 1] - yt)**2
    return np.exp(-r2 / (2 * SIGMA**2))

# ---------------------------------------------------------------------------
# Node generation and MPI broadcast
# ---------------------------------------------------------------------------
if rank == 0:
    nodes, normals, groups = PoissonSquareOne(H)
    print(f"Rank 0: {nodes.shape[0]} nodes, h ≈ {H}.")
else:
    nodes = normals = groups = None

nodes   = comm.bcast(nodes,   root=0)
normals = comm.bcast(normals, root=0)
groups  = comm.bcast(groups,  root=0)

N        = nodes.shape[0]
bc_nodes = groups['boundary:all']

# ---------------------------------------------------------------------------
# Inflow boundary nodes: b(x_i) · n_i < 0
# ---------------------------------------------------------------------------
b_bc    = b_field(nodes[bc_nodes])
n_bc    = normals[bc_nodes]
bn_dot  = np.einsum('ij,ij->i', b_bc, n_bc)
inflow_nodes = bc_nodes[bn_dot < 0.0]

if rank == 0:
    print(f"Inflow nodes: {len(inflow_nodes)}  (b·n < 0 on boundary)")

# ---------------------------------------------------------------------------
# PU-RBF setup
# ---------------------------------------------------------------------------
patches, patches_for_rank = Setup(comm, nodes, normals, N_PATCH)
print(f"Rank {rank}: setup complete with {len(patches)} patches.")

gauss_pts, gauss_wts = GaussPointsAndWeights(comm, patches, N, p=P_QUAD)
SetupGaussEval(comm, patches, gauss_pts)

M_q = len(gauss_wts)
d   = 2

if rank == 0:
    b_max = np.linalg.norm(b_field(nodes), axis=1).max()
    print(f"|b|_max = {b_max:.4f},  CFL = dt·|b|/h ≈ {DT * b_max / H:.3f}")
    print(f"RK4 imaginary-axis limit: dt ≤ 2.83·h/|b| ≈ {2.83 * H / b_max:.4f}")

# ---------------------------------------------------------------------------
# Operator closures
# ---------------------------------------------------------------------------
E0   = ApplyGaussEvalRow(comm, patches, N, M_q)
E0T  = ApplyGaussEvalAdj(comm, patches, N, M_q)
Dk   = [ApplyGaussDerivRow(comm, patches, N, M_q, k) for k in range(d)]
DkT  = [ApplyGaussDerivAdj(comm, patches, N, M_q, k) for k in range(d)]

# Lumped mass vector: m_i = Σ_q w_q Ψ_i(x_q)
# (= M·1 via PU property Σ_j Ψ_j = 1)
m_lump = E0T(gauss_wts)   # (N,)

# Precompute b at Gauss points (constant in time for this problem)
b_gauss = b_field(gauss_pts)   # (M_q, 2)

def advection_rhs(u):
    """
    Skew-symmetric weak advection RHS:
        f = -M_lump⁻¹ · ½(C - Cᵀ) u

    C u  = E₀ᵀ (w * b·∇u)        forward: integrate (b·∇u_h) against PU basis
    Cᵀ u = Σ_l E_l^T(w * b_l * E₀ u)  adjoint transport term

    Inflow BC rows zeroed so those DOFs stay pinned at u_inflow = 0.
    """
    # b·∇u at Gauss points
    b_grad_u = np.zeros(M_q)
    for l in range(d):
        b_grad_u += b_gauss[:, l] * Dk[l](u)

    Cu = E0T(gauss_wts * b_grad_u)               # (N,)

    e0u = E0(u)                                   # E₀ u at Gauss points (M_q,)
    CuT = np.zeros(N)
    for l in range(d):
        CuT += DkT[l](gauss_wts * b_gauss[:, l] * e0u)

    rhs_vec = -0.5 * (Cu - CuT) / m_lump         # lumped-mass inversion
    rhs_vec[inflow_nodes] = 0.0                   # zero forcing at inflow
    return rhs_vec

# ---------------------------------------------------------------------------
# Classical RK4 step
# ---------------------------------------------------------------------------
def rk4_step(u, dt):
    k1 = advection_rhs(u)
    k2 = advection_rhs(u + 0.5 * dt * k1)
    k3 = advection_rhs(u + 0.5 * dt * k2)
    k4 = advection_rhs(u + dt * k3)
    u_new = u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    u_new[inflow_nodes] = 0.0   # re-enforce inflow after update
    return u_new

# ---------------------------------------------------------------------------
# Initial condition
# ---------------------------------------------------------------------------
u      = u_exact(nodes, 0.0)
u_init = u.copy()

if rank == 0:
    n_steps = int(np.ceil(T_END / DT))
    print(f"\nRK4 advection: T = {T_END:.4f},  dt = {DT},  n_steps = {n_steps}")
    t_wall_start = MPI.Wtime()

# ---------------------------------------------------------------------------
# Time integration
# ---------------------------------------------------------------------------
t    = 0.0
step = 0
n_steps = int(np.ceil(T_END / DT))

while step < n_steps:
    dt_actual = min(DT, T_END - t)
    u  = rk4_step(u, dt_actual)
    t += dt_actual
    step += 1

    if rank == 0 and step % PRINT_EVERY == 0:
        u_ex = u_exact(nodes, t)
        err  = np.linalg.norm(u - u_ex) / np.linalg.norm(u_ex)
        print(f"  step {step:5d}  t = {t:.4f}  ({100*t/T_END:.1f}%)"
              f"  ||u - u_ex|| / ||u_ex|| = {err:.3e}")

# ---------------------------------------------------------------------------
# Final diagnostics (rank 0 only)
# ---------------------------------------------------------------------------
if rank == 0:
    t_wall_end  = MPI.Wtime()
    u_ex_final  = u_exact(nodes, T_END)
    err_final   = np.linalg.norm(u - u_ex_final) / np.linalg.norm(u_ex_final)
    print(f"\nFinished {n_steps} steps in {t_wall_end - t_wall_start:.2f} s.")
    print(f"Relative L2 error at T = {T_END:.4f}:  {err_final:.3e}")

    # -----------------------------------------------------------------------
    # Side-by-side plot: initial / final / error
    # -----------------------------------------------------------------------
    tri = Triangulation(nodes[:, 0], nodes[:, 1])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    vmin = u_init.min(); vmax = u_init.max()

    tc0 = axes[0].tripcolor(tri, u_init, shading='gouraud', cmap='RdBu_r',
                            vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Initial  t = 0")
    axes[0].set_aspect('equal')
    plt.colorbar(tc0, ax=axes[0])

    tc1 = axes[1].tripcolor(tri, u, shading='gouraud', cmap='RdBu_r',
                            vmin=vmin, vmax=vmax)
    axes[1].set_title(f"Computed  t = {T_END:.3f}  (one rotation)")
    axes[1].set_aspect('equal')
    plt.colorbar(tc1, ax=axes[1])

    err_field = u - u_ex_final
    tc2 = axes[2].tripcolor(tri, err_field, shading='gouraud', cmap='RdBu_r')
    axes[2].set_title(f"Error  ||e||₂/||u_ex||₂ = {err_final:.2e}")
    axes[2].set_aspect('equal')
    plt.colorbar(tc2, ax=axes[2])

    plt.tight_layout()
    plt.savefig("advection_result.png", dpi=150)
    plt.close()
    print("Saved advection_result.png")