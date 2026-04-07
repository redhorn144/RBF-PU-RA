"""
Eigenvalue analysis of the oversampled PU advection operator following
Tominec, Nazarov & Larsson (2024), "Stability estimates for radial basis
function methods applied to linear scalar conservation laws".

Theory
------
RBF approximant:  u_h(x,t) = Σ_j c_j(t) φ_j(x),   c = Φ_XX⁻¹ u

Require the PDE  ∂u/∂t + c·∇u = 0  at M > N eval points in the ℓ₂ sense:

    E · du/dt  =  -D · u           (M equations, N unknowns)

where
    E  (M×N)  =  PU interpolation at eval pts     (Φ_YX Φ_XX⁻¹ in global RBF)
    D  (M×N)  =  PU advection     at eval pts     (c·∇ applied to u_h at Y)

ℓ₂ normal equations  →  effective N×N semidiscrete ODE:

    du/dt  =  A_eff · u,    A_eff  =  -(EᵀE)⁻¹ Eᵀ D

This is the matrix whose eigenvalues are plotted.

When M = N and E = I  (square Kansa):  A_eff = -D  (standard result).

Figures
-------
  figures/advec_spectrum_os.png  —  three panels:
    Left   : spectrum of A_sq  (square collocation, -D)
    Centre : spectrum of A_eff (oversampled LS, Tominec et al. formula)
    Right  : eigenvalue real-part CDFs for direct comparison

Usage
-----
    mpiexec -n 4 python AdvecSpectrumOS.py
"""
import numpy as np
from mpi4py import MPI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from nodes.SquareDomain import PoissonSquareOne
from source.Setup import Setup, SetupOversampled
from source.Operators import ApplyDeriv, ApplyDerivOS, ApplyInterpOS

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# ── Node / eval-point generation ──────────────────────────────────────────────
# Keep N small: dense eigen-decomposition is O(N³).
h_sol  = 0.075
h_eval = 0.02   # oversampling ratio β = (h_sol/h_eval)² ≈ 1.82

if rank == 0:
    nodes, normals, sol_groups = PoissonSquareOne(h_sol)
    eval_all, _, eval_groups   = PoissonSquareOne(h_eval)

    interior_eval = eval_all[eval_groups['interior']]
    boundary_sol  = nodes[sol_groups['boundary:all']]
    eval_pts      = np.vstack([interior_eval, boundary_sol])

    N = nodes.shape[0]
    M = eval_pts.shape[0]
    boundary_nodes    = sol_groups['boundary:all']
    eval_boundary_idx = np.arange(len(interior_eval), M)

    print(f"N = {N} solution nodes,  M = {M} eval pts  (β = {M/N:.2f})")
else:
    nodes = normals = sol_groups = eval_pts = None
    boundary_nodes = eval_boundary_idx = None

nodes             = comm.bcast(nodes,             root=0)
normals           = comm.bcast(normals,           root=0)
sol_groups        = comm.bcast(sol_groups,        root=0)
eval_pts          = comm.bcast(eval_pts,          root=0)
boundary_nodes    = comm.bcast(boundary_nodes,    root=0)
eval_boundary_idx = comm.bcast(eval_boundary_idx, root=0)

N = nodes.shape[0]
M = eval_pts.shape[0]

# ── Velocity field: solid-body rotation about (0.5, 0.5) ─────────────────────
omega    = 1.0
cx_nodes = -omega * (nodes[:,   1] - 0.5)
cy_nodes =  omega * (nodes[:,   0] - 0.5)
cx_eval  = -omega * (eval_pts[:, 1] - 0.5)
cy_eval  =  omega * (eval_pts[:, 0] - 0.5)

# ── Square patch setup ────────────────────────────────────────────────────────
if rank == 0:
    print("Setting up square patches ...")
patches_sq, _ = Setup(comm, nodes, normals, 30, eval_epsilon=0)

Dx_sq = ApplyDeriv(comm, patches_sq, N, 0, [], [])
Dy_sq = ApplyDeriv(comm, patches_sq, N, 1, [], [])

def advec_sq(u):
    """Square PU: c·∇u at N solution nodes, identity at boundary."""
    v = cx_nodes * Dx_sq(u) + cy_nodes * Dy_sq(u)
    v[boundary_nodes] = u[boundary_nodes]
    return v

# ── Oversampled patch setup ───────────────────────────────────────────────────
if rank == 0:
    print("Setting up oversampled patches ...")
patches_os, _ = SetupOversampled(comm, nodes, eval_pts, normals, 30,
                                  overlap=3, eval_epsilon=0)

# E: PU interpolation at eval pts (N → M)
E_op = ApplyInterpOS(comm, patches_os, N, M)

# D: PU advection at eval pts (N → M), no BCs — raw operator at interior
Dx_os = ApplyDerivOS(comm, patches_os, N, M, 0)
Dy_os = ApplyDerivOS(comm, patches_os, N, M, 1)

def advec_os(u):
    """Oversampled PU: c·∇u at M eval pts (interior only; boundary handled below)."""
    return cx_eval * Dx_os(u) + cy_eval * Dy_os(u)

# ── Assemble dense matrices ───────────────────────────────────────────────────
if rank == 0:
    print(f"Assembling dense E ({M}×{N}) and D ({M}×{N}) by probing ...")

E_dense = np.zeros((M, N))
D_dense = np.zeros((M, N))
A_sq    = np.zeros((N, N))

for j in range(N):
    e_j = np.zeros(N)
    e_j[j] = 1.0
    E_dense[:, j] = E_op(e_j)
    D_dense[:, j] = advec_os(e_j)
    A_sq[:, j]    = advec_sq(e_j)

# Boundary eval rows in E and D: enforce identity mapping
# (eval boundary pts coincide with boundary solution nodes, so E row = e_j and D row = e_j)
E_dense[eval_boundary_idx, :] = 0.0
D_dense[eval_boundary_idx, :] = 0.0
for k, (ei, si) in enumerate(zip(eval_boundary_idx, boundary_nodes)):
    E_dense[ei, si] = 1.0
    D_dense[ei, si] = 1.0   # identity: boundary "advection" = identity (Dirichlet)

# ── A_eff = -(EᵀE)⁻¹ Eᵀ D  (Tominec et al. 2024, eq. for LS semidiscrete ODE) ──
if rank == 0:
    print("Computing A_eff = -(EᵀE)⁻¹ EᵀD ...")
    EtE = E_dense.T @ E_dense    # (N, N)  SPD
    EtD = E_dense.T @ D_dense    # (N, N)

    # Solve EᵀE · X = EᵀD  →  X = (EᵀE)⁻¹ EᵀD
    A_eff = -np.linalg.solve(EtE, EtD)   # (N, N)

    # ── Eigenvalues ───────────────────────────────────────────────────────────
    print("Computing eigenvalues ...")
    eigs_sq  = np.linalg.eigvals(A_sq)
    eigs_eff = np.linalg.eigvals(A_eff)

    max_re_sq  = np.max(np.real(eigs_sq))
    max_re_eff = np.max(np.real(eigs_eff))

    print(f"Square system  max Re(λ) = {max_re_sq:+.4e}  "
          f"({'UNSTABLE' if max_re_sq > 1e-10 else 'stable'})")
    print(f"LS (A_eff)     max Re(λ) = {max_re_eff:+.4e}  "
          f"({'UNSTABLE' if max_re_eff > 1e-10 else 'stable'})")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: square system spectrum
    ax = axes[0]
    ax.scatter(np.real(eigs_sq), np.imag(eigs_sq),
               s=10, alpha=0.7, color='steelblue', zorder=3)
    ax.axvline(0, color='k', lw=0.8, ls='--')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel(r"Re($\lambda$)")
    ax.set_ylabel(r"Im($\lambda$)")
    ax.set_title(f"Square collocation $-D$  ($N={N}$)\n"
                 f"max Re = {max_re_sq:+.2e}")
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Panel 2: LS effective operator spectrum
    ax = axes[1]
    ax.scatter(np.real(eigs_eff), np.imag(eigs_eff),
               s=10, alpha=0.7, color='darkorange', zorder=3)
    ax.axvline(0, color='k', lw=0.8, ls='--')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel(r"Re($\lambda$)")
    ax.set_ylabel(r"Im($\lambda$)")
    ax.set_title(r"LS effective $-(E^\top E)^{-1} E^\top D$  ($N\times N$)"
                 f"\nmax Re = {max_re_eff:+.2e}   ($\\beta={M/N:.2f}$)")
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Panel 3: CDF of Re(λ) — shows how much of the spectrum crosses the axis
    ax = axes[2]
    for eigs, label, color in [
        (eigs_sq,  "Square $-D$",                         "steelblue"),
        (eigs_eff, r"LS $-(E^\top E)^{-1}E^\top D$",     "darkorange"),
    ]:
        re_sorted = np.sort(np.real(eigs))
        ax.plot(re_sorted, np.linspace(0, 1, len(re_sorted)),
                label=label, color=color)
    ax.axvline(0, color='k', lw=0.8, ls='--', label='Re = 0')
    ax.set_xlabel(r"Re($\lambda$)")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("CDF of Re($\\lambda$)\n(fraction left of imaginary axis = stable modes)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Advection operator eigenvalue analysis  —  Tominec et al. (2024) formulation\n"
        f"$h_{{sol}}={h_sol}$,  $h_{{eval}}={h_eval}$,  "
        f"$N={N}$,  $M={M}$,  $\\beta={M/N:.2f}$",
        fontsize=11,
    )
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/advec_spectrum_os.png", dpi=150)
    print("Saved figures/advec_spectrum_os.png")
