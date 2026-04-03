"""
Spectral convergence study for RBF-PU Poisson solver.

Fixes the global node set and sweeps nodes_per_patch, measuring
the L2 error against the eigenmode exact solution:

    u(x,y) = sin(m*pi*x) * sin(n*pi*y)
    -Lap u  = (m^2 + n^2)*pi^2 * sin(m*pi*x) * sin(n*pi*y)

with homogeneous Dirichlet BCs (u=0 on boundary).

Run with:
    mpiexec -n <nprocs> python PoissonSpectralConv.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpi4py import MPI

from nodes.SquareDomain import PoissonSquareOne
from source.Setup import Setup
from source.Operators import ApplyLap
from source.Solver import gmres

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# ── Eigenmode parameters ───────────────────────────────────────────────────────
M, N_MODE = 2, 2                        # wavenumbers
LAMBDA = (M**2 + N_MODE**2) * np.pi**2  # eigenvalue: -Lap u = LAMBDA * u

def u_exact(nodes):
    return np.sin(M * np.pi * nodes[:, 0]) * np.sin(N_MODE * np.pi * nodes[:, 1])

def rhs(nodes):
    # Lap(u_exact) = -LAMBDA * u_exact, so RHS = -LAMBDA * u_exact
    return -LAMBDA * u_exact(nodes)

# ── Sweep parameters ───────────────────────────────────────────────────────────
NODE_SPACING   = 0.025          # fixed global resolution (~1600 nodes)
NODES_PER_PATCH_LIST = [10, 15, 20, 30, 40, 55, 60, 80]
GMRES_TOL = 1e-8

# ── Generate fixed global node set (rank 0 only, then broadcast) ──────────────
if rank == 0:
    nodes, normals, groups = PoissonSquareOne(NODE_SPACING)
    print(f"Global node count: {nodes.shape[0]}")
else:
    nodes = normals = groups = None

nodes   = comm.bcast(nodes,   root=0)
normals = comm.bcast(normals, root=0)
groups  = comm.bcast(groups,  root=0)

bc_groups = np.array([groups['boundary:all']])
BCs       = np.array(["dirichlet"])

f_rhs = rhs(nodes)
f_rhs[bc_groups[0]] = 0.0   # Dirichlet rows → exact value (0 for this eigenmode)

errors = []

for npp in NODES_PER_PATCH_LIST:
    if rank == 0:
        print(f"\n── nodes_per_patch = {npp} ──────────────────────")

    patches, patches_for_rank = Setup(comm, nodes, normals, npp)
    Lap = ApplyLap(comm, patches, nodes.shape[0], bc_groups, BCs)

    solution, num_iters = gmres(comm, Lap, f_rhs, tol=GMRES_TOL, restart=150, maxiter=50)

    if rank == 0:
        u_ex = u_exact(nodes)
        err  = np.linalg.norm(solution - u_ex) / np.linalg.norm(u_ex)
        errors.append(err)
        print(f"  GMRES iters: {num_iters}  |  relative L2 error: {err:.3e}")

# ── Plot spectral convergence (rank 0 only) ────────────────────────────────────
if rank == 0:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(NODES_PER_PATCH_LIST, errors, "o-", color="steelblue", linewidth=2, markersize=7)
    ax.set_xlabel("Nodes per patch", fontsize=13)
    ax.set_ylabel("Relative $L_2$ error", fontsize=13)
    ax.set_title(
        f"Spectral convergence — eigenmode $\\sin({M}\\pi x)\\sin({N_MODE}\\pi y)$\n"
        f"($N={nodes.shape[0]}$ global nodes, node spacing $h={NODE_SPACING}$)",
        fontsize=11,
    )
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.xaxis.set_major_locator(plt.MultipleLocator(10))
    fig.tight_layout()
    savepath = "figures/poisson_spectral_conv.png"
    fig.savefig(savepath, dpi=150)
    print(f"\nConvergence plot saved to {savepath}")
