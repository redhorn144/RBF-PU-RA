"""
PoissonDriverDirect.py

Assembles the full LS-PUM system matrix A explicitly (dense) and solves the
least-squares problem  min ||A x - f||  with numpy.linalg.lstsq.

Purpose: diagnostic / ground-truth comparison for PoissonDriverLS.py.
Runs on a single MPI rank only (rank 0); other ranks do nothing.
"""

from mpi4py import MPI
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

from nodes.SquareDomain import MinEnergySquareOne
from source.LSSetup import SetupPatches
from source.PUWeights import NormalizeWeights
from source.PatchTiling import GenPatchTiling
from source.Operators import ApplyInterp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank != 0:
    comm.barrier()
    raise SystemExit


#------------------------------
# PARAMS  (match PoissonDriverLS.py)
#------------------------------

N_target = 1000
n        = 20
OVERLAP  = 1.5
N_SIDE   = 5


#------------------------------
# Domain
#------------------------------

nodes, normals, groups = MinEnergySquareOne(N_target)
interior_nodes   = nodes[groups['interior']]
boundary_nodes   = nodes[groups['boundary:all']]
boundary_normals = normals[groups['boundary:all']]

all_nodes = np.vstack([interior_nodes, boundary_nodes])
N     = len(all_nodes)
n_int = len(interior_nodes)
d     = all_nodes.shape[1]

print(f"Nodes: {N}  (interior: {n_int}, boundary: {N - n_int})")


#------------------------------
# BCs
#------------------------------

bc_flags  = np.full(len(boundary_nodes), 'd')
bc_values = np.zeros(len(boundary_nodes))


#------------------------------
# Patch tiling & RBF matrices
#------------------------------

centers, r = GenPatchTiling(interior_nodes, boundary_nodes,
                            tiling_choice=("grid", OVERLAP, N_SIDE), min_nodes=n)

# Single-rank: comm.Get_size() == 1 so every patch is local
local_patches = SetupPatches(
    comm, interior_nodes, boundary_nodes, boundary_normals, bc_flags,
    centers, r, n_interp=n, node_layout='vogel',
    K=64, n=16, m=48, eval_epsilon=0, strict=True
)

W = NormalizeWeights(comm, local_patches, N)

P = len(local_patches)
print(f"Patches: {P},  columns: {P * n},  rows: {N}")
print(f"System is {'over' if N >= P*n else 'under'}determined  ({N} x {P*n})")

if np.any(W == 0):
    zero_nodes = np.where(W == 0)[0]
    print(f"WARNING: {len(zero_nodes)} eval nodes have zero PU weight "
          f"(not covered by any patch) — those rows of A will be all-zero.")


#------------------------------
# Assemble A  (N x P*n, dense)
#------------------------------

A = np.zeros((N, P * n))

for p, patch in enumerate(local_patches):
    col = slice(p * n, (p + 1) * n)
    idx = patch.node_indices

    int_mask = (patch.bc_flags == 'i')
    bnd_mask = (patch.bc_flags == 'd')

    # Interior eval nodes: PUM Laplacian (product rule)
    #   A_j = diag(w_bar) L  +  2 Σ_k diag(gw_bar[:,k]) D[k]  +  diag(lw_bar) E
    if np.any(int_mask):
        rows = idx[int_mask]
        block = (  patch.w_bar[int_mask, None]  * patch.L[int_mask]
                 + patch.lw_bar[int_mask, None] * patch.E[int_mask])
        for k in range(d):
            block += 2.0 * patch.gw_bar[int_mask, k, None] * patch.D[k][int_mask]
        A[rows, col] += block

    # Dirichlet boundary eval nodes: PUM interpolation (BC enforcement)
    #   A_j = diag(w_bar) E
    if np.any(bnd_mask):
        rows = idx[bnd_mask]
        block = patch.w_bar[bnd_mask, None] * patch.E[bnd_mask]
        A[rows[:, None], col] += block

nonzero_rows = np.sum(np.any(A != 0, axis=1))
print(f"A assembled:  shape {A.shape},  non-zero rows: {nonzero_rows}")


#------------------------------
# RHS
#------------------------------

f = np.zeros(N)
f[:n_int] = np.sin(np.pi * interior_nodes[:, 0]) * np.sin(np.pi * interior_nodes[:, 1])


#------------------------------
# Solve  min ||A x - f||
#------------------------------

x, residuals, rank_A, sv = np.linalg.lstsq(A, f, rcond=None)

rnorm = np.linalg.norm(A @ x - f)
print(f"lstsq done:  matrix rank = {rank_A},  ||Ax - f|| = {rnorm:.3e},  "
      f"cond(A) ≈ {sv[0]/sv[-1]:.2e}")


#------------------------------
# Interpolate back to eval nodes
#------------------------------

Interp  = ApplyInterp(comm, local_patches, N)
u_eval  = Interp(x)


#------------------------------
# Plot
#------------------------------

tri     = Triangulation(all_nodes[:, 0], all_nodes[:, 1])
u_exact = (- np.sin(np.pi * all_nodes[:, 0])
             * np.sin(np.pi * all_nodes[:, 1])
           / (2.0 * np.pi**2))
err = u_eval - u_exact

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

tc = axes[0].tricontourf(tri, u_eval, levels=30, cmap='viridis')
plt.colorbar(tc, ax=axes[0])
axes[0].set_title('LS-PUM solution  $u_h$  (direct)')
axes[0].set_aspect('equal')

tc2 = axes[1].tricontourf(tri, u_exact, levels=30, cmap='viridis')
plt.colorbar(tc2, ax=axes[1])
axes[1].set_title('Exact  $u = -\\sin(\\pi x)\\sin(\\pi y)/(2\\pi^2)$')
axes[1].set_aspect('equal')

tc3 = axes[2].tricontourf(tri, err, levels=30, cmap='RdBu_r')
plt.colorbar(tc3, ax=axes[2])
axes[2].set_title(f'Error  (max |e| = {np.max(np.abs(err)):.2e})')
axes[2].set_aspect('equal')

plt.tight_layout()
plt.savefig('poisson_direct.png', dpi=150)
print("Saved poisson_direct.png")

comm.barrier()
