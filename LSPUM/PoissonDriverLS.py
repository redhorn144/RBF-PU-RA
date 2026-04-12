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
from source.Operators import ApplyLap, ApplyLapT, ApplyInterp
from source.LSQR import lsqr

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


#------------------------------
# PARAMS
#------------------------------

N_target = 1000      # requested number of eval nodes
n        = 20        # interpolation nodes per patch

# Tiling: grid_tiling computes r = overlap * h * sqrt(d)/2  where h = domain/n_side.
# n_side=5, overlap=1.5 in the unit square  →  h=0.2, r ≈ 0.21
OVERLAP = 1.5
N_SIDE  = 5


#------------------------------
# Domain Generation
#------------------------------
if rank == 0:
    nodes, normals, groups = MinEnergySquareOne(N_target)
    interior_nodes   = nodes[groups['interior']]
    boundary_nodes   = nodes[groups['boundary:all']]
    boundary_normals = normals[groups['boundary:all']]
else:
    interior_nodes   = None
    boundary_nodes   = None
    boundary_normals = None

interior_nodes   = comm.bcast(interior_nodes,   root=0)
boundary_nodes   = comm.bcast(boundary_nodes,   root=0)
boundary_normals = comm.bcast(boundary_normals, root=0)

all_nodes = np.vstack([interior_nodes, boundary_nodes])
N     = len(all_nodes)
n_int = len(interior_nodes)


#------------------------------
# Boundary conditions
#------------------------------

# Homogeneous Dirichlet on all boundaries
bc_flags  = np.full(len(boundary_nodes), 'd')
bc_values = np.zeros(len(boundary_nodes))


#------------------------------
# Patch tiling & RBF matrices
#------------------------------

if rank == 0:
    centers, r = GenPatchTiling(interior_nodes, boundary_nodes,
                                tiling_choice=("grid", OVERLAP, N_SIDE), min_nodes=n)
else:
    centers = None
    r       = None
centers = comm.bcast(centers, root=0)
r       = comm.bcast(r,       root=0)

local_patches = SetupPatches(
    comm, interior_nodes, boundary_nodes, boundary_normals, bc_flags,
    centers, r, n_interp=n, node_layout='vogel',
    K=64, n=16, m=48, eval_epsilon=0, strict=True
)

# NormalizeWeights requires the total eval-node count N
W = NormalizeWeights(comm, local_patches, N)

n_local = len(local_patches)   # patches owned by this rank


#------------------------------
# Operators
#------------------------------

Lap    = ApplyLap  (comm, local_patches, N)
LapT   = ApplyLapT (local_patches)
Interp = ApplyInterp(comm, local_patches, N)


#------------------------------
# RHS  (interior: PDE forcing; boundary rows: Dirichlet = 0)
#------------------------------

f = np.zeros(N)
f[:n_int] = np.sin(np.pi * interior_nodes[:, 0]) * np.sin(np.pi * interior_nodes[:, 1])
# f[n_int:] = bc_values = 0  (already zero)


#------------------------------
# Solve
#------------------------------

x, itn, rnorm = lsqr(comm, Lap, LapT, f, atol=1e-10, btol=1e-10, show=True)

if rank == 0:
    print(f"LSQR converged in {itn} iterations,  ||r|| = {rnorm:.3e}")


#------------------------------
# Interpolate back to eval nodes
#------------------------------

u_eval = Interp(x)    # PUM-interpolated solution at all N eval nodes


#------------------------------
# Plot (rank 0 only)
#------------------------------

if rank == 0:
    tri = Triangulation(all_nodes[:, 0], all_nodes[:, 1])

    # Exact solution: Δu = f = sin(πx)sin(πy)  →  u = -sin(πx)sin(πy)/(2π²)
    u_exact = (- np.sin(np.pi * all_nodes[:, 0])
                 * np.sin(np.pi * all_nodes[:, 1])
               / (2.0 * np.pi**2))
    err = u_eval - u_exact

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    tc = axes[0].tricontourf(tri, u_eval, levels=30, cmap='viridis')
    plt.colorbar(tc, ax=axes[0])
    axes[0].set_title('LS-PUM solution  $u_h$')
    axes[0].set_aspect('equal')

    tc2 = axes[1].tricontourf(tri, err, levels=30, cmap='RdBu_r')
    plt.colorbar(tc2, ax=axes[1])
    axes[1].set_title(
        f'Error  $u_h - u_{{exact}}$   '
        f'(max |e| = {np.max(np.abs(err)):.2e})'
    )
    axes[1].set_aspect('equal')

    plt.tight_layout()
    plt.savefig('poisson_solution.png', dpi=150)
    print("Saved poisson_solution.png")
