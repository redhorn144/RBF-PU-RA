"""
Oversampled strong-form RBF-PU-RA Poisson solver.

Solves -Δu = f on [0,1]² with homogeneous Dirichlet BCs using oversampled
collocation: the PDE is enforced at M > N collocation points, yielding an
overdetermined system A (M×N) solved via LSQR (Paige & Saunders 1982).

Usage:
    mpiexec -n 4 python PoissonDriverOS.py
"""
import numpy as np
from mpi4py import MPI
from nodes.SquareDomain import PoissonSquareOne
from source.Setup import SetupOversampled
from source.Operators import ApplyLapOS, ApplyLapOSTranspose
from source.Solver import lsqr
from source.Plotter import PlotSolution

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# -----------------------------------------------------------------------
# Node generation (rank 0 only, then broadcast)
# -----------------------------------------------------------------------
# h_sol  : solution node spacing → N global unknowns
# h_eval : collocation point spacing (finer) → M > N eval points
# Oversampling ratio ≈ (h_sol / h_eval)^2 in 2D
h_sol  = 0.05
h_eval = 0.025   # gives M/N ≈ (0.025/0.018)² ≈ 1.93

if rank == 0:
    nodes, normals, sol_groups = PoissonSquareOne(h_sol)
    eval_nodes_int, _, eval_groups_int = PoissonSquareOne(h_eval)
    # eval_pts = interior eval pts + boundary solution nodes
    # (boundary pts are shared so BCs are enforced identically)
    interior_eval = eval_nodes_int[eval_groups_int['interior']]
    boundary_sol  = nodes[sol_groups['boundary:all']]
    eval_pts = np.vstack([interior_eval, boundary_sol])

    N = nodes.shape[0]
    M = eval_pts.shape[0]
    N_int_sol  = len(sol_groups['interior'])
    N_bnd      = len(sol_groups['boundary:all'])
    # In eval_pts: interior eval pts come first, then boundary (appended)
    eval_interior_idx  = np.arange(len(interior_eval))
    eval_boundary_idx  = np.arange(len(interior_eval), M)

    print(f"Solution nodes N = {N}  (interior {N_int_sol}, boundary {N_bnd})")
    print(f"Eval points    M = {M}  (interior {len(interior_eval)}, boundary {N_bnd})")
    print(f"Oversampling ratio M/N = {M/N:.2f}")
else:
    nodes = None
    normals = None
    sol_groups = None
    eval_pts = None
    eval_interior_idx = None
    eval_boundary_idx = None

nodes             = comm.bcast(nodes, root=0)
normals           = comm.bcast(normals, root=0)
sol_groups        = comm.bcast(sol_groups, root=0)
eval_pts          = comm.bcast(eval_pts, root=0)
eval_interior_idx = comm.bcast(eval_interior_idx, root=0)
eval_boundary_idx = comm.bcast(eval_boundary_idx, root=0)

N = nodes.shape[0]
M = eval_pts.shape[0]

# -----------------------------------------------------------------------
# Patch setup (oversampled)
# -----------------------------------------------------------------------
patches, patches_for_rank = SetupOversampled(
    comm, nodes, eval_pts, normals,
    nodes_per_patch=30, overlap=3, eval_epsilon=0,
)
if rank == 0:
    print(f"Patch setup complete.")

# -----------------------------------------------------------------------
# Build operators
# -----------------------------------------------------------------------
sol_bnd_nodes  = [sol_groups['boundary:all']]   # solution-space boundary indices
eval_bnd_nodes = [eval_boundary_idx]            # eval-space boundary indices
BCs = np.array(["dirichlet"])

# Forward: ℝᴺ → ℝᴹ
LapOS = ApplyLapOS(comm, patches, N, M, eval_bnd_nodes, sol_bnd_nodes, BCs)
# Adjoint: ℝᴹ → ℝᴺ
LapOST = ApplyLapOSTranspose(comm, patches, N, M, eval_bnd_nodes, sol_bnd_nodes, BCs)

# -----------------------------------------------------------------------
# Build RHS
# -----------------------------------------------------------------------
f_eval = np.zeros(M)
f_eval[eval_interior_idx] = (
    -2 * np.pi**2
    * np.sin(np.pi * eval_pts[eval_interior_idx, 0])
    * np.sin(np.pi * eval_pts[eval_interior_idx, 1])
)
f_eval[eval_boundary_idx] = 0.0

# -----------------------------------------------------------------------
# Solve via LSQR: min‖LapOS(u) - f_eval‖
# -----------------------------------------------------------------------
if rank == 0:
    print("Starting LSQR ...")
    t0 = MPI.Wtime()

solution, num_iters = lsqr(comm, LapOS, LapOST, f_eval, tol=1e-4)

if rank == 0:
    t1 = MPI.Wtime()
    print(f"LSQR converged in {num_iters} iterations, {t1-t0:.2f}s")

    u_exact = np.sin(np.pi * nodes[:, 0]) * np.sin(np.pi * nodes[:, 1])
    error = np.linalg.norm(solution - u_exact) / np.linalg.norm(u_exact)
    print(f"Relative L2 error: {error:.2e}")
    PlotSolution(nodes, solution)
