"""
HeatEquation.py  —  du/dt = Δu  on the unit square, homogeneous Dirichlet BCs.

Spatial:  LS-RBF-PUM (existing Setup / Operators infrastructure).
Temporal: backward Euler — fully implicit, unconditionally stable.

At each step the overdetermined LS system is

    interior rows:  (A_I - dt*A_L) c^{n+1}  =  A_I c^n
    boundary rows:   bc_scale * A_I_bc c^{n+1}  =  0

Normal equations are pre-factored (Cholesky) once; each step is then just
two matrix-vector products and a triangular solve.

Initial condition:  sin(πx)sin(πy) + 0.6 sin(3πx)sin(2πy)
Analytic solution:  each mode decays as exp(-(m²+n²)π²t).
The high-frequency mode (decay rate 13π²) vanishes ~6× faster than the
fundamental (2π²), so the GIF shows a textured initial shape smoothing into
a single symmetric bump.

Output: figures/heat.gif
"""

from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.linalg import cho_factor, cho_solve

from nodes.SquareDomain import MinEnergySquareOne
from source.PatchTiling import LarssonBox2D
from source.LSSetup import Setup
from source.Operators import PoissonRowMatrices, InterpolationRowMatrices, assemble_dense
from source.PUWeights import NormalizeWeights

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

plotfolder = "figures"

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
M          = 2000     # evaluation nodes
n_interp   = 40       # DOFs per patch
bc_scale   = 100.0    # Dirichlet row weight

dt         = 5e-4     # time step
T          = 0.08     # final time
n_steps    = int(T / dt)
save_every = 4        # GIF frame every save_every steps  →  ~40 frames at 12 fps

# ---------------------------------------------------------------------------
# Domain and patch setup
# ---------------------------------------------------------------------------
if rank == 0:
    print("Generating nodes and tiling...")
    eval_nodes, normals, groups = MinEnergySquareOne(M)
    centers, r = LarssonBox2D(H=0.2, xrange=(0, 1), yrange=(0, 1), delta=0.2)
else:
    eval_nodes = normals = groups = centers = r = None

eval_nodes = comm.bcast(eval_nodes, root=0)
normals    = comm.bcast(normals,    root=0)
groups     = comm.bcast(groups,     root=0)
centers    = comm.bcast(centers,    root=0)
r          = comm.bcast(r,          root=0)

bc_flags = np.empty(len(eval_nodes), dtype=str)
bc_flags[groups["boundary:all"]] = 'd'
bc_flags[groups["interior"]]     = 'i'

local_patches = Setup(
    comm, eval_nodes, normals, bc_flags, centers, r,
    n_interp=n_interp, node_layout='vogel', assignment='round_robin',
    K=64, n=16, m=48, eval_epsilon=0,
)
NormalizeWeights(comm, local_patches, M)

# ---------------------------------------------------------------------------
# Assemble dense operators (MPI-distributed, result is identical on all ranks)
# ---------------------------------------------------------------------------
N_patches = len(centers)

if rank == 0:
    print(f"Assembling operators  (M={M}, patches={N_patches}, n_interp={n_interp})...")

Rs_lap    = PoissonRowMatrices(local_patches)
Rs_interp = InterpolationRowMatrices(local_patches)

A_L = assemble_dense(comm, local_patches, Rs_lap,    M, N_patches, n_interp)
A_I = assemble_dense(comm, local_patches, Rs_interp, M, N_patches, n_interp)

# ---------------------------------------------------------------------------
# Time-stepping — rank 0 only
# ---------------------------------------------------------------------------
if rank == 0:
    int_idx = np.asarray(groups["interior"])
    bc_idx  = np.asarray(groups["boundary:all"])

    A_L_int    = A_L[int_idx, :]   # (N_int, N_col)  Laplacian at interior nodes
    A_I_int    = A_I[int_idx, :]   # (N_int, N_col)  interpolation at interior nodes
    A_I_bc     = A_I[bc_idx,  :]   # (N_bc,  N_col)  interpolation at boundary nodes

    N_col = N_patches * n_interp

    # ---- pre-factor normal equations (constant across all steps) --------
    #
    # Full LS system stacked:  [A_step_int; bc_scale * A_I_bc] c = [rhs; 0]
    #   A_step_int = A_I_int - dt * A_L_int
    #
    # Normal equations:
    #   (A_step_int^T A_step_int  +  bc_scale^2 A_I_bc^T A_I_bc) c
    #       = A_step_int^T (A_I_int c^n)
    #
    A_step_int = A_I_int - dt * A_L_int

    ATA  = (A_step_int.T @ A_step_int
            + bc_scale**2 * (A_I_bc.T @ A_I_bc))
    ATA += (1e-14 * np.trace(ATA) / N_col) * np.eye(N_col)
    cho  = cho_factor(ATA)

    # ---- initial condition and coefficient fit ---------------------------
    x, y = eval_nodes[:, 0], eval_nodes[:, 1]

    def u_exact(t):
        return (    np.sin(    np.pi * x) * np.sin(    np.pi * y) * np.exp( -2 * np.pi**2 * t)
                + 0.6 * np.sin(3 * np.pi * x) * np.sin(2 * np.pi * y) * np.exp(-13 * np.pi**2 * t))

    u0 = u_exact(0.0)

    # Fit initial coefficients: min ||A_I c - u0||_2
    c, *_ = np.linalg.lstsq(A_I, u0, rcond=None)

    # ---- march in time --------------------------------------------------
    print(f"Marching {n_steps} steps, dt={dt}, T={T} ...")
    frames = [(0.0, A_I @ c)]

    for step in range(n_steps):
        ATb = A_step_int.T @ (A_I_int @ c)
        c   = cho_solve(cho, ATb)

        t = (step + 1) * dt
        if (step + 1) % save_every == 0:
            frames.append((t, A_I @ c))

    t_f, u_h = frames[-1]
    err = np.max(np.abs(u_h[int_idx] - u_exact(t_f)[int_idx]))
    print(f"t={t_f:.4f}  max|u_h - u_exact| = {err:.3e}")
    print(f"Rendering {len(frames)}-frame GIF ...")

    # ---- render GIF -----------------------------------------------------
    triang = mtri.Triangulation(x, y)
    vmax   = float(np.max(np.abs(frames[0][1])))
    levels = np.linspace(-vmax, vmax, 40)

    fig, ax = plt.subplots(figsize=(5, 4.8))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.02)

    def draw(i):
        ax.clear()
        t_i, u_i = frames[i]
        ax.tricontourf(triang, u_i, levels=levels, cmap='RdBu_r', extend='both')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'du/dt = Δu     t = {t_i:.4f}', fontsize=11)

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=80)
    anim.save(f'{plotfolder}/heat.gif', writer=PillowWriter(fps=12))
    plt.close(fig)
    print(f"Saved {plotfolder}/heat.gif")
