"""
AdvectionDiffusion.py  —  du/dt = ν·Δu + a·∇u  on the unit square,
                           homogeneous Dirichlet BCs.

Spatial:  LS-RBF-PUM with AdvectionDiffusionRowMatrices.
Temporal: backward Euler — unconditionally stable, no CFL constraint.

Time-stepping system at each step (identical structure to HeatEquation.py):

    interior rows:  (A_I - dt * A_AD) c^{n+1}  =  A_I c^n
    boundary rows:   bc_scale * A_I_bc c^{n+1}  =  0

Normal equations pre-factored (Cholesky) once; each step is two matvecs
+ one triangular solve.

Parameters chosen to give Pe ≈ 50 (advection-dominated but not extreme):
    nu = 0.01,  a = (1.0, 0.5),  Gaussian IC at (0.15, 0.25), σ = 0.07.
The bump advects diagonally while diffusing; as it nears the boundaries it
is absorbed by the homogeneous BCs.

Output: figures/advection_diffusion.gif
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
from source.Operators import AdvectionDiffusionRowMatrices, InterpolationRowMatrices, assemble_dense

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

plotfolder = "figures"

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
M          = 2000
n_interp   = 40
bc_scale   = 100.0

nu         = 0.01                    # diffusion coefficient
a          = np.array([1.0, 0.5])    # advection velocity  (Pe ≈ |a|/nu ≈ 112)

dt         = 1e-3                    # backward Euler is stable for any dt
T          = 0.60
n_steps    = int(T / dt)
save_every = 15                      # ~40 frames total

# ---------------------------------------------------------------------------
# Domain and patch setup
# ---------------------------------------------------------------------------
if rank == 0:
    print(f"nu={nu}  a={a}  Pe≈{np.linalg.norm(a)/nu:.0f}")
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

# ---------------------------------------------------------------------------
# Assemble dense operators
# ---------------------------------------------------------------------------
N_patches = len(centers)

if rank == 0:
    print(f"Assembling operators  (M={M}, patches={N_patches}, n_interp={n_interp})...")

Rs_ad     = AdvectionDiffusionRowMatrices(local_patches, a, nu=nu)
Rs_interp = InterpolationRowMatrices(local_patches)

A_AD = assemble_dense(comm, local_patches, Rs_ad,    M, N_patches, n_interp)
A_I  = assemble_dense(comm, local_patches, Rs_interp, M, N_patches, n_interp)

# ---------------------------------------------------------------------------
# Time-stepping — rank 0 only
# ---------------------------------------------------------------------------
if rank == 0:
    int_idx = np.asarray(groups["interior"])
    bc_idx  = np.asarray(groups["boundary:all"])

    A_AD_int   = A_AD[int_idx, :]
    A_I_int    = A_I[int_idx,  :]
    A_I_bc     = A_I[bc_idx,   :]

    N_col = N_patches * n_interp

    # ---- pre-factor normal equations ----
    A_step_int = A_I_int - dt * A_AD_int

    ATA  = (A_step_int.T @ A_step_int
            + bc_scale**2 * (A_I_bc.T @ A_I_bc))
    ATA += (1e-14 * np.trace(ATA) / N_col) * np.eye(N_col)
    cho  = cho_factor(ATA)

    # ---- initial condition: Gaussian bump ----
    x, y  = eval_nodes[:, 0], eval_nodes[:, 1]
    x0, y0, sig = 0.15, 0.25, 0.07
    u0 = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sig**2))
    u0[bc_idx] = 0.0

    # Fit initial coefficients
    c, *_ = np.linalg.lstsq(A_I, u0, rcond=None)

    # ---- free-space analytic reference (no BCs, valid while bump is interior) ----
    def u_free(t):
        sig2t = sig**2 + 2 * nu * t
        return (sig**2 / sig2t) * np.exp(
            -((x - x0 - a[0]*t)**2 + (y - y0 - a[1]*t)**2) / (2 * sig2t))

    # ---- time loop ----
    print(f"Marching {n_steps} steps, dt={dt}, T={T} ...")
    frames = [(0.0, A_I @ c)]

    for step in range(n_steps):
        ATb = A_step_int.T @ (A_I_int @ c)
        c   = cho_solve(cho, ATb)

        t = (step + 1) * dt
        if (step + 1) % save_every == 0:
            frames.append((t, A_I @ c))

    # Error vs free-space solution at t=0.3 (bump still well inside domain)
    t_check = 0.3
    step_check = int(t_check / dt)
    c_check = np.linalg.lstsq(A_I, u0, rcond=None)[0]
    for s in range(step_check):
        ATb = A_step_int.T @ (A_I_int @ c_check)
        c_check = cho_solve(cho, ATb)
    u_check = A_I @ c_check
    err = np.max(np.abs(u_check[int_idx] - u_free(t_check)[int_idx]))
    print(f"Max |u_h - u_free| at t={t_check} (interior): {err:.3e}")
    print(f"Rendering {len(frames)}-frame GIF ...")

    # ---- GIF ----
    triang = mtri.Triangulation(x, y)
    vmax   = float(np.max(frames[0][1]))
    levels = np.linspace(0, vmax, 40)

    fig, ax = plt.subplots(figsize=(5, 4.8))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.02)

    def draw(i):
        ax.clear()
        t_i, u_i = frames[i]
        ax.tricontourf(triang, u_i, levels=levels, cmap='hot_r', extend='both')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(
            f'ν·Δu + a·∇u     ν={nu},  a={tuple(a)},  t={t_i:.3f}',
            fontsize=9)

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=80)
    anim.save(f'{plotfolder}/advection_diffusion.gif', writer=PillowWriter(fps=12))
    plt.close(fig)
    print(f"Saved {plotfolder}/advection_diffusion.gif")
