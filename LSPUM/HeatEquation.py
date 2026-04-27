"""
HeatEquation.py — du/dt = Δu on the unit square, homogeneous Dirichlet BCs.
Backward Euler, LS-RBF-PUM (halo-exchange backend), RAS-preconditioned PCG.
Output: figures/heat.gif
"""

from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.animation import FuncAnimation, PillowWriter

from nodes.SquareDomain import MinEnergySquareOne
from source_halo.PatchTiling import LarssonBox2D
from source_halo.LSSetup import Setup
from source_halo.Operators import HeatStepRowMatrices, InterpolationRowMatrices
from source_halo.Solvers import GenIterativeSolver

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

M, n_interp, bc_scale = 2000, 40, 100.0
dt, T, save_every = 5e-4, 0.08, 4
n_steps = int(T / dt)

# --- domain ---
if rank == 0:
    eval_nodes, normals, groups = MinEnergySquareOne(M)
    centers, r = LarssonBox2D(H=0.2, xrange=(0, 1), yrange=(0, 1), delta=0.2)
else:
    eval_nodes = normals = groups = centers = r = None

eval_nodes = comm.bcast(eval_nodes, root=0)
normals    = comm.bcast(normals,    root=0)
groups     = comm.bcast(groups,     root=0)
centers    = comm.bcast(centers,    root=0)
r          = comm.bcast(r,          root=0)

bc_flags = np.empty(M, dtype=str)
bc_flags[groups["boundary:all"]] = 'd'
bc_flags[groups["interior"]]     = 'i'

patches, halo = Setup(comm, eval_nodes, normals, bc_flags, centers, r,
                      n_interp=n_interp, node_layout='vogel',
                      K=64, n=16, m=48, eval_epsilon=0)

owned    = halo.owned_indices
bc_owned = np.where(bc_flags[owned] == 'd')[0]

def reconstruct(local_cs):
    U_local = np.zeros(M)
    for p, c in zip(patches, local_cs):
        U_local[p.eval_node_indices] += p.w_bar * (p.E @ c)
    U = np.zeros(M)
    comm.Allreduce(U_local, U, op=MPI.SUM)
    return U

# --- exact solution and initial condition ---
x, y = eval_nodes[:, 0], eval_nodes[:, 1]
def u_exact(t):
    return (    np.sin(    np.pi * x) * np.sin(    np.pi * y) * np.exp( -2*np.pi**2*t)
            + 0.6*np.sin(3*np.pi * x) * np.sin(2*np.pi * y) * np.exp(-13*np.pi**2*t))

# fit IC via interpolation solve
solve_ic = GenIterativeSolver(comm, patches, halo, n_interp,
                               InterpolationRowMatrices(patches, bc_scale),
                               preconditioner='block_jacobi', atol=1e-12)
local_cs, *_ = solve_ic(u_exact(0.0)[owned])

# --- time-step solver (RAS preconditioner built once here) ---
solve = GenIterativeSolver(comm, patches, halo, n_interp,
                           HeatStepRowMatrices(patches, dt, bc_scale),
                           preconditioner='ras', atol=1e-8, maxiter=2000)

if rank == 0:
    print(f"Marching {n_steps} steps  dt={dt}  T={T}")

frames = [(0.0, reconstruct(local_cs))]
for step in range(n_steps):
    f_owned = reconstruct(local_cs)[owned]
    f_owned[bc_owned] = 0.0
    local_cs, *_ = solve(f_owned)
    if (step + 1) % save_every == 0:
        frames.append(((step + 1) * dt, reconstruct(local_cs)))

t_f, u_h = frames[-1]
if rank == 0:
    err = np.max(np.abs(u_h[groups["interior"]] - u_exact(t_f)[groups["interior"]]))
    print(f"t={t_f:.4f}  max|u_h - u_exact| = {err:.3e}  ({len(frames)} frames)")
    print("Rendering GIF ...")

    triang = mtri.Triangulation(x, y)
    vmax   = float(np.max(np.abs(frames[0][1])))
    levels = np.linspace(-vmax, vmax, 40)
    fig, ax = plt.subplots(figsize=(5, 4.8))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.02)

    def draw(i):
        ax.clear()
        t_i, u_i = frames[i]
        ax.tricontourf(triang, u_i, levels=levels, cmap='RdBu_r', extend='both')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(f'du/dt = Δu     t = {t_i:.4f}', fontsize=11)

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=80)
    anim.save('figures/heat.gif', writer=PillowWriter(fps=12))
    plt.close(fig)
    print("Saved figures/heat.gif")
