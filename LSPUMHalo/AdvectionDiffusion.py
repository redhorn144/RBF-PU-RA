"""
AdvectionDiffusion.py — du/dt = ν·Δu − a·∇u on the unit square, homogeneous Dirichlet BCs.
BDF3/EXT3 IMEX: diffusion implicit (Helmholtz solve, SAS-preconditioned PCG),
                advection explicit (3rd-order extrapolation).
Output: figures/advection_diffusion.gif
"""

from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.animation import FuncAnimation, PillowWriter

from nodes.SquareDomain import MinEnergySquareOne
from source_halo.PatchTiling import LarssonBox2D
from source_halo.LSSetup import Setup
from source_halo.Operators import (HelmholtzStepRowMatrices, AdvectionRowMatrices,
                                    InterpolationRowMatrices, GenMatFreeOps)
from source_halo.Solvers import GenIterativeSolver

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

M, n_interp, bc_scale = 2000, 40, 100.0
nu = 0.01
a  = np.array([1.0, 0.5])
dt, T, save_every = 1e-3, 0.60, 15
n_steps = int(T / dt)

# --- domain ---
if rank == 0:
    print(f"nu={nu}  a={a}  Pe≈{np.linalg.norm(a)/nu:.0f}")
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

# --- matvec for explicit advection evaluation: g = -a·∇u at owned nodes ---
matvec_adv, _ = GenMatFreeOps(patches, AdvectionRowMatrices(patches, a), halo, n_interp)

# --- three IMEX implicit solvers (SAS preconditioner factored once each) ---
def make_solver(alpha):
    Rs = HelmholtzStepRowMatrices(patches, alpha, dt, nu, bc_scale)
    return GenIterativeSolver(comm, patches, halo, n_interp, Rs,
                              preconditioner='sas', atol=1e-8, maxiter=2000)

solve_bdf1 = make_solver(1.0)
solve_bdf2 = make_solver(1.5)
solve_bdf3 = make_solver(11.0 / 6.0)

# --- initial condition and free-space reference ---
x, y = eval_nodes[:, 0], eval_nodes[:, 1]
x0, y0, sig = 0.15, 0.25, 0.07
u0 = np.exp(-((x - x0)**2 + (y - y0)**2) / (2*sig**2))
u0[groups["boundary:all"]] = 0.0

def u_free(t):
    sig2t = sig**2 + 2*nu*t
    return (sig**2 / sig2t) * np.exp(
        -((x - x0 - a[0]*t)**2 + (y - y0 - a[1]*t)**2) / (2*sig2t))

# fit IC
f0 = u0[owned].copy()
f0[bc_owned] = 0.0
solve_ic = GenIterativeSolver(comm, patches, halo, n_interp,
                               InterpolationRowMatrices(patches, bc_scale),
                               preconditioner='block_jacobi', atol=1e-12)
local_cs, *_ = solve_ic(f0)
c_local = np.concatenate(local_cs)

# --- BDF3/EXT3 time loop ---
if rank == 0:
    print(f"Marching {n_steps} steps  dt={dt}  T={T}  (BDF3/EXT3 IMEX)")

u_hist = [reconstruct(local_cs)[owned]]
g_hist = [matvec_adv(c_local)]

t_check = 0.3
frames, err_check = [(0.0, reconstruct(local_cs))], None

for step in range(n_steps):
    if step == 0:
        f      = u_hist[-1] + dt * g_hist[-1]
        solver = solve_bdf1
    elif step == 1:
        f      = (2*u_hist[-1] - 0.5*u_hist[-2]
                  + dt*(2*g_hist[-1] - g_hist[-2]))
        solver = solve_bdf2
    else:
        f      = (3*u_hist[-1] - 1.5*u_hist[-2] + (1/3)*u_hist[-3]
                  + dt*(3*g_hist[-1] - 3*g_hist[-2] + g_hist[-3]))
        solver = solve_bdf3

    f[bc_owned] = 0.0
    local_cs, *_ = solver(f)
    c_local = np.concatenate(local_cs)

    u_hist.append(reconstruct(local_cs)[owned])
    g_hist.append(matvec_adv(c_local))
    if len(u_hist) > 3:
        u_hist.pop(0)
        g_hist.pop(0)

    t = (step + 1) * dt
    if (step + 1) % save_every == 0:
        u_h = reconstruct(local_cs)
        frames.append((t, u_h))
        if abs(t - t_check) < dt / 2:
            err_check = np.max(np.abs(u_h[groups["interior"]] - u_free(t)[groups["interior"]]))

if rank == 0:
    if err_check is not None:
        print(f"Max |u_h - u_free| at t={t_check} (interior): {err_check:.3e}")
    print(f"Rendering {len(frames)}-frame GIF ...")

    triang = mtri.Triangulation(x, y)
    vmax   = float(np.max(frames[0][1]))
    levels = np.linspace(0, vmax, 40)
    fig, ax = plt.subplots(figsize=(5, 4.8))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.02)

    def draw(i):
        ax.clear()
        t_i, u_i = frames[i]
        ax.tricontourf(triang, u_i, levels=levels, cmap='hot_r', extend='both')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(f'ν·Δu − a·∇u     ν={nu},  a={tuple(a)},  t={t_i:.3f}', fontsize=9)

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=80)
    anim.save('figures/advection_diffusion.gif', writer=PillowWriter(fps=12))
    plt.close(fig)
    print("Saved figures/advection_diffusion.gif")
