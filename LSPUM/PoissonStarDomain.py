"""
LS-RBF-PUM Poisson solver on a 5-pointed star domain.

Manufactured solution:
    u(x, y) = exp(x + y)
   -Lap u   = -2 * exp(x + y)

Non-homogeneous Dirichlet BCs: u = u_exact on the star boundary.

Produces a three-panel figure:
    [exact solution | LS-RBF-PUM solution | pointwise error]

Run with:
    mpiexec -n <nprocs> python PoissonStarDomain.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.tri import Triangulation
from scipy.spatial import Delaunay, cKDTree
from mpi4py import MPI

from nodes.StrangeDomain import MinEnergyStarDomain
from source_halo.PatchTiling import LarssonBox2D
from source_halo.LSSetup import Setup
from source_halo.Operators import PoissonRowMatrices
from source_halo.Solvers import GenIterativeSolver

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ── Manufactured solution ──────────────────────────────────────────────────────
# LSPUM row matrices encode +Δ, so the system solved is  Δu = f_interior.
# For u = exp(x+y),  Δu = 2 exp(x+y).
def u_exact(nodes):
    return np.exp(nodes[:, 0] + nodes[:, 1])

def laplacian_u_exact(nodes):
    return 2.0 * np.exp(nodes[:, 0] + nodes[:, 1])

# ── PUM-weighted global reconstruction (matches HeatEquation.py) ───────────────
def reconstruct(comm, patches, local_cs, M):
    U_local = np.zeros(M)
    for p, c in zip(patches, local_cs):
        U_local[p.eval_node_indices] += p.w_bar * (p.E @ c)
    U = np.zeros(M)
    comm.Allreduce(U_local, U, op=MPI.SUM)
    return U

# ── Parameters ─────────────────────────────────────────────────────────────────
M_TARGET     = 6000      # target number of min-energy nodes in the star
n_interp     = 30
H            = 0.1
delta        = 0.2
bc_scale     = 100.0
atol = btol  = 1e-12
maxiter      = 20000
preconditioner = 'block_jacobi'
reorth       = True
oversample   = 2.0       # min ratio of eval nodes / n_interp per patch (LS over-det.)

# ── Build nodes and patch centers (rank 0, then broadcast) ────────────────────
if rank == 0:
    nodes, normals, groups, star_verts = MinEnergyStarDomain(M_TARGET)
    centers_all, r = LarssonBox2D(H=H, xrange=(0, 1), yrange=(0, 1), delta=delta)

    tree = cKDTree(nodes)
    ball  = [np.asarray(tree.query_ball_point(c, r), dtype=int) for c in centers_all]
    counts = np.array([len(b) for b in ball])

    # Stage 1: keep over-determined patches (needed for RAS Cholesky stability)
    min_eval = int(np.ceil(oversample * n_interp))
    keep = counts >= min_eval

    # Stage 2: rescue any star node not covered by the Stage-1 set.
    # A node is in the POU only if at least one kept patch's ball contains it.
    # Promote rejected patches (greedily, max coverage gain) until every node
    # is covered. Rescue candidates must still be square-LS-feasible (>= n_interp).
    covered = np.zeros(nodes.shape[0], dtype=bool)
    for i in np.where(keep)[0]:
        covered[ball[i]] = True
    while not covered.all():
        cand = np.where(~keep & (counts >= n_interp))[0]
        gains = np.array([np.sum(~covered[ball[i]]) for i in cand])
        if len(cand) == 0 or gains.max() == 0:
            n_orphan = int((~covered).sum())
            raise RuntimeError(
                f"{n_orphan} star nodes are uncoverable by the candidate tiling. "
                f"Reduce H, increase M_TARGET, or lower oversample.")
        winner = cand[int(np.argmax(gains))]
        keep[winner] = True
        covered[ball[winner]] = True
    centers = centers_all[keep]

    n_rescue = int(((counts < min_eval) & keep).sum())
    print(f"M={nodes.shape[0]}  n_interp={n_interp}  "
          f"patches={len(centers)}/{len(centers_all)} kept "
          f"(stage1 min_eval={min_eval}, +{n_rescue} rescued, "
          f"actual min={counts[keep].min()}, max={counts[keep].max()})  "
          f"bc_scale={bc_scale}  ranks={size}")
else:
    nodes = normals = groups = star_verts = centers = r = None

nodes      = comm.bcast(nodes,      root=0)
normals    = comm.bcast(normals,    root=0)
groups     = comm.bcast(groups,     root=0)
star_verts = comm.bcast(star_verts, root=0)
centers    = comm.bcast(centers,    root=0)
r          = comm.bcast(r,          root=0)

M = nodes.shape[0]

# ── BC flags and RHS with non-homogeneous Dirichlet values ────────────────────
bc_flags = np.empty(M, dtype=str)
bc_flags[groups['interior']]     = 'i'
bc_flags[groups['boundary:all']] = 'd'

f = laplacian_u_exact(nodes)
bdy = groups['boundary:all']
# Boundary row in PoissonRowMatrices is  bc_scale * w_bar * E, so the
# forward operator at a boundary node evaluates to  bc_scale * u(i).
# To enforce u(i) = u_exact(i), match RHS to bc_scale * u_exact(i).
f[bdy] = bc_scale * u_exact(nodes[bdy])

# ── Build local patches and halo ──────────────────────────────────────────────
patches, halo = Setup(comm, nodes, normals, bc_flags, centers, r,
                      n_interp=n_interp, node_layout='vogel',
                      assignment='round_robin',
                      K=64, n=16, m=48, eval_epsilon=0)

# ── Solve LS system ────────────────────────────────────────────────────────────
Rs = PoissonRowMatrices(patches, bc_scale)
solve = GenIterativeSolver(comm, patches, halo, n_interp, Rs,
                           preconditioner=preconditioner,
                           atol=atol, btol=btol, maxiter=maxiter,
                           reorth=reorth)
f_owned = f[halo.owned_indices]
local_cs, itn, rnorm = solve(f_owned)
solution = reconstruct(comm, patches, local_cs, M)

# ── Post-process and plot (rank 0 only) ────────────────────────────────────────
if rank == 0:
    u_ex   = u_exact(nodes)
    err_l2 = np.sqrt(np.mean((solution - u_ex) ** 2))
    err    = np.linalg.norm(solution - u_ex) / np.linalg.norm(u_ex)
    print(f"LSQR iters: {itn}  |  rnorm: {rnorm:.3e}  |  "
          f"L2 error: {err_l2:.3e}  |  relative L2 error: {err:.3e}")

    tri    = Delaunay(nodes)
    triang = Triangulation(nodes[:, 0], nodes[:, 1], tri.simplices)
    star_path = Path(np.vstack([star_verts, star_verts[0]]))
    centroids = nodes[tri.simplices].mean(axis=1)
    triang.set_mask(~star_path.contains_points(centroids))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    panels = [
        ("Exact solution  $u = e^{x+y}$",     u_ex,                     "viridis"),
        ("LS-RBF-PUM solution",                solution,                  "viridis"),
        ("Pointwise error  $|u - u_h|$",      np.abs(solution - u_ex),  "magma"),
    ]

    for ax, (title, data, cmap) in zip(axes, panels):
        tc = ax.tripcolor(triang, data, cmap=cmap, shading="gouraud")
        fig.colorbar(tc, ax=ax, shrink=0.85)
        ax.set_title(title, fontsize=11)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.suptitle(
        f"LS-RBF-PUM Poisson on 5-pointed star domain  "
        f"($N={M}$ nodes,  "
        f"$n_{{\\mathrm{{interp}}}}={n_interp}$,  "
        f"rel. $L_2$ error $= {err:.2e}$)",
        fontsize=12,
    )
    fig.tight_layout()
    savepath = "figures/poisson_star_domain.png"
    fig.savefig(savepath, dpi=150)
    print(f"Figure saved to {savepath}")
