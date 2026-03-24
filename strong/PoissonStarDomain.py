"""
Poisson solver on a 5-pointed star domain.

Manufactured solution (method of manufactured solutions):
    u(x, y) = exp(x + y)
   -Lap u   = -2 * exp(x + y)

Non-homogeneous Dirichlet BCs: u = u_exact on the star boundary.

Produces a three-panel figure:
    [exact solution | RBF-PU solution | pointwise error]

Run with:
    mpiexec -n <nprocs> python PoissonStarDomain.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.tri import Triangulation
from scipy.spatial import Delaunay
from mpi4py import MPI

from nodes.StrangeDomain import PoissonStarDomain
from source.Setup import Setup
from source.Operators import ApplyLap
from source.Solver import gmres

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# ── Manufactured solution ──────────────────────────────────────────────────────
def u_exact(nodes):
    return np.exp(nodes[:, 0] + nodes[:, 1])

def rhs(nodes):
    return -2.0 * np.exp(nodes[:, 0] + nodes[:, 1])

# ── Parameters ─────────────────────────────────────────────────────────────────
NODE_SPACING    = 0.01
NODES_PER_PATCH = 30
GMRES_TOL       = 1e-8

# ── Build nodes (rank 0, then broadcast) ──────────────────────────────────────
if rank == 0:
    nodes, normals, groups, star_verts = PoissonStarDomain(NODE_SPACING)
    print(f"Global node count: {nodes.shape[0]}")
else:
    nodes = normals = groups = star_verts = None

nodes      = comm.bcast(nodes,      root=0)
normals    = comm.bcast(normals,    root=0)
groups     = comm.bcast(groups,     root=0)
star_verts = comm.bcast(star_verts, root=0)

bc_groups = np.array([groups['boundary:all']])
BCs       = np.array(["dirichlet"])

# ── Build RHS with non-homogeneous Dirichlet BCs ───────────────────────────────
f_rhs = rhs(nodes)
f_rhs[bc_groups[0]] = u_exact(nodes[bc_groups[0]])

# ── Setup patches and operator ─────────────────────────────────────────────────
patches, patches_for_rank = Setup(comm, nodes, normals, NODES_PER_PATCH)
Lap = ApplyLap(comm, patches, nodes.shape[0], bc_groups, BCs)

# ── Solve ──────────────────────────────────────────────────────────────────────
solution, num_iters = gmres(comm, Lap, f_rhs, tol=GMRES_TOL, restart=150, maxiter=200)

# ── Post-process and plot (rank 0 only) ────────────────────────────────────────
if rank == 0:
    u_ex = u_exact(nodes)
    err  = np.linalg.norm(solution - u_ex) / np.linalg.norm(u_ex)
    print(f"GMRES iters: {num_iters}  |  relative L2 error: {err:.3e}")

    # Build Delaunay triangulation and mask triangles outside the star
    tri     = Delaunay(nodes)
    triang  = Triangulation(nodes[:, 0], nodes[:, 1], tri.simplices)
    star_path = Path(np.vstack([star_verts, star_verts[0]]))
    centroids = nodes[tri.simplices].mean(axis=1)
    triang.set_mask(~star_path.contains_points(centroids))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    panels = [
        ("Exact solution  $u = e^{x+y}$", u_ex,                     "viridis"),
        ("RBF-PU solution",                solution,                  "viridis"),
        ("Pointwise error  $|u - u_h|$",  np.abs(solution - u_ex),  "magma"),
    ]

    for ax, (title, data, cmap) in zip(axes, panels):
        tc = ax.tripcolor(triang, data, cmap=cmap, shading="gouraud")
        fig.colorbar(tc, ax=ax, shrink=0.85)
        ax.set_title(title, fontsize=11)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.suptitle(
        f"Poisson on 5-pointed star domain  "
        f"($N={nodes.shape[0]}$ nodes,  "
        f"$n_{{\\mathrm{{pp}}}}={NODES_PER_PATCH}$,  "
        f"rel. $L_2$ error $= {err:.2e}$)",
        fontsize=12,
    )
    fig.tight_layout()
    savepath = "figures/poisson_star_domain.png"
    fig.savefig(savepath, dpi=150)
    print(f"Figure saved to {savepath}")
