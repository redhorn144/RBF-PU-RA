from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import cKDTree
from source.PatchTiling import BoxGridTiling2D
from nodes.SquareDomain import PoissonSquareOne, MinEnergySquareOne
from source.LSSetup import Setup
from source.Operators import GenLap

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

plotfolder = "figures"

M = 500
beta = 2
alpha = 1.2

if rank == 0:
    print("Generating nodes...")
    #eval_nodes, normals, groups = PoissonSquareOne(r=0.02)
    eval_nodes, normals, groups = MinEnergySquareOne(M)

    print("Tiling domain...")
    centers, r = BoxGridTiling2D(eval_nodes, n_interp=30, oversample_factor=beta, overlap=alpha)

    # Count eval_nodes per patch
    tree = cKDTree(eval_nodes)
    node_counts = np.array([len(idx) for idx in tree.query_ball_point(centers, r)])
    print(f"Nodes per patch — min: {node_counts.min()}, max: {node_counts.max()}, mean: {node_counts.mean():.1f}")

    # Plot patches and nodes
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(eval_nodes[:, 0], eval_nodes[:, 1], s=2, color='steelblue', zorder=2, label='eval nodes')
    for c in centers:
        circle = mpatches.Circle(c, r, fill=False, edgecolor='tomato', linewidth=0.8, alpha=0.6)
        ax.add_patch(circle)
    ax.scatter(centers[:, 0], centers[:, 1], s=15, color='tomato', zorder=3, label='patch centers')
    ax.set_aspect('equal')
    ax.set_title(f'{len(centers)} patches, r={r:.4f}  |  nodes/patch: min={node_counts.min()}, max={node_counts.max()}')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(plotfolder + '/patches.png', dpi=150)
    #plt.show()
    print("Saved patches.png")
else:
    eval_nodes = None
    normals = None
    groups = None
    centers = None
    r = None
comm.bcast(eval_nodes, root=0)
comm.bcast(normals, root=0)
comm.bcast(groups, root=0)
comm.bcast(centers, root=0)
comm.bcast(r, root=0)

bc_flags = np.empty(len(eval_nodes), dtype=str)
bc_flags[groups["boundary:all"]] = 'd'

bc_flags[groups["interior"]] = 'i'

local_patches = Setup(
    comm, eval_nodes, normals, groups, centers, r,
    n_interp=30, node_layout='vogel', assignment='round_robin',
    K=64, n=16, m=48, eval_epsilon=0,
)

lap = GenLap(comm, local_patches, M=M)

local_us = []

for patch in local_patches:
    u_j = -2*np.pi**2*np.sin(np.pi * patch.interp_nodes[:, 0]) * np.sin(np.pi * patch.interp_nodes[:, 1])
    local_us.append(u_j)

UY = lap(local_us)

if rank == 0:
    # Plot UY
    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(eval_nodes[:, 0], eval_nodes[:, 1], c=UY, s=5, cmap='viridis')
    plt.colorbar(sc, ax=ax, label='Laplace(U)')
    ax.set_aspect('equal')
    ax.set_title('Laplace(U) at eval nodes')
    plt.tight_layout()
    plt.savefig(plotfolder + '/laplace_u.png', dpi=150)
    #plt.show()
    print("Saved laplace_u.png")



