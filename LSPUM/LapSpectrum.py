from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import cKDTree
from source.PatchTiling import BoxGridTiling2D, LarssonBox2D
from nodes.SquareDomain import PoissonSquareOne, MinEnergySquareOne
from source.LSSetup import Setup
from source.Operators import 
from source.PUWeights import NormalizeWeights

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

plotfolder = "figures"

M = 2500
n_interp = 60
eval_eps = 0
bc_scale = 10


if rank == 0:
    print("Generating nodes...")
    #eval_nodes, normals, groups = PoissonSquareOne(r=0.02)
    eval_nodes, normals, groups = MinEnergySquareOne(M)

    print("Tiling domain...")
    #centers, r = BoxGridTiling2D(eval_nodes, n_interp=90, oversample_factor=beta, overlap=alpha)
    centers, r = LarssonBox2D(H=0.2, xrange=(0, 1), yrange=(0, 1), delta=0.2)

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
eval_nodes = comm.bcast(eval_nodes, root=0)
normals    = comm.bcast(normals, root=0)
groups     = comm.bcast(groups, root=0)
centers    = comm.bcast(centers, root=0)
r          = comm.bcast(r, root=0)


bc_flags = np.empty(len(eval_nodes), dtype=str)
bc_flags[groups["boundary:all"]] = 'd'

bc_flags[groups["interior"]] = 'i'

local_patches = Setup(
    comm, eval_nodes, normals, bc_flags, centers, r,
    n_interp=n_interp, node_layout='vogel', assignment='round_robin',
    K=64, n=16, m=48, eval_epsilon=eval_eps,
)

NormalizeWeights(comm, local_patches, M)

