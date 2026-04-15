from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial import cKDTree
from source.PatchTiling import BoxGridTiling2D
from nodes.SquareDomain import PoissonSquareOne, MinEnergySquareOne
from source.LSSetup import Setup
from source.Operators import GenLapMatrix, GenInterp
from scipy.sparse.linalg import lsqr
from source.PUWeights import NormalizeWeights

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

plotfolder = "figures"

M = 1000
beta = 6
alpha = 2

if rank == 0:
    print("Generating nodes...")
    #eval_nodes, normals, groups = PoissonSquareOne(r=0.02)
    eval_nodes, normals, groups = MinEnergySquareOne(M)

    print("Tiling domain...")
    centers, r = BoxGridTiling2D(eval_nodes, n_interp=50, oversample_factor=beta, overlap=alpha)

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
    n_interp=30, node_layout='vogel', assignment='round_robin',
    K=64, n=16, m=48, eval_epsilon=0,
)

NormalizeWeights(comm, local_patches, M)

n_interp  = local_patches[0].interp_nodes.shape[0] if local_patches else 0
n_interp  = comm.allreduce(n_interp, op=MPI.MAX)
N_patches = len(centers)

print(f"[{rank}] Building system matrix ({M} x {N_patches * n_interp})...")
A = GenLapMatrix(comm, local_patches, M, N_patches, n_interp, groups["boundary:all"])

if rank == 0:
    # RHS: interior = manufactured forcing, boundary = 0 (sin vanishes on unit-square boundary)
    f = np.zeros(M)
    xi = eval_nodes[groups["interior"]]
    f[groups["interior"]] = -2 * np.pi**2 * np.sin(np.pi * xi[:, 0]) * np.sin(np.pi * xi[:, 1])

    print(f"Solving ({A.shape[0]} x {A.shape[1]}, nnz={A.nnz})...")
    c, *_ = lsqr(A, f, atol=1e-12, btol=1e-12, iter_lim=50000)
else:
    c = None

c = comm.bcast(c, root=0)

local_cs = [c[p.global_pid * n_interp : (p.global_pid + 1) * n_interp] for p in local_patches]

interp = GenInterp(comm, local_patches, M=M)
U = interp(local_cs)

if rank == 0:
    u_exact = np.sin(np.pi * eval_nodes[:, 0]) * np.sin(np.pi * eval_nodes[:, 1])
    error   = np.abs(U - u_exact)
    print(f"Max error: {error.max():.2e},  L2 error: {np.sqrt(np.mean(error**2)):.2e}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sc0 = axes[0].scatter(eval_nodes[:, 0], eval_nodes[:, 1], c=U,     s=5, cmap='viridis')
    sc1 = axes[1].scatter(eval_nodes[:, 0], eval_nodes[:, 1], c=error, s=5, cmap='hot_r')
    plt.colorbar(sc0, ax=axes[0], label='U (computed)')
    plt.colorbar(sc1, ax=axes[1], label='|error|')
    axes[0].set_aspect('equal'); axes[0].set_title('PUM Poisson solution')
    axes[1].set_aspect('equal'); axes[1].set_title('Pointwise error')
    plt.tight_layout()
    plt.savefig(plotfolder + '/poisson_solution.png', dpi=150)
    print("Saved poisson_solution.png")



