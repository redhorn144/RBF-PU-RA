import numpy as np
from scipy.spatial import cKDTree
from mpi4py import MPI

from .Patch import Patch
from .PatchNodes import GenPatchNodes
from .RAHelpers import PhiFactors, StableMatricesLS
from .HaloComm import build_halo_comm
from .PUWeights import NormalizeWeights


def Setup(comm, eval_nodes, normals, bc_flags, centers, r, n_interp=30,
          node_layout='vogel', assignment='block_grid_2d',
          K=64, n=16, m=48, eval_epsilon=0):
    """
    Build local patches, halo communication graph, and normalised PU weights.

    Returns
    -------
    patches : list[Patch]   local patches with w_bar/gw_bar/lw_bar populated
    halo    : HaloComm      precomputed ownership and halo-exchange data
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    patch_nodes_base = GenPatchNodes(n_interp, r, eval_nodes.shape[1], node_layout)

    if rank == 0:
        phi_lus, Er, Es = PhiFactors(patch_nodes_base, K=K)
    else:
        phi_lus = Er = Es = None
    phi_lus = comm.bcast(phi_lus, root=0)
    Er      = comm.bcast(Er,      root=0)
    Es      = comm.bcast(Es,      root=0)

    if assignment == 'block_grid_2d':
        local_patch_indices = BlockGrid2D(rank, size, centers)
    elif assignment == 'round_robin':
        local_patch_indices = RoundRobin(rank, size, len(centers))
    else:
        raise ValueError(f"Unknown assignment method: {assignment!r}")

    # Each rank builds the tree independently (eval_nodes is identical on all ranks)
    tree = cKDTree(eval_nodes)

    local_patches = []
    for i in local_patch_indices:
        c           = centers[i]
        patch_nodes = patch_nodes_base + c
        eval_idxs   = np.asarray(tree.query_ball_point(c, r=r), dtype=int)

        local_eval_nodes = eval_nodes[eval_idxs]

        E, D, L = StableMatricesLS(
            local_eval_nodes, patch_nodes, phi_lus, Er, Es,
            n=n, m=m, eval_epsilon=eval_epsilon,
        )
        AdjustBoundaryMatrices(E, D, L, bc_flags[eval_idxs], normals[eval_idxs])

        local_patches.append(Patch(
            center            = c,
            radius            = r,
            eval_node_indices = eval_idxs,
            eval_nodes        = eval_nodes[eval_idxs],
            normals           = normals[eval_idxs],
            interp_nodes      = patch_nodes,
            bc_flags          = bc_flags[eval_idxs],
            E=E, D=D, L=L,
            global_pid        = i,
        ))

    # Build global patch→rank mapping (one Allreduce, all assignments deterministic)
    patch_rank_local = np.full(len(centers), -1, dtype=np.int32)
    for pid in local_patch_indices:
        patch_rank_local[pid] = rank
    patch_rank = np.empty(len(centers), dtype=np.int32)
    comm.Allreduce(patch_rank_local, patch_rank, op=MPI.MAX)

    halo = build_halo_comm(comm, local_patches, eval_nodes, centers, r, patch_rank)
    NormalizeWeights(local_patches, halo)

    return local_patches, halo


def RoundRobin(rank, size, num_patches):
    return [i for i in range(num_patches) if i % size == rank]


def BlockGrid2D(rank, size, centers):
    """
    Assign patches to ranks as contiguous rectangular blocks on the 2D patch
    grid, minimising inter-rank halo communication.

    Detects the (nx, ny) grid structure from the unique x/y coordinates in
    `centers` (works with any tiling that places centres on a regular grid,
    including LarssonBox2D).  Factorises `size` into (px, py) with px*py==size
    choosing the factorisation that makes each block as square as possible
    relative to the patch grid aspect ratio — this minimises the block
    perimeter (= inter-rank boundary) for a fixed block area.

    Patches are numbered in the same column-major order that LarssonBox2D
    produces: global_pid = ix * ny + iy.

    Parameters
    ----------
    rank    : int           this rank's MPI rank
    size    : int           total number of MPI ranks
    centers : (n_patches, 2) array of patch centre coordinates

    Returns
    -------
    list[int]  global patch indices assigned to this rank
    """
    xs = np.unique(np.round(centers[:, 0], decimals=10))
    ys = np.unique(np.round(centers[:, 1], decimals=10))
    nx, ny = len(xs), len(ys)

    if nx * ny != len(centers):
        raise ValueError(
            f"BlockGrid2D: centres do not form a regular {nx}×{ny} grid "
            f"(got {len(centers)} centres).  Use assignment='round_robin' for "
            f"irregular tilings.")

    px, py = _best_factorization(size, nx, ny)

    bx = rank // py   # x-block index for this rank
    by = rank  % py   # y-block index for this rank

    # np.array_split distributes remainder evenly across the first groups
    x_groups = np.array_split(np.arange(nx), px)
    y_groups = np.array_split(np.arange(ny), py)

    return [int(ix * ny + iy)
            for ix in x_groups[bx]
            for iy in y_groups[by]]


def _best_factorization(P, nx, ny):
    """
    Return (px, py) with px*py == P that minimises |log(block_width/block_height)|,
    i.e. the factorisation that produces the most square blocks on an nx×ny grid.
    """
    best, best_score = (1, P), float('inf')
    for px in range(1, P + 1):
        if P % px:
            continue
        py = P // px
        # block dimensions: nx/px wide, ny/py tall
        score = abs(np.log(nx * py) - np.log(ny * px))
        if score < best_score:
            best_score = score
            best = (px, py)
    return best


def AdjustBoundaryMatrices(E, D, L, bc_flags, full_normals):
    for i in range(E.shape[0]):
        if bc_flags[i] == 'd':
            D[:, i, :] = 0
            L[i, :]    = 0
        elif bc_flags[i] == 'n':
            D[:, i, :] = np.einsum('ki,k->i', D[:, i, :], full_normals[i])
            E[i, :]    = 0
            L[i, :]    = 0
