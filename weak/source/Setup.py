import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from BaseHelpers import *
from Patch import Patch
from scipy.spatial import cKDTree
from RAHelpers import StableFlatMatrices, PHSEvalRows
from mpi4py import MPI
from PUWeights import NormalizeWeights, C2Weight, C2WeightGradient

###################################
# General setup function to create patches, compute their matrices, and normalize weights
###################################
def Setup(comm, nodes, normals, nodes_per_patch, overlap = 3):
    rank = comm.Get_rank()
    
    if rank == 0:
        centers, radii, patch_node_inds = SetupPatches(nodes, 50, overlap=3)
    else:
        centers = None
        radii = None
        patch_node_inds = None

    centers = comm.bcast(centers, root=0)
    radii = comm.bcast(radii, root=0)
    patch_node_inds = comm.bcast(patch_node_inds, root=0)

    num_patches = len(centers)
    patches_for_rank = [i for i in range(num_patches) if i % comm.Get_size() == rank]

    patches = []
    for i in patches_for_rank:
        patch_nodes = nodes[patch_node_inds[i]]
        patch_normals = normals[patch_node_inds[i]]
        patch_center = centers[i]
        patch_radius = radii[i]
        patch_nodes_indices = patch_node_inds[i]
        Patch_Phi, Patch_D, Patch_L = StableFlatMatrices(patch_nodes)
        patch = Patch(center=patch_center, radius=patch_radius, node_indices=patch_nodes_indices, normals=patch_normals,
                        nodes=patch_nodes, Phi=Patch_Phi, D=Patch_D, L=Patch_L, w_bar=None, gw_bar=None, lw_bar=None)
        patches.append(patch)
    
    # Normalize PU weights across all ranks
    NormalizeWeights(comm, patches, patches_for_rank, nodes)

    print(f"Rank {rank} has {len(patches)} patches.")
    
    return patches, patches_for_rank



###################################
# SetupGaussEval: per-patch PHS evaluation data at global Gauss points.
# Must be called after Setup() and GaussPointsAndWeights().
###################################
def SetupGaussEval(comm, patches, gauss_pts):
    """
    For each patch, find the global Gauss points inside its support, compute
    normalized PU weights there, and build PHS eval + gradient matrices.

    PU normalization W(x_q) is computed locally from allgathered patch
    geometries — no Allreduce needed beyond the allgather of centers/radii.

    Parameters
    ----------
    comm      : MPI communicator
    patches   : list of Patch objects (w_bar etc. already set by Setup)
    gauss_pts : (M_q, 2) global Gauss point coordinates from GaussPointsAndWeights
    """
    M_q = gauss_pts.shape[0]
    d   = gauss_pts.shape[1]

    # Allgather all patch geometries so W(x_q) can be computed without Allreduce
    local_centers = np.array([p.center for p in patches]) if patches else np.empty((0, d))
    local_radii   = np.array([p.radius for p in patches]) if patches else np.empty(0)
    all_centers   = np.vstack(comm.allgather(local_centers))
    all_radii     = np.concatenate(comm.allgather(local_radii))

    gauss_tree = cKDTree(gauss_pts)

    for patch in patches:
        gidx = np.array(gauss_tree.query_ball_point(patch.center, patch.radius), dtype=int)

        if len(gidx) == 0:
            n_local = len(patch.node_indices)
            patch.gauss_local_idx = np.empty(0, dtype=int)
            patch.E_gauss         = np.empty((0, n_local))
            patch.GE_gauss        = np.empty((d, 0, n_local))
            patch.w_bar_gauss     = np.empty(0)
            patch.gw_bar_gauss    = np.empty((0, d))
            continue

        local_gauss = gauss_pts[gidx]

        # W(x_q) = Σ_p' w_{p'}(x_q) — computed locally from allgathered geometry
        W  = np.zeros(len(gidx))
        gW = np.zeros((len(gidx), d))
        for c, r in zip(all_centers, all_radii):
            dist = np.linalg.norm(local_gauss - c, axis=1)
            inside = dist < r
            if not np.any(inside):
                continue
            W[inside]  += C2Weight(local_gauss[inside], c, r)
            gW[inside] += C2WeightGradient(local_gauss[inside], c, r)
        W = np.maximum(W, 1e-300)

        w_raw  = C2Weight(local_gauss, patch.center, patch.radius)
        gw_raw = C2WeightGradient(local_gauss, patch.center, patch.radius)

        patch.gauss_local_idx = gidx
        patch.w_bar_gauss     = w_raw / W
        patch.gw_bar_gauss    = gw_raw / W[:, None] - w_raw[:, None] * gW / W[:, None]**2
        patch.E_gauss, patch.GE_gauss = PHSEvalRows(local_gauss, patch.nodes)


####################################
#SetupPatches: called on rank zero to generate the patches and distribute to other ranks
####################################

def SetupPatches(nodes, nodes_per_patch, overlap = 3):

    centers = GenCenters(nodes, nodes_per_patch, overlap)
    patch_node_inds, radii = GenPatches(nodes, centers, nodes_per_patch)

    return centers, radii, patch_node_inds

###################################
# Helper function to generate the patches and their associated data
###################################

def GenCenters(nodes, nodes_per_patch, overlap):
    P = int(overlap * nodes.shape[0] // nodes_per_patch)
    d = nodes.shape[1]
    centers = np.empty((P, d))

    ran_idx = np.random.choice(len(nodes))
    centers[0] = nodes[ran_idx]

    minmaxdist = np.linalg.norm(nodes - centers[0], axis=1)

    for i in range(1, P):
        idx = np.argmax(minmaxdist)
        centers[i] = nodes[idx]
        dist = np.linalg.norm(nodes - centers[i], axis=1)
        np.minimum(minmaxdist, dist, out=minmaxdist)

    return centers

def GenPatches(nodes, centers, nodes_per_patch, radius_scale=1.5):
    tree = cKDTree(nodes)
    patches = np.zeros((centers.shape[0], nodes_per_patch), dtype=int)
    radii = np.zeros(centers.shape[0])


    for i, center in enumerate(centers):
        # Find the nearest nodes to the center
        distances, indices = tree.query(center, k=nodes_per_patch)
        patches[i] = indices
        radii[i] = distances[-1] * radius_scale  # Scale up so all nodes are well inside the support
    return patches, radii