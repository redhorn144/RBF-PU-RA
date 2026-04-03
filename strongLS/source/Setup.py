import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from BaseHelpers import *
from Patch import Patch
from scipy.spatial import cKDTree
from RAHelpers import StableFlatMatrices, StableFlatMatricesOS
from mpi4py import MPI
from PUWeights import NormalizeWeights, NormalizeWeightsEval

###################################
# General setup function to create patches, compute their matrices, and normalize weights
###################################
def Setup(comm, nodes, normals, nodes_per_patch, overlap = 3, eval_epsilon = 0, K = 64, n = 16, m = 48):
    rank = comm.Get_rank()
    
    if rank == 0:
        centers, radii, patch_node_inds = SetupPatches(nodes, nodes_per_patch, overlap=overlap)
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
        Patch_Phi, Patch_D, Patch_L = StableFlatMatrices(patch_nodes, K = K, n = n, m = m, eval_epsilon=eval_epsilon)
        patch = Patch(center=patch_center, radius=patch_radius, node_indices=patch_nodes_indices, normals=patch_normals,
                        nodes=patch_nodes, Phi=Patch_Phi, D=Patch_D, L=Patch_L, w_bar=None, gw_bar=None, lw_bar=None,
                        eval_indices=None, eval_nodes=None, Phi_eval=None, D_eval=None, L_eval=None,
                        w_bar_eval=None, gw_bar_eval=None, lw_bar_eval=None)
        patches.append(patch)
    
    # Normalize PU weights across all ranks
    NormalizeWeights(comm, patches, patches_for_rank, nodes)

    #print(f"Rank {rank} has {len(patches)} patches.")
    
    return patches, patches_for_rank



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

###################################
# Oversampled setup: solution nodes drive patch placement;
# eval_pts (collocation points) can be a denser set.
###################################
def SetupOversampled(comm, nodes, eval_pts, normals, nodes_per_patch, overlap=3,
                     eval_epsilon=0, K=64, n=16, m=48):
    """
    Like Setup, but each patch also carries rectangular (m×n) matrices evaluated
    at eval_pts, enabling an oversampled (least-squares) global system.

    Parameters
    ----------
    nodes           : (N, d) solution nodes — RBF centers and global unknowns
    eval_pts        : (M, d) collocation points where PDE is enforced (M > N typical)
    normals         : (N, d) normals at solution nodes
    nodes_per_patch : target number of solution nodes per patch
    overlap         : patch density factor
    """
    rank = comm.Get_rank()

    if rank == 0:
        centers, radii, patch_node_inds = SetupPatches(nodes, nodes_per_patch, overlap=overlap)
    else:
        centers = None
        radii = None
        patch_node_inds = None

    centers         = comm.bcast(centers, root=0)
    radii           = comm.bcast(radii, root=0)
    patch_node_inds = comm.bcast(patch_node_inds, root=0)

    num_patches = len(centers)
    patches_for_rank = [i for i in range(num_patches) if i % comm.Get_size() == rank]

    eval_tree = cKDTree(eval_pts)

    patches = []
    for i in patches_for_rank:
        patch_nodes   = nodes[patch_node_inds[i]]
        patch_normals = normals[patch_node_inds[i]]
        patch_center  = centers[i]
        patch_radius  = radii[i]
        patch_node_indices = patch_node_inds[i]

        # Find eval_pts within patch support
        eval_idx = eval_tree.query_ball_point(patch_center, patch_radius)
        eval_idx = np.array(sorted(eval_idx), dtype=int)
        patch_eval_nodes = eval_pts[eval_idx]

        # One combined RA pass: square (n×n) + rectangular (m×n) matrices
        Patch_Phi, Patch_D, Patch_L, Patch_Phi_eval, Patch_D_eval, Patch_L_eval = \
            StableFlatMatricesOS(patch_eval_nodes, patch_nodes, K=K, n=n, m=m,
                                 eval_epsilon=eval_epsilon)

        patch = Patch(
            center=patch_center, radius=patch_radius,
            node_indices=patch_node_indices, normals=patch_normals,
            nodes=patch_nodes, Phi=Patch_Phi, D=Patch_D, L=Patch_L,
            w_bar=None, gw_bar=None, lw_bar=None,
            eval_indices=eval_idx, eval_nodes=patch_eval_nodes,
            Phi_eval=Patch_Phi_eval, D_eval=Patch_D_eval, L_eval=Patch_L_eval,
            w_bar_eval=None, gw_bar_eval=None, lw_bar_eval=None,
        )
        patches.append(patch)

    # Normalize PU weights at both solution nodes and eval_pts
    NormalizeWeights(comm, patches, patches_for_rank, nodes)
    NormalizeWeightsEval(comm, patches, patches_for_rank, eval_pts)

    return patches, patches_for_rank


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