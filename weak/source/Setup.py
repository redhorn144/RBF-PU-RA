import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from BaseHelpers import *
from Patch import Patch
from scipy.spatial import cKDTree
from RAHelpers import StableFlatMatrices
from mpi4py import MPI
from PUWeights import NormalizeWeights, C2Weight, C2WeightGradient
from GLLQuad import polar_quad, filter_interior, eval_phi_at_pts, eval_grad_phi_at_pts

###################################
# General setup function to create patches, compute their matrices, and normalize weights
###################################
def Setup(comm, nodes, normals, nodes_per_patch, bdy_indices=None, overlap=3, n_r=12, n_theta=24):
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
        Patch_Phi, Patch_D, Patch_L, Er = StableFlatMatrices(patch_nodes)
        patch = Patch(center=patch_center, radius=patch_radius, node_indices=patch_nodes_indices, normals=patch_normals,
                        nodes=patch_nodes, Phi=Patch_Phi, D=Patch_D, L=Patch_L, w_bar=None, gw_bar=None, lw_bar=None,
                        shape_param=Er)
        patches.append(patch)

    # Normalize PU weights at nodes across all ranks
    NormalizeWeights(comm, patches, patches_for_rank, nodes)

    # Build GLL polar quadrature and normalize PU weights at quad points
    bdy_nodes   = nodes[bdy_indices]   if bdy_indices is not None else None
    bdy_normals = normals[bdy_indices] if bdy_indices is not None else None
    GenQuadrature(patches, centers, radii, bdy_nodes, bdy_normals, n_r, n_theta)

    #print(f"Rank {rank} has {len(patches)} patches.")

    return patches, patches_for_rank


####################################
# SetupPatches: called on rank zero to generate the patches and distribute to other ranks
####################################

def SetupPatches(nodes, nodes_per_patch, overlap=3):
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
        distances, indices = tree.query(center, k=nodes_per_patch)
        patches[i] = indices
        radii[i] = distances[-1] * radius_scale
    return patches, radii


###################################
# GenQuadrature: build GLL polar quadrature on each patch and compute
# normalized PU weights at the quadrature points.
#
# The weight denominator at each quad point q is:
#   W_tot(q) = sum over ALL patches p of C2Weight(q, center_p, radius_p)
# Since all centers/radii are known on every rank, this is computed locally
# with no MPI needed.
###################################

def GenQuadrature(patches, centers, radii, bdy_nodes, bdy_normals, n_r, n_theta):
    """Populate quad_pts, quad_weights, Phi_q, dPhi_q, w_bar_q, gw_bar_q on each patch."""

    for patch in patches:
        pts, weights = polar_quad(patch.center, patch.radius, n_r, n_theta)

        # Filter out points outside the domain (boundary patches)
        if bdy_nodes is not None:
            weights, _ = filter_interior(pts, weights, bdy_nodes, bdy_normals)

        # Only keep points with nonzero weight for efficiency
        mask = weights > 0.0
        pts     = pts[mask]
        weights = weights[mask]

        if pts.shape[0] == 0:
            # Degenerate patch (entirely outside domain) — skip
            patch.quad_pts     = np.empty((0, 2))
            patch.quad_weights = np.empty((0,))
            patch.Phi_q        = np.empty((0, patch.nodes.shape[0]))
            patch.dPhi_q       = np.empty((2, 0, patch.nodes.shape[0]))
            patch.w_bar_q      = np.empty((0,))
            patch.gw_bar_q     = np.empty((0, 2))
            continue

        # Choose a moderate shape parameter so that the Gaussian Phi at nodes
        # is directly invertible (sigma = e * h_nearest ~ 1).
        # e_g = sqrt(n_nodes) / radius gives sigma ~ sqrt(n) * h / R ~ 1
        # for roughly uniform node distributions (h ~ R / sqrt(n)).
        n_loc = patch.nodes.shape[0]
        e_g   = np.sqrt(n_loc) / patch.radius

        # RBF basis and gradients at quad points, using e_g
        Phi_q  = eval_phi_at_pts(pts, patch.nodes, e_g)       # (n_q, n_loc)
        dPhi_q = eval_grad_phi_at_pts(pts, patch.nodes, e_g)  # (d, n_q, n_loc)

        # Same basis at the patch nodes — consistent with Phi_q above
        Phi_nodes = eval_phi_at_pts(patch.nodes, patch.nodes, e_g)  # (n_loc, n_loc)

        # Convert to interpolating shape functions: N_q = Phi_q @ Phi_nodes^{-1}
        # DOFs are nodal values, so N_k(x_j) = delta_{kj}.
        Phi_nodes_inv = np.linalg.inv(Phi_nodes)
        Phi_q  = Phi_q  @ Phi_nodes_inv   # (n_q, n_loc)
        dPhi_q = dPhi_q @ Phi_nodes_inv   # (d, n_q, n_loc)

        # Unnormalized weight for this patch at quad points
        w_p  = C2Weight(pts, patch.center, patch.radius)         # (n_q,)
        gw_p = C2WeightGradient(pts, patch.center, patch.radius) # (n_q, d)

        # Denominator: sum C2Weight over all patches at these quad points
        W_tot  = np.zeros(pts.shape[0])
        gW_tot = np.zeros_like(pts)
        for c, r in zip(centers, radii):
            rho = np.linalg.norm(pts - c, axis=1) / r
            # Only patches whose support covers the point contribute
            inside = rho < 1.0
            if not np.any(inside):
                continue
            W_tot[inside]    += C2Weight(pts[inside], c, r)
            gW_tot[inside]   += C2WeightGradient(pts[inside], c, r)

        # Normalized weight and its gradient (quotient rule)
        w_bar_q  = w_p / W_tot
        gw_bar_q = gw_p / W_tot[:, None] - w_p[:, None] * gW_tot / W_tot[:, None]**2

        patch.quad_pts     = pts
        patch.quad_weights = weights
        patch.Phi_q        = Phi_q
        patch.dPhi_q       = dPhi_q
        patch.w_bar_q      = w_bar_q
        patch.gw_bar_q     = gw_bar_q
