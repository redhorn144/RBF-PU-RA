import numpy as np
import scipy.sparse as sp


###################################
# assemble_poisson
#
# Builds the global Galerkin stiffness matrix K and load vector f for
# the Poisson problem:  -Delta u = rhs_fn(x)
#
# Bilinear form:  a(u, v) = integral_Omega  grad u . grad v  dOmega
#
# On each patch p the trial/test functions are  Psi_{p,j}(x) = w_bar_p(x) phi_j^p(x)
# so the local stiffness contribution is:
#
#   K_p[i,j] = sum_q  W_q  grad(w_bar_p phi_i)(q) . grad(w_bar_p phi_j)(q)
#
# with  grad(w_bar_p phi_j) = phi_j grad(w_bar_p) + w_bar_p grad(phi_j)
###################################

def assemble_poisson(comm, patches, N, rhs_fn):
    """
    Parameters
    ----------
    comm      : MPI communicator
    patches   : list of Patch objects on this rank (already have quad fields)
    N         : total number of global nodes
    rhs_fn    : callable  x -> f(x),  x shape (n, 2)

    Returns
    -------
    K_local : (N, N) scipy sparse CSR — this rank's partial stiffness contributions
    f       : (N,)   numpy array      — global load vector (AllReduced)
    """
    K_local = sp.lil_matrix((N, N))
    f_local = np.zeros(N)

    for patch in patches:
        if patch.quad_pts.shape[0] == 0:
            continue

        idx = patch.node_indices          # (n_loc,)
        W   = patch.quad_weights          # (n_q,)
        wb  = patch.w_bar_q               # (n_q,)
        gw  = patch.gw_bar_q              # (n_q, d)
        P   = patch.Phi_q                 # (n_q, n_loc)
        dP  = patch.dPhi_q                # (d, n_q, n_loc)

        # grad(w_bar * phi_j) at each quad point: (d, n_q, n_loc)
        #   = phi_j(q) * grad_w_bar(q) + w_bar(q) * grad_phi_j(q)
        grad_wphi = P[np.newaxis, :, :] * gw.T[:, :, np.newaxis] \
                  + wb[np.newaxis, :, np.newaxis] * dP   # (d, n_q, n_loc)

        # Local stiffness: K_p[i,j] = sum_q W_q (grad_wphi[:,q,i] . grad_wphi[:,q,j])
        K_loc = np.einsum('q,kqi,kqj->ij', W, grad_wphi, grad_wphi)  # (n_loc, n_loc)

        # Local load: f_p[i] = sum_q W_q  w_bar(q) phi_i(q) rhs(q)
        f_q = rhs_fn(patch.quad_pts)                              # (n_q,)
        f_loc = np.einsum('q,q,qi->i', W, wb * f_q, P)           # (n_loc,)

        # Scatter into global
        K_local[np.ix_(idx, idx)] += K_loc
        f_local[idx] += f_loc

    # AllReduce f only; K stays distributed (one sparse CSR per rank)
    f_global = np.zeros_like(f_local)
    comm.Allreduce(f_local, f_global)

    return K_local.tocsr(), f_global


###################################
# assemble_advection
#
# Builds the global mass matrix M and advection matrix A for:
#   M du/dt + A u = 0
#
# M[i,j] = sum_p  integral  (w_bar_p phi_i)(w_bar_p phi_j) dOmega_p
#
# A[i,j] = sum_p  integral  (c . grad(w_bar_p phi_j)) (w_bar_p phi_i) dOmega_p
#
# cx_fn, cy_fn : callables  x -> c_k(x)  with x shape (n, 2)
###################################

def assemble_advection(comm, patches, N, cx_fn, cy_fn):
    """
    Returns
    -------
    M : (N, N) scipy sparse CSR  (mass matrix)
    A : (N, N) scipy sparse CSR  (advection matrix, non-symmetric)
    """
    M_local = sp.lil_matrix((N, N))
    A_local = sp.lil_matrix((N, N))

    for patch in patches:
        if patch.quad_pts.shape[0] == 0:
            continue

        idx = patch.node_indices
        W   = patch.quad_weights          # (n_q,)
        wb  = patch.w_bar_q               # (n_q,)
        gw  = patch.gw_bar_q              # (n_q, d)
        P   = patch.Phi_q                 # (n_q, n_loc)
        dP  = patch.dPhi_q                # (d, n_q, n_loc)

        # Test function values: w_bar * phi_i  at quad pts  (n_q, n_loc)
        test = wb[:, np.newaxis] * P

        # grad(w_bar * phi_j): (d, n_q, n_loc)
        grad_wphi = P[np.newaxis, :, :] * gw.T[:, :, np.newaxis] \
                  + wb[np.newaxis, :, np.newaxis] * dP

        # Velocity at quad points
        cx_q = cx_fn(patch.quad_pts)  # (n_q,)
        cy_q = cy_fn(patch.quad_pts)  # (n_q,)

        # c . grad(w_bar phi_j): (n_q, n_loc)
        c_dot_grad = cx_q[:, np.newaxis] * grad_wphi[0].T.T \
                   + cy_q[:, np.newaxis] * grad_wphi[1].T.T   # (n_q, n_loc)

        # Mass: M_p[i,j] = sum_q W_q test_i(q) test_j(q)
        M_loc = np.einsum('q,qi,qj->ij', W, test, test)

        # Advection: A_p[i,j] = sum_q W_q test_i(q) (c.grad(w_bar phi_j))(q)
        A_loc = np.einsum('q,qi,qj->ij', W, test, c_dot_grad)

        M_local[np.ix_(idx, idx)] += M_loc
        A_local[np.ix_(idx, idx)] += A_loc

    M_arr = np.zeros((N, N))
    A_arr = np.zeros((N, N))
    comm.Allreduce(M_local.toarray(), M_arr)
    comm.Allreduce(A_local.toarray(), A_arr)

    return sp.csr_matrix(M_arr), sp.csr_matrix(A_arr)


###################################
# apply_dirichlet
#
# Enforce Dirichlet boundary conditions by zeroing the rows and columns
# of K corresponding to boundary nodes, placing 1 on the diagonal,
# and adjusting the RHS accordingly.
###################################

def apply_dirichlet(K, f, bdy_indices, bdy_values):
    """
    Modifies K and f in-place (dense arrays) for Dirichlet BCs.

    Parameters
    ----------
    K           : (N, N) numpy array (dense) — modified in place
    f           : (N,)   numpy array         — modified in place
    bdy_indices : (n_b,) int array of boundary DOF indices
    bdy_values  : (n_b,) prescribed values (often zero)
    """
    # Subtract contribution of known BCs from interior RHS
    f -= K[:, bdy_indices] @ bdy_values

    # Zero boundary rows and columns, then set diagonal = 1
    K[bdy_indices, :] = 0.0
    K[:, bdy_indices] = 0.0
    K[bdy_indices, bdy_indices] = 1.0

    # Set RHS at boundary nodes
    f[bdy_indices] = bdy_values


###################################
# apply_dirichlet_distributed
#
# Sparse, MPI-collective version of apply_dirichlet for use with the
# distributed matvec solver.  K_local is each rank's partial CSR
# contribution; f is already globally AllReduced.
###################################

def apply_dirichlet_distributed(comm, K_local, f, bdy_indices, bdy_values):
    """
    Parameters
    ----------
    comm        : MPI communicator
    K_local     : (N, N) sparse CSR — this rank's partial stiffness
    f           : (N,)   numpy array — global RHS (modified in place)
    bdy_indices : (n_b,) int array
    bdy_values  : (n_b,) prescribed values

    Returns
    -------
    K_local : modified sparse CSR (boundary rows/cols zeroed, diagonal set on rank 0)
    f       : modified RHS
    """
    K_lil = K_local.tolil()

    # Subtract K[:, bdy] @ bdy_values from f, then AllReduce the correction
    correction_local = np.asarray(K_lil[:, bdy_indices] @ bdy_values).ravel()
    correction = np.zeros_like(correction_local)
    comm.Allreduce(correction_local, correction)
    f -= correction
    f[bdy_indices] = bdy_values

    # Zero boundary rows and columns on every rank
    K_lil[bdy_indices, :] = 0.0
    K_lil[:, bdy_indices] = 0.0

    # Set diagonal = 1 on rank 0 only so the AllReduce matvec gives y[b] = x[b]
    if comm.Get_rank() == 0:
        for b in bdy_indices:
            K_lil[b, b] = 1.0

    return K_lil.tocsr(), f
