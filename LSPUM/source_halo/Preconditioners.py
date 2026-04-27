import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.spatial import cKDTree
from mpi4py import MPI

#---------------------------------------------------------------------------
# Preconditioners for the LS-RBF-PUM system.
#
# Each factory returns an (apply_Pinv, apply_PinvT) pair of closures that
# act on the rank-local DOF vector v_local.  The driver (Solvers.py) wraps
# them around the matrix-free matvec / rmatvec as a right preconditioner:
#
#     matvec_prec(y) = matvec(apply_PinvT(y))      # A · P^{-1}
#     rmatvec_prec(u) = apply_Pinv(rmatvec(u))     # P^{-T} · A^T
#
# After the solve the driver recovers x = P^{-1} y = apply_PinvT(y).
# All factories are patch-local; no communication.
#---------------------------------------------------------------------------

def GenBlockJacobi(Rs, n_interp, ridge=1e-14):
    """
    Block-Jacobi preconditioner in factored form.

    For each patch p, M_p = R_p^T R_p is the (n_interp, n_interp) diagonal
    block of A^T A.  Store its Cholesky L_p (lower) so M_p = L_p L_p^T and
    take P = block_diag(L_p^T), giving diag-block-identity for the
    preconditioned normal matrix (P^{-T} A^T A P^{-1})_{pp} = I.  The
    remaining patch-to-patch coupling from overlap is untouched.

    Parameters
    ----------
    Rs       : list[(n_eval_p, n_interp) array]  row matrices (PoissonRowMatrices)
    n_interp : int                               DOFs per patch
    ridge    : float                             tiny diagonal stabiliser

    Returns
    -------
    apply_Pinv  : callable  v_local -> block_diag(L_p^{-1})  v_local
    apply_PinvT : callable  v_local -> block_diag(L_p^{-T})  v_local
    """
    Ls = []
    Ieye = ridge * np.eye(n_interp)
    for R in Rs:
        Mp = R.T @ R + Ieye
        Ls.append(cholesky(Mp, lower=True))

    def apply_PinvT(v_local):
        out = np.empty_like(v_local)
        for pi, L in enumerate(Ls):
            out[pi*n_interp:(pi+1)*n_interp] = solve_triangular(
                L, v_local[pi*n_interp:(pi+1)*n_interp], lower=True, trans=1)
        return out

    def apply_Pinv(v_local):
        out = np.empty_like(v_local)
        for pi, L in enumerate(Ls):
            out[pi*n_interp:(pi+1)*n_interp] = solve_triangular(
                L, v_local[pi*n_interp:(pi+1)*n_interp], lower=True)
        return out

    return apply_Pinv, apply_PinvT


def GenDiagEquil(Rs, n_interp):
    """
    Column-equilibration preconditioner  P = diag(||A[:,j]||).

    Each DOF column lives entirely in its owning patch's rows, so the
    column norm is purely patch-local: ||A[:,j]|| = ||R_p[:,k]||.  P is
    diagonal so P^{-1} = P^{-T} and the two returned closures are the
    same function.

    Parameters
    ----------
    Rs       : list[(n_eval_p, n_interp) array]
    n_interp : int

    Returns
    -------
    apply_Pinv, apply_PinvT : callable  (both apply v_local -> (1/d) v_local)
    """
    d = np.empty(len(Rs) * n_interp)
    for pi, R in enumerate(Rs):
        d[pi*n_interp:(pi+1)*n_interp] = np.linalg.norm(R, axis=0)
    inv_d = np.where(d > 0, 1.0 / d, 1.0)

    def apply(v_local):
        return inv_d * v_local

    return apply, apply


def GenRAS(comm, patches, Rs, n_interp, ridge=1e-14):
    """
    Restricted Additive Schwarz preconditioner for the PUM normal equations.

    For each local patch p, builds an extended local system M̃_p = A_ext^T A_ext
    where A_ext covers all patches in the 1-ring neighborhood N(p).  Factors M̃_p
    via Cholesky.  Returns an exact adjoint pair (apply_Pinv, apply_PinvT) using
    DOF-space gather / scatter-add MPI communication.

    apply_PinvT(y): for each p, gather y_{N(p)}, solve M̃_p^{-1} y_{N(p)},
                    restrict to patch-p block (first n_interp entries).
    apply_Pinv(v):  for each p, solve M̃_p^{-1} [v_p; 0; ...],
                    scatter-add full result to all patches in N(p).

    These satisfy <apply_PinvT(y), z> = <y, apply_Pinv(z)> exactly.

    Parameters
    ----------
    comm      : MPI communicator
    patches   : list[Patch]
    Rs        : list[(n_eval_p, n_interp) ndarray]   PoissonRowMatrices output
    n_interp  : int
    ridge     : float   diagonal stabiliser added to each M̃_p
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    # ------------------------------------------------------------------
    # 1. Allgather patch geometry, global_pids, eval_node_indices, and Rs
    # ------------------------------------------------------------------
    local_centers = np.array([p.center      for p in patches], dtype=float)
    local_radii   = np.array([p.radius      for p in patches], dtype=float)
    local_gpids   = np.array([p.global_pid  for p in patches], dtype=np.int64)

    all_centers_list = comm.allgather(local_centers)
    all_radii_list   = comm.allgather(local_radii)
    all_gpids_list   = comm.allgather(local_gpids)

    nonempty = [c for c in all_centers_list if len(c) > 0]
    all_centers = np.vstack(nonempty) if nonempty else np.zeros((0, 2), dtype=float)
    all_radii   = np.concatenate(all_radii_list)
    all_gpids   = np.concatenate(all_gpids_list)
    all_ranks   = np.concatenate([
        np.full(len(gp), r, dtype=np.int32)
        for r, gp in enumerate(all_gpids_list)
    ])

    gpid_to_gidx = {int(gp): i for i, gp in enumerate(all_gpids)}  # global_pid -> row in all_centers

    # Allgather eval_node_indices (as lists, variable length per patch)
    local_eids = [p.eval_node_indices.tolist() for p in patches]
    all_eids_list = comm.allgather(local_eids)   # all_eids_list[r][pi_local] = list of global eval ids

    # Allgather Rs matrices (flattened)
    local_rs_flat   = np.concatenate([R.flatten() for R in Rs]) if Rs else np.empty(0)
    local_rs_nrows  = np.array([R.shape[0] for R in Rs], dtype=np.int64)
    all_rs_flat_list  = comm.allgather(local_rs_flat)
    all_rs_nrows_list = comm.allgather(local_rs_nrows)

    # Build lookup: global_pid -> (Rs matrix, eval_node_indices ndarray)
    gpid_to_Rs   = {}
    gpid_to_eids = {}
    for r, (rs_flat, rs_nrows, gpids, eids_for_rank) in enumerate(
            zip(all_rs_flat_list, all_rs_nrows_list, all_gpids_list, all_eids_list)):
        offset = 0
        for pi_r, (gpid, nrow, eids) in enumerate(zip(gpids, rs_nrows, eids_for_rank)):
            sz = int(nrow) * n_interp
            gpid_to_Rs[int(gpid)]   = rs_flat[offset:offset+sz].reshape(int(nrow), n_interp)
            gpid_to_eids[int(gpid)] = np.array(eids, dtype=np.int64)
            offset += sz

    # ------------------------------------------------------------------
    # 2. Build 1-ring neighborhood for each local patch
    #    N(p) = {q : dist(center_p, center_q) < radius_p + radius_q}
    #    Patch p is placed first in the ordered list.
    # ------------------------------------------------------------------
    P_total = len(all_gpids)
    neighborhoods = []   # neighborhoods[pi] = ordered list of global_pids; p first
    if P_total > 0:
        tree = cKDTree(all_centers)
        r_max = float(all_radii.max()) if len(all_radii) else 0.0
        for p in patches:
            candidates = tree.query_ball_point(p.center, r=p.radius + r_max)
            nbrs_gpids = []
            for ci in candidates:
                dist = np.linalg.norm(p.center - all_centers[ci])
                if dist < p.radius + all_radii[ci]:
                    nbrs_gpids.append(int(all_gpids[ci]))
            p_gp = p.global_pid
            ordered = [p_gp] + [gp for gp in nbrs_gpids if gp != p_gp]
            neighborhoods.append(ordered)
    else:
        neighborhoods = [[] for _ in patches]

    # ------------------------------------------------------------------
    # 3. Build extended Cholesky factors
    # ------------------------------------------------------------------
    ext_factors = []   # list of (L_ext, ordered_gpids) per local patch
    for pi, p in enumerate(patches):
        ordered_gpids = neighborhoods[pi]
        n_nbrs = len(ordered_gpids)
        n_ext  = n_nbrs * n_interp

        # Union eval set
        all_eids_sets = [set(gpid_to_eids[gp].tolist()) for gp in ordered_gpids]
        U_p = sorted(set().union(*all_eids_sets))
        uid_to_row = {e: i for i, e in enumerate(U_p)}

        # Assemble A_ext
        A_ext = np.zeros((len(U_p), n_ext))
        for qi, gp in enumerate(ordered_gpids):
            Rs_q   = gpid_to_Rs[gp]
            eids_q = gpid_to_eids[gp]
            col_s  = qi * n_interp
            for local_e, global_e in enumerate(eids_q):
                A_ext[uid_to_row[int(global_e)], col_s:col_s+n_interp] = Rs_q[local_e]

        M_ext = A_ext.T @ A_ext + ridge * np.eye(n_ext)
        L_ext = cholesky(M_ext, lower=True)
        ext_factors.append((L_ext, ordered_gpids))

    # ------------------------------------------------------------------
    # 4. Build DOF communication maps via Alltoallv
    #    gather_recv[r] = ordered list of gpids on rank r that I need DOFs from
    #    gather_send[r] = ordered list of MY gpids whose DOFs rank r needs
    # ------------------------------------------------------------------
    gather_recv = {}   # r -> [gpid, ...]
    for nbrs in neighborhoods:
        for gp in nbrs:
            r = int(all_ranks[gpid_to_gidx[gp]])
            if r != rank:
                if r not in gather_recv:
                    gather_recv[r] = []
                if gp not in gather_recv[r]:
                    gather_recv[r].append(gp)

    # Exchange need-lists: tell each rank which gpids we need from them
    send_counts = np.zeros(size, dtype=np.int32)
    for r, gpids in gather_recv.items():
        send_counts[r] = len(gpids)
    recv_counts = np.empty(size, dtype=np.int32)
    comm.Alltoall(send_counts, recv_counts)

    send_buf_nl   = np.concatenate([
        np.array(gather_recv.get(r, []), dtype=np.int64) for r in range(size)
    ]) if any(send_counts) else np.empty(0, dtype=np.int64)
    send_displs_nl = np.zeros(size, dtype=np.int32)
    send_displs_nl[1:] = np.cumsum(send_counts[:-1])
    recv_buf_nl   = np.empty(int(recv_counts.sum()), dtype=np.int64)
    recv_displs_nl = np.zeros(size, dtype=np.int32)
    recv_displs_nl[1:] = np.cumsum(recv_counts[:-1])
    comm.Alltoallv(
        [send_buf_nl, send_counts, send_displs_nl, MPI.INT64_T],
        [recv_buf_nl, recv_counts, recv_displs_nl, MPI.INT64_T],
    )

    gather_send = {}
    for r in range(size):
        if recv_counts[r] > 0:
            s = int(recv_displs_nl[r])
            gather_send[r] = recv_buf_nl[s:s + recv_counts[r]].tolist()

    # Offset maps: gpid -> byte offset within the per-rank buffer
    gather_recv_offsets = {
        r: {gp: i * n_interp for i, gp in enumerate(gpids)}
        for r, gpids in gather_recv.items()
    }
    gather_send_offsets = {
        r: {gp: i * n_interp for i, gp in enumerate(gpids)}
        for r, gpids in gather_send.items()
    }

    # Map local gpid -> slice in v_local
    gpid_to_slice = {p.global_pid: slice(pi*n_interp, (pi+1)*n_interp)
                     for pi, p in enumerate(patches)}

    # ------------------------------------------------------------------
    # 5. apply_sas: symmetric additive Schwarz
    #    gather DOFs -> solve M̃_p^{-1} (full extended) -> scatter-add ALL blocks
    #
    #    apply_sas(v)[q] = Σ_{p: q ∈ N(p)} [M̃_p^{-1} v_{N(p)}][q_in_p]
    #
    #    This is symmetric (<apply_sas(v),w> = <v,apply_sas(w)>) because
    #    each M̃_p is symmetric.  Used as left preconditioner in PCG.
    # ------------------------------------------------------------------
    def apply_sas(v_local):
        # --- Phase 1: gather (receive neighbor DOF values) ---
        reqs = []
        send_bufs_g = {}
        for r, gpids in gather_send.items():
            buf = np.empty(len(gpids) * n_interp)
            for i, gp in enumerate(gpids):
                buf[i*n_interp:(i+1)*n_interp] = v_local[gpid_to_slice[gp]]
            send_bufs_g[r] = buf
            reqs.append(comm.Isend(buf, dest=r, tag=10))

        recv_bufs_g = {}
        for r, gpids in gather_recv.items():
            buf = np.empty(len(gpids) * n_interp)
            recv_bufs_g[r] = buf
            reqs.append(comm.Irecv(buf, source=r, tag=10))

        MPI.Request.Waitall(reqs)

        # --- Phase 2: solve + local scatter-add; pack remote scatter ---
        accum = np.zeros_like(v_local)
        # send_bufs_sc[r] accumulates scatter contributions TO rank r
        send_bufs_sc = {r: np.zeros(len(gpids) * n_interp)
                        for r, gpids in gather_recv.items()}

        for pi, (p, (L_ext, ordered_gpids)) in enumerate(zip(patches, ext_factors)):
            n_nbrs = len(ordered_gpids)
            y_ext = np.empty(n_nbrs * n_interp)
            for qi, gp in enumerate(ordered_gpids):
                if gp in gpid_to_slice:
                    y_ext[qi*n_interp:(qi+1)*n_interp] = v_local[gpid_to_slice[gp]]
                else:
                    r   = int(all_ranks[gpid_to_gidx[gp]])
                    off = gather_recv_offsets[r][gp]
                    y_ext[qi*n_interp:(qi+1)*n_interp] = recv_bufs_g[r][off:off+n_interp]

            # Full extended solve M̃_p^{-1} v_{N(p)}
            ww = solve_triangular(L_ext, y_ext, lower=True)
            z  = solve_triangular(L_ext, ww,    lower=True, trans=1)

            # Scatter-add ALL extended-block contributions to their patches
            for qi, gp in enumerate(ordered_gpids):
                z_q = z[qi*n_interp:(qi+1)*n_interp]
                if gp in gpid_to_slice:
                    accum[gpid_to_slice[gp]] += z_q
                else:
                    r   = int(all_ranks[gpid_to_gidx[gp]])
                    off = gather_recv_offsets[r][gp]
                    send_bufs_sc[r][off:off+n_interp] += z_q

        # --- Phase 3: exchange scatter contributions (transpose of gather) ---
        reqs = []
        recv_bufs_sc = {}
        for r, gpids in gather_send.items():
            buf = np.empty(len(gpids) * n_interp)
            recv_bufs_sc[r] = buf
            reqs.append(comm.Irecv(buf, source=r, tag=11))
        for r, buf in send_bufs_sc.items():
            reqs.append(comm.Isend(buf, dest=r, tag=11))
        MPI.Request.Waitall(reqs)

        for r, buf in recv_bufs_sc.items():
            for i, gp in enumerate(gather_send[r]):
                accum[gpid_to_slice[gp]] += buf[i*n_interp:(i+1)*n_interp]

        return accum

    return apply_sas
