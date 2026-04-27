"""
HaloComm — eval-node ownership and halo-exchange infrastructure.

Assigns each eval node to its "home" patch (nearest patch centre, i.e. Voronoi
on the centre grid) and derives the point-to-point communication pattern that
replaces the global Allreduce in the matvec / rmatvec.

Public API
----------
build_halo_comm(comm, patches, eval_nodes, centers, r_patch) -> HaloComm

HaloComm methods
----------------
mv_exchange(send_bufs)           -> recv_bufs   forward  (matvec direction)
rmv_exchange_1d(owned_1d)        -> recv_bufs   reverse  (rmatvec direction), scalar
rmv_exchange_nd(owned_nd)        -> recv_bufs   reverse, vector (rows = nodes)
"""

import numpy as np
from mpi4py import MPI
from dataclasses import dataclass, field
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Per-patch halo index arrays (precomputed by build_halo_comm)
# ---------------------------------------------------------------------------

@dataclass
class PatchHaloInfo:
    # Eval nodes owned by this rank
    home_mask   : np.ndarray   # (n_eval_p,) bool
    home_lidx   : np.ndarray   # (n_home,) local indices in owned array

    # Eval nodes owned by neighbor ranks
    nbr_ranks   : list         # list of ranks this patch contributes halo to
    nbr_mask    : dict         # {nbr: (n_eval_p,) bool}
    nbr_buf_idx : dict         # {nbr: (n_halo_nbr,) int} indices into mv_send_gidx[nbr]


# ---------------------------------------------------------------------------
# HaloComm
# ---------------------------------------------------------------------------

@dataclass
class HaloComm:
    comm          : object
    rank          : int
    size          : int
    owned_indices : np.ndarray   # (n_owned,) global eval indices this rank owns
    g2l           : np.ndarray   # (M,) g2l[i]=local idx if owned, else -1
    neighbor_ranks: list

    # matvec direction
    mv_send_gidx  : dict         # {nbr: (k,) sorted global indices to send contributions for}
    mv_recv_lidx  : dict         # {nbr: (j,) local indices in owned to add received contributions}

    patch_halo    : list         # list[PatchHaloInfo], one per local patch

    # ------------------------------------------------------------------ exchanges

    def mv_exchange(self, send_bufs, tag=10, n_components=1):
        """
        Forward (matvec-direction) halo exchange.

        send_bufs    : {nbr_rank: 1-D np.ndarray}  contributions to send
        n_components : int  values per eval node (1 for scalars, d for vectors)
        Returns      : {nbr_rank: 1-D np.ndarray}  contributions received
        """
        comm = self.comm
        reqs, recv_bufs = [], {}
        for s in self.neighbor_ranks:
            n = len(self.mv_recv_lidx.get(s, _EMPTY)) * n_components
            if n:
                buf = np.zeros(n)
                recv_bufs[s] = buf
                reqs.append(comm.Irecv(buf, source=s, tag=tag))
        for s, buf in send_bufs.items():
            if len(buf):
                reqs.append(comm.Isend(np.ascontiguousarray(buf, dtype=np.float64),
                                       dest=s, tag=tag))
        MPI.Request.Waitall(reqs)
        return recv_bufs

    def rmv_exchange_1d(self, owned_1d, tag=20):
        """
        Reverse (rmatvec-direction) exchange for a scalar field.

        owned_1d : (n_owned,)  values at owned eval nodes
        Returns  : {nbr: (k,) values for mv_send_gidx[nbr] nodes}
        """
        comm = self.comm
        reqs, recv_bufs = [], {}
        for s in self.neighbor_ranks:
            n = len(self.mv_send_gidx.get(s, _EMPTY))
            if n:
                buf = np.zeros(n)
                recv_bufs[s] = buf
                reqs.append(comm.Irecv(buf, source=s, tag=tag))
        for s in self.neighbor_ranks:
            lidx = self.mv_recv_lidx.get(s, _EMPTY_INT)
            if len(lidx):
                reqs.append(comm.Isend(
                    np.ascontiguousarray(owned_1d[lidx], dtype=np.float64),
                    dest=s, tag=tag))
        MPI.Request.Waitall(reqs)
        return recv_bufs

    def rmv_exchange_nd(self, owned_nd, tag=30):
        """
        Reverse exchange for a vector field.

        owned_nd : (n_owned, d)  values at owned eval nodes
        Returns  : {nbr: (k, d) values for mv_send_gidx[nbr] nodes}
        """
        d = owned_nd.shape[1]
        comm = self.comm
        reqs, recv_bufs = [], {}
        for s in self.neighbor_ranks:
            n = len(self.mv_send_gidx.get(s, _EMPTY))
            if n:
                buf = np.zeros((n, d))
                recv_bufs[s] = buf
                reqs.append(comm.Irecv(buf, source=s, tag=tag))
        for s in self.neighbor_ranks:
            lidx = self.mv_recv_lidx.get(s, _EMPTY_INT)
            if len(lidx):
                reqs.append(comm.Isend(
                    np.ascontiguousarray(owned_nd[lidx], dtype=np.float64),
                    dest=s, tag=tag))
        MPI.Request.Waitall(reqs)
        return recv_bufs


_EMPTY     = np.array([], dtype=np.float64)
_EMPTY_INT = np.array([], dtype=np.int64)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_halo_comm(comm, patches, eval_nodes, centers, r_patch, patch_rank):
    """
    Precompute eval-node ownership and halo communication graph.

    Parameters
    ----------
    comm       : MPI communicator
    patches    : list[Patch]      local patches on this rank (already built)
    eval_nodes : (M, d)           all eval nodes, same on all ranks
    centers    : (n_patches, d)   all patch centres, same on all ranks
    r_patch    : float            patch radius
    patch_rank : (n_patches,) int global patch-id → MPI rank mapping
                                  (computed once in LSSetup and reused here)

    Returns
    -------
    HaloComm
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    M    = len(eval_nodes)

    # ---- 1. Assign each eval node to its home patch (Voronoi on centres) ----
    center_tree       = cKDTree(centers)
    _, node_home_pid  = center_tree.query(eval_nodes)           # (M,) global pid
    node_home_rank    = patch_rank[node_home_pid].astype(np.int32)  # (M,) MPI rank

    # ---- 2. Owned nodes on this rank ----
    owned_indices = np.where(node_home_rank == rank)[0]
    n_owned       = len(owned_indices)
    g2l           = np.full(M, -1, dtype=np.int64)
    g2l[owned_indices] = np.arange(n_owned, dtype=np.int64)

    # ---- 3. Build send_sets from local patches only — O(P_r * n_e) ----
    send_sets = {}
    for p in patches:
        for gidx in p.eval_node_indices:
            nbr = int(node_home_rank[gidx])
            if nbr != rank:
                send_sets.setdefault(nbr, set()).add(int(gidx))

    # ---- 4. Derive recv_sets via Alltoall(v) — eliminates O(P*n_e) global work ----
    # 4a. Tell every rank how many indices we will send it
    send_counts = np.zeros(size, dtype=np.int32)
    for r, s in send_sets.items():
        send_counts[r] = len(s)

    recv_counts = np.empty(size, dtype=np.int32)
    comm.Alltoall(send_counts, recv_counts)

    # 4b. Pack sorted send index arrays and build mv_send_gidx simultaneously
    send_parts = []
    mv_send_gidx = {}
    for r in range(size):
        if r in send_sets:
            arr = np.sort(np.fromiter(send_sets[r], dtype=np.int64))
            mv_send_gidx[r] = arr
            send_parts.append(arr)
        else:
            send_parts.append(np.empty(0, dtype=np.int64))

    send_buf    = np.concatenate(send_parts)
    send_displs = np.zeros(size, dtype=np.int32)
    send_displs[1:] = np.cumsum(send_counts[:-1])

    recv_buf    = np.empty(int(recv_counts.sum()), dtype=np.int64)
    recv_displs = np.zeros(size, dtype=np.int32)
    recv_displs[1:] = np.cumsum(recv_counts[:-1])

    comm.Alltoallv(
        [send_buf, send_counts, send_displs, MPI.INT64_T],
        [recv_buf, recv_counts, recv_displs, MPI.INT64_T],
    )

    # 4c. Unpack into mv_recv_gidx — buffers arrive already sorted
    mv_recv_gidx = {}
    for r in range(size):
        if recv_counts[r] > 0:
            start = int(recv_displs[r])
            mv_recv_gidx[r] = recv_buf[start : start + recv_counts[r]]

    neighbor_ranks = sorted(set(mv_send_gidx) | set(mv_recv_gidx))
    mv_send_gidx   = {s: mv_send_gidx.get(s, np.empty(0, dtype=np.int64)) for s in neighbor_ranks}
    mv_recv_gidx   = {s: mv_recv_gidx.get(s, np.empty(0, dtype=np.int64)) for s in neighbor_ranks}
    mv_recv_lidx   = {s: g2l[mv_recv_gidx[s]] for s in neighbor_ranks}

    # ---- 5. Per-patch halo info ----
    patch_halo = []
    for p in patches:
        eids = p.eval_node_indices.astype(np.int64)
        h_ranks = node_home_rank[eids]

        home_mask = (h_ranks == rank)
        home_lidx = g2l[eids[home_mask]]

        nbr_ranks_p, nbr_mask, nbr_buf_idx = [], {}, {}
        for s in neighbor_ranks:
            if s not in send_sets:
                continue
            mask_s = (h_ranks == s)
            if not mask_s.any():
                continue
            buf_idx = np.searchsorted(mv_send_gidx[s], eids[mask_s]).astype(np.int64)
            nbr_ranks_p.append(s)
            nbr_mask[s]    = mask_s
            nbr_buf_idx[s] = buf_idx

        patch_halo.append(PatchHaloInfo(
            home_mask=home_mask,
            home_lidx=home_lidx,
            nbr_ranks=nbr_ranks_p,
            nbr_mask=nbr_mask,
            nbr_buf_idx=nbr_buf_idx,
        ))

    return HaloComm(
        comm=comm, rank=rank, size=size,
        owned_indices=owned_indices,
        g2l=g2l,
        neighbor_ranks=neighbor_ranks,
        mv_send_gidx=mv_send_gidx,
        mv_recv_lidx=mv_recv_lidx,
        patch_halo=patch_halo,
    )
