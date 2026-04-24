"""
HaloProfile.py — timing and parallel-efficiency analysis.

Compares three configurations:
  allreduce   — original global Allreduce matvec (source/)
  rr          — halo exchange + round_robin patch assignment (source_halo/)
  block       — halo exchange + block_grid_2d assignment   (source_halo/, default)

Run with varying rank counts to observe strong-scaling behaviour:
    mpirun -n 1 python HaloProfile.py
    mpirun -n 2 python HaloProfile.py
    mpirun -n 4 python HaloProfile.py
    mpirun -n 8 python HaloProfile.py

All times via MPI.Wtime(); per-rank min/mean/max reported to expose imbalance.
"""

from mpi4py import MPI
import numpy as np

from nodes.SquareDomain import MinEnergySquareOne
from source_halo.PatchTiling import LarssonBox2D
from source_halo.LSSetup import Setup
from source_halo.Operators import PoissonRowMatrices, GenMatFreeOps
from source_halo.Solvers import GenIterativeSolver

from source.LSSetup  import Setup as Setup_orig
from source.Operators import PoissonRowMatrices as PRM_orig, GenMatFreeOps as GMO_orig
from source.Solvers  import GenIterativeSolver as GIS_orig

N_WARMUP = 5
N_REPS   = 20


def time_fn(fn, n_reps):
    times = np.empty(n_reps)
    for i in range(n_reps):
        MPI.COMM_WORLD.Barrier()
        t0 = MPI.Wtime()
        fn()
        MPI.COMM_WORLD.Barrier()
        times[i] = MPI.Wtime() - t0
    return times


def stats(comm, local_vals):
    arr = np.asarray(local_vals, dtype=float)
    mn  = comm.allreduce(float(arr.min()), op=MPI.MIN)
    mx  = comm.allreduce(float(arr.max()), op=MPI.MAX)
    s   = comm.allreduce(float(arr.sum()), op=MPI.SUM)
    n   = comm.allreduce(len(arr),          op=MPI.SUM)
    return mn, s / n, mx


def section(comm, title):
    if comm.Get_rank() == 0:
        print(f"\n{'─'*72}")
        print(f"  {title}")
        print(f"{'─'*72}")


def hdr(comm):
    if comm.Get_rank() == 0:
        print(f"  {'Metric':<38}  {'allreduce':>9}  {'rr halo':>9}  {'block halo':>10}  unit")
        print(f"  {'-'*70}")


def row3(comm, label, vals, unit="ms", scale=1e3):
    if comm.Get_rank() == 0:
        vs = [f"{v*scale:9.3f}" for v in vals]
        print(f"  {label:<38}  {'  '.join(vs)}  {unit}")


# ---------------------------------------------------------------------------
def build_halo(comm, eval_nodes, normals, bc_flags, centers, r,
               n_interp, assignment):
    patches, halo = Setup(comm, eval_nodes, normals, bc_flags, centers, r,
                          n_interp=n_interp, node_layout='vogel',
                          assignment=assignment,
                          K=64, n=16, m=48, eval_epsilon=0)
    return patches, halo


def build_orig(comm, eval_nodes, normals, bc_flags, centers, r, n_interp):
    return Setup_orig(comm, eval_nodes, normals, bc_flags, centers, r,
                      n_interp=n_interp, node_layout='vogel',
                      assignment='round_robin',
                      K=64, n=16, m=48, eval_epsilon=0)


def topo_row(comm, label, halo):
    n_s = sum(len(v) for v in halo.mv_send_gidx.values())
    n_r = sum(len(v) for v in halo.mv_recv_lidx.values())
    return (len(halo.owned_indices), len(halo.neighbor_ranks), n_s, n_r)


# ---------------------------------------------------------------------------
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    M        = 2000
    n_interp = 40
    H        = 0.2
    delta    = 0.2
    bc_scale = 100.0

    if rank == 0:
        print(f"\n{'═'*72}")
        print(f"  HaloProfile  ranks={size}  M={M}  n_interp={n_interp}  "
              f"patches={int((1/H)**2)}")
        print(f"{'═'*72}")
        eval_nodes, normals, groups = MinEnergySquareOne(M)
        centers, r = LarssonBox2D(H=H, xrange=(0,1), yrange=(0,1), delta=delta)
    else:
        eval_nodes = normals = groups = centers = r = None

    eval_nodes = comm.bcast(eval_nodes, root=0)
    normals    = comm.bcast(normals,    root=0)
    groups     = comm.bcast(groups,     root=0)
    centers    = comm.bcast(centers,    root=0)
    r          = comm.bcast(r,          root=0)

    bc_flags = np.empty(M, dtype=str)
    bc_flags[groups["boundary:all"]] = 'd'
    bc_flags[groups["interior"]]     = 'i'

    f = np.zeros(M)
    ii = groups["interior"]
    xi = eval_nodes[ii]
    f[ii] = -2.0 * np.pi**2 * np.sin(np.pi * xi[:, 0]) * np.sin(np.pi * xi[:, 1])

    # ---- build all three configurations ----
    patches_o               = build_orig(comm, eval_nodes, normals, bc_flags, centers, r, n_interp)
    patches_rr, halo_rr     = build_halo(comm, eval_nodes, normals, bc_flags, centers, r, n_interp, 'round_robin')
    patches_bl, halo_bl     = build_halo(comm, eval_nodes, normals, bc_flags, centers, r, n_interp, 'block_grid_2d')

    # ======================================================================
    # Section 1: topology
    # ======================================================================
    section(comm, "Topology (mean across ranks)")
    if comm.Get_rank() == 0:
        print(f"  {'Metric':<38}  {'allreduce':>9}  {'rr halo':>9}  {'block halo':>10}")
        print(f"  {'-'*70}")

    def topo(halo, field):
        _, me, _ = stats(comm, [field(halo)])
        return me

    for label, fn in [
        ("patches per rank",          lambda h: len(h.patch_halo)),
        ("owned nodes per rank",      lambda h: len(h.owned_indices)),
        ("neighbor ranks",            lambda h: len(h.neighbor_ranks)),
        ("halo send vol (nodes)",     lambda h: sum(len(v) for v in h.mv_send_gidx.values())),
        ("halo recv vol (nodes)",     lambda h: sum(len(v) for v in h.mv_recv_lidx.values())),
    ]:
        v_rr = topo(halo_rr, fn)
        v_bl = topo(halo_bl, fn)
        if comm.Get_rank() == 0:
            print(f"  {label:<38}  {'(global)':>9}  {v_rr:>9.0f}  {v_bl:>10.0f}")

    # These topo() calls are collective — must run on all ranks, outside if-guard
    sv_rr = topo(halo_rr, lambda h: sum(len(v) for v in h.mv_send_gidx.values()))
    sv_bl = topo(halo_bl, lambda h: sum(len(v) for v in h.mv_send_gidx.values()))
    rv_rr = topo(halo_rr, lambda h: sum(len(v) for v in h.mv_recv_lidx.values()))
    rv_bl = topo(halo_bl, lambda h: sum(len(v) for v in h.mv_recv_lidx.values()))
    if comm.Get_rank() == 0:
        print(f"\n  Allreduce volume : {M} floats = {M*8/1024:.1f} KB")
        print(f"  rr  halo volume  : {sv_rr+rv_rr:.0f} floats = {(sv_rr+rv_rr)*8/1024:.1f} KB per rank")
        print(f"  blk halo volume  : {sv_bl+rv_bl:.0f} floats = {(sv_bl+rv_bl)*8/1024:.1f} KB per rank")

    # ======================================================================
    # Section 2: isolated communication cost
    # ======================================================================
    section(comm, "Isolated communication cost  (no computation, mean across ranks)")
    hdr(comm)

    buf_M = np.zeros(M)
    for _ in range(N_WARMUP):
        comm.Allreduce(MPI.IN_PLACE, buf_M, op=MPI.SUM)
    t_ar = time_fn(lambda: comm.Allreduce(MPI.IN_PLACE, buf_M, op=MPI.SUM), N_REPS)

    def bare_mv(halo):
        zs = {s: np.zeros(len(halo.mv_send_gidx[s])) for s in halo.neighbor_ranks}
        for _ in range(N_WARMUP): halo.mv_exchange(zs, tag=90)
        return time_fn(lambda: halo.mv_exchange(zs, tag=90), N_REPS)

    def bare_rmv(halo):
        zo = np.zeros(len(halo.owned_indices))
        for _ in range(N_WARMUP): halo.rmv_exchange_1d(zo, tag=91)
        return time_fn(lambda: halo.rmv_exchange_1d(zo, tag=91), N_REPS)

    t_mv_rr  = bare_mv(halo_rr);  t_rmv_rr  = bare_rmv(halo_rr)
    t_mv_bl  = bare_mv(halo_bl);  t_rmv_bl  = bare_rmv(halo_bl)

    _, me_ar, _    = stats(comm, t_ar)
    _, me_mv_rr, _ = stats(comm, t_mv_rr);  _, me_rmv_rr, _ = stats(comm, t_rmv_rr)
    _, me_mv_bl, _ = stats(comm, t_mv_bl);  _, me_rmv_bl, _ = stats(comm, t_rmv_bl)

    row3(comm, "mv  exchange (mean)",  [me_ar,     me_mv_rr,  me_mv_bl])
    row3(comm, "rmv exchange (mean)",  [me_ar,     me_rmv_rr, me_rmv_bl])

    # ======================================================================
    # Section 3: full matvec / rmatvec
    # ======================================================================
    section(comm, "Full matvec / rmatvec  (computation + comm, mean across ranks)")
    hdr(comm)

    Rs_o  = PRM_orig(patches_o,  bc_scale)
    Rs_rr = PoissonRowMatrices(patches_rr, bc_scale)
    Rs_bl = PoissonRowMatrices(patches_bl, bc_scale)

    mv_o,  rmv_o  = GMO_orig(comm, patches_o,  Rs_o,  M, n_interp)
    mv_rr, rmv_rr = GenMatFreeOps(patches_rr, Rs_rr, halo_rr, n_interp)
    mv_bl, rmv_bl = GenMatFreeOps(patches_bl, Rs_bl, halo_bl, n_interp)

    rng     = np.random.default_rng(seed=7)
    v_rr    = rng.standard_normal(len(patches_rr) * n_interp)
    v_bl    = rng.standard_normal(len(patches_bl) * n_interp)
    u_rr    = rng.standard_normal(len(halo_rr.owned_indices))
    u_bl    = rng.standard_normal(len(halo_bl.owned_indices))
    u_glob  = rng.standard_normal(M)

    for _ in range(N_WARMUP):
        mv_o(v_rr); rmv_o(u_glob)
        mv_rr(v_rr); rmv_rr(u_rr)
        mv_bl(v_bl); rmv_bl(u_bl)

    t_mv_o   = time_fn(lambda: mv_o(v_rr),    N_REPS)
    t_mv_rr  = time_fn(lambda: mv_rr(v_rr),   N_REPS)
    t_mv_bl  = time_fn(lambda: mv_bl(v_bl),   N_REPS)
    t_rmv_o  = time_fn(lambda: rmv_o(u_glob), N_REPS)
    t_rmv_rr = time_fn(lambda: rmv_rr(u_rr),  N_REPS)
    t_rmv_bl = time_fn(lambda: rmv_bl(u_bl),  N_REPS)

    _, me_mv_o,   _ = stats(comm, t_mv_o)
    _, me_mv_rr,  _ = stats(comm, t_mv_rr)
    _, me_mv_bl,  _ = stats(comm, t_mv_bl)
    _, me_rmv_o,  _ = stats(comm, t_rmv_o)
    _, me_rmv_rr, _ = stats(comm, t_rmv_rr)
    _, me_rmv_bl, _ = stats(comm, t_rmv_bl)

    row3(comm, "matvec  (mean)",  [me_mv_o,  me_mv_rr,  me_mv_bl])
    row3(comm, "rmatvec (mean)",  [me_rmv_o, me_rmv_rr, me_rmv_bl])

    if comm.Get_rank() == 0:
        print(f"\n  matvec  speedup vs allreduce:  rr={me_mv_o/me_mv_rr:.2f}x  "
              f"block={me_mv_o/me_mv_bl:.2f}x")
        print(f"  rmatvec speedup vs allreduce:  rr={me_rmv_o/me_rmv_rr:.2f}x  "
              f"block={me_rmv_o/me_rmv_bl:.2f}x")

    # ======================================================================
    # Section 4: full LSQR solve
    # ======================================================================
    section(comm, "Full LSQR solve  (block-Jacobi, atol=btol=1e-10)")

    f_rr = f[halo_rr.owned_indices]
    f_bl = f[halo_bl.owned_indices]

    solve_o  = GIS_orig(comm, patches_o,  M,       n_interp, bc_scale=bc_scale,
                        preconditioner='block_jacobi', atol=1e-10, btol=1e-10, maxiter=5000)
    solve_rr = GenIterativeSolver(comm, patches_rr, halo_rr, n_interp, bc_scale=bc_scale,
                                  preconditioner='block_jacobi', atol=1e-10, btol=1e-10, maxiter=5000)
    solve_bl = GenIterativeSolver(comm, patches_bl, halo_bl, n_interp, bc_scale=bc_scale,
                                  preconditioner='block_jacobi', atol=1e-10, btol=1e-10, maxiter=5000)

    def timed_solve(fn, arg):
        comm.Barrier(); t0 = MPI.Wtime()
        _, itn, rnorm = fn(arg)
        return MPI.Wtime() - t0, itn, rnorm

    t_o,  itn_o,  rn_o  = timed_solve(solve_o,  f)
    t_rr, itn_rr, rn_rr = timed_solve(solve_rr, f_rr)
    t_bl, itn_bl, rn_bl = timed_solve(solve_bl, f_bl)

    if comm.Get_rank() == 0:
        print(f"\n  {'method':<14}  {'iters':>6}  {'rnorm':>10}  {'time (s)':>10}  {'speedup':>8}")
        print(f"  {'-'*54}")
        print(f"  {'allreduce':<14}  {itn_o:>6}  {rn_o:>10.3e}  {t_o:>10.3f}  {'1.00x':>8}")
        print(f"  {'rr halo':<14}  {itn_rr:>6}  {rn_rr:>10.3e}  {t_rr:>10.3f}  {t_o/t_rr:>7.2f}x")
        print(f"  {'block halo':<14}  {itn_bl:>6}  {rn_bl:>10.3e}  {t_bl:>10.3f}  {t_o/t_bl:>7.2f}x")
        print(f"\n  Parallel efficiency (vs 1 rank): run at P=1,2,4,8 and compare solve times")
        print(f"\n{'═'*72}\n")


if __name__ == '__main__':
    main()
