"""
ScalingTest.py — component-level scaling analysis for the halo-exchange backend.

Measures wall-clock time of each separable phase across two axes:

  Size sweep (default) — fixed rank count, loop over M_CONFIGS:
      mpirun -n 4 python ScalingTest.py

  Single configuration — for rank-sweep or weak-scaling analysis:
      mpirun -n 4 python ScalingTest.py --single 10000 0.09

  Rank sweep (run at each rank count, grep SCALING_ROW for a table):
      for P in 1 2 4 8; do
          mpirun -n $P python ScalingTest.py --single 10000 0.09
      done | grep SCALING_ROW

  Weak scaling (M grows with ranks):
      for P in 1 2 4 8; do
          mpirun -n $P python ScalingTest.py --single $((2500*P)) 0.09
      done | grep SCALING_ROW
"""

from mpi4py import MPI
import numpy as np
import sys

from nodes.SquareDomain       import MinEnergySquareOne
from source_halo.PatchTiling  import LarssonBox2D
from source_halo.PatchNodes   import GenPatchNodes
from source_halo.RAHelpers    import PhiFactors
from source_halo.LSSetup      import Setup
from source_halo.HaloComm     import build_halo_comm
from source_halo.PUWeights    import NormalizeWeights
from source_halo.Operators    import PoissonRowMatrices, GenMatFreeOps
from source_halo.Solvers      import GenIterativeSolver

# ---------------------------------------------------------------------------
# Problem constants
# ---------------------------------------------------------------------------
N_INTERP = 40
K, n, m  = 64, 16, 48
DELTA    = 0.2
BC_SCALE = 100.0

# Timing repetitions
N_WARMUP      = 3
N_REPS        = 20   # per-iteration ops (matvec, exchanges, allreduce)
N_REPS_SETUP  =  5   # setup sub-components (heavier, fewer reps)

# Size-sweep configs: H chosen to keep n_eval_p ≈ 3*N_INTERP across all M.
M_CONFIGS = [
    ( 2_000, 0.20),
    ( 5_000, 0.13),
    (10_000, 0.09),
    (20_000, 0.065),
    (50_000, 0.04),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def time_fn(comm, fn, n_reps):
    """
    Barrier-bracketed wall-clock timing.

    Returns an (n_reps,) array of wall times.  All ranks start and end
    together, so every rank records the same global wall time (max across ranks
    is implicit in the barrier).  stats() over the array gives the rep-to-rep
    distribution of that critical-path time.
    """
    times = np.empty(n_reps)
    for i in range(n_reps):
        comm.Barrier()
        t0 = MPI.Wtime()
        fn()
        comm.Barrier()
        times[i] = MPI.Wtime() - t0
    return times


def stats(comm, vals):
    """
    (min, mean, max) of a collection of per-rank values across all ranks.

    Pass a scalar or array.  When used with time_fn output (barrier-bracketed),
    all ranks have identical values, so this gives rep-to-rep statistics.
    When used with a single per-rank scalar (e.g. topology counts), it gives
    the spread across ranks.
    """
    arr = np.asarray(vals, dtype=float).ravel()
    mn  = comm.allreduce(float(arr.min()), op=MPI.MIN)
    mx  = comm.allreduce(float(arr.max()), op=MPI.MAX)
    sm  = comm.allreduce(float(arr.sum()), op=MPI.SUM)
    cnt = comm.allreduce(len(arr),         op=MPI.SUM)
    return mn, sm / cnt, mx


def pct(part, total):
    return f"{100*part/total:5.1f}%" if total > 0 else "  n/a"


def pr(comm, line=""):
    if comm.Get_rank() == 0:
        print(line)


def section(comm, title):
    if comm.Get_rank() == 0:
        print(f"\n{'═'*72}")
        print(f"  {title}")
        print(f"{'═'*72}")


# ---------------------------------------------------------------------------
# Reconstruct patch_rank from already-built patches (needed to re-time
# build_halo_comm without modifying LSSetup internals).
# ---------------------------------------------------------------------------
def recompute_patch_rank(comm, patches, n_patches):
    patch_rank_local = np.full(n_patches, -1, dtype=np.int32)
    for p in patches:
        patch_rank_local[p.global_pid] = comm.Get_rank()
    patch_rank = np.empty(n_patches, dtype=np.int32)
    comm.Allreduce(patch_rank_local, patch_rank, op=MPI.MAX)
    return patch_rank


# ---------------------------------------------------------------------------
# Single-configuration run
# ---------------------------------------------------------------------------
def run_config(comm, M, H):
    rank = comm.Get_rank()
    size = comm.Get_size()

    # ---- domain (rank 0 generates, all ranks receive) ----
    if rank == 0:
        eval_nodes, normals, groups = MinEnergySquareOne(M)
        centers, r = LarssonBox2D(H=H, xrange=(0, 1), yrange=(0, 1), delta=DELTA)
    else:
        eval_nodes = normals = groups = centers = r = None

    eval_nodes = comm.bcast(eval_nodes, root=0)
    normals    = comm.bcast(normals,    root=0)
    groups     = comm.bcast(groups,     root=0)
    centers    = comm.bcast(centers,    root=0)
    r          = comm.bcast(r,          root=0)

    M_actual = len(eval_nodes)
    P        = len(centers)

    bc_flags = np.empty(M_actual, dtype=str)
    bc_flags[groups["boundary:all"]] = 'd'
    bc_flags[groups["interior"]]     = 'i'

    f_global = np.zeros(M_actual)
    ii = groups["interior"]
    xi = eval_nodes[ii]
    f_global[ii] = -2.0 * np.pi**2 * np.sin(np.pi*xi[:, 0]) * np.sin(np.pi*xi[:, 1])

    # ---- total Setup time (single call, barrier-bracketed) ----
    comm.Barrier()
    t0 = MPI.Wtime()
    patches, halo = Setup(comm, eval_nodes, normals, bc_flags, centers, r,
                          n_interp=N_INTERP, node_layout='vogel',
                          assignment='block_grid_2d',
                          K=K, n=n, m=m, eval_epsilon=0)
    comm.Barrier()
    t_setup = MPI.Wtime() - t0

    # ---- topology metrics ----
    P_r_local      = len(patches)
    n_owned_local  = len(halo.owned_indices)
    n_nbr_local    = len(halo.neighbor_ranks)
    n_eval_p_local = (float(np.mean([len(p.eval_node_indices) for p in patches]))
                      if patches else 0.0)
    halo_send_vol  = sum(len(v) for v in halo.mv_send_gidx.values())
    halo_recv_vol  = sum(len(v) for v in halo.mv_recv_lidx.values())

    _, mean_Pr,       _ = stats(comm, [P_r_local])
    _, mean_owned,    _ = stats(comm, [n_owned_local])
    _, mean_nbr,      _ = stats(comm, [n_nbr_local])
    _, mean_n_eval_p, _ = stats(comm, [n_eval_p_local])
    _, mean_send,     _ = stats(comm, [halo_send_vol])
    _, mean_recv,     _ = stats(comm, [halo_recv_vol])

    # ---- PhiFactors + bcast (N_REPS_SETUP reps, includes rank-0 compute) ----
    patch_nodes_base = GenPatchNodes(N_INTERP, r, eval_nodes.shape[1], 'vogel')
    phi_times = []
    for _ in range(N_REPS_SETUP):
        if rank == 0:
            phi_lus, Er, Es = PhiFactors(patch_nodes_base, K=K)
        else:
            phi_lus = Er = Es = None
        comm.Barrier()
        t0 = MPI.Wtime()
        phi_lus = comm.bcast(phi_lus, root=0)
        Er      = comm.bcast(Er,      root=0)
        Es      = comm.bcast(Es,      root=0)
        comm.Barrier()
        phi_times.append(MPI.Wtime() - t0)
    # PhiFactors compute (rank 0) + bcast
    phi_compute_times = []
    for _ in range(N_REPS_SETUP):
        if rank == 0:
            comm.Barrier()
            t0 = MPI.Wtime()
            phi_lus, Er, Es = PhiFactors(patch_nodes_base, K=K)
            phi_compute_times.append(MPI.Wtime() - t0)
            phi_lus = comm.bcast(phi_lus, root=0)
            Er      = comm.bcast(Er,      root=0)
            Es      = comm.bcast(Es,      root=0)
            comm.Barrier()
        else:
            comm.Barrier()
            phi_lus = Er = Es = None
            phi_lus = comm.bcast(phi_lus, root=0)
            Er      = comm.bcast(Er,      root=0)
            Es      = comm.bcast(Es,      root=0)
            comm.Barrier()
    t_phi_bcast   = float(np.mean(phi_times))
    t_phi_compute = float(np.mean(phi_compute_times)) if rank == 0 else 0.0
    t_phi_compute = comm.bcast(t_phi_compute, root=0)

    # ---- build_halo_comm alone (N_REPS_SETUP reps on already-built patches) ----
    patch_rank = recompute_patch_rank(comm, patches, P)
    t_hc = time_fn(comm,
                   lambda: build_halo_comm(comm, patches, eval_nodes,
                                           centers, r, patch_rank),
                   N_REPS_SETUP)
    _, t_halo_comm, _ = stats(comm, t_hc)

    # ---- NormalizeWeights alone (N_REPS_SETUP reps; idempotent) ----
    t_nw = time_fn(comm, lambda: NormalizeWeights(patches, halo), N_REPS_SETUP)
    _, t_norm, _ = stats(comm, t_nw)

    # Derived patch-build time (includes KDTree build inside Setup).
    # Subtract both the rank-0 PhiFactors compute and the subsequent bcast,
    # since both contribute to t_setup sequentially.
    t_patch_build = max(0.0, t_setup - (t_phi_compute + t_phi_bcast) - t_halo_comm - t_norm)

    # ---- operators ----
    Rs              = PoissonRowMatrices(patches, BC_SCALE)
    matvec, rmatvec = GenMatFreeOps(patches, Rs, halo, N_INTERP)

    # ---- matvec / rmatvec (N_REPS reps) ----
    rng   = np.random.default_rng(seed=42)
    v_loc = rng.standard_normal(len(patches) * N_INTERP)
    u_own = rng.standard_normal(n_owned_local)

    for _ in range(N_WARMUP):
        matvec(v_loc)
        rmatvec(u_own)

    t_mv_arr  = time_fn(comm, lambda: matvec(v_loc),  N_REPS)
    t_rmv_arr = time_fn(comm, lambda: rmatvec(u_own), N_REPS)

    _, t_mv_mean,  t_mv_max  = stats(comm, t_mv_arr)
    _, t_rmv_mean, t_rmv_max = stats(comm, t_rmv_arr)

    # ---- halo exchange only (no compute, zero data) ----
    zero_bufs = {s: np.zeros(len(halo.mv_send_gidx[s])) for s in halo.neighbor_ranks}
    zero_own  = np.zeros(n_owned_local)

    for _ in range(N_WARMUP):
        halo.mv_exchange(zero_bufs,  tag=80)
        halo.rmv_exchange_1d(zero_own, tag=81)

    t_cmv_arr  = time_fn(comm, lambda: halo.mv_exchange(zero_bufs, tag=80),     N_REPS)
    t_crmv_arr = time_fn(comm, lambda: halo.rmv_exchange_1d(zero_own, tag=81),  N_REPS)

    _, t_comm_mv,  _ = stats(comm, t_cmv_arr)
    _, t_comm_rmv, _ = stats(comm, t_crmv_arr)

    # ---- scalar Allreduce (LSQR barrier baseline) ----
    for _ in range(N_WARMUP):
        comm.allreduce(0.0, op=MPI.SUM)

    t_ar_arr = time_fn(comm, lambda: comm.allreduce(0.0, op=MPI.SUM), N_REPS)
    _, t_ar_mean, _ = stats(comm, t_ar_arr)

    # ---- full LSQR solve (single timed call) ----
    f_owned = f_global[halo.owned_indices]
    solve   = GenIterativeSolver(comm, patches, halo, N_INTERP, Rs,
                                 preconditioner='sas',
                                 atol=1e-10, btol=1e-10, maxiter=5000)
    comm.Barrier()
    t0 = MPI.Wtime()
    _, itn, rnorm = solve(f_owned)
    comm.Barrier()
    t_solve = MPI.Wtime() - t0

    # ---- print report ----
    comm_frac_mv  = 100.0 * t_comm_mv  / t_mv_mean  if t_mv_mean  > 0 else 0.0
    comm_frac_rmv = 100.0 * t_comm_rmv / t_rmv_mean if t_rmv_mean > 0 else 0.0

    section(comm, f"M={M_actual}  P={P}  n_interp={N_INTERP}  ranks={size}  H={H}")

    pr(comm, f"  Topology (mean across ranks)")
    pr(comm, f"    patches / rank     : {mean_Pr:8.1f}")
    pr(comm, f"    owned nodes / rank : {mean_owned:8.0f}")
    pr(comm, f"    eval nodes / patch : {mean_n_eval_p:8.1f}")
    pr(comm, f"    neighbor ranks     : {mean_nbr:8.1f}")
    pr(comm, f"    halo send (nodes)  : {mean_send:8.0f}")
    pr(comm, f"    halo recv (nodes)  : {mean_recv:8.0f}")

    pr(comm)
    pr(comm, f"  Setup breakdown  (wall-clock; sub-times are mean over {N_REPS_SETUP} reps)")
    pr(comm, f"    Total setup            : {t_setup:8.3f} s")
    pr(comm, f"    ├─ PhiFactors (rank 0) : {t_phi_compute:8.3f} s   ({pct(t_phi_compute, t_setup)})")
    pr(comm, f"    ├─ PhiFactors bcast    : {t_phi_bcast:8.4f} s   ({pct(t_phi_bcast,   t_setup)})")
    pr(comm, f"    ├─ Patch build  (≈)    : {t_patch_build:8.3f} s   ({pct(t_patch_build, t_setup)})")
    pr(comm, f"    ├─ build_halo_comm     : {t_halo_comm:8.3f} s   ({pct(t_halo_comm,   t_setup)})")
    pr(comm, f"    └─ NormalizeWeights    : {t_norm:8.3f} s   ({pct(t_norm,         t_setup)})")

    pr(comm)
    pr(comm, f"  Per-iteration  (mean ms  [max ms] over {N_REPS} reps, barrier-bracketed)")
    pr(comm, f"    matvec                 : {1e3*t_mv_mean:7.3f} ms  [max {1e3*t_mv_max:7.3f}]")
    pr(comm, f"    rmatvec                : {1e3*t_rmv_mean:7.3f} ms  [max {1e3*t_rmv_max:7.3f}]")
    pr(comm, f"    mv  halo exchange only : {1e3*t_comm_mv:7.3f} ms              "
             f"comm fraction {comm_frac_mv:.1f}%")
    pr(comm, f"    rmv halo exchange only : {1e3*t_comm_rmv:7.3f} ms              "
             f"comm fraction {comm_frac_rmv:.1f}%")
    pr(comm, f"    scalar Allreduce       : {1e3*t_ar_mean:7.3f} ms")
    pr(comm, f"    est. barriers/iter     : {1e3*2*t_ar_mean:7.3f} ms  (2 per iter: β norm + α norm)")

    pr(comm)
    pr(comm, f"  PCG (SAS), atol=btol=1e-10)")
    pr(comm, f"    itn={itn}  rnorm={rnorm:.3e}  time={t_solve:.2f} s")

    # Machine-parseable line for rank-sweep and weak-scaling analysis
    pr(comm,
       f"SCALING_ROW"
       f" ranks={size}"
       f" M={M_actual}"
       f" H={H}"
       f" P={P}"
       f" t_setup={t_setup:.4f}"
       f" t_phi_compute={t_phi_compute:.4f}"
       f" t_phi_bcast={t_phi_bcast:.4f}"
       f" t_patch={t_patch_build:.4f}"
       f" t_halo_comm={t_halo_comm:.4f}"
       f" t_norm={t_norm:.4f}"
       f" t_mv={1e3*t_mv_mean:.4f}"
       f" t_rmv={1e3*t_rmv_mean:.4f}"
       f" t_comm_mv={1e3*t_comm_mv:.4f}"
       f" t_comm_rmv={1e3*t_comm_rmv:.4f}"
       f" t_ar={1e3*t_ar_mean:.4f}"
       f" t_solve={t_solve:.4f}"
       f" itn={itn}"
       f" rnorm={rnorm:.4e}"
       f" P_r={mean_Pr:.0f}"
       f" n_owned={mean_owned:.0f}"
    )

    return dict(
        M=M_actual, H=H, P=P,
        t_setup=t_setup, t_phi_compute=t_phi_compute, t_phi_bcast=t_phi_bcast,
        t_patch_build=t_patch_build, t_halo_comm=t_halo_comm, t_norm=t_norm,
        t_mv=t_mv_mean, t_rmv=t_rmv_mean,
        t_comm_mv=t_comm_mv, t_comm_rmv=t_comm_rmv,
        t_ar=t_ar_mean, t_solve=t_solve,
        itn=itn, rnorm=rnorm,
        mean_Pr=mean_Pr, mean_owned=mean_owned,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(f"\n{'═'*72}")
        print(f"  ScalingTest  ranks={size}  n_interp={N_INTERP}  "
              f"K={K}  n={n}  m={m}  DELTA={DELTA}")
        print(f"{'═'*72}")

    if '--single' in sys.argv:
        idx = sys.argv.index('--single')
        M   = int(sys.argv[idx + 1])
        H   = float(sys.argv[idx + 2])
        run_config(comm, M, H)
    else:
        results = []
        for M, H in M_CONFIGS:
            results.append(run_config(comm, M, H))

        # Compact summary table across all sizes
        section(comm, f"Size-sweep summary  (ranks={size})")
        pr(comm,
           f"  {'M':>8}  {'P':>5}  {'P_r':>5}  {'Setup(s)':>9}  "
           f"{'Patch(s)':>9}  {'HaloComm(s)':>11}  "
           f"{'MV(ms)':>7}  {'CommFrac':>9}  {'Solve(s)':>9}  {'itn':>5}")
        pr(comm, f"  {'-'*90}")
        for res in results:
            cf = 100.0 * res['t_comm_mv'] / res['t_mv'] if res['t_mv'] > 0 else 0.0
            pr(comm,
               f"  {res['M']:>8}  {res['P']:>5}  {res['mean_Pr']:>5.0f}  "
               f"{res['t_setup']:>9.2f}  {res['t_patch_build']:>9.2f}  "
               f"{res['t_halo_comm']:>11.3f}  "
               f"{1e3*res['t_mv']:>7.3f}  {cf:>8.1f}%  "
               f"{res['t_solve']:>9.2f}  {res['itn']:>5}")

    if rank == 0:
        print(f"\n{'═'*72}\n")


if __name__ == '__main__':
    main()
