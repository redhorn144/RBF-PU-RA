"""
ScaleProbe.py — measures setup time and RSS memory at increasing M.

Run single-rank to get per-rank baseline, then estimate 16-rank total.
    python ScaleProbe.py
"""
import numpy as np
import time
import resource

from mpi4py import MPI
from nodes.SquareDomain import MinEnergySquareOne
from source_halo.PatchTiling import LarssonBox2D
from source_halo.LSSetup import Setup

comm = MPI.COMM_WORLD

H       = 0.1
delta   = 0.3
n_interp = 40
K, n, m  = 64, 16, 48

# Theoretical n_eval_p per patch (for sanity-check)
r_theory = (1 + delta) * np.sqrt(2) * H / 2

print(f"H={H}  delta={delta}  n_interp={n_interp}  K={K}")
print(f"r={r_theory:.4f}   n_patches={(int(1/H))**2}")
print(f"{'M':>8}  {'n_eval_p':>10}  {'setup(s)':>10}  {'RSS(MB)':>10}  "
      f"{'patch_mem(MB)':>14}  {'extrap16(GB)':>12}")
print("-" * 72)

for M in [2_000, 5_000, 10_000, 20_000, 50_000, 100_000]:
    t0 = time.perf_counter()

    if comm.Get_rank() == 0:
        eval_nodes, normals, groups = MinEnergySquareOne(M)
        centers, r = LarssonBox2D(H=H, xrange=(0, 1), yrange=(0, 1), delta=delta)
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

    patches, halo = Setup(comm, eval_nodes, normals, bc_flags, centers, r,
                          n_interp=n_interp, node_layout='vogel',
                          K=K, n=n, m=m, eval_epsilon=0)

    dt  = time.perf_counter() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # kB on Linux

    # Measure actual patch matrix memory
    patch_bytes = sum(
        p.E.nbytes + p.D.nbytes + p.L.nbytes
        + p.w_bar.nbytes + p.gw_bar.nbytes + p.lw_bar.nbytes
        for p in patches
    )
    mean_n_eval = np.mean([len(p.eval_node_indices) for p in patches]) if patches else 0

    # Extrapolate to 16 ranks: global arrays replicated × 16, patch mem × 16
    global_bytes_per_rank = (eval_nodes.nbytes + normals.nbytes
                              + halo.g2l.nbytes + halo.owned_indices.nbytes)
    extrap_16 = (global_bytes_per_rank * 16 + patch_bytes * 16) / 1e9

    if comm.Get_rank() == 0:
        print(f"{M:>8}  {mean_n_eval:>10.0f}  {dt:>10.2f}  {rss/1e3:>10.1f}  "
              f"{patch_bytes/1e6:>14.1f}  {extrap_16:>12.2f}")
