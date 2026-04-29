"""
HaloSanityCheck.py — correctness tests for the source_halo halo-exchange path.

Run with:
    mpirun -n 4 python HaloSanityCheck.py

Tests
-----
1. Ownership partition  — owned_indices across all ranks is a complete, non-
                          overlapping cover of {0, …, M-1}.
2. PU property          — Σ_p w_bar_p(x_i) = 1 at every eval node.
3. Matvec adjointness   — ⟨A v, u⟩ = ⟨v, A^T u⟩ for random v, u.
4. Source consistency   — matvec(v) via halo matches matvec(v) via the original
                          global Allreduce (Operators from source/).
5. Solve correctness    — LSQR on Δu = f with known exact solution reaches the
                          expected error floor.
"""

from mpi4py import MPI
import numpy as np

from nodes.SquareDomain import MinEnergySquareOne
from source_halo.PatchTiling import LarssonBox2D
from source_halo.LSSetup import Setup
from source_halo.Operators import PoissonRowMatrices, GenMatFreeOps
from source_halo.Solvers import GenIterativeSolver

# For test 4: original Allreduce-based matvec
from source.Operators import (PoissonRowMatrices as PoissonRowMatrices_orig,
                               GenMatFreeOps as GenMatFreeOps_orig)


def _pass(comm, label):
    if comm.Get_rank() == 0:
        print(f"  PASS  {label}")


def _fail(comm, label, detail=""):
    if comm.Get_rank() == 0:
        print(f"  FAIL  {label}" + (f": {detail}" if detail else ""))


def check(comm, ok, label, detail=""):
    if comm.Get_rank() == 0:
        tag = "PASS" if ok else "FAIL"
        print(f"  {tag}  {label}" + (f": {detail}" if detail else ""))


# ---------------------------------------------------------------------------
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    M        = 1000
    n_interp = 30
    H        = 0.25
    delta    = 0.3
    bc_scale = 100.0

    if rank == 0:
        print(f"\n=== HaloSanityCheck  ranks={size}  M={M}  n_interp={n_interp} ===\n")
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

                          K=64, n=16, m=48, eval_epsilon=0)

    # ---------------------------------------------------------------------- #
    # Test 1: ownership partition                                             #
    # ---------------------------------------------------------------------- #
    if rank == 0:
        print("Test 1: ownership partition")

    all_owned = comm.gather(halo.owned_indices, root=0)
    if rank == 0:
        combined = np.concatenate(all_owned)
        unique   = np.unique(combined)
        ok = (len(combined) == M) and np.array_equal(unique, np.arange(M))
        check(comm, ok, f"owned_indices covers {{0..{M-1}}} exactly once",
              f"total={len(combined)} unique={len(unique)}")
    comm.barrier()

    # ---------------------------------------------------------------------- #
    # Test 2: PU property  Σ w_bar_p(x_i) = 1                               #
    # ---------------------------------------------------------------------- #
    if rank == 0:
        print("\nTest 2: PU property")

    W_local = np.zeros(M)
    for p in patches:
        W_local[p.eval_node_indices] += p.w_bar
    W_global = np.zeros(M)
    comm.Allreduce(W_local, W_global, op=MPI.SUM)

    if rank == 0:
        err = np.max(np.abs(W_global - 1.0))
        check(comm, err < 1e-13, "Σ w_bar = 1 at all eval nodes",
              f"max|Σw - 1| = {err:.2e}")

    # ---------------------------------------------------------------------- #
    # Test 3: matvec adjointness  ⟨Av, u⟩ = ⟨v, A^T u⟩                     #
    # ---------------------------------------------------------------------- #
    if rank == 0:
        print("\nTest 3: matvec adjointness")

    Rs         = PoissonRowMatrices(patches, bc_scale)
    matvec, rmatvec = GenMatFreeOps(patches, Rs, halo, n_interp)

    rng = np.random.default_rng(seed=rank + 42)
    n_local  = len(patches) * n_interp
    n_owned  = len(halo.owned_indices)

    v_local  = rng.standard_normal(n_local)
    u_owned  = rng.standard_normal(n_owned)

    Av       = matvec(v_local)       # (n_owned,)
    Atu      = rmatvec(u_owned)      # (n_local,)

    lhs = comm.allreduce(np.dot(Av,  u_owned),  op=MPI.SUM)  # ⟨Av, u⟩
    rhs = comm.allreduce(np.dot(v_local, Atu),  op=MPI.SUM)  # ⟨v, A^T u⟩

    if rank == 0:
        rel_err = abs(lhs - rhs) / (abs(lhs) + 1e-300)
        check(comm, rel_err < 1e-12, "⟨Av, u⟩ = ⟨v, A^T u⟩",
              f"lhs={lhs:.10e}  rhs={rhs:.10e}  rel_err={rel_err:.2e}")

    # ---------------------------------------------------------------------- #
    # Test 4: consistency with original Allreduce-based matvec               #
    # ---------------------------------------------------------------------- #
    if rank == 0:
        print("\nTest 4: halo matvec == original Allreduce matvec")

    Rs_orig = PoissonRowMatrices_orig(patches, bc_scale)
    matvec_orig, _ = GenMatFreeOps_orig(comm, patches, Rs_orig, M, n_interp)

    Av_halo = matvec(v_local)                 # (n_owned,) distributed
    Av_full = matvec_orig(v_local)            # (M,) global

    # Compare at owned indices
    Av_halo_check = Av_halo
    Av_full_check = Av_full[halo.owned_indices]

    local_err  = np.max(np.abs(Av_halo_check - Av_full_check)) if n_owned else 0.0
    global_err = comm.allreduce(local_err, op=MPI.MAX)

    if rank == 0:
        check(comm, global_err < 1e-12,
              "halo matvec matches Allreduce matvec at owned nodes",
              f"max|diff| = {global_err:.2e}")

    # ---------------------------------------------------------------------- #
    # Test 5: LSQR solve error                                                #
    # ---------------------------------------------------------------------- #
    if rank == 0:
        print("\nTest 5: LSQR solve (Poisson Δu = f, exact sol = sin(πx)sin(πy))")

    f = np.zeros(M)
    ii = groups["interior"]
    xi = eval_nodes[ii]
    f[ii] = -2.0 * np.pi**2 * np.sin(np.pi * xi[:, 0]) * np.sin(np.pi * xi[:, 1])

    u_exact = np.sin(np.pi * eval_nodes[:, 0]) * np.sin(np.pi * eval_nodes[:, 1])

    f_owned = f[halo.owned_indices]

    Rs_poisson = PoissonRowMatrices(patches, bc_scale)
    solve = GenIterativeSolver(comm, patches, halo, n_interp, Rs_poisson,
                               preconditioner='sas',
                               atol=1e-10, btol=1e-10, maxiter=10000)
    local_cs, itn, rnorm = solve(f_owned)

    # Reconstruct global PUM interpolant for error measurement
    U_local = np.zeros(M)
    for pi, (p, c) in enumerate(zip(patches, local_cs)):
        U_local[p.eval_node_indices] += p.w_bar * (p.E @ c)
    U = np.zeros(M)
    comm.Allreduce(U_local, U, op=MPI.SUM)

    if rank == 0:
        err_inf = np.max(np.abs(U - u_exact))
        err_l2  = np.sqrt(np.mean((U - u_exact) ** 2))
        converged = itn < 10000
        check(comm, converged and err_inf < 1e-4,
              f"LSQR converged  itn={itn}  rnorm={rnorm:.2e}  "
              f"max|U-u_ex|={err_inf:.2e}  L2={err_l2:.2e}")

    if rank == 0:
        print()


if __name__ == '__main__':
    main()
