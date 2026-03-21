import numpy as np
from scipy.sparse.linalg import spsolve
from mpi4py import MPI
from nodes.SquareDomain import PoissonSquareOne
from source.Setup import Setup
from source.Galerkin import assemble_advection, apply_dirichlet
from source.Plotter import AnimateSolution

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# ── Node generation ───────────────────────────────────────────────────────────
if rank == 0:
    nodes, normals, groups = PoissonSquareOne(0.025)
    print(f"Rank {rank} generated {nodes.shape[0]} nodes.")
else:
    nodes   = None
    normals = None
    groups  = None

nodes   = comm.bcast(nodes,   root=0)
normals = comm.bcast(normals, root=0)
groups  = comm.bcast(groups,  root=0)

N = nodes.shape[0]
bdy_indices = groups['boundary:all']

# ── Patch setup ───────────────────────────────────────────────────────────────
patches, _ = Setup(comm, nodes, normals, 80, bdy_indices=bdy_indices)
print(f"Rank {rank} setup complete with {len(patches)} patches.")

# ── Problem definition ────────────────────────────────────────────────────────
# Solid-body rotation about domain centre (0.5, 0.5) with angular velocity omega.
# Velocity field:  c(x,y) = omega * (-(y-0.5),  (x-0.5))
omega = 1.0

def cx_fn(x): return -omega * (x[:, 1] - 0.5)
def cy_fn(x): return  omega * (x[:, 0] - 0.5)

# ── Galerkin assembly (MPI-collective) ────────────────────────────────────────
if rank == 0:
    t_start = MPI.Wtime()

M, A = assemble_advection(comm, patches, N, cx_fn, cy_fn)

if rank == 0:
    t_assemble = MPI.Wtime()
    print(f"Assembly complete in {t_assemble - t_start:.2f} s")

# ── Apply Dirichlet BCs to M and A ────────────────────────────────────────────
# For the system  (M + dt*A) u^{n+1} = M u^n, Dirichlet rows must enforce
# u[bdy] = 0 at every step. We bake it in by zeroing boundary rows/cols
# of both matrices and placing 1 on the diagonal of the system matrix.
if rank == 0:
    M_arr = M.toarray()
    A_arr = A.toarray()

    # Zero boundary rows and columns of M and A; set diagonal of M to 1
    # so that the mass-weighted BC rows become: 1*u[bdy]^{n+1} = 0.
    zeros_b = np.zeros(len(bdy_indices))
    M_arr[bdy_indices, :] = 0.0
    M_arr[:, bdy_indices] = 0.0
    M_arr[bdy_indices, bdy_indices] = 1.0

    A_arr[bdy_indices, :] = 0.0
    A_arr[:, bdy_indices] = 0.0

# ── Initial condition ─────────────────────────────────────────────────────────
if rank == 0:
    x0, y0, sigma = 0.5, 0.70, 0.07
    u0 = np.exp(-((nodes[:, 0] - x0)**2 + (nodes[:, 1] - y0)**2) / (2.0 * sigma**2))
    u0[bdy_indices] = 0.0
    u = u0.copy()

# ── Implicit Euler time stepping ──────────────────────────────────────────────
# Galerkin system:   M (u^{n+1} - u^n)/dt + A u^{n+1} = 0
#   =>  (M + dt*A) u^{n+1} = M u^n
#
# With the Galerkin mass matrix M (symmetric positive definite for interior DOFs)
# and skew-symmetric-in-the-continuum A, the discrete eigenvalues of M^{-1}A are
# purely imaginary for solid-body rotation — no spurious positive-real modes.
# This means any A-stable scheme (e.g. Crank-Nicolson) conserves amplitude.
# Implicit Euler is used here for simplicity; switch to CN for better accuracy.

if rank == 0:
    T       = 2.0 * np.pi / omega
    dt      = 0.2
    n_steps = int(round(T / dt))

    snapshots = [u.copy()]
    times     = [0.0]

    t = 0.0
    for step in range(n_steps):
        lhs = M_arr + dt * A_arr
        rhs = M_arr @ u
        u = spsolve(lhs, rhs)
        u[bdy_indices] = 0.0
        t += dt
        snapshots.append(u.copy())
        times.append(t)

        idx = np.argmax(u)
        print(f"  step {step+1:3d}  t={t:.2f}  peak={u.max():.4f}"
              f"  loc=({nodes[idx, 0]:.3f},{nodes[idx, 1]:.3f})")

    t_end = MPI.Wtime()
    print(f"Time stepping complete: {n_steps} steps in {t_end - t_assemble:.2f} s")

    err = np.linalg.norm(u - u0) / np.linalg.norm(u0)
    print(f"Relative L2 error after one revolution: {err:.3e}")

    AnimateSolution(nodes, snapshots, times, savepath="advection.gif", fps=5)
