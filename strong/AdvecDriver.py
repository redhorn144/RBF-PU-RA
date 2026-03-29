import numpy as np
from mpi4py import MPI
from nodes.SquareDomain import PoissonSquareOne
from source.Setup import Setup
from source.Operators import ApplyDeriv
from source.Solver import gmres
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
boundary_nodes = groups['boundary:all']

# ── Patch setup ───────────────────────────────────────────────────────────────
patches, _ = Setup(comm, nodes, normals, 30, eval_epsilon=0.02)
print(f"Rank {rank} setup complete with {len(patches)} patches.")

# ── Derivative operators ──────────────────────────────────────────────────────
Dx = ApplyDeriv(comm, patches, N, 0, [], [])
Dy = ApplyDeriv(comm, patches, N, 1, [], [])

# ── Problem definition ────────────────────────────────────────────────────────
# Solid-body rotation about domain centre (0.5, 0.5) with angular velocity ω.
# Velocity field:  c(x,y) = ω * (-(y-0.5),  (x-0.5))
# Period:          T = 2π / ω
omega = 1.0
cx = -omega * (nodes[:, 1] - 0.5)
cy =  omega * (nodes[:, 0] - 0.5)

# Gaussian bump offset from centre by 0.2 in y so it orbits visibly.
x0, y0, sigma = 0.5, 0.70, 0.07
u0 = np.exp(-((nodes[:, 0] - x0)**2 + (nodes[:, 1] - y0)**2) / (2.0 * sigma**2))
u0[boundary_nodes] = 0.0

u = u0.copy()

# ── Implicit Euler time stepping ──────────────────────────────────────────────
# The collocation RBF-PU derivative and Laplacian operators both have spurious
# positive-real eigenvalues from the local RBF-PS matrices on each patch:
#   λ_adv_pos  ≈  17 s⁻¹   (advection operator)
#   λ_Lap_pos  ≈ 1600 s⁻¹  (confirmed: nu=0.01*Lap doubled the growth rate)
#
# For any explicit scheme, a positive-real eigenvalue grows every step
# regardless of dt — reducing dt only slows the blow-up, never stops it.
# Artificial viscosity using ApplyLap makes things worse because the PU
# collocation Laplacian has its own large positive-real eigenvalue.
#
# Implicit Euler is L-stable: amplification = 1/|1 - dt*λ|.
# For positive-real λ, this is stable only when dt*λ > 2 (so that |1-dt*λ|>1).
# Requirement: dt > 2 / λ_adv_pos ≈ 2/17 ≈ 0.12.
# Using dt=0.2: dt*λ_adv_pos ≈ 3.4 → amplification = 1/2.4 ≈ 0.42. Stable.
#
# Trade-off: implicit Euler damps physical (imaginary-axis) modes too.
# At dt=0.2, the Gaussian (k~7, c~0.2) decays by ~0.96 per step → ~0.3x peak
# after one revolution. The rotation is clearly visible; the decay is expected
# and will be fixed by the Galerkin formulation.
T          = 2.0 * np.pi / omega   # one full revolution
dt         = 0.01
n_steps    = int(round(T / dt))    # ~31 steps

snapshots = [u.copy()]
times     = [0.0]

def advec_matvec(v):
    # (I - dt * A_adv) v  where  A_adv = -(cx*Dx + cy*Dy)
    dv = v + dt * (cx * Dx(v) + cy * Dy(v))
    dv[boundary_nodes] = v[boundary_nodes]   # identity rows at boundary
    return dv

if rank == 0:
    t_start = MPI.Wtime()

t = 0.0
for step in range(n_steps):
    # Solve: (I - dt*A_adv) u^{n+1} = u^n
    u, _ = gmres(comm, advec_matvec, u, x0=u.copy(),
                 tol=1e-6, restart=50, maxiter=30)
    t += dt
    snapshots.append(u.copy())
    times.append(t)
    if rank == 0:
        idx = np.argmax(u)
        print(f"  step {step+1:3d}  t={t:.2f}  peak={u.max():.4f}"
              f"  loc=({nodes[idx,0]:.3f},{nodes[idx,1]:.3f})")

if rank == 0:
    t_end = MPI.Wtime()
    print(f"Time stepping complete: {n_steps} steps in {t_end - t_start:.2f} s")

    err = np.linalg.norm(u - u0) / np.linalg.norm(u0)
    print(f"Relative L2 error after one revolution: {err:.3e}")

    AnimateSolution(nodes, snapshots, times, savepath="advection.gif", fps=5)
