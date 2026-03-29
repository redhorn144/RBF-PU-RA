import numpy as np
from mpi4py import MPI
from nodes.SquareDomain import PoissonSquareOne
from source.Patch import Patch
from source.Setup import Setup
from source.Operators import ApplyDeriv, ApplyDerivAdj, ApplyMassMul
from source.Quadrature import PatchLocalWeights
from source.Solver import gmres
from source.Plotter import PlotSolution

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    nodes, normals, groups = PoissonSquareOne(0.025)
    print(f"Rank {rank} generated {nodes.shape[0]} nodes.")
else:
    nodes = None
    normals = None
    groups = None

nodes = comm.bcast(nodes, root=0)
normals = comm.bcast(normals, root=0)
groups = comm.bcast(groups, root=0)

N = nodes.shape[0]
bc_nodes = groups['boundary:all']

patches, patches_for_rank = Setup(comm, nodes, normals, 100)
print(f"Rank {rank} setup complete with {len(patches)} patches.")

# Compute quadrature weights
q = PatchLocalWeights(comm, patches, N)
if rank == 0:
    print(f"Quadrature weights: min={q.min():.2e}, max={q.max():.2e}, all positive={np.all(q > 0)}")

# Build operators: D_k, D_k^T, and mass multiply (diagonal Q)
d = nodes.shape[1]
Dk = [ApplyDeriv(comm, patches, N, k) for k in range(d)]
DkT = [ApplyDerivAdj(comm, patches, N, k) for k in range(d)]
MassMul = ApplyMassMul(comm, patches, N, q)

# Weak Laplacian: A u = sum_l D_l^T Q D_l u  with Dirichlet row replacement
def WeakLap(u):
    result = np.zeros(N)
    for l in range(d):
        g = Dk[l](u)        # g = D_l u
        g = q * g            # g = Q D_l u
        result += DkT[l](g)  # result += D_l^T Q D_l u
    result[bc_nodes] = u[bc_nodes]
    return result

# RHS: weak form b = Q f, with Dirichlet BC rows set to 0
# Weak form solves -Δu = f, so f = +8π² sin(2πx)sin(2πy) (positive, not Δu)
f = 8*np.pi**2*np.sin(2*np.pi * nodes[:, 0]) * np.sin(2*np.pi * nodes[:, 1])
rhs = q * f
rhs[bc_nodes] = 0.0

print(f"Rank {rank} starting GMRES solve...")
if rank == 0:
    t_start = MPI.Wtime()
solution, num_iters = gmres(comm, WeakLap, rhs, tol=1e-4, restart=100, maxiter=100)

if rank == 0:
    t_end = MPI.Wtime()
    print(f"GMRES solve complete in {t_end - t_start:.2f} seconds.")
    print(f"GMRES converged in {num_iters} iterations.")
    u_exact = np.sin(2*np.pi * nodes[:, 0]) * np.sin(2*np.pi * nodes[:, 1])
    error = np.linalg.norm(solution - u_exact) / np.linalg.norm(u_exact)
    print(f"Relative L2 error: {error:.2e}")
    PlotSolution(nodes, solution)
