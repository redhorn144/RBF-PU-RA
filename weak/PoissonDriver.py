import numpy as np
from mpi4py import MPI
from nodes.SquareDomain import PoissonSquareOne
from source.Setup import Setup, SetupGaussEval
from source.Operators import ApplyWeakLap, AssembleGaussRHS
from source.Quadrature import GaussPointsAndWeights
from source.Solver import gmres
from source.Plotter import PlotSolution

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    nodes, normals, groups = PoissonSquareOne(0.02)
    print(f"Rank {rank} generated {nodes.shape[0]} nodes.")
else:
    nodes = None
    normals = None
    groups = None

nodes   = comm.bcast(nodes,   root=0)
normals = comm.bcast(normals, root=0)
groups  = comm.bcast(groups,  root=0)

N = nodes.shape[0]
bc_nodes = groups['boundary:all']

patches, patches_for_rank = Setup(comm, nodes, normals, 100)
print(f"Rank {rank} setup complete with {len(patches)} patches.")

# Global Delaunay non-nodal Gauss quadrature (rank 0, then broadcast)
gauss_pts, gauss_wts = GaussPointsAndWeights(comm, patches, N, p=4)

# Per-patch PHS eval matrices at the global Gauss points
SetupGaussEval(comm, patches, gauss_pts)

WeakLap = ApplyWeakLap(comm, patches, N, gauss_wts, bc_nodes=bc_nodes)

def f_fn(pts):
    return 8*np.pi**2 * np.sin(2*np.pi*pts[:, 0]) * np.sin(2*np.pi*pts[:, 1])

rhs = AssembleGaussRHS(comm, patches, N, gauss_pts, gauss_wts, f_fn)
rhs[bc_nodes] = 0.0

print(f"Rank {rank} starting GMRES solve...")
if rank == 0:
    t_start = MPI.Wtime()
solution, num_iters = gmres(comm, WeakLap, rhs, tol=1e-10, restart=100, maxiter=100)

if rank == 0:
    t_end = MPI.Wtime()
    print(f"GMRES solve complete in {t_end - t_start:.2f} seconds.")
    print(f"GMRES converged in {num_iters} iterations.")
    u_exact = np.sin(2*np.pi * nodes[:, 0]) * np.sin(2*np.pi * nodes[:, 1])
    error = np.linalg.norm(solution - u_exact) / np.linalg.norm(u_exact)
    print(f"Relative L2 error: {error:.2e}")
    PlotSolution(nodes, solution)
